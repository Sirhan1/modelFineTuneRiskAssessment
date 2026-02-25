from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import chain
from typing import Callable, Iterable, Iterator, Literal, Sequence, cast

import torch
from torch.func import functional_call, vmap
from torch.func import grad as torch_grad

from .types import SensitivitySubspace
from .utils import (
    batch_size,
    build_parameter_slices,
    flatten_parameter_grads,
    move_to_device,
    named_trainable_parameters,
    resolve_device,
    slice_batch,
)

LossFn = Callable[[torch.nn.Module, object], torch.Tensor]


@dataclass
class FisherConfig:
    top_rank: int = 32
    target_explained_variance: float | None = 0.9
    max_samples: int = 128
    max_parameters: int = 500_000
    top_weight_count: int = 256
    device: str = "auto"
    parameter_names: Sequence[str] | None = None
    gradient_collection: Literal["auto", "loop", "vmap"] = "loop"
    vmap_chunk_size: int | None = None
    auto_probe_samples: int = 2
    auto_probe_atol: float = 1e-6
    auto_probe_rtol: float = 1e-4
    subspace_method: Literal["svd", "randomized_svd", "diag_topk"] = "svd"
    randomized_oversample: int = 8
    randomized_niter: int = 1


class _FunctionalModelProxy(torch.nn.Module):
    """Proxy module that evaluates forward with functional parameters/buffers.

    Important: loss functions may inspect `model.parameters()` for explicit
    regularization terms. We override parameter/buffer iteration to expose the
    functional tensors used by `functional_call`, keeping loop and vmap
    gradient collection numerically consistent.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self._base_module = module
        self._functional_params = params
        self._functional_buffers = buffers

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return cast(
            torch.Tensor,
            functional_call(
                self._base_module,
                (self._functional_params, self._functional_buffers),
                args=args,
                kwargs=kwargs,
            ),
        )

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        _ = recurse
        _ = remove_duplicate
        for name, value in self._functional_params.items():
            key = f"{prefix}.{name}" if prefix else name
            yield key, cast(torch.nn.Parameter, value)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        _ = recurse
        for _, value in self.named_parameters():
            yield value

    def named_buffers(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        _ = recurse
        _ = remove_duplicate
        for name, value in self._functional_buffers.items():
            key = f"{prefix}.{name}" if prefix else name
            yield key, value

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        _ = recurse
        for _, value in self.named_buffers():
            yield value


class FisherSubspaceAnalyzer:
    """Estimate a skill-specific FIM and extract its top eigenspace."""

    def __init__(self, config: FisherConfig | None = None):
        self.config = config or FisherConfig()

    def analyze(
        self,
        model: torch.nn.Module,
        dataloader: Iterable[object],
        loss_fn: LossFn,
        parameter_names: Sequence[str] | None = None,
    ) -> SensitivitySubspace:
        device = resolve_device(self.config.device)
        model = model.to(device)

        selected_names = parameter_names if parameter_names is not None else self.config.parameter_names
        names, params = named_trainable_parameters(model, selected_names)
        if not params:
            raise ValueError("No trainable parameters selected for Fisher analysis.")

        total_params = sum(p.numel() for p in params)
        if total_params > self.config.max_parameters:
            raise ValueError(
                f"Selected parameter count {total_params:,} exceeds max_parameters="
                f"{self.config.max_parameters:,}. Restrict parameter_names for MVP runs."
            )

        gradients = self._collect_per_sample_gradients(
            model=model,
            parameter_names=names,
            params=params,
            dataloader=dataloader,
            loss_fn=loss_fn,
            max_samples=self.config.max_samples,
            device=device,
        )
        if not gradients:
            raise ValueError("No gradients were collected from the safety dataloader.")

        grad_matrix = torch.stack(gradients)  # [m, n]
        if grad_matrix.dtype in (torch.float16, torch.bfloat16):
            grad_matrix = grad_matrix.float()
        fisher_diag = (grad_matrix * grad_matrix).mean(dim=0)

        top_eigvals, top_eigvecs = self._estimate_subspace(grad_matrix, fisher_diag)
        fisher_diag_cpu = fisher_diag.detach().cpu()

        top_k = min(self.config.top_weight_count, fisher_diag_cpu.numel())
        top_scores, top_indices = torch.topk(fisher_diag_cpu, k=top_k)

        parameter_slices = build_parameter_slices(names, params)
        module_sums: dict[str, float] = {}
        module_counts: dict[str, int] = {}
        for p in parameter_slices:
            module_name = p.name.rsplit(".", 1)[0] if "." in p.name else p.name
            block = fisher_diag_cpu[p.start : p.end]
            module_sums[module_name] = module_sums.get(module_name, 0.0) + float(block.sum().item())
            module_counts[module_name] = module_counts.get(module_name, 0) + int(block.numel())
        module_scores = {
            name: module_sums[name] / float(module_counts[name])
            for name in module_sums
        }

        return SensitivitySubspace(
            parameter_slices=parameter_slices,
            fisher_eigenvalues=top_eigvals,
            fisher_eigenvectors=top_eigvecs,
            fisher_diagonal=fisher_diag_cpu,
            module_scores=module_scores,
            top_weight_indices=top_indices,
            top_weight_scores=top_scores,
        )

    def _collect_per_sample_gradients(
        self,
        model: torch.nn.Module,
        parameter_names: Sequence[str],
        params: Sequence[torch.nn.Parameter],
        dataloader: Iterable[object],
        loss_fn: LossFn,
        max_samples: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        strategy = self.config.gradient_collection
        if strategy == "loop":
            return self._collect_per_sample_gradients_loop(
                model=model,
                params=params,
                dataloader=dataloader,
                loss_fn=loss_fn,
                max_samples=max_samples,
                device=device,
            )

        if strategy == "vmap":
            return self._collect_per_sample_gradients_vmap(
                model=model,
                parameter_names=parameter_names,
                params=params,
                dataloader=dataloader,
                loss_fn=loss_fn,
                max_samples=max_samples,
                device=device,
            )

        return self._collect_per_sample_gradients_auto(
            model=model,
            parameter_names=parameter_names,
            params=params,
            dataloader=dataloader,
            loss_fn=loss_fn,
            max_samples=max_samples,
            device=device,
        )

    def _collect_per_sample_gradients_auto(
        self,
        model: torch.nn.Module,
        parameter_names: Sequence[str],
        params: Sequence[torch.nn.Parameter],
        dataloader: Iterable[object],
        loss_fn: LossFn,
        max_samples: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        iterator = iter(dataloader)
        try:
            first_batch = next(iterator)
        except StopIteration:
            return []

        probe_samples = min(max_samples, max(1, self.config.auto_probe_samples))
        loop_probe = self._collect_per_sample_gradients_loop(
            model=model,
            params=params,
            dataloader=[first_batch],
            loss_fn=loss_fn,
            max_samples=probe_samples,
            device=device,
        )

        use_vmap = False
        try:
            vmap_probe = self._collect_per_sample_gradients_vmap(
                model=model,
                parameter_names=parameter_names,
                params=params,
                dataloader=[first_batch],
                loss_fn=loss_fn,
                max_samples=probe_samples,
                device=device,
            )
            use_vmap = self._gradient_probe_allclose(loop_probe, vmap_probe)
            if not use_vmap:
                warnings.warn(
                    "Auto Fisher backend selected loop because vmap/loop parity check failed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except Exception as exc:
            warnings.warn(
                f"Auto Fisher backend selected loop because vmap probe failed: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )

        full_dataloader = chain([first_batch], iterator)
        if use_vmap:
            return self._collect_per_sample_gradients_vmap(
                model=model,
                parameter_names=parameter_names,
                params=params,
                dataloader=full_dataloader,
                loss_fn=loss_fn,
                max_samples=max_samples,
                device=device,
            )
        return self._collect_per_sample_gradients_loop(
            model=model,
            params=params,
            dataloader=full_dataloader,
            loss_fn=loss_fn,
            max_samples=max_samples,
            device=device,
        )

    def _collect_per_sample_gradients_loop(
        self,
        model: torch.nn.Module,
        params: Sequence[torch.nn.Parameter],
        dataloader: Iterable[object],
        loss_fn: LossFn,
        max_samples: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        was_training = model.training
        model.eval()

        grads: list[torch.Tensor] = []
        try:
            for batch in dataloader:
                batch = move_to_device(batch, device)
                bsz = batch_size(batch)
                for i in range(bsz):
                    sample = slice_batch(batch, i)
                    model.zero_grad(set_to_none=True)
                    loss = loss_fn(model, sample)
                    if loss.ndim > 0:
                        loss = loss.mean()
                    torch.autograd.backward(loss)
                    grad = flatten_parameter_grads(params).detach()
                    # CPU SVD does not support fp16/bf16. Promote for stable Fisher decomposition.
                    if grad.dtype in (torch.float16, torch.bfloat16):
                        grad = grad.float()
                    grad = grad.cpu()
                    grads.append(grad)
                    if len(grads) >= max_samples:
                        return grads
            return grads
        finally:
            if was_training:
                model.train()

    def _collect_per_sample_gradients_vmap(
        self,
        model: torch.nn.Module,
        parameter_names: Sequence[str],
        params: Sequence[torch.nn.Parameter],
        dataloader: Iterable[object],
        loss_fn: LossFn,
        max_samples: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        was_training = model.training
        model.eval()

        base_named_params = dict(model.named_parameters())
        base_named_buffers = dict(model.named_buffers())
        trainable_names = list(parameter_names)
        trainable_params = tuple(params)

        def single_sample_loss(
            trainable_values: tuple[torch.Tensor, ...],
            sample: object,
        ) -> torch.Tensor:
            merged_params: dict[str, torch.Tensor] = {
                name: param
                for name, param in base_named_params.items()
            }
            for name, value in zip(trainable_names, trainable_values, strict=False):
                merged_params[name] = value
            proxy = _FunctionalModelProxy(
                model,
                params=merged_params,
                buffers=base_named_buffers,
            )
            loss = loss_fn(proxy, sample)
            if loss.ndim > 0:
                loss = loss.mean()
            return loss

        grads: list[torch.Tensor] = []
        chunk_size = self.config.vmap_chunk_size
        try:
            for batch in dataloader:
                batch = move_to_device(batch, device)
                bsz = batch_size(batch)
                if bsz <= 0:
                    continue

                in_dims = _batch_in_dims(batch)
                per_sample_grad_fn = vmap(
                    torch_grad(single_sample_loss),
                    in_dims=(None, in_dims),
                )
                step = chunk_size if chunk_size is not None and chunk_size > 0 else bsz
                for start in range(0, bsz, step):
                    end = min(start + step, bsz)
                    chunk = _slice_batch_range(batch, start, end)
                    grads_tree = per_sample_grad_fn(trainable_params, chunk)
                    flat_chunk = torch.cat([g.reshape(end - start, -1) for g in grads_tree], dim=1)
                    if flat_chunk.dtype in (torch.float16, torch.bfloat16):
                        flat_chunk = flat_chunk.float()
                    flat_chunk = flat_chunk.detach().cpu()
                    for row in flat_chunk:
                        grads.append(row)
                        if len(grads) >= max_samples:
                            return grads
            return grads
        finally:
            if was_training:
                model.train()

    def _gradient_probe_allclose(
        self,
        loop_probe: list[torch.Tensor],
        vmap_probe: list[torch.Tensor],
    ) -> bool:
        if len(loop_probe) != len(vmap_probe):
            return False
        if not loop_probe and not vmap_probe:
            return True
        if not loop_probe or not vmap_probe:
            return False
        left = torch.stack(loop_probe)
        right = torch.stack(vmap_probe)
        return torch.allclose(
            left,
            right,
            atol=self.config.auto_probe_atol,
            rtol=self.config.auto_probe_rtol,
        )

    def _estimate_subspace(
        self,
        grad_matrix: torch.Tensor,
        fisher_diag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        method = self.config.subspace_method
        if method == "svd":
            return self._subspace_from_full_svd(grad_matrix)
        if method == "randomized_svd":
            return self._subspace_from_randomized_svd(grad_matrix)
        if method == "diag_topk":
            return self._subspace_from_diag_topk(fisher_diag)
        raise ValueError(f"Unsupported subspace_method: {method}")

    def _subspace_from_full_svd(self, grad_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = grad_matrix.shape[0]
        _, singular_values, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
        eigvals = (singular_values ** 2) / float(m)
        eigvecs = vh.T
        max_rank = min(self.config.top_rank, eigvals.numel())
        d = self._select_rank_from_explained_variance(eigvals, max_rank)
        return eigvals[:d].detach().cpu(), eigvecs[:, :d].detach().cpu()

    def _subspace_from_randomized_svd(self, grad_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m, n = grad_matrix.shape
        target_rank = min(self.config.top_rank, m, n)
        if target_rank == 0:
            return torch.empty(0), torch.empty((n, 0))

        oversample = max(0, self.config.randomized_oversample)
        sketch_rank = min(target_rank + oversample, min(m, n))
        omega = torch.randn(n, sketch_rank, device=grad_matrix.device, dtype=grad_matrix.dtype)
        y = grad_matrix @ omega
        for _ in range(max(0, self.config.randomized_niter)):
            y = grad_matrix @ (grad_matrix.T @ y)
        q, _ = torch.linalg.qr(y, mode="reduced")
        b = q.T @ grad_matrix
        _, singular_values, vh = torch.linalg.svd(b, full_matrices=False)
        eigvals = (singular_values ** 2) / float(m)
        eigvecs = vh.T
        d = self._select_rank_from_explained_variance(eigvals[:target_rank], target_rank)
        return eigvals[:d].detach().cpu(), eigvecs[:, :d].detach().cpu()

    def _subspace_from_diag_topk(self, fisher_diag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = fisher_diag.numel()
        d = min(self.config.top_rank, n)
        if d == 0:
            return torch.empty(0), torch.empty((n, 0))
        values, indices = torch.topk(fisher_diag, k=d)
        d = self._select_rank_from_explained_variance(values, d)
        values = values[:d]
        indices = indices[:d]
        if d == 0:
            return torch.empty(0), torch.empty((n, 0))
        basis = torch.zeros(n, d, dtype=fisher_diag.dtype, device=fisher_diag.device)
        basis[indices, torch.arange(d, device=fisher_diag.device)] = 1.0
        return values.detach().cpu(), basis.detach().cpu()

    def _select_rank_from_explained_variance(
        self,
        values: torch.Tensor,
        max_rank: int,
    ) -> int:
        if max_rank <= 0:
            return 0
        target = self.config.target_explained_variance
        if target is None:
            return max_rank
        if target <= 0.0:
            return 1
        if target >= 1.0:
            return max_rank

        clipped = torch.clamp(values[:max_rank], min=0.0)
        total = float(clipped.sum().item())
        if total <= 0.0:
            return max_rank

        cumulative = torch.cumsum(clipped, dim=0) / total
        reached = torch.nonzero(cumulative >= target, as_tuple=False)
        if reached.numel() == 0:
            return max_rank
        return int(reached[0].item()) + 1


def _batch_in_dims(batch: object) -> object:
    if torch.is_tensor(batch):
        if batch.ndim == 0:
            return None
        return 0
    if isinstance(batch, dict):
        return {k: _batch_in_dims(v) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_batch_in_dims(x) for x in batch)
    if isinstance(batch, list):
        return [_batch_in_dims(x) for x in batch]
    return None


def _slice_batch_range(batch: object, start: int, end: int) -> object:
    if torch.is_tensor(batch):
        if batch.ndim == 0:
            return batch
        return batch[start:end]
    if isinstance(batch, dict):
        return {k: _slice_batch_range(v, start, end) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_slice_batch_range(x, start, end) for x in batch)
    if isinstance(batch, list):
        return [_slice_batch_range(x, start, end) for x in batch]
    return batch
