from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

from .types import SensitivitySubspace
from .utils import (
    batch_size,
    build_parameter_slices,
    flatten_parameter_grads,
    move_to_device,
    named_trainable_parameters,
    slice_batch,
)

LossFn = Callable[[torch.nn.Module, object], torch.Tensor]


@dataclass
class FisherConfig:
    top_rank: int = 32
    max_samples: int = 128
    max_parameters: int = 500_000
    top_weight_count: int = 256
    device: str = "cpu"
    parameter_names: Sequence[str] | None = None


class FisherSubspaceAnalyzer:
    """Estimate a skill-specific FIM and extract its top eigenspace."""

    def __init__(self, config: FisherConfig | None = None):
        self.config = config or FisherConfig()

    def analyze(
        self,
        model: torch.nn.Module,
        dataloader: Iterable[object],
        loss_fn: LossFn,
    ) -> SensitivitySubspace:
        device = torch.device(self.config.device)
        model = model.to(device)

        names, params = named_trainable_parameters(model, self.config.parameter_names)
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
            params=params,
            dataloader=dataloader,
            loss_fn=loss_fn,
            max_samples=self.config.max_samples,
            device=device,
        )
        if not gradients:
            raise ValueError("No gradients were collected from the safety dataloader.")

        grad_matrix = torch.stack(gradients)  # [m, n]
        m = grad_matrix.shape[0]
        fisher_diag = (grad_matrix * grad_matrix).mean(dim=0)

        # If G has per-sample score rows, F = (1/m) G^T G with eigenpairs from SVD(G).
        _, singular_values, vh = torch.linalg.svd(grad_matrix, full_matrices=False)
        eigvals = (singular_values ** 2) / float(m)
        eigvecs = vh.T

        d = min(self.config.top_rank, eigvals.numel())
        top_eigvals = eigvals[:d].detach().cpu()
        top_eigvecs = eigvecs[:, :d].detach().cpu()
        fisher_diag_cpu = fisher_diag.detach().cpu()

        top_k = min(self.config.top_weight_count, fisher_diag_cpu.numel())
        top_scores, top_indices = torch.topk(fisher_diag_cpu, k=top_k)

        parameter_slices = build_parameter_slices(names, params)
        module_scores = {
            p.name: float(fisher_diag_cpu[p.start : p.end].mean().item())
            for p in parameter_slices
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
                    loss.backward()
                    grad = flatten_parameter_grads(params).detach().cpu()
                    grads.append(grad)
                    if len(grads) >= max_samples:
                        return grads
            return grads
        finally:
            if was_training:
                model.train()
