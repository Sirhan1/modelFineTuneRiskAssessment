from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from .types import CurvatureCouplingResult, SensitivitySubspace
from .utils import flatten_tensors, move_to_device, named_trainable_parameters, resolve_device

LossFn = Callable[[torch.nn.Module, object], torch.Tensor]


@dataclass
class CurvatureConfig:
    device: str = "auto"
    max_batches: int = 1


class CurvatureCouplingAnalyzer:
    """Estimate AIC Condition 3 via directional derivative H g."""

    def __init__(self, config: CurvatureConfig | None = None):
        self.config = config or CurvatureConfig()

    def analyze(
        self,
        model: torch.nn.Module,
        dataloader: Iterable[object],
        loss_fn: LossFn,
        subspace: SensitivitySubspace,
    ) -> CurvatureCouplingResult:
        device = resolve_device(self.config.device)
        model = model.to(device)

        selected_names = [p.name for p in subspace.parameter_slices]
        _, params = named_trainable_parameters(model, include_names=selected_names)
        if not params:
            raise ValueError("No trainable parameters found for curvature analysis.")

        loss = self._mean_loss(model, dataloader, loss_fn, device)
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        g = flatten_tensors(grads, params)

        # Directional derivative of the gradient field along itself: (∇g)g = H g.
        dot = torch.dot(g, g.detach())
        hvp = torch.autograd.grad(dot, params, allow_unused=True)

        g_vec = g.detach().cpu()
        a_vec = flatten_tensors(hvp, params).detach().cpu()

        basis = subspace.fisher_eigenvectors
        eigvals = torch.clamp(subspace.fisher_eigenvalues, min=0.0)

        g_coeff = basis.T @ g_vec
        epsilon_hat = float(g_coeff.norm().item())

        a_coeff = basis.T @ a_vec
        projected_a = basis @ a_coeff

        gamma_hat = float(torch.sqrt(torch.sum(eigvals * (a_coeff ** 2))).item())
        projected_acc_norm = float(projected_a.norm().item())

        return CurvatureCouplingResult(
            gamma_hat=gamma_hat,
            epsilon_hat=epsilon_hat,
            acceleration_norm=float(a_vec.norm().item()),
            projected_acceleration_norm=projected_acc_norm,
        )

    def _mean_loss(
        self,
        model: torch.nn.Module,
        dataloader: Iterable[object],
        loss_fn: LossFn,
        device: torch.device,
    ) -> torch.Tensor:
        total = None
        count = 0
        for i, batch in enumerate(dataloader):
            if i >= self.config.max_batches:
                break
            batch = move_to_device(batch, device)
            loss = loss_fn(model, batch)
            if loss.ndim > 0:
                loss = loss.mean()
            total = loss if total is None else total + loss
            count += 1

        if count == 0 or total is None:
            raise ValueError("Fine-tuning dataloader yielded no batches.")

        return total / float(count)
