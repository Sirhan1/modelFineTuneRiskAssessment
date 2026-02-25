"""Curvature coupling diagnostics for alignment risk.

Academic grounding:
- [AIC-2026] https://arxiv.org/pdf/2602.15799

See docs/SOURCES.md for section/page-level mapping to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from .types import CurvatureCouplingResult, SensitivitySubspace
from .utils import (
    batch_size,
    flatten_tensors,
    move_to_device,
    named_trainable_parameters,
    resolve_device,
)

LossFn = Callable[[torch.nn.Module, object], torch.Tensor]


@dataclass
class CurvatureConfig:
    device: str = "auto"
    max_batches: int = 1
    force_eval: bool = True


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
        *,
        max_batches_override: int | None = None,
    ) -> CurvatureCouplingResult:
        device = resolve_device(self.config.device)
        model = model.to(device)
        max_batches = max_batches_override if max_batches_override is not None else self.config.max_batches

        selected_names = [p.name for p in subspace.parameter_slices]
        _, params = named_trainable_parameters(
            model,
            include_names=selected_names,
            strict=True,
        )
        if not params:
            raise ValueError("No trainable parameters found for curvature analysis.")

        was_training = model.training
        if self.config.force_eval:
            model.eval()
        try:
            loss = self._mean_loss(model, dataloader, loss_fn, device, max_batches=max_batches)
        finally:
            if self.config.force_eval and was_training:
                model.train()

        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        g = flatten_tensors(grads, params)

        # Directional derivative of the gradient field along itself: (∇g)g = H g.
        dot = torch.dot(g, g.detach())
        hvp = torch.autograd.grad(dot, params, allow_unused=True)

        basis = subspace.fisher_eigenvectors.detach().cpu()
        eigvals = torch.clamp(subspace.fisher_eigenvalues.detach().cpu(), min=0.0).to(dtype=basis.dtype)

        g_vec = g.detach().to(device=basis.device, dtype=basis.dtype)
        a_vec = flatten_tensors(hvp, params).detach().to(device=basis.device, dtype=basis.dtype)

        g_coeff = basis.T @ g_vec
        epsilon_hat = float(torch.sqrt(torch.sum(eigvals * (g_coeff ** 2))).item())

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
        *,
        max_batches: int,
    ) -> torch.Tensor:
        total = None
        total_weight = 0
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = move_to_device(batch, device)
            loss = loss_fn(model, batch)
            try:
                weight = batch_size(batch)
            except ValueError:
                # Synthetic/test iterables may provide metadata-only batches.
                weight = int(loss.shape[0]) if loss.ndim > 0 and loss.shape[0] > 0 else 1
            if weight <= 0:
                raise ValueError("Fine-tuning batch size must be positive.")
            if loss.ndim > 0:
                loss = loss.mean()
            weighted_loss = loss * float(weight)
            total = weighted_loss if total is None else total + weighted_loss
            total_weight += weight

        if total_weight == 0 or total is None:
            raise ValueError("Fine-tuning dataloader yielded no batches.")

        return total / float(total_weight)
