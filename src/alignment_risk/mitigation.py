"""AlignGuard-style LoRA mitigation losses.

Academic grounding:
- [ALIGNGUARD-2025] https://arxiv.org/pdf/2508.02079
- [AIC-2026] https://arxiv.org/pdf/2602.15799

See docs/SOURCES.md for section/page-level mapping to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Sequence, cast

import torch

from .types import SensitivitySubspace
from .utils import flatten_tensors, named_trainable_parameters

HType = Literal["identity", "fisher_diagonal", "custom"]


@dataclass
class AlignGuardConfig:
    """Hyperparameters for AlignGuard-style LoRA mitigation."""

    lambda_a: float = 0.25
    lambda_t: float = 0.50
    lambda_nc: float = 0.10
    alpha: float = 0.50
    beta: float = 4.0
    tau: float = 0.01
    h_type: HType = "fisher_diagonal"
    epsilon: float = 1e-12


@dataclass
class AlignGuardLossBreakdown:
    task_loss: torch.Tensor
    alignment_penalty: torch.Tensor
    task_stability_penalty: torch.Tensor
    collision_riemannian: torch.Tensor
    collision_geodesic: torch.Tensor
    collision_penalty: torch.Tensor
    total_loss: torch.Tensor


def _validate_alignguard_config(cfg: AlignGuardConfig) -> None:
    nonnegative_scalars = {
        "lambda_a": cfg.lambda_a,
        "lambda_t": cfg.lambda_t,
        "lambda_nc": cfg.lambda_nc,
        "beta": cfg.beta,
        "tau": cfg.tau,
    }
    for name, value in nonnegative_scalars.items():
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"AlignGuardConfig.{name} must be finite and >= 0.")

    if not isfinite(cfg.alpha) or not (0.0 <= cfg.alpha <= 1.0):
        raise ValueError("AlignGuardConfig.alpha must be finite and in [0, 1].")
    if not isfinite(cfg.epsilon) or cfg.epsilon <= 0.0:
        raise ValueError("AlignGuardConfig.epsilon must be finite and > 0.")


def project_onto_subspace(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coeffs = basis.T @ vector
    return basis @ coeffs


def decompose_update(
    update_vector: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_a = project_onto_subspace(update_vector, basis)
    delta_t = update_vector - delta_a
    return delta_a, delta_t


def fisher_weighted_alignment_penalty(
    delta_a: torch.Tensor,
    basis: torch.Tensor,
    eigenvalues: torch.Tensor,
) -> torch.Tensor:
    coeffs = basis.T @ delta_a
    return torch.sum(torch.clamp(eigenvalues, min=0.0) * (coeffs ** 2))


def riemannian_overlap_penalty(
    delta_a: torch.Tensor,
    delta_t: torch.Tensor,
    *,
    beta: float,
    tau: float,
) -> torch.Tensor:
    eta = 1.0 + beta * torch.sigmoid(torch.abs(delta_a + delta_t) - tau)
    return torch.mean(eta * torch.abs(delta_a * delta_t))


def geodesic_overlap_penalty(
    delta_a: torch.Tensor,
    delta_t: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    numerator = torch.dot(delta_a, delta_t) ** 2
    denominator = (delta_a.norm() ** 2) * (delta_t.norm() ** 2) + eps
    return cast(torch.Tensor, numerator / denominator)


class AlignGuardLoRARegularizer:
    """Compute AlignGuard-style regularized loss terms for LoRA training.

    This class captures a reference adapter state and penalizes drift in the
    Fisher-sensitive component while keeping task-adaptive drift controlled.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        subspace: SensitivitySubspace,
        parameter_names: Sequence[str],
        config: AlignGuardConfig | None = None,
        custom_h_diagonal: torch.Tensor | None = None,
    ):
        self.config = config or AlignGuardConfig()
        _validate_alignguard_config(self.config)

        names, params = named_trainable_parameters(model, include_names=parameter_names)
        if not params:
            raise ValueError("No trainable parameters selected for AlignGuard regularization.")

        self.parameter_names = names
        self.params = list(params)

        device = self.params[0].device
        dtype = self.params[0].dtype

        self.basis = subspace.fisher_eigenvectors.to(device=device, dtype=dtype)
        self.eigenvalues = torch.clamp(subspace.fisher_eigenvalues.to(device=device, dtype=dtype), min=0.0)
        self.fisher_diagonal = torch.clamp(subspace.fisher_diagonal.to(device=device, dtype=dtype), min=0.0)

        self._custom_h_diagonal = None
        if custom_h_diagonal is not None:
            custom = custom_h_diagonal.to(device=device, dtype=dtype).reshape(-1)
            if custom.numel() != self.fisher_diagonal.numel():
                raise ValueError(
                    "custom_h_diagonal size mismatch: expected "
                    f"{self.fisher_diagonal.numel()} values, got {custom.numel()}."
                )
            self._custom_h_diagonal = custom

        self.reference_vector = self._current_parameter_vector().detach().clone()

    def reset_reference(self) -> None:
        """Reset regularization anchor to the current parameter state."""
        self.reference_vector = self._current_parameter_vector().detach().clone()

    def regularized_loss(
        self,
        task_loss: torch.Tensor,
        *,
        h_diagonal: torch.Tensor | None = None,
    ) -> AlignGuardLossBreakdown:
        if task_loss.ndim > 0:
            task_loss = task_loss.mean()

        delta = self._update_vector()
        delta_a, delta_t = decompose_update(delta, self.basis)

        align_base = fisher_weighted_alignment_penalty(delta_a, self.basis, self.eigenvalues)
        alignment_penalty = self.config.lambda_a * align_base

        h_diag = self._resolve_h_diagonal(h_diagonal)
        task_stability_base = torch.sum(h_diag * (delta_t ** 2))
        task_stability_penalty = self.config.lambda_t * task_stability_base

        col_rm = riemannian_overlap_penalty(
            delta_a,
            delta_t,
            beta=self.config.beta,
            tau=self.config.tau,
        )
        col_geo = geodesic_overlap_penalty(
            delta_a,
            delta_t,
            eps=self.config.epsilon,
        )
        collision_penalty = self.config.lambda_nc * (
            self.config.alpha * col_rm + (1.0 - self.config.alpha) * col_geo
        )

        total_loss = task_loss + alignment_penalty + task_stability_penalty + collision_penalty

        return AlignGuardLossBreakdown(
            task_loss=task_loss,
            alignment_penalty=alignment_penalty,
            task_stability_penalty=task_stability_penalty,
            collision_riemannian=col_rm,
            collision_geodesic=col_geo,
            collision_penalty=collision_penalty,
            total_loss=total_loss,
        )

    def _current_parameter_vector(self) -> torch.Tensor:
        return flatten_tensors(self.params, self.params)

    def _update_vector(self) -> torch.Tensor:
        current = self._current_parameter_vector()
        ref = self.reference_vector.to(device=current.device, dtype=current.dtype)
        return current - ref

    def _resolve_h_diagonal(self, h_diagonal: torch.Tensor | None) -> torch.Tensor:
        if h_diagonal is not None:
            candidate = h_diagonal.to(
                device=self.fisher_diagonal.device,
                dtype=self.fisher_diagonal.dtype,
            ).reshape(-1)
            if candidate.numel() != self.fisher_diagonal.numel():
                raise ValueError(
                    "h_diagonal size mismatch: expected "
                    f"{self.fisher_diagonal.numel()} values, got {candidate.numel()}."
                )
            return torch.clamp(candidate, min=0.0)

        if self.config.h_type == "identity":
            return torch.ones_like(self.fisher_diagonal)

        if self.config.h_type == "fisher_diagonal":
            return torch.clamp(self.fisher_diagonal, min=0.0)

        if self.config.h_type == "custom":
            if self._custom_h_diagonal is None:
                raise ValueError(
                    "h_type='custom' requires custom_h_diagonal at initialization "
                    "or h_diagonal in regularized_loss()."
                )
            return torch.clamp(self._custom_h_diagonal, min=0.0)

        raise ValueError(f"Unsupported h_type: {self.config.h_type}")
