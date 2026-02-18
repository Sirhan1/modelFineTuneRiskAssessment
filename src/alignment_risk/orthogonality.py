from __future__ import annotations

import torch

from .types import InitialRiskScore, SensitivitySubspace


def project_onto_subspace(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coefficients = basis.T @ vector
    return basis @ coefficients


def initial_risk_from_update(
    update_vector: torch.Tensor,
    subspace: SensitivitySubspace,
    *,
    orthogonality_threshold: float = 0.05,
) -> InitialRiskScore:
    """
    Compute the initial overlap with the safety subspace.

    projected_ratio is ||P_M(delta)|| / ||delta||, which is the cosine between
    update direction and the closest direction inside subspace M.
    """
    update = update_vector.detach().cpu()
    basis = subspace.fisher_eigenvectors

    update_norm = float(update.norm().item())
    if update_norm == 0.0:
        return InitialRiskScore(
            cosine_to_subspace=0.0,
            projected_ratio=0.0,
            update_norm=0.0,
            projected_norm=0.0,
            trigger_curvature_check=True,
        )

    projected = project_onto_subspace(update, basis)
    projected_norm = float(projected.norm().item())
    ratio = projected_norm / (update_norm + 1e-12)
    ratio = max(0.0, min(1.0, ratio))

    return InitialRiskScore(
        cosine_to_subspace=ratio,
        projected_ratio=ratio,
        update_norm=update_norm,
        projected_norm=projected_norm,
        trigger_curvature_check=ratio <= orthogonality_threshold,
    )
