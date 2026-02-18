from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class ParameterSlice:
    name: str
    start: int
    end: int
    shape: Tuple[int, ...]


@dataclass
class SensitivitySubspace:
    parameter_slices: List[ParameterSlice]
    fisher_eigenvalues: torch.Tensor
    fisher_eigenvectors: torch.Tensor
    fisher_diagonal: torch.Tensor
    module_scores: Dict[str, float]
    top_weight_indices: torch.Tensor
    top_weight_scores: torch.Tensor


@dataclass
class InitialRiskScore:
    cosine_to_subspace: float
    projected_ratio: float
    update_norm: float
    projected_norm: float
    trigger_curvature_check: bool


@dataclass
class CurvatureCouplingResult:
    gamma_hat: float
    epsilon_hat: float
    acceleration_norm: float
    projected_acceleration_norm: float


@dataclass
class SafetyForecast:
    steps: np.ndarray
    projected_drift: np.ndarray
    quartic_lower_bound: np.ndarray
    estimated_loss: np.ndarray
    collapse_step: Optional[int]


@dataclass
class RiskAssessmentReport:
    subspace: SensitivitySubspace
    initial_risk: InitialRiskScore
    curvature: CurvatureCouplingResult
    forecast: SafetyForecast
    warning: str
