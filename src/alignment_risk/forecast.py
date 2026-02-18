from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import SafetyForecast


@dataclass
class ForecastConfig:
    max_steps: int = 2000
    step_size: float = 1.0
    collapse_loss_threshold: float = 0.1


def forecast_stability(
    lambda_min: float,
    gamma: float,
    epsilon: float,
    config: ForecastConfig | None = None,
) -> SafetyForecast:
    """
    Build a lower-bound style forecast from Theorem 6.2 + Corollary 6.3.

    Drift lower bound (ignoring O(t^3)): ||F^(1/2) P delta(t)|| >= (gamma / 2)t^2 - epsilon t.
    Utility drop proxy: Delta u ~ 0.5 * lambda * drift^2.
    Quartic asymptote: Delta u ~= (lambda * gamma^2 / 8) t^4.
    """
    cfg = config or ForecastConfig()

    steps = np.arange(cfg.max_steps + 1, dtype=float)
    t = steps * cfg.step_size

    projected_drift = np.maximum(0.5 * gamma * (t ** 2) - epsilon * t, 0.0)
    estimated_loss = 0.5 * lambda_min * (projected_drift ** 2)
    quartic_lower_bound = (lambda_min * (gamma ** 2) / 8.0) * (t ** 4)

    crossings = np.flatnonzero(estimated_loss >= cfg.collapse_loss_threshold)
    collapse_step = int(crossings[0]) if crossings.size else None

    return SafetyForecast(
        steps=steps,
        projected_drift=projected_drift,
        quartic_lower_bound=quartic_lower_bound,
        estimated_loss=estimated_loss,
        collapse_step=collapse_step,
    )
