"""Stability forecast utilities derived from AIC-style lower bounds.

Academic grounding:
- [AIC-2026] https://arxiv.org/pdf/2602.15799

See docs/SOURCES.md for section/page-level mapping to this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import SafetyForecast


@dataclass
class ForecastConfig:
    max_steps: int = 2000
    step_size: float = 1.0
    collapse_loss_threshold: float = 0.1


def _validate_forecast_config(cfg: ForecastConfig) -> None:
    if cfg.max_steps < 0:
        raise ValueError("ForecastConfig.max_steps must be >= 0.")
    if cfg.step_size <= 0.0:
        raise ValueError("ForecastConfig.step_size must be > 0.")
    if cfg.collapse_loss_threshold < 0.0:
        raise ValueError("ForecastConfig.collapse_loss_threshold must be >= 0.")


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
    _validate_forecast_config(cfg)

    steps = np.arange(cfg.max_steps + 1, dtype=int)
    times = steps.astype(float) * cfg.step_size

    projected_drift = np.maximum(0.5 * gamma * (times ** 2) - epsilon * times, 0.0)
    estimated_loss = 0.5 * lambda_min * (projected_drift ** 2)
    quartic_lower_bound = (lambda_min * (gamma ** 2) / 8.0) * (times ** 4)

    crossings = np.flatnonzero(estimated_loss >= cfg.collapse_loss_threshold)
    collapse_step = int(crossings[0]) if crossings.size else None
    collapse_time = float(times[collapse_step]) if collapse_step is not None else None

    return SafetyForecast(
        steps=steps,
        times=times,
        projected_drift=projected_drift,
        quartic_lower_bound=quartic_lower_bound,
        estimated_loss=estimated_loss,
        collapse_step=collapse_step,
        collapse_time=collapse_time,
    )
