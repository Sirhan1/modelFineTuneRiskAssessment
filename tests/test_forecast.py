import numpy as np

from alignment_risk.forecast import ForecastConfig, forecast_stability


def test_forecast_identifies_collapse() -> None:
    cfg = ForecastConfig(max_steps=100, step_size=1.0, collapse_loss_threshold=0.5)
    out = forecast_stability(lambda_min=1.0, gamma=0.4, epsilon=0.0, config=cfg)

    assert out.collapse_step is not None
    assert out.estimated_loss[out.collapse_step] >= cfg.collapse_loss_threshold


def test_quartic_curve_is_monotonic_for_positive_coefficients() -> None:
    out = forecast_stability(
        lambda_min=2.0,
        gamma=0.2,
        epsilon=0.0,
        config=ForecastConfig(max_steps=30, step_size=1.0),
    )
    diffs = np.diff(out.quartic_lower_bound)
    assert np.all(diffs >= 0.0)
