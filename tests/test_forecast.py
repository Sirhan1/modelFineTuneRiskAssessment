import numpy as np
import pytest

from alignment_risk.forecast import ForecastConfig, forecast_stability


def test_forecast_identifies_collapse() -> None:
    cfg = ForecastConfig(max_steps=100, step_size=1.0, collapse_loss_threshold=0.5)
    out = forecast_stability(lambda_min=1.0, gamma=0.4, epsilon=0.0, config=cfg)

    assert out.collapse_step is not None
    assert out.collapse_time is not None
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


def test_forecast_rejects_negative_max_steps() -> None:
    cfg = ForecastConfig(max_steps=-1, step_size=1.0, collapse_loss_threshold=0.1)
    with pytest.raises(ValueError, match="max_steps"):
        forecast_stability(lambda_min=1.0, gamma=0.2, epsilon=0.1, config=cfg)


def test_forecast_rejects_non_positive_step_size() -> None:
    cfg = ForecastConfig(max_steps=10, step_size=0.0, collapse_loss_threshold=0.1)
    with pytest.raises(ValueError, match="step_size"):
        forecast_stability(lambda_min=1.0, gamma=0.2, epsilon=0.1, config=cfg)


def test_forecast_exposes_physical_time_axis_from_step_size() -> None:
    cfg = ForecastConfig(max_steps=10, step_size=0.5, collapse_loss_threshold=0.01)
    out = forecast_stability(lambda_min=1.0, gamma=1.0, epsilon=0.0, config=cfg)

    assert np.allclose(out.times, out.steps.astype(float) * cfg.step_size)
    if out.collapse_step is not None:
        assert out.collapse_time == pytest.approx(float(out.times[out.collapse_step]))
