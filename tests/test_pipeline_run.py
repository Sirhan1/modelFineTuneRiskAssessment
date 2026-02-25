import numpy as np
import pytest
import torch

from alignment_risk.pipeline import AlignmentRiskPipeline, PipelineConfig
from alignment_risk.types import (
    CurvatureCouplingResult,
    ParameterSlice,
    SafetyForecast,
    SensitivitySubspace,
)


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_pipeline_run_accepts_single_pass_fine_tune_iterable() -> None:
    torch.manual_seed(0)
    model = _TinyClassifier()

    x_safety = torch.randn(8, 4)
    y_safety = torch.randint(0, 2, (8,))
    safety_loader = [(x_safety, y_safety)]

    x_ft = torch.randn(8, 4)
    y_ft = torch.randint(0, 2, (8,))

    def fine_tune_generator() -> object:
        # Intentionally one-shot iterable (generator) to verify run() can consume it twice safely.
        yield (x_ft[:4], y_ft[:4])
        yield (x_ft[4:], y_ft[4:])

    def loss_fn(model_: torch.nn.Module, batch: object) -> torch.Tensor:
        x, y = batch  # type: ignore[misc]
        logits = model_(x)
        return torch.nn.functional.cross_entropy(logits, y)

    cfg = PipelineConfig(mode="full", learning_rate=1e-2)
    cfg.fisher.max_samples = 8
    cfg.fisher.top_rank = 2
    cfg.fisher.max_parameters = 128
    cfg.curvature.max_batches = 1
    cfg.forecast.max_steps = 20

    report = AlignmentRiskPipeline(cfg).run(
        model=model,
        safety_dataloader=safety_loader,
        safety_loss_fn=loss_fn,
        fine_tune_dataloader=fine_tune_generator(),
        fine_tune_loss_fn=loss_fn,
    )

    assert report.forecast.steps.shape[0] == cfg.forecast.max_steps + 1


def test_pipeline_forecast_uses_lr_scaled_curvature_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _TinyClassifier()
    cfg = PipelineConfig(mode="full", learning_rate=0.2)
    pipeline = AlignmentRiskPipeline(cfg)

    subspace = SensitivitySubspace(
        parameter_slices=[ParameterSlice(name="linear.weight", start=0, end=2, shape=(2,))],
        fisher_eigenvalues=torch.tensor([3.0, 1.0]),
        fisher_eigenvectors=torch.eye(2),
        fisher_diagonal=torch.tensor([3.0, 1.0]),
        module_scores={"linear.weight": 2.0},
        top_weight_indices=torch.tensor([0, 1]),
        top_weight_scores=torch.tensor([3.0, 1.0]),
    )

    class _FisherStub:
        def analyze(self, **kwargs: object) -> SensitivitySubspace:
            return subspace

    class _CurvatureStub:
        def analyze(self, **kwargs: object) -> CurvatureCouplingResult:
            return CurvatureCouplingResult(
                gamma_hat=7.0,
                epsilon_hat=5.0,
                acceleration_norm=0.0,
                projected_acceleration_norm=0.0,
            )

    captured: dict[str, float] = {}

    def _fake_forecast(
        lambda_min: float,
        gamma: float,
        epsilon: float,
        config: object,
    ) -> SafetyForecast:
        captured["lambda_min"] = lambda_min
        captured["gamma"] = gamma
        captured["epsilon"] = epsilon
        return SafetyForecast(
            steps=np.array([0.0]),
            times=np.array([0.0]),
            projected_drift=np.array([0.0]),
            quartic_lower_bound=np.array([0.0]),
            estimated_loss=np.array([0.0]),
            collapse_step=None,
            collapse_time=None,
        )

    pipeline.fisher_analyzer = _FisherStub()  # type: ignore[assignment]
    pipeline.curvature_analyzer = _CurvatureStub()  # type: ignore[assignment]
    monkeypatch.setattr(
        "alignment_risk.pipeline.forecast_stability",
        _fake_forecast,
    )
    monkeypatch.setattr(
        pipeline,
        "_estimate_initial_update",
        lambda *args, **kwargs: torch.zeros(2),
    )

    pipeline.run(
        model=model,
        safety_dataloader=[],
        safety_loss_fn=lambda m, b: torch.tensor(0.0),
        fine_tune_dataloader=[None],
        fine_tune_loss_fn=lambda m, b: torch.tensor(0.0),
    )

    assert captured["lambda_min"] == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert captured["epsilon"] == pytest.approx(0.2 * 5.0, rel=1e-6, abs=1e-6)
    assert captured["gamma"] == pytest.approx((0.2**2) * 7.0, rel=1e-6, abs=1e-6)


def test_pipeline_adaptive_curvature_refinement_can_override_borderline_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyClassifier()
    cfg = PipelineConfig(mode="full", learning_rate=0.1)
    cfg.curvature.max_batches = 1
    cfg.adaptive_curvature_refine = True
    cfg.adaptive_curvature_max_batches = 4
    cfg.adaptive_curvature_trigger_fraction = 0.5
    cfg.forecast.collapse_loss_threshold = 0.1
    pipeline = AlignmentRiskPipeline(cfg)

    subspace = SensitivitySubspace(
        parameter_slices=[ParameterSlice(name="linear.weight", start=0, end=2, shape=(2,))],
        fisher_eigenvalues=torch.tensor([2.0, 1.0]),
        fisher_eigenvectors=torch.eye(2),
        fisher_diagonal=torch.tensor([2.0, 1.0]),
        module_scores={"linear.weight": 1.5},
        top_weight_indices=torch.tensor([0, 1]),
        top_weight_scores=torch.tensor([2.0, 1.0]),
    )

    class _FisherStub:
        def analyze(self, **kwargs: object) -> SensitivitySubspace:
            return subspace

    class _CurvatureStub:
        def analyze(self, **kwargs: object) -> CurvatureCouplingResult:
            override = kwargs.get("max_batches_override")
            if override is None:
                return CurvatureCouplingResult(  # borderline/high
                    gamma_hat=4.0,
                    epsilon_hat=2.0,
                    acceleration_norm=0.0,
                    projected_acceleration_norm=0.0,
                )
            return CurvatureCouplingResult(  # refined/lower risk
                gamma_hat=1.0,
                epsilon_hat=1.0,
                acceleration_norm=0.0,
                projected_acceleration_norm=0.0,
            )

    calls: list[float] = []

    def _fake_forecast(
        lambda_min: float,
        gamma: float,
        epsilon: float,
        config: object,
    ) -> SafetyForecast:
        calls.append(gamma)
        # First call should trigger refinement, second should not.
        if len(calls) == 1:
            est = np.array([0.06])
        else:
            est = np.array([0.01])
        return SafetyForecast(
            steps=np.array([0.0]),
            times=np.array([0.0]),
            projected_drift=np.array([0.0]),
            quartic_lower_bound=np.array([0.0]),
            estimated_loss=est,
            collapse_step=None,
            collapse_time=None,
        )

    pipeline.fisher_analyzer = _FisherStub()  # type: ignore[assignment]
    pipeline.curvature_analyzer = _CurvatureStub()  # type: ignore[assignment]
    monkeypatch.setattr("alignment_risk.pipeline.forecast_stability", _fake_forecast)
    monkeypatch.setattr(pipeline, "_estimate_initial_update", lambda *args, **kwargs: torch.zeros(2))

    report = pipeline.run(
        model=model,
        safety_dataloader=[],
        safety_loss_fn=lambda m, b: torch.tensor(0.0),
        fine_tune_dataloader=[None],
        fine_tune_loss_fn=lambda m, b: torch.tensor(0.0),
    )

    assert len(calls) == 2
    assert report.curvature.gamma_hat == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert "automatically refined" in report.warning


def test_initial_update_respects_force_eval_and_restores_training_mode() -> None:
    class _DropModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.drop = torch.nn.Dropout(p=0.5)
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(self.drop(x))

    model = _DropModel()
    model.train()
    cfg = PipelineConfig(mode="full", learning_rate=1e-3)
    cfg.curvature.force_eval = True
    pipeline = AlignmentRiskPipeline(cfg)

    x = torch.randn(4, 4)
    y = torch.randint(0, 2, (4,))

    out = pipeline._estimate_initial_update(
        model=model,
        dataloader=[(x, y)],
        loss_fn=lambda m, b: torch.nn.functional.cross_entropy(m(b[0]), b[1]),  # type: ignore[index]
        selected_names=["linear.weight", "linear.bias"],
        learning_rate=cfg.learning_rate,
    )

    assert out.numel() == model.linear.weight.numel() + model.linear.bias.numel()
    assert model.training


def test_pipeline_rejects_invalid_adaptive_refinement_fraction() -> None:
    pipeline = AlignmentRiskPipeline(PipelineConfig())
    pipeline.config.adaptive_curvature_trigger_fraction = 1.5
    with pytest.raises(ValueError, match="adaptive_curvature_trigger_fraction"):
        pipeline._validate_runtime_config()


def test_pipeline_lora_non_strict_returns_skipped_report_when_no_lora_params() -> None:
    model = _TinyClassifier()
    cfg = PipelineConfig(mode="lora", require_lora_match=False)
    cfg.forecast.max_steps = 5
    report = AlignmentRiskPipeline(cfg).run(
        model=model,
        safety_dataloader=[],
        safety_loss_fn=lambda m, b: torch.tensor(0.0),
        fine_tune_dataloader=[],
        fine_tune_loss_fn=lambda m, b: torch.tensor(0.0),
    )

    assert report.subspace.parameter_slices == []
    assert report.curvature.gamma_hat == 0.0
    assert report.forecast.collapse_step is None
    assert "skipped" in report.warning.lower()


def test_pipeline_skipped_lora_report_never_sets_collapse_at_zero_threshold() -> None:
    model = _TinyClassifier()
    cfg = PipelineConfig(mode="lora", require_lora_match=False)
    cfg.forecast.max_steps = 5
    cfg.forecast.collapse_loss_threshold = 0.0

    report = AlignmentRiskPipeline(cfg).run(
        model=model,
        safety_dataloader=[],
        safety_loss_fn=lambda m, b: torch.tensor(0.0),
        fine_tune_dataloader=[],
        fine_tune_loss_fn=lambda m, b: torch.tensor(0.0),
    )

    assert report.forecast.collapse_step is None
    assert report.forecast.collapse_time is None
    assert "skipped" in report.warning.lower()


def test_pipeline_restores_original_model_device_after_run(monkeypatch: pytest.MonkeyPatch) -> None:
    class _LogicalDeviceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.logical_device = "cpu"
            self.to_calls: list[str] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def to(self, device: object, *args: object, **kwargs: object):  # type: ignore[override]
            _ = args
            _ = kwargs
            self.logical_device = str(device)
            self.to_calls.append(self.logical_device)
            return self

    model = _LogicalDeviceModel()
    cfg = PipelineConfig(mode="full")
    pipeline = AlignmentRiskPipeline(cfg)

    subspace = SensitivitySubspace(
        parameter_slices=[ParameterSlice(name="weight", start=0, end=1, shape=(1,))],
        fisher_eigenvalues=torch.tensor([1.0]),
        fisher_eigenvectors=torch.tensor([[1.0]]),
        fisher_diagonal=torch.tensor([1.0]),
        module_scores={"weight": 1.0},
        top_weight_indices=torch.tensor([0]),
        top_weight_scores=torch.tensor([1.0]),
    )

    class _FisherStub:
        def analyze(self, **kwargs: object) -> SensitivitySubspace:
            _ = kwargs
            model.to("cuda")
            return subspace

    class _CurvatureStub:
        def analyze(self, **kwargs: object) -> CurvatureCouplingResult:
            _ = kwargs
            return CurvatureCouplingResult(
                gamma_hat=0.0,
                epsilon_hat=0.0,
                acceleration_norm=0.0,
                projected_acceleration_norm=0.0,
            )

    pipeline.fisher_analyzer = _FisherStub()  # type: ignore[assignment]
    pipeline.curvature_analyzer = _CurvatureStub()  # type: ignore[assignment]
    monkeypatch.setattr(pipeline, "_estimate_initial_update", lambda *args, **kwargs: torch.zeros(1))
    monkeypatch.setattr(
        pipeline,
        "_uniform_model_device",
        lambda m: torch.device(m.logical_device),  # type: ignore[arg-type]
    )

    pipeline.run(
        model=model,
        safety_dataloader=[None],
        safety_loss_fn=lambda m, b: torch.tensor(0.0),
        fine_tune_dataloader=[None],
        fine_tune_loss_fn=lambda m, b: torch.tensor(0.0),
    )

    assert model.logical_device == "cpu"
    assert "cuda" in model.to_calls
