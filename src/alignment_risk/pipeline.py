"""End-to-end orchestration of AIC diagnostics and optional LoRA mitigation.

Academic grounding:
- [AIC-2026] https://arxiv.org/pdf/2602.15799
- [ALIGNGUARD-2025] https://arxiv.org/pdf/2508.02079

See docs/SOURCES.md for section/page-level mapping to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
import torch

from .curvature import CurvatureConfig, CurvatureCouplingAnalyzer
from .fisher import FisherConfig, FisherSubspaceAnalyzer
from .forecast import ForecastConfig, forecast_stability
from .mitigation import AlignGuardConfig, AlignGuardLoRARegularizer
from .orthogonality import initial_risk_from_update
from .types import (
    CurvatureCouplingResult,
    InitialRiskScore,
    RiskAssessmentReport,
    SafetyForecast,
    SensitivitySubspace,
)
from .utils import (
    flatten_tensors,
    move_to_device,
    named_trainable_parameters,
    resolve_device,
    select_parameter_names_for_mode,
)

LossFn = Callable[[torch.nn.Module, object], torch.Tensor]
PipelineMode = Literal["full", "lora"]


@dataclass
class PipelineConfig:
    fisher: FisherConfig = field(default_factory=FisherConfig)
    curvature: CurvatureConfig = field(default_factory=CurvatureConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    mode: PipelineMode = "full"
    lora_name_markers: Sequence[str] = ("lora_", "lora_A", "lora_B")
    require_lora_match: bool = True
    learning_rate: float = 1e-4
    orthogonality_threshold: float = 0.05
    adaptive_curvature_refine: bool = True
    adaptive_curvature_max_batches: int = 4
    adaptive_curvature_trigger_fraction: float = 0.5
    trust_region_warning_ratio: float = 5.0
    restore_model_device: bool = True


class AlignmentRiskPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.fisher_analyzer = FisherSubspaceAnalyzer(self.config.fisher)
        self.curvature_analyzer = CurvatureCouplingAnalyzer(self.config.curvature)

    def run(
        self,
        model: torch.nn.Module,
        safety_dataloader: Iterable[object],
        safety_loss_fn: LossFn,
        fine_tune_dataloader: Iterable[object],
        fine_tune_loss_fn: LossFn,
    ) -> RiskAssessmentReport:
        self._validate_runtime_config()
        original_device = self._uniform_model_device(model)
        try:
            selected_param_names = self._resolve_analysis_parameter_names(model)
            if not selected_param_names:
                if self.config.mode == "lora" and not self.config.require_lora_match:
                    return self._build_skipped_lora_report()
                raise ValueError("No trainable parameters selected for analysis.")

            fine_tune_batches = self._collect_fine_tune_batches(fine_tune_dataloader)
            if not fine_tune_batches:
                raise ValueError("Fine-tuning dataloader yielded no batches.")

            subspace = self.fisher_analyzer.analyze(
                model=model,
                dataloader=safety_dataloader,
                loss_fn=safety_loss_fn,
                parameter_names=selected_param_names,
            )

            initial_update = self._estimate_initial_update(
                model=model,
                dataloader=fine_tune_batches[:1],
                loss_fn=fine_tune_loss_fn,
                selected_names=[p.name for p in subspace.parameter_slices],
                learning_rate=self.config.learning_rate,
            )

            initial_risk = initial_risk_from_update(
                initial_update,
                subspace,
                orthogonality_threshold=self.config.orthogonality_threshold,
            )

            curvature = self.curvature_analyzer.analyze(
                model=model,
                dataloader=fine_tune_batches,
                loss_fn=fine_tune_loss_fn,
                subspace=subspace,
            )

            lambda_min = self._stable_lambda_min(subspace)
            forecast = self._forecast_from_curvature(lambda_min, curvature)
            curvature_refined = False
            if self._should_refine_curvature(forecast):
                refined_batches = max(
                    self.config.curvature.max_batches + 1,
                    self.config.adaptive_curvature_max_batches,
                )
                curvature = self.curvature_analyzer.analyze(
                    model=model,
                    dataloader=fine_tune_batches[:refined_batches],
                    loss_fn=fine_tune_loss_fn,
                    subspace=subspace,
                    max_batches_override=refined_batches,
                )
                forecast = self._forecast_from_curvature(lambda_min, curvature)
                curvature_refined = True

            trust_ratio = float(
                forecast.projected_drift.max()
                / max(initial_risk.update_norm, 1e-12)
            )
            local_validity_warning = trust_ratio >= self.config.trust_region_warning_ratio
            warning = self._build_warning(
                forecast.collapse_step,
                initial_risk.trigger_curvature_check,
                curvature_refined=curvature_refined,
                local_validity_warning=local_validity_warning,
            )

            return RiskAssessmentReport(
                subspace=subspace,
                initial_risk=initial_risk,
                curvature=curvature,
                forecast=forecast,
                warning=warning,
            )
        finally:
            if self.config.restore_model_device:
                self._restore_model_device(model, original_device)

    def _forecast_from_curvature(
        self,
        lambda_min: float,
        curvature: CurvatureCouplingResult,
    ) -> SafetyForecast:
        # Map local gradient/curvature terms to per-step displacement scale.
        epsilon = self.config.learning_rate * curvature.epsilon_hat
        gamma = (self.config.learning_rate ** 2) * curvature.gamma_hat
        return forecast_stability(
            lambda_min=lambda_min,
            gamma=gamma,
            epsilon=epsilon,
            config=self.config.forecast,
        )

    def _stable_lambda_min(self, subspace: SensitivitySubspace) -> float:
        eigvals = torch.clamp(subspace.fisher_eigenvalues, min=0.0)
        if eigvals.numel() == 0:
            return 0.0
        # Use a true lower-quantile floor so the "lambda_min" proxy never exceeds
        # the smallest retained eigenvalue for small-rank spectra.
        return float(torch.quantile(eigvals, q=0.1, interpolation="lower").item())

    def _should_refine_curvature(self, forecast: SafetyForecast) -> bool:
        if not self.config.adaptive_curvature_refine:
            return False
        if self.config.adaptive_curvature_max_batches <= self.config.curvature.max_batches:
            return False

        threshold = self.config.forecast.collapse_loss_threshold
        trigger = threshold * self.config.adaptive_curvature_trigger_fraction
        estimated_loss = forecast.estimated_loss
        collapse_step = forecast.collapse_step
        max_steps = self.config.forecast.max_steps

        if collapse_step is None:
            return float(estimated_loss[-1]) >= trigger
        return int(collapse_step) >= int(max_steps * self.config.adaptive_curvature_trigger_fraction)

    def build_lora_mitigator(
        self,
        model: torch.nn.Module,
        subspace: SensitivitySubspace,
        *,
        config: AlignGuardConfig | None = None,
    ) -> AlignGuardLoRARegularizer:
        if self.config.mode != "lora":
            raise ValueError(
                "build_lora_mitigator() is intended for PipelineConfig.mode='lora'. "
                "Switch the pipeline mode or construct AlignGuardLoRARegularizer directly."
            )

        selected_names = [p.name for p in subspace.parameter_slices]
        return AlignGuardLoRARegularizer(
            model,
            subspace=subspace,
            parameter_names=selected_names,
            config=config,
        )

    def _resolve_analysis_parameter_names(self, model: torch.nn.Module) -> list[str]:
        return select_parameter_names_for_mode(
            model=model,
            mode=self.config.mode,
            include_names=self.config.fisher.parameter_names,
            lora_name_markers=self.config.lora_name_markers,
            require_lora_match=self.config.require_lora_match,
        )

    def _estimate_initial_update(
        self,
        model: torch.nn.Module,
        dataloader: Iterable[object],
        loss_fn: LossFn,
        selected_names: list[str],
        learning_rate: float,
    ) -> torch.Tensor:
        device = resolve_device(self.config.curvature.device)
        model = model.to(device)

        _, params = named_trainable_parameters(model, include_names=selected_names)
        if not params:
            raise ValueError("No matching trainable parameters found for update estimate.")

        was_training = model.training
        if self.config.curvature.force_eval:
            model.eval()
        try:
            for batch in dataloader:
                batch = move_to_device(batch, device)
                loss = loss_fn(model, batch)
                if loss.ndim > 0:
                    loss = loss.mean()
                grads = torch.autograd.grad(loss, params, allow_unused=True)
                grad_vec = flatten_tensors(grads, params).detach().cpu()
                return -learning_rate * grad_vec
        finally:
            if self.config.curvature.force_eval and was_training:
                model.train()

        raise ValueError("Fine-tuning dataloader yielded no batches.")

    def _collect_fine_tune_batches(self, dataloader: Iterable[object]) -> list[object]:
        required_batches = max(1, self.config.curvature.max_batches)
        if self.config.adaptive_curvature_refine:
            required_batches = max(required_batches, self.config.adaptive_curvature_max_batches)

        out: list[object] = []
        for batch in dataloader:
            out.append(batch)
            if len(out) >= required_batches:
                break
        return out

    def _build_skipped_lora_report(self) -> RiskAssessmentReport:
        subspace = SensitivitySubspace(
            parameter_slices=[],
            fisher_eigenvalues=torch.empty(0),
            fisher_eigenvectors=torch.empty((0, 0)),
            fisher_diagonal=torch.empty(0),
            module_scores={},
            top_weight_indices=torch.empty(0, dtype=torch.long),
            top_weight_scores=torch.empty(0),
        )
        initial_risk = InitialRiskScore(
            cosine_to_subspace=0.0,
            projected_ratio=0.0,
            update_norm=0.0,
            projected_norm=0.0,
            trigger_curvature_check=False,
        )
        curvature = CurvatureCouplingResult(
            gamma_hat=0.0,
            epsilon_hat=0.0,
            acceleration_norm=0.0,
            projected_acceleration_norm=0.0,
        )
        steps = np.arange(self.config.forecast.max_steps + 1, dtype=int)
        times = steps.astype(float) * self.config.forecast.step_size
        zeros = np.zeros_like(times)
        forecast = SafetyForecast(
            steps=steps,
            times=times,
            projected_drift=zeros,
            quartic_lower_bound=zeros,
            estimated_loss=zeros,
            collapse_step=None,
            collapse_time=None,
        )
        return RiskAssessmentReport(
            subspace=subspace,
            initial_risk=initial_risk,
            curvature=curvature,
            forecast=forecast,
            warning=(
                "No trainable LoRA parameters selected (require_lora_match=False); "
                "risk analysis was skipped."
            ),
        )

    def _uniform_model_device(self, model: torch.nn.Module) -> torch.device | None:
        params = list(model.parameters())
        if not params:
            return None
        first = params[0].device
        for param in params[1:]:
            if param.device != first:
                return None
        return first

    def _restore_model_device(
        self,
        model: torch.nn.Module,
        original_device: torch.device | None,
    ) -> None:
        if original_device is None:
            return
        current = self._uniform_model_device(model)
        if current is None or current == original_device:
            return
        model.to(original_device)

    def _validate_runtime_config(self) -> None:
        if not np.isfinite(self.config.learning_rate) or self.config.learning_rate <= 0.0:
            raise ValueError("PipelineConfig.learning_rate must be finite and > 0.")
        if not (0.0 <= self.config.orthogonality_threshold <= 1.0):
            raise ValueError("PipelineConfig.orthogonality_threshold must be in [0, 1].")
        if (
            not np.isfinite(self.config.trust_region_warning_ratio)
            or self.config.trust_region_warning_ratio <= 0.0
        ):
            raise ValueError("PipelineConfig.trust_region_warning_ratio must be finite and > 0.")

        frac = self.config.adaptive_curvature_trigger_fraction
        if not (0.0 < frac <= 1.0):
            raise ValueError(
                "PipelineConfig.adaptive_curvature_trigger_fraction must be in (0, 1]."
            )
        if self.config.adaptive_curvature_max_batches < 1:
            raise ValueError("PipelineConfig.adaptive_curvature_max_batches must be >= 1.")
        if self.config.curvature.max_batches < 1:
            raise ValueError("CurvatureConfig.max_batches must be >= 1.")

    @staticmethod
    def _build_warning(
        collapse_step: int | None,
        triggered_curvature_check: bool,
        *,
        curvature_refined: bool,
        local_validity_warning: bool,
    ) -> str:
        if collapse_step is not None:
            message = f"Collapse predicted at step {collapse_step}."
        elif triggered_curvature_check:
            message = (
                "Low initial overlap detected. "
                "No collapse crossed threshold in the current horizon."
            )
        else:
            message = "No collapse predicted within the configured step horizon."

        if curvature_refined:
            message += " Curvature estimate was automatically refined on additional batches."
        if local_validity_warning:
            message += (
                " Forecast entered a large-drift regime; local quadratic approximation may be less reliable."
            )
        return message
