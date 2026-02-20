from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Sequence

import torch

from .curvature import CurvatureConfig, CurvatureCouplingAnalyzer
from .fisher import FisherConfig, FisherSubspaceAnalyzer
from .forecast import ForecastConfig, forecast_stability
from .mitigation import AlignGuardConfig, AlignGuardLoRARegularizer
from .orthogonality import initial_risk_from_update
from .types import RiskAssessmentReport, SensitivitySubspace
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
        selected_param_names = self._resolve_analysis_parameter_names(model)

        subspace = self.fisher_analyzer.analyze(
            model=model,
            dataloader=safety_dataloader,
            loss_fn=safety_loss_fn,
            parameter_names=selected_param_names,
        )

        initial_update = self._estimate_initial_update(
            model=model,
            dataloader=fine_tune_dataloader,
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
            dataloader=fine_tune_dataloader,
            loss_fn=fine_tune_loss_fn,
            subspace=subspace,
        )

        lambda_min = float(subspace.fisher_eigenvalues[-1].item())
        forecast = forecast_stability(
            lambda_min=lambda_min,
            gamma=curvature.gamma_hat,
            epsilon=initial_risk.projected_norm,
            config=self.config.forecast,
        )

        warning = self._build_warning(forecast.collapse_step, initial_risk.trigger_curvature_check)

        return RiskAssessmentReport(
            subspace=subspace,
            initial_risk=initial_risk,
            curvature=curvature,
            forecast=forecast,
            warning=warning,
        )

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

        for batch in dataloader:
            batch = move_to_device(batch, device)
            loss = loss_fn(model, batch)
            if loss.ndim > 0:
                loss = loss.mean()
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            grad_vec = flatten_tensors(grads, params).detach().cpu()
            return -learning_rate * grad_vec

        raise ValueError("Fine-tuning dataloader yielded no batches.")

    @staticmethod
    def _build_warning(collapse_step: int | None, triggered_curvature_check: bool) -> str:
        if collapse_step is not None:
            return f"Collapse Predicted at Step {collapse_step}."
        if triggered_curvature_check:
            return "Low initial overlap detected: running curvature check was necessary." \
                " No collapse crossed threshold in the current horizon."
        return "No collapse predicted within the configured step horizon."
