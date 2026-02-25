import pytest
import torch

from alignment_risk.mitigation import AlignGuardConfig, AlignGuardLoRARegularizer, decompose_update
from alignment_risk.pipeline import AlignmentRiskPipeline, PipelineConfig
from alignment_risk.types import SensitivitySubspace
from alignment_risk.utils import build_parameter_slices, named_trainable_parameters


class _TinyLoraModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(4, 4)
        self.lora_A = torch.nn.Parameter(torch.randn(2, 4) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.randn(4, 2) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = x @ self.lora_A.t() @ self.lora_B.t()
        return self.base(x) + delta


def _toy_subspace(model: torch.nn.Module, names: list[str]) -> SensitivitySubspace:
    selected_names, params = named_trainable_parameters(model, include_names=names)
    slices = build_parameter_slices(selected_names, params)
    n = sum(p.numel() for p in params)

    d = min(2, n)
    q, _ = torch.linalg.qr(torch.randn(n, d))
    eigvals = torch.linspace(1.0, 0.5, d)
    fisher_diag = torch.linspace(0.1, 1.0, n)

    module_scores = {
        s.name: float(fisher_diag[s.start : s.end].mean().item())
        for s in slices
    }
    top_scores, top_idx = torch.topk(fisher_diag, k=min(4, n))

    return SensitivitySubspace(
        parameter_slices=slices,
        fisher_eigenvalues=eigvals,
        fisher_eigenvectors=q,
        fisher_diagonal=fisher_diag,
        module_scores=module_scores,
        top_weight_indices=top_idx,
        top_weight_scores=top_scores,
    )


def test_decompose_update_reconstructs_vector() -> None:
    basis, _ = torch.linalg.qr(torch.randn(8, 3))
    vector = torch.randn(8)

    delta_a, delta_t = decompose_update(vector, basis)

    assert torch.allclose(delta_a + delta_t, vector, atol=1e-6)
    assert abs(float(torch.dot(delta_a, delta_t).item())) < 1e-5


def test_alignguard_regularizer_adds_penalties_after_drift() -> None:
    model = _TinyLoraModel()
    selected = ["lora_A", "lora_B"]
    subspace = _toy_subspace(model, selected)

    regularizer = AlignGuardLoRARegularizer(
        model,
        subspace=subspace,
        parameter_names=selected,
        config=AlignGuardConfig(lambda_a=0.2, lambda_t=0.3, lambda_nc=0.1),
    )

    x = torch.randn(4, 4)
    y = torch.randn(4, 4)

    first = regularizer.regularized_loss(torch.nn.functional.mse_loss(model(x), y))
    assert torch.isclose(first.alignment_penalty, torch.tensor(0.0), atol=1e-8)
    assert torch.isclose(first.task_stability_penalty, torch.tensor(0.0), atol=1e-8)
    assert torch.isclose(first.collision_penalty, torch.tensor(0.0), atol=1e-8)

    with torch.no_grad():
        model.lora_A.add_(0.05)

    second = regularizer.regularized_loss(torch.nn.functional.mse_loss(model(x), y))
    assert float(second.alignment_penalty.item()) > 0.0
    assert float(second.task_stability_penalty.item()) > 0.0
    assert float(second.total_loss.item()) > float(second.task_loss.item())


def test_custom_h_type_requires_diagonal() -> None:
    model = _TinyLoraModel()
    selected = ["lora_A", "lora_B"]
    subspace = _toy_subspace(model, selected)

    regularizer = AlignGuardLoRARegularizer(
        model,
        subspace=subspace,
        parameter_names=selected,
        config=AlignGuardConfig(h_type="custom"),
    )

    loss = torch.nn.functional.mse_loss(model(torch.randn(2, 4)), torch.randn(2, 4))
    with pytest.raises(ValueError, match="h_type='custom'"):
        regularizer.regularized_loss(loss)


def test_pipeline_build_lora_mitigator() -> None:
    model = _TinyLoraModel()
    selected = ["lora_A", "lora_B"]
    subspace = _toy_subspace(model, selected)

    pipeline = AlignmentRiskPipeline(PipelineConfig(mode="lora"))
    mitigator = pipeline.build_lora_mitigator(model, subspace)
    assert isinstance(mitigator, AlignGuardLoRARegularizer)


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (AlignGuardConfig(lambda_a=-0.1), "lambda_a"),
        (AlignGuardConfig(alpha=1.1), "alpha"),
        (AlignGuardConfig(epsilon=0.0), "epsilon"),
    ],
)
def test_alignguard_rejects_invalid_hyperparameters(
    cfg: AlignGuardConfig,
    match: str,
) -> None:
    model = _TinyLoraModel()
    selected = ["lora_A", "lora_B"]
    subspace = _toy_subspace(model, selected)

    with pytest.raises(ValueError, match=match):
        AlignGuardLoRARegularizer(
            model,
            subspace=subspace,
            parameter_names=selected,
            config=cfg,
        )
