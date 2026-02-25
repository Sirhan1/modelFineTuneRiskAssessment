import pytest
import torch

from alignment_risk.curvature import CurvatureConfig, CurvatureCouplingAnalyzer
from alignment_risk.types import ParameterSlice, SensitivitySubspace


class _QuadraticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_curvature_epsilon_hat_is_fisher_weighted_projection() -> None:
    model = _QuadraticModel()

    subspace = SensitivitySubspace(
        parameter_slices=[ParameterSlice(name="w", start=0, end=2, shape=(2,))],
        fisher_eigenvalues=torch.tensor([4.0, 1.0]),
        fisher_eigenvectors=torch.eye(2),
        fisher_diagonal=torch.tensor([4.0, 1.0]),
        module_scores={"w": 2.5},
        top_weight_indices=torch.tensor([0, 1]),
        top_weight_scores=torch.tensor([4.0, 1.0]),
    )

    analyzer = CurvatureCouplingAnalyzer(CurvatureConfig(max_batches=1, device="cpu"))

    def loss_fn(model_: torch.nn.Module, batch: object) -> torch.Tensor:
        # L = 0.5 * ||w||^2 -> g = w, H = I, so H g = g.
        return 0.5 * (model_.w ** 2).sum()

    out = analyzer.analyze(model, dataloader=[None], loss_fn=loss_fn, subspace=subspace)

    # sqrt(4*2^2 + 1*3^2) = sqrt(25) = 5
    assert out.epsilon_hat == pytest.approx(5.0, rel=1e-6, abs=1e-6)
    assert out.gamma_hat == pytest.approx(5.0, rel=1e-6, abs=1e-6)


def test_curvature_handles_mixed_precision_subspace_projection() -> None:
    class _HalfQuadraticModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.float16))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = _HalfQuadraticModel()
    subspace = SensitivitySubspace(
        parameter_slices=[ParameterSlice(name="w", start=0, end=2, shape=(2,))],
        fisher_eigenvalues=torch.tensor([4.0, 1.0], dtype=torch.float32),
        fisher_eigenvectors=torch.eye(2, dtype=torch.float32),
        fisher_diagonal=torch.tensor([4.0, 1.0], dtype=torch.float32),
        module_scores={"w": 2.5},
        top_weight_indices=torch.tensor([0, 1]),
        top_weight_scores=torch.tensor([4.0, 1.0]),
    )

    analyzer = CurvatureCouplingAnalyzer(CurvatureConfig(max_batches=1, device="cpu"))

    def loss_fn(model_: torch.nn.Module, batch: object) -> torch.Tensor:
        _ = batch
        return 0.5 * (model_.w**2).sum()

    out = analyzer.analyze(model, dataloader=[None], loss_fn=loss_fn, subspace=subspace)
    assert out.epsilon_hat == pytest.approx(5.0, rel=1e-3, abs=1e-3)
    assert out.gamma_hat == pytest.approx(5.0, rel=1e-3, abs=1e-3)
