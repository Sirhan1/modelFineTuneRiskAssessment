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


def test_curvature_mean_loss_is_sample_weighted_across_variable_batch_sizes() -> None:
    class _ScalarModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.w

    model = _ScalarModel()
    analyzer = CurvatureCouplingAnalyzer(CurvatureConfig(max_batches=10, device="cpu"))

    batch_large = (torch.ones(4, 1), torch.zeros(4, 1))
    batch_small = (torch.full((1, 1), 10.0), torch.zeros(1, 1))

    def loss_fn(model_: torch.nn.Module, batch: object) -> torch.Tensor:
        x, y = batch  # type: ignore[misc]
        pred = model_(x)
        return ((pred - y) ** 2).mean()

    mean_loss = analyzer._mean_loss(
        model,
        dataloader=[batch_large, batch_small],
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        max_batches=10,
    )

    x_all = torch.cat([batch_large[0], batch_small[0]], dim=0)
    y_all = torch.cat([batch_large[1], batch_small[1]], dim=0)
    expected = loss_fn(model, (x_all, y_all))

    assert float(mean_loss.item()) == pytest.approx(float(expected.item()), rel=1e-6, abs=1e-6)


def test_curvature_respects_subspace_parameter_order() -> None:
    class _TwoParamModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor([1.0]))
            self.b = torch.nn.Parameter(torch.tensor([2.0]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = _TwoParamModel()
    # Intentionally reverse order relative to model.named_parameters().
    subspace = SensitivitySubspace(
        parameter_slices=[
            ParameterSlice(name="b", start=0, end=1, shape=(1,)),
            ParameterSlice(name="a", start=1, end=2, shape=(1,)),
        ],
        fisher_eigenvalues=torch.tensor([4.0, 1.0]),
        fisher_eigenvectors=torch.eye(2),
        fisher_diagonal=torch.tensor([4.0, 1.0]),
        module_scores={"b": 4.0, "a": 1.0},
        top_weight_indices=torch.tensor([0, 1]),
        top_weight_scores=torch.tensor([4.0, 1.0]),
    )

    analyzer = CurvatureCouplingAnalyzer(CurvatureConfig(max_batches=1, device="cpu"))

    def loss_fn(model_: torch.nn.Module, batch: object) -> torch.Tensor:
        _ = batch
        # g = [d/da, d/db] = [10a, b] = [10, 2]
        return 0.5 * (10.0 * (model_.a**2).sum() + (model_.b**2).sum())

    out = analyzer.analyze(model, dataloader=[None], loss_fn=loss_fn, subspace=subspace)
    # Subspace order is [b, a], so weighted norm is sqrt(4*2^2 + 1*10^2).
    expected = (4.0 * (2.0**2) + 1.0 * (10.0**2)) ** 0.5
    assert out.epsilon_hat == pytest.approx(expected, rel=1e-6, abs=1e-6)
