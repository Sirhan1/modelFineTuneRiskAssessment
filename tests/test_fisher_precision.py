import torch

from alignment_risk.fisher import FisherConfig, FisherSubspaceAnalyzer


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(6))


class _HalfGradientAnalyzer(FisherSubspaceAnalyzer):
    def _collect_per_sample_gradients(  # type: ignore[override]
        self,
        model: torch.nn.Module,
        parameter_names: list[str],
        params: list[torch.nn.Parameter],
        dataloader: object,
        loss_fn: object,
        max_samples: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        n = sum(p.numel() for p in params)
        return [torch.randn(n, dtype=torch.float16) for _ in range(4)]


def test_fisher_analysis_promotes_half_precision_gradients_for_cpu_svd() -> None:
    analyzer = _HalfGradientAnalyzer(
        FisherConfig(
            top_rank=3,
            target_explained_variance=None,
            max_samples=4,
            max_parameters=128,
            top_weight_count=4,
        )
    )
    model = _TinyModel()

    # Loss and dataloader are unused by the override but required by API.
    out = analyzer.analyze(model, dataloader=[], loss_fn=lambda m, b: torch.tensor(0.0))

    assert out.fisher_eigenvalues.dtype == torch.float32
    assert out.fisher_diagonal.dtype == torch.float32
    assert out.fisher_eigenvalues.numel() == 3
