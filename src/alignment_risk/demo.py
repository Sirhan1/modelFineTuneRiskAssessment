from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .pipeline import AlignmentRiskPipeline, PipelineConfig
from .visualization import plot_module_sensitivity, plot_safety_forecast, sensitivity_rows


def run_demo(output_dir: str = "artifacts") -> None:
    torch.manual_seed(7)

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 2),
    )

    x_safety = torch.randn(64, 16)
    y_safety = (x_safety[:, :4].sum(dim=1) > 0).long()

    x_ft = torch.randn(64, 16)
    y_ft = (x_ft[:, 4:8].sum(dim=1) > 0).long()

    safety_loader = DataLoader(TensorDataset(x_safety, y_safety), batch_size=16, shuffle=False)
    ft_loader = DataLoader(TensorDataset(x_ft, y_ft), batch_size=16, shuffle=False)

    def loss_fn(model_: torch.nn.Module, batch: object) -> torch.Tensor:
        x, y = batch  # type: ignore[misc]
        logits = model_(x)
        return torch.nn.functional.cross_entropy(logits, y)

    config = PipelineConfig(
        learning_rate=1e-2,
    )
    config.fisher.max_parameters = 20_000
    config.fisher.max_samples = 32
    config.fisher.top_rank = 8
    config.forecast.max_steps = 800
    config.forecast.collapse_loss_threshold = 0.02

    report = AlignmentRiskPipeline(config).run(
        model=model,
        safety_dataloader=safety_loader,
        safety_loss_fn=loss_fn,
        fine_tune_dataloader=ft_loader,
        fine_tune_loss_fn=loss_fn,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_module_sensitivity(report.subspace, out_dir / "sensitivity_map.png")
    plot_safety_forecast(report.forecast, out_dir / "safety_decay_forecast.png")

    print("=== Alignment Risk Demo ===")
    print(f"Initial overlap cosine: {report.initial_risk.cosine_to_subspace:.4f}")
    print(f"Curvature coupling gamma_hat: {report.curvature.gamma_hat:.4f}")
    print(report.warning)
    print("Top-10 sensitive weights:")
    for row in sensitivity_rows(report.subspace, top_k=10):
        print(
            f"  #{row['rank']}: {row['parameter']}[{row['parameter_offset']}]"
            f" score={row['score']:.6f}"
        )


if __name__ == "__main__":
    run_demo()
