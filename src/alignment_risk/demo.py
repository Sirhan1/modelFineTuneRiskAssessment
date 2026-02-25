from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import torch
from torch.utils.data import DataLoader, TensorDataset

from .pipeline import AlignmentRiskPipeline, PipelineConfig
from .utils import resolve_device
from .visualization import plot_module_sensitivity, plot_safety_forecast, sensitivity_rows


class ToyLoraLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        delta_weight = self.lora_B @ self.lora_A
        delta = torch.nn.functional.linear(x, delta_weight, None)
        return base + delta


def _set_lora_only_trainable(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if "lora_" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False


def _fmt_sci(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}e}"


def run_demo(output_dir: str = "artifacts", *, mode: Literal["full", "lora"] = "full") -> None:
    torch.manual_seed(7)
    device = resolve_device("auto")

    model = torch.nn.Sequential(
        ToyLoraLinear(16, 32, rank=4),
        torch.nn.Tanh(),
        ToyLoraLinear(32, 2, rank=2),
    )
    if mode == "lora":
        _set_lora_only_trainable(model)

    x_safety = torch.randn(64, 16)
    y_safety = (x_safety[:, :4].sum(dim=1) > 0).long()

    x_ft = torch.randn(64, 16)
    y_ft = (x_ft[:, 4:8].sum(dim=1) > 0).long()

    safety_loader = DataLoader(TensorDataset(x_safety, y_safety), batch_size=16, shuffle=False)
    ft_loader = DataLoader(TensorDataset(x_ft, y_ft), batch_size=16, shuffle=False)

    def loss_fn(model_: torch.nn.Module, batch: object) -> torch.Tensor:
        x, y = cast(tuple[torch.Tensor, torch.Tensor], batch)
        logits = model_(x)
        return torch.nn.functional.cross_entropy(logits, y)

    config = PipelineConfig(
        learning_rate=1e-2,
    )
    config.mode = mode
    config.fisher.device = str(device)
    config.curvature.device = str(device)
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
    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print(f"Initial overlap cosine: {report.initial_risk.cosine_to_subspace:.4f}")
    print(f"Curvature coupling gamma_hat: {_fmt_sci(report.curvature.gamma_hat)}")
    print(f"Curvature coupling epsilon_hat: {_fmt_sci(report.curvature.epsilon_hat)}")
    print(f"Acceleration norm: {_fmt_sci(report.curvature.acceleration_norm)}")
    print(
        "Projected acceleration norm: "
        f"{_fmt_sci(report.curvature.projected_acceleration_norm)}"
    )
    print(report.warning)
    print("Top-10 sensitive weights:")
    for row in sensitivity_rows(report.subspace, top_k=10):
        score = float(row["score"])
        print(
            f"  #{row['rank']}: {row['parameter']}[{row['parameter_offset']}]"
            f" score={score:.6e}"
        )


if __name__ == "__main__":
    run_demo()
