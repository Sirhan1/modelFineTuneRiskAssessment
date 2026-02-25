from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from .types import SafetyForecast, SensitivitySubspace


def sensitivity_rows(
    subspace: SensitivitySubspace,
    *,
    top_k: int = 128,
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    k = min(top_k, subspace.top_weight_indices.numel())
    for i in range(k):
        flat_idx = int(subspace.top_weight_indices[i].item())
        score = float(subspace.top_weight_scores[i].item())
        param_name, local_idx = _flat_to_named_index(subspace, flat_idx)
        rows.append(
            {
                "rank": i + 1,
                "flat_index": flat_idx,
                "parameter": param_name,
                "parameter_offset": local_idx,
                "score": score,
            }
        )
    return rows


def plot_module_sensitivity(subspace: SensitivitySubspace, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    names = list(subspace.module_scores.keys())
    values = [subspace.module_scores[n] for n in names]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
    plt.ylabel("Mean Fisher diagonal")
    plt.title("Sensitivity Map (module-level)")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_safety_forecast(forecast: SafetyForecast, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    x = forecast.times
    plt.figure(figsize=(8, 4))
    plt.plot(x, forecast.estimated_loss, label="Estimated Safety Decay")
    plt.plot(x, forecast.quartic_lower_bound, label="Quartic Asymptote", linestyle="--")
    if forecast.collapse_step is not None:
        assert forecast.collapse_time is not None
        plt.axvline(
            forecast.collapse_time,
            color="red",
            linestyle=":",
            label=f"Collapse step {forecast.collapse_step}",
        )
    plt.xlabel("Fine-tuning time")
    plt.ylabel("Predicted alignment loss")
    plt.title("AIC Stability Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def _flat_to_named_index(subspace: SensitivitySubspace, flat_index: int) -> tuple[str, int]:
    for p in subspace.parameter_slices:
        if p.start <= flat_index < p.end:
            return p.name, flat_index - p.start
    return "<unknown>", flat_index
