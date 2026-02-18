# alignment-risk (template)

A Python package template for pre-training risk diagnostics inspired by
*The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety* (arXiv:2602.15799).

This template implements an MVP pipeline for the four checks you asked for:

1. **Alignment Sharpness (Low-Rank Sensitivity)**
   - Estimates a skill-specific Fisher Information Matrix (FIM).
   - Extracts top Fisher eigenvectors as the alignment-sensitive subspace.
   - Produces a sensitivity map (module scores + top sensitive weights).

2. **False Sense of Security (Initial Orthogonality)**
   - Computes overlap / cosine to the sensitive subspace for the initial update.
   - Emits an `InitialRiskScore` and trigger flag for second-order checks.

3. **Smoke Detector (Curvature Coupling)**
   - Estimates acceleration term `(∇g)g = H g` using Hessian-vector products.
   - Computes AIC-inspired coupling score `gamma_hat = ||F^(1/2) P (H g)||`.

4. **Quartic Warning**
   - Forecasts safety decay from the paper's local form:
     - Drift lower bound: `||F^(1/2)PΔθ(t)|| ≳ (gamma/2) t^2 - epsilon t`
     - Quartic asymptote: `Δu(t) ≈ (lambda * gamma^2 / 8) t^4`
   - Emits `Collapse Predicted at Step X` when threshold is crossed.

## Install

```bash
pip install -e .
```

## Quick start

Run the synthetic demo:

```bash
alignment-risk demo --output-dir artifacts
```

This writes:

- `artifacts/sensitivity_map.png`
- `artifacts/safety_decay_forecast.png`

## Use with your model

You provide:

- a `torch.nn.Module`
- a safety dataloader + loss function (for FIM of a safety skill)
- a fine-tuning dataloader + loss function (for update + curvature checks)

```python
import torch
from alignment_risk import AlignmentRiskPipeline, PipelineConfig

pipeline = AlignmentRiskPipeline(PipelineConfig())
report = pipeline.run(
    model=model,
    safety_dataloader=safety_loader,
    safety_loss_fn=safety_loss_fn,
    fine_tune_dataloader=ft_loader,
    fine_tune_loss_fn=ft_loss_fn,
)

print(report.warning)
print(report.initial_risk)
print(report.curvature)
```

## Notes

- This is a **template** implementation; full-scale LLM use needs batching, parameter sharding,
  and matrix-free eigensolvers.
- For an MVP, constrain analysis to selected parameter groups via `FisherConfig.parameter_names`.
- The forecast is a local diagnostic approximation, not a formal safety guarantee.
