# alignment-risk

`alignment-risk` is a Python package for pre-flight alignment risk diagnostics during fine-tuning.
It estimates whether updates are likely to drift into safety-sensitive directions before you commit to a run.

Full documentation site: [sirhan1.github.io/modelFineTuneRiskAssessment](https://sirhan1.github.io/modelFineTuneRiskAssessment/)

## Quick Start (1 minute)

Install:

```bash
pip install alignment-risk
```

Run the built-in demo:

```bash
alignment-risk demo --output-dir artifacts
```

Outputs:
- `artifacts/sensitivity_map.png`
- `artifacts/safety_decay_forecast.png`

## Installation

PyPI:

```bash
pip install alignment-risk
```

Local development (Apple Silicon convenience):

```bash
make setup
source .venv/bin/activate
```

Local development (manual, cross-platform):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Basic Usage (Python API)

```python
from alignment_risk import AlignmentRiskPipeline, PipelineConfig

config = PipelineConfig(mode="lora")  # "full" or "lora"
pipeline = AlignmentRiskPipeline(config)

report = pipeline.run(
    model=model,
    safety_dataloader=safety_loader,
    safety_loss_fn=safety_loss_fn,
    fine_tune_dataloader=ft_loader,
    fine_tune_loss_fn=ft_loss_fn,
)

print(report.warning)
print(report.forecast.collapse_step)
```

## CLI

```bash
alignment-risk --help
alignment-risk demo --output-dir artifacts
alignment-risk demo --mode lora --output-dir artifacts
```

## What It Computes

1. Low-rank safety sensitivity subspace from empirical Fisher geometry.
2. Initial overlap risk (projection of first update into sensitive subspace).
3. Curvature coupling risk (second-order drift signal).
4. Quartic-style stability forecast and collapse-step estimate.

## Modes

- `full`: analyze all selected trainable parameters.
- `lora`: analyze only trainable LoRA adapter parameters (`lora_`, `lora_A`, `lora_B` by default).

## LoRA Mitigation (AlignGuard-style)

After `mode="lora"` risk analysis, attach a regularizer to penalize drift in sensitive directions:

```python
from alignment_risk import AlignmentRiskPipeline, PipelineConfig, AlignGuardConfig

config = PipelineConfig(mode="lora")
pipeline = AlignmentRiskPipeline(config)

report = pipeline.run(
    model=model,
    safety_dataloader=safety_loader,
    safety_loss_fn=safety_loss_fn,
    fine_tune_dataloader=ft_loader,
    fine_tune_loss_fn=ft_loss_fn,
)

mitigator = pipeline.build_lora_mitigator(
    model,
    report.subspace,
    config=AlignGuardConfig(lambda_a=0.25, lambda_t=0.5, lambda_nc=0.1, alpha=0.5),
)

task_loss = ft_loss_fn(model, batch)
breakdown = mitigator.regularized_loss(task_loss)
breakdown.total_loss.backward()
```

Use `mitigator.reset_reference()` to re-anchor regularization at the current adapter state.

## Performance / Accuracy Controls

`FisherConfig` supports speed/accuracy tradeoffs:

- `gradient_collection`: `"loop"` (default), `"auto"`, `"vmap"`.
- `subspace_method`: `"svd"`, `"randomized_svd"`, `"diag_topk"`.
- `vmap_chunk_size`: optional chunking for lower memory.
- `target_explained_variance`: auto-rank selection (default `0.9`).

Example:

```python
from alignment_risk import AlignmentRiskPipeline, PipelineConfig

config = PipelineConfig()
config.fisher.gradient_collection = "vmap"
config.fisher.subspace_method = "randomized_svd"
config.fisher.vmap_chunk_size = 16
```

## Theory and Math References

- **[AIC-2026]** Springer, Max, et al. (2026). *The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety*. arXiv:2602.15799v1. PDF: [https://arxiv.org/pdf/2602.15799](https://arxiv.org/pdf/2602.15799)
- **[ALIGNGUARD-2025]** Das, Amitava, et al. (2025). *AlignGuard-LoRA: Alignment-Preserving Fine-Tuning via Fisher-Guided Decomposition and Riemannian-Geodesic Collision Regularization*. arXiv:2508.02079v1. PDF: [https://arxiv.org/pdf/2508.02079](https://arxiv.org/pdf/2508.02079)

Detailed internal mappings and equations:
- `docs/SOURCES.md`
- `docs/MATH.md`

## Development

```bash
make install
make test
make lint
make typecheck
make build
make check-dist
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution workflow details.
