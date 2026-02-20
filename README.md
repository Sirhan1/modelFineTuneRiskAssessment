# alignment-risk

`alignment-risk` is a Python package for pre-flight alignment risk diagnostics during fine-tuning.
It estimates whether a fine-tuning trajectory is likely to drift into safety-sensitive weight directions.

## Installation

```bash
pip install alignment-risk
```

For local development on Apple Silicon:

```bash
make setup
source .venv/bin/activate
```

## Quickstart

Run the included synthetic demo:

```bash
alignment-risk demo --output-dir artifacts
alignment-risk demo --mode lora --output-dir artifacts
```

Generated outputs:
- `artifacts/sensitivity_map.png`
- `artifacts/safety_decay_forecast.png`

## Python API

```python
from alignment_risk import AlignmentRiskPipeline, PipelineConfig

config = PipelineConfig(mode="lora")  # or mode="full"
pipeline = AlignmentRiskPipeline(config)
report = pipeline.run(
    model=model,
    safety_dataloader=safety_loader,
    safety_loss_fn=safety_loss_fn,
    fine_tune_dataloader=ft_loader,
    fine_tune_loss_fn=ft_loss_fn,
)

print(report.warning)
```

## Modes

- `full`: analyze all selected trainable parameters (standard full fine-tuning).
- `lora`: analyze only trainable LoRA adapter parameters (names containing `lora_`, `lora_A`, `lora_B`).

## LoRA mitigation (AlignGuard-style)

After running risk analysis in `mode="lora"`, you can attach a regularizer that adds:
- Fisher-weighted safety penalty on alignment-sensitive update component,
- task-stability penalty on the orthogonal component,
- collision penalties (Riemannian + geodesic) to reduce interference.

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

Call `mitigator.reset_reference()` when you want to re-anchor regularization to the current adapter state.

## What the package computes

1. Low-rank safety sensitivity from Fisher geometry.
2. Initial overlap risk (first-order projection into safety subspace).
3. Curvature coupling risk (second-order directional drift).
4. Quartic-style stability forecast and collapse-step warning.

## Repository layout

- `src/alignment_risk/`: package source code.
- `tests/`: test suite.
- `examples/`: runnable examples.
- `scripts/`: setup and project automation scripts.
- `pyproject.toml`: packaging metadata and build config.

## Development commands

```bash
make test
make lint
make typecheck
make build
make check-dist
```

## Publishing workflow (when ready)

1. Bump version in `src/alignment_risk/__about__.py`.
2. Run quality checks: `make test lint typecheck`.
3. Build artifacts: `make build`.
4. Validate artifacts: `make check-dist`.
5. Upload with Twine to TestPyPI/PyPI.
