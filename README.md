# alignment-risk

`alignment-risk` is a Python package for pre-flight alignment risk diagnostics during fine-tuning.
It estimates whether a fine-tuning trajectory is likely to drift into safety-sensitive weight directions.

## Installation

Install from PyPI:

```bash
pip install alignment-risk
```

For local development:

Apple Silicon convenience setup:

```bash
make setup
source .venv/bin/activate
```

Manual setup (cross-platform):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

Run the included synthetic demo:

```bash
alignment-risk --help
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

## Fisher Performance Options

`FisherConfig` now supports user-selectable speed/accuracy tradeoffs:

- `gradient_collection`: `"loop"` (default), `"auto"`, or `"vmap"`.
- `subspace_method`: `"svd"` (exact), `"randomized_svd"` (faster approximate), or `"diag_topk"` (fastest rough approximation).
- `vmap_chunk_size`: optional chunking for lower memory with `vmap`.
- `target_explained_variance`: auto-select Fisher rank by retained energy (default `0.9`) within `top_rank`.

When using `gradient_collection="auto"`, the analyzer runs a small loop-vs-vmap
parity probe and falls back to loop if the gradients disagree.

Example:

```python
from alignment_risk import AlignmentRiskPipeline, PipelineConfig

config = PipelineConfig()
config.fisher.gradient_collection = "vmap"
config.fisher.subspace_method = "randomized_svd"
config.fisher.vmap_chunk_size = 16
```

## Reliability Defaults

The pipeline now enables several mitigations by default without requiring extra user inputs:

- Adaptive curvature refinement on borderline forecasts (re-runs curvature on more batches).
- Consistent eval-mode gradient probes for Fisher, initial update estimate, and curvature (dropout/BN consistency).
- More robust forecast curvature floor via a stable low-quantile Fisher eigenvalue.
- LoRA mode honors explicit `fisher.parameter_names` allowlists (does not require name-marker matches).

## Theory sources

This implementation is grounded in the following primary references:

- **[AIC-2026]** Springer, Max, et al. (2026). *The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety*. arXiv:2602.15799v1. PDF: [https://arxiv.org/pdf/2602.15799](https://arxiv.org/pdf/2602.15799)
- **[ALIGNGUARD-2025]** Das, Amitava, et al. (2025). *AlignGuard-LoRA: Alignment-Preserving Fine-Tuning via Fisher-Guided Decomposition and Riemannian-Geodesic Collision Regularization*. arXiv:2508.02079v1. PDF: [https://arxiv.org/pdf/2508.02079](https://arxiv.org/pdf/2508.02079)

See `docs/SOURCES.md` for module-by-module mapping from repository code to sections/pages in these papers.

## Repository layout

- `src/alignment_risk/`: package source code.
- `tests/`: test suite.
- `examples/`: runnable examples.
- `scripts/`: setup and project automation scripts.
- `pyproject.toml`: packaging metadata and build config.

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

## Documentation map

- `CONTRIBUTING.md`: local dev and contribution checklist.
- `docs/PUBLISHING.md`: release and publishing guide.
- `docs/CITATIONS.md`: canonical references and starter BibTeX entries.
- `docs/SOURCES.md`: source-to-paper mapping and derivation notes.

## Publishing workflow (when ready)

Follow `docs/PUBLISHING.md` end-to-end.
