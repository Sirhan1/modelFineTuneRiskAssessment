# Contributing

Thanks for contributing to `alignment-risk`.

This guide is optimized for fast onboarding: set up the project, make a change, run checks, and open a PR.

## Quick Start

### 1. Clone and enter the repo

```bash
git clone https://github.com/sirhan1/modelFineTuneRiskAssessment.git
cd modelFineTuneRiskAssessment
```

### 2. Set up your environment

Apple Silicon (convenience script):

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

### 3. Verify setup

```bash
make test
```

## Typical Contribution Workflow

### 1. Create a branch

```bash
git checkout -b my-change
```

### 2. Implement your change

- Keep changes focused and small when possible.
- Add or update tests for behavior changes.
- Update docs/examples when user-facing behavior changes.

### 3. Run local checks before opening a PR

```bash
make test
make lint
make typecheck
```

Optional docs/build checks:

```bash
make docs-build
make check-dist
```

### 4. Open a PR

In your PR description, include:

1. What changed.
2. Why it changed.
3. How you validated it (commands and outcomes).
4. Any follow-up work or known limitations.

## What We Expect in PRs

- Tests for new public behavior or bug fixes.
- Backward-compatible public API changes (or clear migration notes).
- Consistent docs in `README.md` for usage-level changes.
- Consistent docs in `docs/MATH.md` and `docs/SOURCES.md` when equations/logic/source mapping changes.

## Project Commands (Reference)

```bash
make install      # install package + dev deps into .venv
make test         # run pytest
make lint         # run ruff
make typecheck    # run mypy
make demo         # run synthetic demo
make docs-build   # strict mkdocs build
make check-dist   # build + twine checks
make clean        # remove venv/cache/build artifacts
```

## Release Maintainers Checklist

1. Update version in both `pyproject.toml` and `src/alignment_risk/__about__.py`.
2. Run checks: `make test`, `make lint`, `make typecheck`.
3. Build and validate artifacts: `make check-dist`.
4. Publish with Twine (see `docs/PUBLISHING.md`).
