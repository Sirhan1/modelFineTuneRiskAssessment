# Contributing

## Local setup

Apple Silicon quick setup:

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

## Pre-PR checklist

```bash
make test
make lint
make typecheck
```

## Release checklist

1. Update version in both `pyproject.toml` and `src/alignment_risk/__about__.py`.
2. Run quality checks (`make test`, `make lint`, `make typecheck`).
3. Build and validate distributions (`make check-dist`).
4. Publish with Twine (see `docs/PUBLISHING.md` for full steps).

## Notes

- Keep public API exports in `src/alignment_risk/__init__.py` stable.
- Add tests for any new public behavior.
- Keep README/docs examples in sync with CLI and config defaults.
- Keep `docs/MATH.md` and `docs/SOURCES.md` updated when math or decision logic changes.
