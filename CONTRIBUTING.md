# Contributing

## Local dev setup

```bash
make setup
source .venv/bin/activate
make test
make lint
```

## Release checklist

1. Update version in `src/alignment_risk/__about__.py`.
2. Ensure tests and lint pass.
3. Build distributions with `make build`.
4. Validate with `make check-dist`.
5. Upload to TestPyPI/PyPI using Twine.

## Notes

- Keep public API exports in `src/alignment_risk/__init__.py` stable.
- Add tests for any new public behavior.
