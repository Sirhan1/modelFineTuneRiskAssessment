# Publishing Guide

## One-time setup

1. Create accounts on TestPyPI and PyPI.
2. Generate API tokens.
3. Configure `~/.pypirc` or use environment variables for Twine.

## Release steps

1. Update `version` in `pyproject.toml` and `__version__` in `src/alignment_risk/__about__.py`.
2. Run quality checks:
   - `make test`
   - `make lint`
   - `make typecheck`
3. Build and verify distributions:
   - `make check-dist` (this target runs `make build` first)
4. Upload to TestPyPI:
   - `python -m twine upload --repository testpypi dist/*`
5. Validate install from TestPyPI.
6. Upload to PyPI:
   - `python -m twine upload dist/*`
