# Publishing Guide (CI/CD First)

This guide is for automated releases with GitHub Actions.
Recommended approach: PyPI Trusted Publishing (OIDC), not long-lived API tokens.

## Release Model (Recommended)

1. Pull requests run CI (tests, lint, typecheck, build check).
2. Merging to `main` keeps the branch releasable.
3. Creating a version tag (`vX.Y.Z`) triggers publish workflow.
4. Workflow builds once, publishes to TestPyPI, then publishes to PyPI.

## One-Time Setup

### 1. Create package projects

1. Create accounts on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org).
2. Ensure project name exists: `alignment-risk`.

### 2. Configure Trusted Publishing on PyPI and TestPyPI

In each index (PyPI and TestPyPI), add a Trusted Publisher pointing to this repo:

1. Owner: `sirhan1`
2. Repository: `modelFineTuneRiskAssessment`
3. Workflow filename: your publishing workflow file (for example `publish.yml`)
4. Environment name:
   - `testpypi` for TestPyPI publisher
   - `pypi` for PyPI publisher

Use exact workflow filename and environment names to match your GitHub workflow.

### 3. Configure GitHub environments

In GitHub repo settings, create environments:

1. `testpypi`
2. `pypi`

Recommended protections:

1. Require reviewers for `pypi`.
2. Restrict allowed branches/tags (for example, only release tags).

## Versioning

Before release, bump version in both files:

1. `pyproject.toml` (`[project].version`)
2. `src/alignment_risk/__about__.py` (`__version__`)

Use semantic versioning (`MAJOR.MINOR.PATCH`).

## Pre-Release Checks

Run before tagging:

```bash
make test
make lint
make typecheck
make check-dist
```

## Suggested CI Workflow (PR + main)

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install --upgrade pip
      - run: pip install -e ".[dev]"
      - run: make test
      - run: make lint
      - run: make typecheck
      - run: make check-dist
```

## Suggested Publish Workflow (Trusted Publishing)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish

on:
  push:
    tags:
      - "v*"
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install --upgrade pip
      - run: pip install -e ".[dev]"
      - run: make check-dist
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/*

  publish-testpypi:
    needs: build
    runs-on: ubuntu-latest
    environment: testpypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  publish-pypi:
    needs: publish-testpypi
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Notes:

1. Build artifacts once and reuse them for both indexes.
2. Keep `environment` values exactly aligned with Trusted Publisher config.
3. For Trusted Publishing, avoid calling publish from a reusable workflow.

## Release Procedure

1. Merge release PR into `main`.
2. Create and push tag:

```bash
git tag v0.1.1
git push origin v0.1.1
```

3. Watch `publish.yml` in GitHub Actions.
4. Verify package on TestPyPI and PyPI.
5. Create GitHub release notes from the tag.

## Rollback and Recovery

If publish fails:

1. Fix workflow/config issue.
2. Delete broken tag locally and remotely if needed.
3. Bump to next patch version (do not overwrite released files on PyPI).
4. Re-tag and republish.

## Emergency Manual Publishing (Fallback)

Use this only if CI is unavailable:

```bash
make check-dist
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

Token-based uploads are fallback-only when Trusted Publishing cannot be used.
