# Releasing `event-core`

This repository publishes only `event-core`.

## 1) Build locally

```bash
python -m pip install --upgrade build
python -m build
```

## 2) Tag to trigger automatic build

The core build workflow runs on tags matching `v*` in this repository.

Example:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## 3) Publish to PyPI

1. Create the PyPI project `event-core`.
2. In PyPI, configure trusted publishing for this repository + workflow `.github/workflows/publish-pypi.yml`.
3. In GitHub Actions, run `Publish event-core to PyPI`.
