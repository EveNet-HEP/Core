# Releasing `event-core`

This repository publishes only `event-core`.

## Auto publish flow

- Tag push matching `v*` triggers `.github/workflows/build-core.yml`.
- `build-core` job builds and validates artifacts.
- `publish-core` runs only if `build-core` succeeds and uploads to PyPI.

No separate manual publish workflow is required.

## What you must set on PyPI

Set up **Trusted Publishing** for package `event-core`:

1. Package name: `event-core` (create it if it does not exist yet).
2. Publisher type: GitHub Actions.
3. GitHub owner: your org/user.
4. GitHub repository: your `event-core` repository.
5. Workflow filename: `build-core.yml`.
6. Environment name: `pypi`.

Notes:
- First release may require creating a **pending publisher** in PyPI before the workflow can publish.
- You do not need a `PYPI_API_TOKEN` secret when using trusted publishing.

## One-time GitHub requirement

Create GitHub Actions environment `pypi` in this repository (Settings -> Environments).

## Release steps

1. Bump version in `setup.py`.
2. Commit and push.
3. Tag and push:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow will build, then auto-publish on success.
