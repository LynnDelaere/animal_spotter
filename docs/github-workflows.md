# GitHub Workflows & Branch Protection

Animal Spotter relies on three GitHub Actions workflows. All of them currently
trigger on pull requests targeting `main` and run on the self-hosted runner that
lives in the Docker stack. This page mirrors what is in `.github/workflows/`
right now so branch protection can be configured accurately.

---

## Workflow overview

### 1. Code Quality (`.github/workflows/code-quality.yml`)
- **Trigger**: `pull_request` → `main`
- **Runner**: `self-hosted`
- **Steps**:
  - Set up a fresh virtual environment and install `requirements.txt`.
  - Run Black (format check) and Ruff linting.
  - Run mypy for type checking.
  - Run documentation-oriented tools (pydocstyle, interrogate).
  - Run Bandit and pip-audit (both marked `|| true`, so they surface warnings
    but do not fail the job right now).
- **Job / check name**: `quality`

### 2. Tests (`.github/workflows/tests.yml`)
- **Trigger**: `pull_request` → `main`
- **Runner**: `self-hosted`
- **Steps**:
  - Same venv bootstrap + dependency installation as the quality job.
  - Execute `pytest tests --cov=src --cov-report=xml`.
- **Job / check name**: `test`
- **Python versions**: single run on the version installed on the runner
  (currently Python 3.10). There is no matrix for 3.11+.

### 3. Docker Build (`.github/workflows/docker-build.yml`)
- **Trigger**: `pull_request` → `main`
- **Runner**: `ubuntu-latest` (GitHub-hosted)
- **Steps**:
  - Validate `docker/docker-compose.yml` via `docker compose ... config`.
  - Build all images with `docker compose build --no-cache`.
  - Smoke-test the compose stack (up, wait 10s, `docker compose ps`, down).
- **Job / check name**: `docker-build-test`

---

## Branch protection recommendations

To require these checks before merging into `main`:

1. Open **Settings → Branches** in GitHub.
2. Add or edit a rule for `main` with:
   - **Require a pull request before merging** (at least one approval).
   - **Require status checks to pass before merging** with the following checks:
     - `quality`
     - `test`
     - `docker-build-test`
   - Optionally enable “Require branches to be up to date” to enforce rebase/merge
     before completion.

Because the quality and test jobs run on the self-hosted runner, make sure the
runner container is online before merging PRs—otherwise checks will stay queued.

---

## Local parity

Developers can mirror the CI steps locally to shorten iteration time:

```bash
# Format + lint + types
black --check .
ruff check .
mypy .

# Tests + coverage
pytest tests --cov=src --cov-report=term-missing

# Docker validation
cd docker
docker compose -f docker-compose.yml config
docker compose build
docker compose up -d
docker compose down
```

Running these commands before opening a PR should yield the same results as the
workflows described above.

---

## Future improvements

Potential enhancements (not yet implemented):
- Add `push` triggers on `main` to verify direct commits.
- Expand the test workflow to a Python version matrix.
- Make security tools (Bandit, pip-audit) blocking instead of advisory.
- Publish artifacts (coverage, reports) for easier inspection.

Track these ideas in issues so the workflow files and this document stay in sync.
