# Pre-commit Hooks & Local Checks

This project uses `pre-commit` to enforce formatting, linting, docstrings, type checks, and coverage thresholds before code is committed. Running these locally helps your commits pass GitHub Actions smoothly.

## Quick Setup
- Install tooling in your virtualenv:
  - `pip install pre-commit`
  - `pip install -r requirements.txt` (includes formatters/linters used in CI)
- Register hooks: `pre-commit install`

## What Runs on Commit
Configured in `.pre-commit-config.yaml`:
- `black`: formats Python code.
- `ruff` with `--fix`: lints and auto-fixes issues.
- `ruff-format`: ensures consistent formatting.
- `pydocstyle --convention=google src`: checks docstrings in `src/`.
- `interrogate -c 50 -v src`: verifies docstring coverage ≥ 50%.
- `mypy --ignore-missing-imports`: static type checks.

Hooks run at the `commit` stage. If a hook modifies files (e.g., `black`, `ruff`), your commit is blocked so you can `git add` the changes and re-commit.

## Run Everything Before Committing
To run all hooks across the repo:
- `pre-commit run --all-files`

## Individual Commands (useful for quick checks)
- Format:
  - `black .`
  - `ruff format .` (or `pre-commit run ruff-format --all-files`)
- Lint:
  - `ruff .` (add `--fix` to auto-fix)
- Docstrings:
  - `pydocstyle --convention=google src`
  - `interrogate -c 50 -v src`
- Types:
  - `mypy --ignore-missing-imports .`

## Keeping Hooks Up-to-date
- Update hook versions: `pre-commit autoupdate`
- Optional PR automation is enabled in CI to auto-fix/auto-update on a schedule.

## Troubleshooting
- Hooks reformat files: re-add changes and re-commit.
- Failing docstring coverage: add concise docstrings to public modules/functions/classes in `src/`.
- Types failing: add type hints or adjust interfaces; avoid `ignore` unless necessary.

## CI Parity
Running `pre-commit run --all-files` locally mirrors what GitHub Actions checks, minimizing surprises in PRs.
