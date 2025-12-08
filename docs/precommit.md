# Pre-commit Hooks & Local Checks

Animal Spotter uses `pre-commit` hooks to mirror the checks enforced by the
Code Quality workflow (Black, Ruff, mypy, docstring tooling). Keeping hooks up to
date locally prevents surprises once GitHub Actions runs.

## Quick start
```bash
pip install -r requirements.txt  # installs the same tool versions as CI
pip install pre-commit
pre-commit install
```

Hooks run at the `pre-commit` stage. If a hook edits files (Black/Ruff), Git
blocks the commit so you can `git add` the changes and retry.

## Hooks currently configured
From `.pre-commit-config.yaml`:

| Hook | Purpose | Notes |
| --- | --- | --- |
| `black` | Format Python files. | Uses Python 3.13 runtime to match the tool’s latest version. |
| `ruff` (`--fix`) | Lint + auto-fix violations. | Aligns with CI’s `ruff check .`. |
| `ruff-format` | Additional formatter from Ruff. | Ensures consistent style in files Black does not touch. |
| `pydocstyle --convention=google src` | Enforce docstring style. |
| `interrogate -c 50 -v src` | Enforce ≥50% docstring coverage. |
| `mypy --ignore-missing-imports` | Static type checking (tests/ excluded). | Installs `types-requests` + `types-PyYAML`. |

Run all hooks manually with:
```bash
pre-commit run --all-files
```

## Equivalent ad-hoc commands
Useful when iterating rapidly or debugging hook failures:
```bash
black .
ruff check . --fix
ruff format .
pydocstyle --convention=google src
interrogate -c 50 -v src
mypy --ignore-missing-imports .
```

## Updating hooks
- Bump versions: `pre-commit autoupdate`.
- CI (`.pre-commit-config.yaml` → `ci` block) is configured to open quarterly
  auto-update PRs and to allow autofix PRs when hooks modify files.

## Troubleshooting
- **Formatting changes** → re-add files after hooks modify them.
- **Docstring coverage fails** → add concise docstrings to public modules,
  classes and functions in `src/`.
- **mypy errors** → add type hints or adjust stubs; avoid `# type: ignore`
  unless justified.

Running `pre-commit run --all-files` before pushing should give you the same
results as the Code Quality GitHub Action (`quality` job).

## Troubleshooting
- Hooks reformat files: re-add changes and re-commit.
- Failing docstring coverage: add concise docstrings to public modules/functions/classes in `src/`.
- Types failing: add type hints or adjust interfaces; avoid `ignore` unless necessary.

## CI Parity
Running `pre-commit run --all-files` locally mirrors what GitHub Actions checks, minimizing surprises in PRs.
