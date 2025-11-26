# GitHub Workflows & Branch Protection Setup

This project uses GitHub Actions to ensure code quality, security, and proper testing before merging to main.

## Workflows Overview

### 1. **Code Quality** (`code-quality.yml`)
- **Triggers**: On pull requests and pushes to `main` and `develop`
- **What it does**:
  - Checks code formatting with **Black**
  - Checks import sorting with **isort**
  - Lints code with **flake8**
  - Type checks with **mypy**
  - Auto-formats code on PR branches (commits back formatted code)

### 2. **Tests** (`tests.yml`)
- **Triggers**: On pull requests and pushes to `main` and `develop`
- **What it does**:
  - Runs pytest test suite
  - Tests on Python 3.10 and 3.11
  - Generates code coverage reports
  - Uploads coverage to Codecov (optional)

### 3. **Docker Build** (`docker-build.yml`)
- **Triggers**: When Docker-related files change
- **What it does**:
  - Validates docker-compose.yml syntax
  - Builds Docker images
  - Tests that containers can start successfully


## Setting Up Branch Protection Rules

To enforce that all checks must pass before merging, follow these steps on GitHub:

### Step 1: Navigate to Branch Protection Settings
1. Go to your repository on GitHub
2. Click **Settings** → **Branches**
3. Under "Branch protection rules", click **Add rule**

### Step 2: Configure Protection for `main` Branch
Set the following settings:

#### Basic Settings
- **Branch name pattern**: `main`
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: `1` (or more for team projects)
  - ✅ Dismiss stale pull request approvals when new commits are pushed
  - ✅ Require review from Code Owners (optional)

#### Status Checks
- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging
  - **Search and select these required checks**:
    - `format-and-lint` (from Code Quality workflow)
    - `test` (from Tests workflow)
    - `docker-build-test` (from Docker Build workflow)
    - `dependency-scan` (from Security Scan workflow)

#### Additional Settings
- ✅ **Require conversation resolution before merging**
- ✅ **Require signed commits** (recommended for security)
- ✅ **Require linear history** (optional, prevents merge commits)
- ✅ **Include administrators** (apply rules to admins too)
- ✅ **Allow force pushes**: ❌ (disabled)
- ✅ **Allow deletions**: ❌ (disabled)

### Step 3: Save Changes
Click **Create** or **Save changes** at the bottom.

### Step 4: Repeat for `develop` Branch (if applicable)
If you use a `develop` branch, repeat the same process with slightly relaxed rules if needed.

## Local Development Setup

### Install Pre-commit Hooks (Recommended)
Pre-commit hooks run checks before each commit, catching issues early:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# (Optional) Run checks on all files
pre-commit run --all-files
```

### Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### Manual Formatting
```bash
# Format code
black .
isort .

# Check formatting (no changes)
black --check .
isort --check .

# Lint code
flake8 .

# Type check
mypy src/
```

## Workflow Behavior

### On Pull Request Creation
1. All workflows run automatically
2. PR cannot be merged until all required checks pass
3. If formatting is wrong, the workflow will auto-commit fixes (for PR branches only)
4. Review the auto-committed changes and push if needed

### On Push to Main/Develop
1. All workflows run to verify the merged code
2. Weekly security scans continue to monitor for new vulnerabilities

## Testing the Setup

To test that everything works:

1. Create a new branch:
   ```bash
   git checkout -b test/workflow-test
   ```

2. Make a small change (e.g., add a comment to a Python file)

3. Commit and push:
   ```bash
   git add .
   git commit -m "test: verify workflows"
   git push -u origin test/workflow-test
   ```

4. Create a pull request on GitHub

5. Verify that all workflow checks appear and run

## Troubleshooting

### Workflow Not Running?
- Check that the workflow files are in `.github/workflows/`
- Ensure YAML syntax is valid
- Check the "Actions" tab on GitHub for errors

### Check Failing?
- Click on the failed check in the PR to see detailed logs
- Run the same commands locally to debug
- Use pre-commit hooks to catch issues before pushing

### Auto-formatting Not Working?
- Ensure the branch is not `main` (auto-format only works on feature branches)
- Check that the GitHub Actions bot has write permissions

<!-- ## Project-Specific Recommendations

For this **Animal Spotter** project on Radxa Rock 5B, consider adding:

1. **Model Conversion Validation**: Add a workflow to validate ONNX → .rknn conversion
2. **Dataset Validation**: Check that downloaded datasets meet expected format
3. **ARM64 Build Tests**: Test builds on ARM architecture (if using self-hosted runners)
4. **Model Performance Tests**: Benchmark inference speed on sample images
5. **Documentation Builds**: Auto-generate docs from docstrings

Would you like me to add any of these specialized workflows? -->
