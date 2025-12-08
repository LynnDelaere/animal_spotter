# Animal Spotter

Animal Spotter fine-tunes Facebook's DETR model on a curated subset of Open
Images V7 animal classes. The repository covers the full workflow: downloading
data, converting it to COCO, syncing with MinIO, training/evaluating DETR, and
running the stack locally through Docker (MinIO, a self-hosted GitHub runner and
an all-in-one Python dev container).

---

## Key capabilities
- **Data ingestion** – scripts to pull filtered Open Images subsets, convert
  them to COCO JSON and sync files via MinIO.
- **Model training** – fine-tune `facebook/detr-resnet-50` with Hugging Face
  Transformers, including early stopping, TensorBoard logging and mixed
  precision on CUDA.
- **Evaluation / exploration** – a ready-to-run notebook for quick inference on
  local checkpoints.
- **Dockerized infrastructure** – a compose stack with MinIO, the self-hosted
  Actions runner used in CI, and a reproducible Python workspace container.
- **CI/CD** – GitHub Actions workflows for linting, tests and Docker builds.

---

## Repository layout
| Path | Description |
| --- | --- |
| `configs/` | Dataset definition (`dataset_v1.yaml`) controlling class filters, limits and paths. |
| `data/` | Downloaded images + metadata and COCO conversions (ignored in Git). |
| `docs/` | Architecture, workflow, Docker and pre-commit notes. |
| `docker/` | Compose stack (`docker-compose.yml`), service-specific Dockerfiles and `.env.example`. |
| `models/` | Local training outputs (ignored). |
| `src/data/` | Data utilities (download/convert/upload/sync). |
| `src/training/` | DETR dataset class and training script. |
| `src/notebooks/` | Notebooks such as `evalutation_detr.ipynb` for inference demos. |
| `tests/` | Pytest suite covering the dataset class and helpers. |

---

## Requirements
- Python 3.10+
- `pip` (or an alternative such as `uv`)
- Optional: Docker + Docker Compose v2 (for MinIO, the runner and dev container)
- Optional: NVIDIA drivers + Container Toolkit (only needed if you re-enable GPU
  passthrough for the dev container)

---

## Local Python setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` (if present) or create `.env` in the repo root and add
   the MinIO credentials used by the scripts:
   ```
   MINIO_ENDPOINT=localhost:9000
   MINIO_ROOT_USER=...
   MINIO_ROOT_PASSWORD=...
   ```
3. Optional: install the pre-commit hooks defined in `docs/precommit.md`:
   ```bash
   pre-commit install
   ```

---

## Dataset workflow (Open Images ➜ COCO ➜ MinIO)
The `configs/dataset_v1.yaml` file specifies class names, download limits and
local/remote paths. Most scripts take `--config configs/dataset_v1.yaml`.

1. **Download Open Images subset**
   ```bash
   python src/data/download_dataset_raw.py --config configs/dataset_v1.yaml
   ```
   - Images end up in `data/images/{train,validation,test}`.
   - Metadata lives under `data/metadata/`.

2. **Convert to COCO JSON + class map**
   ```bash
   python src/data/convert_to_coco.py --config configs/dataset_v1.yaml
   ```
   - Produces `data/processed/v1/{train,validation,test}.json` and
     `classes.yaml`.

3. **Sync with MinIO (optional)**
   ```bash
   python src/data/upload_dataset_to_minio.py --config configs/dataset_v1.yaml --type both
   ```
   - Reads credentials from `.env`, creates the bucket if missing and uploads
     both raw images and processed COCO outputs.

4. **Download from MinIO instead of the public source (optional)**
   ```bash
   python src/data/download_dataset_from_minio.py --config configs/dataset_v1.yaml
   ```
   - Mirrors the configured prefixes back into `data/`.

---

## Training DETR
```bash
python src/training/train_detr.py
```
- Uses `facebook/detr-resnet-50` as the base checkpoint with label mappings
  derived from `classes.yaml`.
- Saves checkpoints under `models/detr-finetuned/`, processor files alongside
  the model, and TensorBoard logs to `logs/`.
- Selects CUDA automatically when available (and falls back to CPU otherwise).
- Adjust hyperparameters in `TrainingArguments` inside the script or clone it if
  you need multiple experiment configurations.

---

## Evaluation / inference notebook
- Open `src/notebooks/evalutation_detr.ipynb`.
- The notebook adds the repo root to `sys.path`, loads a local checkpoint from
  `models/detr-finetuned/` and runs inference on sample images.
- When running outside Docker, ensure your `.venv` has the same dependencies as
  `requirements.txt` to avoid version mismatch warnings.

---

## Docker stack
Documented in detail in `docs/docker_containers.md`. Quick start:
```bash
cd docker
cp .env.example .env       # fill in MinIO + RUNNER_* values
docker compose up -d --build
```
Services:
- **MinIO** (`minio_service`): S3-compatible storage used by the scripts.
- **Self-hosted runner** (`animal-spotter-runner`): GitHub Actions runner with a
  persistent `runner-data/` volume.
- **Python app container** (`animal-spotter-app`): lightweight `python:3.10-slim`
  image that installs `requirements.txt`, mounts the repo to `/workspace`, and
  exposes a shell for notebooks/training:
  ```bash
  docker compose exec animal-spotter-app bash
  ```

The compose file already defines health checks for all services. Stop the stack
with `docker compose down`. Runner data remains on the host so you do not need a
new token after restarts.

---

## Testing & linting
Run the entire test suite:
```bash
pytest -q
```

Helpful extras:
```bash
ruff check src tests
ruff format src tests
mypy src
```
Pre-commit hooks in `.pre-commit-config.yaml` mirror the CI checks (see
`docs/precommit.md` for details).

---

## CI/CD
GitHub Actions workflows live under `.github/workflows/`. Highlights:
- **`python-ci.yml`** (lint + tests) – triggers on pull requests and pushes to
  `main`.
- **`docker-build.yml`** – validates `docker/docker-compose.yml`, builds the
  images and ensures the stack can start/stop cleanly.
- **`pre-commit.yml`** – enforces formatting and static analysis.

`docs/github-workflows.md` documents the triggers and environment settings for
each workflow. The repository also includes a reusable Dockerized runner, so CI
jobs relying on specialized hardware can be scheduled onto your own runners.

---

## Further reading
- `docs/architecture.md` – high-level system overview.
- `docs/docker_containers.md` – deeper dive into the Docker setup.
- `docs/github-workflows.md` – workflow reference.
- `docs/use_case.md` – the motivating use case and product assumptions.

Contributions are welcome. Please open an issue or pull request if you have
ideas to improve the training pipeline, dataset coverage, or infrastructure.
