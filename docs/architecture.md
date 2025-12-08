# Architecture Overview

Animal Spotter is a containerized ML workflow for training and evaluating a DETR
object detector on a filtered subset of the Open Images V7 animal classes. The
repository is structured around a data pipeline (ingest → convert → sync), a
training stack, and optional infrastructure for storage and CI.

## High-level flow
1. **Ingest** – pull Open Images metadata and images for selected classes via
	 `src/data/download_dataset_raw.py` using `configs/dataset_v1.yaml`.
2. **Convert** – produce COCO-style annotations and class mapping with
	 `src/data/convert_to_coco.py`, writing to `data/processed/v1/`.
3. **Sync (optional)** – push/pull raw and processed data to/from MinIO using
	 `upload_dataset_to_minio.py` and `download_dataset_from_minio.py`.
4. **Train** – fine-tune `facebook/detr-resnet-50` via
	 `src/training/train_detr.py`, saving checkpoints to `models/detr-finetuned/`
	 and TensorBoard logs to `logs/`.
5. **Evaluate** – explore checkpoints in
	 `src/notebooks/evalutation_detr.ipynb` or downstream scripts.

## Core components
- **Data layer**: local `data/` directory for images/metadata/COCO JSON; MinIO
	(S3-compatible) as optional remote storage. Configured through `.env` and
	`configs/dataset_v1.yaml`.
- **Data processing utilities** (`src/data/`):
	- `download_dataset_raw.py` – fetch Open Images subset.
	- `convert_to_coco.py` – build COCO JSON and class list.
	- `upload_dataset_to_minio.py` / `download_dataset_from_minio.py` – bidirectional sync with MinIO.
	- `minio_datamanagement.py` – client and bucket helpers.
- **Training stack** (`src/training/`):
	- `dataset_class.py` – COCO-based dataset wrapper with validation.
	- `train_detr.py` – Hugging Face Transformers training loop with early
		stopping, mixed precision on CUDA, and TensorBoard logging.
- **Model artifacts**: stored under `models/detr-finetuned/` (ignored in Git)
	with processor files saved alongside the checkpoint.
- **Observability**: training logs in `logs/` for TensorBoard; script/stdout
	logging otherwise.

## Infrastructure (Docker)
Defined in `docker/` and documented in `docs/docker_containers.md`:
- **MinIO**: local object storage for datasets and artifacts.
- **Self-hosted GitHub Actions runner**: mirrors CI environment; mounts
	`runner-data/` for persistence.
- **Python app container**: installs `requirements.txt`, mounts the repo to
	`/workspace`, and serves as a reproducible dev/train environment.
Use `docker compose up -d --build` to start the stack; connect with
`docker compose exec animal-spotter-app bash`.

## CI/CD
GitHub Actions workflows enforce formatting, linting, tests, and Docker build
checks (see `docs/github-workflows.md`). The self-hosted runner container lets
you offload jobs requiring local data or specialized hardware.

## Configuration
- Dataset and paths: `configs/dataset_v1.yaml` (class list, limits, raw and
	processed dirs, MinIO prefixes).
- Secrets and endpoints: `.env` (e.g., `MINIO_ENDPOINT`, `MINIO_ROOT_USER`,
	`MINIO_ROOT_PASSWORD`).
- Python tooling: `pyproject.toml` (Black, isort, Ruff, mypy, pytest).

## Environments
- **Local**: Python venv + scripts/notebooks against the `data/` directory.
- **Dockerized**: same code mounted into the Python app container for parity
	with CI and for isolated deps.
- **CI**: GitHub Actions on hosted or self-hosted runners using the same
	workflows as local pre-commit hooks.