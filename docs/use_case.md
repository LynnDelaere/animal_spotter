# Animal Spotter – Use Cases & Value

Animal Spotter fine-tunes a DETR (Detection Transformer) model on a curated
subset of Open Images V7 animal classes. The resulting detector can be deployed
as part of wildlife monitoring, conservation, or camera trap analytics systems.
This document summarises the primary use cases, user stories, and benefits that
the current codebase supports.

---

## Target scenarios

| Scenario | Description | Supported features |
| --- | --- | --- |
| Wildlife research | Process camera-trap footage/images to detect animals across multiple classes. | COCO conversion pipeline, DETR training script, evaluation notebook. |
| Park/conservation monitoring | Run periodic training jobs when new labelled data arrives and track results over time. | Training outputs saved under `models/`, logging to TensorBoard, MinIO storage sync to share datasets. |
| ML experimentation | Iterate on DETR fine-tuning configurations, swap datasets, and test new augmentation pipelines. | Modular dataset config (`configs/dataset_v1.yaml`), dataset class in `src/training/dataset_class.py`, dockerized dev environment. |
| CI/CD validation | Guarantee that code changes do not break the training pipeline or infrastructure. | GitHub Actions workflows, self-hosted runner container, pre-commit hooks. |

---

## Primary user stories

1. **Data engineer** downloads raw Open Images data, filters it to specific
   animal classes and exports COCO annotations using the scripts under
   `src/data/`.
2. **ML engineer** fine-tunes DETR on the processed dataset, inspects TensorBoard
   logs under `logs/`, and saves checkpoints to `models/detr-finetuned/`.
3. **Reviewer/researcher** loads `src/notebooks/evalutation_detr.ipynb`,
   visualizes detections and validates that the model meets accuracy thresholds.
4. **Ops/DevOps** runs the Docker stack (`docker/docker-compose.yml`) to provide
   local S3 storage (MinIO), the self-hosted GitHub runner, and a reproducible
   Python environment for experimentation.
5. **CI maintainer** relies on `.github/workflows/` to enforce linting, testing,
   and Docker build checks before merging changes into `main`.

---

## Benefits

- **End-to-end reproducibility** – configuration files, scripts, and Docker
  services allow anyone to recreate the dataset and training results.
- **Infrastructure parity** – the same Docker compose stack powers local
  development as well as the self-hosted runner used in CI.
- **Extendibility** – thanks to the dataset configuration and modular code, new
  species or data sources can be added without rewriting the pipeline.
- **Quality gates** – pre-commit hooks and GitHub Actions reduce regressions in
  data processing, model training and infrastructure code.

---

## Future opportunities

- Serve the fine-tuned model through an API (e.g., FastAPI) for real-time
  inference.
- Automate scheduled retraining jobs that pull fresh MinIO data.
- Add monitoring dashboards for model drift using the MinIO-backed datasets.

These enhancements fit naturally with the current repository structure and can
be tracked via issues/roadmap items.
