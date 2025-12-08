# Docker services

The `docker/` directory contains all containers needed locally or in CI for
Animal Spotter:

- **MinIO**: S3-compatible storage that serves as a local replacement for the
  cloud bucket. Training and evaluation scripts can use the same S3 API without
  relying on an external provider.
- **Self-hosted GitHub Actions runner**: builds this repository in CI and runs
  tasks that need GPU or network access. The runner runs in its own container
  and binds its working directory (`runner-data`) on the host so configuration
  persists across restarts.
- **Python app container**: contains all Python dependencies from
  `requirements.txt` and mounts the entire repository at `/workspace`. Use this
  container for local development, notebooks, and training jobs so the
  environment stays identical to CI. 

## Usage

1. Create a `.env` file in the `docker/` directory:

   ```bash
   cd docker
   cp .env.example .env
   ```

   Fill in the MinIO credentials (`MINIO_ROOT_*`) and all `RUNNER_*` variables.
   Request a new `RUNNER_TOKEN` via GitHub > Settings > Actions > Runners >
   "New self-hosted runner".

2. Start the services:

   ```bash
   docker compose up -d
   ```

   Add `--build` the first time so the Python and runner images get built.

3. Verify MinIO is reachable at `http://localhost:9001`, the runner is active on
   GitHub, and the Python container is running (`docker compose ps`). You can
   find runner logs and the `_work` path in `docker/github-runner/runner-data`
   (ignored in Git). Use `docker compose exec animal-spotter-app bash` to open a
   shell inside the Python container and run scripts.

Stop services with `docker compose down`. Runner configuration persists in
`runner-data`, so you do not need a new token after a restart.
