#!/bin/bash
set -e

cd /home/runner

# First step: configure the runner if it hasn't been configured yet
if [ ! -f .runner ]; then
  : "${RUNNER_URL:?Environment variable RUNNER_URL is required}"
  : "${RUNNER_TOKEN:?Environment variable RUNNER_TOKEN is required}"

  echo "Configuring runner for ${RUNNER_URL}..."

  ./config.sh \
    --unattended \
    --url "${RUNNER_URL}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME:-$(hostname)}" \
    --labels "${RUNNER_LABELS:-self-hosted,animal-spotter}" \
    --work "_work" \
    --replace

  echo "Runner configured."
else
  echo ".runner file already exists, skipping configuration."
fi

echo "Starting runner..."
./run.sh
