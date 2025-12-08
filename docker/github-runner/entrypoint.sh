#!/bin/bash
set -euo pipefail

export RUNNER_ALLOW_RUNASROOT=1

RUNNER_HOME="${RUNNER_HOME:-/home/runner}"
RUNNER_DIST_DIR="${RUNNER_DIST_DIR:-/opt/actions-runner}"
RUNNER_USER="${RUNNER_USER:-runner}"

mkdir -p "${RUNNER_HOME}"

if [ ! -f "${RUNNER_HOME}/config.sh" ]; then
  echo "Initialising runner home from ${RUNNER_DIST_DIR}..."
  cp -R "${RUNNER_DIST_DIR}/." "${RUNNER_HOME}/"
fi

chown -R "${RUNNER_USER}:${RUNNER_USER}" "${RUNNER_HOME}"
cd "${RUNNER_HOME}"

if [ ! -f .runner ]; then
  : "${RUNNER_URL:?Environment variable RUNNER_URL is required}"
  : "${RUNNER_TOKEN:?Environment variable RUNNER_TOKEN is required}"

  echo "Configuring runner for ${RUNNER_URL}..."
  runuser -u "${RUNNER_USER}" -- ./config.sh \
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
exec runuser -u "${RUNNER_USER}" -- ./run.sh
