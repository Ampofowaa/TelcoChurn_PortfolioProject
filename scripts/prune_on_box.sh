#!/usr/bin/env bash
# Runs ON the EC2 box, only after the smoke test has passed.
set -euo pipefail
docker image prune -f
