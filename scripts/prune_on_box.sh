#!/usr/bin/env bash
# Runs ON the EC2 box, only after the smoke test has passed.
set -euo pipefail
# -a: without it only dangling images go, and every SHA-tagged release stayed on
# disk forever until the 19 GB root volume filled.
docker image prune -af
