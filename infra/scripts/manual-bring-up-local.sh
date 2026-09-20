#!/usr/bin/env bash
set -euo pipefail

# One-off manual bring-up of Group 9's serving stack (PHASE_12A_TODO.md's
# last unchecked Group 9 item), run BEFORE Group 11's CD pipeline exists.
# Run this from your own terminal with real AWS access — never via a Claude
# Code `!`-prefixed command (sandboxed, no AWS creds/SSM access).
#
# Phases A-D run here, locally. Phase E runs inside an SSM session on the
# box itself (manual-bring-up-on-box.sh is the reference for that — paste
# it into the session, it isn't executed from here). Phase F (verification)
# is a handful of curls from wherever you have internet access, after E.
#
# Usage: run from the repo root: bash infra/scripts/manual-bring-up-local.sh

REGION="us-east-1"
ACCOUNT_ID="742306314937"
REPO_ROOT="$(pwd)"

# Git Bash's MSYS layer rewrites a bare leading-slash argument like
# /telco-churn/api-key into a Windows path (e.g. C:/Program Files/Git/telco-churn/api-key)
# before handing it to aws.exe, which then rejects it with "Parameter name
# must be a fully qualified name." Harmless on Linux/macOS; required here.
export MSYS_NO_PATHCONV=1

if [ ! -f "pyproject.toml" ]; then
  echo "Run this from the repo root (pyproject.toml not found here)." >&2
  exit 1
fi

echo "=== Assuming terraform-admin (MFA) - sourced directly in this shell, not a"
echo "    subshell, so the exported session credentials survive for every"
echo "    phase below, not just the immediately-following command. ==="
source infra/scripts/assume-admin.sh

echo "=== Phase A: terraform apply (t3.small bump + new deploy-config files) ==="
cd infra
terraform plan
terraform apply
# terraform apply prompts for its own yes/no confirmation - intentionally
# not passing -auto-approve here.
cd "$REPO_ROOT"
echo "Note: the instance_type change causes AWS to stop/modify/start the box"
echo "in place (same instance ID, same EBS volumes/data) - expect ~1-2 min"
echo "of SSM connectivity loss during that resize, not a full re-provision."
echo

echo "=== Phase B: set the real API key (Group 7 left this as a placeholder) ==="
aws ssm put-parameter \
  --name /telco-churn/api-key \
  --type SecureString \
  --value "$(openssl rand -hex 32)" \
  --overwrite \
  --region "$REGION"
echo "Real API key set. Fetch it when you need it for verification/README:"
echo "  aws ssm get-parameter --name /telco-churn/api-key --with-decryption --query Parameter.Value --output text --region $REGION"
echo

echo "=== Phase C: build + push real api/ui images to ECR ==="
TAG="$(git rev-parse --short HEAD)"
API_REPO="$(cd infra && terraform output -json ecr_repository_urls | python -c 'import json,sys; print(json.load(sys.stdin)["api"])')"
UI_REPO="$(cd infra && terraform output -json ecr_repository_urls | python -c 'import json,sys; print(json.load(sys.stdin)["ui"])')"
echo "Tagging with git SHA: $TAG (ECR repos are image_tag_mutability=IMMUTABLE -"
echo "re-running this script against the same commit will fail on push; commit"
echo "something first, or manually pick a different tag, if you need a retry)."

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

docker build -f docker/api/Dockerfile -t "${API_REPO}:${TAG}" .
docker push "${API_REPO}:${TAG}"

docker build -f docker/ui/Dockerfile -t "${UI_REPO}:${TAG}" .
docker push "${UI_REPO}:${TAG}"
echo

echo "=== Phase D: point the api-image/ui-image SSM params at the real tags ==="
aws ssm put-parameter --name /telco-churn/api-image --type String \
  --value "${API_REPO}:${TAG}" --overwrite --region "$REGION"
aws ssm put-parameter --name /telco-churn/ui-image --type String \
  --value "${UI_REPO}:${TAG}" --overwrite --region "$REGION"
echo

INSTANCE_ID="$(cd infra && terraform output -raw instance_id)"
echo "=== Local phases done. Instance: $INSTANCE_ID ==="
echo "Next: start an SSM session and run infra/scripts/manual-bring-up-on-box.sh's"
echo "commands there (it's a reference to paste from, not something this script runs):"
echo
echo "  aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo
