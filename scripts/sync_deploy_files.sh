#!/usr/bin/env bash
# Uploads the deploy files from the current working tree to the S3 prefix the
# box syncs from. Shared by cd.yml's forward sync and its rollback (which runs
# it after checking out the previous commit's copies), so the two can never
# drift apart on what counts as a deploy file.
set -euo pipefail

bucket="telco-churn-mlflow-${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID must be set}"

aws s3 sync infra/deploy/ "s3://${bucket}/deploy-config/" \
    --exclude "RUNBOOK.md" --exclude "user-data.sh.tpl" --exclude "Caddyfile;C/*"
aws s3 cp docker/mlflow/Dockerfile "s3://${bucket}/deploy-config/docker/mlflow/Dockerfile"
