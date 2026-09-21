#!/usr/bin/env bash
# Usage: scripts/ssm_run.sh <local-script> [timeout-seconds]
#
# Ships a local script to the EC2 box via ssm:SendCommand, waits for it, prints
# the box's stdout/stderr, and exits non-zero if the remote script did. The box
# is found by its Project=telco-churn tag — the same scope iam.tf's
# SendCommandTarget statement grants, so no instance ID is needed anywhere.
set -euo pipefail

script="$1"
timeout="${2:-900}"
REGION="us-east-1"

# base64 sidesteps every quoting problem of embedding a script in JSON params.
# The script is written to a temp file and run with stdin from /dev/null. It
# used to be piped into `bash`, and any command in it that reads stdin
# (`docker compose run`/`exec`) swallowed the rest of the script - later lines
# never ran while SSM still reported Success.
encoded=$(base64 -w0 "${script}")

command_id=$(aws ssm send-command \
    --document-name AWS-RunShellScript \
    --targets "Key=tag:Project,Values=telco-churn" \
    --parameters "commands=[\"f=\$(mktemp) && echo ${encoded} | base64 -d > \$f && bash \$f < /dev/null; rc=\$?; rm -f \$f; exit \$rc\"]" \
    --timeout-seconds "${timeout}" \
    --region "${REGION}" \
    --query Command.CommandId --output text)
echo "==> ${script}: SSM command ${command_id}"

# The invocation row appears a moment after send-command returns.
instance_id=""
for _ in $(seq 1 20); do
    instance_id=$(aws ssm list-command-invocations --command-id "${command_id}" \
        --region "${REGION}" --query 'CommandInvocations[0].InstanceId' --output text 2>/dev/null || true)
    [ -n "${instance_id}" ] && [ "${instance_id}" != "None" ] && break
    sleep 3
done
[ -n "${instance_id}" ] && [ "${instance_id}" != "None" ] || { echo "no instance received the command"; exit 1; }

deadline=$((SECONDS + timeout + 60))
while :; do
    status=$(aws ssm get-command-invocation --command-id "${command_id}" \
        --instance-id "${instance_id}" --region "${REGION}" --query Status --output text)
    case "${status}" in
        Success | Failed | Cancelled | TimedOut) break ;;
    esac
    [ "${SECONDS}" -lt "${deadline}" ] || { echo "gave up polling (last status: ${status})"; exit 1; }
    sleep 5
done

aws ssm get-command-invocation --command-id "${command_id}" \
    --instance-id "${instance_id}" --region "${REGION}" \
    --query '[StandardOutputContent, StandardErrorContent]' --output text

echo "==> ${script}: ${status}"
[ "${status}" = "Success" ]
