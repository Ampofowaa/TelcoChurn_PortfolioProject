# Must match the mlflow version in uv.lock (enforced by
# tests/unit/test_mlflow_version_parity.py). This server shares the `mlflow`
# tracking database with the api, ui and training clients, and a newer MLflow
# server silently migrates that database to its own schema on its first real
# data request - after which every older client refuses it ("Detected
# out-of-date database schema"). A Dependabot bump to v3.16.1 against 3.14
# clients took live predictions down on 2026-10-08. Upgrade both sides in one
# change, never this image alone.
FROM ghcr.io/mlflow/mlflow:v3.14.0
# boto3: the official image doesn't bundle it, and MLflow's S3 artifact
# repository needs it installed to serve artifacts from an s3:// root
# (--artifacts-destination s3://... --serve-artifacts) - same gap Group 6.5
# found and fixed for the training dependency tree.
# mlflow[auth]: the image also lacks Flask-WTF, which `--app-name basic-auth`
# (infra/deploy/compose.prod.yml) needs for CSRF - without it the server dies
# on import. Pinned to the base image's own version so pip resolves only the
# extra's dependencies and never swaps mlflow itself.
RUN pip install --no-cache-dir psycopg2-binary boto3 "mlflow[auth]==3.14.0"
