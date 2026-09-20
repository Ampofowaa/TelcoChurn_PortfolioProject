FROM ghcr.io/mlflow/mlflow:v3.14.0
# boto3: the official image doesn't bundle it, and MLflow's S3 artifact
# repository needs it installed to serve artifacts from an s3:// root
# (--artifacts-destination s3://... --serve-artifacts) - same gap Group 6.5
# found and fixed for the training dependency tree.
RUN pip install --no-cache-dir psycopg2-binary boto3
