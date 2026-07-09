-- Bootstraps the 'mlflow' database on the same Postgres server/instance as
-- POSTGRES_DB. The mlflow service (docker-compose.yml) points its
-- --backend-store-uri at this database, but the Postgres image only
-- auto-creates POSTGRES_DB itself — nothing else creates this one.
-- Runs before 001_create_raw.sql (numeric prefix), though order doesn't
-- matter here since it targets a separate database.
CREATE DATABASE mlflow;
