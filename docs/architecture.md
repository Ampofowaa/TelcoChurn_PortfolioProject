# Architecture

> This diagram reflects the target architecture. It will be updated as each phase is completed.

```mermaid
flowchart TD
    A[Raw CSV\nIBM Telco Dataset] -->|ingest| B[(Postgres\ncustomers_raw)]
    B -->|validate| C[Pandera\n5 Quality Gates]
    C -->|feature engineering| D[SQL Views +\nColumnTransformer]
    D -->|train| E[LightGBM +\nOptuna + MLflow]
    E -->|calibrate + threshold| F[MLflow Model\nRegistry champion]
    F -->|serve| G[FastAPI\n/predict]
    G -->|UI| H[Streamlit\nDemo App]
    G -->|metrics| I[Prometheus\n+ Grafana]
    G -->|logs| J[Evidently\nDrift Monitor]
    J -->|trigger| K[Prefect\nRetrain Flow]
    K -->|re-runs| D
```

## Components

| Component | Tool | Phase |
|---|---|---|
| Data ingestion | Postgres 16 + SQLAlchemy | 1 |
| Data validation | Pandera | 2 |
| Feature engineering | SQL views + sklearn ColumnTransformer | 4 |
| Model training | LightGBM + Optuna + MLflow | 5 |
| Calibration + threshold | CalibratedClassifierCV + cost matrix | 6 |
| Model registry | MLflow (champion / challenger) | 7 |
| Pipeline reproducibility | DVC | 8 |
| Serving | FastAPI + uvicorn | 9 |
| Demo UI | Streamlit | 9 |
| Orchestration | Prefect 3 | 10 |
| CI/CD | GitHub Actions + AWS ECR + App Runner | 11–12 |
| Monitoring | Prometheus + Grafana + Evidently | 13 |
