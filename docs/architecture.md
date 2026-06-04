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

## ML Workflow — a loop, not a straight line

The system diagram above shows the infrastructure flow. The diagram below shows the **modelling lifecycle**, including the two feedback loops that a linear summary hides.

```mermaid
flowchart TD
    A[Data Ingestion] --> B[Data Validation]
    B --> C[EDA]
    C --> D[Feature Engineering]
    D --> E[Baseline Models]
    E --> F["Error Analysis 1 — generative\nblind-spot profiling on baseline FNs"]
    F -.->|"hypothesis-driven\nfeatures back to FE"| D
    F --> G[Hyperparameter Tuning — Optuna]
    G --> H[Calibration + Threshold]
    H --> I[Sealed Test Evaluation]
    I --> J["Error Analysis 2 — confirmatory\nSHAP + FN/FP profiling of final model"]
    J --> K[Business Review]
    K -.->|"cost assumptions\nrevised"| H
    K -.->|"drift or\nnew data"| D
```

**Solid arrows** — main linear flow. **Dashed arrows** — feedback loops. Full rationale for both loops is in `ANALYSIS.md` §0.

---

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
