"""Shared fixtures for integration tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx2
import mlflow
import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from testcontainers.postgres import PostgresContainer

from telco_churn.data.ingest import ingest
from telco_churn.features.preprocessing import build_preprocessor
from telco_churn.serving.crm_data import (
    CrmGenerationParams,
    generate_crm_rows,
    load_crm,
    setup_crm_schema,
)
from telco_churn.utils.paths import compose_config, reset_active_config

# fastapi.testclient.TestClient imports the third-party `httpx` module by
# name; this project's dev group pins `httpx2` (a drop-in fork), not
# `httpx`, so nothing provides a real `httpx` distribution to import.
# alias_httpx() registers httpx2 into sys.modules['httpx'] once, here,
# before any test module gets a chance to import fastapi.testclient —
# conftest.py is collected first, so every integration test importing
# TestClient afterwards sees a resolvable `httpx`.
httpx2.alias_httpx()

__all__: list[str] = []


@pytest.fixture(autouse=True)
def _reset_active_config() -> None:
    """Clear any installed config between tests.

    Every fixture here that calls activate_config() already resets it in its
    own try/finally around the in-process calls that need it, so this is a
    backstop rather than load-bearing — mirrors tests/unit/conftest.py's
    fixture of the same name.
    """
    reset_active_config()


# ---------------------------------------------------------------------------
# Shared serving fixtures — tests/integration/test_api.py and
# test_serving_parity.py both need a real Postgres-backed customers_raw
# table and a real MLflow-registered champion, so the heavy setup lives
# here once rather than duplicated across both files.
# ---------------------------------------------------------------------------

# The customerid/tenure/monthlycharges triple is what both files assert
# against; every other CustomerFeatures field is filled with a fixed,
# schema-valid value that plays no role in the model's own committed
# feature space (serving_champion below trains on tenure/monthlycharges
# only, mirroring tests/unit/test_predict.py's raw-feature fixture).
SERVING_CUSTOMERS: list[dict[str, Any]] = [
    {"customerid": "serve-0001", "tenure": 5, "monthlycharges": 30.0},
    {"customerid": "serve-0002", "tenure": 40, "monthlycharges": 75.0},
    {"customerid": "serve-0003", "tenure": 70, "monthlycharges": 110.0},
]

_SERVING_CSV_TEMPLATE = (
    "{customerid},Female,0,Yes,No,{tenure},Yes,No,DSL,No,Yes,No,No,No,No,"
    "Month-to-month,Yes,Electronic check,{monthlycharges:.2f},"
    "{totalcharges:.2f},No\n"
)
_SERVING_CSV_HEADER = (
    "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,"
    "MultipleLines,InternetService,OnlineSecurity,OnlineBackup,"
    "DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,"
    "PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\n"
)


def _serving_csv_text() -> str:
    rows = [_SERVING_CSV_HEADER]
    for customer in SERVING_CUSTOMERS:
        rows.append(
            _SERVING_CSV_TEMPLATE.format(
                customerid=customer["customerid"],
                tenure=customer["tenure"],
                monthlycharges=customer["monthlycharges"],
                totalcharges=customer["monthlycharges"] * max(customer["tenure"], 1),
            )
        )
    return "".join(rows)


_CRM_PARAMS = CrmGenerationParams(
    random_state=42,
    tenure_advance_min_months=1,
    tenure_advance_max_months=6,
    contract_upgrade_probability=0.08,
    totalcharges_noise_scale=0.02,
)


@pytest.fixture(scope="module")
def serving_postgres_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Ephemeral Postgres, seeded with SERVING_CUSTOMERS in both
    customers_raw and customers_crm, exposed as a bare (driverless)
    connection URL.

    customer_lookup.py reads customers_crm, not customers_raw — seeding both
    keeps GET /customer/{id} and /predict/batch's ID-resolution path exercised
    against the same table serving/app.py actually queries in production.

    Bare 'postgresql://' rather than testcontainers' default
    '+psycopg2'-qualified URL: utils/db.py::get_engine() (used here to seed
    via ingest()) resolves the default psycopg2 dialect from a bare URL
    fine, and get_async_engine()'s _to_asyncpg_url rewrite — the actual
    serving/app.py request path — only fires on that exact 'postgresql://'
    prefix.
    """
    with PostgresContainer("postgres:16") as pg:
        url = pg.get_connection_url(driver=None)
        engine = create_engine(url)
        csv_path = tmp_path_factory.mktemp("serving_csv") / "serving_customers.csv"
        csv_path.write_text(_serving_csv_text(), encoding="utf-8")
        ingest(csv_path, engine)
        raw_df = pd.read_sql_table("customers_raw", engine)
        crm_rows = generate_crm_rows(raw_df, _CRM_PARAMS)
        setup_crm_schema(engine)
        load_crm(crm_rows, engine)
        engine.dispose()
        yield url


_CHAMPION_COMMITTED_FEATURES = ["tenure", "monthlycharges"]
_CHAMPION_THRESHOLD = 0.4
_CHAMPION_SEED = 1


def _fit_champion_pipeline(seed: int) -> CalibratedClassifierCV:
    rng = np.random.default_rng(seed)
    n = 200
    tenure = rng.integers(0, 72, size=n)
    monthlycharges = rng.uniform(20, 120, size=n)
    logit = -0.03 * tenure + 0.02 * monthlycharges - 1.0
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, prob), name="churn")
    X = pd.DataFrame({"tenure": tenure, "monthlycharges": monthlycharges})
    preprocessor = build_preprocessor([], [], _CHAMPION_COMMITTED_FEATURES)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LGBMClassifier(
                    random_state=seed,
                    n_estimators=30,
                    num_leaves=7,
                    min_child_samples=5,
                    class_weight="balanced",
                    n_jobs=1,
                    verbose=-1,
                ),
            ),
        ]
    )
    calibrated = CalibratedClassifierCV(
        pipeline,
        method="sigmoid",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        ensemble=False,
    )
    calibrated.fit(X, y)
    return calibrated


@pytest.fixture(scope="module")
def serving_champion(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A real calibrated pipeline, fit on tenure/monthlycharges, registered
    and aliased `champion` under configs/config.yaml's own literal
    mlflow.registered_model_name ('telco-churn-pipeline') — app.py's lifespan
    calls compose_config() with no overrides, so that name can't be
    redirected per-test; only mlflow.tracking_uri (env-resolved) can, which
    is what isolates this from every other test's registry.

    Logs golden_predictions.json in calibrate.py's own artifact shape (Task
    3/CLAUDE.md's register.py smoke-check format) so
    test_serving_parity.py can check served probabilities against it at
    cfg.register.golden_atol, the same tolerance register.py's own
    pre-promotion check uses.
    """
    seed_root = tmp_path_factory.mktemp("serving_champion_mlflow")
    tracking_uri = f"sqlite:///{seed_root / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        "test_serving_champion",
        artifact_location=(seed_root / "artifacts").as_uri(),
    )
    mlflow.set_experiment(experiment_id=experiment_id)

    registered_model_name = str(compose_config().mlflow.registered_model_name)
    golden_atol = float(compose_config().register.golden_atol)

    calibrated = _fit_champion_pipeline(_CHAMPION_SEED)
    golden_X = pd.DataFrame(
        {
            "tenure": [c["tenure"] for c in SERVING_CUSTOMERS],
            "monthlycharges": [c["monthlycharges"] for c in SERVING_CUSTOMERS],
        }
    )
    golden_p_hat = calibrated.predict_proba(golden_X)[:, 1]
    signature = infer_signature(golden_X, calibrated.predict_proba(golden_X))

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_dict(
            {"feature_selection": {"model_features": _CHAMPION_COMMITTED_FEATURES}},
            "training_manifest.json",
        )
        mlflow.log_dict(
            {
                "threshold": _CHAMPION_THRESHOLD,
                "scenario": "base",
                "scenarios": {
                    "base": {
                        "threshold": _CHAMPION_THRESHOLD,
                        "costs": {"c": 60.0, "r": 0.3, "ltv": 500.0, "arpu": 70.0},
                    },
                },
            },
            "threshold/threshold.json",
        )
        mlflow.log_dict(
            {
                "purpose": (
                    "serving-parity fixture - reproducibility only; scores "
                    "are in-sample and are not performance evidence"
                ),
                "customerid": [c["customerid"] for c in SERVING_CUSTOMERS],
                "rows": golden_X.to_dict(orient="records"),
                "p_hat": golden_p_hat.tolist(),
            },
            "calibration/golden_predictions.json",
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=calibrated,
            name="calibrated_model",
            signature=signature,
            input_example=golden_X,
            pyfunc_predict_fn="predict_proba",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=registered_model_name,
        )
    version = str(model_info.registered_model_version)
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(
        registered_model_name, version, "logged_model_id", model_info.model_id
    )
    client.set_registered_model_alias(registered_model_name, "champion", version)

    return {
        "tracking_uri": tracking_uri,
        "registered_model_name": registered_model_name,
        "model_version": version,
        "run_id": run_id,
        "threshold": _CHAMPION_THRESHOLD,
        "committed_features": _CHAMPION_COMMITTED_FEATURES,
        "golden_customerids": [c["customerid"] for c in SERVING_CUSTOMERS],
        "golden_p_hat": golden_p_hat.tolist(),
        "golden_atol": golden_atol,
    }


@pytest.fixture
def serving_env(
    monkeypatch: pytest.MonkeyPatch,
    serving_champion: dict[str, Any],
    serving_postgres_url: str,
) -> dict[str, Any]:
    """Point the real MLFLOW_TRACKING_URI/POSTGRES_URL env vars app.py's
    lifespan reads (via Hydra's ${oc.env:...} resolvers) at this module's
    seeded champion/Postgres, for the duration of one test.

    monkeypatch.setenv, not a bare os.environ mutation, so the previous
    value (or absence) is restored automatically at test teardown —
    serving/app.py's lifespan calls load_dotenv() with its default
    override=False, so a real developer .env's own MLFLOW_TRACKING_URI/
    POSTGRES_URL would never leak in ahead of these anyway, but restoring
    afterwards still matters for any other test in the same session that
    reads these vars.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(serving_champion["tracking_uri"]))
    monkeypatch.setenv("POSTGRES_URL", serving_postgres_url)
    return {"champion": serving_champion, "postgres_url": serving_postgres_url}


@pytest.fixture
def compose_config_with_auth_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force serving.auth.enabled=true for one test.

    serving.auth.enabled is a plain literal in configs/serving/api.yaml, not
    an ${oc.env:...} resolver, and app.py's lifespan calls compose_config()
    with no overrides (see serving_champion's docstring above for why
    per-test config redirection is otherwise limited to env-resolved keys) —
    so the only way to flip it for one test is to monkeypatch the
    compose_config name app.py imported into its own module namespace.
    """
    import telco_churn.serving.app as app_module

    real_compose_config = app_module.compose_config

    def _compose_with_auth_enabled(overrides: list[str] | None = None) -> Any:
        cfg = real_compose_config(overrides)
        cfg.serving.auth.enabled = True
        return cfg

    monkeypatch.setattr(app_module, "compose_config", _compose_with_auth_enabled)


@pytest.fixture
def serving_customer_payload() -> Any:
    """Factory: a schema-valid 19-field CustomerFeatures/PredictRequest body
    for a given (customerid, tenure, monthlycharges) — every other field is
    fixed, since serving_champion's committed feature space never reads them.
    A fixture (rather than a plain module-level function) so both test_api.py
    and test_serving_parity.py can use it with no cross-file import — pytest
    resolves it through normal fixture injection instead.
    """

    def _make(customerid: str, tenure: int, monthlycharges: float) -> dict[str, Any]:
        return {
            "customerid": customerid,
            "gender": "Female",
            "seniorcitizen": 0,
            "has_partner": "Yes",
            "dependents": "No",
            "tenure": tenure,
            "phoneservice": "Yes",
            "multiplelines": "No",
            "internetservice": "DSL",
            "onlinesecurity": "No",
            "onlinebackup": "Yes",
            "deviceprotection": "No",
            "techsupport": "No",
            "streamingtv": "No",
            "streamingmovies": "No",
            "contract_type": "Month-to-month",
            "paperlessbilling": "Yes",
            "paymentmethod": "Electronic check",
            "monthlycharges": monthlycharges,
            "totalcharges": monthlycharges * max(tenure, 1),
        }

    return _make
