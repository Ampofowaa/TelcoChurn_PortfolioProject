# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions map to project phases, not strict Semantic Versioning: a new phase bumps MINOR
(e.g. Phase 5 starts at `0.5.0`); QA passes and sub-steps within a phase bump PATCH.
See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap.

---

## [Unreleased]

---

## [0.10.0] - 2026-08-22 — Phase 10a-i: Data Infrastructure — Alembic + Prediction Tables

*Adopts Alembic as this project's schema-migration tool (`prediction_outcomes` is the first table that can't be regenerated from the CSV) and lands the durable capture path for production inference: `prediction_log` (every `/predict`/`/predict/batch` call) and `prediction_outcomes` (matured churn labels). Full design in `prediction_logging_plan.md` Part B/C.*

### Added
- **`alembic/`, `alembic.ini`, `src/telco_churn/data/tables.py`** — four linear migrations (`customers_raw` → `customers_crm` → `prediction_log` → `prediction_outcomes`), each with a real `downgrade()`. `utils/db.py::apply_migrations()` is the belt-and-braces call `data/ingest.py`, `serving/crm_data.py`, and `serving/outcomes.py` now run at CLI start instead of executing raw DDL.
- **`src/telco_churn/serving/prediction_log.py`** — `build_log_rows()` (pure) + `write_log_rows()` (async I/O boundary): one `prediction_log` row per scored customer, capturing `feature_snapshot`, the serving model's identity, `dual_score_mode`/challenger/champion probabilities whenever a challenger was evaluated, and `resolution_kind`. Wired into `serving/app.py`'s `POST /predict`/`POST /predict/batch` via `BackgroundTasks` — fires after the response is sent, fail-open (log-and-swallow plus a new `prediction_log_write_failures_total` Prometheus counter). Config-gated (`configs/serving/api.yaml: prediction_log.enabled`).
- **`src/telco_churn/serving/outcomes.py`** — `build_outcome_row()` (pure) + `write_outcomes()` (`INSERT ... ON CONFLICT (customerid, observed_at, source) DO NOTHING`); `__main__` CLI (`python -m telco_churn.serving.outcomes`, `make record-outcome`) for manually recording a matured churn label. Raises normally on failure — no live request path to protect.
- **`serving/schemas.py::PredictRequest.resolution_kind`**, **`customer_lookup.py::resolve_batch_rows`**'s fifth return value — client-/server-declared provenance (`id_only` / `full_inline` / `partial_override`) for how a scored row was actually built, threaded through to `prediction_log`. `ui/streamlit_app.py`'s Score a Customer tab sets it from whether a Fetch preceded the Predict click.
- **`Makefile`** — `alembic-upgrade`, `record-outcome` targets.
- Test suite: `tests/unit/test_prediction_log.py` (6), `test_outcomes.py` (3); `tests/integration/test_alembic_migrations.py` (up/down/up roundtrip), `test_outcomes_subprocess.py` (success, `ON CONFLICT` dedup, exit-1); three new `prediction_log`-focused tests in `test_api.py` plus a `compose_config_with_prediction_log_disabled` fixture.

### Changed
- **`serving/predict.py::predict_single`** — now returns `(PredictResponse, ScoredBatch)` instead of just the response, so `/predict`'s background prediction-log write doesn't re-score the row a second time.
- **`sql/schema/001_create_raw.sql`, `sql/schema/003_create_customers_crm.sql`** — deleted, not archived (superseded by Alembic migrations 001–002; leaving them would race Alembic on every fresh Docker volume via the `docker-entrypoint-initdb.d` mount). `000_create_mlflow_db.sql`/`002_create_optuna_schema.sql` stay outside Alembic's scope — MLflow and Optuna self-manage those schemas.
- Local Docker Postgres reinitialized (`docker compose down -v`) and the full training cycle re-run end-to-end (`dvc repro --force` through `register.py`) to mint champion version 1 against the fresh Alembic-managed schema.

---

## [0.9.1] - 2026-08-20 — Phase 9 sub-step: `customers_crm` live-lookup source

*Gives `GET /customer/{customerid}` and `/predict/batch`'s ID-resolution path a source genuinely distinct from the frozen training snapshot — the Score a Customer tab's "Fetch customer" no longer silently reuses `customers_raw` and presents it as current. See `prediction_logging_plan.md` Part A.*

### Added
- **`sql/schema/003_create_customers_crm.sql`**, **`src/telco_churn/serving/crm_data.py`** — new `customers_crm` table: a seeded "current state" derivation of `customers_raw` (`generate_crm_rows`), never touching the read-only raw table. Tenure advances +1–6 months, contract only upgrades (never downgrades), `totalcharges` carries forward plus accrued charge and seeded noise; everything else held fixed. `setup_crm_schema`/`load_crm` mirror `data/ingest.py`'s idempotent truncate-and-reload pattern. `make crm-data` runs it (after `db-up`/ingest).
- **`serving/schemas.py::CustomerLookupResponse`** — wraps `CustomerFeatures` with a `crm_snapshot_at` provenance timestamp; `crm_snapshot_at` stays off `CustomerFeatures` itself since it isn't a model feature.
- Test suite: `tests/unit/test_crm_data.py` (pure generator — determinism, bounds, contract-never-downgrades), `tests/integration/test_crm_data_postgres.py` (schema/load round-trip + the required `__main__` subprocess test).

### Changed
- **`serving/customer_lookup.py::lookup_customers`** — reads `customers_crm`, not `customers_raw`.
- **`serving/app.py::get_customer`** — `GET /customer/{customerid}` now returns `CustomerLookupResponse{features, crm_snapshot_at}` instead of bare `CustomerFeatures`.
- **`ui/streamlit_app.py::_render_lookup_tab`** — unwraps the new response shape; the success message is now genuinely dynamic (*"Loaded customer X (CRM snapshot as of ...)"*) instead of a static caption on a query that was actually hitting the training table.
- **`tests/integration/conftest.py::serving_postgres_url`** — also seeds `customers_crm` from the ingested `customers_raw`, since serving now resolves lookups against the CRM table.
- **`tests/integration/test_serving_parity.py`** — the ID-only golden-parity test asserted served probabilities matched `golden_predictions.json` at `atol=1e-9`; since ID-only resolution now goes through `customers_crm`'s seeded tenure nudge, bit-exact parity against the frozen training values is no longer the right invariant. Replaced with a self-consistency check: an ID-only `/predict/batch` request must match an inline `/predict` request built from that same customer's current `GET /customer/{id}` row.
- **`examples/sample_batch_predictions.csv`** — regenerated from `customers_crm` (47 dev-partition ID-only rows, up from 5, still keeping the one inline-prospect and two partial-override demo rows) so the Batch Prediction tab exercises real scale, not just the three item shapes. Now reproducible via **`scripts/generate_sample_batch_predictions.py`** — deterministic (`dev_ids() ∩ customers_crm`, sorted, first 47), also prints the 5 customer IDs referenced below for the Score a Customer tab's "Fetch customer" demo. **`README.md`** — Quick Start's `curl` example and the Score a Customer tab's "Fetch customer" walkthrough both point at customer IDs that actually resolve against the current data.

---

## [0.9.0] - 2026-08-18 — Phase 9: Serving + Streamlit UI

*Stands up the champion-serving path end to end: FastAPI (`/predict`, `/predict/batch`, `/customer/{id}`, `/health`, `/ready`, `/metrics`) with hot-reload and a capacity-constrained contact policy, plus a config-gated shadow/canary mechanism that continuously dual-scores traffic against a `challenger` whenever one exists. Streamlit UI and both Docker images complete a locally-runnable full stack (`docker compose up`). The champion alias still flips on an offline gate alone — shadow/canary is real routing/logging/metrics machinery, not a live-traffic validation gate, since this dataset has no live traffic to validate against (`ANALYSIS.md` §9 item 13).*

### Added
- **`src/telco_churn/models/policy_config.py::expected_value_per_customer`**, **`serving/contact_policy.py::select_contacts`** — capacity-constrained contact policy: ranks by expected value, caps at `contact_capacity`/`campaign_budget`.
- **`src/telco_churn/serving/schemas.py`** — Pydantic v2 request/response models; `customerid` pass-through only, stripped before `predict_proba`. Structural guard ties `CustomerFeatures`'s field set to `FEATURE_SCHEMA`'s raw columns (`tests/unit/test_schemas.py`).
- **`src/telco_churn/serving/predict.py`** — champion load at startup; `resolve_serving_model` TTL-polls `champion`/`challenger`, diffing logged dependencies against what's installed (`models/environment_parity.py`) before hot-reloading — fail-open on a refresh problem, fail-closed on the first load. Shadow/canary routing (`resolve_champion_model`/`resolve_challenger_model`, consistent-hash canary bucketing on `customerid` salted by the challenger's own `model_version`) and five Prometheus metrics (`predictions_total`, `predicted_probability`, `predictions_above_threshold_total`, `shadow_canary_score_distance`, `shadow_canary_agreement_total`). SHAP explainer and threshold policy reload in lockstep with the model inside one `ModelBundle`.
- **`src/telco_churn/serving/app.py`** — `POST /predict` (probability + optional SHAP explanation), `POST /predict/batch` (three item shapes, partial success, 500-row cap), `GET /customer/{id}`, `GET /health` (liveness only), `GET /ready` (503 pre-load, graceful SIGTERM drain), `GET /metrics`. Optional API-key auth (`configs/serving/api.yaml: auth.enabled`, off by default), never applied to `/health`/`/ready`/`/metrics`.
- **`src/telco_churn/ui/streamlit_app.py`** — Lookup, Manual/what-if, Bulk CSV, and "About this model" tabs; widgets sourced from `features/schema.py`'s constraint sets.
- **`docker/api/Dockerfile`, `docker/ui/Dockerfile`** — multi-stage, built from `uv.lock`. `docker-compose.yml` gains `fastapi`/`streamlit` services, unprofiled so bare `docker compose up -d` brings up the full stack.
- **`scripts/smoke_test_serving.sh`**, `make smoke-test-serving` — curls `/ready`, `/predict`, `/customer/<id>` against a running compose stack.
- **`docs/architecture.md`** — new Shadow/Canary Serving section: mechanism flowchart, Prometheus metric labelling scheme, and a real log/metrics snippet from real dev-partition customers run through the actual mechanism.
- **`examples/sample_batch_predictions.csv`** — 8 real dev-partition customers demonstrating all three `/predict/batch` item shapes (ID-only, full inline "new prospect," ID-plus-partial-override) in one upload; referenced from the Streamlit Bulk CSV tab and the README Quick Start.
- Test suite: `tests/unit/test_predict.py` (23), `test_contact_policy.py`, `test_schemas.py`, `test_environment_parity.py`; `tests/integration/test_api.py` (9), `test_predict_subprocess.py`, `test_serving_parity.py`; `tests/streamlit/test_streamlit_app.py`. New `make test-serving` scoped target.

### Changed
- **`docker-compose.yml`** — `postgres`/`mlflow` lose `profiles: [infra]` (Compose refuses to resolve an unprofiled service's `depends_on` into an inactive profile); infra-only startup is now `docker compose up -d postgres mlflow`. Updated to match in `CLAUDE.md`, `Makefile` (`db-up`/`db-down`), `PROJECT_PLAN.md`, `notebooks/00-data-ingestion.ipynb`.
- **`ANALYSIS.md`** §9 — gains items 13–16 on the monitoring/shadow-canary stack's mechanism-vs-live-traffic-validation boundary, capacity-pool interference between canary and champion cohorts, and the per-model cost-scenario staleness risk between a champion and a later challenger.
- **`README.md`** — Phase 9 marked done (Project Status, pipeline diagram, Tech Stack); restructured into Results → Dataset → Tech Stack → Pipeline → Modelling → Pipeline Versioning (DVC) → Serving, each of the last three a mechanism-only table linking out to `ANALYSIS.md`/`docs/architecture.md` for rationale rather than restating it; Quick Start gains a step to bring up the full compose stack and try `/predict`/`/predict/batch`/the Streamlit UI; Project Structure lists `serving/`, `ui/`, `docker/`, `examples/`.

### Fixed
- **`src/telco_churn/serving/app.py`** — `lifespan()` was `await`ing the initial model load directly in ASGI startup; uvicorn's own `Server.startup()` only opens its listening socket *after* lifespan startup returns, so with no champion yet to resolve, `/health` (meant to be reachable unconditionally) was unreachable too, not just `/ready`. Fixed by moving the initial load onto the same background task the hot-reload poll already runs on. Caught by `tests/integration/test_predict_subprocess.py`, a real uvicorn subprocess over a real socket.
- **`src/telco_churn/serving/predict.py::score_request`** — canary-only mode (shadow disabled) was scoring the challenger against the full batch instead of only the bucketed subset; a real canary router never double-scores every request the way shadow does.
- **`src/telco_churn/serving/predict.py::_canary_bucket_mask`** — bucketing now salted with the challenger's own `model_version`, so the canary population reshuffles the next time `challenger` moves to a different version instead of being the same customer subset on every canary this project ever runs.
- **`src/telco_churn/serving/app.py::require_api_key`** — compared the API key with `!=`, not constant-time; fixed with `secrets.compare_digest`.
- **`docker-compose.yml`** — MLflow 3.14's server-side Host-header allowlist rejected the Compose DNS name (`mlflow`) `fastapi`/`streamlit` reach it through; fixed via `MLFLOW_SERVER_ALLOWED_HOSTS` with the needed `:*` wildcard entries.
- **`src/telco_churn/ui/streamlit_app.py::_render_bulk_tab`** — no client-side row-count check against `batch.max_size`; now shown against the live limit and disables submission over it, instead of only surfacing the API's `413` after upload.

---

## [0.8.0] - 2026-08-15 — Phase 8: DVC Pipeline Wrap

*Wraps `ingest → validate → split → features → train → calibrate → threshold → evaluate → error_analysis` as a 9-stage, content-hashed DVC DAG, reproducible end to end via `dvc repro`. Required extracting registry writes out of `calibrate.py`/`evaluate.py`/`error_analysis.py` into `register.py` as the sole registry-mutating module, switching model resolution to run-id receipts, and splitting the automated gate verdict from the human review.*

### Added
- **`dvc.yaml`, `dvc.lock`, `.dvc/`, `.dvcignore`** — 9 stages with explicit `deps`/`params`/`outs`; receipts (`cache: false`) stand in for side effects DVC can't hash (Postgres rows, MLflow runs). No remote yet — local cache only through Phase 11.
- **`Makefile`** — `repro`/`dag`/`metrics`/`params` targets; `dvc.lock` merge-conflict policy documented (never hand-edit, regenerate via `dvc repro`).
- **`data/validate.py::write_validation_receipt`** — unconditional, deterministic writer for `reports/validation_receipt.json`, giving `split`/`features` a DVC-enforced "validate must pass first" edge.
- **`data/ingest.py`** — emits `reports/ingest_receipt.json` as `ingest`'s DVC out.
- **`data/schema.py`** — `CleanedSchema` deleted; its one live invariant promoted onto `RawSchema`. `validate_clean` and the `cleaned` param removed with it.
- **`configs/policy/threshold.yaml` → `reports/policy/threshold.yaml`** — moved so it can be a DVC `metrics:` out of the `threshold` stage.
- **`utils/mlflow.py`** — receipt read/write helpers (`{train,calibrate,eval,error_analysis}_receipt`) and `resolve_model_identifier`; every stage resolves its upstream model by run-id/version override or receipt.
- **`models/gate.py`** — comparative-regime recall guardrail gains a non-inferiority check alongside its absolute floor (`GateBars.recall_non_inferiority_margin`). `tests/unit/test_gate.py` 28 → 33 tests.
- **`models/train/feature_selection.py`** (new) — `run_feature_selection_step`, the mechanism deciding `features/schema.py::COMMITTED_FEATURES` via paired-bootstrap ablation. `tests/unit/test_train_feature_selection.py` (11 tests).
- **PR B1.** New `models/review.py` + `configs/review/default.yaml` — standalone CLI stamping append-only `promotion_review.json` onto the eval run, independent of Jupyter. `tests/unit/test_review.py` (4), `tests/integration/test_review_subprocess.py` (2).
- **`models/register.py::register_challenger`** — extracted from `calibrate.py`: mints the calibrated pipeline as a registry version, tags `promotion_status: pending`, verifies reload parity, points `challenger`.
- **`models/register.py::refresh_champion_reference`** — re-points a promoted champion's `eval_run_id`/gate-criteria tags at a fresh out-of-band re-evaluation. `tests/unit/test_register.py` (+6), `tests/integration/test_register_subprocess.py` (+2).
- **PR B2.** New `models/shap_values.py` (`unwrap_calibrated_pipeline`/`compute_shap_values`), shared by `calibrate.py` (dev) and `error_analysis.py` (test). `tests/unit/test_shap_values.py` (5).
- **`models/calibrate.py`** — logs dev-SHAP once per cycle (`dev_shap_values.parquet`, `calibration/dev_shap_summary.json`), the evidence `threshold.py`'s V3 veto binds on.
- **`models/threshold.py`** — dev-OOF pre-seal screen binds a second criterion, V3 (direction sanity), alongside the calibration-slope band. New `configs/threshold/default.yaml` keys `v3_top_k_features`/`v3_min_direction_magnitude`, derived from a real dev-SHAP ranking (`ANALYSIS.md` §0).
- **PR C.** Five new leaf modules so `calibrate.py`/`threshold.py`/`evaluate.py`/`error_analysis.py`/`register.py` stop cross-importing: `models/calibration_metrics.py`, `models/artifacts.py`, `models/dev_features.py`, `models/policy_config.py`, `utils/hashing.py`.
- **`tests/unit/test_architecture.py`** — new `test_no_module_imports_from_a_dunder_main_bearing_module`, enforcing the cross-import ban above.
- **`notebooks/05-evaluation-and-error-analysis.ipynb`** — new "Incumbent comparison" subsection rendering `metrics["incumbent_summary"]` and the comparative-regime bootstrap deltas.

### Changed
- **`models/review.py`** — `notes` is now hard-required; logs where a reviewer finds this cycle's MLflow diagnostics before validating input.
- **`models/evaluate.py`** — `load_fitted_model` takes an explicit `model_uri`; step orchestrators take an already-resolved `(run_id, model_version[, model_uri])` instead of re-deriving it; `load_incumbent_proba` reads the champion's own `test_predictions.parquet` instead of re-scoring it live, checked against `champion_data_content_hash`.
- **`models/evaluate.py::resolve_evaluation_champion`** — explicit `champion_version` override that never touches the live `champion` alias; omitted, falls back to a live alias read.
- **`models/threshold.py`** — `model_version` stamp replaced with `logged_model_id`; `validation_payload` gains `screen_passed`.
- **`models/error_analysis.py`** — dev-OOF diagnostics resolved via `evaluate.load_dev_oof_diagnostics(run_id, cfg)` over MLflow instead of a local file; SHAP-based direction-sanity check is now a reported-only re-audit of the feature set `threshold.py` already vetoed on dev.
- **`models/gate.py`** — comparative-regime PR-AUC selection now also requires the candidate to clear the cold-start bar, not only a positive Δ; Brier guardrail reinstates the absolute BSS floor alongside its non-inferiority check.
- **`models/gate.py`, `models/evaluate.py`** — `decide_promotion` takes an explicit `regime` parameter instead of an unused `incumbent` argument; no longer returns a `review` field.
- **`models/calibrate.py`** — no longer performs any registry write; fits, logs, and writes `reports/calibrate_receipt.json` only.
- **`models/evaluate.py`, `models/error_analysis.py`** — stop writing model-version tags; `register.py` is now the sole writer.
- **`models/register.py`** — reads `promotion_review.json`'s last entry for the human verdict; a `rejected` verdict routes through `_tag_rejected` like every other reject path.
- **`configs/config.yaml`** — `paths.processed_data` is now a plain literal, not an `${oc.env:...}` interpolation DVC can't hash; overrides go through Hydra CLI overrides via new `utils/paths.py::activate_config`/`reset_active_config`.
- **`data/ingest.py`** — `--csv-path` CLI flag removed; raw CSV path now set via `paths.raw_data=<path>` like every other entry point.
- **`models/train/feature_freeze.py` → `feature_audit.py`** (renamed) — no longer runs the ~85,000-scoring-pass keep-vs-reduce comparison every cycle; committed features now read from `features/schema.py::COMMITTED_FEATURES`; the module only audits the committed set, doesn't decide it.
- **`features/accessor.py`, `features/build.py`** — processed-features artifact moved from `telco_churn_processed.csv` to `telco_churn_features.parquet`.
- **`models/train/__main__.py`** — automated pipeline now runs Steps 3-5 directly against the frozen model family; Steps 1-2 (family comparison) moved to notebook-only.
- **`models/train/selection_review.py` → `feature_selection.py`** (renamed) — the module decides `COMMITTED_FEATURES`, doesn't just review it.
- **`CLAUDE.md`** — Model Registry section rewritten to describe `register.py` as the sole entry point for minting/tagging/pointing/flipping.
- **`README.md`** — Tests/Coverage badges refreshed to 835 passed / 51 skipped, 95.7% coverage.

### Removed
- **`models/train/feature_freeze.py`** — the automated `decision == "reduced"` branch and its full-vs-reduced figures; the underlying comparison functions remain, called only from the notebook's on-demand review.
- **`configs/config.yaml`, `configs/training/feature_selection.yaml`** — dead CV/bootstrap config keys, replaced by module-level constants once their automated-pipeline call sites were removed.

### Fixed
- **`tests/integration/test_train_subprocess.py`** — a receipt-writer leak into the tracked working directory; now `tmp_path`-scoped like the other subprocess suites.
- **`models/evaluate.py`, `models/register.py`** — the four `champion`-alias lookups now check MLflow's `error_code` instead of catching every `MlflowException` as "not found," so a transient error can no longer silently flip the gate to cold-start regime.
- **`models/register.py::rollback_champion`** — no longer raises on a first-ever promotion with nothing tagged `promoted`; unsets the alias and records a `rolled_back` event.
- **`models/calibrate.py`** — `promotion_status: pending` now tagged immediately after mint, before the parity check — previously a parity failure left the version untagged, invisible to the rollback rule and the pending-reaper.
- **`models/register.py`** — the `promoted` tag now only lands after the post-flip drift-reference/model-card build succeeds; a failure rolls the alias back and re-raises.
- **`models/error_analysis.py`** — per-feature SHAP metrics capped to `top_k_shap_features` instead of one metric key per one-hot-expanded feature.
- **Phases 1-7 QA pass (13 smaller items)** — `CleanedSchema`/`validate_clean` wired into `build.py`'s `__main__`; an Optuna `max_depth`/`num_leaves` coupling attempt reverted after it excluded the 1-SE rule's preferred region; new `capacity_budget_check` diagnostic; Optuna's Postgres schema now created from a versioned `sql/schema/002_create_optuna_schema.sql` file; assorted stale-docstring/comment fixes and one mis-named test corrected.
- **`.gitignore`** — the six per-cycle receipts gitignored instead of git-committed — a committed receipt hash-matched `dvc.lock` on a fresh clone even though the fresh backend had never run, so `dvc repro` skipped stages against empty infra.
- **`src/telco_churn/models/train/log_model.py`** — `_build_training_manifest`'s `model_comparison` section (the recomputed paired-Δ vs LogReg) replaced by `model_family_committed` (`model_family`, `decision_reference`, `decision_run_id` — referenced from `common.py`, never recomputed). `run_model_logging_step` drops the now-unused `comparison` parameter.
- **`src/telco_churn/models/train/candidates.py`, `comparison.py`** — module docstrings updated: notebook-only (`03a-model-selection.ipynb`), not called from the automated pipeline. `run_candidate_step`/`run_comparison_step`/`run_diagnostics_step` gain explicit `cv_folds`/`cv_repeats`/`n_bootstrap`/`segment_n_bootstrap` parameters (defaulting to the new `_FAMILY_REVIEW_*` module constants below) so notebook call sites need no override while test fixtures still can, same pattern `reduced_set_bootstrap_test`'s `n_bootstrap` parameter already established.
- **`notebooks/03a-model-selection.ipynb`** — rewritten from a passive renderer ("already ran via `python -m telco_churn.models.train`") to the on-demand trigger: §1 now calls `run_candidate_step`/`run_comparison_step` directly against a freshly loaded dev partition, logging its own `dummy_prior`/`logreg_cv`/`lgbm_default`/`model_comparison` MLflow runs, then prints the `model_comparison` run id to cite as `COMMITTED_MODEL_FAMILY_DECISION_RUN_ID`. The bias/variance diagnostic loop (§4) reuses that same loaded dev partition instead of a second independent load.
- **`src/telco_churn/models/train/common.py`** — `_load_dev_features` now routes through `features/accessor.py::load_features()` instead of its own `_load_processed` CSV reader, closing the two-paths-to-one-file drift the Phase 8 CSV→Parquet swap below would otherwise have opened. `_load_dev_features` drops its now-unused `cfg` parameter.
- **`src/telco_churn/features/accessor.py`, `features/build.py`** — processed-features artifact moved from `telco_churn_processed.csv` (`pd.read_csv`/`to_csv`) to `telco_churn_features.parquet` (`pd.read_parquet`/`to_parquet`), landed early on the phase-8-prereqs branch ahead of the rest of Phase 8's `dvc.yaml` wiring. `FEATURES_FILENAME` is the single place the name/format is declared; `make features` now needs a re-run to regenerate the local artifact under the new name.
- **`src/telco_churn/models/train/feature_freeze.py`** — `run_selection_step` renamed to `run_feature_audit_step`. The old name implied an action ("select"/"freeze") the function doesn't perform at runtime: the committed feature set is frozen out-of-band, as a one-line `COMMITTED_FEATURES` edit reviewed via PR after a notebook run — this function only reads that already-frozen list and logs a permutation-importance/SHAP audit against it. Propagated through `models/train/__init__.py`, `models/__init__.py`, `models/train/__main__.py`, `models/calibrate.py`'s docstring, and the corresponding test/notebook references. `models/train/__main__.py` also gains a short comment block naming all five pipeline steps (Steps 1-2 notebook-only, Steps 3-5 run here) above the call sites, so the entry point is legible without cross-referencing three other files.
- **`src/telco_churn/features/select.py`** — `run_selection_cv`'s per-fold return grows two fields, `importance_table` and `group_importance`, both aggregated (mean) across all 100 folds from the fold's own already-fitted `PermutationImportanceSelector` (`_fit_and_score_selection_fold` now returns the fold's `permutation_importance_table_`/`group_importance_` instead of discarding them after reading `survivors_`) — no extra fit, since every fold already computes these while deciding that fold's survivors. `feature_selection.py` consumes these directly instead of calling a second, separately-fit `mint_committed_list`: a single all-dev fit is a different computation from 100 refits on different folds/seeds and can disagree with the fold-survival picture (the "minting defect" `PROJECT_PLAN.md` documents — a feature can survive one fluky all-dev fit despite low fold stability, or the reverse), so `recommended_committed_features`, the logged `permutation_importance_table`'s `survived` column, and the SHAP audit's `committed` flag are now all derived from the one 100-fold-averaged signal (`stability >= 0.5`, majority vote) rather than two independently-drawn fits that could quietly disagree. Two stale docstring references to `train.run_candidate_step` (`run_selection_cv`) fixed to name the actual full-feature candidate (`feature_selection.py`'s own `cv_score_candidate` call — a different candidate answering a different question). `tests/unit/test_select.py` gains 2 tests (37 total); `mint_committed_list` itself is unchanged and still used as-is by `feature_freeze.py`'s per-cycle diagnostic, where a single fast fit is the right cost/robustness tradeoff for a non-decision-making snapshot.
- **`notebooks/03b-feature-selection.ipynb`** — §3's "On-demand ablation review" rewritten from inlined orchestration code to a thin call to `run_feature_selection_step`, the same on-demand-trigger pattern `03a-model-selection.ipynb` already uses for the model-family decision. §1's dev-data load switched from a hand-built path reading the retired `telco_churn_processed.csv` (stale since the Phase 8 CSV→Parquet rename — would have raised `FileNotFoundError`) to `features/accessor.py::load_features()`.
- **`src/telco_churn/models/train/selection_review.py` → `feature_selection.py`** (renamed, same day) — the old name undersold what the module does: it's the mechanism that decides `COMMITTED_FEATURES`, not a passive review of a decision made elsewhere. `run_selection_review_step` → `run_feature_selection_step` accordingly. The `selection_review` run-name prefix/stage tag and the `telco-churn-feature-selection-review` experiment name are unchanged — those describe the run's place in a periodic review *cadence*, a different axis from what the Python module/function is named.
- **`src/telco_churn/models/train/common.py`** — new `_plot_shap_audit(shap_audit, high_shap_dropouts, title)`, mirroring `_plot_bootstrap_delta`'s existing split (pure computation stays in `features/select.py`; plotting lives here, shared by callers). Horizontal bar chart of mean(|SHAP|) per feature — green = committed, red = not committed, gold = a `flag_high_shap_dropouts`-flagged feature (a dropped feature ranking above a committed one), so the exact check that function computes is visible on the chart, not just inferable by counting bar positions against an implicit top-N cutoff. Called by both `feature_freeze.py` and `feature_selection.py`, logged as `figures/shap_importance_audit.png` alongside the existing CSV (never chart-only — `CLAUDE.md`: "log the array, not only the summary or the chart derived from it"). `_plot_bootstrap_delta`'s own docstring, stale since the model-family retirement (still said "feature_freeze.py's full-vs-reduced" — that logic moved to `feature_selection.py`), corrected in the same pass.
- **`src/telco_churn/models/train/feature_freeze.py`, `feature_selection.py`** — MLflow artifacts moved off their blanket `selection/`/`review/` path prefix onto the run root, with only the two figures now under `figures/` — matching `CLAUDE.md`'s already-stated rule for `calibrate.py`/`threshold.py`/`evaluate.py` ("sub-paths only where they separate different kinds of output — `figures/`, `slices/`. Do not prefix them `evaluation/`: the run already is the evaluation"), which `feature_freeze.py`/`feature_selection.py` hadn't been brought in line with. `notebooks/03b-feature-selection.ipynb`'s three artifact-download calls updated to match (§1's whole-run download drops its `artifact_path="selection"` scoping; §2's `group_importance.json` and §3's new `shap_importance_audit.png` drop their prefixes).
- **`notebooks/03b-feature-selection.ipynb`** — §3 gains a rendered chart: downloads and displays `run_feature_selection_step`'s own `figures/shap_importance_audit.png` (scored against `recommended_committed_features`, not whatever `COMMITTED_FEATURES` is today) rather than leaving the ablation's SHAP evidence as text-only numbers. Deliberately does not touch §4's existing chart, which cross-references SHAP rank against the *permutation-importance* `survived` column — a different comparison (two independent signals agreeing or not), not a duplicate.
- **`notebooks/03b-feature-selection.ipynb`** — restructured so the on-demand ablation review is the first analytical section (§2), not the third — mirroring `03a-model-selection.ipynb`'s decide-first structure, but *not* copying its always-run pattern: `03a`'s review has no cheap fallback (there's no routinely-logged family-comparison artifact to load instead), while `03b`'s does (`run_feature_audit_step` already logs a permutation-importance/SHAP audit every training cycle). Unconditionally re-running the ~85,000-scoring-pass review on every notebook open would reintroduce the exact cost the Phase 8 retirement removed from the automated pipeline, just relocated into notebook habit. New `RUN_ON_DEMAND_REVIEW` flag (default `False`) gates the review cell; the cheap per-cycle diagnostic (old §1/§2/§4, renumbered §3/§4/§6) always renders regardless. `run_feature_selection_step`'s SHAP-audit display cell is conditioned on `review is not None`. Re-executed end-to-end via `jupyter nbconvert --execute` with the flag at its default — 0 errors.
- **`src/telco_churn/features/build.py`** — `__main__`'s output path now resolves via `features/accessor.py::features_path()` instead of hand-building `get_project_root() / cfg.paths.processed_data`, closing the last non-canonical read of that config key ahead of the new architecture guard (see Added, `test_architecture.py`).
- **`pyproject.toml`** — ruff `select` gains `TRY`, `LOG`, `PTH`. Ignored `TRY003` (raise-vanilla-args) and `PTH123` (builtin-open) as bulk pre-existing patterns not worth a same-day mass refactor; the 5 `TRY301` (raise-within-try) sites were fixed directly.
- **`src/telco_churn/models/calibrate.py`, `models/error_analysis.py`, `models/evaluate.py`, `models/register.py`, `models/threshold.py`** — each `__main__` block's required-CLI-arg validation (`calibration.run_id`, `*.model_version`) moved into a small `_require_*(cfg) -> str` inner function instead of raising directly inside the surrounding `try`, per the new `TRY301` lint rule (see `pyproject.toml` above).
- **`src/telco_churn/models/train/feature_freeze.py` → `feature_audit.py`** (renamed, plus `tests/unit/test_train_feature_freeze.py` → `test_train_feature_audit.py`) — the module name no longer matched what it does: it doesn't freeze or decide the committed feature set (a hand-maintained `features/schema.py::COMMITTED_FEATURES` constant, decided by `feature_selection.py`'s on-demand review), it only audits it every training cycle. Companion to the earlier `run_selection_step` → `run_feature_audit_step` function rename documented above, which the module name had been left out of step with. Propagated through `models/train/__init__.py`, `models/calibrate.py`, `features/select.py`, `feature_selection.py` and `common.py`'s docstrings, and prose references in `ANALYSIS.md`, `PROJECT_PLAN.md`, `PHASE_8_PREREQ_TASKS.md`, `docs/architecture.md`, `configs/training/feature_selection.yaml`, the `Makefile`, and `notebooks/03b-feature-selection.ipynb`. Its module and `_RUN_DESCRIPTION` docstrings also corrected to describe the audit-only behavior accurately (previously read "freeze the committed feature set," and cited "the Step 3 ablation" for the notebook review — colliding with this module's own Step-3 identity in the automated pipeline).
- **PR B1 — promotion-decision integrity.** `src/telco_churn/models/gate.py::decide_promotion` no longer returns a `review` field (`{regime, gate, criteria}` only — a pure function has no business emitting a field about a process it knows nothing about). `record_review`'s signature changes from `(decision, verdict, notes, approver, reviewed_at, direction_sanity_check_fired)` to `(promotion_review, decision, verdict, notes, approver, reviewed_at)`: it now appends one entry to `promotion_review.json`'s `entries` list instead of stamping fields onto a copy of `decision`, and drops `direction_sanity_check_fired` entirely — that was a machine fact (error_analysis.py's V3 outcome) duplicated into a human document that already has its own source for it (`error_analysis_run_id`, resolved via tag). `promotion_decision.json` (evaluate.py's own DVC-track-eligible out) is now single-authored and never mutated after evaluate.py writes it — the review split closes a real defect: notebook 05's closing cell used to write the human verdict back into the same file evaluate.py produces, so a subsequent `dvc repro`/`dvc checkout` would regenerate it and silently discard the approval.
- **`src/telco_churn/models/calibrate.py`** — no longer performs any registry write, and no longer calls `register.register_challenger` at all (an earlier pass on this branch had it call the extracted function via a function-local import; that call site is now removed — see the call-site-decoupling entry below). `run_calibration_step` fits and logs the calibrated pipeline (a pure, deterministic file transform), tags the run's `calibrated_model_id`/`calibrated_model_uri`, and writes `reports/calibrate_receipt.json` (`{run_id, logged_model_id, model_uri}` — never `model_version`). `_tag_new_version_pending`/`_verify_reload_parity`/`_point_challenger_alias` moved to `register.py` verbatim as part of the new function; `_REGISTRY_DESCRIPTION`/`_PENDING_VERSION_DESCRIPTION` constants moved with them.
- **Call-site decoupling (2026-08-10): `register.register_challenger` runs as its own CLI step, never called from `calibrate.py`.** `calibrate.py::run_calibration_step` no longer imports or calls `register_challenger` — the two now communicate only through `reports/calibrate_receipt.json` and the `calibrated_model_id`/`calibrated_model_uri` run tags `calibrate.py` sets, exactly like every other stage boundary in this pipeline. `register_challenger`'s signature changed to `(cfg, run_id, model_uri, logged_model_id)` and its reload-parity check now loads `golden_predictions.json` off the run instead of comparing against an in-memory array, since a second CLI process cannot receive one. `utils/mlflow.py::resolve_model_identifier`'s "neither given" default path changed to match: it resolves `run_id` from the receipt and falls through into the same registry lookup (`resolve_model_version_from_run_id`) the explicit-`run_id` path already used, rather than trusting a `model_version` the receipt no longer carries — this is `threshold.py`/`evaluate.py`/`error_analysis.py`'s default resolution "switching from `model_version` to `run_id`," concretely. New `utils/mlflow.py::resolve_calibrated_run` resolves `(run_id, model_uri, logged_model_id)` for the mint CLI, from an explicit `register.run_id` override (reading the run tags directly) or the receipt. `tests/unit/test_architecture.py`'s `_MAIN_MODULE_IMPORT_ALLOWANCES` drops the now-unused `models/register.py: {models/calibrate.py}` row — the two modules no longer cross-import at all. See `PROJECT_PLAN.md`'s Phase 8 Prerequisites, "Extract `calibrate.py`'s registration call," for the full before/after.
- **`src/telco_churn/models/evaluate.py`, `models/error_analysis.py`** — stop writing any model-version tag. The four gate-criteria tags + `eval_run_id` (`evaluate.py`) and `error_analysis_run_id` (`error_analysis.py`) are now written by `register.py` itself — the module that already mints the version and is now the sole writer of every registry-mutating call in the cycle — resolved from the tag if a prior invocation already wrote it, else from `evaluate.py`'s/`error_analysis.py`'s new `reports/eval_receipt.json`/`reports/error_analysis_receipt.json` on the version's first registration pass. Fixes a real ownership gap: after B1 moves minting downstream of human review, `evaluate.py`/`error_analysis.py` run *before* a registry version exists to tag.
- **`src/telco_churn/models/register.py`** — `_check_review_approval` replaced by reading `promotion_review.json`'s `entries[-1]` off the eval run (never `decision["review"]`, which no longer exists). A human `rejected` verdict now routes through `_tag_rejected` — the same path the other three reject sites (`gate_fail`, `smoke_check_failed`, `post_flip_parity_failed`) already use — fixing a live defect where a deliberate human rejection was left at `promotion_status: pending`, indistinguishable from a crash artifact and eligible for the Phase 14 pending-reaper. `model_card.json`'s `human_review` section is now sourced from the resolved review entry (gains an `approver` field) instead of `decision`.
- **`notebooks/05-evaluation-and-error-analysis.ipynb`** — closing cell rewritten for the new `record_review` signature: fetches `promotion_review.json` from the eval run if one already exists (so a second review appends rather than overwrites), calls `record_review`, and re-logs the result as `promotion_review.json` — never rewrites `promotion_decision.json`. Drops the `direction_sanity_check_fired` variable/print (superseded — traceable via `error_analysis_run_id` instead).
- **`CLAUDE.md`** — "Phase 6 `calibrate.py` performs the single registration" rewritten to describe `register.py` as the sole entry point for minting/tagging/pointing/flipping; the tag-ownership rule corrected to name `register.py` as the writer of the four gate-criteria tags, on every version it processes (not only promoted ones, per the registry-legibility rationale already stated); `model_card.json`'s `error_analysis_run_id` provenance line corrected. `PROJECT_PLAN.md`'s Phase 7 Verification line updated to describe the receipt/tag/review-CLI flow instead of a `review: pending` field this PR removes.
- **PR C — stage-entry-point extraction (rewiring).** `expected_calibration_error`/`murphy_decomposition` (now in `calibration_metrics.py`) drop their `cfg: DictConfig` parameter in favour of explicit `n_bins: int, strategy: str` — callers (`calibrate.py::select_calibration_method`, `evaluate.py::sealed_test_calibration_report`, `diagnostics.py::sliced_calibration`) resolve `cfg.calibration.ece_n_bins`/`ece_strategy` themselves before calling in. `calibrate.py`, `threshold.py`, `evaluate.py`, `error_analysis.py`, `register.py`, `economics.py`, `diagnostics.py`, `utils/mlflow.py` all updated to import from the five new modules/`gate.py` instead of defining these symbols locally or importing them from each other; `register.py` now imports nothing from `calibrate.py`, `threshold.py`, or `evaluate.py` at all (the item the new architecture guard was written to hold permanently, not just check once).
- **`CLAUDE.md`** — closes out the Phase 8 prereqs' remaining "Doc corrections" backlog now that A/B0/B1/B2/C are all complete. "Test set touched once" restated around the guards that actually enforce it (`test_only_evaluate_binds_the_test_partition`/`test_test_ids_is_never_called_outside_evaluate`) instead of the previously-false "`X_test`/`y_test` imported in exactly one place" wording (`error_analysis.py` carries local variables of those names, sourced from `evaluate.py`'s stamped output, not a second import of the sealed split). "— enforced by `tests/unit/test_architecture.py`" pointers added to every remaining rule with a guard (alias-resolution, `__all__`, `exc_info=True`, random-state, the subprocess-test requirement). New Code Style rule documenting PR C's `test_no_module_imports_from_a_dunder_main_bearing_module` guard, which had no corresponding prose at all before this pass. `__all__`'s "public module" scope defined (`__main__.py` entry points and non-re-exporting `__init__.py` placeholders are exempt — `serving/`, `ui/`, `monitoring/`, `utils/__init__.py`, root `__init__.py`). Random-state rule clarified: every stage reads its own `random_state`/`random_seed` config key rather than one shared/interpolated value, and why (DVC's per-stage `params:` hashing).
- **`src/telco_churn/models/evaluate.py`** — new `resolve_evaluation_champion`, closing the last open Phase 8 DVC prerequisite (`PROJECT_PLAN.md`'s undeclared champion-alias dependency note). `configs/evaluate/default.yaml` gains `champion_version` (default `null`): an explicit override (a version number, or the literal `"none"` to pin the cold-start regime) is read verbatim and never touches the `champion` registry alias — the alias is externally-mutable state a DVC `cmd` string cannot declare as a dep. Omitting the override falls back to a live alias read, unchanged from today's behaviour, for interactive/notebook use. `_compute_promotion_decision` now calls this resolver instead of `resolve_champion_version` directly. `tests/unit/test_evaluate.py` (+3 tests), `tests/integration/test_evaluate_subprocess.py` (+1 test, the explicit-override path).
- **`src/telco_churn/models/evaluate.py`** — `load_incumbent_proba` no longer loads the champion's fitted pipeline and re-scores it live; it reads the champion's own `test_predictions.parquet` off its own eval run instead (resolved via `resolve_incumbent_summary`'s now-returned `eval_run_id`, so both functions share one registry lookup instead of two) and reindexes it onto the candidate's sealed-test row order by `customerid`. Removes a second per-cycle model deserialization/inference pass and a second source of cross-environment nondeterminism in the comparative gate (pinning *which* version is champion, `resolve_evaluation_champion` above, doesn't by itself freeze *how it's scored*). Raises loudly, naming the champion version and its eval run, on a customerid-set mismatch (the canonical split moved since the champion was last evaluated) or a label mismatch for a shared customerid. `resolve_incumbent_summary`'s returned dict gains `eval_run_id`. `tests/unit/test_evaluate.py` (+3 tests for the new alignment/error paths).
- **`tests/integration/test_register_subprocess.py`** — new `test_evaluate_cli_comparative_regime_reads_champion_historical_predictions`, closing a real coverage gap: no test previously ran `evaluate.py`'s comparative regime end-to-end against a real, live-registered champion (`test_evaluate_subprocess.py`'s throwaway registry is cold-start-only by design). Promotes `reviewed_model`'s candidate to champion, mints and threshold-screens a second independent candidate (new `_mint_and_threshold_second_candidate` helper), then runs `evaluate.py`'s real CLI with an explicit `evaluate.champion_version` override and asserts `promotion_decision.json`'s `regime == "comparative"` and `metrics.json`'s `incumbent_summary.eval_run_id` matches the champion's own eval run — the property `load_incumbent_proba`'s rework above depends on.
- **`src/telco_churn/models/evaluate.py`** — `load_incumbent_proba` gains a `champion_data_content_hash` parameter, checked against the current processed-features file's own `features_sha256()` before anything is downloaded: a customerid/label match alone can't rule out the feature pipeline having changed under an unchanged customer set (e.g. a new engineered column added to every row), so the champion's historical predictions are only trusted when they were computed against the same feature file. Every `evaluate.py` run now tags its own eval run with `data_content_hash` (`_log_evaluation_run`, alongside the existing `costs_config_hash` tag); `resolve_incumbent_summary`'s returned dict and missing-tag check both gain `data_content_hash`. `tests/unit/test_evaluate.py` (+2 tests: the mismatch path, the missing-tag path).

### Removed
- **`src/telco_churn/models/train/feature_freeze.py`** — `feature_freeze.py`'s `decision == "reduced"` branch and `_plot_selection_figures`' full-vs-reduced PR-curve/bootstrap-delta figures — the automated pipeline no longer computes or branches on this comparison. `run_selection_cv`/`reduced_set_bootstrap_test` themselves are **not** removed from `src/`: they remain in `features/select.py`, fully tested, called only from the notebook's on-demand review (see Added/Changed above) — an earlier same-day pass deleted them outright before this decoupled design replaced it.
- **`configs/config.yaml`** — `training_setup.cv_n_jobs`, dead config once `run_selection_cv`'s automated-pipeline call site was removed. The notebook's on-demand review passes `n_jobs` as a literal instead, since nothing Hydra-configured calls it anymore.
- **`configs/training/feature_selection.yaml`** — `bootstrap_n_samples` (ablation-only; replaced by `select.py::ABLATION_N_BOOTSTRAP`, since only the notebook uses it now and nothing automated needs a CLI-overridable value).
- **`configs/config.yaml`** — `training_setup.cv_folds`, `cv_repeats`, `bootstrap_n_samples`, `segment_bootstrap_n_samples` — dead config once `run_candidate_step`/`run_comparison_step`'s automated-pipeline call site was removed (see Removed, `__main__.py`). Replaced by `_FAMILY_REVIEW_CV_FOLDS`/`_FAMILY_REVIEW_CV_REPEATS` (`candidates.py`) and `_FAMILY_REVIEW_N_BOOTSTRAP`/`_FAMILY_REVIEW_SEGMENT_N_BOOTSTRAP` (`comparison.py`), same pattern as `configs/training/feature_selection.yaml`'s `bootstrap_n_samples` retirement above. `training_setup.delta_threshold`/`fixed_recall_thresholds` stay — both are also read by `calibrate.py`/`evaluate.py`.

### Fixed
- **`tests/integration/test_train_subprocess.py`** — `test_train_main_cli_exits_zero` never overrode `paths.reports`, so `run_model_logging_step`'s new `write_train_receipt` call (see Added above) resolved against `config.yaml`'s real `"reports"` default and leaked a `reports/train_receipt.json` into the tracked working directory on every run. Now overrides `paths.reports` to a `tmp_path`-scoped directory, same as the other four subprocess suites, and asserts the receipt's contents from there.
- **`src/telco_churn/models/evaluate.py`, `models/register.py`** — the four `champion`-alias/registry lookups (`resolve_champion_version`, `rollback_champion`'s current-champion read, `_append_promotion_event`, `champion_history`) each caught the entire `MlflowException` hierarchy and treated any failure as "not found," so a transient MLflow/network/auth error during evaluation could silently switch the gate from the comparative regime to cold-start against a healthy incumbent. All four now check `error_code`, re-raising anything that isn't a genuine not-found (`RESOURCE_DOES_NOT_EXIST` for a never-registered model, plus `INVALID_PARAMETER_VALUE` for a registered model with no `champion` alias set yet — the two real cold-start shapes MLflow's SqlAlchemy-backed registry reports).
- **`src/telco_churn/models/register.py`** — `rollback_champion` no longer raises when no version is tagged `promotion_status=promoted`; it now unsets the `champion` alias and records a `rolled_back` event with `version: None`. Previously the automated post-flip-parity abort path (`_flip_and_confirm`) hit this as a `RuntimeError` on a first-ever (cold-start) promotion, which skipped `_tag_rejected` and left `champion` pointing at a candidate that had just failed its own serving check.
- **`src/telco_churn/models/calibrate.py`** — `run_calibration_step` tagged a newly minted registry version `promotion_status=pending` only after `_verify_reload_parity` passed, so a parity failure left the version completely untagged — invisible to both the tag-based rollback rule and the Phase 14 pending-reaper, a permanent, unreapable orphan. `_register_challenger_version` split into `_tag_new_version_pending` (runs immediately after `log_model` returns, before the parity check) and `_point_challenger_alias` (unchanged, still gated on parity passing).
- **`src/telco_churn/models/register.py`** — `run_registration_step` set `promotion_status=promoted` only after `_build_and_log_drift_reference`/`_build_and_log_model_card` ran post-flip, so a failure in either left `champion` pointing at a version never tagged `promoted` — invisible to the tag-based rollback rule, eligible for the pending-reaper despite being the live champion, and stuck on any retry (`_reverify_incumbent` would find champion already moved off the recorded incumbent and refuse). New `_complete_promotion` wraps the drift-reference/model-card build and the promoted tag write as one unit; any failure rolls the alias back to the prior promoted champion via `rollback_champion` and re-raises, leaving `promotion_status` at `pending` so a retry stays valid.
- **`src/telco_churn/models/evaluate.py`** — `_compute_core_test_metrics` now emits a `logger.warning("cost_scenario_spread_too_narrow", ...)` when `sealed_test_business_impact`'s `parameter_spread_dominates_sampling` is `False`, mirroring `threshold.py`'s `threshold_argmax_disagreement` pattern. Previously this diagnostic was only visible by hand-reading the JSON artifact, which Phase 10's unattended weekly retrain never does.
- **`src/telco_churn/models/error_analysis.py`** — `_log_error_analysis_run` capped per-feature `shap_importance_*` metrics to the same `top_k_shap_features` config value already gating V3's dependence plots, instead of logging one metric per feature in the (one-hot-expanded) feature space. Unbounded per-feature metric keys are the same "hundreds of metric keys" clutter problem `CLAUDE.md`'s fairness-metrics rule exists to avoid; the full ranking remains available in `shap_values.parquet`/`error_analysis.json`.
- **`src/telco_churn/models/register.py`** — `check_environment_parity` now emits a `logger.warning("environment_parity_pin_not_found", ...)` for any package whose logged requirement line doesn't match the expected exact `package==version` pin, instead of silently dropping it from both the matched and mismatched sets. A package that can't be parsed (extras syntax, a VCS install, a naming mismatch) was previously excluded from the drift check with no signal, giving false confidence that the environment was fully verified.
- **`Makefile`** — `test-models` target was stale, predating PR B1/B2/C's new test files: `test_review.py`, `test_calibration_metrics.py`, `test_artifacts.py`, `test_policy_config.py`, and `test_shap_values.py` were silently excluded from the scoped run and its coverage figure. Added all five to the target's file list; `test_mlflow.py` deliberately left out since it covers `utils/mlflow.py`, outside this target's `--cov=src/telco_churn/models` scope. Found while working through the Phase 8 prereqs' own Verification checklist — `make test-models` now passes 527 tests at 96.59% coverage on `models/`, up from a stale run that never exercised roughly a sixth of the package's test files.
- **`configs/config.yaml`** — `paths.processed_data`'s comment carried a stale sequencing warning ("that helper must land in the same change as this line, or...") about `activate_config()` — written before that helper existed, describing a landing risk that resolved successfully in PR A and has been true ever since. Trimmed to keep only the still-live rationale (the DVC `params:`-hashing constraint on why this key stays a plain literal).
- **`notebooks/05-evaluation-and-error-analysis.ipynb`** — artifact/figure loading switched from reading the local `reports/`/`reports/figures/` mirror directly to resolving through MLflow by run id (`reports/eval_receipt.json`/`reports/error_analysis_receipt.json` as the bootstrap pointers, then `mlflow.artifacts.load_dict`/`download_artifacts`), matching `04-calibration-and-threshold.ipynb`'s existing convention. Caught mid-fix: the local mirror was stale relative to current code (missing the `direction_sanity_check_test` key from the earlier rename), which the local-path read had been masking — an MLflow-run read surfaces that kind of schema drift immediately instead. Re-ran `evaluate.py`/`error_analysis.py` to regenerate both receipts against current code and re-executed the notebook end-to-end, 0 errors.
- **Phases 1-7 QA pass (13 items)**:
  - `data/schema.py`/`features/build.py` — `CleanedSchema`/`validate_clean()` now runs automatically in `build.py`'s `__main__`, previously reachable only from a manual notebook cell. Narrowed to `RawSchema`'s own columns before validating — checking the full post-join feature dataframe tripped `CleanedSchema`'s `strict=True` on `charge_per_service`.
  - `models/train/tuning.py`/`configs/tuning/optuna.yaml` — tried coupling `max_depth`'s sampled low to `num_leaves.high` (`ceil(log2(num_leaves.high))`) so no trial could draw a `max_depth` too shallow to reach its own `num_leaves`; reverted after a re-run showed it excluded the shallow-tree region (`max_depth=4`, `num_leaves=6`) the 1-SE rule actually preferred and lowered both the raw-best and 1-SE CV PR-AUC — `max_depth` and `num_leaves` are independent LightGBM regularizers, and it's valid for either to bind first.
  - `models/economics.py`/`models/evaluate.py`/`models/register.py` — new `capacity_budget_check` compares each cost scenario's contact count/spend against `configs/costs.yaml`'s `contact_capacity`/`campaign_budget`, logging a warning and a signed excess metric per scenario (diagnostic only, never gating). Surfaced in `economics.json`, notebook 05, and `model_card.json`.
  - `data/checks.py` — documented why `churn` stays in `_NULL_CHECKED_COLS` despite double-reporting with Gate 3 on a null label.
  - `data/ingest.py` — corrected the post-merge row-count-mismatch guard's comment to describe the actual failure path (`IntegrityError` out of `_merge_from_staging`, not a silent per-row skip).
  - `features/generate.py`/`notebooks/02a-feature-discovery.ipynb` — extracted `charge_per_service`'s formula (previously hand-duplicated in the notebook) into `compute_service_count`/`compute_charge_per_service`, with a new parity test asserting it matches `sql/features/charge_per_service.sql` row-for-row.
  - `features/accessor.py` — corrected `features_sha256()`'s stale "not yet used" docstring; it's called by every `models/train/*` module and `evaluate.py`.
  - `notebooks/01-eda.ipynb` — confirmed (no code change) that the headline significance tables use `compute_significance_screen`'s pooled, BH-corrected p-values, never the standalone uncorrected functions.
  - `sql/schema/002_create_optuna_schema.sql` (new)/`models/train/tuning.py` — Optuna's isolated Postgres schema is now created from a versioned DDL file, matching the `sql/schema/*.sql` convention `customers_raw` follows, instead of an inline Python string.
  - `models/train/comparison.py` — stated the LGBM tie-break rationale directly at the call site instead of only in `ANALYSIS.md`.
  - `tests/integration/test_register_subprocess.py` — renamed a test whose name contradicted its own assertion (`exits_one_on_gate_fail` → `exits_zero_and_rejects_on_gate_fail`; a clean gate-fail rejection exits 0).
  - `notebooks/02a-feature-discovery.ipynb`/`features/generate.py` — factored the 9-lap discovery notebook's repeated 4-screen gate block (serving → redundancy → OOF fit → PR-AUC → importance → decision) into `generate.py::run_lap()`/`LapEvaluation`.
  - `utils/mlflow.py` — the training-cycle experiment's `mlflow.note.content` was missing the threshold-derivation stage and didn't mention it's expandable in the UI; both fixed.
- **`.gitignore`, `reports/{ingest,validation,train,calibrate,eval,error_analysis}_receipt.json`** — the six `cache: false` per-cycle receipts are gitignored instead of git-committed, closing the fresh-Postgres/MLflow blind spot Phase 8's fresh-clone check surfaced: a git-committed receipt hash-matched `dvc.lock` on a brand-new clone even though the fresh backend had never actually run, so `dvc repro` skipped `ingest`/`train`/`calibrate` against infra with no data or matching run. Gitignored, DVC's own missing-output check now reruns the whole chain automatically on a genuinely fresh environment — no manual `--force` workaround needed. `CLAUDE.md`'s Data Handling section, `PROJECT_PLAN.md`'s Phase 10 open question, `CONTRIBUTING.md`, and `README.md`'s quick start updated accordingly; `PHASE_8_TASKS.md`'s verification finding kept as a historical record with an addendum.

---

## [0.7.6] - 2026-08-02 — Code Quality Pass: Orchestrator Decomposition & Test/CI Hardening

*Every Phase 5–7 `run_*_step` function had grown into a single 250–540 line block mixing load/compute/plot/MLflow-log concerns; this pass splits each into named, single-purpose helper functions with no behavior change, dedupes a handful of copy-pasted blocks, and separately lands leftover CI/warning-suppression fixes and test-coverage backfill from Phase 7 QA. Full suite: 815 passed, 51 skipped, 95.71% coverage.*

### Changed
- **`models/error_analysis.py`, `evaluate.py`, `register.py`, `calibrate.py`, `threshold.py`** — each `run_*_step` orchestrator decomposed into named stage helpers; bodies shrank from 410–474 lines to 63–137.
- **`models/train/tuning.py`, `feature_freeze.py`, `log_model.py`, `comparison.py`** — same decomposition; all four now call `utils.mlflow.ensure_experiment_metadata` instead of hand-rolling MLflow setup.
- **`models/explain.py`, `features/select.py`** — duplicated signed-direction and preprocessor/selector construction logic extracted to shared helpers.
- **`models/error_analysis.py`** — dropped its private reimplementation of `utils.mlflow.resolve_logged_model_id`.

### Fixed
- **`utils/mlflow.py`** — silences three confirmed-benign MLflow warnings at `resolve_tracking_uri`.
- **`features/select.py`, `models/error_analysis.py`, `models/explain.py`, `features/preprocessing.py`** — migrated off shap's deprecated `TreeExplainer.shap_values()`.
- **`models/calibrate.py`, `models/evaluate.py`** — `calibration_slope`'s bootstrap fit avoids an sklearn 1.8 migration-shim warning; sealed-test bootstrap CIs suppress a benign all-NaN warning.
- **`.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `pyproject.toml`** — mypy hook dependencies expanded to match `uv.lock`; unit tests run under `pytest-xdist`; integration tests split into a scheduled/on-demand job.
- **`tests/`** — fixtures scoped to module level for parallel-safe execution; backfilled coverage for `utils/stats.py`/`utils/logging.py`.

---

## [0.7.5] - 2026-07-30 — Docs Accuracy Pass: Phase 7 Status Sync

*README.md and PROJECT_PLAN.md still described Phase 7b (registry promotion) as in progress or not started after 0.7.3 had already closed it — this pass syncs both against the actual shipped code, extends the architecture doc to cover a workflow it was missing, and clears stray root-level screenshots.*

### Changed
- **`README.md`** — Phase 7b status corrected to done; pipeline diagram's registration step updated to reflect the drift-baseline/model-card work it now includes; Project Structure listing gains `models/register.py`, `models/drift_reference.py`, `configs/register/`; the `docs/architecture.md` pointer now names all four diagrams it links to (System Architecture, ML Workflow, Data Flow, MLflow Layout — was only naming the first two); Quick Start gains the previously-undocumented `make split` and `make register` steps; test/coverage badges refreshed to 771 passed / 51 skipped, 95.31% coverage.
- **`Makefile`**, **`CONTRIBUTING.md`** — added the missing `register` target (Phase 7b had no `make` entry point despite being fully built); `test-models` now covers `test_register.py`/`test_drift_reference.py`; both docs' workflow lists gain `make split`, previously undocumented despite being a hard prerequisite for `make train`.
- **`PROJECT_PLAN.md`** — Phase 7 checklist row corrected from "Not started" to done.
- **`docs/architecture.md`** — new **Emergency Rollback** diagram covering `register.py::rollback_champion`'s two triggers (automated post-flip parity failure vs. manual invocation) and `champion_history`'s append-only promotion log; this flow was previously only referenced in one prose sentence.
- **`ANALYSIS.md` §9** — cross-checked every figure against `notebooks/_archive/EDA-original.ipynb` and this project's own artifacts (`error_analysis.json`, `metrics.json`, `calibration_summary.json`, `reports/feature_discovery/provenance.json`); one item cited an archive figure (~50% joint false-positive rate) that was never actually measured there — corrected to the archive's real single-axis rates (54.0% fiber optic, 60.1% month-to-month). Section regrouped into four subsections (Model Behaviour & Blind Spots, Business & Economic Assumptions, Methodology & Engineering Trade-offs, Data & Production-Readiness Constraints) and renumbered; fixed a pre-existing off-by-one in the section's own intro note (pointed at item #11 for a claim that's actually item #12/#13's). Every `§9 #N` citation in `PROJECT_PLAN.md` (8 instances) updated to match.
- **`ANALYSIS.md` §10** — added a short cross-reference to §4's existing RFECV/embedded-method deferral note, rather than restating it.

### Removed
- Nine untracked root-level screenshot PNGs (`landing_page.png`, `model_registry.png`, `logged_models.png`, etc.) that were never moved into `reports/figures/`.

---

## [0.7.4] - 2026-07-30 — Phase 7 Completion: Pre-Seal Dev-OOF Screen, Full-Refit Descoping & First Champion Promotion

*Closes the gap between what `CLAUDE.md`/0.7.3 already documented as `register.py`'s contract and what the rest of the pipeline actually wrote: `evaluate.py`/`error_analysis.py` now tag the model version `eval_run_id`/`error_analysis_run_id` `register.py` resolves by. Adds Phase 6's pre-seal screen (`threshold.py` re-runs `ANALYSIS.md` §0's V1/V2/V2b dev-OOF diagnostics and the calibration-slope band check before the sealed test set is ever touched), retires the full-data-refit design (`refit.py` was never implemented and is no longer planned — `register.py` promotes the exact artifact `evaluate.py` scored), and runs the project's first real promotion: `telco-churn-pipeline` v1 → `champion`, 2026-07-30 13:00:37 UTC.*

### Added
- **`src/telco_churn/models/threshold.py`** — `run_threshold_step`'s new last step: screens the calibration slope against `ANALYSIS.md` §0's `[0.80, 1.25]` band (`gate.py::slope_passes`) and computes V1/V2/V2b on the dev-OOF vector, raising `RuntimeError` before the one-time sealed-test evaluation runs on a badly-calibrated candidate. Writes `dev_oof_predictions.parquet`/`dev_oof_diagnostics.json`.
- **`src/telco_churn/models/diagnostics.py`** — the V1/V2/V2b computation moved here from `evaluate.py` so `threshold.py`'s dev-OOF screen and `evaluate.py`'s sealed-test slices share one implementation.
- **`src/telco_churn/utils/mlflow.py`** — `resolve_model_run_id`, `resolve_logged_model_id`, `load_model_promotion_bars`, `ensure_experiment_metadata`, and description-setting helpers for MLflow experiment/run/registry/model overview pages.
- **`tests/unit/test_diagnostics.py`** (+274), **`test_threshold.py`** (+246), **`test_calibrate.py`** (+81); **`tests/integration/test_threshold_subprocess.py`** (+72), **`test_evaluate_subprocess.py`** (+49), **`test_error_analysis_subprocess.py`** (+28) — dev-OOF screen and `eval_run_id`/`error_analysis_run_id` tag coverage.

### Changed
- **`src/telco_churn/models/evaluate.py`** — no longer computes V1/V2/V2b itself; fetches `threshold.py`'s `dev_oof_diagnostics.json` by explicit `run_id`. Now tags the model version `eval_run_id`.
- **`src/telco_churn/models/error_analysis.py`** — reads `calibrate.py`'s dev-OOF vector via MLflow instead of a local file mirror; now tags the model version `error_analysis_run_id`.
- **`src/telco_churn/models/gate.py`** — `_slope_passes` made public (`slope_passes`); reused by `threshold.py`'s dev-OOF screen.
- **`src/telco_churn/models/calibrate.py`** — model-version tag `refit_scope` renamed `training_data_scope: dev`; artifacts now logged under a `calibration/` subpath instead of the run root.
- **`configs/config.yaml`, `configs/threshold/default.yaml`, `configs/calibration/default.yaml`** — `register: default` wired into config composition; `threshold` gains `n_bootstrap`; `calibration` gains `golden_n_rows`.
- **`ANALYSIS.md`** §0 rewritten in plainer language for the reviewer audience; §8 records the real promotion event (`telco-churn-pipeline` v1 → `champion`, 2026-07-30 13:00:37 UTC).
- **`CLAUDE.md`, `PROJECT_PLAN.md`** — every `refit.py`/`refit_scope` reference removed; Phase 6's checklist gains the pre-seal dev-OOF screen.

### Removed
- **The full-data-refit design (`refit.py`)** — never implemented, descoped from Phase 7: `register.py` promotes the exact artifact `evaluate.py` scored, no second fit. `refit_scope: dev | full` tag retired with it.

---

## [0.7.3] - 2026-07-28 — Phase 7 Completion: Registry Promotion & Drift Reference

*Closes Phase 7: the promotion gate's verdict is now acted on — `register.py` flips the `champion` alias, builds the drift-monitoring baseline, and assembles the stakeholder-facing model card, completing the evaluate → gate → register cycle.*

### Added
- **`src/telco_churn/models/register.py`** — `run_registration_step`: reads the persisted gate verdict, a two-phase-commit serving smoke check (pre-flip by version URI, post-flip through the `champion` alias with automatic rollback on failure), writes `drift_reference.json`/`model_card.json`. `rollback_champion` selects the highest `promotion_status: promoted` version, never by version arithmetic.
- **`src/telco_churn/models/drift_reference.py`** — `build_reference`: pure builder for the champion's drift-monitoring baseline.
- **`configs/register/default.yaml`** — registry promotion step configuration.
- **`tests/unit/test_register.py`** (17), **`test_drift_reference.py`**, **`tests/integration/test_register_subprocess.py`** (3) — step-order, rollback, and environment-mismatch coverage.

### Changed
- **`src/telco_churn/models/calibrate.py`** — tags every newly minted version `promotion_status: pending` at mint time; logs `golden_predictions.json` (customerid-pinned dev rows + reference scores) as the independent reference `register.py`'s serving-parity check verifies against.

---

## [0.7.2] - 2026-07-27 — Repo Hygiene: Makefile, CONTRIBUTING, and Docs

*Tooling and documentation pass alongside Phase 7 — no modelling logic changed.*

### Added
- **`Makefile`** — self-documenting `help` target (default goal); fail-fast guards on `RUN_ID`/`MODEL_VERSION` for `calibrate`/`threshold`/`evaluate`/`error-analysis`; `clean`, `pre-commit`, `mlflow-ui` targets; idempotency guard on `data`.
- **`README.md`** — MIT license and Python-version badges, table of contents, links to `docs/architecture.md`, `CHANGELOG.md`, and `ANALYSIS.md` §9 Known Limitations, an Author section.
- **`LICENSE`** — MIT.

### Changed
- **`CONTRIBUTING.md`** — rewritten to defer to `CLAUDE.md` for governing conventions (branch/commit rules) instead of duplicating them; make-commands table replaced with a pointer to `make help` plus the common first-run workflow, so it can't go stale the way the old hand-maintained table did; corrected the `nbstripout`/`fix-notebook-outputs` hook descriptions.
- **`CLAUDE.md`** — `make train`'s Key Commands description corrected (was `dvc repro`; actually `python -m telco_churn.models.train` — DVC pipeline wrapping is still Phase 8).

---

## [0.7.1] - 2026-07-27 — Phase 7 QA: Cross-Notebook Consistency Audit

*A full crosscheck of every notebook's rendered narrative against its own output, and against `ANALYSIS.md`, surfaced one functional bug and several stale figures — none affecting the gate decision or shipped model.*

### Fixed
- **`src/telco_churn/models/evaluate.py`** — `promotion_decision_payload` never carried its own `eval_run_id`, so the notebook's human-review cell logged the reviewed verdict onto the evaluated model's *training* run instead of the `evaluation` run that actually holds it — silently leaving the real run's copy stuck at `review: pending` forever. Now stamped inside the run context before logging.
- **`notebooks/02a-feature-discovery.ipynb`** — seven `LapRecord` hypothesis/`eda_anchor` strings held stale pp/dollar figures from before a `make_split` ordering fix, contradicted by their own adjacent, freshly-computed cells; corrected, and `reports/feature_discovery/provenance.json`/`.md` regenerated.
- **`notebooks/03b-feature-selection.ipynb`** — a markdown cell's CI/Δ/fold-win-rate numbers didn't match the code output directly above it; corrected.
- **`notebooks/03a-model-selection.ipynb`**, **`ANALYSIS.md`** §4a/§4b — stale LightGBM-vs-LogReg training-time figures, and the §4b bootstrap CI/fold-win-rate cross-referenced in §5, corrected to match the actual re-run.
- All 9 non-archived notebooks re-executed end to end; fragmented stream outputs normalized via `scripts/fix_notebook_outputs.py`.

### Added
- **`tests/integration/test_evaluate_subprocess.py`** — asserts `promotion_decision.json`'s `eval_run_id` is distinct from the evaluated model's own `run_id`, regression-covering the fix above.

---

## [0.7.0] - 2026-07-27 — Phase 7: Sealed Test-Set Evaluation, Error Analysis & Human Review

*One-time evaluation of the champion candidate against the sealed test set, closing the loop `ANALYSIS.md` §0's promotion gate defines: PR-AUC-driven selection, three veto-only guardrails, and a pre-registered human review (V1–V3). Gate result: pass (cold start); human review: approved. Registry promotion (`register.py`) remains open, tracked in `PROJECT_PLAN.md`; a standalone full-data refit step was subsequently descoped from Phase 7 (see 0.7.3 — `register.py` promotes the already-evaluated candidate directly, no second fit).*

### Added
- **`src/telco_churn/models/evaluate.py`** — `run_evaluation_step`: resolves the model by explicit version (never by alias), scores the sealed test set once, computes ranking/classification/calibration/business-impact metrics plus per-slice robustness and fairness views, calls `gate.py::decide_promotion`, and logs a dedicated `evaluation` MLflow run.
- **`src/telco_churn/models/gate.py`** — `decide_promotion` (pure function implementing `ANALYSIS.md` §0's cold-start/comparative regimes) and `record_review` (stamps the human verdict onto the persisted gate decision).
- **`src/telco_churn/models/economics.py`** — expected-value scenarios, retention-rate/cost/LTV sensitivity (tornado diagram), and break-even heatmap over the three cost scenarios.
- **`src/telco_churn/models/explain.py`**, **`error_analysis.py`** — SHAP explainability (global importance, beeswarm, direction sanity check V3) and error diagnosis (cohort scan, near-miss vs. confident-failure split, value-weighted error analysis).
- **`src/telco_churn/models/plots.py`** — `pr_curve_points`, `roc_curve_points`, `decile_lift_table`, `classification_summary_points`.
- **`src/telco_churn/models/diagnostics.py`** — `segment_bootstrap_ci`, `segment_decision_rates` for the per-slice robustness/fairness checks (V1/V2/V2b).
- **`src/telco_churn/models/calibrate.py::murphy_decomposition`** — Brier = reliability − resolution + uncertainty, reused by `evaluate.py`'s calibration report.
- **`src/telco_churn/utils/stats.py`** — `paired_bootstrap_metric_ci`, `bootstrap_metric_ci` — row-resampling bootstrap for set-level metrics (PR-AUC) that have no per-row decomposition, distinct from the existing per-row `paired_bootstrap_ci`.
- **`configs/evaluate/default.yaml`**, **`configs/error_analysis/default.yaml`**, **`configs/model_promotion.yaml`** — Phase 7 step configuration.
- **`configs/costs.yaml`** — `contact_capacity`, `campaign_budget` — operational limits used by the business-impact scenarios and their sensitivity checks.
- **`notebooks/05-evaluation-and-error-analysis.ipynb`** — renders the gate criteria, ranking/calibration/business-impact detail, disaggregated robustness & fairness, error analysis, and SHAP explainability; closing cell records the human review verdict.
- Test suite: `test_evaluate.py`, `test_economics.py`, `test_explain.py`, `test_error_analysis.py`, `test_gate.py`, `test_plots.py`, plus two subprocess integration tests (`test_evaluate_subprocess.py`, `test_error_analysis_subprocess.py`). Suite now 740 passed / 47 skipped, 96.46% coverage.

### Changed
- **`ANALYSIS.md`** §7 rewritten with the real sealed-test result (previously an archived-notebook placeholder): PR-AUC 0.670, recall 0.698, BSS 0.301, calibration slope 0.992 — gate pass, human review approved. §0 tightened for readability (plain-language summary added, repeated calibration-guardrail rationale consolidated) without changing any rule.
- **`README.md`** — headline results replaced with the real sealed-test figures; Project Status, pipeline diagram, and Quick Start updated through Phase 7's evaluation step.
- **`PROJECT_PLAN.md`** — Phase 7 deliverable description expanded (V1/V2/V2b/V3 framework, notebook 05 description).

---

## [0.6.3] - 2026-07-18 — Phase 7 Prerequisites: Tree-Count Scaling & Dataset Lineage (Run 2)

*`n_estimators` was derived from Optuna's early-stopped median on each CV fold's carved-down training partition and applied unscaled to the larger final dev fit, systematically under-boosting every shipped model. Scaling it moves CV PR-AUC/BSS/ECE and the empirical threshold checks. A second, independent fix extends Run 1's dataset-lineage logging to the three step-level MLflow runs it missed.*

### Added
- `models/train/log_model.py` — `n_estimators` scaled by `n_final_fit / n_fold_fit` before the final dev fit, tree count 94 → **147** (`ANALYSIS.md` §4c).
- `models/train/common.py::_log_dev_input` — dataset-lineage `log_input` now fires from the `model_comparison`, `feature_selection`, and `tuning_study` runs.
- `models/calibrate.py::calibration_slope` — adds an analytic (Wald) CI cross-checking the bootstrap CI; surfaced an intercept miscalibration (`ANALYSIS.md` §9 #14).
- `configs/calibration/default.yaml` — `method` pinned from `auto` to **`sigmoid`**.

### Changed
- **Run 2 measurement** — dev CV PR-AUC (sigmoid) 0.6669 → 0.6684, BSS 0.3098 → 0.3111, ECE 0.0217 → 0.0222, base-scenario empirical threshold 0.4428 → 0.4150 (closed-form `t* = 0.3941` unchanged). Notebooks `03a`–`03c`/`04` re-executed.

---

## [0.6.2] - 2026-07-16 — Phase 7 Prerequisites: MLflow Lineage & Evidence-Persistence Fixes (Run 1)

*Four logging/lineage defects found while designing Phase 7, all in already-shipped Phase 5/6 code — none change a computed number. Verified via a hard reset and full rebuild: every existing number in `ANALYSIS.md` §4–§6 reproduced identically.*

### Added
- `configs/config.yaml`, `utils/mlflow.py::resolve_tracking_uri` — fallback now resolves to `sqlite:///mlflow.db`, not the bare `mlruns` file store (raises as of MLflow 3.14).
- `models/train/log_model.py`, `models/calibrate.py` — `logged_model_id` persisted to `training_manifest.json` and as a model-version tag.
- `models/train/candidates.py` — logged dataset `source` resolves through `features/accessor.py::features_path()` instead of a hardcoded path.
- `models/calibrate.py::calibration_slope` — Cox calibration slope + bootstrap CI, logged in `calibration_summary.json` alongside the dev-OOF probability vector, previously computed and discarded.
- `models/calibrate.py`, `models/threshold.py` — dev calibration/threshold metrics now logged as MLflow metrics, not only inside JSON artifacts.
- `models/threshold.py::expected_value_at_threshold`, `costs_config_hash` — new pure functions; `configs/policy/threshold.yaml` pinned by `costs_config_hash` instead of a model stamp.

### Changed
- **Run 1 reproducibility audit** — registry holds exactly one version; every family-comparison delta, frozen feature set, tuned hyperparameter, calibration diagnostic, and threshold value matched `ANALYSIS.md` exactly.

---

## [0.6.1] - 2026-07-14 — Phase 6 QA Pass: Test Isolation & Calibration-Selection Hardening

*A QA pass over Phase 6 found the calibrate/threshold unit tests were not hermetic (silently falling through to real project data), the `calibration.method='auto'` path had two untested branches, and `switch_decision` had three inconsistent shapes depending on which branch produced it. None change the currently-registered model.*

### Fixed
- **`tests/unit/test_calibrate.py`, `test_threshold.py`** — tests fell through to real disk instead of synthetic fixtures; added a `sandboxed_dev_features` fixture redirecting `load_dev_features` to the synthetic `dev_split` fixture. Verified hermetic.
- **`src/telco_churn/models/calibrate.py::select_calibration_method`** — in `'auto'` mode, sigmoid (the fallback whenever isotonic is disqualified) was never itself PR-AUC-gated, unlike pinned mode. Added the same gate check, raising `ValueError` rather than silently registering a degraded model.
- **`CLAUDE.md`** — Data Handling section corrected `features/io.py` → `features/accessor.py::load_features()`.

### Added
- **`tests/unit/test_accessor.py`** — dedicated tests for `features/accessor.py`: path override, schema-validation failures, hash stability. Now at 100% line coverage.
- **`tests/unit/test_calibrate.py`** — two tests closing previously-uncovered branches of `select_calibration_method`'s `'auto'` mode. `calibrate.py` now at 98% line coverage.

### Changed
- **`src/telco_churn/models/calibrate.py::select_calibration_method`** — `switch_decision` now always returns the same six keys regardless of which branch produced it, instead of a 2-key dict in one branch and 6-key in another.
- **`ANALYSIS.md`** §5 — notes the sigmoid-vs-isotonic switch's paired bootstrap CI resamples only 5 folds, a small effective space; not load-bearing for the shipped v1 decision.

---

## [0.6.0] - 2026-07-13 — Phase 6: Calibration & Cost-Sensitive Threshold

*Wraps the tuned LightGBM pipeline in sigmoid calibration (pooled Brier 0.1345, BSS 0.31) and derives the production threshold from a closed-form cost model (`t* = c/(r×LTV)`), not an empirical cost-matrix cutoff. Base scenario ships at `t* = 0.3941` (30.8% contact rate), agreeing with its empirical argmax-EV check. Performs the training cycle's single MLflow registration, pointing `challenger` at the calibrated pipeline.*

### Added
- **`src/telco_churn/models/calibrate.py`** — `CalibratedClassifierCV(ensemble=False)` cross-fit on the dev set; sigmoid selected over isotonic via a PR-AUC-preservation gate. Registers as `telco-churn-pipeline` / `challenger` (`ANALYSIS.md` §5).
- **`src/telco_churn/models/threshold.py`** — closed-form `t* = c / (r × LTV)`, replacing the classical cost-matrix rule (`ANALYSIS.md` §6).
- **`src/telco_churn/models/plots.py`** — reliability-diagram, EV-curve, and retention-sensitivity plotting helpers.
- **`src/telco_churn/features/accessor.py`** — `load_features()`, the single accessor owning the processed-features path/format/hash.
- **`src/telco_churn/utils/stats.py::paired_bootstrap_ci`** — generic paired-bootstrap CI, extracted from `comparison.py`.
- `configs/calibration/default.yaml`, `configs/costs.yaml`, `configs/threshold/default.yaml`; `notebooks/04-calibration-and-threshold.ipynb`.
- `tests/unit/test_calibrate.py`, `test_threshold.py`, `test_mlflow.py`, plus subprocess integration tests — suite now 420 passed / 38 skipped, 94.15% coverage.

### Changed
- **`ANALYSIS.md`** — added §5 Probability Calibration and §6 Business Impact & Threshold Selection; corrected several inaccuracies inherited from the exploratory-pass narrative.
- **`README.md`** — results, data-splits, and pipeline diagram corrected to match the real pipeline; extended through calibrate → threshold.
- **`Makefile`** — `train` runs the module directly; `calibrate`/`threshold` targets added.
- **`configs/config.yaml`** — registers `calibration`/`threshold` Hydra defaults.
- **Notebooks** (`00`, `01`, `02a`, `02b`, `03c`, `04`) — cross-references completed; stream-output fragmentation normalized via `scripts/fix_notebook_outputs.py`.

---

## [0.5.4] - 2026-07-11 — Phase 5→6 Bridge: Registration Boundary Remediation

*Phase 5 shipped before `CLAUDE.md`'s registry boundary was written, and had registered the uncalibrated pipeline as `challenger` — not a valid rollback target under that boundary. This bridge removes registration from Phase 5 (deferred to Phase 6's `calibrate.py`), fixes two defects in the same code paths, and rebuilds from a hard reset to prove it reproduces from zero.*

### Changed
- **`src/telco_churn/models/train/log_model.py`** (replaces `registration.py`) — logs the tuned `Pipeline` without `registered_model_name=` or a `challenger` alias; adds `training_manifest["logged_model_uri"]` as the permanent handle Phase 6 resolves by.
- **`tests/unit/test_train_log_model.py`** (renamed) — asserts `search_registered_models()` stays empty after Step 5.
- **`ANALYSIS.md`, `notebooks/03c-hyperparameter-tuning.ipynb`, `PROJECT_PLAN.md`** — registration narrative and Phase 5 deliverable updated to log-only, no registration.

### Fixed
- **`models/train/common.py::_resolve_tracking_uri`** — a bare relative tracking URI resolved to a Windows path MLflow's store registry rejected (`urlparse` read the drive letter as scheme `'c'`); now anchored via `.as_uri()`.
- **MLflow test-fixture artifact leak** — several fixtures left the artifact root defaulting to `./mlruns`, leaking run directories into the repo on every test run. Fixed via a shared `conftest.py` fixture with an explicit `artifact_location`.
- **Logged model signature declared `predict()` output, not `predict_proba()`** — `pyfunc_predict_fn` left at its default; now explicit.

---

## [0.5.3] - 2026-07-09 — QA: Step 2/3 Orchestrator Test Coverage

*A Phase 5 QA pass found that `run_comparison_step` and `run_selection_step` — the Step 2 and
Step 3 top-level entry points — had no direct unit test, unlike their sibling step orchestrators
(`run_candidate_step`, `run_tuning_step`, `run_registration_step`); they were only exercised
indirectly through the full subprocess integration test. No production code changed.*

### Added
- **`tests/unit/test_train_feature_freeze.py`** (7 tests, new file) — direct coverage of
  `run_selection_step`: `committed_features`/SHAP-audit consistency with the `reduced`/`full`
  decision branch, a wiring cross-check against `bootstrap_test`'s own decision, and the MLflow
  artifact/tag contract via a real `MlflowClient`.
- **`tests/unit/test_train_comparison.py`** — +4 tests covering `run_comparison_step` directly:
  return-keys contract, decision cross-checked against `bootstrap_comparison` called independently
  on the same inputs, the comparison/diagnostics MLflow artifact contract, and the tag contract.

### Fixed
- Coverage on `models/train/comparison.py` and `models/train/feature_freeze.py` raised from
  60%/20% to 100%/100% (full suite: 350 → 361 tests; 86.5% → 93.0% overall).

---

## [0.5.2] - 2026-07-08 — Phase 5 Steps 4-5: Hyperparameter Tuning & Challenger Registration

*A 50-trial TPE Optuna study tunes LightGBM on the frozen feature set. The 1-SE rule selects a more-regularized trial than raw-best: CV PR-AUC rises from 0.6582 to 0.6659 while the train-vs-CV gap narrows from 0.1362 to 0.0328. Registered as `telco-churn-pipeline` / `challenger` — uncalibrated, not serving-ready.*

### Added
- **`src/telco_churn/models/train/tuning.py`** — Step 4: `run_tuning_step`, a Postgres-backed Optuna TPE study with content-addressed study naming so an incompatible trial pool can never mix silently. `select_best_trial` (`argmax`/`1se`), `boundary_hit_check`.
- **`src/telco_churn/models/train/registration.py`** — Step 5: refits the selected trial on all of development, asserts log→reload→predict parity, registers `telco-churn-pipeline` / `challenger`.
- **`configs/tuning/optuna.yaml`** — search space and study knobs (`n_trials=50`, `cv_folds=5`, `selection_rule=1se`).
- **`training_manifest.json`** (logged at registration) — git SHA, data hash, hyperparameters, CV PR-AUC, paired-Δ vs. LogReg, `tuning_summary`.
- **`tests/unit/test_train_tuning.py`** (24), **`test_train_registration.py`** (4); **`tests/integration/test_train_subprocess.py`** (2) — 1-SE selection, boundary-hit, reload-parity, full Steps 1-5 composition coverage.

### Changed
- **`ANALYSIS.md`** §4c — Optuna result recorded: selected trial vs. raw-best, boundary-hit check clears on all 8 hyperparameters; registration result: `telco-churn-pipeline` v1, alias `challenger`, `n_estimators=59`.

---

## [0.5.1] - 2026-07-03 — Phase 5 Step 3: Permutation-Importance Feature Selection

*Replaces Step 3's gain-based null-importance selector with permutation importance vs. a synthetic noise-decoy column, adds a non-gating SHAP audit, and replaces the keep-vs-reduce adoption test with a paired-bootstrap test. Re-run: the full 20-feature set is retained decisively (Δ = 0.0173, 95% CI [0.0104, 0.0246], `material_full_win`).*

### Added
- **`src/telco_churn/features/select.py`** — `PermutationImportanceSelector` replaces `NullImportanceSelector`; `compute_shap_audit()`; `reduced_set_bootstrap_test()` replaces the unpaired within-CI adoption check.
- **`shap`** added as a project dependency.

### Changed
- **`src/telco_churn/models/train/feature_freeze.py`** — calls the new SHAP audit; adopts the reduced set by default, overriding to full only on `material_full_win`.
- **`configs/training/selection.yaml`** — `n_permutations`/`cutoff_percentile` renamed `n_repeats`/`noise_floor_margin`.
- **`tests/unit/test_select.py`, `notebooks/03b-feature-selection.ipynb`, `ANALYSIS.md` §4, `PROJECT_PLAN.md`** — rewritten for the new method and re-run result.

### Fixed
- **`tests/unit/test_train_common.py`** — asserted the old renamed config keys; updated.

---

## [0.5.0] - 2026-07-03 — Phase 5 Steps 1-2: Model Selection (Candidate Comparison & Diagnostics)

*Candidate bake-off — `DummyClassifier`, `LogisticRegressionCV`, default-config LightGBM — on one shared `RepeatedStratifiedKFold`. LightGBM adopted under the pre-registered paired-bootstrap rule: Δ = +0.0071 PR-AUC, 95% CI [+0.0029, +0.0113], clears the Δ*=0.005 threshold (`material_lgbm_win`). Non-gating fixed-recall and per-segment diagnostics logged alongside, never deciding the family.*

### Added
- **`src/telco_churn/features/schema.py`** — `FeatureSchema` frozen dataclass replacing bare `list[str]` column-group constants.
- **`src/telco_churn/features/preprocessing.py`** — `build_linear_preprocessor` for the `DummyClassifier`/`LogisticRegressionCV` baselines.
- **`src/telco_churn/models/train/common.py`** — shared helpers reused by every training step.
- **`src/telco_churn/models/train/candidates.py`** — Step 1: CV-scores the three candidates; hard-assertion leakage canary on the dummy candidate.
- **`src/telco_churn/models/train/comparison.py`** — Step 2: `bootstrap_comparison`, paired bootstrap on Δ = AP(LGBM) − AP(LogReg).
- **`src/telco_churn/models/diagnostics.py`** — pure helpers: `fixed_recall_profile`, `segment_oof_errors`, `segment_bootstrap_delta`, `generalization_gap`, `learning_curve_points`.
- **`src/telco_churn/models/train/__main__.py`** — CLI entry point.
- **`configs/training/lightgbm.yaml`**, **`configs/training/logreg.yaml`**; Postgres-backed MLflow tracking server (`docker/mlflow/`, `docker-compose.yml`).
- **`tests/unit/test_train_candidates.py`** (4), **`test_train_comparison.py`** (13), **`test_train_common.py`** (15), **`test_diagnostics.py`** (25).

### Changed
- **`ANALYSIS.md`** — Step 2 result recorded: `material_lgbm_win` (Δ = +0.0071, 95% CI [+0.0029, +0.0113]).

---

## [0.4.3] - 2026-07-02 — Canonical Split Refactor (Phase 4a Rework)

*Seals the dev/test split as a canonical artifact before feature discovery runs, closing the
"test set touched once" gap Phase 4a previously accepted as a documented limitation. Phase 4a
discovery re-run on the dev partition only; adopted feature set unchanged (`charge_per_service`).
14 new tests.*

### Added
- **`src/telco_churn/data/split.py`** — canonical `customerid`/`churn` partition: `make_split()`,
  `write_split()`/`load_split()`, `dev_ids()`/`test_ids()`/`partition()`; `__main__` CLI writes
  `datasets/processed/split_manifest.parquet`.
- **`tests/unit/test_split.py`** — 14 tests: determinism, order-invariance, stratification,
  disjointness, full coverage, manifest round-trip, `partition()` membership assertion.
- **`make split`** Makefile target, between `validate` and `features`; `test-data` scoped target.

### Changed
- **`notebooks/02a-feature-discovery.ipynb`** — re-run on the dev partition only (via
  `data.split.partition()`), superseding the earlier full-dataset run; adopted set unchanged.
- **`ANALYSIS.md`** §3a/§4 — OOF PR-AUC and blind-spot figures updated to the dev-only rerun.
- **`PROJECT_PLAN.md`** — added the Canonical Data Split step before Phase 4a; rewrote the
  Phase 4a split-timing caveat from "documented limitation" to "resolved".

### Fixed
- **`make_split()`** — sorted `ids`/`labels` by `customerid` before splitting, since
  `sklearn.train_test_split`'s stratified sampling is positionally order-sensitive even with a
  fixed `random_state`; an unordered source (e.g. Postgres `SELECT *`) could otherwise select a
  different partition than a CSV-ordered source.

---

## [0.4.2] - 2026-06-27 — Phase 4a/4b: Feature Discovery & Engineering

*Phase 4a ran a nine-lap structured feature search; `charge_per_service` was the sole adoption.
Phase 4b pruned all rejected candidates from the feature pipeline. Feature set: 20 columns
(19 raw IBM + `charge_per_service`). 206 unit tests; 94.97% coverage.*

### Added
- **`notebooks/02a-feature-discovery.ipynb`** — nine-lap search: OOF error profile → hypothesis
  → four-screen gate (leakage, redundancy, ΔPR-AUC, permutation importance).
- **`src/telco_churn/features/generate.py`** — discovery machinery: OOF predictor,
  blind-spot profiler (`profile_false_negatives`), four-screen adoption gate
  (`serving_available`, `redundancy_screen`, `candidate_importance`, `adoption_gate`),
  bootstrap PR-AUC CI, and provenance writer; typed result dataclasses (`LapRecord`,
  `AdoptionDecision`, `RedundancyResult`, `ImportanceResult`).
- **`src/telco_churn/utils/stats.py`** — `abs_corr()` (absolute Spearman), `cramers_v()`,
  and `vif_single()`; shared statistical helpers across `data.eda` and `features.generate`.
- **`reports/feature_discovery/provenance.json`** — per-lap gate outputs (screen results, ΔPR-AUC,
  CI, importance, decision); machine-readable audit trail.
- **`reports/feature_discovery/adopted_features.json`** — frozen adopted-feature list
  (`["charge_per_service"]`).
- **Provenance cross-check** (`tests/unit/test_build.py`) — reads `adopted_features.json`; asserts
  every adopted feature appears in the column groups; tripwire if Phase 4a reruns.

### Changed
- **`notebooks/02b-feature-engineering.ipynb`** — renamed from `02-feature-engineering.ipynb`;
  rewritten as a verification wrapper: loads the feature view, renders the 20-column inventory,
  confirms output shape; Phase 4a outcome summary in the opening cell.
- **`src/telco_churn/features/build.py`** — `PYTHON_ENGINEERED_COLS`, module-load assertion guard,
  and `_add_python_features()` removed; `build_feature_df()` is now a schema-validated pass-through.
- **`src/telco_churn/features/schema.py`** — `tenure_cohort` field and three constant sets removed;
  `FeatureOutputSchema` collapsed to a `coerce=False` subclass with no additional columns.
- **`sql/features/customer_features.sql`** — `tenure_buckets` JOIN and `tenure_cohort` column removed.
- **`src/telco_churn/features/sql_features.py`** — `_SQL_FILES` updated to two entries;
  `tenure_buckets.sql` reference removed.
- **`tests/unit/test_build.py`** — complete rewrite; H1/H2/H3 tests (~15) removed; 14 tests remain
  covering dtype invariants, null propagation, column-count stability, and provenance cross-check.

### Removed
- **`sql/features/tenure_buckets.sql`** — `tenure_cohort` rejected at Phase 4a gate (screen 4).
- **`PYTHON_ENGINEERED_COLS`** from `features/build.py` and `features/__init__.py`.
- **Cross-schema invariant tests** (`tests/unit/test_schema.py`) — removed with the
  `FeatureOutputSchema` columns that depended on Python-engineered features.

---

## [0.4.1] - 2026-06-12 — Post-Phase 4 QA Hardening

*Addresses correctness and code-quality gaps identified across Phases 1–4. Highlights: ingest
rewritten to the staging-table upsert pattern; `RawSchema` nullability corrected for 19 columns;
`clean_dataframe()` placeholder removed; Gate 5 null-check set made schema-derived. No modelling
logic changed. 169 unit tests; 94.97% coverage.*

### Added
- **`src/telco_churn/utils/paths.py`** — `get_project_root()` walks upward from `__file__` to
  the first directory containing `pyproject.toml`. Canonical path anchor for all `src/` modules;
  replaces three separate inline copies scattered across `__main__` blocks.
- **`data/__init__.py`** — was empty; now re-exports the `data` package's public API with a
  matching `__all__`.
- **`ingest.setup_schema(engine)`** (`data/ingest.py`) — new public function; executes
  `001_create_raw.sql` via `CREATE TABLE IF NOT EXISTS` so the PRIMARY KEY is never silently
  dropped between runs.
- **`eda.inspect_missing(df)`** (`data/eda.py`) — returns missingness-context rows for each null
  column; supports MCAR/MAR/MNAR analysis in the EDA notebook.
- **`__all__`** defined in every public module under `src/` — `data/schema.py`, `data/checks.py`,
  `data/validate.py`, `data/ingest.py`, `data/eda.py`, `features/schema.py`, `features/build.py`.
  Controls the star-import surface and documents the intended public API per module.
- **Module-load assertion guard** (`features/build.py`) — asserts `PYTHON_ENGINEERED_COLS ⊆` the
  union of the four typed feature lists at import time; raises immediately if the sets diverge.
- **Integration tests** (`tests/integration/test_validate_postgres.py`) — subprocess tests
  covering `validate.py __main__` exit 0 (valid data) and exit 1 (blocking gate failure).
- **Unit tests** (`tests/unit/test_schema.py`) — 5 tests: DDL↔`RawSchema` column parity (no
  Docker required) and `CleanedSchema`↔`FeatureOutputSchema` cross-schema invariant consistency.
- **Unit tests** (`tests/unit/test_ingest.py`) — 2 tests added: `test_missing_column_raises` and
  `test_extra_column_raises` covering `load_raw_csv` column-validation.
- **Unit tests** (`tests/unit/test_checks.py`) — 1 test:
  `test_null_rate_check_covers_all_non_nullable_schema_columns` verifying Gate 5 monitors all 19
  non-nullable columns, not just the original hardcoded 3.
- **Unit tests** (`tests/unit/test_eda.py`) — 7 tests added for `inspect_missing`.

### Changed
- **`ingest.py` rewritten** (`data/ingest.py`) — `to_sql(if_exists='replace')` replaced with an
  atomic staging-table upsert; `ingest()` now calls `validate_raw(strict=True)` before writing.
- **`001_create_raw.sql`** — 19 columns upgraded from nullable to `NOT NULL`, four with matching
  `CHECK` constraints.
- **`RawSchema` nullability** (`data/schema.py`) — 19 fields corrected to `nullable=False`;
  `ingest.py` now derives the required column set from the schema instead of a hardcoded list.
- **`_NULL_CHECKED_COLS`** (`data/checks.py`) — schema-derived from `RawSchema` non-nullable
  columns instead of a hardcoded 3-column set.
- **`eda.py` column constants** — changed from `list[str]` to `Final[tuple[str, ...]]`.
- **`eda.compute_vif()`** — `sklearn.LinearRegression` replaced with `numpy.linalg.lstsq`,
  eliminating a sklearn dependency for one function; warns when any VIF is `inf`.
- **`eda.detect_outliers()`, `compute_chi2_tests()`, `compute_mann_whitney()`, `compute_vif()`**
  — mutable list default arguments replaced with `None` + in-body defaults.
- **`CLAUDE.md`** Code Style — three rules added: `__all__` required in every public module;
  `get_project_root()` instead of bare relative paths; `exc_info=True` on `logger.error()`.
- **`PROJECT_PLAN.md`** — Phase 8 deliverables: Alembic note added.

### Fixed
- **`clean_dataframe()` removed** (`data/validate.py`) — was a Phase 2 placeholder that
  duplicated the Phase 5 `ColumnTransformer`/`SimpleImputer` responsibility. 6 unit tests
  removed alongside it.
- **`validate_raw` / `validate_clean` duplication** — identical gate-assembly blocks extracted
  into a private `_run_gates()` helper.
- **`validate.py __main__`** — `validate_raw(strict=False)` replaced with `strict=True` +
  `except ValidationError`; `pragma: no cover` removed (now covered by integration test).
- **Bare `OmegaConf.load` paths** — fixed in `ingest.py __main__`, `validate.py __main__`, and
  `sql_features.py __main__`; all now use `get_project_root() / "configs" / "config.yaml"`.
  Inline `_project_root()` copy in `sql_features.py` removed.
- **`_REPORTS_DIR`** (`validate.py`) — was `Path("reports/validation")` (CWD-relative); replaced
  with `get_project_root() / "reports" / "validation"`.
- **Typed exception handlers** — bare `except Exception` replaced with typed handlers across
  `ingest.py`, `validate.py`, and `features/build.py __main__` blocks; all include `exc_info=True`.
- **`eda_df` fixture** (`tests/unit/test_eda.py`) — was generating structurally invalid data
  (`multiplelines` independent of `phoneservice`); now respects IBM Telco structural constraints.
- **Integration test imports** — `test_ingest_postgres.py`, `test_sql_features_postgres.py`,
  `test_validate_postgres.py` imported `get_project_root` from `tests/unit/helpers.py` via pytest
  sys.path manipulation; replaced with `from telco_churn.utils.paths import get_project_root`.
  Duplicate removed from `helpers.py`.
- **`sql/features/tenure_buckets.sql`** — stale `pd.cut` reference removed; implementation uses
  `CASE WHEN`.
- **`sql/features/charge_per_service.sql`** — comment added documenting the intentional
  `phoneservice`/`multiplelines` double-count in `service_count`.

---

## [0.4.0] - 2026-06-08 — Phase 4: Feature Engineering (SQL + Python)

*Builds features in two layers: SQL views in Postgres and four hypothesis-driven Python columns.
Returns a raw, untransformed feature DataFrame — ColumnTransformer fitting and train/val/test
split are Phase 5 responsibilities.*

### Added
- **SQL feature views** (`sql/features/`) — `tenure_buckets.sql` (four tenure cohorts),
  `charge_per_service.sql` (monthly charges ÷ active service count), and `customer_features.sql`
  (join view read by the Python feature builder). Built idempotently via `build_sql_features(engine)`.
- **Python feature engineering** (`src/telco_churn/features/build.py`) — `build_feature_df(df)`
  adds four columns: `is_long_month_to_month` (H1), `monthly_to_total_ratio` (H2),
  `fiber_contract` and `dsl_contract` (H3a/b — contract × internet-service interactions).
  Returns the raw augmented DataFrame; y extraction is Phase 5's responsibility. NaN preserved
  for 11 zero-tenure rows for Phase 5 imputation.
- **Column group exports** — `BINARY_STR_COLS`, `BINARY_INT_COLS`, `MULTI_CAT_COLS`, `NUMERIC_COLS`,
  `PYTHON_ENGINEERED_COLS` exported from `build.py` and surfaced via `features/__init__.py` as the
  public API for Phase 5.
- **SQL feature runner** (`src/telco_churn/features/sql_features.py`) — executes the three SQL
  files in dependency order via SQLAlchemy; idempotent (`CREATE OR REPLACE VIEW`).
- **Feature engineering notebook** (`notebooks/02b-feature-engineering.ipynb`) — thin wrapper
  rendering SQL feature distributions and Python feature validation outputs.
- **Unit tests** (`tests/unit/test_build.py`) — 37 tests covering H1, H2, H3a, H3b correctness,
  NaN propagation, column count invariant, target/ID exclusion, and no input mutation. Includes
  `hypothesis` property-based tests.
- **Integration tests** (`tests/integration/test_sql_features_postgres.py`) — verifies SQL views
  are created correctly against a live Postgres instance; skips when Docker is not running.

### Changed
- `ANALYSIS.md §3` rewritten with Phase 4 results: SQL feature evidence tables, Python feature
  hypothesis → evidence → result for H1–H3, and the 25-column feature inventory.
- `PROJECT_PLAN.md` Phase 4 spec cleaned up; Phase 5 updated to own `ColumnTransformer`
  definition and fitting; `FeatureSchema` frozen dataclass added to Phase 5 deliverables.
- `CLAUDE.md` Testing section — two rules added: `__main__` CLI entry points require an
  integration test covering the full pipeline path; each new package requires a scoped
  `make test-<package>` Makefile target.

### Fixed
- `make test-features` added to `Makefile` — scoped to `--cov=src/telco_churn/features`; Phase 4
  tests run in isolation without a false `fail_under=80` failure from uncovered Phase 2 modules.
- `_PYTHON_ENGINEERED` renamed to `PYTHON_ENGINEERED_COLS` in `features/build.py`, exported from
  `features/__init__.py`, and import in integration test updated to the public path.
- `CustomerFeaturesSchema.monthlycharges` now declares `lt=np.inf`; `FeatureOutputSchema.monthly_to_total_ratio`
  drops `lt=np.inf` — `inf` is now unreachable by construction rather than caught on output.
- `customer_features.sql` LEFT JOIN intent undocumented; comment added noting `customers_raw` is
  the authoritative row source and Pandera `nullable=False` is the downstream guard.
- `_make_feature_row` fixture hardcoded `charge_per_service` as `/ 2`; corrected to derive from
  service flags via the SQL formula (correct for default configuration is `/ 3`).
- `test_build_feature_df_invalid_tenure_cohort_raises` added — invalid `tenure_cohort` category
  was not covered by existing input schema guard tests.
- Hypothesis `max_examples` raised from 40 → 100 on both property tests.
- `build_feature_df` docstring corrected from "H1–H3" to "H1, H2, H3a, and H3b".
- `exclude_lines` was under `[tool.coverage.run]` instead of `[tool.coverage.report]`; coverage.py
  silently ignored it. Moved to the correct section — suite coverage corrected from 86.61% to 97.64%.
- H1 boundary (`tenure > 24`) had no fencepost tests at `tenure=24` or `tenure=25`; two tests added.
- Four queries in `test_sql_features_postgres.py` used f-string interpolation into raw SQL;
  replaced with SQLAlchemy bound parameters.
- `build_feature_df` previously raised an opaque `KeyError` when SQL view columns were missing;
  now raises a named `SchemaError` via `CustomerFeaturesSchema` input validation.
- `SELECT *` in `build.py __main__` replaced with an explicit column list from
  `BINARY_STR_COLS + BINARY_INT_COLS + MULTI_CAT_COLS + NUMERIC_COLS`.
- `customerid` was excluded from `telco_churn_processed.csv`; now written as the first column.
- `make test-integration` was a silent no-op — ran without `--run-integration` so all integration
  tests were skipped. Flag added to the target.
- `make features` target was missing from the `Makefile`; added to complete the
  `ingest → validate → features → train` chain.

---

## [0.3.0] - 2026-06-06 — Phase 3: EDA Notebook

*Promotes all EDA logic from the original research notebook into a testable, importable `src/`
module. The EDA notebook becomes a thin rendering wrapper; statistical helpers become reusable
production code.*

### Added
- **EDA helper library** (`src/telco_churn/data/eda.py`) — seven public functions: IQR outlier
  detection, per-group churn rates, chi-squared + Cramér's V, Mann-Whitney U + rank-biserial r,
  Pearson correlation matrix, top-N target correlations, and VIF. All functions handle edge cases
  (empty DataFrame, NaN, constant columns, perfect collinearity) via `warnings.warn` rather than
  raising.
- **VIF without `statsmodels`** — derived from `sklearn.LinearRegression` R², eliminating a
  50 MB+ dependency.
- **Column constants** — `CAT_FEATURES`, `NUM_FEATURES`, `BINARY_INT_FEATURES`, `TARGET` exported
  as the single source of truth for column lists across EDA, validation, and feature engineering.
- **EDA notebook** (`notebooks/01-eda.ipynb`) — thin wrapper covering class imbalance, univariate
  distributions, bivariate churn-rate analysis, statistical tests with effect sizes, correlation
  heatmap, VIF table, and a contract × internet-service interaction that motivates Phase 4
  engineering. Original archived at `notebooks/_archive/EDA-original.ipynb`.
- **Unit tests** (`tests/unit/test_eda.py`) — 50+ tests covering all seven functions under normal
  inputs, missing values, wrong dtype, and empty DataFrame. Warning emission verified with
  `pytest.warns`.

---

## [0.2.2] - 2026-06-03 — Pre-Phase 3 Cleanup

*Fixes identified in a post-QA audit. No modelling logic changed.*

### Added
- Two missing schema constraint tests added to `tests/unit/test_checks.py`: invalid
  `contract_type` value and unexpected extra column. Test count: 49 → 51; coverage holds at
  80.73 %.

### Changed
- `CLAUDE.md` corrected: source-of-truth pointers, test file references, and Phase 7 notebook
  name fixed.

### Fixed
- `mirrors-mypy` pre-commit hook bumped from `v1.13.0` → `v2.1.0` to match `mypy>=2.1.0` in
  `pyproject.toml`; eliminates "passes locally, fails in CI" type errors.
- Pre-commit mypy hook now runs `mypy src/` with `pass_filenames: false` — matches CI exactly
  and avoids duplicate-module errors from the two `conftest.py` files.
- `tests/` added to mypy `exclude` list in `pyproject.toml`; type-checking scoped to `src/` only.

---

## [0.2.1] - 2026-06-03 — QA & Standards Hardening

*Addresses code quality and documentation gaps identified in a review against industry DS
standards. No modelling logic changed.*

### Added
- `CHANGELOG.md`, `CONTRIBUTING.md`, and `ANALYSIS.md` added.
- `src/telco_churn/py.typed` added (PEP 561) so downstream type checkers pick up inline types.
- `fail_under = 80` added to `pyproject.toml` — failing coverage is caught locally before CI.
- Minimal GitHub Actions CI pipeline (`.github/workflows/ci.yml`): lint, type-check, and unit
  tests with coverage on every push and PR.

### Changed
- `PROJECT_PLAN.md` restored and rewritten — Phases 3, 4, and the body of Phase 5 were missing
  due to file corruption; all 15 phases now present. Phase status table added.
- README rewritten as a project landing page; modelling rationale moved to `ANALYSIS.md`.
- Validation thresholds (min row count, max null rate) moved from source code to
  `configs/config.yaml`.
- Data source path now configurable via `--csv-path` CLI flag.

### Fixed
- Integration tests now skip automatically when Docker is not running rather than failing with a
  connection error. Pass `--run-integration` to run them explicitly.

---

## [0.2.0] - 2026-06-01 — Phase 2: Data Validation

*Establishes automated data quality checks that run before any modelling begins.*

### Added
- **Five validation gates** with two severity levels — ERROR (blocking) and WARNING (non-blocking,
  logged). All five pass on the IBM Telco dataset:
  - Gate 1 — Schema (ERROR): Pandera validates column presence, types, ranges, and allowed
    categorical values.
  - Gate 2 — Duplicate IDs (ERROR): asserts `customerid` is unique.
  - Gate 3 — Churn labels (ERROR): asserts `churn` is binary with no nulls.
  - Gate 4 — Unexpected TotalCharges nulls (WARNING): flags nulls only where `tenure > 0`; the
    11 zero-tenure nulls are expected and ignored.
  - Gate 5 — Distribution sanity (WARNING): row count below 1,000 or null rate above 5 % on
    key columns.
- **Schema inheritance** (`RawSchema` → `CleanedSchema`) — `CleanedSchema` additionally requires
  `totalcharges` to be non-null, verifying imputation ran before downstream stages.
- **`clean_dataframe()`** — median imputation for the 11 known NULL `totalcharges` rows;
  preserves all 7,043 customers.
- **Validation reports** written to `reports/validation/<timestamp>/` on failure — `summary.csv`
  per gate and `<gate>_failures.csv` for offending rows.

### Fixed
- Analysis notebooks now render correctly on GitHub. Notebook format upgraded and cell identifiers
  standardised; pre-commit hook added to prevent regression.

---

## [0.1.0] - 2026-05-31 — Phase 1: Data Ingestion

*Moves the raw CSV into Postgres, establishing the foundation for all downstream SQL-based
feature engineering.*

### Added
- **Raw data ingestion** (`src/telco_churn/data/ingest.py`) — loads CSV into `customers_raw`
  table. Retains all 7,043 rows including the 11 zero-tenure customers with null `TotalCharges`.
- **Idempotent ingestion** (`if_exists="replace"`) — re-running never creates duplicates.
- **SQL schema** (`sql/schema/001_create_raw.sql`) — explicit column types; `customerid` as
  primary key. Applied automatically on container first start.
- **Column name normalisation** — names lowercased and SQL reserved words renamed at ingest time
  (`partner` → `has_partner`, `contract` → `contract_type`).
- **Structured logging** — records row counts and table names on every run.
- **Postgres in Docker** (`docker-compose.yml`, `infra` profile) — reproducible local database
  with a healthcheck.
- **Unit and integration tests** — unit tests cover parsing; integration tests use
  `testcontainers` to verify the full CSV → Postgres path including idempotency.

---

## [0.0.1] - 2026-05-28 — Phase 0: Project Foundation

*Establishes the development environment, tooling, and project structure that all subsequent
phases build on.*

### Added
- **Reproducible environment** via `uv` + `pyproject.toml` + `uv.lock`.
- **Pre-commit hooks** — `ruff`, `black`, `mypy --strict` (src/ only), `detect-secrets`, and
  standard file checks.
- **Configuration-driven design** via Hydra — paths, random seed, MLflow settings, and tunable
  parameters in `configs/config.yaml`.
- **Structured JSON logging** via `structlog` — machine-readable, compatible with CloudWatch
  and Grafana.
- **Project directory skeleton** — `src/`, `tests/`, `configs/`, `sql/`, `pipelines/`, `docs/`,
  `notebooks/`, `datasets/`.
- **`Makefile` shortcuts** — `make lint`, `make test`, `make validate`, `make train`.
- **Architecture diagram** (`docs/architecture.md`) documenting the intended end-to-end system
  design.

---

<!-- Version comparison links: the `origin` remote (github.com/Ampofowaa/TelcoChurn_PortfolioProject)
already exists, but no `vX.Y.Z` tags have been pushed yet, so these links would 404 if uncommented
now. Un-comment once tags are pushed for each released version below.
[Unreleased]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.7.6...v0.8.0
[0.7.6]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.7.5...v0.7.6
[0.7.5]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.7.4...v0.7.5
[0.7.4]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.7.3...v0.7.4
[0.7.3]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.6.3...v0.7.0
[0.6.3]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.5.4...v0.6.0
[0.5.4]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.4.3...v0.5.0
[0.4.3]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/releases/tag/v0.0.1
-->
