# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions map to project phases, not strict Semantic Versioning: a new phase bumps MINOR
(e.g. Phase 5 starts at `0.5.0`); QA passes and sub-steps within a phase bump PATCH.
See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap.

---

## [Unreleased]

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

*One-time evaluation of the champion candidate against the sealed test set, closing the loop `ANALYSIS.md` §0's promotion gate defines: PR-AUC-driven selection, three veto-only guardrails, and a pre-registered human review (V1–V3). Gate result: pass (cold start); human review: approved. Full-data refit and registry promotion (`refit.py`/`register.py`) remain open, tracked in `PROJECT_PLAN.md`.*

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

*The one modelling fix in the Prerequisites section, shipped alone per the two-run ordering
Run 1 established: `n_estimators` was derived from Optuna's early-stopped median on each CV
fold's carved-down training partition (3,606 rows) and applied unscaled to the larger final
dev fit (5,634 rows), systematically under-boosting every shipped model. Scaling it moves CV
PR-AUC, BSS, ECE, and the empirical threshold checks — all attributable to this one change,
confirmed against Run 1's now-established baseline. A second, independently-additive fix
extends Run 1's dataset-lineage logging to the three step-level MLflow runs it missed.*

### Added
- `src/telco_churn/models/train/log_model.py` — `n_estimators` scaled by
  `n_final_fit / n_fold_fit` before the final dev fit, taking the shipped tree count from 94 to
  **147**; derivation and diagnostic in `ANALYSIS.md` §4c.
- `src/telco_churn/models/train/common.py::_log_dev_input` — dataset-lineage `log_input` now
  fires from the `model_comparison`, `feature_selection`, and `tuning_study` runs, which
  previously opened without recording what dataset they touched.
- `src/telco_churn/models/calibrate.py::calibration_slope` — adds an analytic (Wald) CI
  cross-checking the existing bootstrap CI (they agree closely); surfaced an intercept
  miscalibration the slope alone is blind to. Finding recorded in `ANALYSIS.md` §9 limitation
  #14.
- `configs/calibration/default.yaml` — `method` pinned from `auto` to **`sigmoid`**, the
  winner resolved from Run 2's `calibration_summary.json`.
- `ANALYSIS.md` — records the tree-count scaling derivation (§4c), the calibration-intercept
  finding and new limitation #14, and the resulting Δ across the calibration and threshold
  tables.

### Changed
- **Run 2 measurement** — dev CV PR-AUC (sigmoid) 0.6669 → 0.6684, BSS 0.3098 → 0.3111,
  ECE 0.0217 → 0.0222, base-scenario empirical threshold 0.4428 → 0.4150 (within its bootstrap
  CI; closed-form `t* = 0.3941` unchanged). Notebooks `03a`–`03c` and `04` re-executed against
  the Run 2 pipeline.

---

## [0.6.2] - 2026-07-16 — Phase 7 Prerequisites: MLflow Lineage & Evidence-Persistence Fixes (Run 1)

*Four logging/lineage defects found while designing Phase 7, all in already-shipped Phase 5/6
code — none of which changes a computed number. Verified via a hard reset (Postgres volume,
`mlruns/`, `mlflow.db`) and full rebuild: every existing number in `ANALYSIS.md` §4–§6
reproduced identically, confirming the pipeline is deterministic and these fixes are inert.*

### Added
- `configs/config.yaml`, `src/telco_churn/utils/mlflow.py::resolve_tracking_uri` — tracking-URI
  fallback now resolves to `sqlite:///mlflow.db`, not the bare `mlruns` file store (raises as of
  MLflow 3.14).
- `src/telco_churn/models/train/log_model.py`, `src/telco_churn/models/calibrate.py` —
  `logged_model_id` persisted to `training_manifest.json` and as a model-version tag, giving the
  registry a path to the `LoggedModel` Phase 7's evaluation logging attaches metrics to.
- `src/telco_churn/models/train/candidates.py` — logged dataset `source` now resolves through
  `features/accessor.py::features_path()` instead of a hardcoded path.
- `src/telco_churn/models/calibrate.py::calibration_slope` — Cox calibration slope + bootstrap CI,
  logged in `calibration_summary.json` alongside the dev-OOF probability vector
  (`dev_oof_predictions.parquet`), previously computed and discarded.
- `src/telco_churn/models/calibrate.py`, `src/telco_churn/models/threshold.py` — dev calibration/
  threshold metrics (`dev_brier`, `dev_bss`, `t_star_{scenario}`, etc.) now logged as MLflow
  metrics, not only inside JSON artifacts; the EV curve persisted as `ev_curve.parquet`.
- `src/telco_churn/models/threshold.py::expected_value_at_threshold`, `costs_config_hash` — new
  pure functions; `configs/policy/threshold.yaml` pinned by `costs_config_hash` instead of a
  model stamp.

### Changed
- **Run 1 reproducibility audit** — registry holds exactly one version (`refit_scope: dev`,
  aliased `challenger`); every family-comparison delta, frozen feature set, tuned hyperparameter,
  calibration diagnostic, and threshold value matched `ANALYSIS.md` exactly. Clears Run 2 to
  isolate its own effect against this baseline.

---

## [0.6.1] - 2026-07-14 — Phase 6 QA Pass: Test Isolation & Calibration-Selection Hardening

*A QA pass over Phase 6 found the calibrate/threshold unit tests were not hermetic
(silently falling through to real, gitignored project data instead of the synthetic
fixtures every other test uses), the `calibration.method='auto'` decision path — the
mode that actually produced the shipped model — had two branches with no real test
coverage, and `calibration_summary.json`'s `switch_decision` field had three
inconsistent shapes depending on which branch produced it. All are fixed; none change
the currently-registered model or its documented Phase 6 numbers.*

### Fixed
- **`tests/unit/test_calibrate.py`, `tests/unit/test_threshold.py`** — `run_calibration_step`/
  `run_threshold_step` tests called `load_dev_features()` with no override, which falls
  through to real disk (`load_features()` → `partition()` → `load_split()`) reading the
  gitignored `datasets/processed/` directory; passed only because that directory happened
  to exist locally. Added a `sandboxed_dev_features` fixture (`test_calibrate.py`) and the
  equivalent inline patch in `test_threshold.py`'s `registered_model_version` fixture,
  redirecting both `calibrate.load_dev_features` and `threshold.py`'s separate
  `from ... import load_dev_features` binding to the synthetic `dev_split` fixture.
  Verified hermetic by re-running with `PROCESSED_DATA_DIR` pointed at a nonexistent path.
- **`src/telco_churn/models/calibrate.py::select_calibration_method`** — in
  `calibration.method='auto'` mode, only isotonic's per-fold AP was checked against
  `pr_auc_gate_passes`; sigmoid (the fallback method whenever isotonic is disqualified)
  was never itself gated, unlike pinned mode. Added the same gate check to sigmoid,
  raising `ValueError` rather than silently registering a model with degraded ranking.
- **`CLAUDE.md`** — Data Handling section referenced `features/io.py::load_features()`;
  corrected to `features/accessor.py::load_features()`, the actual Phase 6 module name.

### Added
- **`tests/unit/test_accessor.py`** — dedicated unit tests for
  `src/telco_churn/features/accessor.py` (previously untested directly): `features_path`
  override behavior, `load_features` schema-validation failures (missing column, wrong
  dtype, zero-byte file, header-only file), and `features_sha256` hash stability/content-
  sensitivity. `features/accessor.py` now at 100% line coverage; wired into
  `Makefile`'s `test-features` target.
- **`tests/unit/test_calibrate.py`** — two tests closing previously-uncovered branches of
  `select_calibration_method`'s `'auto'` mode: `isotonic_disqualified_pr_auc_gate` (real,
  unmocked run) and the real Brier-bootstrap switch decision (isotonic forced past the
  PR-AUC gate, everything else genuinely computed). `calibrate.py` now at 98% line
  coverage.

### Changed
- **`src/telco_churn/models/calibrate.py::select_calibration_method`** —
  `switch_decision` is now always present in the return value with the same six keys
  (`method`, `decision_rule`, `delta_brier_obs`, `delta_brier_ci_lower`,
  `delta_brier_ci_upper`, `n_bootstrap`) regardless of which branch produced it —
  previously absent entirely in pinned mode and a 2-key dict when isotonic was
  disqualified, versus the full 6-key dict from a real Brier-bootstrap decision.
  `run_calibration_step` reads it via direct indexing instead of `selection.get(...)`.
- **`ANALYSIS.md`** §5 — added a note on the sigmoid-vs-isotonic switch decision's
  paired bootstrap CI: it resamples only `outer_cv_folds` (= 5) paired fold-level Brier
  differences, a small effective resample space; not load-bearing for the shipped v1
  decision since the PR-AUC gate resolved it first, but worth revisiting if a future
  retrain's decision actually reaches the Brier bootstrap.

---

## [0.6.0] - 2026-07-13 — Phase 6: Calibration & Cost-Sensitive Threshold

*Wraps the tuned LightGBM pipeline in sigmoid calibration (pooled Brier 0.1345, Brier
Skill Score 0.31 — 16.5% better than uncalibrated) and derives the production decision
threshold from a closed-form cost model (`t* = c/(r×LTV)`), not an empirical
cost-matrix cutoff — the point where the model becomes a business decision, not just a
ranking. Base scenario ships at `t* = 0.3941` (30.8% contact rate), agreeing with its
empirical argmax-EV check. This phase performs the training cycle's single MLflow
registration, pointing `challenger` at the calibrated pipeline; promotion to `champion`
remains Phase 7's job.*

### Added
- **`src/telco_churn/models/calibrate.py`** — `CalibratedClassifierCV(ensemble=False)`
  cross-fit on the dev set; sigmoid selected over isotonic via a PR-AUC-preservation gate.
  Registers the calibrated pipeline as `telco-churn-pipeline` / `challenger` — the training
  cycle's single registration. Detail in `ANALYSIS.md` §5.
- **`src/telco_churn/models/threshold.py`** — closed-form operating threshold
  `t* = c / (r × LTV)`, replacing the classical cost-matrix rule. Ships `threshold.json` /
  `configs/policy/threshold.yaml` with per-scenario thresholds and an empirical argmax-EV
  agreement check. Detail in `ANALYSIS.md` §6.
- **`src/telco_churn/models/plots.py`** — reliability-diagram, expected-value-curve, and
  retention-sensitivity plotting helpers.
- **`src/telco_churn/features/accessor.py`** — `load_features()`, the single accessor owning
  the processed-features path, format, and `sha256` content hash.
- **`src/telco_churn/utils/mlflow.py`** — `resolve_tracking_uri()`, extracted from
  `models/train/common.py` for reuse.
- **`src/telco_churn/utils/stats.py::paired_bootstrap_ci`** — generic paired-bootstrap CI,
  extracted from `comparison.py` and reused by `threshold.py`.
- `configs/calibration/default.yaml`, `configs/costs.yaml`, `configs/threshold/default.yaml`
  — calibration, cost-scenario, and threshold-derivation config.
- `notebooks/04-calibration-and-threshold.ipynb` — reliability diagrams, cost curves,
  threshold-by-scenario and sensitivity plots.
- `tests/unit/test_calibrate.py`, `test_threshold.py`, `test_mlflow.py`,
  `tests/integration/test_calibrate_subprocess.py`, `test_threshold_subprocess.py` —
  suite now 420 passed / 38 skipped, 94.15% coverage (up from 371 tests in Phase 5).

### Changed
- **`ANALYSIS.md`** — added §5 Probability Calibration and §6 Business Impact & Threshold
  Selection; corrected several inaccuracies inherited from the exploratory-pass narrative
  (nested CV mechanics, isotonic-overfitting explanation, ARPU characterization) and removed
  an unreproducible recall/precision figure.
- **`README.md`** — results table, data-splits description, and pipeline diagram corrected to
  match the real Phase 5/6 pipeline; tech stack and quick start extended through
  calibrate → threshold.
- **`Makefile`** — `train` target now runs the module directly (was a premature `dvc repro`);
  `calibrate`/`threshold` targets added; `test-models` file list fixed.
- **`configs/config.yaml`** — registers `calibration`/`threshold` Hydra defaults; adds
  `paths.figures`/`paths.policy`/`paths.costs_config`.
- **`PROJECT_PLAN.md`** — Phase 6 marked done.
- **Notebooks** (`00`, `01`, `02a`, `02b`, `03c`, `04`) — cross-references and "continue to"
  pointers completed end-to-end; stream-output fragmentation (a Windows/ipykernel artifact)
  normalized via `scripts/fix_notebook_outputs.py`.

---

## [0.5.4] - 2026-07-11 — Phase 5→6 Bridge: Registration Boundary Remediation

*Phase 5 shipped before the registry boundary in `CLAUDE.md` § "Run artifacts vs. registry
versions" was written, and had registered the uncalibrated pipeline as `challenger` — not a valid
rollback target under that boundary. This bridge removes registration from Phase 5 (deferred to
Phase 6's `calibrate.py`), fixes two defects surfaced in the same code paths, and rebuilds the
pipeline from a hard reset (`mlruns/` and the Postgres volume both cleared) to prove it still
reproduces from zero — `mlruns/1/` went from 223 leaked/historical run directories to 3.*

### Changed
- **`src/telco_churn/models/train/log_model.py`** (replaces `registration.py`) —
  `run_model_logging_step` logs the tuned `Pipeline` onto the `tuning_study` run without
  `registered_model_name=` or a `challenger` alias; adds `training_manifest["logged_model_uri"]`
  as the permanent handle Phase 6 resolves by.
- **`tests/unit/test_train_log_model.py`** (renamed from `test_train_registration.py`) — asserts
  `search_registered_models()` stays empty after Step 5, and that the logged signature declares a
  float output.
- **`ANALYSIS.md`** §4c "Registration — `challenger`" renamed to "Model logging (not registered)";
  **`notebooks/03c-hyperparameter-tuning.ipynb`** — registration narrative and the two cells that
  resolved runs via the `challenger` alias updated to resolve by run recency/tag instead.
- **`PROJECT_PLAN.md`** Phase 5 deliverable updated to log-only, no registration.

### Fixed
- **`models/train/common.py::_resolve_tracking_uri`** — a bare relative tracking URI (the default
  on any fresh clone with no `.env`) resolved to a Windows path MLflow's store registry rejected
  (`urlparse` read the drive letter as scheme `'c'`); now anchored via `.as_uri()`.
- **MLflow test-fixture artifact leak** — five unit-test fixtures plus the subprocess integration
  test pointed the tracking URI at a tmp SQLite store but left the artifact root defaulting to
  `./mlruns`, leaking run directories into the repo on every test run. Fixed via a shared
  `conftest.py` fixture with an explicit `artifact_location`.
- **Logged model signature declared `predict()` output (int labels), not `predict_proba()`** —
  `pyfunc_predict_fn` was left at its default, so the pyfunc flavour served argmax labels while the
  signature described probabilities; now explicit.

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

*A 50-trial TPE Optuna study tunes LightGBM on the frozen feature set (single stratified 5-fold,
`average_precision`; `n_estimators` resolved per fold by early stopping). The 1-SE rule selects a
more-regularized trial than raw-best: CV PR-AUC rises from 0.6582 (reduced set, default
hyperparameters) to 0.6659 while the train-vs-CV gap narrows from 0.1362 to 0.0328. The tuned
pipeline is registered as `telco-churn-pipeline` / `challenger` — uncalibrated and un-thresholded,
not serving-ready (Phase 6/7).*

### Added
- **`src/telco_churn/models/train/tuning.py`** — Step 4: `run_tuning_step`, a Postgres-backed
  Optuna TPE study with content-addressed study naming (data hash + committed features + search
  space + CV scheme, so an incompatible trial pool can never mix silently).
- `select_best_trial` (`argmax`/`1se`) and `boundary_hit_check`; nested per-trial MLflow child
  runs under one `tuning_study` parent run.
- **`src/telco_churn/models/train/registration.py`** — Step 5: `run_registration_step` refits the
  selected trial on all of development, logs the pipeline as pyfunc, asserts a log→reload→predict
  parity check, and registers `telco-churn-pipeline` / alias `challenger`.
- **`configs/tuning/optuna.yaml`** — search space and study knobs (`n_trials=50`, `cv_folds=5`,
  `selection_rule=1se`, `pruner=median`, `sampler_seed=42`).
- **`training_manifest.json`** (logged at registration) — git SHA, DVC data hash, hyperparameters,
  CV PR-AUC, paired-Δ vs. LogReg, and `tuning_summary`.
- **`tests/unit/test_train_tuning.py`** (24), **`test_train_registration.py`** (4) — 1-SE
  selection-rule branches, boundary-hit warnings, idempotent re-run, reload-parity failure path.
- **`tests/integration/test_train_subprocess.py`** (2) — subprocess test of the full Steps 1-5
  composition path.

### Changed
- **`ANALYSIS.md`** §4c — Optuna study result recorded: selected trial (1-SE rule) vs. raw-best;
  boundary-hit check clears on all 8 tuned hyperparameters; CV PR-AUC 0.6582 → 0.6659, train–CV gap
  0.1362 → 0.0328. Registration result recorded: `telco-churn-pipeline` version 1, alias
  `challenger`, `n_estimators=59` fixed on the full-development refit.

---

## [0.5.1] - 2026-07-03 — Phase 5 Step 3: Permutation-Importance Feature Selection

*Replaces Step 3's gain-based null-importance selector with permutation importance measured
against a synthetic noise-decoy column — the same rule Phase 4 discovery's Screen 4 already uses,
now shared end-to-end and model-agnostic by construction, unlike gain. Adds a non-gating SHAP
audit over the full feature space (flagging any dropped feature that would outrank a kept one),
and replaces the keep-vs-reduce adoption test with a paired-bootstrap test mirroring Step 2's
model-family decision, rather than an unpaired interval-containment check. Re-running Step 3
against real data: the full 20-feature set is retained, and decisively so under the paired test
(Δ = 0.0173, 95% CI [0.0104, 0.0246], `material_full_win`) — sharper than the near-miss the
earlier unpaired test reported for the same underlying gap.*

### Added
- **`src/telco_churn/features/select.py`** — `PermutationImportanceSelector` replaces
  `NullImportanceSelector` (grouped permutation importance vs. a synthetic decoy column);
  `compute_shap_audit()` adds a non-gating SHAP diagnostic; `reduced_set_bootstrap_test()`
  replaces the unpaired within-CI adoption check with a paired-bootstrap test.
- **`shap`** added as a project dependency (`pyproject.toml`).

### Changed
- **`src/telco_churn/models/train/feature_freeze.py`** — calls the new SHAP audit after minting
  the committed list; adopts the reduced set by default, overriding to full only on
  `material_full_win`.
- **`configs/training/selection.yaml`** — `n_permutations`/`cutoff_percentile` renamed to
  `n_repeats`/`noise_floor_margin`; adds its own `inner_val_size` key.
- **`tests/unit/test_select.py`** — rewritten onto the new API; adds permutation-importance,
  SHAP-audit, and bootstrap-test coverage.
- **`ANALYSIS.md`** §4 — rewritten with the real re-run result: full 20-feature set retained
  (Δ = 0.0173, CI [0.0104, 0.0246]); flags §5's Optuna results as stale pending re-run.
- **`notebooks/03b-feature-selection.ipynb`** — rewritten to render the paired-bootstrap test
  and permutation-importance/SHAP tables; re-executed against the real MLflow run.
- **`PROJECT_PLAN.md`**, **`docs/phase-5-tasks.md`** — Step 3 description and adoption test
  rewritten for the new method.

### Fixed
- **`tests/unit/test_train_common.py::test_compose_config_loads_expected_structure`** — asserted
  the old `selection.n_permutations`/`cutoff_percentile` config keys; updated to the renamed keys.

---

## [0.5.0] - 2026-07-03 — Phase 5 Steps 1-2: Model Selection (Candidate Comparison & Diagnostics)

*Candidate bake-off — `DummyClassifier(strategy='prior')`, `LogisticRegressionCV`, and default-config
LightGBM — on one shared `RepeatedStratifiedKFold` over the canonical dev split. LightGBM is
adopted as the modelling family under the pre-registered paired-bootstrap decision rule:
Δ = +0.0071 PR-AUC, 95% CI [+0.0029, +0.0113], clears the Δ*=0.005 materiality threshold
(`material_lgbm_win`). Non-gating fixed-recall and per-segment robustness/fairness diagnostics are
logged alongside the decision — they flag concerns but never decide the family (CLAUDE.md's
one-metric invariant).*

### Added
- **`src/telco_churn/features/schema.py`** — `FeatureSchema` frozen dataclass replacing the bare
  `list[str]` column-group constants previously in `features/build.py`.
- **`src/telco_churn/features/preprocessing.py`** — `build_linear_preprocessor` (OHE + scaling,
  tenure-cohort binning) for the `DummyClassifier`/`LogisticRegressionCV` baselines.
- **`src/telco_churn/models/train/common.py`** — shared helpers reused by every training step:
  CV scoring, default hyperparameters, dev-split loading, tracking-URI/git/DVC resolution.
- **`src/telco_churn/models/train/candidates.py`** — Step 1: CV-scores the three candidates on
  one shared `RepeatedStratifiedKFold(10×10)`; hard-assertion leakage canary on the dummy
  candidate; each candidate logged as its own MLflow run.
- **`src/telco_churn/models/train/comparison.py`** — Step 2: `bootstrap_comparison`, a paired
  bootstrap on Δ = AP(LGBM) − AP(LogReg) under the pre-registered `Δ*=0.005` decision rule.
- `run_diagnostics_step` — fixed-recall precision/F1 profile and per-segment robustness/fairness
  flags; both logged but never gating.
- **`src/telco_churn/models/diagnostics.py`** — pure helpers: `fixed_recall_profile`,
  `segment_oof_errors`, `segment_bootstrap_delta`, `generalization_gap`, `learning_curve_points`.
- **`src/telco_churn/models/train/__main__.py`** — CLI entry point; runs Steps 1-2 (3-5 once the
  family is confirmed), exits 1 on invalid data or a broken leakage canary.
- **`configs/training/lightgbm.yaml`**, **`configs/training/logreg.yaml`** — candidate
  hyperparameters plus shared LightGBM determinism/imbalance knobs.
- **`docker/mlflow/Dockerfile`**, **`sql/schema/000_create_mlflow_db.sql`**, `docker-compose.yml`
  `mlflow` service — Postgres-backed MLflow tracking server, started via
  `docker compose --profile infra up -d`.
- **`tests/unit/test_train_candidates.py`** (4), **`test_train_comparison.py`** (13),
  **`test_train_common.py`** (15), **`test_diagnostics.py`** (25) — leakage-canary, metric-
  logging, decision-rule, and segment-diagnostics coverage.

### Changed
- **`ANALYSIS.md`** — Step 2 model-selection result recorded: `material_lgbm_win`
  (Δ = +0.0071, 95% CI [+0.0029, +0.0113]).

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
[Unreleased]: https://github.com/Ampofowaa/TelcoChurn_PortfolioProject/compare/v0.7.2...HEAD
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
