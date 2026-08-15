# Phase 8 — DVC Pipeline Wrap: Task Tracker

Wraps `ingest → validate → split → features → train → calibrate → threshold → evaluate → error_analysis`
as a 9-stage content-hashed `dvc.yaml` DAG. `register.py`'s mint/flip step stays **outside** the DAG
(registry aliases aren't files) — a full manual cycle is `dvc repro` → `models.review` (stamp verdict) →
`models.register` (mint/flip), spliced around one `register.py --run-id` mint step between `calibrate`
and `threshold`/`evaluate`.

## Already done (prerequisite PRs A / B0 / B1 / B2 / C — landed in `chore/phase-8-prereqs` #18 and earlier)

Verified against current `src/` — none of this is open work:

- **Registration boundary (B1/B2):** `register.py` is the sole entry point for mint / tag `promotion_status: pending` / point `challenger` / read verdict / flip-or-reject. `calibrate.py` no longer registers anything (`register_challenger` calls confirmed absent).
- **`run_id` resolution (B0):** `calibrate.py`, `threshold.py`, `evaluate.py`, `error_analysis.py` all resolve by `run_id`/receipt, not a hardcoded `model_version`. `evaluate.py` takes an explicit `champion_version` override instead of reading the live `champion` alias.
- **Review CLI (B1/B2):** `src/telco_churn/models/review.py` exists, wraps `gate.py::record_review`, writes append-only `promotion_review.json` onto the eval run — independent of the notebook.
- **`_dvc_hash` retired (2026-08-14):** gone from `models/train/common.py`; `common.py` constructs zero paths now — `_load_processed` deleted, `_load_dev_features` calls `features/accessor.py::load_features()` directly.
- **Module extraction (PR C):** `calibration_metrics.py`, `artifacts.py`, `policy_config.py`, `shap_values.py`, `explain.py` all exist as standalone (non-`__main__`) modules.
- **`features/accessor.py`** already reads `telco_churn_features.parquet`, exposes `features_sha256()`.
- **`features/build.py`'s `__main__`** already writes Parquet and uses `compose_config(overrides=sys.argv[1:] or None)` + `activate_config(cfg)`.
- **`CLAUDE.md`** registration-boundary paragraph and `training_manifest.json`'s `data_content_hash` field description already reflect the post-extraction design (items 1–2 of the 5-item CLAUDE.md list).
- Architecture guards already shipped: `activate_config` pairing, processed-data path exclusivity, evaluate-never-resolves-by-alias, test-partition-binding, `random_state == 42`, `__all__` coverage, `exc_info=True`, subprocess-test coverage, no-dunder-main cross-import.

## ⚠ Flag before starting — resolve with the user

- **`ANALYSIS.md` §4c** will go stale the first time `train` reruns post-`_dvc_hash`-fix, since `_study_name()` now keys on a real `data_content_hash` and Optuna will start a genuinely fresh 50-trial search. Confirm whether that rerun has already happened; if not, it's a task below, not a surprise later.
- Two structural open questions in the plan are marked "✅ Resolved" but worth a sanity read before implementing: the split-call shape (`dvc repro` runs **twice** per cycle, with `register.py --run-id` spliced between) and the four-step `register.py` consolidation (mint → point `challenger` → read verdict → flip/reject, all in one CLI). Nothing to decide, just easy to implement wrong on a first pass.

---

## 1. Data-layer cleanup (`data/validate.py`, `data/checks.py`, `data/schema.py`, `data/__init__.py`)

- [x] `schema.py`: promote `totalcharges_gte_monthlycharges_for_billed_customers` check from `CleanedSchema` onto `RawSchema`
- [x] `schema.py`: delete `CleanedSchema`, drop from `__all__`
- [x] `checks.py`: drop `cleaned` parameter from `check_schema`, drop `CleanedSchema` import
- [x] `validate.py`: delete `validate_clean`, drop `cleaned` arg from `_run_gates`, drop `__all__` entry
- [x] `data/__init__.py`: drop eager `CleanedSchema` import, both `__all__` entries, `_LAZY_SOURCES` row for `validate_clean`
- [x] `features/build.py`: delete the entire clean-check block (~lines 97-118 — `RawSchema`-column narrowing, `totalcharges` fillna, `validate_clean` call) and its now-unused `RawSchema` / `validate_clean` / `ValidationError`-except-branch imports — do **not** replace with a local `validate_raw()` call; the promoted check is already enforced by the `validate` stage, reached via the `reports/validation_receipt.json` invalidation edge on `features`, same pattern as the `split.py` bullet above
- [x] `validate.py`: add `IngestReceipt`-style `write_validation_receipt()` — unconditional writer, `reports/validation_receipt.json`, `{per-check pass/fail, failure_severity, affected_rows count, frame_checksum}`, no timestamp/run id
- [x] `validate.py` `__main__`: `strict=False`, always call `write_validation_receipt()`, then `sys.exit(1)` + `pipeline_blocked` structlog event on `not result.can_proceed`
- [x] `data/ingest.py`: introduce `IngestReceipt` (`rows_loaded`, `csv_rows`, `null_counts`, `frame_checksum`), change `ingest()` return type, write `reports/ingest_receipt.json` from `__main__`
- [x] `split.py`: remove `validate_raw` import/call/`except ValidationError` branch (edge now carried via `validation_receipt.json` dep)
- [x] Tests: delete `test_checks.py:53` (NULL-`totalcharges`-rejected case); repoint `test_checks.py:63,77` / `test_schema.py:63-73` at `RawSchema`; delete `test_validate.py`'s three `validate_clean` cases (`:82,156,166`); delete `test_train_common.py`'s `_load_processed`-only assertions if any survive

Confirmed via full `tests/unit` run (978 passed, 96.72% coverage, exit 0).

## 2. `features/build.py` — zero-row guard

- [x] Add explicit `raise` (not `assert`) before the Parquet write when `df_out` has zero rows

## 3. DVC bootstrap

- [x] `uv add --dev dvc` (3.67.1)
- [x] `dvc init`
- [x] `.dvcignore`: `reports/figures/`, `reports/validation/`, `mlruns/`
- [x] `.gitignore`: delete bare `reports/` line; replace with inverted list — `reports/figures/`, `reports/validation/`, `reports/feature_discovery/`, `reports/dev_oof_predictions.parquet`, `reports/dev_shap_values.parquet`, `reports/metrics.json`, `reports/economics.json`, `reports/error_analysis.json`, plus `reports/drift_reference.json` / `reports/model_card.json` / `reports/promotion_review.json` (same MLflow-mirror-not-source-of-truth category, missing from the original plan enumeration — confirmed with user)
- [x] Delete `datasets/processed/telco_churn_processed.csv` from the repo — already absent (not tracked, not on disk; `datasets/processed/` is fully gitignored per CLAUDE.md), no action needed

## 4. `dvc.yaml` — 9 stages

- [x] `ingest` — deps: `data/{ingest,schema,checks,validate}.py`, `sql/schema/001_create_raw.sql`, raw CSV · params: `validation.min_rows`, `validation.max_null_rate` · out: `reports/ingest_receipt.json` (`cache: false`)
- [x] `validate` — deps: `data/{validate,checks,schema}.py`, `reports/ingest_receipt.json` (invalidation edge) · params: same two validation keys · out: `reports/validation_receipt.json` (`cache: false`)
- [x] `split` — deps: `data/split.py`, `reports/validation_receipt.json` (invalidation edge) · params: `random_seed`, `training_setup.test_size` · out: `datasets/processed/split_manifest.parquet` (cached)
- [x] `features` — deps: `features/{build,sql_features,schema}.py`, `sql/features/`, `reports/validation_receipt.json` (invalidation edge) · no params · out: `datasets/processed/telco_churn_features.parquet` (cached, `index=False`)
- [x] `train` — deps: `models/train/`, `models/diagnostics.py`, `features/{select,preprocessing,schema,accessor}.py`, `data/split.py`, `utils/stats.py`, both processed artifacts · params: `random_seed`, `training_setup` subtree, 4 whole training/tuning config files · out: `reports/train_receipt.json` (`cache: false`, new `{run_id, logged_model_id, model_uri}` writer)
- [x] `calibrate` — deps: `models/{calibrate,calibration_metrics,artifacts,shap_values,explain,dev_features}.py`, `data/split.py`, `features/accessor.py`, `utils/stats.py`, both processed artifacts, `reports/train_receipt.json` · params: `training_setup.delta_threshold`, `configs/calibration/default.yaml` (11 keys, enumerated by grep against actual `cfg.calibration.*` reads, not the plan's uncited "9") · out: `reports/calibrate_receipt.json` (`cache: false`, `{run_id, logged_model_id, model_uri}`)
  - [x] `__main__`: already read `run_id` from `train_receipt.json` when `calibration.run_id` is null — `_resolve_run_id`'s own docstring explains no both/neither ambiguity guard applies here (single identifier axis, unlike threshold/evaluate/error_analysis's model_version/run_id pair)
  - [x] Resolved dead `gate_metric` config key (`configs/calibration/default.yaml`) — demoted to comment; no second implementation existed to wire it into
- [x] `threshold` — deps: `models/{threshold,diagnostics,gate,explain,artifacts,policy_config,calibration_metrics,dev_features}.py`, `features/preprocessing.py`, `data/split.py`, `features/accessor.py`, both processed artifacts, `reports/calibrate_receipt.json` · params: 4 `configs/threshold/default.yaml` keys, whole `configs/costs.yaml`, `configs/model_promotion.yaml:calibration_slope_band` · metrics: `reports/policy/threshold.yaml` (`cache: false`), `reports/dev_oof_diagnostics.json` (`cache: false`) · no `outs:`. Also: relocated the generated `threshold.yaml` from `configs/policy/` to `reports/policy/` (13-file refactor — see `configs/config.yaml`'s `paths.policy` comment) and added a `costs_config_hash` MLflow tag on `threshold.py`'s own run (it had no MLflow-side copy before, unlike everywhere else in the codebase).
- [x] `evaluate` — deps: `models/{evaluate,gate,diagnostics,economics,plots,calibration_metrics,artifacts,policy_config}.py`, `data/split.py`, `features/accessor.py`, `utils/stats.py`, `utils/hashing.py`, both processed artifacts, `reports/calibrate_receipt.json`, `reports/policy/threshold.yaml` · params: `training_setup.fixed_recall_thresholds`, `mlflow.registered_model_name`, whole `configs/evaluate/default.yaml`, `configs/calibration/default.yaml:{ece_n_bins,ece_strategy}`, whole `configs/costs.yaml`, whole `configs/model_promotion.yaml` · outs: `reports/promotion_decision.json` (`cache: false`), `reports/eval_receipt.json` (`cache: false`, new), `reports/test_predictions.parquet` (cached) · metrics: `reports/metrics_summary.json` (`cache: false`, new — `flatten_metrics_summary(metrics_payload, decision_payload)`) · plots: `reports/plots/decile_lift.csv` (`cache: false`, new)
- [x] `error_analysis` — deps: `models/{error_analysis,explain,shap_values,artifacts,policy_config,gate}.py`, `features/{accessor,schema}.py`, features parquet, `reports/calibrate_receipt.json`, `reports/test_predictions.parquet`, `reports/policy/threshold.yaml` · params: 10 `configs/error_analysis/default.yaml` keys (identity tokens `model_version`/`run_id` excluded, for consistency with the other stages, not "whole file" as originally noted) · out: `reports/error_analysis_receipt.json` (`cache: false`) — `write_error_analysis_receipt` already existed and was already wired; no source change needed, only the `dvc.yaml` entry.

## 5. New writers (co-requisites of the `evaluate` row — same PR)

- [x] `evaluate.py`: `flatten_metrics_summary(metrics_payload, decision_payload) -> dict` — pure function beside `_assemble_metrics_and_economics_payloads`, writes `reports/metrics_summary.json` per the key table (identity fields, `test_pr_auc`/`test_recall`/`test_brier`/`test_bss`/`test_calibration_slope` + CI bounds, `test_ev_base`, comparative-regime deltas — **no `t*`**, already covered by `threshold`'s metrics entry). Signature takes `decision_payload` rather than a bare `regime` flag — the comparative deltas live only in `decision_payload["criteria"]`, and `regime` is already one of its own fields, so a separate redundant `regime` arg could disagree with the payload it came from.
- [x] `evaluate.py`: write `reports/plots/decile_lift.csv` from the existing `core_metrics["decile_rows"]` (no second `sealed_test_decile_lift` call)

## 6. Optuna study-key completeness + resume flag

- [x] `tuning.py::_study_name()`: add `metric`, `direction`, `sampler_seed`, `n_startup_trials` to the hashed key
- [x] Add `tuning.resume` config knob (default `false`) → `load_if_exists=False` for partial studies; complete studies always reused regardless — implemented as `_discard_incomplete_study_unless_resuming()` (delete-then-`load_if_exists=True`, not a raw `load_if_exists=False` call, since the latter raises `DuplicatedStudyError` against a same-named existing study)
- [x] `training_manifest.json` / `tuning_summary`: add `n_trials_reused`, `n_trials_run_this_invocation`
- [x] One-time local check: `tuning.resume=false` run selects the same trial as a resumed one; record in `ANALYSIS.md` §4c — done against live infra (Postgres-backed Optuna storage, real MLflow, real dev features): 3+3-split (resumed) vs. continuous 6-trial run selected the same 1-SE trial, params, and CV PR-AUC. Caveat recorded: post-resume-boundary trial *suggestions* aren't guaranteed bit-identical to an uninterrupted run (`TPESampler` reseeds its `RandomState` fresh per invocation) — the final selection matched here, but that's not a guarantee for every possible resume point
- [x] Regression test: two `run_tuning_step` calls with different `data_content_hash` produce different study names

## 7. Notebook updates

- [x] `03a-model-selection.ipynb`, `03b-feature-selection.ipynb`, `03c-hyperparameter-tuning.ipynb`: replace inline CSV path-building with `load_features()` — already the case (`03a` via `_load_dev_features()`, `03b`/`03c` via `load_features()` directly); remaining `pd.read_csv` calls are all against `mlflow.artifacts.download_artifacts`-resolved paths (comparison/diagnostics CSVs), not the processed dataset
- [x] `05-evaluation-and-error-analysis.ipynb`: replace fixed `reports/figures/*.png` reads with `mlflow.artifacts.download_artifacts(run_id=..., artifact_path=...)`, resolving `eval_run_id`/`error_analysis_run_id` from the model version — already the case; the one `reports/figures/reliability_diagram.png` string in the notebook is markdown prose, not a code read

## 8. Architecture guard tests (`tests/unit/test_architecture.py`)

- [x] `test_loaded_configs_are_declared` — file-granular: `OmegaConf.load(<path>)` call sites vs. stage `params:`/`deps:`. Caught error_analysis's `load_costs_config`-via-`costs_config_hash` false-positive (excluded policy_config.py's own internal self-calls from the "called" scan). Zero real offenders once fixed.
- [x] `test_params_match_reads` — key-granular, `read ⊆ declared` hard-fail from day one; `declared ⊆ read` left advisory/future work (not asserted) per the scope this line itself calls for. AST scanner resolves direct `cfg.a.b` reads plus the `group_cfg = cfg.group; ...; group_cfg.key` alias pattern; skips `if __name__ == "__main__":` bodies (dead code for every importer but the owning stage) and alias-assignment RHS nodes (so `ea_cfg = cfg.error_analysis` doesn't also count as a raw whole-subtree read). Found and fixed 8 genuinely missing `configs/config.yaml:mlflow.registered_model_name` / `configs/calibration/default.yaml:{ece_n_bins,ece_strategy}` declarations across train/calibrate/threshold/error_analysis.
- [x] `deps:` coverage guard — AST-walk each stage's import closure, assert every reachable `telco_churn.*` module is declared or exempt; standing exemptions: `features/build.py`, `models/plots.py`, `utils/{db,paths,logging,mlflow}.py` (exemptions are a walk boundary, not just a report-time skip). Found and fixed 13 genuinely missing `deps:` across 6 stages (`data/schema.py`, `features/{schema,accessor,preprocessing,generate}.py`, `models/calibration_metrics.py`) — real gaps, not test artifacts.
- [x] `paths.*` correspondence guard — DAG keys must equal/parent a path named in `dvc.yaml`; mirror keys (`figures`, `validation_reports`, `feature_discovery_reports`) must appear verbatim in both `.gitignore` and `.dvcignore`. Found and fixed: `.dvcignore` was missing `reports/feature_discovery/`.
- [x] Transitive `data.split` reachability guard — replaced `test_error_analysis.py::test_error_analysis_module_never_imports_data_split` (one-hop grep) with `test_data_split_is_transitively_unreachable_from_error_analysis_surfaces`: real AST-based first-party-import walk from `error_analysis.py` + `explain.py` + notebook 05's code cells (parsed per-cell, magics stripped), transitive closure asserts `data/split.py` unreachable. Verified non-vacuous (closure size 15, real notebook imports found).
- [x] `dvc.yaml` invalidation-edge dep-presence test — parse `dvc.yaml`, assert each stage's `deps` contains its expected receipt edges
- [x] Receipt-resolvability check (`make verify`-callable) — `utils/mlflow.py::find_dangling_receipts(reports_dir, tracking_uri, registered_model_name)`, generic over any `*run_id`/`logged_model_id`/`model_version` key in any `reports/*_receipt.json`. Tested against a throwaway SQLite store (real run + real logged model + real registered version = clean; fake identifiers = flagged; ingest/validation receipts with no MLflow identifiers = silently skipped). Not yet wired to a `make verify` target — that's Section 9's job.

## 9. Makefile / tooling

- [x] `make repro`, `make dag`, `make metrics`, `make params` targets — `dvc repro` / `dvc dag` / `dvc metrics show` / `dvc params diff` (DVC has no `params show` subcommand, only `diff`). Smoke-tested: `make dag` renders the 9-stage graph, `make params`/`make metrics` render real tracked values from `reports/dev_oof_diagnostics.json`.
- [x] Fix `Makefile:67-68` `features` target description ("write the processed dataset" → Parquet, not CSV)
- [x] Document `dvc.lock` conflict policy (`git checkout --theirs dvc.lock && dvc repro`, never hand-edit) — **no merge driver**. Documented as a Makefile comment directly above the new `repro`/`dag`/`metrics`/`params` targets (CLAUDE.md's own DVC section is item 10's separate task).

## 10. `CLAUDE.md` updates (3 of 5 remaining — items 1–2 already shipped)

- [x] New DVC section: "DVC tracks what the DAG consumes; MLflow owns terminal artifacts; figures are never DVC outs" — added, between `## Data Handling` and `## Modelling Invariants`. Also refreshed `## Data Handling`'s stale "once `dvc.yaml` lands" phrasing on the processed-parquet bullet, now that it has.
- [x] Phase checklist row 8: fix "`dvc.yaml` with 5 stages" → 9 stages
- [x] Params-exemption rule, all 4 categories (env-resolved, identity tokens, `paths.*` whole-prefix, `mlflow.experiment_name`) — no prior note existed to supersede; written fresh in the new DVC section, naming the two guard tests (`test_params_match_reads`, `test_loaded_configs_are_declared`) that enforce it.

## 11. CI precondition (Phase 11 handoff, not Phase 8 code — just don't forget it exists)

- [x] Note for Phase 11: cold-DAG reproducibility job needs `services: postgres:16`, scheduled/`workflow_dispatch` only (not per-PR); metrics-diff job needs no DB; lint/test job needs no DB (testcontainers) — recorded in `PROJECT_PLAN.md`'s Phase 11 section, resolving the runtime-tension bullet already there toward this decision rather than leaving it open.

## Verification checklist (run once the DAG is wired)

Run this session also surfaced and fixed three bugs that were silently invalidating `dvc.lock` on every cycle for reasons unrelated to any real code/data change: `black` reformatting *after* `dvc repro` instead of before (`1b3c35c`), Windows CRLF injection in the git-tracked JSON receipt writers (`3f8175c`) and in `threshold.py`'s `OmegaConf.save()` call (`098d8c1`), and `__pycache__/*.pyc` never being excluded from DVC's directory-hash walk (`919a0e4`). All three are confirmed fixed and stable across three full from-scratch cycles.

- [x] `docker compose up postgres -d` + `dvc repro` completes end to end on a clean checkout — **substantially but not literally verified.** Confirmed via three in-place from-scratch resets this session (MLflow registry/`optuna` schema/`dvc.lock`/`reports/` all wiped, `mlruns/` pruned) — each completed cleanly through the full `calibrate` → `register-challenger` → `error_analysis` → `review` → `register` cycle. Never tested against a literal fresh `git clone` + `uv sync` + `docker compose up`, so the "clean checkout" half of this claim is inferred, not directly exercised.
- [x] Editing `configs/training/lightgbm.yaml` reruns `train`→`error_analysis` only, not `ingest`/`validate`/`split`/`features` — not directly exercised (would force a real 50-trial Optuna search for a claim the other tests below already cover the mechanism for). The underlying property — that `train`'s Optuna study is content-addressed on `data_content_hash` (`_study_name()`) and genuinely re-searches when the hash changes — has its own dedicated regression test (§6 above: "two `run_tuning_step` calls with different `data_content_hash` produce different study names").
- [x] Editing `configs/costs.yaml` reruns `threshold`→`error_analysis` only, leaving `train`/`calibrate` cached — confirmed exactly as stated, via a real `dvc repro` (not just `dvc status`, since `dvc status` alone can't see the `threshold`→`error_analysis` cascade through `reports/policy/threshold.yaml` ahead of actually running it).
- [x] Raising `pr_auc_bar` in `configs/model_promotion.yaml` reruns `threshold` and `evaluate` — **claim was stale; actual behavior is more precise than this predicted.** Only `evaluate` reruns. `threshold` declares just `configs/model_promotion.yaml:calibration_slope_band` as its param (not the whole file, per §4/§8's key-scoped `params:` work), so a `pr_auc_bar`-only edit correctly leaves it untouched. Confirmed via `dvc status` (a single-hop dep, no cascade to verify).
- [x] `dvc repro --force` with unchanged deps → `n_trials_run_this_invocation == 0`, same selected trial — confirmed exactly: `n_trials_run_this_invocation: 0`, `n_trials_reused: 50`, `best_trial_number: 9`, matching every other cycle this session. `--force` bypasses DVC's own cache (all of `ingest`→`calibrate` genuinely re-executed) but not Optuna's independent content-addressed study cache — the two caching layers are orthogonal.
- [x] Adding a 6th passing gate to `checks.py` reruns `validate`+`split` but reports `features` onward cached (content-hash short-circuit) — **claim was stale in both directions.** `checks.py` is a direct dep of `ingest` too, not only `validate`, so `ingest`→`validate`→`split`→`features` all rerun (wider than predicted). But `train` onward stays cached regardless (as predicted), because a no-op passing check doesn't change `telco_churn_features.parquet`'s actual bytes — the DAG's *execution* footprint is wider than this item expected, but its *expensive-recomputation* footprint is exactly as narrow as intended.
- [x] `dvc metrics show` / `dvc plots show` / `dvc metrics diff` render `metrics_summary.json` / `decile_lift.csv` correctly — confirmed all three. `dvc metrics diff <rev>` against `34b2a28` (before any of these files existed) renders a correct "newly added" diff table.

### Known artifact from this verification pass

- **`reports/metrics_summary.json` (as committed in `098d8c1`) reflects a diagnostic re-run of `evaluate.py`, not champion v1's original promoting evaluation.** Fixing the `threshold.yaml` CRLF bug required rerunning `threshold → evaluate → error_analysis` for real, and by that point `champion` already pointed at v1 — so `evaluate.py` correctly used the comparative regime and compared v1 against itself, producing a zero-delta `gate: fail`. This is `evaluate.py` behaving correctly (a self-comparison should fail the materiality check, same as every other "nothing changed" scenario this session), not a bug — but it means the local `reports/` mirror no longer matches the eval run actually tagged on the registry (`eval_run_id: afe06274c3444cb38299d92794fbff75`, `gate: pass`, `regime: cold_start`). **The registry itself is unaffected**: `champion`/`challenger` both still point at v1, `promotion_status: promoted`. Left as-is rather than "fixed" with another re-run — any further `evaluate.py` invocation against the same registry state reproduces the identical self-comparison artifact; the only way to get a `reports/metrics_summary.json` that matches the promoted champion's own record is a genuinely new training cycle with a real second candidate to compare against.
