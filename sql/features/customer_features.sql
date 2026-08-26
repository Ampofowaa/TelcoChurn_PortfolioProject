-- Final feature view: all raw columns plus the one SQL-derived feature.
-- has_partner and contract_type keep the ingest.py names (partner/contract were renamed
-- on load); build.py uses these names in its column lists.
-- LEFT JOIN: training_pool is the authoritative row source. If a dependent view ever
-- filters rows, NULLs surface here and are caught by Pandera nullable=False in
-- build_feature_df — preferred over INNER JOIN silently dropping rows from training.
--
-- Sourced from training_pool, not customers_raw (Phase 10a-ii) — the
-- unified retraining feed covering both the original CSV-seeded population
-- and every past/future reserve
-- cohort, so this view (and charge_per_service) is written once and never
-- needs a per-source duplicate. Joined on training_pool_id, never bare
-- customerid, since a customerid can legitimately recur (once from the
-- one-time seed, once more per matured reserve cohort). reserve_month is
-- exposed here so a caller can scope the read to a fold-forward cycle
-- (features/build.py's cycle-scoped query path) — features/build.py's
-- default v1/cold-start read still selects only reserve_month IS NULL, so
-- today's behavior is unchanged. training_pool_id/reserve_month are appended
-- last, not inserted before customerid — Postgres's CREATE OR REPLACE VIEW
-- only allows appending new columns, never reordering or inserting ahead of
-- the existing ones.
CREATE OR REPLACE VIEW customer_features AS
SELECT
    r.customerid,
    r.gender,
    r.seniorcitizen,
    r.has_partner,
    r.dependents,
    r.tenure,
    r.phoneservice,
    r.multiplelines,
    r.internetservice,
    r.onlinesecurity,
    r.onlinebackup,
    r.deviceprotection,
    r.techsupport,
    r.streamingtv,
    r.streamingmovies,
    r.contract_type,
    r.paperlessbilling,
    r.paymentmethod,
    r.monthlycharges,
    r.totalcharges,
    r.churn,  -- NULL for new customers in serving; training queries only.
    cps.charge_per_service,
    r.training_pool_id,
    r.reserve_month
FROM training_pool           AS r
LEFT JOIN charge_per_service AS cps ON cps.training_pool_id = r.training_pool_id;
