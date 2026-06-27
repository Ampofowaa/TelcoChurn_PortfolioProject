-- Final feature view: all raw columns plus the one SQL-derived feature.
-- has_partner and contract_type keep the ingest.py names (partner/contract were renamed
-- on load); build.py uses these names in its column lists.
-- LEFT JOIN: customers_raw is the authoritative row source. If a dependent view ever
-- filters rows, NULLs surface here and are caught by Pandera nullable=False in
-- build_feature_df — preferred over INNER JOIN silently dropping rows from training.
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
    cps.charge_per_service
FROM customers_raw           AS r
LEFT JOIN charge_per_service AS cps ON cps.customerid = r.customerid;
