-- Active service count: nine binary flags summed to normalise monthlycharges.
--   CASE WHEN ... THEN 1 ELSE 0 END — portable across Postgres, DuckDB, SQLite.
--   phoneservice and multiplelines are separate line items: a customer with both
--   pays for two distinct services, so both contribute 1 to service_count (intended).
--   internetservice uses <> 'No' (not = 'Yes') so DSL and Fiber optic both count.
--   GREATEST(..., 1) mirrors .clip(lower=1) — prevents divide-by-zero for the
--   rare customer with no phone and no internet service.
CREATE OR REPLACE VIEW charge_per_service AS
SELECT
    customerid,
    monthlycharges / GREATEST(service_count, 1) AS charge_per_service
FROM (
    SELECT
        customerid,
        monthlycharges,
        (
            CASE WHEN phoneservice     = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN multiplelines    = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN internetservice <> 'No'  THEN 1 ELSE 0 END +
            CASE WHEN onlinesecurity   = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN onlinebackup     = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN deviceprotection = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN techsupport      = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN streamingtv      = 'Yes' THEN 1 ELSE 0 END +
            CASE WHEN streamingmovies  = 'Yes' THEN 1 ELSE 0 END
        ) AS service_count
    FROM customers_raw
) AS base;
