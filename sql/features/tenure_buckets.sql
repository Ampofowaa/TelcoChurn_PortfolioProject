-- Right-closed bands: tenure ≤ 12 → '0–12 mo', 13–24 → '13–24 mo', 25–48 → '25–48 mo', 49+ → '49+ mo'.
CREATE OR REPLACE VIEW tenure_buckets AS
SELECT
    customerid,
    tenure,
    CASE
        WHEN tenure <= 12 THEN '0–12 mo'
        WHEN tenure <= 24 THEN '13–24 mo'
        WHEN tenure <= 48 THEN '25–48 mo'
        ELSE                   '49+ mo'
    END AS tenure_cohort
FROM customers_raw;
