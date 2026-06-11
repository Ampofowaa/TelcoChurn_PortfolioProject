-- Right-closed bands matching pd.cut(bins=[0,12,24,48,72], right=True, include_lowest=True):
-- tenure=12 → '0–12 mo', tenure=13 → '13–24 mo'.
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
