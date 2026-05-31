-- customers_raw: verbatim copy of the IBM Telco Customer Churn source data.
-- TotalCharges is NULLABLE — 11 zero-tenure customers have no first bill (whitespace in CSV).
-- churn is the binary-encoded target (0 = No, 1 = Yes); the raw "Churn" text column is dropped on ingest.
-- All column names are lowercased at ingest time to avoid quoted-identifier friction in SQL.
CREATE TABLE IF NOT EXISTS customers_raw (
    customerid        VARCHAR(20) PRIMARY KEY,
    gender            VARCHAR(10),
    seniorcitizen     SMALLINT,
    has_partner       VARCHAR(3),
    dependents        VARCHAR(3),
    tenure            SMALLINT,
    phoneservice      VARCHAR(3),
    multiplelines     VARCHAR(25),
    internetservice   VARCHAR(25),
    onlinesecurity    VARCHAR(25),
    onlinebackup      VARCHAR(25),
    deviceprotection  VARCHAR(25),
    techsupport       VARCHAR(25),
    streamingtv       VARCHAR(25),
    streamingmovies   VARCHAR(25),
    contract_type     VARCHAR(20),
    paperlessbilling  VARCHAR(3),
    paymentmethod     VARCHAR(45),
    monthlycharges    NUMERIC(8, 2),
    totalcharges      NUMERIC(10, 2) NULL,
    churn             SMALLINT NOT NULL
);
