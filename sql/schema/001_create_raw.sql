-- customers_raw: verbatim copy of the IBM Telco Customer Churn source data.
-- TotalCharges is NULLABLE — 11 zero-tenure customers have no first bill (whitespace in CSV).
-- churn is the binary-encoded target (0 = No, 1 = Yes); the raw "Churn" text column is dropped on ingest.
-- All column names are lowercased at ingest time to avoid quoted-identifier friction in SQL.
CREATE TABLE IF NOT EXISTS customers_raw (
    customerid        VARCHAR(20) PRIMARY KEY,
    gender            VARCHAR(10)  NOT NULL,
    seniorcitizen     SMALLINT     NOT NULL CHECK (seniorcitizen IN (0, 1)),
    has_partner       VARCHAR(3)   NOT NULL,
    dependents        VARCHAR(3)   NOT NULL,
    tenure            SMALLINT     NOT NULL CHECK (tenure >= 0),
    phoneservice      VARCHAR(3)   NOT NULL,
    multiplelines     VARCHAR(25)  NOT NULL,
    internetservice   VARCHAR(25)  NOT NULL,
    onlinesecurity    VARCHAR(25)  NOT NULL,
    onlinebackup      VARCHAR(25)  NOT NULL,
    deviceprotection  VARCHAR(25)  NOT NULL,
    techsupport       VARCHAR(25)  NOT NULL,
    streamingtv       VARCHAR(25)  NOT NULL,
    streamingmovies   VARCHAR(25)  NOT NULL,
    contract_type     VARCHAR(20)  NOT NULL,
    paperlessbilling  VARCHAR(3)   NOT NULL,
    paymentmethod     VARCHAR(45)  NOT NULL,
    monthlycharges    NUMERIC(8, 2)  NOT NULL CHECK (monthlycharges >= 0),
    totalcharges      NUMERIC(10, 2) NULL,
    churn             SMALLINT     NOT NULL CHECK (churn IN (0, 1))
);
