-- customers_crm: simulated "current" customer state, derived from customers_raw
-- by serving/crm_data.py. Gives GET /customer/{customerid} and /predict/batch's
-- ID-resolution path a source genuinely distinct from the frozen training
-- snapshot (customers_raw) — see prediction_logging_plan.md Part A.
-- No churn column: this table is never a label source, only a feature lookup.
CREATE TABLE IF NOT EXISTS customers_crm (
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
    crm_snapshot_at   TIMESTAMPTZ  NOT NULL
);
