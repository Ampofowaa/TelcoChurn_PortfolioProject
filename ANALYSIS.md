# Analysis & Modelling Decisions

Full modelling rationale for the [Telco Customer Churn](README.md) portfolio project.
Covers problem framing, EDA, feature engineering, model selection, hyperparameter tuning,
error analysis, SHAP explainability, calibration, threshold optimisation, and business impact.

All analysis is reproducible from `notebooks/_archive/EDA-original.ipynb`.
The `src/` package implements the production-ready version of the same logic.

> *Section numbers follow the analytical lifecycle and do not correspond to implementation phase numbers in `PROJECT_PLAN.md`.*

---

## Table of Contents

0. [Problem Framing & Cost Definition](#0-problem-framing--cost-definition)
1. [Data Ingestion & Quality Checks](#1-data-ingestion--quality-checks)
2. [EDA & Statistical Testing](#2-eda--statistical-testing)
3. [Feature Engineering](#3-feature-engineering)
4. [Baseline Models](#4-baseline-models)
5. [Hyperparameter Tuning (Optuna)](#5-hyperparameter-tuning-optuna)
6. [Error Analysis](#6-error-analysis)
7. [SHAP Explainability](#7-shap-explainability)
8. [Probability Calibration](#8-probability-calibration)
9. [Business Impact & Threshold Selection](#9-business-impact--threshold-selection)
10. [Final Test-Set Results](#10-final-test-set-results)
11. [Production Refit & Model Registration](#11-production-refit--model-registration)
12. [Known Limitations](#12-known-limitations)
13. [Recommendations & Next Steps](#13-recommendations--next-steps)

---

## 0. Problem Framing & Cost Definition

This step locks the rules that govern every modelling decision in §§1–11. It is documented here — before any EDA or model results — so that the cost-sensitive threshold, the choice of recall as the primary metric, and the class imbalance handling all have a traceable origin.

### Prediction unit

A **single customer at a scoring cycle** (default: monthly). The model produces one churn-probability score per customer per run. All metrics (Recall, Precision, P&L) are computed per customer, not per account or household.

### Label definition and horizon

`Churn = 1` if the customer cancelled service **within the current billing cycle** (approximately 30 days). The label is derived directly from the IBM Telco dataset's `Churn` column (Yes/No, recoded to 1/0). It is a **binary, point-in-time label** — there is no survival horizon to tune and no soft-churn intermediate class in this dataset.

The label captures *revealed* churn (cancellation has occurred), not *predicted intent*. This means the model is trained to identify customers who have the same profile as those who eventually cancelled — not customers who are currently dissatisfied but have not yet acted.

### Decision the score feeds

The score feeds a **proactive retention intervention decision**: whether to include a customer in the upcoming outreach cycle (discount offer, contract upgrade, service credit). The decision is binary per customer per cycle. A contacted customer either receives an offer or does not; there is no tiered response modelled at this stage (tiered outreach is a §13 recommendation).

This framing makes the cost structure asymmetric and well-defined (see below). It has two metric consequences: **PR-AUC is the primary model selection and promotion metric** — it summarises precision-recall performance across all thresholds and is the promotion gate used in §§5 and 7. **Recall at the production threshold is the primary business metric** — once the model is deployed at the optimised production threshold (§9), the question the business asks is "how many churners did we catch this cycle?", and missed churners receive no offer and are lost.

### FN vs FP cost structure

| Error type | Business consequence | Base-scenario cost |
|---|---|---|
| **False Negative (FN)** | Churner not contacted — no retention offer issued. Customer cancels; LTV is lost. | ~$172 opportunity cost (30 % retention success × $575 LTV − $0 spend) |
| **False Positive (FP)** | Non-churner contacted unnecessarily. Retention offer issued at cost; customer was going to stay anyway. | ~$68 direct spend per intervention |

**FN/FP cost ratio ≈ 2.5:1** under the base scenario. Missing a churner costs approximately 2.5× more than a wasted offer. This ratio has two concrete consequences:

1. The optimal decision threshold is **shifted left of 0.5** — accepting more FPs to recover more churners is cost-rational.
2. **Recall is prioritised over Precision** as the primary metric. A model that catches fewer churners is not compensated by being more precise, because the asymmetry means false negatives are the more expensive error.

The cost parameters ($68/intervention, $575 LTV, 30 % success rate) are illustrative and derived from plausible telecom industry benchmarks. They are not Finance-validated; see §12 Known Limitation #8. The three-scenario bracket (Conservative / Base / Optimistic) in §9 exists specifically to stress-test decisions against this uncertainty.

### Baseline to beat

Two reference points bound the performance target:

| Baseline | Recall | Precision | Notes |
|---|---|---|---|
| Stratified random (population rate) | 26.5 % | 26.5 % | Floor — any useful model must beat this |
| Heuristic: flag all month-to-month customers | ~88 % | ~43 % | Cheap, non-ML rule; captures the dominant risk profile. Values are EDA-derived approximations (§1), not fitted model results. |

**Heuristic derivation:** month-to-month customers = 3,875; total churners = 1,869; month-to-month churn rate ≈ 43 % (§2 bivariate analysis) → ~1,666 churners captured. Recall = 1,666 / 1,869 ≈ **88 %**. Precision = 1,666 / 3,875 ≈ **43 %** (equals the segment churn rate by definition — flagging an entire group sets precision equal to that group's base rate).

The month-to-month heuristic sets a high recall bar but has two structural weaknesses. First, the ~12 % of churners it misses (~203 customers on annual or two-year contracts) are disproportionately long-tenure, higher-LTV customers — each FN in this segment costs more than the $172 average above, so the apparent 88 % recall understates the business cost of the misses. Second, the heuristic assigns equal weight to all 3,875 flagged customers; it cannot distinguish near-certain churners from borderline cases, making cost-efficient prioritisation of the outreach budget impossible. The ML model must therefore do more than match aggregate recall and precision — it must recover high-value churners the heuristic ignores and produce calibrated probability scores that enable threshold-optimised, cost-rational triage. This is also why the recall at the cost-optimised production threshold (§9) is set below 88 %: chasing the heuristic's recall ceiling requires flagging an ever-larger share of the customer base, flooding the outreach budget with low-confidence contacts. The economically rational operating point accepts a lower recall in exchange for materially higher precision and a better overall cost outcome.

### Success criterion

The model is considered fit for production if, at the cost-optimised threshold on the **sealed test set**:

| Criterion | Gate | Rationale |
|---|---|---|
| PR-AUC | ≥ 0.60 | Primary ranking metric — threshold-free and imbalance-appropriate at a 27 % positive rate; ROC-AUC is optimistic under class imbalance and is not used as a gate |
| Recall at the optimised production threshold | ≥ 0.75 | Primary business metric — proportion of churners caught at the deployed operating point |
| P&L vs random (base scenario) | Positive — model P&L > random-targeting P&L | Confirms the model adds economic value over an unguided contact budget |
| No test-set information used before final evaluation | Structural requirement — threshold derived from OOF predictions only | Preserves the "test set touched once" invariant (§9) |

These gates were set before the test set was opened. The final test-set results in §10 are evaluated against them exactly once.

---

## 1. Data Ingestion & Quality Checks

**Dataset:** 7,043 rows × 20 columns. Ingested from `datasets/raw/Telco-Customer-Churn.csv` into the `customers_raw` Postgres table via `ingest()`.

**Five automated gates — all PASS:**

| Gate | Severity | Check | Result |
|---|---|---|---|
| 1 | ERROR | Schema — column presence, types, value ranges, categoricals | PASS |
| 2 | ERROR | Duplicate `customerid` values | PASS — 0 duplicates |
| 3 | ERROR | Binary `churn` labels, no missing values | PASS |
| 4 | WARNING | Unexpected NULL `totalcharges` (non-zero tenure) | PASS — 0 unexpected NULLs |
| 5 | ERROR/WARNING | Row count ≥ 1,000; critical-column null rates ≤ 5 % | PASS — 7,043 rows; all rates 0.0 % |

ERROR gates block the Postgres load; WARNING gates emit a diagnostic but allow the pipeline to proceed.

**11 NULL `totalcharges` rows (0.16 %):** All have `tenure = 0` — brand-new customers whose first bill has not yet been generated. Gate 4 passes because all nulls are on zero-tenure rows (the expected structural case). Missing value treatment and the decision to retain these rows are discussed in §2.

The full gate walkthrough — rendered results and failure detail tables — is in [`notebooks/00-data-ingestion.ipynb`](notebooks/00-data-ingestion.ipynb).

---

## 2. EDA & Statistical Testing

### Key findings

- **Churn signal follows a clear hierarchy.** Contract type and tenure dominate; a cluster of service-related features (internet service, security/support add-ons, payment method, charges) contribute moderate signal; gender and phone service contribute nothing. A few features dominate, many contribute moderately, and a handful contribute nothing — but the distribution is not sparse enough to call concentrated.
- **The three leading signals share variance.** Contract type, tenure, and fiber optic internet service are correlated — month-to-month customers inherently have shorter tenure; fiber optic customers pay more and churn at a higher rate. In bivariate analysis each appears strong in isolation; in a fitted model they compete for the same variance.
- **The add-on service cluster signals disengagement, not causation.** All six add-on services are structurally tied to internet service — no-internet customers are coded "No internet service" across all six simultaneously. Their elevated churn correlation is partly structural and partly a disengagement signal: customers planning to leave do not invest in add-on services. Causality cannot be resolved from cross-sectional data alone.
- **Charge features move in opposite directions and must both be retained.** Higher monthly charges predict higher churn (fiber optic concentration); higher total charges predict lower churn (only long-tenure customers accumulate large totals). They carry independent information despite their correlation — only ~8.7 % of customers have `TotalCharges ≈ MonthlyCharges × tenure` exactly.
- **Class imbalance is moderate but consequential.** The dataset is 73.5 % No-churn / 26.5 % Churn (5,174 / 1,869; 2.77:1 ratio). A naive accuracy-maximising model achieves 73.5 % accuracy while identifying zero churners. **PR-AUC is the primary model selection and promotion metric** — more informative than ROC-AUC at this imbalance ratio; Recall at the deployed threshold is the primary business metric (see §0).

### Univariate distributions

**Numeric features:**
- `tenure` is bimodal (U-shaped): a spike at 0–5 months (new customers at highest inherent risk), a broad plateau from ~10–65 months (the stable retained cohort), and a second concentration at 65–72 months (long-term loyals) — consistent with a survival distribution. The 0–12 month cohort churns at ~47 %; by 49+ months that falls below 10 %.
- `monthlycharges` shows a two-tier structure: a sharp spike at $18–20 (basic phone-only plans), then a broad spread to $120 with density skewed toward $75–120 (bundled internet + add-on packages).
- `totalcharges` is right-skewed, with a long tail of high-value, long-tenure customers. Billing amounts shift over time for ~91 % of customers, so `totalcharges` carries signal independent of the other two numeric features.

**Categorical features:**
- **Demographics:** gender is near-balanced (~51 % male); senior citizens represent ~16 % of the base; customers with a partner or dependents are ~41 % and ~30 % respectively — the base skews toward younger, independent adults.
- **Services:** ~90 % have phone service; internet service splits across fiber optic (~44 %), DSL (~34 %), and no internet (~22 %). Security and support add-ons skew heavily toward "No" — the ~22 % without internet cannot subscribe, and month-to-month customers show lower uptake. Streaming add-ons are more evenly split.
- **Contract and billing:** month-to-month contracts dominate (~55 %); paperless billing is the majority preference (~59 %); electronic check is the most common payment method (~34 %).

**Outliers:**
All three numeric features have zero IQR-flagged outliers. The bounds are wide because the features span their full natural ranges — tenure 0–72 months (by contract design), monthlycharges ~$18–$119, totalcharges $0–$8,684. The right-skew in `totalcharges` reflects a genuine business pattern, not contamination. All values are retained.

**Categorical cardinality:**
Excluding `customerid` (7,043 unique values — one per customer), all 15 modelling features have 2–4 distinct values — low cardinality throughout, so one-hot encoding is safe with no dimensionality or sparsity concern.

### Missing values

Only `totalcharges` has nulls — 11 rows (0.16 %), all with `tenure = 0` (brand-new customers whose first bill had not yet been generated). All 11 are **non-churners**: a missingness indicator would carry no predictive signal because there are no churners in this group to separate from non-churners. The missingness is purely structural — determined by the billing cycle, not by customer behaviour or intent. **Decision: rows retained**; imputed via `SimpleImputer(strategy='median')` inside the training pipeline.

### Bivariate analysis & effect sizes

Chi-squared + Cramér's V for categorical features (V > 0.30 = strong, 0.10–0.30 = moderate, < 0.10 = weak); Mann-Whitney U + rank-biserial r for numeric features. With n = 7,043, p-values are near zero for any real effect — **effect size magnitude, not p-value, drives the ordering below.**

| Feature | Method | Effect size | Key finding |
|---|---|---|---|
| Contract type | Cramér's V | **0.41** (strong) | Month-to-month: ~43 % churn; One year: ~11 %; Two year: < 3 % |
| tenure | Rank-biserial r | **−0.48** | Churners average 18 months; non-churners 38 months |
| OnlineSecurity | Cramér's V | 0.35 (strong) | Churn ~2× higher without the service |
| TechSupport | Cramér's V | 0.34 (strong) | Churn ~2× higher without the service |
| InternetService | Cramér's V | 0.32 (strong) | Fiber optic carries disproportionately high churn |
| PaymentMethod | Cramér's V | 0.30 (strong) | Electronic check is the highest-churn payment method |
| TotalCharges | Rank-biserial r | −0.30 | Churners accumulate less ($1,532 vs $2,555) before leaving |
| MonthlyCharges | Rank-biserial r | +0.24 | Churners pay *more* per month ($74 vs $61) — fiber optic concentration |
| OnlineBackup | Cramér's V | 0.29 | Non-subscribers churn more; partly structural (internet cluster), partly disengagement |
| DeviceProtection | Cramér's V | 0.28 | Same pattern as OnlineBackup |
| PaperlessBilling | Cramér's V | 0.19 | Elevated churn — proxy for month-to-month and fiber optic mix, not a direct effect |
| Dependents | Cramér's V | 0.16 | Without dependents: above-average churn |
| SeniorCitizen | Cramér's V | 0.15 | ~42 % churn vs ~24 % — likely mediated by contract type and internet service |
| Partner | Cramér's V | 0.15 | Without a partner: above-average churn |
| MultipleLines | Cramér's V | 0.04 | Statistically significant but practically negligible |
| **PhoneService** | Cramér's V | **0.01** (p = 0.34) | **Non-predictor** |
| **Gender** | Cramér's V | **0.008** (p = 0.49) | **Non-predictor** |

**Interpretive notes:**

- **Fiber optic churn** operates through two overlapping channels: a cost channel (average $91.50/month vs $58.10 for DSL — a 57 % premium) and a potential service quality channel. The data cannot separate the two; feature importance in the modelling phase will clarify each channel's contribution.
- **Payment method and paperless billing reflect commitment depth, not direct drivers.** Automated payment methods (bank transfer, credit card) are associated with lower churn — consistent with greater friction to cancel. Paperless billing's elevated churn is a proxy for the month-to-month and fiber optic mix.
- **Senior citizen churn** (~42 % vs ~24 %) is likely mediated by contract type and internet service; the EDA cannot decompose the independent age contribution. No systematic age-based under-service is evident in the error analysis (§6).
- **`PhoneService_Yes`** has VIF = ∞ (perfectly collinear with `MultipleLines_No phone service`) yet V = 0.01 — the clearest illustration that VIF measures collinearity between features, not relevance to the target.

### Interaction effects

- **Contract type shows an asymmetric correlation with churn.** `contract_type_Month-to-month` correlates positively at r ≈ +0.40, while `contract_type_Two year` correlates negatively at only r ≈ −0.30 — lack of commitment pushes harder toward churn than long-term commitment protects against it.
- **Contract type × internet service compound interaction.** Within each internet tier, contract type dominates — fiber optic churn falls from 54.6 % (month-to-month) to 7.2 % (two-year), a 47 pp drop. Within each contract tier, fiber optic customers always churn more than DSL. The gap between fiber optic and DSL *narrows* as contracts lengthen (22 pp on month-to-month, 10 pp on one-year, 5 pp on two-year), confirming that contract lock-in partially suppresses the fiber optic risk premium. The highest-risk cohort is specifically *fiber optic + month-to-month*:

| Contract type | Fiber optic | DSL | No internet |
|---|---|---|---|
| Month-to-month | 54.6 % | 32.2 % | 18.9 % |
| One year | 19.3 % | 9.3 % | 2.5 % |
| Two year | 7.2 % | 1.9 % | 0.8 % |

### Feature correlations

Three key patterns emerge from the full encoded correlation matrix:

- **`TotalCharges` is the hub of the numeric feature cluster.** It correlates strongly with both `tenure` (longer-tenured customers accumulate more total charges) and `monthlycharges` (higher monthly spend compounds over time). As noted in Key findings, billing amounts shift over time for ~91 % of customers, so `totalcharges` retains independent predictive signal despite this correlation.
- **Add-on service features form a tight structural cluster.** OnlineSecurity, OnlineBackup, TechSupport, DeviceProtection, StreamingTV, and StreamingMovies correlate strongly with one another and with InternetService — no-internet customers receive "No internet service" across all six add-on columns simultaneously, making the cluster identification-equivalent rather than independently informative.
- **Most other feature pairs carry weak-to-moderate correlations**, indicating the remaining features contribute relatively independent signal.

### Multicollinearity (VIF)

VIF > 10 flags severe multicollinearity. **14 of 30 encoded features** exceed this threshold:

- **9 features return VIF = ∞:** `internetservice_No` and the six `_No internet service` add-on dummies are perfectly collinear — the same customers carry all seven simultaneously. `phoneservice_Yes` and `multiplelines_No phone service` form the second infinite cluster — customers without phone service are always coded "No phone service" across MultipleLines.
- **5 features return high finite VIF:** `monthlycharges` (≈ 866, driven by internet service tier and `totalcharges` accumulation), `internetservice_Fiber optic` (≈ 149), `streamingtv_Yes` and `streamingmovies_Yes` (≈ 24 each), and `totalcharges` (≈ 11).

The remaining 16 features all have VIF ≤ 10, with `seniorcitizen` (≈ 1.15) and `gender_Male` (1.00) effectively orthogonal to all other features.

**All features are retained — none are dropped on VIF grounds.** The practical impact of multicollinearity is model-family-dependent: tree-based methods (LightGBM, XGBoost, RandomForest) are immune — each split is evaluated independently and collinearity does not distort estimates. Linear models are materially affected and would require feature consolidation or regularisation. The null-importance experiment during model training (§5) is the explicit gate for any feature pruning.

---

## 3. Feature Engineering

Features are built in two layers: **SQL views in Postgres** (tenure bucketing, charge-per-service ratio) and **Python-engineered features** (four hypothesis-driven columns). The SQL layer handles transformations that are efficient in a relational database and reusable across any downstream query; the Python layer adds signals that require conditional logic or ratios across multiple raw columns. Together they produce a 25-column feature DataFrame passed to the training pipeline.

All features are derived from current account attributes available at CRM serving time — no historical aggregates, future data, or target-adjacent signals.

---

### SQL-engineered features

#### `tenure_cohort` — tenure bucketing

**What it is:** Each customer is assigned to one of four tenure bands — 0–12 mo, 13–24 mo, 25–48 mo, 49+ mo — based on how long they have been a customer.

**Why it matters:** Churn risk does not decline uniformly with each additional month of tenure — it drops in meaningful jumps at cohort boundaries. A customer who has just crossed into a later cohort behaves noticeably differently from one still in the previous group.

| Cohort | Customers | Churn rate |
|---|---|---|
| 0–12 mo | 2,186 | **47.4 %** |
| 13–24 mo | 1,024 | 28.7 % |
| 25–48 mo | 1,594 | 20.4 % |
| 49+ mo | 2,239 | **9.5 %** |

The 0–12 mo cohort churns at roughly **5× the rate** of the 49+ mo cohort.

**Design decision:** Both `tenure` (exact month count) and `tenure_cohort` (risk tier) are kept. Exact tenure tells the model *where* a customer sits within a group; the cohort label tells it *which risk tier* they belong to. The two carry complementary information and neither subsumes the other.

---

#### `charge_per_service` — monthly charge per active subscription

**What it is:** Each customer's monthly charges divided by the number of active service subscriptions they hold (phone, multiple lines, internet, and six add-on services — nine binary flags in total).

**Why it matters:** A raw monthly bill can be misleading. A customer paying $80/month across eight subscriptions is getting reasonable value per service; one paying the same for two is paying a steep price per subscription. `charge_per_service` puts all customers on a like-for-like basis.

The distribution splits into four natural clusters with a sharp churn step at the $24 threshold:

| Band | Avg services | Median monthly ($) | Churn rate |
|---|---|---|---|
| $10–16 | 5.6 | 69.80 | 15 % |
| $16–24 | 3.1 | 50.65 | 31 % |
| **$24–30** | **2.7** | **75.05** | **57 %** |
| $30+ | 2.0 | 70.05 | 59 % |

The step from 31 % to 57 % at the $24 mark is the sharpest in the distribution. Churners have a median of $18.94 per service versus $15.53 for non-churners — a signal that is not visible in raw `monthlycharges` alone because a high bill could simply mean the customer has many services rather than that they are overpaying for each one.

---

### Python-engineered features

Three hypothesis-driven features (H1–H3) derived from the EDA's identified blind spots, producing four engineered columns — H3 contributes two (`fiber_contract` and `dsl_contract`, one per internet tier). Each targets a segment where raw features send a misleading signal to the model.

---

#### H1: `is_long_month_to_month` — exit-barrier blind spot

**Hypothesis:** Customers on a month-to-month contract who have stayed past 24 months look loyal by tenure alone, but they have never committed to a longer term — raw tenure underestimates their churn risk.

**Evidence:** Month-to-month churn falls steadily across cohorts (51.4 % → 37.7 % → 32.9 % → 26.0 %) but never fully settles — even at 49+ months, MTM customers still churn at roughly the dataset average. By contrast, two-year contract customers show near-zero churn in the first two cohorts and remain below 4 % throughout.

| Cohort | MTM churn | Two-year churn | Gap (pp) |
|---|---|---|---|
| 0–12 mo | 51.4 % | 0.0 % | 51.4 |
| 13–24 mo | 37.7 % | 0.0 % | 37.7 |
| 25–48 mo | 32.9 % | 2.2 % | 30.7 |
| 49+ mo | 26.0 % | 3.3 % | 22.7 |

The gap narrows but never closes — confirming the interaction is real.

**Result:** 1,144 customers flagged (16.2 % of the dataset). Of those, **30.9 % churn** versus **25.7 %** for the rest — a **5.2 pp gap**. Roughly 1 in 3 customers who have been on a monthly contract for over two years will leave, a rate higher than the dataset average despite their long tenure.

---

#### H2: `monthly_to_total_ratio` — thin billing history

**Hypothesis:** `TotalCharges` grows automatically with tenure and is mainly a proxy for seniority. A low total could mean a loyal new customer or someone about to leave after a short stay — the two are indistinguishable from `TotalCharges` alone. Dividing `MonthlyCharges` by `TotalCharges` produces a ratio close to 1 for someone who just started (total ≈ one month's bill) and falling toward 0 as billing history accumulates. This separates the ambiguous low-`TotalCharges` group into customers who are simply new versus those whose account history is too thin to assess.

**Result:** Churners have a median ratio of **0.103** versus **0.027** for non-churners — roughly **4× higher**. Translating to approximate tenure (ratio ≈ 1/tenure): the median churner has been a customer for ~10 months; the median non-churner for ~37 months.

The churner population is a genuine mix: a large cluster of new customers at early-exit risk (>0.2 ratio), plus a meaningful long-tenure tail (<0.05 ratio) — customers who eventually leave after years of service. Non-churners concentrate heavily at the low end.

*Note: 11 zero-tenure rows have `TotalCharges = NULL` and therefore `monthly_to_total_ratio = NaN`. These are preserved in the feature DataFrame and imputed by `SimpleImputer(strategy='median')` in the training pipeline.*

---

#### H3: `fiber_contract` / `dsl_contract` — internet × contract interaction

**Hypothesis:** Fiber optic customers churn at the highest rate of any internet tier; month-to-month customers churn at the highest rate of any contract type. These two risk factors do not combine additively. A fiber optic customer on a two-year contract has made a commitment that partially offsets the fiber churn tendency; a fiber optic customer on a month-to-month plan has both risk factors at once with nothing to offset either. The individual `contract_type` and `internetservice` columns each carry this information separately but cannot represent when the two coincide.

**Evidence:** Churn rates vary substantially *within* each internet tier depending on contract type:

| Contract type | Fiber optic | DSL | No internet |
|---|---|---|---|
| Month-to-month | 54.6 % | 32.2 % | 18.9 % |
| One year | 19.3 % | 9.3 % | 2.5 % |
| Two year | 7.2 % | 1.9 % | 0.8 % |

Fiber optic spans **47 pp** across contract types; DSL spans **30 pp**. This within-tier variation cannot be recovered from the individual encoded columns alone.

**Design decision:** Two 4-level categoricals rather than binary MTM flags or a full 9-level crossing. Binary MTM flags would capture only the highest-risk cell in each tier and discard the remaining variation. The full 9-level crossing would encode all combinations, including the seven sitting at or below average, adding noise without signal. The chosen design creates one categorical per high-risk internet tier — `fiber_contract` takes values `Month-to-month_Fiber optic`, `One year_Fiber optic`, `Two year_Fiber optic`, and `Not Fiber optic`; `dsl_contract` mirrors this for DSL. Customers on no internet service are handled by the individual encoded columns; their churn rates vary by contract type (18.9 % month-to-month, 2.5 % one-year, 0.8 % two-year) but all fall below the dataset average, and the contract_type encoding already captures this variation.

---

### Feature inventory

| Group | Count | Examples |
|---|---|---|
| Binary (OHE drop-if-binary) | 7 | `gender`, `seniorcitizen`, `is_long_month_to_month` |
| Multi-category (OHE) | 13 | `contract_type`, `tenure_cohort`, `fiber_contract`, `dsl_contract` |
| Numeric (impute → scale) | 5 | `tenure`, `monthlycharges`, `charge_per_service`, `monthly_to_total_ratio` |
| **Total** | **25** | |

The `ColumnTransformer` definition and fitting are model training responsibilities, applied to the training split only to prevent test-set statistics from leaking into preprocessing parameters.

---

## 4. Baseline Models

Three models trained at minimum-viable-baseline settings (not tuned). Selection criterion: **5-fold stratified CV recall on the training set**. Val set used as diagnostic holdout only — never influenced model selection.

| Model | Val Precision | Val Recall | Val F1 | Val ROC-AUC | Val PR-AUC | CV Recall (5-fold) |
|---|---|---|---|---|---|---|
| LightGBM | 0.495 | **0.900** | 0.638 | **0.867** | **0.696** | **0.843 ± 0.011** |
| XGBoost | 0.519 | 0.813 | 0.634 | 0.810 | 0.592 | 0.766 ± 0.006 |
| RandomForest | 0.589 | 0.727 | 0.651 | 0.820 | 0.612 | 0.668 ± 0.021 |

**LightGBM selected** — leads on CV recall (0.843), val recall (0.900), ROC-AUC (0.867), and PR-AUC (0.696). PR-AUC separates models more cleanly than ROC-AUC at this class imbalance ratio.

**Why accuracy is a misleading metric here:** RandomForest achieved 79.3 % accuracy vs LightGBM's 72.9 %, but the gap is entirely driven by non-churner accuracy. At a 2.8:1 class imbalance, accuracy rewards over-predicting the majority class. Recall on the positive class is the primary metric.

### Pre-tuning diagnostics

| Metric | Value | Threshold | Flag |
|---|---|---|---|
| Train recall | 0.979 | — | — |
| CV recall | 0.843 | — | — |
| Train–CV gap | **+0.136** | 0.021 (= 2 × CV std) | **[OVERFIT]** |
| CV std | 0.011 | — | Stable, reproducible overfitting |

Clear overfitting at default settings, but CV std = 0.011 confirms it is consistent — making LightGBM an ideal Optuna regularisation target.

---

## 5. Hyperparameter Tuning (Optuna)

| Setting | Value |
|---|---|
| Algorithm | Tree-structured Parzen Estimator (TPE) |
| Trials | 50 |
| Objective | 5-fold CV recall on training set only (val/test never exposed) |
| Search space | `num_leaves`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree` |

**Best hyperparameters (MLflow run ef055674):**

```
colsample_bytree: 0.8125   max_depth: 3          learning_rate: 0.01369
min_child_samples: 43      reg_lambda: 4.167      n_estimators: 110
reg_alpha: 0.333           subsample: 0.5535      num_leaves: 133
```

### Post-tuning bias/variance check

| Model | Train Recall | CV Recall | Train–CV Gap | Flag |
|---|---|---|---|---|
| Baseline LightGBM | 0.979 | 0.843 | +0.136 | OVERFIT |
| **Tuned LightGBM** | **0.949** | **0.946** | **+0.003** | **OK** |

- Gap reduced by **~98 %** (+0.136 → +0.003), within the 0.021 noise floor.
- CV recall improved by **+0.103** (0.843 → 0.946).
- Learning curve: train and CV lines nearly parallel and flat. The model is not data-hungry and has found the correct regularisation operating point for this dataset size.

### Baseline vs tuned — val set (threshold = 0.35)

| Metric | Baseline | Tuned | Δ |
|---|---|---|---|
| Recall | 0.900 | **0.940** | +0.040 |
| Precision | 0.495 | 0.405 | −0.089 |
| F1 | 0.638 | 0.566 | −0.072 |
| ROC-AUC | 0.867 | 0.855 | −0.012 |
| TP / FN | 135 / 15 | 141 / 9 | +6 TP, −6 FN |
| FP | 138 | 207 | +69 |

**McNemar's test:** χ² = 48.66, p < 0.0001 — the improvement in error pattern is statistically significant. Discordant pairs: b₁₀ = 8 (tuned recovers churners baseline misses), b₀₁ = 71 (tuned over-flags non-churners baseline got right). The b₀₁ > b₁₀ asymmetry is the expected cost of recall optimisation at a shared threshold — not model degradation.

---

## 6. Error Analysis

All analysis is on the **tuned `best_pipe`** on the val set (threshold = 0.35). Calibration preserves rank order but cannot be used for SHAP (see §7).

### Missed churners — profile (9 FNs, 6.0 % FN rate)

| Dimension | Missed churners (FN) | Caught churners (TP) | Gap |
|---|---|---|---|
| Mean tenure | 42.22 months | 17.42 months | +24.80 months |
| Mean MonthlyCharges | $58.30 | $78.80 | −$20.50 |
| Mean TotalCharges | $2,932 | $1,536 | +$1,396 |

**Contract split:** All 9 missed churners hold non-month-to-month contracts (6 × One year, 3 × Two year, 0 × Month-to-month).

**Structural blind spot:** The dominant learned signal is *short tenure + high charges + month-to-month contract*. Customers who churn after years of lower-cost service break this pattern entirely. The model applies a near-irrefutable "committed customer" prior to annual-contract, long-tenure holders.

**Score distribution:**
- 6 near-miss FNs with scores 0.242–0.338 (just below the 0.35 threshold — a threshold problem)
- 3 deep-miss FNs with scores 0.148–0.184 (a feature problem — the model is actively confidently wrong)

### False positive profile (207 FPs, 50 % FP rate)

FPs have a nearly identical feature profile to actual churners: 80.7 % month-to-month, 53 % fiber optic, 69 % no online security. There is no clean spatial boundary separating FPs from true negatives — the model cannot distinguish the minority who share the risk profile but remain loyal. Resolving this requires loyalty or satisfaction signals not present in the current feature set.

### Subgroup FN rates

| Subgroup | FN rate | Interpretation |
|---|---|---|
| Contract = Two year | 1.000 | Near-total blind spot |
| Contract = One year | 0.538 | Major blind spot |
| tenure 49–72 months | Highest by band | Highest-LTV customers; hardest to anticipate |
| DSL / No internet | Above average | Model under-serves lower-cost segments |
| Fiber optic | Lowest | Classic risk profile — thoroughly learned |
| SeniorCitizen = 1 | Low (similar to non-seniors) | No systematic age-based under-service — positive fairness result |

---

## 7. SHAP Explainability

Using `shap.TreeExplainer` on `best_pipe` (exact tree values, not kernel approximation). Analysis conducted on both val and test sets; top-10 ranking is identical across both — confirming pipeline consistency.

### Global feature importance (mean |SHAP| — top 10)

| Rank | Feature | Direction | Key pattern |
|---|---|---|---|
| 1 | `Contract_Month-to-month` | Positive | Mean |SHAP| ~3× the next feature; drives churn for nearly all customers |
| 2 | `OnlineSecurity_No` | Positive | Bimodal — subscribers cluster left (protective), non-subscribers right (risky) |
| 3 | `InternetService_Fiber optic` | Positive | Compounding risk with MonthlyCharges confirmed |
| 4 | `tenure` | Negative | Clean monotone: short tenure → high risk; steeper effect for month-to-month (interaction) |
| 5 | `TechSupport_No` | Positive | Same bimodal pattern as OnlineSecurity |
| 6 | `MonthlyCharges` | Positive | Effect moderated by contract type and internet tier |
| 7 | `PaymentMethod_Electronic check` | Positive | Independent signal beyond contract type |
| 8 | `Contract_Two year` | Negative | Strong protective — near-irrefutable commitment signal |
| 9 | `StreamingMovies_Yes` | Positive | Small but consistent marginal value |
| 10 | `TotalCharges` | Negative | Retained despite high VIF — billing history non-redundancy confirmed |

**`gender` and `SeniorCitizen` rank #23 and #40 respectively** (mean |SHAP| ≈ 0.000). Churn scores are driven entirely by behavioral and contractual features, not demographic proxies. Safe to document in a model card as non-influential.

### Key interactions confirmed by dependence plots

- **tenure × Contract:** Month-to-month customers retain high positive SHAP even at moderate tenure — the contract type sustains churn risk regardless of seniority.
- **MonthlyCharges × Fiber optic:** Fiber optic customers show systematically higher SHAP at the same charge level — two risk factors compound rather than add.

### Individual explanations (waterfall plots)

**High-confidence churner (score = 0.790):** month-to-month + 1-month tenure + fiber optic + no online security + no tech support — every feature points in the same direction.

**High-confidence non-churner (score = 0.148):** Not month-to-month = largest single protective factor; reinforced by two-year contract, 70-month tenure, $19.80/month, security and tech support subscribed.

**Most convincingly missed churner (FN, score = 0.025):** Holds no month-to-month contract flag (SHAP −0.62); two-year contract (−0.13); tenure = 55 months (−0.07). The model's "annual contract = committed customer" heuristic overpowers all other signals.

### EDA vs SHAP agreement

| Feature | EDA rank | SHAP rank | Divergence |
|---|---|---|---|
| Contract | 1 | 1 | Agreement |
| tenure | 2 | 4 | OnlineSecurity ranks above tenure in SHAP — marginal contribution in conditional context |
| OnlineSecurity | 3 | 2 | Elevated vs bivariate rank |
| TechSupport | 4 | 5 | Agreement |
| InternetService | 5 | 3 | Agreement |
| gender | Non-significant | #23 (≈0) | Agreement |
| SeniorCitizen | Confounded | #40 (≈0) | Agreement — mediation confirmed |

---

## 8. Probability Calibration

Raw LightGBM probabilities are poorly calibrated — useful for ranking but not for expected-value calculations.

### Pre- vs post-calibration (val set)

| Metric | Uncalibrated (`best_pipe`) | Calibrated (`cal_pipe`) | Change |
|---|---|---|---|
| Brier Score | 0.1671 | **0.1317** | −0.0355 (−21 %) |
| Brier Skill Score (BSS) | 0.1438 | **0.3254** | +0.1816 |
| ROC-AUC | 0.855 | 0.855 | 0.000 — rank order preserved |

**Method:** `CalibratedClassifierCV(best_pipe, method='sigmoid', cv=5)`. Sigmoid chosen over isotonic: each calibration fold has only ~1,014 OOF points, at which isotonic produces a zigzag reliability curve. Sigmoid fits two parameters and is immune to this sparsity artifact.

BSS rises from 0.14 (below the 0.20 meaningful-calibration guideline) to 0.33 — the calibrated model explains 33 % of available calibration skill above the naive baseline.

**Residual calibration gap:** Slight over-prediction in the 0.22–0.45 range; more pronounced under-prediction above 0.58 (actual rates ~0.60–0.80 exceed predicted scores by up to ~10 pp). This gap cannot be closed with post-hoc calibration alone — it requires richer features or additional data.

**`cal_pipe` is used for all subsequent scoring.** `best_pipe` is retained for SHAP only, as `CalibratedClassifierCV` wraps an ensemble of 5 internal clones that `TreeExplainer` cannot cleanly penetrate.

### Bootstrap confidence intervals (1,000 resamples, val set)

| Metric | Point Estimate | 95 % CI | Width |
|---|---|---|---|
| Recall | 0.760 | [0.687, 0.822] | 0.135 |
| Precision | 0.576 | [0.505, 0.644] | 0.139 |
| F1 | 0.655 | [0.593, 0.708] | 0.115 |
| ROC-AUC | **0.856** | [0.820, 0.888] | 0.068 |
| PR-AUC | 0.684 | [0.602, 0.764] | 0.162 |

**SLA-safe lower bounds (95 % confidence):** Recall ≥ 0.687, ROC-AUC ≥ 0.820.

---

## 9. Business Impact & Threshold Selection

### Business cost parameters (illustrative — replace with Finance actuals)

Three scenarios bracket realistic business assumptions:

| Parameter | Conservative | Base | Optimistic |
|---|---|---|---|
| ARPU | $55.90 | $79.90 | $94.40 |
| 1-year LTV | $402 | $575 | $680 |
| Retention success rate | 20 % | 30 % | 40 % |
| Cost per intervention | $22 | $68 | $135 |
| Value per TP | $58 | $104 | $137 |
| Opportunity cost per FN | $80 | $172 | $272 |

### Production threshold derivation

The val-set cost minimum is **not** used as the production threshold — it would leak val-set information. The threshold is derived from **OOF predictions on the training set** (5-fold CV with `cal_pipe`), eliminating all leakage.

| Method | Threshold | Rationale |
|---|---|---|
| Val-set cost min | ~0.03 | Diagnostic only — not shipped |
| **OOF cost min (production)** | **0.2956** | No val/test leakage — this is shipped |

The OOF and val cost curves have minima at consistent thresholds (~0.03–0.36), confirming no distribution shift between training and validation populations.

**OOF threshold results by scenario (val set diagnostic):**

| Scenario | Threshold | Recall | Precision | F1 | TP | FP | FN | P&L (val) |
|---|---|---|---|---|---|---|---|---|
| Conservative | 0.2464 | 0.840 | 0.512 | 0.636 | 126 | 120 | 24 | $4,718 |
| **Base (production)** | **0.2956** | **0.800** | **0.531** | **0.638** | **120** | **106** | **30** | **$5,332** |
| Optimistic | 0.3596 | 0.760 | 0.588 | 0.663 | 114 | 80 | 36 | $4,818 |

### Lift analysis (val set)

| Population | Churners captured | Lift | Notes |
|---|---|---|---|
| Top 10 % by score | 30.7 % of all churners | **3.03×** | Highest-density operating point |
| Top 20 % | 52.7 % | 2.22× | Half of all churners at one-fifth of budget |
| Top 30 % (≈ KS point) | 70.0 % | 1.75× | Max separation point |
| Production threshold (~40 %) | 80.0 % | ~1.0× | Marginal lift near random at boundary |
| Bottom 50 % | Below baseline | < 1.0× | De-prioritise — expected negative ROI |

---

## 10. Final Test-Set Results

One-time evaluation on the sealed test set (n = 1,409; 374 churners). No modelling decisions were made after seeing these numbers.

### Classification metrics

| Metric | Value |
|---|---|
| Production threshold | 0.2956 (OOF cost-optimised) |
| **Recall** | **0.7861** |
| **Precision** | 0.5241 |
| **F1** | 0.6289 |
| **ROC-AUC** | **0.8413** |
| PR-AUC | 0.6511 |
| Brier Score | 0.1383 |
| Brier Skill Score | 0.2907 |
| KS | 0.5310 at 39.6 % of ranked population |

**Confusion matrix (threshold = 0.2956, n = 1,409):**

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | TN = 768 | FP = 267 |
| **Actual Yes** | FN = 80 | TP = 294 |

561 customers contacted (39.8 % of test set).

### Train / val / test metric trajectory

| Split | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Train | 0.8015 | 0.5264 | 0.6354 | 0.8508 | 0.6638 |
| Val | 0.8000 | 0.5310 | 0.6383 | 0.8560 | 0.6845 |
| **Test** | **0.7861** | **0.5241** | **0.6289** | **0.8413** | **0.6511** |
| Val → Test Δ | −0.0139 | −0.0069 | −0.0094 | −0.0147 | −0.0334 |

All metrics decline uniformly Val → Test; no rank-order reversals; Train ≈ Val confirms overfitting remediation was successful.

### Lift & gains (test set — definitive)

| Decile | Churn rate | Lift | Cumulative gain |
|---|---|---|---|
| Top 10 % | 72.3 % | **2.73×** | 27.3 % of all churners |
| Top 20 % | 61.7 % | 2.32× | 50.5 % |
| Top 30 % | 43.3 % | 1.63× | 66.8 % |
| Top 40 % (production budget) | 31.2 % | 1.18× | 78.6 % |
| Deciles 5–10 | Declining | < 1.0× | Below baseline |

The model delivers **+27 percentage-point improvement** over random targeting at the same 40 % budget.

### Business impact (test set)

**Model vs random (same 561-contact budget):**
- Random targeting would recover 149 churners by chance (base rate × budget)
- Model recovers 294 — **+145 more (+97 %)**

**P&L by scenario:**

| Scenario | Model P&L | Random P&L | P&L Uplift |
|---|---|---|---|
| Conservative | $11,296 | −$362 | **$11,658** |
| Base | $12,567 | −$12,446 | **$25,012** |
| Optimistic | $4,233 | −$35,207 | **$39,440** |

> Note: Optimistic P&L is lower than Base despite higher LTV because $135 per contact × 267 FPs = $36,045 in outreach spend nearly offsets the gains. Intervention cost is the binding constraint in the high-cost scenario.

**Annualised projection (scaled to full 7,045-customer base, 12 cycles/year):**

| Scenario | Per run (full base) | Annual |
|---|---|---|
| Conservative | $56,462 | **$677,544** |
| Base | $62,817 | **$753,806** |
| Optimistic | $21,159 | **$253,908** |

### Segment breakdown (test set)

**By contract type:**

| Contract | n | n_churn | Recall | FP rate |
|---|---|---|---|---|
| Month-to-month | 773 | 329 | 0.891 | 0.601 |
| **One year** | **300** | **36** | **0.028** | **0.000** |
| **Two year** | **336** | **9** | **0.000** | **0.000** |

**By internet service:**

| InternetService | n | n_churn | Recall | FP rate |
|---|---|---|---|---|
| Fiber optic | 613 | 252 | 0.885 | 0.540 |
| DSL | 484 | 97 | 0.619 | 0.129 |
| None | 312 | 25 | 0.440 | 0.077 |

**By tenure band:**

| Tenure | n | n_churn | Recall | FP rate |
|---|---|---|---|---|
| 0–12 months | 449 | 215 | 0.884 | 0.543 |
| 13–24 months | 206 | 60 | 0.850 | 0.384 |
| **25+ months** | **754** | **99** | **0.535** | **0.128** |

### SHAP on the test set

SHAP is applied to the LightGBM base estimator extracted from fold 0 of `cal_pipe`. Top-10 feature ranking on the sealed test set is **identical** to the validation-set ranking — zero rank shifts across the top 10. SHAP base value: **−0.3009** (log-odds space).

**Protected attributes:**

| Attribute | Mean \|SHAP\| | Rank of 40 | Assessment |
|---|---|---|---|
| `Gender: Male` | 0.0000 | #23 | Negligible |
| `Partner: Yes` | 0.0000 | #21 | Negligible |
| `SeniorCitizen` | 0.0000 | #40 | Negligible (last) |

All three protected attributes contribute zero marginal signal at the individual prediction level — consistent with the validation-set finding.

### Full model performance history

| Stage | Threshold | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline LightGBM (val) | 0.35 | 0.900 | 0.495 | 0.638 | 0.867 |
| Tuned LightGBM (val) | 0.35 | 0.940 | 0.405 | 0.566 | 0.855 |
| Calibrated (val) | 0.35 | 0.760 | 0.576 | 0.655 | 0.856 |
| Calibrated + OOF threshold (val) | 0.2956 | 0.800 | 0.531 | 0.638 | 0.856 |
| **Final test set** | **0.2956** | **0.7861** | **0.5241** | **0.6289** | **0.8413** |

---

## 11. Production Refit & Model Registration

### Rationale

All hyperparameter, calibration, and threshold decisions were finalised on held-out data before the production refit. Retraining on the full 7,043-customer dataset provides the production model with all available signal without contaminating the §10 benchmark figures.

### Full-data refit

| Property | Value |
|---|---|
| Training data | Full dataset: train + val + test (7,043 rows; 1,869 churners, 5,174 non-churners) |
| Model architecture | LightGBM + sigmoid calibration (`CalibratedClassifierCV`, cv=5) |
| Hyperparameters | Unchanged from Optuna best (§5) — no re-tuning on full data |
| `scale_pos_weight` | Recalculated on full-data class ratio (5,174 / 1,869 ≈ 2.77) |
| Features | 40 (same as training pipeline) |
| MLflow run | `f81665fa` (experiment `"Telco Churn - Final Model"`) |

### Production threshold re-derivation

The §9 OOF threshold (0.2956) was derived on `X_train` only. With the full dataset, OOF cost minimisation was re-run under the base scenario:

| Threshold source | Threshold | Dataset | Notes |
|---|---|---|---|
| §9 OOF cost min | 0.2956 | Train set only | Used for §10 test-set evaluation |
| **Production OOF cost min** | **0.3596** | **Full dataset** | **Production-shipped value** |

The production threshold rises by 0.064, reflecting the broader distribution of the full dataset (proportionally more long-tenure, non-month-to-month customers shift the cost-optimal point rightward).

### OOF performance on full data (threshold = 0.3596)

| Metric | OOF value |
|---|---|
| Recall | 0.717 |
| Precision | 0.569 |
| F1 | 0.634 |
| PR-AUC (avg precision) | 0.644 |

> OOF metrics on the full dataset are not directly comparable to §10 test-set figures — the test set is now included in training. These are a distributional sanity check only; the §10 test-set numbers remain the authoritative benchmark.

### MLflow model registration

Registered model: **`telco-churn-pipeline`** (version 1), aliases `champion` and `challenger`.

**Logged artifacts:**

| Artifact | Description |
|---|---|
| `model/` | MLflow pyfunc format — `mlflow.sklearn.load_model()` |
| `feature_columns.txt` | Ordered list of 40 feature names |
| `preprocessing.pkl` | Fitted preprocessor for offline feature transformation |
| `model_card.json` | Parameters, dataset description, run provenance |

```bash
# Load the champion model by alias
mlflow.sklearn.load_model("models:/telco-churn-pipeline@champion")
```

---

## 12. Known Limitations

1. **Annual/multi-year contract churners are a near-total blind spot.** One-year and two-year contract holders have FN rates of 0.972 and 1.000 respectively. The model learned "annual contract = committed customer" as a near-irrefutable heuristic.

2. **Long-tenure, low-cost segment under-served.** Customers in the 25+ month tenure band have a 0.465 FN rate. Long tenure is treated as a loyalty signal, but duration can be a lagging indicator for quietly disengaging customers.

3. **No uplift / persuadability modelling.** The model identifies *who will churn*, not *who will respond to a retention offer*. Without A/B test data, the model cannot separate persuadables from lost causes.

4. **Residual calibration gap at high scores.** The calibrated model under-predicts churn probability for high-scoring customers by up to ~10 pp above score ≈ 0.58. Expected-value calculations will understate financial exposure for the highest-risk customers.

5. **No re-contact suppression.** Customers flagged in one cycle will appear in the next scoring run with the same feature profile. Without a 90-day suppression window, the system risks contact fatigue.

6. **Production monitoring not yet deployed.** No PSI or prediction quality monitoring is currently in place to trigger re-training.

7. **Feature set cannot reduce FPs for loyal high-risk-profile customers.** ~50 % of non-churners who share the fiber optic + month-to-month + no add-on profile are incorrectly flagged. No loyalty, satisfaction, or recency-of-service-change signals are available.

8. **Business cost parameters are illustrative.** The $68/intervention and $575/LTV figures are reasonable but not Finance-validated.

---

## 13. Recommendations & Next Steps

### Immediate (model v1 deployment)

1. **Deploy at threshold 0.2956** for the base cost scenario. Revisit threshold if intervention cost assumptions change.
2. **Implement tiered outreach:** cheap outreach (email/SMS) for scores in 0.30–0.50; expensive retention offers only for scores > 0.50.
3. **Apply a 90-day re-contact suppression window** to prevent intervention fatigue.

### Short-term (next model iteration)

4. **Address annual-contract blind spot:** Explore contract-type-specific sub-models or a lower intervention threshold applied selectively to annual-contract holders who show secondary risk signals.
5. **Instrument an A/B test** on the first cohort of flagged customers. This data is the prerequisite for uplift modelling.
6. **Add engagement/recency features** (last service change date, usage trend, support ticket volume) to separate loyal high-risk-profile customers from genuine pre-churners.

### Ongoing

7. **Deploy PSI monitoring** for score distribution drift. Trigger re-evaluation when PSI > 0.2.
8. **Recalibrate the threshold annually** using updated OOF predictions on a full-data refit.
9. **Replace illustrative business parameters** with Finance-validated figures before committing to P&L projections.
