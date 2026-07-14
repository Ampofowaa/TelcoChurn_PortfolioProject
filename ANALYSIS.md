# Analysis & Modelling Decisions

Full modelling rationale for the [Telco Customer Churn](README.md) portfolio project.
Covers problem framing, EDA, feature engineering, model selection, hyperparameter tuning,
calibration, threshold optimisation, business impact, final test-set results, error analysis,
SHAP explainability, and production refit.

This document is the authoritative modelling record. Where it diverges from earlier exploratory analysis, this document and `src/` are what's authoritative, per a stated reason recorded here or in commit/PR history.
The `src/` package implements the production-ready version of the current logic.

> *Section numbers follow the analytical lifecycle and do not correspond to implementation phase numbers in `PROJECT_PLAN.md`.*

---

## Table of Contents

0. [Problem Framing & Cost Definition](#0-problem-framing--cost-definition)
1. [Data Ingestion & Quality Checks](#1-data-ingestion--quality-checks)
2. [EDA & Statistical Testing](#2-eda--statistical-testing)
3. [Feature Discovery & Engineering](#3-feature-discovery--engineering)
4. [Model Selection, Feature Selection & Hyperparameter Tuning](#4-model-selection-feature-selection--hyperparameter-tuning)
5. [Probability Calibration](#5-probability-calibration)
6. [Business Impact & Threshold Selection](#6-business-impact--threshold-selection)
7. [Final Test-Set Results](#7-final-test-set-results)
8. [Production Refit & Model Registration](#8-production-refit--model-registration)
9. [Known Limitations](#9-known-limitations)
10. [Recommendations & Next Steps](#10-recommendations--next-steps)

---

## 0. Problem Framing & Cost Definition

This step locks the rules that govern every modelling decision in §§1–8. It is documented here — before any EDA or model results — so that the cost-sensitive threshold, the choice of recall as the headline reported metric, and the class imbalance handling all have a traceable origin.

### Prediction unit

A **single customer at a scoring cycle** (default: monthly). The model produces one churn-probability score per customer per run. All metrics (Recall, Precision, P&L) are computed per customer, not per account or household.

### Label definition and horizon

`Churn = 1` if the customer cancelled service **within the current billing cycle** (approximately 30 days). The label is derived directly from the IBM Telco dataset's `Churn` column (Yes/No, recoded to 1/0). It is a **binary, point-in-time label** — there is no survival horizon to tune and no soft-churn intermediate class in this dataset.

The label captures *revealed* churn (cancellation has occurred), not *predicted intent*. This means the model is trained to identify customers who have the same profile as those who eventually cancelled — not customers who are currently dissatisfied but have not yet acted.

### Decision the score feeds

The score feeds a **proactive retention intervention decision**: whether to include a customer in the upcoming outreach cycle (discount offer, contract upgrade, service credit). The decision is binary per customer per cycle. A contacted customer either receives an offer or does not; there is no tiered response modelled at this stage (tiered outreach is a §10 recommendation).

This framing makes the cost structure asymmetric and well-defined (see below). It has two metric consequences: **PR-AUC is the primary model selection and promotion metric** — it summarises precision-recall performance across all thresholds and is the promotion gate used in §4 and §8. **Recall at the production threshold is the headline business-facing number, not a second thing anything is optimised for** — once the model is deployed at the optimised production threshold (§6), it is the first question the business will ask ("how many churners did we catch this cycle?"), because a missed churner receives no offer and is lost. The threshold itself is chosen to maximise expected value (§6's `t* = c / (r × LTV)`), never recall — a policy that only maximised recall would contact every customer, which is exactly the "treat all" baseline the P&L-vs-baseline figure (§7) exists to beat. Recall is worth reporting because it is legible to a non-technical audience in a way PR-AUC is not, but it is read alongside that expected-value figure, never in place of it: recall with no precision/cost context cannot distinguish a profitable campaign from an expensive one.

### Cost structure — cost attaches to the *action*, not to the error

The intervention is paid whenever we contact someone. We do not know whether they were going to churn at the moment we dial. So the decision table is over (action, true state), and the correct cell to interrogate is the **true positive** — which is *not* free. Four quantities drive it: `q`, the probability a customer churns; `r`, the retention rate — the probability a retention offer actually works once made; `LTV`, the customer's lifetime value if retained; and `c`, the total cost of the intervention — contacting the customer plus any discount offered.

| | **contact** (spend `c`) | **do nothing** |
|---|---|---|
| **churner** (prob `q`) | `−c + r·LTV` — offer works with prob `r` | `0` |
| **non-churner** (prob `1−q`) | `−c + LTV` — offer wasted, customer stays anyway | `LTV` |

Contact if and only if `E[contact] > E[do nothing]`. The `(1−q)·LTV` terms cancel and the rule collapses to:

> **Contact if and only if `q · r · LTV > c`, i.e. the operating threshold is `t* = c / (r × LTV)`.**

This shifts the threshold to the right relative to the archived pass, which selected under a cost function that charged the intervention only to false positives — treating a correctly-identified churner as free to contact. Here it is not: contacting costs `c` regardless of the outcome. **The actual cost parameters, the derived thresholds for all three scenarios, and the shipped value are §6's job, not this section's** — §0 establishes the rule the numbers get plugged into, not the numbers themselves.

The familiar error-cost view is a *diagnostic*, not the decision rule:

| Error type | Business consequence | Cost |
|---|---|---|
| **False Negative (FN)** | Churner not contacted. Customer cancels; the recoverable share of LTV is lost. | `r · LTV` |
| **False Positive (FP)** | Non-churner contacted. Offer issued at cost; customer was going to stay anyway. | `c` |

Whichever of `r·LTV` and `c` is larger sets which error the threshold implicitly favours avoiding — §6 reports the actual ratio per scenario, and it is not the same across scenarios (cheaper, less generous interventions and costlier, more generous ones don't move `r·LTV` and `c` proportionally). This diagnostic does **not** by itself give the threshold: `C_FP/(C_FP + C_FN)` presumes correct decisions are free, and a true positive costs `c` like any other contact. *(Selection remains governed by PR-AUC alone — which error the cost structure favours avoiding is a statement about the cost structure, not a second selection metric.)*

#### The retention rate `r` is the dominant uncertainty — not the model

The operating threshold `t*` is inversely proportional to the retention rate `r`, and `r` is the one parameter that **cannot be estimated from this dataset** — it is the fraction of contacted churners the offer actually saves, and measuring it requires intervening on customers and observing what happens. The assumed value, 0.30, is an industry benchmark drawn from a literature range of 0.15–0.40, not something observable in this dataset.

A plausible range of `r` moves the threshold far more than any realistic improvement in PR-AUC would move the operating point. **The single most consequential number in the deployment decision is a benchmark guess, not a model output.** §6 has the full retention-rate sensitivity sweep and plot with the actual numbers.

**`r` stops being a guess once the model is actually deployed and contacting customers.** From that point on, the system logs two things: who was contacted, and — once enough time has passed to know for sure — whether they actually stayed. `performance_check.py` joins the two in a `prediction_outcomes` table, which turns "of the customers we contacted, what fraction did we retain?" into a number that can be counted directly instead of assumed, and `t*` can be re-derived from that real `r` instead of a benchmark. That data doesn't exist yet — it requires live deployment and time for outcomes to mature — so until it does, the three-scenario bracket (Conservative / Base / Optimistic) in §6 exists to show how much the decision would shift under different plausible values of `r`, rather than presenting one falsely-precise number.

The cost parameters are illustrative, derived from plausible telecom industry benchmarks, and not Finance-validated; see §9 Known Limitation #8 and §6 for the actual values.

### Success criterion

The model is considered fit for production if, at the cost-optimised threshold on the **sealed test set**:

| Criterion | Gate | Rationale |
|---|---|---|
| PR-AUC | ≥ 0.60 | Primary ranking metric — threshold-free and imbalance-appropriate at a 27 % positive rate; ROC-AUC is optimistic under class imbalance and is not used as a gate |
| Recall at the optimised production threshold | ≥ 0.65 | Primary business metric — proportion of churners caught at the deployed operating point |
| No test-set information used before final evaluation | Structural requirement — threshold derived from OOF predictions only | Preserves the "test set touched once" invariant (§7) |

These gates were set before the test set was opened. The final test-set results in §7 are evaluated against them exactly once.

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
- **Class imbalance is moderate but consequential.** The dataset is 73.5 % No-churn / 26.5 % Churn (5,174 / 1,869; 2.77:1 ratio). A naive accuracy-maximising model achieves 73.5 % accuracy while identifying zero churners. **PR-AUC is the primary model selection and promotion metric** — more informative than ROC-AUC at this imbalance ratio; recall at the deployed threshold is the headline business-facing number, though the threshold itself is chosen to maximise expected value, not recall (see §0).

### Univariate distributions

**Numeric features:**
- `tenure` is bimodal (U-shaped): a spike at 0–5 months (new customers at highest inherent risk), a broad plateau from ~10–65 months (the stable retained cohort), and a second concentration at 65–72 months (long-term loyals) — consistent with a survival distribution. The 0–12 month cohort churns at ~47 %; by 49+ months that falls below 10 %.
- `monthlycharges` shows a two-tier structure: a sharp spike at $18–20 (basic phone-only plans), then a broad spread to $120 with density skewed toward $75–120 (bundled internet + add-on packages).
- `totalcharges` is right-skewed, with a long tail of high-value, long-tenure customers. Billing amounts shift over time for ~91 % of customers, so `totalcharges` carries signal independent of the other two numeric features.

**Categorical features:**
- **Demographics:** gender is near-balanced (~51 % male); senior citizens represent ~16 % of the base; customers with a partner or dependents are ~48 % and ~30 % respectively — the base skews toward younger, independent adults.
- **Services:** ~90 % have phone service; internet service splits across fiber optic (~44 %), DSL (~34 %), and no internet (~22 %). Security and support add-ons skew heavily toward "No" — the ~22 % without internet cannot subscribe, and month-to-month customers show lower uptake. Streaming add-ons are more evenly split.
- **Contract and billing:** month-to-month contracts dominate (~55 %); paperless billing is the majority preference (~59 %); electronic check is the most common payment method (~34 %).

**Outliers:**
All three numeric features have zero IQR-flagged outliers. The bounds are wide because the features span their full natural ranges — tenure 0–72 months (by contract design), monthlycharges ~$18–$119, totalcharges ~$19–$8,685. The right-skew in `totalcharges` reflects a genuine business pattern, not contamination. All values are retained.

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
| OnlineSecurity | Cramér's V | 0.35 (strong) | Churn ~3× higher without the service |
| TechSupport | Cramér's V | 0.34 (strong) | Churn ~3× higher without the service |
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
| **PhoneService** | Cramér's V | **0.01** (p = 0.32) | **Non-predictor** |
| **Gender** | Cramér's V | **0.0086** (p = 0.47) | **Non-predictor** |

**Interpretive notes:**

- **Fiber optic churn** operates through two overlapping channels: a cost channel (average $91.50/month vs $58.10 for DSL — a 57 % premium) and a potential service quality channel. The data cannot separate the two; feature importance in the modelling phase will clarify each channel's contribution.
- **Payment method and paperless billing reflect commitment depth, not direct drivers.** Automated payment methods (bank transfer, credit card) are associated with lower churn — consistent with greater friction to cancel. Paperless billing's elevated churn is a proxy for the month-to-month and fiber optic mix.
- **Senior citizen churn** (~42 % vs ~24 %) is likely mediated by contract type and internet service; the EDA cannot decompose the independent age contribution. No systematic age-based under-service is evident in the error analysis (§7).
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

**All features are retained — none are dropped on VIF grounds.** The practical impact of multicollinearity is model-family-dependent: tree-based methods (LightGBM, XGBoost, RandomForest) are immune — each split is evaluated independently and collinearity does not distort estimates. Linear models are materially affected and would require feature consolidation or regularisation. The permutation-importance experiment during model training (§4) is the explicit gate for any feature pruning.

The full EDA — rendered distributions, bivariate charts, interaction heatmaps, and VIF tables — is in [`notebooks/01-eda.ipynb`](notebooks/01-eda.ipynb).

---

## 3. Feature Discovery & Engineering

### 3a. Feature Discovery

Feature discovery precedes feature engineering to identify which new columns are worth building. The search follows two strategies: **error-driven** — profiling the base model's false negatives to find subgroups it systematically misses, then hypothesising a feature that targets each blind spot directly — and **domain-derived** — using knowledge of telecom pricing and service structure to construct row-level ratios and counts that no single tree split can replicate on its own. Both strategies feed the same evaluation pipeline: a baseline LightGBM trained on the 19 raw IBM columns (using 5-fold out-of-fold, or OOF, predictions — each row scored by a fold that never saw it during training — PR-AUC = **0.6387**, 95 % bootstrap CI: [0.6113, 0.6658]) sets the performance floor, and every candidate is assessed through a four-screen adoption gate, cheapest-first.

The 7,043-row dataset is split 80/20 into a **dev partition** (5,634 rows) and a held-out **test set** (1,409 rows) by the canonical split module (`data/split.py`, `datasets/processed/split_manifest.parquet`). This split is sealed once, before feature discovery or any downstream modelling step runs, and the test set is touched exactly once — at final evaluation (§7). Discovery below runs on the **dev partition only**, so no candidate-adoption decision touches held-out data.

**Four-screen gate:**

The four gates assess each candidate in sequence:

- **Serving availability** — can this column be computed at prediction time from raw data alone?
- **Redundancy** — is it too similar to a column already in the adopted set?
- **Performance** — does adding it improve the model's ranking score (PR-AUC) by at least 0.0015?
- **Importance** — does the model actually use it, or is the signal already captured elsewhere?

| Screen | Name | Rejection trigger |
|---|---|---|
| 1 | Serving availability | Column not computable from raw CSV at inference time |
| 2 | Redundancy (soft gate) | Spearman \|r\| ≥ 0.85 (numeric) or Cramér's V ≥ 0.70 (categorical); flags but does not auto-reject |
| 3 | Performance | Global OOF PR-AUC delta < 0.0015 |
| 4 | Importance | Permutation importance ≤ `max(noise_decoy, 0) + 0.005` |

Screen 4 is the empirical backstop for collective or cross-type redundancy that Screen 2's pairwise check cannot catch.

**Error-driven laps (1–6).** The base FN error profile surfaces three blind spots:

| Blind spot | FN rate | Missed churners |
|---|---|---|
| Two-year contract | 94.9 % | 37 |
| Tenure 55–72 months | 55.9 % | 62 |
| One-year contract | 51.5 % | 67 |

*Rationale.* **One-year and two-year contract blind spots (Laps 1–4):** these groups churn at ~11 % and ~3 % respectively versus 43 % for month-to-month, so the model assigns nearly everyone in them a low churn score — even customers who show clear warning signs like fiber optic internet or high monthly charges. Each lap tests whether explicitly flagging a specific high-risk combination within that segment (e.g. "one-year contract AND fiber optic," "two-year contract AND above-median charge") could give the model a dedicated signal for those customers, rather than treating the whole contract group the same way. **Tenure blind spot (Laps 5–6):** the model reads tenure as a smooth number, but churn risk drops steeply at specific thresholds rather than declining gradually. Lap 5 tests whether bucketing tenure into lifecycle cohorts (0–12 mo, 13–24 mo, etc.) makes those risk tiers explicit enough to improve predictions; Lap 6 tests a flag for customers who have remained on a month-to-month contract for over 55 months — a "never upgraded despite years of opportunity" signal.

All six error-driven candidates are pairwise conjunctions of existing columns. All six are rejected.

| Lap | Feature | Description | Outcome | Reason |
|---|---|---|---|---|
| 1 | `two_year_fiber` | `contract_type = Two year` AND `internetservice = Fiber optic` | REJECTED (Screen 4) | PR-AUC passed Screen 3 (+0.0017, 0.6387→0.6404), but the model gains no independent signal from the flag once measured against the full context — importance 0.0001, below the 0.0054 floor |
| 2 | `two_year_high_charge` | `contract_type = Two year` AND `monthlycharges` > dataset median | REJECTED (Screen 4) | PR-AUC also passed Screen 3 (+0.0017, 0.6387→0.6404 — an identical delta to Lap 1, both candidates converge on the same fiber-optic-heavy high-charge subset), but importance (0.0009) fell below the 0.0050 floor |
| 3 | `one_year_streaming` | `contract_type = One year` AND at least one streaming service subscribed | REJECTED (Screen 3) | PR-AUC fell (−0.0009, 0.6387→0.6378), below the 0.0015 minimum |
| 4 | `one_year_high_charge` | `contract_type = One year` AND `monthlycharges` > dataset median | REJECTED (Screen 3) | PR-AUC fell (−0.0003, 0.6387→0.6384), below the 0.0015 minimum |
| 5 | `tenure_cohort` | `tenure` bucketed into 5 lifecycle cohorts (0–12, 13–24, 25–48, 49–65, 65+ months) | REJECTED (Screen 3) | PR-AUC fell (−0.0041, 0.6387→0.6346) — the largest drop of any candidate; the buckets discard information the raw continuous `tenure` column already carries |
| 6 | `long_tenure_mtm` | `tenure` > 55 months AND `contract_type = Month-to-month` | REJECTED (Screen 3) | PR-AUC delta +0.0013 — positive but below the 0.0015 minimum |

*Structural conclusion:* Two candidates (Laps 1–2) clear the PR-AUC bar but fail the importance backstop — LightGBM already recovers the same pairwise conjunction from its own splits, so the pre-computed flag adds no independent signal once the model has both source columns. The other four (Laps 3–6) fail earlier, directly at the PR-AUC screen — three with a negative delta and one with a positive delta too small to clear the threshold. Useful features must require operations no single tree split can replicate — row-level division or within-row summation.

**Domain-derived laps (7–9).**

*Rationale.* Laps 1–6 showed that recombining columns the model already has never helps. The next question is whether features requiring operations the model genuinely cannot perform on its own — row-level division and within-row summation — can add signal. **`charge_per_service`** tests whether normalising monthly spend by number of active services reveals a per-unit cost signal invisible in raw charges alone. **`num_add_on_services`** tests whether collapsing all optional subscriptions into a single count captures a "switching cost depth" signal the individual add-on flags carry only in aggregate. **`monthly_to_total_ratio`** is included as a deliberate decoy to verify the gate correctly rejects a ratio that looks like new information but reduces to a near-perfect transformation of a column the model already has.

| Lap | Feature | Description | Outcome | Reason |
|---|---|---|---|---|
| 7 | `charge_per_service` | monthlycharges ÷ num_active_services | **ADOPTED** | Per-unit cost is genuinely new information not recoverable from raw charges alone — all four screens passed (importance 0.0080, PR-AUC +0.0033, 0.6387→0.6420) |
| 8 | `num_add_on_services` | Count of 7 optional add-on subscriptions | REJECTED (Screen 3) | Screen 2 nearly flagged it (max_corr 0.792, close to the 0.85 threshold) — the seven individual add-on flags are already in the model; their sum adds nothing collectively that they do not cover separately (PR-AUC −0.0021, 0.6420→0.6399) |
| 9 | `monthly_to_total_ratio` | monthlycharges ÷ totalcharges | REJECTED (Screen 2 flagged, Screen 3 failed) | Reduces to ≈ 1/`tenure` (Spearman \|r\| = 0.999) — confirmed decoy; the model already has `tenure` directly (PR-AUC −0.0020, 0.6420→0.6400) |

The domain-derived search confirmed the core insight from Laps 1–6: a feature must require operations the model cannot perform through its own splits. `charge_per_service` met that bar — dividing monthly charges by service count is a relationship no single split on either column can replicate. The other two did not: the add-on count was collectively redundant with its seven source flags despite passing Screen 2's pairwise check, and the charges ratio reduced to approximately `1/tenure` (Spearman |r| = 0.999), making it mathematically equivalent to a column the model already holds regardless of how it was framed.

**Backward elimination.** Removing `charge_per_service` drops PR-AUC from 0.6420 to 0.6387 (Δ = +0.0033), comfortably above the 0.0015 threshold — the feature holds its place in the full adopted set.

**Result:** 1 feature adopted from 9 candidates. OOF PR-AUC: 0.6387 → **0.6420**. Adopted set: `charge_per_service`, frozen to `reports/feature_discovery/adopted_features.json`.

The full lap-by-lap trail — per-screen outputs, provenance records, and diagnostic plots — is in [`notebooks/02a-feature-discovery.ipynb`](notebooks/02a-feature-discovery.ipynb).

---

### 3b. Feature Engineering

One feature survived the feature discovery process: `charge_per_service`. Feature engineering builds it into the production pipeline alongside the 19 raw IBM columns, producing a 20-column feature set for model training.

`charge_per_service` is a SQL-engineered feature that normalises each customer's monthly bill by the number of active service subscriptions they hold — nine binary flags in total (`phoneservice`, `multiplelines`, `internetservice`, and six add-on services).

The full 20-column inventory — grouped by type and always in sync with the codebase — is in [`notebooks/02b-feature-engineering.ipynb`](notebooks/02b-feature-engineering.ipynb).

---

## 4. Model Selection, Feature Selection & Hyperparameter Tuning

### 4a. Model Selection

#### Data split & evaluation strategy

The dev/test split is established once in §3a — the 5,634-row dev partition and the 1,409-row sealed test set, touched exactly once at final evaluation (§7). `train.py` imports the canonical split manifest (`data/split.py` / `datasets/processed/split_manifest.parquet`) directly rather than deriving its own — the two are the same split by construction, not just by convention. All comparisons and tuning below run on the dev partition using 10-fold cross-validation repeated 10 times (`configs/config.yaml: training_setup.cv_folds/cv_repeats`, chosen to shrink the mean PR-AUC estimate's variance on ~5,600 dev rows), which gives more reliable estimates than a single held-out slice on a dataset of this size.

#### Model family scope

Only LightGBM and `LogisticRegressionCV` (the strongest interpretable linear baseline) are compared. Other tree ensembles (XGBoost, RandomForest) are skipped: on tabular data they typically perform within noise of each other, so comparing several at default config would just compare defaults, not tuned ceilings — added maintenance cost for no decision value. LightGBM is committed up front for build-specific reasons: fast exact TreeSHAP for the SHAP explainability work (§7) and its Streamlit surfacing, training speed for Optuna and the weekly retrain cadence, and clean interaction with `class_weight`-based imbalance handling.

**Native categorical splits vs. one-hot encoding (deliberate, not an oversight):** LightGBM supports native categorical splitting, often preferred over OHE for trees. OHE is used here instead because every categorical in this feature set has ≤ 4 unique values (the column-blowup cost OHE is usually criticized for is negligible), it keeps the tree and linear preprocessing pipelines structurally uniform, and it yields cleaner per-level SHAP attribution for the Streamlit top-5 contributions view.

#### Candidate comparison (PR-AUC, RSKF 10×10)

LightGBM and LogisticRegressionCV are compared on PR-AUC: threshold-free, aligned with the §0 cost structure, and more discriminating than ROC-AUC at this 2.77:1 class imbalance, since ROC-AUC is scored against the large negative class and can look nearly identical across models that rank the minority churn class very differently. Each model uses its own preprocessor (tree vs. linear — see `03a-model-selection.ipynb`), so OHE encoding choices don't handicap either candidate, and class imbalance is handled via `class_weight='balanced'` for both rather than SMOTE/resampling — chosen for three reasons: the ~27% positive rate is mild imbalance, and resampling pays off more at extreme ratios; the feature space is mostly one-hot, and SMOTE interpolates incoherently across dummy columns; and calibration is a later deliverable (§5) that resampling would undermine — base-rate-altering resamplers decalibrate probabilities while reweighting barely shifts them, and the operating threshold gets set explicitly at that point anyway (§6). Identical fold indices are shared across all three candidates (one `RepeatedStratifiedKFold` instance, instantiated once) so scores are paired by construction — the precondition for the bootstrap comparison below. A `DummyClassifier(strategy='prior')` — a baseline that ignores every customer feature and simply predicts the overall churn rate for everyone — is included as a safeguard: since it has no real signal to work with, a strong score from it would mean the target has somehow leaked into the features. `train.py` asserts its ROC-AUC sits at chance and its PR-AUC matches the churn rate, and aborts the run if either check fails.

| Candidate | CV PR-AUC | ± std | Train s/fold |
|---|---|---|---|
| `dummy_prior` | 0.265 | 0.001 | 0.00 |
| `logreg_cv` | 0.651 | 0.033 | 0.89 |
| `lgbm_default` | 0.658 | 0.035 | 0.25 |

**Safeguard check passed:** the dummy classifier scored 0.265, matching the dev-set churn rate (0.265) — confirming the real candidates' higher scores reflect genuine predictive signal, not a broken eval harness or leaked target.

LightGBM leads by ~0.007 PR-AUC points; ROC-AUC is close and non-contradictory (0.844 for both candidates). LogReg is markedly more expensive to train here (`LogisticRegressionCV`'s inner 5-fold search over 10 `C` values costs ~3.6× LightGBM's single fit per outer fold — 0.89s vs. 0.25s) — a genuine simplicity-vs-cost tradeoff LogReg does *not* win on, despite its interpretability appeal.

#### Precision/F1-at-fixed-recall profile

Threshold-dependent diagnostics (precision, F1) are not reported at the default 0.5 cutoff — they are reported as a profile over OOF predictions at three fixed recall targets, computed identically for both candidates:

| Recall target | LogReg precision | LogReg F1 | LogReg threshold | LightGBM precision | LightGBM F1 | LightGBM threshold |
|---|---|---|---|---|---|---|
| 0.70 | 0.573 | 0.631 | 0.596 | 0.574 | 0.631 | 0.572 |
| 0.80 | 0.516 | 0.628 | 0.485 | 0.517 | 0.628 | 0.453 |
| 0.90 | 0.445 | 0.595 | 0.341 | 0.446 | 0.596 | 0.269 |

Precision and F1 are essentially indistinguishable between candidates at every recall target — reinforcing the near-linear finding above. The threshold column is the actual cutoff each candidate needed to hit that recall target on its own OOF predictions; it differs more between the two models (e.g. 0.341 vs. 0.269 at the 0.90 target) because their predicted-probability distributions aren't shaped the same way, but neither is a committed decision threshold. These profiles are threshold-*planning* diagnostics, not selection tools (PR-AUC alone decides the family); the operating threshold itself is set later (§6).

#### Disaggregated robustness & fairness check (flag-only)

Pre-registered, computed on OOF predictions for all three candidates, and explicitly non-gating — aggregate PR-AUC still decides the family regardless of what this check finds.

**Robustness** (`contract_type`, `tenure_cohort`, `internetservice`): per-segment PR-AUC tracks each segment's own churn-rate prevalence for both candidates (e.g. LightGBM: 0.428 churn rate → 0.694 PR-AUC on month-to-month; 0.029 → 0.070 on two-year contracts) — expected for a prevalence-sensitive metric, not a robustness failure. A paired row-level bootstrap (`segment_bootstrap_delta`, 1,000 resamples) puts a 95% CI on the LightGBM-minus-LogReg gap in each segment: 4 of 11 exclude zero, split evenly between the two candidates. LightGBM has a genuine edge among 0–12m-tenure customers (Δ = +0.025, CI [0.005, 0.047]; the highest-churn tenure band, 47.4%) and no-internet customers (Δ = +0.092, CI [0.024, 0.164], the widest gap of any segment; one of the lowest-churn segments, 7.5%) — spanning both risk tiers, not just the high-churn one. LogReg edges out on 13–24m tenure (Δ = −0.036, CI [−0.066, −0.001]) and 49–65m tenure (Δ = −0.046, CI [−0.084, −0.001]). Every other split — including the full contract-type breakdown, where the point estimates are themselves mixed (LogReg ahead on one-year and two-year contracts, LightGBM ahead on month-to-month) — straddles zero and isn't distinguishable from sampling noise.

**Fairness** (`gender`, `seniorcitizen`, `has_partner`, `dependents` — the four protected/quasi-protected axes, per this section's policy of measurement over exclusion (see "Protected attributes & fairness policy" below)): per-subgroup PR-AUC gaps track each subgroup's own churn-rate prevalence for both candidates (e.g. LightGBM: 0.726 PR-AUC for senior citizens, 42.0% churn rate, vs. 0.632 for non-seniors, 23.6% churn rate) — the same prevalence-driven pattern as the robustness segments, not evidence of differential treatment. The same CI test is unambiguous here: all 8 of 8 fairness segments straddle zero — no demographic split shows a LightGBM-vs-LogReg gap distinguishable from sampling noise, including `has_partner` (Δ = −0.003, CI [−0.026, 0.016]), which looked like a hairline LogReg edge before the CI was applied. So the family choice doesn't trade fairness for LightGBM's other advantages — a conclusion resting on a formal test, not eyeballed gaps. **No disparity is flagged for follow-up at this stage.**

#### Bootstrap selection decision

To judge whether LightGBM's lead is real or a quirk of fold assignment, the gap in PR-AUC (Δ = AP(LightGBM) − AP(LogReg)) is stress-tested with a paired bootstrap: the 100 paired fold scores are resampled 10,000 times, producing a 95% percentile CI on Δ. The percentile CI *is* the test — there is no separate significance-level gate layered on top. The pre-registered decision rule (fixed before seeing results) has three branches against a materiality threshold Δ\* = 0.005:

| Branch | Condition | Outcome |
|---|---|---|
| `lgbm_win` | CI excludes 0 in LGBM's favour and Δ ≥ Δ\* | Adopt LightGBM on the evidence |
| `tie` | CI includes 0, or excludes 0 but \|Δ\| < Δ\* | Practical tie — adopt LightGBM on the build-specific rationale above (SHAP, speed, calibration, continuity), not on this comparison |
| `logreg_win` | CI excludes 0 in LogReg's favour and Δ ≤ −Δ\* | LogReg confidently and materially wins — ship LogReg instead |

| | |
|---|---|
| Observed gap (LightGBM − LogReg) | +0.007 |
| 95% confidence interval | [+0.002, +0.012] — excludes zero, entirely in LightGBM's favour |
| Probability the gap is ≤ 0 | 0.20% (p = 0.0020, informational only) |
| Materiality threshold (Δ\*) | 0.005 |
| **Branch fired** | **`lgbm_win`** |
| **Verdict** | **LightGBM adopted on the evidence — the bootstrap shows a genuine, material PR-AUC advantage over LogReg, reinforced by (not resting on) the build-specific rationale above** |

**This comparison uses default LightGBM as a conservative floor** — hyperparameter tuning happens later in this section, after the feature-selection freeze below, and cannot run earlier without violating select-then-tune. The asymmetry cuts in the safe direction: if untuned LightGBM already beats tuned-to-its-ceiling LogReg, tuning can only widen the gap in LightGBM's favour. The margin reported here is refreshed after the tuning subsection's result against the tuned model — recorded once that lands.

#### Generative diagnostic loop: bias/variance and OOF segment profiling

After the family is confirmed (LightGBM), a diagnostic pass on the development set (sealed test untouched) asks *how* the model is performing, not just how well: is it underfitting, overfitting, or failing on a specific customer slice? Three explanations are distinguished — **bias** (features can't capture the pattern, needs a new feature), **variance** (the model overfits what it has, needs regularisation or fewer features), and **localized segment failure** (a specific slice underperforms, needs a targeted feature) — so a tuning problem isn't mistaken for a feature gap, or vice versa. Both checks are non-gating; PR-AUC alone still decides any feature change.

**Bias/variance.** Default-config LightGBM (identical config to the `lgbm_default` candidate above) is refit via the same `RepeatedStratifiedKFold(10×10)` scheme, this time capturing both in-sample (training-fold) and held-out (validation-fold) PR-AUC per fold:

| | |
|---|---|
| Train PR-AUC (mean) | 0.7945 |
| CV PR-AUC (mean) | 0.6583 |
| Train − CV gap (variance) | 0.1362 |
| Lift over Dummy floor (bias) | +0.3933 (0.6583 − 0.265) |

The gap is wide — well above the ~0.05 healthy guideline — meaning default LightGBM (100 trees, `num_leaves=20`) overfits the ~4,500-row training folds noticeably. Three signals rule out bias: training PR-AUC (0.7945) is already far above the Dummy floor, so the model fits the pattern; the held-out lift over that floor (+0.3933) confirms it generalises; and the learning curve's train-CV gap narrows from 0.316 at 901 rows to 0.153 at 4,507 rows as training size grows 5× — a bias problem wouldn't shrink with more data. CV PR-AUC is still climbing (0.610 → 0.652) without plateauing, so more data would likely help further — a data-acquisition note (§9), not a feature gap.

A single 80/20 early-stopping run makes the overfitting concrete round by round (illustrative only, not the fold-averaged number driving the decision): validation PR-AUC peaks at round 41 (0.6608), then drifts down to 0.6513 by round 100 as the model starts overfitting, while training PR-AUC climbs the whole time, from 0.812 at round 1 to 0.914 at round 100 — the model keeps "improving" on data it has already memorised, well past the point where validation peaked.

This is a **variance** signal, not a bias one — and simplifying the model is itself a standard variance remedy, so both the hyperparameter regularisation search in the tuning subsection below (`reg_alpha`/`reg_lambda`, `num_leaves`, `min_child_samples`) and the feature-count reduction tested in the next section are legitimate responses to this gap, not just the former.

**OOF segment profiling.** Out-of-fold predictions from the bias/variance CV run above are disaggregated across four cuts: contract type; a tenure band (< 12m / 12–36m / > 36m); a service-count quintile (each customer's count of nine active phone/internet/add-on services, split into five equal-sized groups, 1 = fewest active services and 5 = most); and a `charge_per_service` outlier flag (customers above the 95th percentile):

| Segment | Value | n | Churn rate | PR-AUC | 95% CI |
|---|---|---|---|---|---|
| Contract | Month-to-month | 3,095 | 0.428 | 0.694 | [0.667, 0.721] |
| Contract | One year | 1,178 | 0.110 | 0.188 | [0.150, 0.236] |
| Contract | Two year | 1,361 | 0.029 | 0.070 | [0.043, 0.118] |
| Tenure band | < 12m | 1,756 | 0.474 | 0.754 | [0.722, 0.785] |
| Tenure band | 12–36m | 1,480 | 0.254 | 0.547 | [0.496, 0.605] |
| Tenure band | > 36m | 2,398 | 0.120 | 0.360 | [0.311, 0.422] |
| Service quintile | 1 (fewest) | 1,666 | 0.190 | 0.644 | [0.595, 0.694] |
| Service quintile | 2 | 687 | 0.438 | 0.726 | [0.673, 0.778] |
| Service quintile | 3 | 1,521 | 0.347 | 0.662 | [0.617, 0.707] |
| Service quintile | 4 | 742 | 0.255 | 0.663 | [0.593, 0.730] |
| Service quintile | 5 (most) | 1,018 | 0.158 | 0.499 | [0.424, 0.579] |
| `charge_per_service` outlier (> p95) | outlier | 282 | 0.621 | 0.827 | [0.765, 0.883] |
| `charge_per_service` outlier (> p95) | normal | 5,352 | 0.247 | 0.626 | [0.599, 0.652] |

Every segment's PR-AUC clears its own churn-rate floor by a healthy margin, and a 95% CI on each segment's PR-AUC (a single-model bootstrap computed inline in the notebook, distinct from `segment_bootstrap_delta`) confirms this holds under sampling uncertainty too — its lower bound still clears the floor everywhere. The margin is narrowest on the two smallest, lowest-churn segments — One year contracts (floor 0.110, CI lower bound 0.150) and Two year contracts (floor 0.029, CI lower bound 0.043, ~39 churners) — but even there it clears comfortably; every other segment's margin is at least 3× wider. Service-count churn is non-monotonic (quintile 2 peaks at 43.8%), a genuine pattern the model tracks without degrading. **No systematic failure pattern is identified** — with no segment to target and no bias signal from the checks above, feature selection (below) proceeds on the current 20-column set without additional feature engineering.

The full model-selection walkthrough — the candidate comparison, bootstrap decision, and OOF segment/fairness profiling — is in [`notebooks/03a-model-selection.ipynb`](notebooks/03a-model-selection.ipynb).

### 4b. Feature Selection (input-space freeze)

Selected once against default LightGBM after family commitment, inside CV on the train set. The surviving columns are frozen into the model input space shared by tuning (below), calibration (§5), and serving. Selection runs once, not every retrain: the feature space is small (~20 named features) and stable, so rerunning it on a schedule risks more than it gains — a borderline feature like `paymentmethod` (49/100 fold stability) could flip in or out between retrains from data-split noise alone, breaking tuning's fixed-input assumption and making model versions hard to compare. So the list is frozen instead: downstream stages share the same fixed columns, and the diff between `feature_space.txt` (everything produced) and `feature_columns.txt` (what survived) records what was dropped.

**Method.** Each feature is scored using **permutation importance**: how much the model's performance (PR-AUC) drops when that feature's values are shuffled into random noise, holding everything else fixed — a bigger drop means the feature was actually being used. A synthetic **decoy column**, built from pure noise, sets the bar for what "no real signal" looks like: a feature survives only if its measured importance beats the decoy's by a comfortable margin (`max(decoy_importance, 0) + 0.005`; `configs/training/selection.yaml`). This is model-agnostic — it doesn't lean on any one algorithm's internal bookkeeping — which is why it replaced this build's original approach (LightGBM's own "gain" importance, prone to overstating high-cardinality or frequently-split features; Boruta-SHAP was also considered and rejected for a thinly-maintained dependency with no clean sklearn API). A categorical's one-hot columns are always shuffled and judged together, so it is kept or dropped as a whole, never partially.

Two checks guard against this method misleading:

1. **Fluke check.** Refit across all 100 CV folds and track each feature's survival rate. 8 of the 10 all-dev survivors are selected in at least 70 of 100 folds (`tenure`/`contract_type` every time), but `streamingtv` — the thinnest all-dev survivor — is picked in only 5 of 100 folds, and `paymentmethod` in just 33. Two dropped features cross over the other way: `paperlessbilling` (55/100) and `charge_per_service` (46/100) are both selected more often than either `streamingtv` or `paymentmethod`, despite failing the all-dev floor outright.
2. **Correlated-credit check.** Shuffling correlated features one at a time can under-count shared signal, so each correlated cluster is re-tested together before accepting an individual failure. The `tenure`/`totalcharges`/`monthlycharges` trio all clear the floor alone (no rescue needed; `tenure`–`totalcharges` correlate at 0.89, the strongest pair in the trio). The 7-feature `internetservice` add-on cluster (VIF = ∞, §8b `01-eda.ipynb`) has 4 individual survivors (`internetservice`, `onlinesecurity`, `techsupport`, `streamingtv`), but the other 3 (`onlinebackup`, `deviceprotection`, `streamingmovies`) didn't need rescuing either — credit-splitting never dragged a real signal under the floor.

**Deciding keep vs. reduce, and the outcome.** Same mechanism as the model-family decision above: refit the full and reduced (10-feature) candidates on the same 100 folds, then bootstrap Δ = mean(AP_full) − mean(AP_reduced), paired fold by fold. Result: reduced **0.6490** vs. full **0.6580** — Δ = **0.0100**, 95% CI **[0.0060, 0.0140]**, p < 0.0001, full wins 65 of 100 folds — clears the materiality bar, so **`full_features_win`** fires: **the full set is retained, 20 of 20 kept**. Below, 10 of 20 features individually clear their own decoy floor; the other 10 — including all four protected/quasi-protected attributes (see below) — stay anyway, since this aggregate test, not the per-feature table, governs adoption. That table is a diagnostic audit trail only; it never touches the sealed test set.

| Feature | Real importance | Decoy floor | Survived (all-dev) | Per-fold stability |
|---|---:|---:|:---:|---:|
| `tenure` | 0.0880 | 0.0050 | ✅ | 100/100 |
| `contract_type` | 0.0840 | 0.0050 | ✅ | 100/100 |
| `internetservice` | 0.0280 | 0.0050 | ✅ | 95/100 |
| `totalcharges` | 0.0240 | 0.0050 | ✅ | 91/100 |
| `monthlycharges` | 0.0200 | 0.0050 | ✅ | 94/100 |
| `multiplelines` | 0.0160 | 0.0050 | ✅ | 72/100 |
| `paymentmethod` | 0.0130 | 0.0050 | ✅ | 33/100 |
| `techsupport` | 0.0090 | 0.0050 | ✅ | 88/100 |
| `onlinesecurity` | 0.0080 | 0.0050 | ✅ | 70/100 |
| `streamingtv` | 0.0060 | 0.0050 | ✅ | 5/100 |
| `paperlessbilling` | 0.0050 | 0.0050 | ❌ | 55/100 |
| `streamingmovies` | 0.0040 | 0.0050 | ❌ | 3/100 |
| `seniorcitizen` | 0.0030 | 0.0050 | ❌ | 4/100 |
| `onlinebackup` | 0.0030 | 0.0050 | ❌ | 7/100 |
| `charge_per_service` | 0.0010 | 0.0050 | ❌ | 46/100 |
| `deviceprotection` | -0.0000 | 0.0050 | ❌ | 0/100 |
| `phoneservice` | -0.0010 | 0.0050 | ❌ | 0/100 |
| `dependents` | -0.0010 | 0.0050 | ❌ | 3/100 |
| `has_partner` | -0.0010 | 0.0050 | ❌ | 0/100 |
| `gender` | -0.0010 | 0.0050 | ❌ | 5/100 |

*(Decoy importance on the all-dev fit was -0.0060; the floor is `max(-0.0060, 0) + 0.005 = 0.005` for every row.)*

**Reasons for keeping the failed features.** Most of the 10 failing features carry real, statistically significant univariate correlation with churn per `01-eda.ipynb` §7's Cramér's V — the add-on trio `onlinebackup`/`deviceprotection`/`streamingmovies` (V = 0.23–0.29), `paperlessbilling` (V = 0.19), and the protected-attribute trio `seniorcitizen`/`has_partner`/`dependents` (V = 0.15–0.16). That's redundancy, not noise: the signal doesn't survive as *marginal* contribution once the rest of the feature set is already in the model — though not a simple one-bigger-feature story, since the add-on cluster's own rescue check above found no credit-splitting, and `seniorcitizen` is VIF-orthogonal to everything else (§8b `01-eda.ipynb`). `gender` and `phoneservice` are the one clean case where EDA and permutation importance both agree there's no signal at all (V ≈ 0.01, both fail to reject the null). `charge_per_service` — the sole engineered feature adopted during feature discovery, constructed in LAP 7 of `02a-feature-discovery.ipynb` — fails its own floor here (0.0010 vs. 0.0050, 46/100 fold stability) despite passing a different, incremental-signal screen at adoption time (§3a); `multiplelines` runs the opposite way, weak in EDA (V = 0.04) but real here. None of this changes the outcome: every one of the 20 stays regardless of its individual read, because the decision above is a single full-vs-reduced choice, not a per-feature filter.

**SHAP audit (diagnostic only).** A second, different lens on the same 20 features: `compute_shap_audit` fits one more default-config LightGBM and measures each feature's average contribution to individual predictions — not the performance drop from removing it, like permutation importance measures. It never decides keep/drop; it's a cross-check only. The two methods mostly agree on *which* features matter, but not completely: 9 of the 10 permutation-importance survivors occupy 9 of the top 10 SHAP ranks. The exception is `streamingtv`, the thinnest survivor (real importance 0.006, barely above the 0.005 floor), which falls to 12th by SHAP (mean |SHAP| 0.105) — behind two features that failed the decoy floor outright: `charge_per_service` (8th, 0.167) and `paperlessbilling` (11th, 0.113). Among the features both methods agree on, ordering still reshuffles: `contract_type` and `tenure` swap the top two spots (SHAP rates `contract_type` more than double `tenure`'s score — 1.115 vs. 0.465 — the reverse of permutation importance's order, 0.084 vs. 0.088), and `paymentmethod` jumps from a modest 7th-of-10 permutation-importance rank (0.013) to 3rd by SHAP (0.351) — plausibly because SHAP credits its role in individual predictions more than one shuffle-and-measure pass captures.

**Tradeoff and future consideration.** The full set's edge costs something too: 10 extra columns to validate and monitor, for a ~1.4% relative PR-AUC gain — a team prioritizing simplicity or a smaller fairness-audit surface (e.g. dropping `gender`) could reasonably prefer the reduced set instead, with a different Δ\* set in advance rather than a different reading of the same evidence. A finer-grained method like `RFECV` could identify exactly which 1-2 features are safe to drop individually, but isn't worth adopting now: run naively it reintroduces the stepwise-selection bias the single pre-registered test avoids, and run correctly (nested CV, to avoid that bias) it costs meaningfully more compute — either way, for a marginal payoff at this feature-set size. Worth revisiting only if serving cost or audit-surface reduction becomes a real priority.

The full feature-selection walkthrough — permutation importance, the fluke and correlated-credit checks, and the full-vs-reduced bootstrap decision — is in [`notebooks/03b-feature-selection.ipynb`](notebooks/03b-feature-selection.ipynb).

#### Protected attributes & fairness policy

All four protected / quasi-protected attributes — `gender` (sex), `seniorcitizen` (age), `has_partner` (marital status), `dependents` (familial status) — **remain model inputs; none is hand-excluded.**

1. **Benefit, not a denial.** The model drives a retention offer, not a credit or employment decision — the domains where statutory protection binds. Demographic targeting is standard practice in marketing.
2. **They carry genuine univariate churn signal** (three of four; see §2) — a real predictive case for eligibility, even though it is not the final word (see "Reasons for keeping the failed features" above).
3. **Fairness is enforced by measurement.** Per-group PR-AUC parity is evaluated across all four axes at candidate-selection time (§4, "Disaggregated robustness & fairness check") and again on the champion during error analysis (§7). Keeping the attributes available through candidate comparison and selection is what makes that measurement possible; the fairness *monitoring* commitment stands regardless of which axes feature selection ultimately keeps as model inputs.

### 4c. Hyperparameter Tuning (Optuna)

Tunes LightGBM only, on the input space frozen by Feature Selection above (§4b: `full_features_win`, all 20 features retained). PR-AUC (`average_precision`) is the sole study objective, consistent with the one-metric invariant.

Study configuration:

| Setting | Value |
|---|---|
| Algorithm | Tree-structured Parzen Estimator (TPE), `sampler_seed=42`, `n_startup_trials=10` |
| Trials | 50 requested |
| Pruner | `MedianPruner` — kills a trial mid-CV once it's clearly behind the running median |
| CV scheme | Single stratified 5-fold (`cv_folds=5`) — lighter than the candidate decision's repeated scheme, since 50 trials of early-stopped fits already sets the tuning budget |
| `n_estimators` | Not searched — set per fold by early stopping on `average_precision` (ceiling 2000, `early_stopping_rounds=50`); each trial's value is the median tree count across its own 5 folds |
| Search space | `num_leaves` (5–200), `learning_rate` (log-uniform, 0.005–0.1), `min_child_samples` (10–100), `subsample` (0.5–1.0), `colsample_bytree` (0.5–1.0), `reg_alpha` (0.0–10.0), `reg_lambda` (0.0–10.0), `max_depth` (3–12) |
| Best-trial rule | **1-SE** (Breiman/glmnet convention) — most-regularized trial within one standard error of the raw-best trial's own CV score, not bare argmax |

**Study hygiene (four checks recorded):**

1. **Pruning** — MedianPruner adopted; **47 of 50 trials completed, 3 pruned** mid-CV, saving the budget for configurations plausibly competitive with the running median.
2. **Boundary-hit check** — none of the 8 selected hyperparameters sit on its searched range's edge (all `False`); the widened ranges above comfortably contain the optimum.
3. **Selection rule** — raw argmax (trial 13, CV PR-AUC 0.6668) is **not** adopted. The 1-SE rule picks **trial 10** (CV PR-AUC 0.6621, within one SE = 0.0060 of trial 13) for far fewer `num_leaves` (6 vs. 151) and about 60% of the trees (94 vs. 161) — `num_leaves` is LightGBM's documented main complexity control under leaf-wise growth, though this particular study's own fANOVA ranking (now seeded for reproducibility) doesn't reflect that theoretical primacy: `max_depth` dominates at 22.5% of total importance, with `num_leaves` mid-pack at 9.9%. Net effect: a ~0.0047 PR-AUC sacrifice for a materially simpler configuration.
4. **Convergence** — the running-best curve is flat from trial 13 onward (every trial from 14 through 50 lands at or below it — a wide plateau, not a narrow cutoff); the study is not still climbing at the 50-trial budget.

**Selected hyperparameters (trial 10, 1-SE rule; MLflow run `tuning_study` → nested `trial_009`) vs. LightGBM defaults:**

| Hyperparameter | Default | Tuned (1-SE) |
|---|---:|---:|
| `num_leaves` | 20 | 6 |
| `learning_rate` | 0.1 | 0.0575 |
| `min_child_samples` | 50 | 74 |
| `subsample` | 1.0 | 0.8645 |
| `colsample_bytree` | 1.0 | 0.8856 |
| `max_depth` | −1 (unbounded) | 4 |
| `reg_alpha` | 0.0 | 0.7404 |
| `reg_lambda` | 0.0 | 3.5847 |
| `n_estimators` | 100 | 94 (median over folds) |

#### Full → reduced → tuned bias/variance progression

Three bias/variance reads chain across the decide→optimize boundary: ① the full-feature default model, ② the reduced-feature default model, and ③ the tuned model — all read via `diagnostics.generalization_gap` on the same `RepeatedStratifiedKFold(10×10)` scheme:

| Stage | Train PR-AUC | CV PR-AUC | Train − CV gap |
|---|---:|---:|---:|
| ① Full (20 features, default) | 0.7945 | 0.6583 | 0.1362 |
| ② Reduced (20 features, default) | 0.7945 | 0.6583 | 0.1362 |
| ③ Tuned (20 features, 1-SE) | 0.6944 | 0.6690 | **0.0254** |

Selection alone (①→②) shows no movement this cycle, since Feature Selection (§4b) retained the full set — confirming feature count was never the primary overfit lever here. Tuning (②→③) is what does the work: the gap **shrinks 81%** (0.1362 → 0.0254) while CV PR-AUC simultaneously **improves** (0.6583 → 0.6690) — the default model was overfitting, fitting the training data (0.7945) far better than it generalized to unseen data (0.6583). Tuning deliberately gives up some of that training fit (down to 0.6944) to curb the overfitting, and it pays off: performance on unseen data still goes up (0.6583 → 0.6690) rather than just holding steady. (The tuned CV PR-AUC here, 0.6690, is computed on the 10×10 RSKF scheme for this comparison and differs slightly from the Optuna study's own reported 0.6621, which is a single 5-fold value from a different fold assignment — not a discrepancy, a different measuring instrument.)

#### Model logging (not registered)

The tuned pipeline (`tree_preprocessor` + LightGBM at the 1-SE hyperparameters above, `n_estimators=94` fixed — no further early stopping on this final fit) is refit on all of development and logged as one `Pipeline` — not the bare estimator — onto the *same* MLflow run as the `tuning_study` parent (`run_id 6450decd98cc40ea96f49a6a9de56b39`), per the packaging checklist below:

| Check | Result |
|---|---|
| Model signature + input example | Logged (`mlflow.sklearn.log_model`) |
| Log → reload → predict parity | **Passed** — reloaded pipeline's predictions match the in-memory model exactly on a 5-row dev sample |
| `training_manifest.json` | Populated: git SHA, DVC data hash, hyperparameters, `feature_space` (20) / `feature_columns` (20), CV PR-AUC, paired-Δ vs LogReg with full bootstrap CI, and `tuning_summary` — the engineering audit trail |
| Registered | **Not at this stage** — logged only (`models:/m-30fc7b1c87c844ff98d6545d79277104`), no `registered_model_name`, no alias |

**This logged pipeline is the raw tuned model** — uncalibrated, un-thresholded, and not evaluated on the sealed test set. Nothing downstream should serve it directly: model calibration (`models/calibrate.py`) performs the training cycle's *single* registration — pointing the `challenger` alias at the calibrated artifact, not this one — Phase 7 evaluates on the sealed test once and promotes `champion`, and only `champion` is ever loaded by serving (Phase 9). Registering an intermediate, uncalibrated artifact here would leave two versions per cycle with no way to tell which is the valid rollback target — every registered version must be a valid one, and an uncalibrated intermediate is not.

The full hyperparameter-tuning walkthrough — the Optuna study, its hygiene checks, and the full→reduced→tuned bias/variance progression — is in [`notebooks/03c-hyperparameter-tuning.ipynb`](notebooks/03c-hyperparameter-tuning.ipynb).

---

## 5. Probability Calibration

A model can rank churners correctly while still getting their actual probabilities wrong — ranking and calibration are different things, and PR-AUC (§4's selection metric) only measures the first. That gap matters here: §6 decides whether to contact a customer by comparing expected benefit — churn probability times retention rate times customer value — against the cost of reaching out (`q·r·LTV > c` in §6's notation). That comparison only works if `q` is a real probability of churn, not just a relative score. And the raw probabilities really are off — LightGBM's imbalance handling (`class_weight='balanced'`, §4) skews every score toward the positive class as a side effect of correcting for class imbalance. This section fixes that distortion so §6's threshold is built on real probabilities, not skewed ones.

**Method:** The table below comes from a nested cross-validation, not a single `CalibratedClassifierCV` call. An **outer** `StratifiedKFold(5, shuffle=True, random_state=42)` drives `cross_val_predict` for every candidate — uncalibrated, sigmoid, isotonic — over the same fold assignment, so every row in the comparison is paired. For sigmoid and isotonic, what gets fit per outer fold is `CalibratedClassifierCV(pipeline, method=method, cv=StratifiedKFold(5, shuffle=True, random_state=42), ensemble=False)` — an **inner** 5-fold that happens to share the outer layer's exact parameters but is a distinct set of splits, used internally to fit the calibrator. `ensemble=False` means each per-fold `CalibratedClassifierCV` refits the base pipeline once on its own outer-training rows and fits one calibrator on that data's inner-OOF predictions — no separate validation split anywhere in the chain.

The winning method is then refit once more, on all of development, to produce what actually gets registered — the same `ensemble=False` collapse again yields a single `calibrated_classifiers_[0].estimator`, the uncalibrated `Pipeline` (preprocessor + LightGBM).

### Method selection (outer-OOF, development set)

| Method | Per-fold mean PR-AUC | Pooled Brier | ECE | BSS |
|---|---|---|---|---|
| Dummy prior (BSS reference) | 0.2654 | 0.1949 | 0.0000 | 0.0000 |
| Uncalibrated | 0.6669 | 0.1611 | 0.1446 | 0.1735 |
| **Sigmoid (selected)** | **0.6669** | **0.1345** | **0.0217** | **0.3098** |
| Isotonic | 0.6466 | 0.1339 | 0.0112 | 0.3131 |

**Selection: sigmoid, via `isotonic_disqualified_pr_auc_gate`.** Isotonic has the best Brier score and Expected Calibration Error (ECE) of the three, but its per-fold mean PR-AUC (0.6466) falls 0.0203 below uncalibrated — past the pre-registered materiality gate (Δ\* = 0.005) — so it's disqualified regardless of its calibration edge. Sigmoid leaves PR-AUC untouched by construction while still recovering most of isotonic's calibration gain.

**The PR-AUC gate decided this outright — the Brier-bootstrap switch test (`brier_switch_decision`) never ran.** That matters for how much weight to put on that test if a future retrain reaches it: it resamples only `outer_cv_folds` (= 5) paired fold-level Brier differences, and a 10,000-draw percentile bootstrap over just 5 original values has a small effective resample space (5⁵ = 3,125 distinct draws) — its CI is directionally sound but coarser than a bootstrap built from more units. It isn't live for the shipped v1 decision, since the PR-AUC gate resolved things first; it would only bind if a future retrain's isotonic candidate clears that gate and the choice actually comes down to the Brier bootstrap — worth raising `outer_cv_folds` at that point for a less noisy read.

**Raw Brier is hard to read in isolation** — its scale depends on the class base rate, not on how good the model is. `DummyClassifier(strategy='prior')`, cross-validated over the identical outer folds, gives a reference point: `BSS = 1 − Brier_candidate/Brier_dummy` is 0 by construction for that reference, and sigmoid's **BSS ≈ 0.31** means it recovers 31 % of the calibration skill achievable over just predicting the base rate for everyone.

**Expected Calibration Error (ECE)** — the weighted mean gap between each bin's predicted and observed rate, in probability points, 0 being perfect — is what the reliability diagram below visualizes bin-by-bin. Concretely: in the uncalibrated model, customers scored around 87 % likely to churn actually churned only about 76 % of the time — an ~11-point overconfidence gap, the largest of any bin. Sigmoid calibration corrects this across the whole curve, not just at that one extreme — its residual errors are small and go in both directions (a little low at low scores, a little high at high scores) rather than leaning the same way everywhere, which is the qualitative signature of a well-calibrated model.

The full calibration walkthrough — the reliability diagram and the method-selection diagnostics rendered from the actual run — is in [`notebooks/04-calibration-and-threshold.ipynb`](notebooks/04-calibration-and-threshold.ipynb).

---

## 6. Business Impact & Threshold Selection

A calibrated probability alone doesn't tell the retention team what to do — it needs a cutoff: contact the customer if their probability clears some threshold `t*`, leave them alone otherwise. §0 already derived the rule for where that cutoff belongs: contact a customer if and only if the expected payoff of doing so — their churn probability (`q`) times the retention rate (`r`) times their lifetime value (`LTV`) — exceeds the total intervention cost (`c`). That collapses to a closed form, `t* = c / (r × LTV)`. What §0 didn't have was any of the actual numbers: `q` wasn't a calibrated probability yet (§5 fixes that), and `r`, `LTV`, and `c` were still unresolved. This section supplies both, resolving the cost parameters from `configs/costs.yaml` and plugging the calibrated `q` into the rule, to ship the real operating threshold.

### Cost parameters, from `configs/costs.yaml`

Three scenarios bracket realistic business assumptions, resolved from the development-set churner population. Beyond `r`, two more quantities feed `t* = c/(r·LTV)` — `LTV` and `c` — and Average Revenue Per User (ARPU) is the starting point for both:

> **`LTV = ARPU × gross_margin × horizon_months`** — ARPU is a monthly *revenue* rate, sourced from `MonthlyCharges`; multiplying by `gross_margin` converts it to a monthly *profit* rate, which is then projected forward across the 1-year horizon.

> **`c = outreach_cost + (ARPU × discount_rate × discount_months)`** — the intervention's total cost is *contacting* the customer (`outreach_cost`) plus the *discount offered* as the retention incentive.

`discount_months` (3) is fixed across all three scenarios too. `outreach_cost`, `discount_rate`, and `retention_rate` vary by scenario, from a cheaper, less generous, less effective conservative intervention through the base case to a costlier, more generous, more effective optimistic one.

**`r` is different from the other parameters — it can't be measured at all, not just from this dataset** (see Known Limitations below). It's taken from a literature range instead, which is why the sensitivity sweep below is a headline result, not a footnote.

| Parameter | Conservative | Base | Optimistic |
|---|---|---|---|
| ARPU quantile of churner `MonthlyCharges` | P25 | P50 | P75 |
| ARPU | $56.68 | $79.60 | $94.20 |
| `gross_margin` | 0.60 | 0.60 | 0.60 |
| `horizon_months` | 12 | 12 | 12 |
| **1-year LTV** = `ARPU × 0.60 × 12` | **$408.06** | **$573.12** | **$678.24** |
| `outreach_cost` | $5 (automated SMS/email) | $20 (call-centre agent) | $50 (retention specialist) |
| `discount_rate` | 10% | 20% | 30% |
| `discount_months` | 3 | 3 | 3 |
| Discount offer = `ARPU × discount_rate × 3` | $17.00 | $47.76 | $84.78 |
| **Total intervention cost `c`** = `outreach_cost + discount offer` | **$22.00** | **$67.76** | **$134.78** |
| `retention_rate` `r` (industry benchmark — not observable in this dataset) | 20% | 30% | 40% |

### Derived thresholds

Each scenario's threshold is checked two ways. The **closed form** calculates it directly from the cost math above (`c/(r·LTV)`) — pure arithmetic, no data involved. The **empirical check** goes the other direction: it tries every possible cutoff on real, calibrated probabilities to find which one would actually have performed best historically, then resamples that estimate thousands of times to build a plausible range (a 95% confidence interval), since a single historical best is noisy. The question that matters: does the closed-form answer fall inside that range?

| Scenario | `t*` (closed form) | Empirical argmax-EV | 95% bootstrap CI | Within CI? | Implied contact rate |
|---|---|---|---|---|---|
| Conservative | 0.2696 | 0.2672 | [0.212, 0.332] | ✓ | 40.3% |
| **Base (shipped)** | **0.3941** | 0.4428 | [0.364, 0.532] | ✓ | 30.8% |
| Optimistic | 0.4968 | 0.5515 | [0.494, 0.593] | ✓ | 24.5% |

All three scenarios' closed-form `t*` falls inside the bootstrap CI of the empirical argmax-EV threshold — the theoretical cost-derived cutoff agrees with where realized expected value actually peaks on held-out data, for every scenario, not just the shipped one.

**Base is the shipped operating point** (`t* = 0.3941`) — the other two are stored reference alternatives, not automatically-active fallbacks.

Neither extreme pays: contacting everyone costs money outright (sharply for base/optimistic, less so for conservative's cheaper contacts), and contacting no one gives up on every customer who could have been talked into staying. Between those extremes, expected value stays close to its best across a **broad plateau** of cutoffs before dropping to $0 near `t = 1` — nearby thresholds perform almost as well as the peak, so a slightly-off `t*` costs little, which is why each scenario's `t*` doesn't need to sit exactly at the empirical peak to be a good choice.

### Retention-rate sensitivity (base scenario, cost/LTV held fixed)

§0 already flagged `r` — the retention rate — as the model's dominant source of uncertainty; this table makes that concrete. Holding the base scenario's cost and LTV fixed, it re-derives `t*` — the operating threshold — at five different assumed values of `r`, showing how much the shipped threshold would move if the true retention rate turns out different from the 0.30 assumption.

| `r` | 0.15 | 0.20 | **0.30** | 0.40 | 0.45 |
|---|---|---|---|---|---|
| `t*` | 0.788 | 0.591 | **0.394** | 0.296 | 0.263 |

Halving the assumed retention rate exactly doubles the threshold — `t*` and `r` are related by pure inverse proportionality (`c` and `LTV` held fixed), since each contact is "worth less" in expectation when it's less likely to work. This is the single most consequential unmeasured assumption in the whole cost model, and the direction of the error matters: if real-world `r` turns out lower than the assumed 0.30, the shipped threshold is too low and the team contacts more customers than the offer's actual success rate justifies; if `r` is higher, the threshold is too conservative and leaves customers uncontacted who could have been saved — the table above brackets both directions, from 0.263 (`r`=0.45) to 0.788 (`r`=0.15).

The full threshold derivation — the per-scenario cost breakdown, the closed-form-vs-empirical-argmax comparison, the expected-value curves, and the retention-rate sensitivity sweep, all rendered from the actual run — is in [`notebooks/04-calibration-and-threshold.ipynb`](notebooks/04-calibration-and-threshold.ipynb).

### Known limitations of the cost model

- **Horizon/offer-length mismatch.** LTV is credited over the full `horizon_months` (12) once a customer is retained, but the discount that (in expectation) retains them only runs for `discount_months` (3). The model implicitly assumes a 3-month offer buys a full year of loyalty; if churn risk resumes once the discount lapses, realized LTV recovered per successful intervention is overstated.
- **`r` is a single global constant, not a per-customer effect.** The true retention probability almost certainly varies by customer (contract type, tenure, price sensitivity) — this is a uniform-treatment-effect assumption, not an estimated uplift function `τ(x)`. A customer-level uplift model would let `q·r(x)·LTV(x) > c` replace the current population-average rule.
- **`r` cannot be recovered from this project's data, even under future retraining** — the raw dataset is a static snapshot with no experimental holdout or live customer feed, and resolving it requires an actual randomized retention-offer experiment (out of scope here). It's still the single highest-leverage input to validate, given how directly it moves `t*` (the sensitivity sweep above) — and since `t*` depends only on the cost parameters, not the model, a measured `r` could simply replace the assumed value in `configs/costs.yaml`, with no re-tuning, re-calibration, or model change required.

---

## 7. Final Test-Set Results

> **⚠ Archived pass — this whole section, including its Error Analysis and SHAP Explainability subsections, describes the pre-Phase-6 notebook's own evaluation of `best_pipe`/`cal_pipe` at threshold 0.2956, not this project's real pipeline.** None of the numbers below come from the actual `telco-churn-pipeline` registry, the real calibrated `challenger`, or the closed-form threshold derived in §6. They are kept as the historical record of the archived pass — the archive is a comparison notebook, not a bar current work must clear. This entire section is rewritten once Phase 7's `evaluate.py` and `05-error-analysis.ipynb` run for real against the sealed test set and the actual champion.

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

### Full model performance history

| Stage | Threshold | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline LightGBM (val) | 0.35 | 0.900 | 0.495 | 0.638 | 0.867 |
| Tuned LightGBM (val) | 0.35 | 0.940 | 0.405 | 0.566 | 0.855 |
| Calibrated (val) | 0.35 | 0.760 | 0.576 | 0.655 | 0.856 |
| Calibrated + OOF threshold (val) | 0.2956 | 0.800 | 0.531 | 0.638 | 0.856 |
| **Final test set** | **0.2956** | **0.7861** | **0.5241** | **0.6289** | **0.8413** |

### Error Analysis

All analysis is on the **tuned `best_pipe`** on the val set (threshold = 0.35). Calibration preserves rank order but cannot be used for SHAP (see the SHAP Explainability subsection below).

#### Missed churners — profile (9 FNs, 6.0 % FN rate)

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

#### False positive profile (207 FPs, 50 % FP rate)

FPs have a nearly identical feature profile to actual churners: 80.7 % month-to-month, 53 % fiber optic, 69 % no online security. There is no clean spatial boundary separating FPs from true negatives — the model cannot distinguish the minority who share the risk profile but remain loyal. Resolving this requires loyalty or satisfaction signals not present in the current feature set.

#### Subgroup FN rates

| Subgroup | FN rate | Interpretation |
|---|---|---|
| Contract = Two year | 1.000 | Near-total blind spot |
| Contract = One year | 0.538 | Major blind spot |
| tenure 49–72 months | Highest by band | Highest-LTV customers; hardest to anticipate |
| DSL / No internet | Above average | Model under-serves lower-cost segments |
| Fiber optic | Lowest | Classic risk profile — thoroughly learned |
| SeniorCitizen = 1 | Low (similar to non-seniors) | No systematic age-based under-service — positive fairness result |

### SHAP Explainability

Using `shap.TreeExplainer` on `best_pipe` (exact tree values, not kernel approximation). Analysis conducted on both val and test sets; top-10 ranking is identical across both — confirming pipeline consistency.

#### Global feature importance (mean |SHAP| — top 10)

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

#### Key interactions confirmed by dependence plots

- **tenure × Contract:** Month-to-month customers retain high positive SHAP even at moderate tenure — the contract type sustains churn risk regardless of seniority.
- **MonthlyCharges × Fiber optic:** Fiber optic customers show systematically higher SHAP at the same charge level — two risk factors compound rather than add.

#### Individual explanations (waterfall plots)

**High-confidence churner (score = 0.790):** month-to-month + 1-month tenure + fiber optic + no online security + no tech support — every feature points in the same direction.

**High-confidence non-churner (score = 0.148):** Not month-to-month = largest single protective factor; reinforced by two-year contract, 70-month tenure, $19.80/month, security and tech support subscribed.

**Most convincingly missed churner (FN, score = 0.025):** Holds no month-to-month contract flag (SHAP −0.62); two-year contract (−0.13); tenure = 55 months (−0.07). The model's "annual contract = committed customer" heuristic overpowers all other signals.

#### EDA vs SHAP agreement

| Feature | EDA rank | SHAP rank | Divergence |
|---|---|---|---|
| Contract | 1 | 1 | Agreement |
| tenure | 2 | 4 | OnlineSecurity ranks above tenure in SHAP — marginal contribution in conditional context |
| OnlineSecurity | 3 | 2 | Elevated vs bivariate rank |
| TechSupport | 4 | 5 | Agreement |
| InternetService | 5 | 3 | Agreement |
| gender | Non-significant | #23 (≈0) | Agreement |
| SeniorCitizen | Confounded | #40 (≈0) | Agreement — mediation confirmed |

#### SHAP on the test set

SHAP is applied to the LightGBM base estimator extracted from fold 0 of `cal_pipe`. Top-10 feature ranking on the sealed test set is **identical** to the validation-set ranking — zero rank shifts across the top 10. SHAP base value: **−0.3009** (log-odds space).

> **⚠ "Fold 0" is an artefact of the archived `ensemble=True` calibrator and must not be carried into `src/`.** Under Phase 6's `ensemble=False` there is one base estimator, not five, and the access path is `calibrated.calibrated_classifiers_[0].estimator` — no fold indexing, no averaging, no `best_pipe` kept alongside. See the note in §5. Phase 7's error analysis is written against that path.

**Protected attributes:**

| Attribute | Mean \|SHAP\| | Rank of 40 | Assessment |
|---|---|---|---|
| `Gender: Male` | 0.0000 | #23 | Negligible |
| `Partner: Yes` | 0.0000 | #21 | Negligible |
| `SeniorCitizen` | 0.0000 | #40 | Negligible (last) |

All three protected attributes contribute zero marginal signal at the individual prediction level — consistent with the validation-set finding.

---

## 8. Production Refit & Model Registration

> **⚠ Archived pass — this whole section describes the archived notebook's own full-data refit and registration, not this project's real pipeline.** No production refit has happened yet in `src/` — that requires `refit.py` and `register.py`, which run after `evaluate.py` scores the sealed test set (§7, also archived). The run ID, threshold progression (0.2956 → 0.3596), and `champion`/`challenger` alias state below are all from the archived notebook. As of §6, the real registry holds one version of `telco-churn-pipeline`, aliased only `challenger` — no `champion` exists yet. This section is rewritten once evaluation, refit, and registration run for real.

### Rationale

All hyperparameter, calibration, and threshold decisions were finalised on held-out data before the production refit. Retraining on the full 7,043-customer dataset provides the production model with all available signal without contaminating the §7 benchmark figures.

### Full-data refit

| Property | Value |
|---|---|
| Training data | Full dataset: train + val + test (7,043 rows; 1,869 churners, 5,174 non-churners) |
| Model architecture | LightGBM + sigmoid calibration (`CalibratedClassifierCV`, cv=5) |
| Hyperparameters | Unchanged from Optuna best (§4) — no re-tuning on full data |
| `scale_pos_weight` | Recalculated on full-data class ratio (5,174 / 1,869 ≈ 2.77) |
| Features | 40 (same as training pipeline) |
| MLflow run | `f81665fa` (experiment `"Telco Churn - Final Model"`) |

### Production threshold re-derivation

The §6 OOF threshold (0.2956) was derived on `X_train` only. With the full dataset, OOF cost minimisation was re-run under the base scenario:

| Threshold source | Threshold | Dataset | Notes |
|---|---|---|---|
| §6 OOF cost min | 0.2956 | Train set only | Used for §7 test-set evaluation |
| **Production OOF cost min** | **0.3596** | **Full dataset** | **Production-shipped value** |

The production threshold rises by 0.064, reflecting the broader distribution of the full dataset (proportionally more long-tenure, non-month-to-month customers shift the cost-optimal point rightward).

### OOF performance on full data (threshold = 0.3596)

| Metric | OOF value |
|---|---|
| Recall | 0.717 |
| Precision | 0.569 |
| F1 | 0.634 |
| PR-AUC (avg precision) | 0.644 |

> OOF metrics on the full dataset are not directly comparable to §7 test-set figures — the test set is now included in training. These are a distributional sanity check only; the §7 test-set numbers remain the authoritative benchmark.

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

## 9. Known Limitations

1. **Annual/multi-year contract churners are a near-total blind spot.** One-year and two-year contract holders have FN rates of 0.972 and 1.000 respectively. The model learned "annual contract = committed customer" as a near-irrefutable heuristic.

2. **Long-tenure, low-cost segment under-served.** Customers in the 25+ month tenure band have a 0.465 FN rate. Long tenure is treated as a loyalty signal, but duration can be a lagging indicator for quietly disengaging customers.

3. **No uplift / persuadability modelling.** The model identifies *who will churn*, not *who will respond to a retention offer*. Without A/B test data, the model cannot separate persuadables from lost causes.

4. **Residual calibration gap at high scores.** The calibrated model under-predicts churn probability for high-scoring customers by up to ~10 pp above score ≈ 0.58. Expected-value calculations will understate financial exposure for the highest-risk customers.

5. **No re-contact suppression.** Customers flagged in one cycle will appear in the next scoring run with the same feature profile. Without a 90-day suppression window, the system risks contact fatigue.

6. **Production monitoring not yet deployed.** No PSI or prediction quality monitoring is currently in place to trigger re-training.

7. **Feature set cannot reduce FPs for loyal high-risk-profile customers.** ~50 % of non-churners who share the fiber optic + month-to-month + no add-on profile are incorrectly flagged. No loyalty, satisfaction, or recency-of-service-change signals are available.

8. **Business cost parameters are illustrative.** The $68/intervention and $575/LTV figures are reasonable but not Finance-validated.

9. **Feature discovery redundancy screen has a mixed-type gap.** Screen 2 in the lap framework checks numeric-vs-numeric relationships (Pearson) and categorical-vs-categorical (Cramér's V) but has no branch for categorical-derived-from-numeric (e.g. `tenure_cohort` vs `tenure`). Screen 4 — permutation importance given all adopted features — is the empirical backstop: a feature that adds no marginal signal because the model already has the underlying numeric column is identified and rejected regardless of type. On the dev-partition run, `tenure_cohort` (Lap 5) was redundant enough to fail earlier, directly at Screen 3 (PR-AUC fell −0.0041) — Screen 2 raised no flag, since it has no cross-type branch, but the feature never reached Screen 4. `two_year_fiber` (Lap 1) demonstrates the Screen 4 backstop directly on this run: Screen 2 passed (max_corr 0.389), Screen 3 passed (+0.0017 PR-AUC), and Screen 4 correctly rejected it (importance 0.0001, below the 0.0054 floor) once measured against the full adopted context.

10. **Learning curve had not plateaued at Phase 5 Step 2c.** CV PR-AUC was still rising at the maximum training size in the Steps 2c/2d generative diagnostic loop (0.613 → 0.655 from 20%→100% of the dev-training folds) — more historical data would plausibly still improve ranking quality. Not acted on in Phase 5 (no feature was engineered in response); flagged as a Phase 10 retrain / data-acquisition consideration. **This is the empirical justification for the Phase 7 full-data refit** (`models/refit.py`): on this project's own evidence the extra 1,409 rows are still buying ranking quality, so the refit is not a ritual.

11. **Calibrated probabilities are calibrated *to development-set prevalence* — prevalence drift invalidates both the calibration and the threshold.** LightGBM trains with `class_weight='balanced'` (`models/train/common.py`), which systematically inflates scores toward the positive class. `CalibratedClassifierCV` corrects this because sklearn fits the calibrator on **unweighted** out-of-fold data, mapping the reweighted scores back to true posteriors against the real ~26.5 % dev prevalence. That is the right behaviour, and it is the mechanism that makes the closed-form threshold `t* = c/(r × LTV)` valid at all — `t*` is only Bayes-optimal against *honest* posteriors. The dependency runs both ways: if production churn prevalence shifts away from 26.5 %, the calibration map is stale, the posteriors are biased, and `t*` is being applied to numbers that no longer mean what it assumes. Neither a PR-AUC check nor a reliability diagram computed on old data will catch this. **This is the hook for Phase 13 drift monitoring:** track prevalence alongside feature PSI, and treat a sustained prevalence shift as a re-calibration trigger, not merely a retrain trigger — they are different remedies.

---

## 10. Recommendations & Next Steps

### Immediate (model v1 deployment)

1. ~~**Deploy at threshold 0.2956** for the base cost scenario.~~ **Superseded — see §0 and §6.** 0.2956 was selected by minimising a cost function that charges the intervention only to false positives; at that threshold the marginal contact has an expected value of **−$17.01**. The base-scenario operating point is `t* = c/(r × LTV) = `**`0.3941`**, derived and shipped from `configs/costs.yaml` (§6). Revisit whenever the cost parameters change — and note that `t*` is far more sensitive to the retention rate `r` than to anything the model does.
2. **Implement tiered outreach:** cheap outreach (email/SMS) for scores in the band just below the deployed threshold; expensive retention offers only for high-confidence scores. *(The archived 0.30–0.50 band was anchored to the superseded threshold — bands should be re-derived from the real `t* = 0.3941`.)*
3. **Apply a 90-day re-contact suppression window** to prevent intervention fatigue.

### Short-term (next model iteration)

4. **Address annual-contract blind spot:** Explore contract-type-specific sub-models or a lower intervention threshold applied selectively to annual-contract holders who show secondary risk signals.
5. **Instrument an A/B test** on the first cohort of flagged customers. This data is the prerequisite for uplift modelling.
6. **Add engagement/recency features** (last service change date, usage trend, support ticket volume) to separate loyal high-risk-profile customers from genuine pre-churners.

### Ongoing

7. **Deploy PSI monitoring** for score distribution drift. Trigger re-evaluation when PSI > 0.2.
8. **Recalibrate the threshold annually** using updated OOF predictions on a full-data refit.
9. **Replace illustrative business parameters** with Finance-validated figures before committing to P&L projections.
