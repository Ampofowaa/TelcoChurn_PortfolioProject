# Telco Customer Churn — End-to-End ML Project

> **Predict which telecom customers are likely to churn and quantify the revenue impact of early intervention.**
> Full MLOps lifecycle: data validation → EDA → feature engineering → model selection → hyperparameter tuning → error analysis → SHAP explainability → probability calibration → cost-sensitive threshold optimisation → full-data production refit + model registration.

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Pipeline Overview](#2-pipeline-overview)
3. [EDA & Statistical Testing](#3-eda--statistical-testing)
4. [Data Quality & Missing Values](#4-data-quality--missing-values)
5. [Feature Engineering](#5-feature-engineering)
6. [Baseline Models](#6-baseline-models)
7. [Hyperparameter Tuning (Optuna)](#7-hyperparameter-tuning-optuna)
8. [Error Analysis](#8-error-analysis)
9. [SHAP Explainability](#9-shap-explainability)
10. [Probability Calibration](#10-probability-calibration)
11. [Business Impact & Threshold Selection](#11-business-impact--threshold-selection)
12. [Final Test-Set Results](#12-final-test-set-results)
13. [Production Refit & Model Registration](#13-production-refit--model-registration)
14. [Known Limitations](#14-known-limitations)
15. [Recommendations & Next Steps](#15-recommendations--next-steps)
16. [Tech Stack](#16-tech-stack)
17. [Quick Start](#17-quick-start)
18. [Project Structure](#18-project-structure)

---

## 1. Dataset

| Property | Detail |
|---|---|
| Source | IBM Telco Customer Churn (public) |
| Rows | 7,043 customers |
| Features | 20 (demographics, account info, 9 service flags) |
| Target | `Churn` — left within the last month |
| Class split | 73.5 % No / 26.5 % Yes (2.8:1 imbalance) |
| Missing data | 11 `TotalCharges` NaN (0.16 %) — zero-tenure customers with no first bill |

**Feature groups:**
- **Target:** `Churn` (Yes/No)
- **Services (9):** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account (6):** tenure, Contract, PaymentMethod, PaperlessBilling, MonthlyCharges, TotalCharges
- **Demographics (4):** gender, SeniorCitizen, Partner, Dependents

**Numeric summary:**

| Feature | Mean | Std | Range |
|---|---|---|---|
| tenure | 32.4 mo | 24.6 | 0–72 |
| MonthlyCharges | $64.76 | $30.09 | $18.25–$118.75 |
| TotalCharges | — | — | Skewed right; long tail |

---

## 2. Pipeline Overview

```
Data Loading → Data Validation (5 gates) → EDA + Statistical Testing
    → Preprocessing → Baseline Models (3) → Feature Engineering (error-driven)
    → Model Selection → Optuna Tuning (50 trials)
    → Error Analysis → SHAP Explainability → Probability Calibration
    → Cost-Sensitive Threshold → Business Impact Analysis
    → Final Test-Set Evaluation (sealed) → Production Refit → Model Registration
```

### Data splits

All splits are 3-way stratified and fixed before any modelling begins. The test set is sealed until the final evaluation step.

| Split | n | Churners | Role |
|---|---|---|---|
| Train | 5,070 | ~26.5 % | Model fitting + CV folds |
| Val | 564 | 150 (26.6 %) | All diagnostics (§§11–14, never used for selection) |
| Test | 1,409 | 374 (26.5 %) | Final evaluation only — sealed until all decisions finalised |

---

## 3. EDA & Statistical Testing

Statistical significance and effect sizes validated with **Chi-square + Cramér's V** (categorical) and **Mann–Whitney U + rank-biserial r** (numeric) on the full 7,043 rows. With n = 7,043, effect size magnitude is the primary lens — p-values are all < 0.001 and are informative only as a filter.

### Top predictors by effect size

| Feature | Method | Effect size | Key finding |
|---|---|---|---|
| Contract type | Cramér's V | **0.41** (strong) | Month-to-month: ~43 % churn; One year: ~11 %; Two year: < 3 % |
| tenure | Rank-biserial r | **0.48** | Churners leave at median ~10 months; non-churners average 38 months |
| OnlineSecurity | Cramér's V | 0.35 (strong) | Churn ~2× higher without the service |
| TechSupport | Cramér's V | 0.34 (strong) | Churn ~2× higher without the service |
| InternetService | Cramér's V | 0.32 (strong) | Fiber optic carries disproportionately high churn |
| PaymentMethod | Cramér's V | 0.30 (strong) | Electronic check is the highest-churn payment method |
| TotalCharges | Rank-biserial r | 0.30 | Churners accumulate less ($1,532 vs $2,555) before leaving |
| MonthlyCharges | Rank-biserial r | −0.24 | Churners pay *more* per month ($74 vs $61) — fiber optic concentration |
| OnlineBackup | Cramér's V | 0.29 | Near-strong predictor |
| DeviceProtection | Cramér's V | 0.28 | Near-strong predictor |

### Non-predictors

- **PhoneService** (V = 0.01, p = 0.34) and **gender** (V = 0.008, p = 0.49) are statistically non-significant.
- `PhoneService_Yes` has the highest finite VIF (~1,774) yet is not predictive of churn — a clean illustration of why multicollinearity screening does **not** replace significance testing. VIF measures collinearity between features, not relevance to the target.

### Key interaction effects

- **Senior citizens** show higher churn not as an independent age effect but because 70.7 % of seniors are on month-to-month contracts (vs 52.0 % for non-seniors) and 72.8 % are on fiber optic (vs 38.4 %). The age signal is entirely mediated by contract type and internet service. `SeniorCitizen` does not appear in the top 15 SHAP features.
- **Fiber optic churn** operates through two overlapping channels: a cost channel (average $91.50/month vs $58.10 for DSL, a 57 % monthly premium) and a potential service quality channel. Both `InternetService_Fiber optic` and `MonthlyCharges` retain independent gain and SHAP signal, confirming compounding rather than additive risk.
- **TotalCharges** remains independently informative despite high VIF (10.8). Only 8.7 % of customers have TotalCharges = MonthlyCharges × tenure exactly; mean billing deviation is $45.09, confirming TotalCharges captures genuine plan history beyond simple arithmetic.

### Charge feature directionality (important)

- High `MonthlyCharges` → **higher churn risk** (fiber optic concentration)
- High `TotalCharges` → **lower churn risk** (only long-tenure customers accumulate high totals)
- These move in opposite directions because they measure different things. Both are retained.

### Multicollinearity (VIF)

Several features have infinite or very high VIF due to structural dependencies (all add-on `_No internet service` dummies are linearly dependent on `InternetService_No`). This is a linear model problem only. **Tree-based ensembles split one feature at a time and are immune to collinearity.** No columns were dropped on this basis.

---

## 4. Data Quality & Missing Values

**Five automated pass/fail gates — all PASS:**

| Check | Status | Detail |
|---|---|---|
| No duplicate customerIDs | PASS | 0 duplicates |
| Missing values < 1% per column | PASS | Max 0.16 % in `TotalCharges` |
| tenure ≥ 0 | PASS | 0 negative values |
| MonthlyCharges > 0 | PASS | 0 non-positive values |
| Churn values in {Yes, No} | PASS | Exactly two unique values |

**Missing value treatment:**
- 11 `TotalCharges` NaN rows (0.16 %) belong to customers with `tenure = 0` — first bill not yet issued.
- **Decision: rows retained**; imputed via `SimpleImputer(strategy='median')` inside the preprocessing pipeline.
- No outliers removed: IQR bounds encompass the full range for all three numeric features. Tree-based ensembles are robust to extreme values.

---

## 5. Feature Engineering

Feature engineering was driven by **FN profiling** on the baseline model (error-driven iteration) rather than domain intuition alone.

### Identified blind spots (val set, baseline LightGBM)

| Subgroup | FN rate | n (val) | Reliability |
|---|---|---|---|
| Contract = One year | 0.538 | 13 | Most reliable problem area |
| DSL + tenure ≥ 25 months | 0.333 | 6 | Directional, small n |
| Month-to-month | 0.037 | — | Best-caught segment |
| Fiber optic | 0.019 | — | Best-caught segment |

The model thoroughly learns the high-risk profile (short tenure + fiber optic + month-to-month) but applies a strong "loyal customer" prior to customers who churn after years of moderate-cost service.

### Three hypothesis-driven features tested

| Feature | Construction | Hypothesis |
|---|---|---|
| `is_long_month_to_month` | tenure > 24 AND month-to-month | Long-tenured month-to-month customers have low tenure signal but still lack exit barriers |
| `charge_per_service` | MonthlyCharges ÷ count of active services | Over-charged, under-served customers — high cost relative to value received |
| `monthly_to_total_ratio` | MonthlyCharges ÷ TotalCharges | Isolates newly-expensive customers with insufficient billing history |

### Adoption decision

| Metric | Baseline | With engineered features | Gate |
|---|---|---|---|
| Overall CV recall | 0.8431 | 0.8394 | PASS (Δ < 0.005 tolerance) |
| Subgroup recall (benchmark FN group) | 0.3333 | 0.1667 | **FAIL (direction reversed)** |

**Decision: all three engineered features discarded.** Both gates must pass; subgroup gate failed.

**Root cause:** LightGBM constructs interaction splits internally. H1/H2/H3 interactions are already approximated via tree splits on the raw features; explicit engineering added redundant signal and marginal multicollinearity without improving the target blind spot.

**Limitation acknowledged:** The DSL + long-tenure blind spot is real but the val subgroup is too small (n = 6) for reliable measurement. This is a first-pass finding, not a closed issue.

---

## 6. Baseline Models

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

## 7. Hyperparameter Tuning (Optuna)

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

## 8. Error Analysis

All analysis is on the **tuned `best_pipe`** on the val set (threshold = 0.35). Calibration preserves rank order but cannot be used for SHAP (see §9).

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

## 9. SHAP Explainability

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

## 10. Probability Calibration

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

## 11. Business Impact & Threshold Selection

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

## 12. Final Test-Set Results

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

The model delivers **+27 percentage-point improvement** over random targeting at the same 40 % budget. Headroom to the perfect model at 40 % budget: 21.4 pp.

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

### SHAP Explainability on the Test Set

SHAP analysis is applied to the LightGBM base estimator extracted from fold 0 of `cal_pipe` (feature names are identical across all five calibration folds). Feature names are stripped of pipeline prefixes (`onehot__`, `numeric__`, `binary__`) for readability. SHAP base value on the test set: **−0.3009** (log-odds space).

#### Global feature importance — consistency with validation

The top-10 feature ranking on the sealed test set is **identical** to the §9 validation-set ranking, with zero rank shifts across the top 10. This confirms no feature inconsistency between the training and production pipelines.

| Rank | Feature | Direction | Mean \|SHAP\| |
|---|---|---|---|
| 1 | `Contract: Month-to-month` | Positive | ~0.57 (~3× rank #2) |
| 2 | `OnlineSecurity: No` | Positive | — |
| 3 | `InternetService: Fiber optic` | Positive | — |
| 4 | `Tenure` | Negative | — |
| 5 | `TechSupport: No` | Positive | — |
| 6 | `MonthlyCharges` | Positive | — |
| 7 | `PaymentMethod: Electronic check` | Positive | — |
| 8 | `Contract: Two year` | Negative | — |
| 9 | `StreamingMovies: Yes` | Positive | — |
| 10 | `TotalCharges` | Negative | — |

`Contract: Month-to-month` dominates at mean |SHAP| ≈ 0.57 — approximately three times the next feature. All beeswarm directions mirror the §9 validation analysis.

#### Individual prediction explanations — all four outcomes

**Missed churner — FN (churn probability 0.025, below threshold 0.2956)**

Three suppressors stack to push the churn probability far below threshold. `Contract: Month-to-month = 0` contributes SHAP −0.62 — the model's single largest loyalty signal. `Contract: Two year = 1` adds a second strong suppressor (SHAP −0.13), and `Tenure = 55 months` reads as embedded loyalty rather than a lagging indicator of disengagement (SHAP −0.07). The absence of fiber optic removes the model's primary churn risk signal entirely. All three blind-spot conditions are simultaneously active — non-monthly contract, long tenure, and non-fiber internet — compounding into a churn probability the model has no mechanism to push above threshold.

**Over-flagged non-churner — FP (churn probability 0.732, above threshold 0.2956)**

Every risk factor the model has learned activates simultaneously. `Contract: Month-to-month = 1` dominates (SHAP +0.65), compounded by 1 month of tenure (+0.27), `OnlineSecurity: No` (+0.19), `InternetService: Fiber optic` (+0.16), and `TechSupport: No` (+0.11). The model is not wrong to flag this profile; the FP arises because within this high-risk configuration a minority of customers stay, and no available feature in the current set separates stayers from leavers.

**Most confident correct churn flag — TP (churn probability 0.738, above threshold 0.2956)**

The feature pattern is near-identical to the FP case above. `Contract: Month-to-month = 1` dominates (SHAP +0.59), compounded by very short tenure (+0.38), `OnlineSecurity: No` (+0.11), `InternetService: Fiber optic` (+0.09), and `TechSupport: No` (+0.07). The TP and FP cases are distinguishable only by ground truth — not by any feature the model has access to. Both are new month-to-month fiber subscribers with no add-on services; the two outcomes reflect the model operating at the limit of what the current feature set allows.

**Most confident correct non-churn flag — TN (churn probability 0.020, below threshold 0.2956)**

The same suppressor stack as the FN case, applied to a customer who genuinely stays. `Contract: Two year = 1` dominates as the model's strongest loyalty signal, reinforced by long tenure and the absence of fiber optic. The TN and the FN above share a near-identical feature profile — long contract, long tenure, non-fiber internet — and the model produces the same low churn score for both. The difference is ground truth: here the suppressors are working as intended. The same mechanism that creates the annual-contract blind spot produces the correct answer when the customer's intent aligns with what the features signal.

#### Blind-spot segment analysis

SHAP is computed separately for each FN sub-segment identified in §8 and §12 Segment Breakdown, revealing the specific suppressor combinations that prevent the model from detecting churn in each group.

**Non-monthly contract FNs (n = 44)**

`Contract: Month-to-month = 0` (all blue, far left) dominates every customer in this segment — the model has learned that non-monthly contracts overwhelmingly predict loyalty and assigns a large negative SHAP that pushes the churn score far below threshold. Two-year holders are maximally suppressed: `Contract: Two year = 1` adds a second large negative push, with no offsetting signal of comparable magnitude. One-year holders receive a small positive SHAP from `Contract: One year = 1`, which partially offsets the dominant suppression but falls far short of threshold. The root cause: all 44 customers genuinely churn, yet `Month-to-month = 0` is strong enough to override every other risk signal available to the model. The heuristic is statistically correct in aggregate — non-monthly contracts do predict loyalty most of the time — but the model has no mechanism to identify the minority who hold a non-monthly contract and churn anyway.

**Long-tenure (≥ 25 mo) FNs (n = 46)**

`Contract: Month-to-month` remains the dominant feature. The majority of long-tenure FNs also hold non-monthly contracts, so the contractual suppressor and the tenure suppressor stack simultaneously — the same compound blind spot observed in the FN waterfall above, now confirmed across 46 customers. A smaller sub-group holds month-to-month contracts; their positive M2M SHAP (≈ +0.3) is visible but insufficient to cross threshold when compounded with long-tenure suppression. Secondary risk signals — `OnlineSecurity: No`, `InternetService: Fiber optic`, high `MonthlyCharges` — are present but too weak to overcome the joint suppression. Long tenure operates as a lagging indicator: by the time a customer has 25+ months of history, the model reads duration as loyalty rather than a risk of quiet disengagement.

**DSL FNs (n = 37)**

`Contract: Month-to-month` produces the same bimodal split visible in all panels. The DSL-specific suppressor is `InternetService: Fiber optic = 0`: since all 37 customers have DSL, fiber optic is absent for every one of them, producing a uniform cluster of blue dots far left in the beeswarm. The model learned fiber optic as its primary internet-tier churn signal; the absence of fiber is treated as active evidence of lower risk, pushing the score down regardless of all other features. Most other features scatter near zero, though `OnlineSecurity`, `TechSupport`, and `MonthlyCharges` show modest contributions. DSL customers are suppressed by the model's strong negative SHAP for the fiber optic signal they do not carry — there is no secondary feature providing comparable positive signal for this segment.

**No Internet FNs (n = 14)**

The most complete suppression of the four segments. Unlike DSL customers who are held down by one missing internet feature, No Internet customers accumulate negative SHAP from multiple internet-related features simultaneously: `InternetService: Fiber optic = 0`, `OnlineSecurity` and `TechSupport` encoded as 0 ("No internet service", not "No"), and streaming features likewise — each contributing a separate negative SHAP push. The suppression is structural: the model's churn-risk signals are all internet-related, and No Internet customers can never activate them, so even genuine behavioral signals like a month-to-month contract are buried by the accumulated negative SHAPs — the model cannot score them above threshold regardless of their behavior.

#### Protected attributes and fairness

| Attribute | Mean \|SHAP\| | Rank of 40 | Assessment |
|---|---|---|---|
| `Gender: Male` | 0.0000 | #23 | Negligible |
| `Partner: Yes` | 0.0000 | #21 | Negligible |
| `SeniorCitizen` | 0.0000 | #40 | Negligible (last) |
| Mean across all 40 features | 0.0378 | — | Baseline for comparison |

All three protected attributes contribute zero marginal signal at the individual prediction level on the test set, consistent with the §9 validation finding. Churn scores are driven entirely by behavioural and contractual features. These results are safe to document in the model card as confirming no material demographic influence on predictions.

---

### Full model performance history

| Stage | Threshold | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline LightGBM (val) | 0.35 | 0.900 | 0.495 | 0.638 | 0.867 |
| Tuned LightGBM (val) | 0.35 | 0.940 | 0.405 | 0.566 | 0.855 |
| Calibrated (val) | 0.35 | 0.760 | 0.576 | 0.655 | 0.856 |
| Calibrated + OOF threshold (val) | 0.2956 | 0.800 | 0.531 | 0.638 | 0.856 |
| **Final test set** | **0.2956** | **0.7861** | **0.5241** | **0.6289** | **0.8413** |

---

## 13. Production Refit & Model Registration

### Rationale

All hyperparameter, calibration, and threshold decisions were finalised on held-out data (val and sealed test sets) before the production refit. Retraining on the full 7,043-customer dataset provides the production model with all available signal without contaminating the §12 benchmark figures.

### 13.1 Full-Data Refit

| Property | Value |
|---|---|
| Training data | Full dataset: train + val + test (7,043 rows; 1,869 churners, 5,174 non-churners) |
| Model architecture | LightGBM + sigmoid calibration (`CalibratedClassifierCV`, cv=5) |
| Hyperparameters | Unchanged from Optuna best (§7) — no re-tuning on full data |
| `scale_pos_weight` | Recalculated on full-data class ratio (5,174 / 1,869 ≈ 2.77) |
| Features | 40 (same as training pipeline) |
| MLflow run | `f81665fa` (experiment `"Telco Churn - Final Model"`) |

### 13.2 Production Threshold Re-derivation

The §11 OOF threshold (0.2956) was derived on `X_train` only. With the full dataset, OOF cost minimisation was re-run under the base scenario (ARPU $79.90, LTV $575, intervention cost $68):

| Threshold source | Threshold | Dataset | Notes |
|---|---|---|---|
| §11 OOF cost min | 0.2956 | Train set only | Used for §12 test-set evaluation |
| **§17 OOF cost min** | **0.3596** | **Full dataset** | **Production-shipped value** |

The production threshold rises by 0.064 relative to the train-only derivation, reflecting the broader distribution of the full dataset (proportionally more long-tenure, non-month-to-month customers shift the cost-optimal point rightward).

### 13.3 OOF Performance on Full Data

Out-of-fold metrics at production threshold 0.3596 (5-fold stratified CV across all 7,043 rows):

| Metric | OOF value |
|---|---|
| Recall | 0.717 |
| Precision | 0.569 |
| F1 | 0.634 |
| PR-AUC (avg precision) | 0.644 |
| Churn rate (full dataset) | 26.5 % |

> OOF metrics on the full dataset are not directly comparable to §12 test-set figures — the test set is now included in training. These are provided as a distributional sanity check only; the §12 test-set numbers remain the authoritative evaluation benchmark.

### 13.4 MLflow Model Registration

The fitted `cal_pipe_full` is logged to the MLflow Model Registry (experiment `"Telco Churn - Final Model"`, run `111afb57`, run name `LightGBM_prod_registered`). The registered model is named **`telco-churn-pipeline`** (version 1) and carries two aliases: `champion` (current best model) and `challenger`.

**Logged parameters:**

| Parameter | Value |
|---|---|
| `base_model` | LightGBM |
| `calibration_method` | sigmoid |
| `calibration_cv` | 5 |
| `dataset` | train+val+test |
| `prod_threshold` | 0.3596 |
| `evaluated_run_id` | `971c9002…` (cal_pipe §10 run) |

**Logged artifacts:**

| Artifact | Description |
|---|---|
| `model/` | MLflow pyfunc format — load with `mlflow.sklearn.load_model()` |
| `feature_columns.txt` | Ordered list of 40 feature names from `preprocessor.get_feature_names_out()` |
| `preprocessing.pkl` | Fitted preprocessor for offline feature transformation |
| `model_card.json` | Serialised parameters, dataset description, and run provenance |

**Alias-based promotion:** Version 1 carries the `champion` alias — the production-ready designation in MLflow 2.x. To promote a new version, reassign the alias: `client.set_registered_model_alias("telco-churn-pipeline", "champion", <new_version>)`.

### 13.5 Production Serving Integration

The FastAPI serving pipeline (`src/serving/inference.py`) loads the registered model from `/app/model` (Docker container path). `feature_columns.txt` enforces column order at inference time, ensuring one-hot encoded columns from the API request payload are aligned with the training-time column sequence regardless of `pd.get_dummies()` ordering.

```bash
# Load the champion (production) model by alias — MLflow 2.x style
mlflow.sklearn.load_model("models:/telco-churn-pipeline@champion")

# Or pin to an explicit version
mlflow.sklearn.load_model("models:/telco-churn-pipeline/1")
```

---

## 14. Known Limitations

### 1. Annual/multi-year contract churners are a near-total blind spot

One-year and two-year contract holders have FN rates of 0.972 and 1.000 respectively on the test set. The model learned "annual contract = committed customer" as a near-irrefutable heuristic. Customers who churn despite holding annual contracts are systematically missed. This is the highest-priority limitation for the next model iteration.

### 2. Long-tenure, low-cost segment under-served

Customers in the 25+ month tenure band have a 0.465 FN rate. Long tenure is treated as a loyalty signal, but duration can be a lagging indicator for quietly disengaging customers. No "sustained disengagement" feature exists in the current set.

### 3. No uplift / persuadability modelling

The model identifies *who will churn*, not *who will respond to a retention offer*. Four customer types exist: sure stayers (intervention wasted), persuadables (the real target), lost causes (no intervention will work), and sleeping dogs (intervention may backfire). Without A/B test data distinguishing treated and untreated outcomes, the model cannot separate persuadables from lost causes. This means some intervention budget is spent on customers who would have stayed anyway (wasted) and customers who cannot be retained at any price (lost causes).

### 4. Residual calibration gap at high scores

The calibrated model under-predicts churn probability for high-scoring customers by up to ~10 pp above score ≈ 0.58. Expected-value calculations that multiply score × LTV will understate financial exposure for the highest-risk customers. This gap requires additional features or training data to close.

### 5. No re-contact suppression

Customers flagged in one cycle and contacted will appear in the next scoring run with the same feature profile. Without a 90-day suppression window, the intervention system risks contact fatigue and diminished offer redemption rates.

### 6. Production monitoring not yet deployed

Score distribution drift (PSI) and prediction quality will degrade as the customer base evolves. No monitoring pipeline is currently in place to trigger re-training.

### 7. Feature set cannot reduce FPs for loyal high-risk-profile customers

~50 % of non-churners who share the fiber optic + month-to-month + no add-on profile are incorrectly flagged. The model has no loyalty, satisfaction, usage frequency, or recency-of-service-change signals to distinguish them from actual churners.

### 8. Business cost parameters are illustrative

The $68/intervention and $575/LTV figures used to derive the production threshold are reasonable but not Finance-validated. Threshold and P&L figures should be recalculated with confirmed actuals before committing to a budget.

---

## 15. Recommendations & Next Steps

### Immediate (model v1 deployment)

1. **Deploy at threshold 0.2956** for the base cost scenario. Revisit threshold if intervention cost assumptions change — a 2.5:1 FN-to-FP penalty ratio is sensitive to the cost/LTV estimate.
2. **Implement tiered outreach:** cheap outreach (email/SMS) for scores in 0.30–0.50; expensive retention offers (discounts, contract sweeteners) only for scores > 0.50. This reduces the wasted outreach cost on low-confidence flags.
3. **Apply a 90-day re-contact suppression window** to prevent intervention fatigue and allow post-contact conversion data to feed back into the model.

### Short-term (next model iteration)

4. **Address annual-contract blind spot:** Explore contract-type-specific sub-models, contract-age features, or a lower intervention threshold applied selectively to annual-contract holders who show secondary risk signals (e.g., high MonthlyCharges, no add-ons, electronic check payment).
5. **Instrument an A/B test** on the first cohort of flagged customers (treat vs. hold-out control). This data is the prerequisite for uplift modelling and will allow the next model to target persuadables rather than all predicted churners.
6. **Add engagement/recency features** (last service change date, usage trend, support ticket volume) to provide a signal that separates loyal high-risk-profile customers from genuine pre-churners. The current FP rate for fiber optic + month-to-month customers cannot be reduced without this.

### Ongoing

7. **Deploy PSI monitoring** for score distribution drift. Trigger a re-evaluation when PSI > 0.2 on the churn score distribution.
8. **Recalibrate the threshold annually** using updated OOF predictions on a full-data refit with refreshed business cost parameters.
9. **Replace illustrative business parameters** with Finance-validated ARPU, LTV, and intervention cost figures before committing to P&L projections.

### Deployment (next phase)

10. **REST API serving:** Expose the registered `champion` model via a FastAPI endpoint (`POST /predict`) accepting the 19 raw customer features and returning a churn probability and binary flag.
11. **Web UI:** Wrap the API with a Gradio interface for business stakeholders to score individual customers without writing code.
12. **Containerisation:** Package the serving stack with Docker for environment-consistent deployment.
13. **CI/CD:** GitHub Actions pipeline to build and push the Docker image to Docker Hub on merge to main; manual ECS service update to complete the Fargate deployment.
14. **Cloud deployment:** Host on AWS Fargate behind an Application Load Balancer for scalable, serverless inference.

---

## 16. Tech Stack

| Layer | Tool |
|---|---|
| Data validation | Great Expectations |
| Experiment tracking | MLflow (file-based) |
| Modelling | LightGBM, XGBoost, scikit-learn RandomForest |
| Hyperparameter tuning | Optuna (TPE sampler, 50 trials) |
| Calibration | `CalibratedClassifierCV` (sigmoid, cv=5) |
| Explainability | SHAP (`TreeExplainer`, exact) |

---

## 17. Quick Start

**Prerequisites:** Python 3.9+, Jupyter, and the project dependencies installed (`pip install -r requirements.txt`).

**Step 1 — Clone the repository and navigate into it**

```bash
git clone <repo-url>
cd telco-customer-churn
```

**Step 2 — Add the dataset**

Download the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place `Telco-Customer-Churn.csv` in `data/raw/`. This folder is gitignored and must be created manually.

**Step 3 — Open the notebook**

```bash
jupyter notebook notebooks/EDA.ipynb
```

This opens Jupyter in your browser. Run cells top to bottom — all analysis, modelling, and results are self-contained in `EDA.ipynb`.

**Step 4 — Browse experiment runs (optional)**

> **Note:** This step requires the notebook (Step 2) to have been run at least once. Running the notebook generates the `mlruns/` tracking store locally — a fresh clone will have no experiments to show until then.

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open [http://localhost:5000](http://localhost:5000) in your browser to explore logged metrics, parameters, and model artifacts across all experiments.

---

## 18. Project Structure

```
├── notebooks/
│   └── EDA.ipynb             # Full EDA + modelling notebook (§§1–17)
├── requirements.txt
└── README.md
```

> **Locally generated (gitignored — not committed):**
> - `data/raw/` — download the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `Telco-Customer-Churn.csv` here before running the notebook.
> - `mlruns/` — created automatically when the notebook runs and logs experiments to MLflow.

### Notebook section map

| Section | Content |
|---|---|
| §§1–5 | Data loading, initial EDA, missing value audit, outlier detection |
| §§6–8b | Bivariate analysis, correlation, multicollinearity (VIF), chi-square + Cramér's V, Mann–Whitney U |
| §9 | Key EDA insights summary |
| §10 | ML pipeline: preprocessing, baselines, feature engineering (error-driven), Optuna tuning |
| §11 | Model evaluation: baseline vs tuned, McNemar's test, threshold sweep |
| §12 | Error analysis: FN/FP deep dives, subgroup FN rates, score distribution |
| §13 | SHAP: global importance, waterfall, dependence plots, protected attribute check |
| §14 | Probability calibration (sigmoid), bootstrap CIs, cost-sensitive threshold (OOF) |
| §15 | Lift analysis, decile table, KS statistic |
| §16 | Final test-set evaluation (sealed), segment breakdown, business P&L, annualised projections |
| §17 | Production refit on full data, MLflow model registration |
