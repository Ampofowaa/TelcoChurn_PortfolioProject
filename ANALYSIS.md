# Analysis & Modelling Decisions

Full modelling rationale for the [Telco Customer Churn](README.md) portfolio project.
Covers problem framing, EDA, feature engineering, model selection, hyperparameter tuning,
calibration, threshold optimisation, business impact, final test-set results, error analysis,
SHAP explainability, and registry promotion.

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
8. [Model Registration & Promotion](#8-model-registration--promotion)
9. [Known Limitations](#9-known-limitations)
10. [Recommendations & Next Steps](#10-recommendations--next-steps)

---

## 0. Problem Framing & Cost Definition

This section locks the rules that govern every modelling decision in §§1–8: what's predicted, what it costs to act on that prediction, and how a model earns the right to serve it. It is written before any EDA or model results, so the cost-sensitive threshold, the choice of recall as the headline business metric, and the promotion gate all have a traceable origin rather than being justified after the fact.

**In short:** each month, the model scores every customer's probability of cancelling. Above a cost-justified cutoff, that customer gets a retention offer; below it, they don't. The cutoff itself is set by a simple rule — contact a customer only when the expected savings from a successful offer outweigh what the offer costs — not by any target hit-rate. A model earns the right to serve if it ranks churners well (the one metric that decides), and doesn't fail on catch-rate, basic probabilistic skill, or having honest probabilities (three checks that can only disqualify, never promote). A human reviewer then gets one more veto, decided in advance of seeing results, so nobody can rationalize a decision after the fact.

### Prediction unit, label, and decision

A **single customer at a scoring cycle** (default: monthly) — one churn-probability score per customer per run; every metric (recall, precision, EV) is computed per customer, not per household or account.

`Churn = 1` if the customer cancelled within the current billing cycle (~30 days) — the IBM Telco dataset's own `Churn` column, recoded 1/0. It's a binary, point-in-time label with no survival horizon and no soft-churn class, and it captures *revealed* churn (already cancelled), not predicted intent — the model learns the profile of customers who *did* leave, not ones who are merely dissatisfied.

The score feeds one decision: include this customer in the upcoming retention cycle (discount, upgrade, service credit) or don't. Binary, per customer, per cycle — no tiered response yet (a §10 recommendation).

Two metric consequences follow: **PR-AUC is the sole selection and promotion metric** (§4's family comparison, the gate below) — it summarises precision/recall across every threshold, and nothing else gets a vote on which model is *better*. **Recall at the shipped threshold is the headline business number, not a second optimisation target** — it's the first question the business asks ("how many churners did we catch?"), but the threshold itself is chosen to maximise expected value (below), never to hit a recall target — a recall-only policy would just contact everyone. Recall is reported because it's legible to a non-technical audience in a way PR-AUC isn't, but always alongside the expected-value figure (§7), never instead of it — recall with no cost context can't tell a profitable campaign from an expensive one.

### Cost structure — cost attaches to the action, not the error

Contacting a customer costs money whether or not they were actually going to leave — so the decision table is over (action, true state), and the cell that matters is the **true positive**, which is *not free*. Four quantities drive it: `q` (churn probability), `r` (retention rate — the odds an offer actually works), `LTV` (customer lifetime value if retained), `c` (total intervention cost).

| | Contact (spend `c`) | Do nothing |
|---|---|---|
| **Churner** (prob `q`) | `−c + r·LTV` | `0` |
| **Non-churner** (prob `1−q`) | `−c + LTV` (offer wasted) | `LTV` |

Contact iff `E[contact] > E[do nothing]`. The `(1−q)·LTV` terms cancel, giving:

> **Contact iff `q · r · LTV > c` — i.e. the operating threshold is `t* = c / (r × LTV)`.**

This is a deliberate departure from the archived exploratory pass, which charged the intervention only to false positives (treating a correctly-caught churner as free). Here it isn't — every contact costs `c` regardless of outcome. **The actual cost figures, the derived thresholds, and the shipped value are §6's job** — this section fixes the rule, not the numbers.

The familiar error-cost framing is a *diagnostic*, not the decision rule itself:

| Error | Consequence | Cost |
|---|---|---|
| False negative | Churner not contacted, offer never made | `r · LTV` |
| False positive | Non-churner contacted for nothing | `c` |

Whichever of `r·LTV` and `c` is larger sets which error the threshold implicitly favours avoiding (§6 has the real ratio per scenario — it isn't constant, since cheaper/costlier interventions don't move `r·LTV` and `c` proportionally). This does **not** hand you the threshold on its own: `C_FP/(C_FP+C_FN)` assumes correct decisions are free, and a true positive costs `c` like anything else. Selection is still PR-AUC alone — which error the cost structure favours avoiding is a fact about the cost structure, not a second selection metric.

#### The retention rate `r` is the biggest uncertainty — and it isn't the model's

`t*` moves inversely with `r`, and `r` — the fraction of contacted churners an offer actually saves — **can't be measured from this dataset**; measuring it means intervening and observing the outcome. The shipped value, 0.30, is an industry benchmark (literature range 0.15–0.40), not something this data can tell us. A plausible range of `r` swings the operating threshold far more than any realistic PR-AUC improvement would — **the single most consequential number in this deployment is a benchmark guess, not a model output** (§6 has the full sensitivity sweep).

`r` stops being a guess once the model is live: `performance_check.py` will join who-was-contacted against who-actually-stayed, turning "what fraction did we retain?" into a measured number `t*` can be re-derived from. Until that data exists, §6's three-scenario bracket (Conservative/Base/Optimistic) shows how much the decision shifts under different plausible values of `r`, instead of presenting one falsely-precise number. The cost parameters throughout are industry-plausible estimates, not Finance-validated (§9 limitation #7).

**`LTV` is one number per scenario, not per customer — the most consequential simplification in this cost model.** `configs/costs.yaml` derives it from quantiles of churner `MonthlyCharges`, even though `LTV_i = MonthlyCharges_i × gross_margin × horizon_months` is available per-row for free. Two things follow from not using it: (1) with `c`, `r`, `LTV` constant, expected value is a strictly increasing function of `p_i` alone — ranking by EV is *identical* to ranking by churn probability, so a $110/month customer and a $20/month customer at the same score are treated as equally worth saving; (2) a per-customer `LTV` would replace the single global `t*` with a per-customer cut (`p_i · r · LTV_i > c`), which under a capacity constraint becomes a genuine top-K-by-value policy rather than top-K-by-score. Adopting it now would re-open this section and §6's already-closed threshold work, invalidating the shipped `t* = 0.3941`. It's the **leading v2 item**, pairing naturally with uplift modelling (§9 #3): *who's worth saving* and *who can be saved* are exactly what the current model assumes away.

### Success criterion — the promotion gate

The gate decides whether a model is fit to serve, defined once, here. `models/gate.py` implements it (`decide_promotion`); `evaluate.py` calls it and persists the verdict the moment sealed-test metrics exist; `register.py` only **reads** that verdict, never recomputes it. A divergence between this section and the code is a bug in the code.

**One metric decides which model is better; three separate checks can each block it regardless** — like a job candidate who aces the interview (PR-AUC) but still needs a clean background check on three fronts (recall, Brier, calibration). Any one disqualifying result ends it, as finally as losing the interview would. None of the three can win the job for a candidate — each can only cost them it.

| Role | Metric(s) | Power |
|---|---|---|
| **Selection** | PR-AUC | The only metric that decides which model is *better*. Can admit. |
| **Guardrail** | Recall at `t*`; Brier skill score; **calibration slope** | Can veto a model selection already admitted. Never promotes. |
| **Diagnostic** | ROC-AUC, precision, F1, lift/gains, EV, per-segment PR-AUC | Reported only. No power over the gate. |

All four checks run in one pass (`pr_auc_passed and recall_passed and brier_passed and slope_passed`) — not a staged pipeline; describing a guardrail as being "asked of" an admitted model is about what a "no" *means*, not about execution order.

**Two regimes**, because the registry starts empty:

**Cold start (the first promotion, Phase 7) — no incumbent, so every bar is absolute,** fixed before the test set opened and checked once against §7's real results:

| Criterion | Role | Bar | Why |
|---|---|---|---|
| PR-AUC | selection | ≥ 0.60 | Threshold-free ranking metric, appropriate at 27% prevalence where ROC-AUC runs optimistic. |
| Recall at `t*` | guardrail | ≥ 0.65 | The business floor — ranking well while catching two-thirds of nothing isn't shippable. |
| Brier skill score | guardrail | > 0 | Basic probabilistic skill vs. the base-rate floor — without this, `t*` rests on nothing. |
| **Calibration slope** | guardrail | CI entirely outside [0.80, 1.25] fails | Catches what Brier can miss (below): slope 1 = perfectly calibrated, < 1 = overconfident. Fails only on clear evidence, not a merely wide estimate. |

One asymmetry: PR-AUC/recall/BSS are checked against the *point estimate*, not a confidence bound — requiring the CI to clear these bars would silently raise them (at n = 1,409, a lower-bound "≥0.60" behaves like "≥0.65"). The slope's CI-based rule runs the opposite way, making it the more *lenient* check, not stricter.

**Why two calibration guardrails, not one.** Brier is a proper score, but `Brier = reliability − resolution + uncertainty` (Murphy's decomposition) blends calibration with *ranking* — a challenger can improve its Brier purely by ranking better while its calibration quietly degrades, sailing past a Brier-only check. That matters here specifically because `t* = c/(r×LTV)` is only Bayes-optimal against honest posteriors — a miscalibrated model means the shipped threshold silently stops being the right rule, and no PR-AUC check would catch it, since ranking is exactly what still looks fine. ECE, the obvious alternative, is binning-dependent, improper, and estimation-biased — reported as a diagnostic, never gating. The **calibration slope** (regress `y` on `logit(p)`) is binning-free, parameter-free, and can't be bought with better ranking, which is what makes it the second, gate-worthy guardrail — applied as an absolute bar in both regimes, like recall, because "are these posteriors honest" has a right answer independent of the incumbent. It's computed in `evaluate.py` on the sealed test set (not in `calibrate.py`, which fits the calibration and logs the slope into `calibration_summary.json` but only needs Brier to pick sigmoid vs. isotonic) and checked twice — once at n=5,634 (`threshold.py`'s pre-seal dev-OOF screen, Phase 6's last step, screening `calibrate.py`'s already-logged slope — higher-powered but slightly optimistic, since it screens the same probabilities that informed the calibration-method choice) and once at n=1,409 (test, honest but only powered to catch gross miscalibration). Neither certifies calibration on its own; either can disqualify, which is all a veto needs.

**The band's edges are pre-registered policy, not derived from this data** (§9 limitation #13 has the validation status).

**Comparative regime (every promotion after the first) — selection becomes comparative, and every guardrail keeps its absolute floor.** Once a champion exists, a new candidate is judged against *it*, not against a fixed bar — and that comparison has to account for statistical noise, or it will mislead. If a new model were promoted every time its measured PR-AUC came out even slightly ahead of the champion's, that rule would misfire constantly: when two models are genuinely equally good, ordinary measurement noise makes one look better than the other roughly half the time, purely by chance. Promoting on that basis would replace the champion for no real reason on a regular basis — and worse, each accidental "win" becomes the new baseline the *next* comparison has to beat, so the bar quietly ratchets upward on noise while the model's actual quality goes nowhere.

Two of the three guardrails (recall, Brier) layer a **non-inferiority check against the incumbent on top of** — never instead of — their absolute floor. A floor alone can pass a candidate that is a large regression from a much stronger incumbent (recall 0.90 → 0.66 both clear the 0.65 floor); a non-inferiority check alone can pass a lineage that drifts downward one small, individually-tolerated dip at a time, since it only ever compares against the most recent predecessor and never against a fixed reference — the failure mode pharmacovigilance calls "biocreep." Requiring both closes each gap without opening the other: the floor re-anchors the lineage against drift, the non-inferiority check catches a regression the floor is too permissive to see. Calibration slope has no incumbent-relative form — "are these posteriors honest" doesn't get easier just because the incumbent's were also dishonest — so it stays a pure absolute check, identical to cold start.

The fix is a **paired bootstrap with a pre-registered materiality threshold** — the same surface §4's model-family choice already uses. Pairing (both models scored on the same rows) cancels shared uncertainty, so the CI on the *difference* is far tighter than either model's own marginal CI — §4's full-vs-reduced comparison resolved a CI of [0.0050, 0.0110] (±0.003) against marginal CIs an order of magnitude wider.

| Criterion | Role | Rule | Why |
|---|---|---|---|
| PR-AUC | selection | Promote iff the candidate's own PR-AUC still clears the cold-start bar (≥ 0.60) **and** the CI on Δ = AP(challenger) − AP(champion) excludes 0 in the challenger's favour **and** Δ ≥ 0.005 | Mirrors §4: the CI is the test, the materiality threshold stops a real-but-trivial gain from moving a production alias. The absolute floor is layered on top, not replaced by the Δ: without it, a champion lineage is only ever checked against its immediate predecessor, so a later increase to the 0.60 bar would never re-bind an existing lineage — each new candidate only has to out-pace a champion that itself was never re-measured against the new floor. |
| Recall at `t*` | guardrail | Blocks the promotion if recall < 0.65, absolute, **or** if the CI on Δ = recall(challenger) − recall(champion) lies entirely **below −0.01** | The 0.65 floor doesn't get lowered just because the model currently in production has slipped — a struggling incumbent is something monitoring should flag, not an excuse to accept a weaker replacement. But the floor alone can't catch a candidate that clears 0.65 while still being a large regression from a much stronger incumbent (0.90 → 0.66 both clear it) — the non-inferiority check adds that, vetoing only on confident evidence of real harm, never merely for failing to prove there is none. |
| **Calibration slope** | guardrail | Blocks the promotion if the CI lies entirely outside [0.80, 1.25], absolute | Checks whether the new candidate's own probabilities can be trusted, independent of how the old model did. It also closes a loophole: a model that ranks customers better but gives less trustworthy probabilities could otherwise slip through undetected. |
| Brier | guardrail | Blocks the promotion if BSS ≤ 0 (vs. the `DummyClassifier` baseline), absolute, **or** if the CI on Δ = Brier(challenger) − Brier(champion) lies entirely **above +0.005** | The non-inferiority check alone tolerates a small dip every cycle by design, so a real gain in ranking never gets blocked by trivial Brier noise — but that also means a lineage that is merely non-inferior to its immediate predecessor, cycle after cycle, could drift arbitrarily far below a fixed reference with nothing ever re-checking it against one ("biocreep," the same failure mode non-inferiority drug trials guard against). The absolute BSS floor — identical to the cold-start bar — re-anchors the lineage each cycle, the same role the PR-AUC floor plays above. |

A challenger that passes is promoted directly — `register.py` flips `champion` onto exactly the version this comparison scored, no separate refit.

Structurally, in both regimes: no test-set information is used before final evaluation — `t*` comes from dev-only out-of-fold predictions (§6), preserving *test set touched once*.

### The pre-seal automated screen (V3)

Before the sealed test set is ever opened, `threshold.py` runs one more automated, binding check alongside the calibration-slope band already described above: **V3, direction sanity** — do the model's most influential factors push risk the way the data actually shows they should (e.g. longer tenure should *lower* risk, never raise it)? A reversed factor this influential means the model latched onto a coincidence in the training data, not a real pattern, and no aggregate score would catch it.

This runs on the champion's own **development-partition** SHAP values (`calibrate.py` logs the full matrix plus a per-feature `(mean_abs_shap, direction)` summary the moment the pipeline is fit), never the sealed test set — the same discipline that keeps `t*` itself dev-derived. The check ranks every transformed feature by mean |SHAP|, takes the top `v3_top_k_features` (**8** — derived once by hand off the dev ranking: mean-|SHAP| forms a tight plateau across ranks 4–8 before a real elbow at rank 8→9, roughly 4.7× the plateau's own step size), and compares each survivor's direction against the established EDA relationship. A feature whose `|direction|` falls below `v3_min_direction_magnitude` (**0.3**) is excluded from the check rather than risking a veto on a sign too unstable to trust — every one of the derivation run's actual top-8 features cleared 0.87, so the floor exists purely to guard a future retrain, not to soften today's check. A violation — or a top-8 with nothing left to check once weak/unmatched features are excluded — halts the pipeline before `evaluate.py` ever touches the seal.

`error_analysis.py` re-runs the identical check on the sealed test set afterward, against the *same* dev-derived feature set (no independent test-side cutoff) — reported only, never a second gate, but a dev-pass/test-fail pairing is itself a generalisation signal worth a reviewer's attention.

### The human review

Automated checks alone aren't enough: a model can post an excellent overall score while quietly failing one specific customer group — and that wouldn't show up in the aggregate numbers. So before the very first model goes live, a human reviewer also checks it, using the error-analysis notebook and model card, and stamps an approve/reject verdict.

Three checks (V1, V2, V2b) are computed and shown to the reviewer and the model card, but none of them can block promotion on their own. The sealed test set is simply too small to draw a reliable conclusion about any one customer group — the two-year-contract segment, for example, has only around ten churners in it, nowhere near enough to trust. So these three are reported now for visibility, and their real enforcement moves to ongoing production monitoring once the model is live and there's far more data to check them against.

| # | Criterion | Surface | Action if triggered |
|---|---|---|---|
| V1 | **Segment collapse** — the model ranks one customer segment (by contract type, tenure, or internet service) no better than chance, for that segment specifically. | dev OOF | Reported only — feeds the model card and ongoing per-segment monitoring. |
| V2 | **Fairness gap, at the actual decision** — across gender, senior-citizen status, having a partner, and having dependents: one group is offered help notably less often, or missed notably more often, than another, at the threshold actually used. | dev OOF | Reported only — a fairness gap here is a business judgment call, discussed in §6, and tracked going forward. |
| V2b | **Per-group calibration** — the model's probability estimates are honest overall, but not for one specific customer group. | dev OOF | Reported only, same reasoning as V2. |
| V3 | **Direction sanity** — see the pre-seal automated screen above. | dev SHAP, pre-seal | **Blocks promotion automatically, before the reviewer ever sees the model** — already resolved by the time this review happens. |

**Why V2 checks the actual decision, not just the ranking.** Two models can rank customers equally well overall, yet still produce very different real outcomes once a cutoff is applied — one might end up contacting far fewer members of some group, or missing more of that group's actual churners. Since ranking quality alone can hide that kind of unevenness, V2 looks at what actually happens at the threshold in use: who gets contacted and who gets missed is what affects real people, not the ranking underneath it.

**This review can only approve or reject — never "fix and try again."** If it turns up something that genuinely needs a model change, the right response is to reject and go back to feature engineering — not quietly patch the model and re-check it against the same sealed test set, which the reviewer has by then already seen too much of for a second look to be trustworthy. The decision is recorded in `reports/promotion_decision.json` and `reports/promotion_review.json`, and the model cannot be promoted without both.

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

Chi-squared + Cramér's V for categorical features (V > 0.30 = strong, 0.10–0.30 = moderate, < 0.10 = weak); Mann-Whitney U + rank-biserial r for numeric features. With n = 7,043, p-values are near zero for any real effect — **effect size magnitude, not p-value, drives the ordering below.** The two non-predictors below are cited by `p_adj` rather than the raw p-value — a Benjamini-Hochberg correction across the 19 categorical and numeric features tested against the same target, correcting for running that many simultaneous tests.

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
| **PhoneService** | Cramér's V | **0.01** (p_adj = 0.33) | **Non-predictor** |
| **Gender** | Cramér's V | **0.0086** (p_adj = 0.47) | **Non-predictor** |

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

The dev/test split is established once in §3a — the 5,634-row dev partition and the 1,409-row sealed test set, touched exactly once at final evaluation (§7). `models/train/` imports the canonical split manifest (`data/split.py` / `datasets/processed/split_manifest.parquet`) directly rather than deriving its own — the two are the same split by construction, not just by convention. All comparisons and tuning below run on the dev partition using 10-fold cross-validation repeated 10 times (`configs/config.yaml: training_setup.cv_folds/cv_repeats`, chosen to shrink the mean PR-AUC estimate's variance on ~5,600 dev rows), which gives more reliable estimates than a single held-out slice on a dataset of this size.

#### Model family scope

Only LightGBM and `LogisticRegressionCV` (the strongest interpretable linear baseline) are compared. Other tree ensembles (XGBoost, RandomForest) are skipped: on tabular data they typically perform within noise of each other, so comparing several at default config would just compare defaults, not tuned ceilings — added maintenance cost for no decision value. LightGBM is committed up front for build-specific reasons: fast exact TreeSHAP for the SHAP explainability work (§7) and its Streamlit surfacing, training speed for Optuna and the monthly retrain cadence, and clean interaction with `class_weight`-based imbalance handling.

**Native categorical splits vs. one-hot encoding (deliberate, not an oversight):** LightGBM supports native categorical splitting, often preferred over OHE for trees. OHE is used here instead because every categorical in this feature set has ≤ 4 unique values (the column-blowup cost OHE is usually criticized for is negligible), it keeps the tree and linear preprocessing pipelines structurally uniform, and it yields cleaner per-level SHAP attribution for the Streamlit top-5 contributions view.

#### Candidate comparison (PR-AUC, RSKF 10×10)

LightGBM and LogisticRegressionCV are compared on PR-AUC: threshold-free, aligned with the §0 cost structure, and more discriminating than ROC-AUC at this 2.77:1 class imbalance, since ROC-AUC is scored against the large negative class and can look nearly identical across models that rank the minority churn class very differently. Each model uses its own preprocessor (tree vs. linear — see `03a-model-selection.ipynb`), so OHE encoding choices don't handicap either candidate, and class imbalance is handled via `class_weight='balanced'` for both rather than SMOTE/resampling — chosen for three reasons: the ~27% positive rate is mild imbalance, and resampling pays off more at extreme ratios; the feature space is mostly one-hot, and SMOTE interpolates incoherently across dummy columns; and calibration is a later deliverable (§5) that resampling would undermine — base-rate-altering resamplers decalibrate probabilities while reweighting barely shifts them, and the operating threshold gets set explicitly at that point anyway (§6). Identical fold indices are shared across all three candidates (one `RepeatedStratifiedKFold` instance, instantiated once) so scores are paired by construction — the precondition for the bootstrap comparison below. A `DummyClassifier(strategy='prior')` — a baseline that ignores every customer feature and simply predicts the overall churn rate for everyone — is included as a safeguard: since it has no real signal to work with, a strong score from it would mean the target has somehow leaked into the features. `models/train/candidates.py` asserts its ROC-AUC sits at chance and its PR-AUC matches the churn rate, and aborts the run if either check fails.

| Candidate | CV PR-AUC | ± std | Train s/fold |
|---|---|---|---|
| `dummy_prior` | 0.265 | 0.001 | 0.00 |
| `logreg_cv` | 0.651 | 0.033 | 1.31 |
| `lgbm_default` | 0.658 | 0.035 | 0.35 |

**Safeguard check passed:** the dummy classifier scored 0.265, matching the dev-set churn rate (0.265) — confirming the real candidates' higher scores reflect genuine predictive signal, not a broken eval harness or leaked target.

LightGBM leads by ~0.007 PR-AUC points; ROC-AUC is close and non-contradictory (0.844 for both candidates). LogReg is markedly more expensive to train here (`LogisticRegressionCV`'s inner 5-fold search over 10 `C` values costs ~3.7× LightGBM's single fit per outer fold — 1.31s vs. 0.35s) — a genuine simplicity-vs-cost tradeoff LogReg does *not* win on, despite its interpretability appeal.

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

**Retirement — decoupled from the automated pipeline, not deleted.** Steps 1-2 (`models/train/candidates.py::run_candidate_step`, `comparison.py::run_comparison_step`) once ran automatically inside `models/train/__main__.py`, on every training cycle — that call site is gone: re-deciding the model family every cycle (including Phase 10's monthly retrain) bought nothing beyond reconfirming an answer that had already stopped moving, and made cycle-to-cycle model versions harder to compare, the same problem §4b's feature-selection retirement solved for the adjacent axis. The functions still live in `models/train/`, fully tested — they now run only from [`notebooks/03a-model-selection.ipynb`](notebooks/03a-model-selection.ipynb)'s on-demand review, deliberately, on a real trigger (a new candidate model family, a drift signal, or a scheduled periodic review), never automatically. The committed family is frozen into `models/train/common.py::COMMITTED_MODEL_FAMILY` — a hand-maintained constant, edited only via reviewed PR — not recomputed by reading this record. The decision above is recorded in MLflow run `c405f6fe31454c3a9899423680644411` (`models/train/common.py::COMMITTED_MODEL_FAMILY_DECISION_RUN_ID`).

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

### 4b. Feature Selection (concluded ablation + importance diagnostic)

Scored once against default LightGBM after family commitment, inside CV on the train set (MLflow run `de4fac8e`). This is a **concluded ablation, not a step that re-runs**: the keep-vs-reduce question below was decided once and is now a fixed constant — every training cycle commits the full ~20-feature `FEATURE_SCHEMA` space, the same columns shared by tuning (below), calibration (§5), and serving. What still runs every cycle is cheaper and answers a different question: one all-dev permutation-importance fit plus a non-gating SHAP audit, neither of which restricts the input space — they exist to explain it (feeding the model card, `error_analysis.py`, and stakeholder fairness explanation), not to decide it.

**Decoupled from the automated pipeline, not deleted.** The CV wrapper that produced the keep-vs-reduce comparison below (`run_selection_cv`, `reduced_set_bootstrap_test`) once ran automatically inside `models/train/feature_audit.py`, on every training cycle — that call site is gone: replaying the same test on truncated repeat subsets showed `full_features_win` from 3 of the planned 10 repeats onward, so re-deriving the full 100-fold comparison every cycle (~85,000 scoring passes: 100 folds × a LightGBM fit + ~850 `predict_proba` calls) bought nothing beyond reconfirming an answer that had already stopped moving. The functions themselves still live in `features/select.py`, fully tested — they now run only from `notebooks/03b-feature-selection.ipynb`'s on-demand review section, deliberately, on a real trigger (a new engineered feature, a per-feature drift signal, or a scheduled periodic review), never automatically. See `CHANGELOG.md` for the code-level history, including an earlier same-day pass that deleted these functions outright before this decoupled design replaced it. The committed feature set they decide is frozen into `features/schema.py::COMMITTED_FEATURES` — a hand-maintained constant, edited only via reviewed PR — not recomputed by reading this record.

**Method.** Each feature is scored using **permutation importance**: how much the model's performance (PR-AUC) drops when that feature's values are shuffled into random noise, holding everything else fixed — a bigger drop means the feature was actually being used. A synthetic **decoy column**, built from pure noise, sets the bar for what "no real signal" looks like: a feature survives only if its measured importance beats the decoy's by a comfortable margin (`max(decoy_importance, 0) + 0.005`; `configs/training/feature_selection.yaml`). This is model-agnostic — it doesn't lean on any one algorithm's internal bookkeeping — which is why it replaced this build's original approach (LightGBM's own "gain" importance, prone to overstating high-cardinality or frequently-split features; Boruta-SHAP was also considered and rejected for a thinly-maintained dependency with no clean sklearn API). A categorical's one-hot columns are always shuffled and judged together, so it is kept or dropped as a whole, never partially.

Two checks guard against this method misleading:

1. **Fluke check.** Refit across all 100 CV folds and track each feature's survival rate. 7 of the 10 all-dev survivors are selected in at least 70 of 100 folds (`tenure`/`contract_type` every time), but three survivors fall well short: `multiplelines` in 60 of 100 folds, `paymentmethod` in 38, and `streamingtv` — the thinnest all-dev survivor — in just 8. Two dropped features cross over the other way: `charge_per_service` (69/100) is selected more often than all three of those survivors, and `paperlessbilling` (51/100) exceeds `paymentmethod` and `streamingtv`, despite both failing the all-dev floor outright.
2. **Correlated-credit check.** Shuffling correlated features one at a time can under-count shared signal, so each correlated cluster is re-tested together before accepting an individual failure. The `tenure`/`totalcharges`/`monthlycharges` trio all clear the floor alone (no rescue needed; `tenure`–`totalcharges` correlate at 0.89, the strongest pair in the trio). The 7-feature `internetservice` add-on cluster (VIF = ∞, §8b `01-eda.ipynb`) has 4 individual survivors (`internetservice`, `onlinesecurity`, `techsupport`, `streamingtv`), but the other 3 (`onlinebackup`, `deviceprotection`, `streamingmovies`) didn't need rescuing either — credit-splitting never dragged a real signal under the floor.

**Deciding keep vs. reduce, and the outcome.** Same mechanism as the model-family decision above: refit the full and reduced (10-feature) candidates on the same 100 folds, then bootstrap Δ = mean(AP_full) − mean(AP_reduced), paired fold by fold. Result: reduced **0.6490** vs. full **0.6580** — Δ = **0.0080**, 95% CI **[0.0050, 0.0110]**, p < 0.0001, full wins 69 of 100 folds — clears the materiality bar, so **`full_features_win`** fires: **the full set is retained, 20 of 20 kept**. Below, 10 of 20 features individually clear their own decoy floor; the other 10 — including all four protected/quasi-protected attributes (see below) — stay anyway, since this aggregate test, not the per-feature table, governs adoption. That table is a diagnostic audit trail only; it never touches the sealed test set.

| Feature | Real importance | Decoy floor | Survived (all-dev) | Per-fold stability |
|---|---:|---:|:---:|---:|
| `tenure` | 0.0880 | 0.0050 | ✅ | 100/100 |
| `contract_type` | 0.0840 | 0.0050 | ✅ | 100/100 |
| `internetservice` | 0.0280 | 0.0050 | ✅ | 91/100 |
| `totalcharges` | 0.0240 | 0.0050 | ✅ | 96/100 |
| `monthlycharges` | 0.0200 | 0.0050 | ✅ | 95/100 |
| `multiplelines` | 0.0160 | 0.0050 | ✅ | 60/100 |
| `paymentmethod` | 0.0130 | 0.0050 | ✅ | 38/100 |
| `techsupport` | 0.0090 | 0.0050 | ✅ | 87/100 |
| `onlinesecurity` | 0.0080 | 0.0050 | ✅ | 75/100 |
| `streamingtv` | 0.0060 | 0.0050 | ✅ | 8/100 |
| `paperlessbilling` | 0.0050 | 0.0050 | ❌ | 51/100 |
| `streamingmovies` | 0.0040 | 0.0050 | ❌ | 0/100 |
| `seniorcitizen` | 0.0030 | 0.0050 | ❌ | 1/100 |
| `onlinebackup` | 0.0030 | 0.0050 | ❌ | 4/100 |
| `charge_per_service` | 0.0010 | 0.0050 | ❌ | 69/100 |
| `deviceprotection` | -0.0000 | 0.0050 | ❌ | 0/100 |
| `phoneservice` | -0.0010 | 0.0050 | ❌ | 0/100 |
| `dependents` | -0.0010 | 0.0050 | ❌ | 1/100 |
| `has_partner` | -0.0010 | 0.0050 | ❌ | 0/100 |
| `gender` | -0.0010 | 0.0050 | ❌ | 2/100 |

*(Decoy importance on the all-dev fit was -0.0060; the floor is `max(-0.0060, 0) + 0.005 = 0.005` for every row.)*

**Reasons for keeping the failed features.** Most of the 10 failing features carry real, statistically significant univariate correlation with churn per `01-eda.ipynb` §7's Cramér's V — the add-on trio `onlinebackup`/`deviceprotection`/`streamingmovies` (V = 0.23–0.29), `paperlessbilling` (V = 0.19), and the protected-attribute trio `seniorcitizen`/`has_partner`/`dependents` (V = 0.15–0.16). That's redundancy, not noise: the signal doesn't survive as *marginal* contribution once the rest of the feature set is already in the model — though not a simple one-bigger-feature story, since the add-on cluster's own rescue check above found no credit-splitting, and `seniorcitizen` is VIF-orthogonal to everything else (§8b `01-eda.ipynb`). `gender` and `phoneservice` are the one clean case where EDA and permutation importance both agree there's no signal at all (V ≈ 0.01, both fail to reject the null). `charge_per_service` — the sole engineered feature adopted during feature discovery, constructed in LAP 7 of `02a-feature-discovery.ipynb` — fails its own floor here (0.0010 vs. 0.0050, 69/100 fold stability) despite passing a different, incremental-signal screen at adoption time (§3a); `multiplelines` runs the opposite way, weak in EDA (V = 0.04) but real here. None of this changes the outcome: every one of the 20 stays regardless of its individual read, because the decision above is a single full-vs-reduced choice, not a per-feature filter.

**SHAP audit (diagnostic only).** A second, different lens on the same 20 features: `_compute_shap_audits` fits one more default-config LightGBM and measures each feature's average contribution to individual predictions — not the performance drop from removing it, like permutation importance measures. It never decides keep/drop; it's a cross-check only. The two methods mostly agree on *which* features matter, but not completely: 9 of the 10 permutation-importance survivors occupy 9 of the top 10 SHAP ranks. The exception is `streamingtv`, the thinnest survivor (real importance 0.006, barely above the 0.005 floor), which falls to 12th by SHAP (mean |SHAP| 0.105) — behind two features that failed the decoy floor outright: `charge_per_service` (8th, 0.167) and `paperlessbilling` (11th, 0.113). Among the features both methods agree on, ordering still reshuffles: `contract_type` and `tenure` swap the top two spots (SHAP rates `contract_type` more than double `tenure`'s score — 1.115 vs. 0.465 — the reverse of permutation importance's order, 0.084 vs. 0.088), and `paymentmethod` jumps from a modest 7th-of-10 permutation-importance rank (0.013) to 3rd by SHAP (0.351) — plausibly because SHAP credits its role in individual predictions more than one shuffle-and-measure pass captures.

**Tradeoff and future consideration.** The full set's edge costs something too: 10 extra columns to validate and monitor, for a ~1.5% relative PR-AUC gain — a team prioritizing simplicity or a smaller fairness-audit surface (e.g. dropping `gender`) could reasonably prefer the reduced set instead, with a different Δ\* set in advance rather than a different reading of the same evidence. A finer-grained method like `RFECV` could identify exactly which 1-2 features are safe to drop individually, but isn't worth adopting now: run naively it reintroduces the stepwise-selection bias the single pre-registered test avoids, and run correctly (nested CV, to avoid that bias) it costs meaningfully more compute — either way, for a marginal payoff at this feature-set size. Worth revisiting only if serving cost or audit-surface reduction becomes a real priority.

The permutation-importance diagnostic that still runs every cycle — the table above, the fluke and correlated-credit checks, and the demographic joint-importance result — is walked through in [`notebooks/03b-feature-selection.ipynb`](notebooks/03b-feature-selection.ipynb). That same notebook's on-demand review section can genuinely redo the full-vs-reduced ablation against current data (logged to its own `telco-churn-feature-selection-review` MLflow experiment); its standing result — cited here, not re-derived — is recorded in this section and in MLflow run `de4fac8e`.

#### Protected attributes & fairness policy

All four protected / quasi-protected attributes — `gender` (sex), `seniorcitizen` (age), `has_partner` (marital status), `dependents` (familial status) — **remain model inputs; none is hand-excluded.**

1. **Benefit, not a denial.** The model drives a retention offer, not a credit or employment decision — the domains where statutory protection binds. Demographic targeting is standard practice in marketing.
2. **They carry genuine univariate churn signal** (three of four; see §2) — a real predictive case for eligibility, even though it is not the final word (see "Reasons for keeping the failed features" above).
3. **Fairness is enforced by measurement.** Per-group PR-AUC parity is evaluated across all four axes at candidate-selection time (§4, "Disaggregated robustness & fairness check") and again on the champion during error analysis (§7). Keeping the attributes available through candidate comparison and selection is what makes that measurement possible; the fairness *monitoring* commitment stands regardless of which axes feature selection ultimately keeps as model inputs.
4. **The block's collective importance is now priced, not just its members' individual scores.** `configs/training/feature_selection.yaml`'s `correlated_groups` declares `[gender, seniorcitizen, has_partner, dependents]` alongside the existing correlated trios above (added at the Phase 8 prerequisites pass, after run `de4fac8e`, so it postdates the table above and was verified separately against current dev data). All four fail the individual decoy floor on their own (gender −0.00088, seniorcitizen 0.00300, has_partner −0.00104, dependents −0.00113, vs. a 0.005 floor), which is exactly the rescue's trigger condition — no member survives alone — so the joint permutation runs and returns the block's collective importance: **`group_importance = 0.00116`**, still below the 0.005 floor. The number is the answer a fairness-motivated removal of this block would need — it says the four together buy the model at most ~0.0012 PR-AUC, cheap to give up — while the `survived`/`rescued` flags stay `False` either way: clearing the floor was never a precondition for removing the block on governance grounds, only a measurement of what removing it costs.

### 4c. Hyperparameter Tuning (Optuna)

Tunes LightGBM only, on the input space frozen by Feature Selection above (§4b: `full_features_win`, all 20 features retained — a fixed constant since the retirement, not a per-run decision). PR-AUC (`average_precision`) is the sole study objective, consistent with the one-metric invariant.

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

**Study hygiene (five checks recorded):**

1. **Pruning** — MedianPruner adopted; **47 of 50 trials completed, 3 pruned** mid-CV, saving the budget for configurations plausibly competitive with the running median.
2. **Boundary-hit check** — none of the 8 selected hyperparameters sit on its searched range's edge (all `False`); the widened ranges above comfortably contain the optimum.
3. **Selection rule** — raw argmax (trial 13, CV PR-AUC 0.6668) is **not** adopted. The 1-SE rule picks **trial 10** (CV PR-AUC 0.6621, within one SE = 0.0060 of trial 13) for far fewer `num_leaves` (6 vs. 151) and about 91% of the trees (147 vs. 161) — `num_leaves` is LightGBM's documented main complexity control under leaf-wise growth, though this particular study's own fANOVA ranking (now seeded for reproducibility) doesn't reflect that theoretical primacy: `max_depth` dominates at 22.5% of total importance, with `num_leaves` mid-pack at 9.9%. Net effect: a ~0.0047 PR-AUC sacrifice for a materially simpler configuration — `num_leaves` carries essentially all of that simplicity gain, since the tree count itself is barely reduced.
4. **Convergence** — the running-best curve is flat from trial 13 onward (every trial from 14 through 50 lands at or below it — a wide plateau, not a narrow cutoff); the study is not still climbing at the 50-trial budget.
5. **Resume continuity** — checked once against live infra (Postgres-backed Optuna storage, real MLflow server, real dev features — not the InMemoryStorage/synthetic-fixture unit tests), since `_discard_incomplete_study_unless_resuming` needed a run spanning an actual process/study-reload boundary to test meaningfully. A 6-trial study split across two invocations (3 trials fresh, then 3 more via `tuning.resume=true`) picked the same 1-SE trial as a single continuous 6-trial run — identical trial number, hyperparameters, and CV PR-AUC. Trials sampled before the resume boundary matched exactly across both runs; trials sampled *after* it did not (`TPESampler` reseeds its own `RandomState` fresh on every `_build_optuna_study` call rather than persisting RNG state across the resume boundary, so the search trajectory past that point is an independent seeded continuation, not a bit-for-bit replay of what an uninterrupted run would have sampled). This is expected — Optuna's storage persists trial *results*, not sampler internal state — and immaterial to this check's outcome since the winning trial fell before the boundary; it does mean a resumed study is not guaranteed to reproduce an uninterrupted run trial-for-trial past the resume point, only that the resulting *selection* was unaffected here.

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
| `n_estimators` | 100 | 147 (median over folds, scaled to the final-fit row count) |

#### Tree-count scaling for the final fit

`n_estimators` is not searched directly — each Optuna trial's value is the median early-stopped tree count across its own 5 CV folds, measured on a per-fold training partition of 3,606 rows. The full-development refit trains on 5,634 rows, large enough to earn more boosting rounds before early stopping would trigger, so the raw median (94) is scaled by `n_final_fit / n_fold_fit ≈ 1.5624` to a shipped count of **147**.

A two-count diagnostic checks this scaling against the model's own data rather than trusting the formula alone: refitting the tuned spec at both counts on the same CV folds gives CV PR-AUC 0.6669 at 94 trees versus **0.6684** at 147 — scaling improves the score, not just changes it.

#### Full → reduced → tuned bias/variance progression

Three bias/variance reads chain across the decide→optimize boundary: ① the full-feature default model, ② the reduced-feature default model, and ③ the tuned model — all read via `diagnostics.generalization_gap` on the same `RepeatedStratifiedKFold(10×10)` scheme:

| Stage | Train PR-AUC | CV PR-AUC | Train − CV gap |
|---|---:|---:|---:|
| ① Full (20 features, default) | 0.7945 | 0.6583 | 0.1362 |
| ② Reduced (20 features, default) | 0.7945 | 0.6583 | 0.1362 |
| ③ Tuned (20 features, 1-SE) | 0.7045 | 0.6701 | **0.0343** |

Selection alone (①→②) shows no movement this cycle, since Feature Selection (§4b) retained the full set — confirming feature count was never the primary overfit lever here. Tuning (②→③) is what does the work: the gap **shrinks 75%** (0.1362 → 0.0343) while CV PR-AUC simultaneously **improves** (0.6583 → 0.6701) — the default model was overfitting, fitting the training data (0.7945) far better than it generalized to unseen data (0.6583). Tuning deliberately gives up some of that training fit (down to 0.7045) to curb the overfitting, and it pays off: performance on unseen data still goes up (0.6583 → 0.6701) rather than just holding steady. (The tuned CV PR-AUC here, 0.6701, is computed on the 10×10 RSKF scheme for this comparison and differs slightly from the Optuna study's own reported 0.6621, which is a single 5-fold value from a different fold assignment — not a discrepancy, a different measuring instrument.)

#### Model logging (not registered)

The tuned pipeline (`tree_preprocessor` + LightGBM at the 1-SE hyperparameters above, `n_estimators=147` fixed — no further early stopping on this final fit) is refit on all of development and logged as one `Pipeline` — not the bare estimator — onto the *same* MLflow run as the `tuning_study` parent (`run_id 7949ad79ea1d4c25bd4976b14a31bb4d`), per the packaging checklist below:

| Check | Result |
|---|---|
| Model signature + input example | Logged (`mlflow.sklearn.log_model`) |
| Log → reload → predict parity | **Passed** — reloaded pipeline's predictions match the in-memory model exactly on a 5-row dev sample |
| `training_manifest.json` | Populated: git SHA, data content hash, hyperparameters, `feature_space` (20) / `feature_columns` (20), CV PR-AUC, paired-Δ vs LogReg with full bootstrap CI, and `tuning_summary` — the engineering audit trail |
| Registered | **Not at this stage** — logged only (`models:/m-f5b3a275f9de4b2281965cbb4ff48c3b`), no `registered_model_name`, no alias |

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
| Uncalibrated | 0.6684 | 0.1604 | 0.1402 | 0.1770 |
| **Sigmoid (selected)** | **0.6684** | **0.1343** | **0.0222** | **0.3111** |
| Isotonic | 0.6474 | 0.1342 | 0.0122 | 0.3117 |

**Selection: sigmoid, via `isotonic_disqualified_pr_auc_gate`.** Isotonic has the best Brier score and Expected Calibration Error (ECE) of the three, but its per-fold mean PR-AUC (0.6474) falls 0.0210 below uncalibrated — past the pre-registered materiality gate (Δ\* = 0.005) — so it's disqualified regardless of its calibration edge. Sigmoid leaves PR-AUC untouched by construction while still recovering most of isotonic's calibration gain.

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

#### Net benefit per catch and margin ratio

Two more figures fall out of the same cost table, and they matter for reading the expected-value comparison below correctly. **Net benefit per catch** (`retention rate × LTV − cost`) prices a single hit — a customer who was genuinely about to churn, got contacted, and stayed — netting the retention payoff against the cost of that one contact. It says nothing about the customers contacted at the same threshold who were never going to churn in the first place: those **wasted contacts** still cost `c` each but return $0, since there's no churn for the offer to prevent. **Margin ratio** (`payoff ÷ cost`, where payoff = `r × LTV`) restates the same trade-off as a multiple — how many times over a successful catch pays back what it cost to attempt.

| Scenario | Net benefit per catch | Margin ratio |
|---|---|---|
| Conservative | $59.61 | 3.71× |
| **Base** | **$104.18** | **2.54×** |
| Optimistic | $136.52 | 2.01× |

Optimistic has the largest net benefit per catch of the three but the thinnest margin — its cost scales with both a higher assumed ARPU and a higher discount rate, so the cost side grows faster than the payoff does. Conservative is the reverse: the smallest net benefit, but the fattest margin. This is the piece that explains why a bigger per-catch payoff doesn't automatically win in aggregate, below.

### Derived thresholds

Each scenario's threshold is checked two ways. The **closed form** calculates it directly from the cost math above (`c/(r·LTV)`) — pure arithmetic, no data involved. The **empirical check** goes the other direction: it tries every possible cutoff on real, calibrated probabilities to find which one would actually have performed best historically, then resamples that estimate thousands of times to build a plausible range (a 95% confidence interval), since a single historical best is noisy. The question that matters: does the closed-form answer fall inside that range?

| Scenario | `t*` (closed form) | Empirical argmax-EV | 95% bootstrap CI | Within CI? | Implied contact rate |
|---|---|---|---|---|---|
| Conservative | 0.2696 | 0.2497 | [0.2106, 0.2586] | ✗ | 40.3% |
| **Base (shipped)** | **0.3941** | 0.4150 | [0.3913, 0.5084] | ✓ | 30.9% |
| Optimistic | 0.4968 | 0.5474 | [0.4849, 0.6076] | ✓ | 24.1% |

Base and optimistic's closed-form `t*` fall inside their empirical argmax-EV bootstrap CI — the theoretical cost-derived cutoff agrees with where realized expected value actually peaks on held-out data. **Conservative is the exception:** its closed-form threshold (0.2696) falls just outside the empirical CI's upper edge (0.2586) — a real disagreement (≈0.011, about a fifth of the CI's width), not sampling noise. It does not affect the shipped policy, since base is what ships and base still agrees. Nor does it indicate broken calibration: the calibration slope (§5, ≈1.01) is an aggregate summary across the whole probability range and can pass comfortably while one narrow slice (roughly 0.25–0.27) is locally a little off — exactly the kind of local wobble an aggregate slope is built to miss. `threshold.py` logs this as a diagnostic warning; nothing is blocked.

**Base is the shipped operating point** (`t* = 0.3941`) — the other two are stored reference alternatives, not automatically-active fallbacks.

Neither extreme pays: contacting everyone costs money outright (sharply for base/optimistic, less so for conservative's cheaper contacts), and contacting no one gives up on every customer who could have been talked into staying. Between those extremes, expected value stays close to its best across a **broad plateau** of cutoffs before dropping to $0 near `t = 1` — nearby thresholds perform almost as well as the peak, so a slightly-off `t*` costs little, which is why each scenario's `t*` doesn't need to sit exactly at the empirical peak to be a good choice.

### Expected value at the shipped thresholds

| Scenario | `t*` | EV at `t*` ($/customer) |
|---|---|---|
| Conservative | 0.2696 | $8.51 |
| **Base** | **0.3941** | **$10.45** |
| Optimistic | 0.4968 | $9.62 |

Base outperforms optimistic in aggregate despite optimistic's larger net benefit per catch ($136.52 vs. base's $104.18, above) — the margin-ratio table explains the reversal: optimistic's margin (2.01×) is the thinnest of the three, and its cost per false alarm ($134.78, vs. base's $67.76) is the highest. A thinner margin and a costlier miss combine with a lower contact rate (24.1% vs. base's 30.9%) to pull optimistic's whole-base average below base's, even though its per-success dollar figure is the largest. A bigger per-catch payoff does not guarantee a bigger aggregate return once contact volume and margin are both in play — the reason base, not optimistic, is the better policy despite scoring lower on the single number ($136.52) that looks most impressive in isolation.

### Retention-rate sensitivity (base scenario, cost/LTV held fixed)

§0 already flagged `r` — the retention rate — as the model's dominant source of uncertainty; this table makes that concrete. Holding the base scenario's cost and LTV fixed, it re-derives `t*` — the operating threshold — at five different assumed values of `r`, showing how much the shipped threshold would move if the true retention rate turns out different from the 0.30 assumption.

| `r` | 0.15 | 0.20 | **0.30** | 0.40 | 0.45 |
|---|---|---|---|---|---|
| `t*` | 0.788 | 0.591 | **0.394** | 0.296 | 0.263 |

Halving the assumed retention rate exactly doubles the threshold — `t*` and `r` are related by pure inverse proportionality (`c` and `LTV` held fixed), since each contact is "worth less" in expectation when it's less likely to work. This is the single most consequential unmeasured assumption in the whole cost model, and the direction of the error matters: if real-world `r` turns out lower than the assumed 0.30, the shipped threshold is too low and the team contacts more customers than the offer's actual success rate justifies; if `r` is higher, the threshold is too conservative and leaves customers uncontacted who could have been saved — the table above brackets both directions, from 0.263 (`r`=0.45) to 0.788 (`r`=0.15).

The full threshold derivation — the per-scenario cost breakdown, the closed-form-vs-empirical-argmax comparison, the expected-value curves, and the retention-rate sensitivity sweep, all rendered from the actual run — is in [`notebooks/04-calibration-and-threshold.ipynb`](notebooks/04-calibration-and-threshold.ipynb).

### Dev-OOF diagnostics — reported, non-gating

`threshold.py`'s last step, immediately after deriving the threshold above, re-runs three dev-OOF diagnostics §0 defines as V1, V2, and V2b, on the 5,634-row development OOF set (~4× the sealed test set's churners) — more statistical power than the 1,409-row test set has on thin segments. **V1, segment collapse** (a segment's PR-AUC CI below its own churn-rate floor): 0 flagged. **V2b, per-group calibration collapse** (a group's slope CI entirely outside [0.80, 1.25]): 0 flagged. **V2, fairness disparity** (>10pp equal-opportunity or demographic-parity gap): 3 axes flagged — `seniorcitizen`, `has_partner`, `dependents` — each tracking a real, pre-existing churn-rate gap in the population (§2 EDA: seniors ~42% vs. ~24%; no-partner ~33% vs. ~20%; no-dependents ~31% vs. ~16%), not a proxy pattern the model invented. A flag from any of V1/V2/V2b changes nothing about the promotion decision — it only feeds the model card and ongoing monitoring.

A separate, fourth check on this same dev-OOF surface *is* a §0 guardrail: the aggregate **calibration-slope band check** (the dev-OOF half of §0's calibration-slope guardrail), applying the [0.80, 1.25] band once across the whole set rather than per group. The point is to protect the sealed test set, which can only be spent once: a model whose calibration is already dishonest isn't worth spending that one shot on. A failure here stops the process immediately, before the model ever reaches the test-set evaluation step, so the test set stays unspent for a better-calibrated attempt. This isn't a formal rejection — the model doesn't get marked as having failed a review; it simply never finishes this cycle, and sits in a holding state (`pending`, not `rejected`) until it's re-calibrated and tried again.

### Known limitations of the cost model

- **Horizon/offer-length mismatch.** LTV is credited over the full `horizon_months` (12) once a customer is retained, but the discount that (in expectation) retains them only runs for `discount_months` (3). The model implicitly assumes a 3-month offer buys a full year of loyalty; if churn risk resumes once the discount lapses, realized LTV recovered per successful intervention is overstated.
- **`r` is a single global constant, not a per-customer effect.** The true retention probability almost certainly varies by customer (contract type, tenure, price sensitivity) — this is a uniform-treatment-effect assumption, not an estimated uplift function `τ(x)`. A customer-level uplift model would let `q·r(x)·LTV(x) > c` replace the current population-average rule.
- **`r` cannot be recovered from this project's data, even under future retraining** — the raw dataset is a static snapshot with no experimental holdout or live customer feed, and resolving it requires an actual randomized retention-offer experiment (out of scope here). It's still the single highest-leverage input to validate, given how directly it moves `t*` (the sensitivity sweep above) — and since `t*` depends only on the cost parameters, not the model, a measured `r` could simply replace the assumed value in `configs/costs.yaml`, with no re-tuning, re-calibration, or model change required.

---

## 7. Final Test-Set Results

One-time evaluation on the sealed test set (n = 1,409; 374 churners), scored once by `evaluate.py` and diagnosed once by `error_analysis.py` — the same 1,409 customers held out since the original split, untouched by feature engineering, training, tuning, or calibration. Model version 1 evaluated cold-start (no incumbent champion to compare against). **Gate result: pass.** **Human review: approved** (Richlove Frimpong, 2026-07-27).

### Promotion gate — the four criteria

| Criterion | Role | Value | Bar / band | 95% CI |
|---|---|---|---|---|
| PR-AUC | selection | **0.670** | ≥ 0.60 | [0.619, 0.714] |
| Recall (base threshold) | guardrail | **0.698** | ≥ 0.65 | [0.651, 0.743] |
| Brier skill score | guardrail | **0.301** | > 0 | — |
| Calibration slope | guardrail | **0.992** | [0.80, 1.25] | [0.891, 1.100] |

Only PR-AUC can promote; recall, BSS, and calibration slope can only veto a model PR-AUC already admitted. All four pass, and none are borderline — PR-AUC clears its bar by 0.07, recall by 0.05 (with the whole CI above 0.65), and the calibration slope sits within 0.008 of the 1.0 ideal.

### Ranking — PR-AUC vs. ROC-AUC

PR-AUC **0.670** (95% CI [0.619, 0.714]) against a **0.265** dummy-prior floor (the score of guessing the churn rate for everyone). ROC-AUC — reported as a diagnostic only, never gated — is **0.848** (95% CI [0.828, 0.868]): the ~27% churn prevalence lets ROC-AUC look strong while still missing real churners, since false positives are always a small fraction of the much larger non-churner class. PR-AUC doesn't share that blind spot, which is why it drives the decision.

#### Classification performance at the three shipped thresholds

| Scenario | Threshold | Recall | Precision | F1 | TP | FP | FN | TN | Contact rate |
|---|---|---|---|---|---|---|---|---|---|
| Conservative | 0.270 | 0.789 [0.747, 0.829] | 0.526 | 0.631 | 295 | 266 | 79 | 769 | 39.8% |
| **Base (shipped)** | **0.394** | **0.698 [0.651, 0.743]** | 0.593 | 0.641 | 261 | 179 | 113 | 856 | 31.2% |
| Optimistic | 0.497 | 0.575 [0.525, 0.626] | 0.623 | 0.598 | 215 | 130 | 159 | 905 | 24.5% |

Base is what's shipped: it flags 261 of 374 churners for outreach while missing 113, and separately contacts 179 non-churners for nothing. Its recall — the number the gate's guardrail actually checks — clears 0.65 with its entire CI above the bar. Optimistic's recall would **not** have cleared the guardrail even at the top of its own CI (0.626 < 0.65), had it been shipped as the base scenario instead. F1 stays flat across all three (0.60–0.64) because the shipped threshold is chosen to maximize expected value, not F1.

A fixed-recall sweep of the same underlying curve shows what more aggressive targeting would cost: 70% recall (precision 0.59) lands almost exactly on base; 80% needs precision 0.52; 90% needs precision 0.47 — roughly a coin flip on every flagged customer.

#### Calibration

| Quantity | Value |
|---|---|
| Brier (candidate) | 0.1363 |
| Brier (dummy-prior floor) | 0.1950 |
| Brier skill score | 0.301 |
| ECE (diagnostic, not gated) | 0.0346 |
| Calibration slope | 0.992 [0.891, 1.100] |

Murphy's decomposition (Brier = reliability − resolution + uncertainty) shows the 0.301 BSS is earned honestly: reliability (calibration error) is tiny at 0.0020, resolution (discrimination) does almost all the work at 0.0598, and uncertainty (0.1950) is fixed by the data, not the model. The calibration slope — binning-free and immune to being bought by better ranking, unlike Brier — sits almost exactly at the 1.0 ideal with room on both sides of its CI before the [0.80, 1.25] guardrail band. The reliability diagram (10 quantile bins, ~141 rows each) tracks the diagonal closely; the largest single gap (predicted 0.62 vs. observed 0.52 in one bin, the model mildly overconfident) is offset by the opposite pattern in the top bin (predicted 0.72 vs. observed 0.77, mildly underconfident) — errors that cancel rather than compound in one direction.

### Ranking diagnostics — gains, lift, and the decile table

| Decile | Churn rate | Lift | Cumulative capture | Perfect-ranker ceiling | Headroom |
|---|---|---|---|---|---|
| 1 (top 10%) | 76.6% | 2.89× | 28.9% | 37.7% | 8.8pp |
| 2 | 51.8% | 1.95× | 48.4% | 75.4% | 27.0pp |
| 3 (~top 30%) | 50.4% | 1.90× | 67.4% | 100% | 32.6pp |
| 4 | 30.5% | 1.15× | 78.9% | 100% | 21.1pp |
| 5 | 26.2% | 0.99× | 88.8% | 100% | 11.2pp |
| 6–10 | ≤ 12.1% | ≤ 0.45× | 93.3% → 100% | 100% | ≤ 6.7pp |

A perfect ranker would already hold every churner by decile 3 (base churn rate ≈27%, so the top 30% has room for all of them). At that same decile 3, random targeting would capture ~30%, the model captures **67.4%**, and the ceiling is 100% — more than double random, with headroom shrinking from 32.6pp at decile 3 to under 1pp by decile 8. Lift drops below 1.0 starting at decile 5, and deciles 6–10 combined add only 42 more churners out of 374 — exactly the flat-tail shape a well-ranking model should produce. `mean_predicted` tracks `churn_rate` decile-by-decile with the same mild, offsetting over/under-confidence already seen in the reliability diagram, not a directional bias.

### Business impact — expected value across three cost scenarios

| Scenario | Threshold | Contacted | Campaign cost | Retained revenue | EV (95% CI) | EV vs. treat-all | Break-even retention |
|---|---|---|---|---|---|---|---|
| Conservative | 0.270 | 561 (39.8%) | $12,343 | $24,076 | **$11,732** [$9,496, $13,719] | −$479 | 10.3% |
| Base (shipped) | 0.394 | 440 (31.2%) | $29,814 | $44,875 | **$15,061** [$11,215, $18,604] | −$31,170 | 19.9% |
| Optimistic | 0.497 | 345 (24.5%) | $46,499 | $58,329 | **$11,830** [$6,794, $16,578] | −$88,440 | 31.9% |

All three scenarios beat both baselines — `treat-all` is negative throughout, `treat-none` is $0 by construction. Cost and revenue both scale up together across scenarios (~4× from conservative to optimistic), yet EV stays in a narrow **$11.7k–$15.1k** band — a range narrower than sampling noise: the cross-scenario spread ($3,329) is smaller than the widest single scenario's own bootstrap CI width ($9,784, optimistic), so the ranking between scenarios is not a confident signal even though all three clear both baselines.

A ±20% sensitivity sweep shows `retention_rate` and `ltv` as the two biggest swing factors (±$12.74 per customer around a $10.69 base — they enter the EV formula as a product, so an equal percentage miss on either moves revenue by the same amount); `cost` moves EV less (±$8.46) — even a 20% cost overrun leaves EV solidly positive. At the base scenario's true cost per contact ($67.76, not just the $20 labor component), break-even sits at 19.9% retention against a 30% assumption — real room for error before the campaign stops paying off.

**Bottom line: deploy at the base scenario.** All three scenarios are profitable, but base is the only one both financially comfortable (a real margin to its break-even rate) and operationally executable (440 contacts fits under the 500-contact operational capacity; conservative's 561 does not).

### Disaggregated results — robustness & fairness

Sliced two ways on the sealed test set: **robustness** (`contract_type`, `tenure_cohort`, `internetservice`) and **protected/quasi-protected** axes (`gender`, `seniorcitizen`, `has_partner`, `dependents`). Neither surface gates promotion at this stage — reported for the model card and for production monitoring, with a companion check on the larger dev-OOF sample (§6) for anything the thinner test-set slices can't distinguish from noise.

Among robustness axes, the model is strongest exactly where the campaign spends its budget: month-to-month contracts (PR-AUC 0.704) and 0–12-month tenure (0.782) rank confidently, while two-year contracts (0.157, 2.7% churn rate) and 65+-month tenure (0.371) carry wide CIs from thin churner counts, not worse ranking. Among protected axes, `gender` is nearly identical (0.678 female vs. 0.667 male); `seniorcitizen`, `has_partner`, and `dependents` all show real ranking gaps, but in the direction of the group that churns more (e.g. seniors rank *better*, 0.758 vs. 0.639).

At the shipped threshold, `dependents` shows the largest fairness gap: customers without dependents are contacted more than twice as often (37.5% vs. 15.7%) and missed less often (28.0% vs. 45.8% FNR — a 35.1pp equal-opportunity gap) than customers with dependents. `has_partner` (19.4pp) and `seniorcitizen` (12.3pp, in the opposite direction — seniors are contacted *more* and missed *less*) show smaller versions of the same pattern; `gender`'s gaps stay small (≤3.8pp). Translated to dollars, the `dependents` split is starkest: customers without dependents net $14,499 EV vs. just $561 for customers with dependents, whose missed revenue ($5,502) exceeds what's retained ($3,611). Every slice's calibration-slope CI overlaps the [0.80, 1.25] band except two-year contracts (2.294, from only ~9 churners — a thin-support artifact, not real miscalibration, and one that never reaches an outreach decision since two-year contracts sit at a 0% selection rate anyway).

These gaps track the same real, pre-existing churn-rate differences already noted against the dev-OOF V2 flag (§6) — a risk-targeted campaign contacting a higher-risk group more and missing fewer of its churners is doing what it should, not exhibiting an unexplained preference. The choice of a single EV-maximizing threshold accepting this as the cost of uneven risk (vs. deliberately subsidizing outreach to lower-risk groups) is a policy question for the business to revisit each retrain cycle, not a modelling defect.

### Error analysis — where, how, and what it costs

**Missed churners (false negatives) cluster on two distinct blind spots.** The dominant one is long tenure plus a long-term contract: 92.8% FN rate at 61–72 months, 99.2% on one-year contracts, 100% on two-year — a near-absolute "long tenure + long contract = safe" prior. This alone misses 169 real churners (130 one-year, 39 two-year), and they are not low-value: one/two-year churners average $85–87/month, more than the $73/month average for the month-to-month churners the model does catch. A second, separate blind spot is low monthly charges (75% FN rate in the cheapest bin) and no internet service (90%) — customers who never trigger the high-charges signal the model otherwise relies on.

**False positives cluster on two different patterns.** New sign-ups on flexible terms (51.2% FP rate at 0–6 months tenure, overwhelmingly month-to-month and paying by electronic check) and moderate-tenure customers paying more per month who simply stay anyway (fiber optic, no add-ons, high charges — 27–51% FP rate). Both patterns are a consequence of the same structural gap: no feature in this dataset measures loyalty or satisfaction directly, so nothing cleanly separates a genuinely at-risk customer from one who merely looks like one on paper.

**Most errors are confident, not borderline.** At `t* = 0.3941 ± 0.0586`, only 9.7% of false negatives and 12.8% of false positives fall inside that near-miss band — the rest are confident failures a threshold change cannot fix. Missed churners scatter thinly across the entire low-score range (many near 0); false alarms bunch tightly around 0.55–0.65, comfortably above the cutoff, not barely over it. Both are feature problems, not threshold problems.

**Errors are not evenly distributed by customer value.** Splitting by `MonthlyCharges` decile (a ~5.5× revenue-at-risk gap between the cheapest and priciest tenth), the FN rate is high at both ends and low in the middle: the two cheapest deciles miss 73–86% of churners (each miss cheap), but the FN rate climbs back to 54.8% in the single most expensive decile (~$108/month) after a healthy 15.8–18.2% in deciles 6–9 — the model's highest-value blind spot is concentrated, not a general degradation at high value.

### Model explainability — SHAP

**Global importance.** Two features carry over a third of total signal: `Contract Type: Month-to-month` (mean |SHAP| 0.647) and `Tenure` (0.431). A second tier — Monthly Charges, Tech Support: No, Contract Type: Two year, Online Security: No, Internet Service: Fiber optic, Payment Method: Electronic check (0.17–0.22 each) — brings the top 8 to 73% of total signal. `SeniorCitizen`, `Gender`, `Dependents`, and `Partner` all rank below every commercial feature except Paperless Billing, keeping demographic signal marginal by construction, not by post-hoc filtering.

**Direction sanity check — the one check in this section that can veto.** All 8 top-ranked features' observed SHAP direction agree with the established EDA relationship (`observed_direction` all correctly signed, 0 violations): month-to-month, high monthly charges, no tech support, no online security, and fiber optic all push toward churn; a two-year contract and long tenure both push away from it. No sign flip means the model didn't learn a backwards relationship on any feature this influential — **check passed.**

**Signed cohort SHAP** explains the two error types as opposite failure modes. False negatives are a **dilution** problem: the same features that argue "churn" strongly for true positives (Month-to-month +0.70, Tenure +0.45) argue it only weakly for missed churners (+0.12, barely positive) — the signal goes quiet rather than reversing. False positives are a **collision** problem: the same features argue "churn" for false alarms almost as strongly as for real churners (+0.68 vs. +0.70) — these customers look like real churners on every feature the model relies on most. Neither is fixable by moving the threshold. A closer look at the annual-contract false negatives specifically shows Monthly Charges *does* react in the right direction (median $100 for missed churners vs. $61 for true negatives on the same contract type, Mann-Whitney p = 6×10⁻¹⁰) — it's just outvoted roughly 2:1 by Contract Type's opposing pull, a weighting problem fixable via an interaction feature or reweighted refit, not a missing-signal problem like the tenure/loyalty gap.

**What this means going forward:** closing the false-negative/false-positive overlap needs a genuine loyalty or satisfaction signal this dataset doesn't have — not a different threshold. Two nearer-term, cheaper fixes exist without new data: manually flag long-contract, high-bill customers for outreach today (a targeting rule), and engineer a spend × contract-length interaction feature or reweight the next refit (a modelling fix). Phone-only customers with no internet service are a separate, smaller blind spot needing new feature engineering, since almost every warning signal the model has is internet-service-based.

#### Business takeaways

- **Today, no model change needed:** manually flag long-contract (one/two-year) customers whose monthly bill sits in the top 25% for retention outreach — a spreadsheet filter. The model already reacts to their spend, just not strongly enough to outweigh Contract Type's pull toward "safe."
- **Front-load retention effort into a customer's first few months.** Tenure produces the single widest score swing of any feature — a 1-month customer can be pushed further toward risk than almost anyone else in the dataset — so early tenure is where a fixed outreach budget buys the most risk reduction per customer.
- **Next cycle, a modelling fix:** an engineered spend × contract-length interaction feature, or a reweighted refit, would let the model catch these annual-contract, high-bill customers itself rather than relying on a manual filter.
- **Longer-term, needs new data, not a targeting rule:** phone-only customers with no internet service are nearly invisible to the model, since almost every signal it relies on is internet-service-based — this is a feature-engineering gap, not a threshold one.
- **The root structural gap:** nothing in this dataset measures loyalty or satisfaction directly. That single missing signal explains both failure types at once — quietly-leaving long-tenure customers the model reads as safe, and new sign-ups who look risky but were never leaving — and closing it would do more for both error rates than any threshold or calibration change.

### Human review & promotion decision

Every automated criterion above passed; the direction sanity check found no violations; segment and per-group calibration collapse were both clean on the dev-OOF surface (§6). The one flagged item — the fairness-disparity gap on `seniorcitizen`/`has_partner`/`dependents`, also flagged by the dev-OOF V2 check (§6) — was reviewed and judged consistent with genuinely higher churn risk in these groups rather than proxy discrimination, and is tracked for ongoing monitoring rather than treated as a blocker. **Verdict: approved** (Richlove Frimpong, 2026-07-30), clearing the model for registration in §8.

The full sealed-test walkthrough — the classification, calibration, and ranking diagnostics; the disaggregated robustness/fairness slices; the error analysis and SHAP explainability findings; and the direction-sanity check (V3, the one automatic veto) — all rendered from the actual run — is in [`notebooks/05-evaluation-and-error-analysis.ipynb`](notebooks/05-evaluation-and-error-analysis.ipynb). The promotion verdict itself is stamped separately, outside the notebook, via `models/review.py`.

---

## 8. Model Registration & Promotion

**Run for real on 2026-07-30.** `telco-churn-pipeline` version 1 — already registered as `challenger` once calibration finished — cleared the cold-start gate and was aliased `champion` at `2026-07-30 13:00:37 UTC`. It is the first and, as of this writing, only champion this project has had.

### Rationale

Hyperparameters, calibration, and the operating threshold were all finalised on held-out data before evaluation (§4–§6). The model evaluated in §7 is exactly the artifact that gets registered, **directly, with no separate full-data refit** — there is nothing left to retrain before it serves.

The no-refit choice is itself a trade-off, not a default: §9's learning-curve finding shows the extra 1,409 test rows would plausibly still buy some ranking quality, but on an already-flattening curve the estimated gain is modest, and refitting into the sealed test set would permanently forfeit the project's only holdout on a dataset with no time axis to ever produce a fresh one. That trade was judged not worth it — see §9 for the full reasoning.

### Design decisions in `register.py`

The four gate criteria and their cold-start bars are §7's table, not repeated here — §7 is where they're computed and where the "gate: pass" verdict is first reached. What's specific to registration is that `register.py` never recomputes any of it: it resolves `promotion_decision.json` from the exact evaluation run tagged onto this model version, hashes `metrics.json` fresh and checks that hash against the one the decision stamped, and only then reads `"gate": "pass"` off the file. The same tag-based resolution extends to `error_analysis.json`, read via the model version's own `error_analysis_run_id` tag — never a local `reports/` path, since a fixed path reflects whichever run last executed on this machine, not necessarily this model version's own cycle (`reports/` is still written to disk for human inspection, but nothing here reads it back). The human reviewer (Ampofowaa, `2026-07-30 13:00:07 UTC`) stamped `"review": "approved"` onto that same file in `notebooks/05-evaluation-and-error-analysis.ipynb`, noting the V2 fairness gaps on `dependents`/`has_partner` (§7) as tracked-not-blocking and confirming V3 fired no direction-sanity violations. `register.py` refuses to act without that stamp present (`register.require_review: true`).

**Validate, then promote — the alias only moves after every check has already passed.** `register.py` verifies the environment, model schema, and golden-prediction parity against the candidate version *before* touching the alias. Only on a pass does it flip `champion` — and then it reloads *through the alias* and reconfirms golden parity a second time, rolling back immediately if that fails. The two-sided check exists because the failure it guards against isn't "the model is bad," it's "the alias points at something other than what was just verified" — a distinct and equally real failure mode.

**`promotion_status` starts at `pending` and resolves within its own cycle.** A version that crashes mid-evaluation is left at `pending` rather than at some inferred bad state — the safe default is the one a crash produces for free, not one that has to be written on every failure path. A `pending` version older than one training cycle is therefore a crash artifact: it never earned `promoted` (no rollback value) and was never decided (no audit value).

**Rollback selects on the `promotion_status` tag, never on version arithmetic.** "Roll back to version N−1" is the wrong operation here: a rejected or crashed cycle can leave a higher-numbered version with no alias and no `promoted` tag, indistinguishable from a legitimate former champion by number alone. `rollback_champion()` accepts an explicit target version but still refuses one not tagged `promoted`; without one, it defaults to the highest version still tagged `promoted`, excluding the current champion itself — otherwise a rollback could silently resolve right back to where it started, since `promotion_status` is never cleared once set.

**Every promotion and rollback appends to an ordered `promotion_log`** tag on the registered model itself (not per-version), so "what was champion before this one" is a direct index into an append-only history rather than a tag-scan reconstruction.

### MLflow model registration

Registered model: **`telco-churn-pipeline`**, version **1**, alias **`champion`**.

**Artifacts on the registered run** (the first four logged at training/calibration time, the last two written by `register.py` itself, at promotion):

| Artifact | Written by | Description |
|---|---|---|
| `model/` | `calibrate.py` | MLflow pyfunc — `mlflow.sklearn.load_model()` |
| `feature_space.txt` / `feature_columns.txt` | `models/train/log_model.py` | Full feature space vs. the subset that survived selection into the model |
| `preprocessing.pkl` | `models/train/log_model.py` | Fitted `ColumnTransformer` |
| `training_manifest.json` | `models/train/log_model.py` | Git SHA, data content hash, hyperparameters, tuning summary |
| `registration/drift_reference.json` | `register.py` | Monitoring baseline |
| `registration/model_card.json` | `register.py` | Stakeholder-facing summary |

```bash
# Load the champion model by alias
mlflow.sklearn.load_model("models:/telco-churn-pipeline@champion")
```

### Drift-monitoring baseline & the model card

`register.py` closes the cycle by building and logging two artifacts onto the promoted run, neither hand-authored, each for a different audience:

- **`drift_reference.json`** — the baseline a future scheduled drift check and monitoring dashboards will compare live traffic against. Per-feature reference distributions from the 5,634-row dev training population, plus an out-of-sample prediction-score distribution built by unioning dev-OOF and sealed-test predictions (all 7,043 rows, each scored by a model that never trained on it). The score half specifically must come from out-of-sample scores — the champion's own in-sample scores are systematically sharper than anything it will produce in production, and baselining on them would report permanent phantom drift against honest live traffic. A rollback that re-points `champion` at an earlier version carries its own reference with it (read by resolving the alias, never a fixed path), so a rollback can't leave live traffic being compared against the wrong model's baseline.
  - **Capture method: the training set itself, not a live-traffic window.** A drift baseline can generally be captured one of three ways — the training set, a vetted early-production window, or a rolling window of recent production data. This project uses the first, by necessity as much as choice: `build_reference()` (`drift_reference.py`) takes the dev training population and the unioned out-of-sample score vector directly, since there is no live traffic to draw a trusted-window or rolling-window baseline from (§9 #13 — the dataset is a static 7,043-row export, not a stream). The other two methods assume a production feed this project doesn't have; a real deployment with live traffic would be a candidate to switch to one of them.
  - **Minted via MLflow, checked later by Evidently — two separate tools for two separate jobs.** `register.py` builds `build_reference()`'s output and writes it with a plain `client.log_dict(...)` call onto the promoted run, as the last step of the promotion sequence (after the alias flip and golden-parity reconfirm), inside the same rollback-guarded unit as `model_card.json` — a failure here rolls `champion` back to the prior promoted version rather than leaving it pointed at a version with no baseline. Evidently has no part in minting it; once built (Phase 10's `drift_check.py`, Phase 13's `monitoring/drift.py`), its role is purely to *consume* this artifact — fetch it by resolving the `champion` alias, fetch the current window, and compute the comparison (PSI etc.). The reference is a registry-owned artifact independent of whichever drift-detection library ends up reading it.
- **`registration/model_card.json`** — the stakeholder-facing artifact: a business reader, not a pipeline, is the consumer. Assembled entirely from `metrics.json`, `economics.json`, `error_analysis.json`, and `promotion_decision.json` — its executive summary, error-pattern narrative, and driver summary are all computed fresh from those artifacts at registration time, so nothing is hand-transcribed and none of it can silently go stale under a future retrain the way a hand-written card would. Baselines throughout are `treat-none`, `treat-all`, and the `DummyClassifier` prevalence floor — not LogReg, the linear candidate the model-family comparison in §4 already eliminated. A business reader needs to know what the model buys over doing nothing or guessing the base rate, not how it stacks up against an alternative that was never going into production.

### Alias lifecycle

`challenger` only ever moves in one place — `calibrate.py`, when the next training cycle mints a new version. No path in `evaluate.py` or `register.py` re-points or clears it when a candidate is rejected, so a rejected candidate stays visibly aliased `challenger` until a later cycle's mint supersedes it. This is a deliberate split, not an oversight: `challenger` is a **positional** pointer ("the most recently trained candidate"), while the **verdict** ("did it pass") lives entirely on the `promotion_status` model-version tag (`pending`/`promoted`/`rejected`) — collapsing the two into one signal is exactly what MLflow's now-deprecated `stages` field did, and exactly what its replacement (aliases + tags, mirroring SageMaker's `ModelApprovalStatus`) is designed to avoid.

---

## 9. Known Limitations

> **Two kinds of entry live here, and they are marked.** Items tagged **⚠ UNVERIFIED** were empirical claims *inherited from the exploratory pass* — measured on a different model, at a superseded threshold (0.2956, now `t* = 0.3941`), under a different calibration. They were **not** established facts about the shipped champion, and were recorded as open questions rather than findings until Phase 7's error analysis re-derived each one against the real model. **All four have since been rewritten or retired against that evidence; none remain tagged.**
>
> Everything else is structural — true of this problem and this dataset regardless of which model is fitted — and stands on its own evidence. *(Two former entries have been removed: "no re-contact suppression" was a roadmap item, not a limitation, and already appears under §10's short-term recommendations; "production monitoring not yet deployed" was a phase status, and is superseded by #13, which makes the sharper point.)*
>
> Grouped below by kind — model behaviour, business assumptions, methodology/engineering trade-offs, and data/production constraints — rather than by discovery order. Cross-references elsewhere in the codebase (`PROJECT_PLAN.md`, `register.py`) cite items by this numbering; if this section is ever reordered again, those citations must move with it.

### Model Behaviour & Blind Spots

*Confirmed, measured properties of what the shipped champion does and does not catch.*

1. **Annual/multi-year contract churners are a near-total blind spot at the shipped threshold — confirmed on the champion.** The model misses 129 of 130 dev-OOF churners on one-year contracts (99.2% FN rate) and all 39 on two-year contracts (100%), measured at `t* = 0.3941` — close to the archived pass's 0.972/1.000. This is a threshold-placement effect, not a ranking collapse: two-year contracts' PR-AUC (0.063, 95% CI [0.041, 0.104]) still clears its own churn-rate floor (0.029), so V1 does not fire — the model ranks that segment's churners above its non-churners well enough, it simply never predicts a probability above `t*` for any of them. The "annual contract = committed customer" heuristic the archived pass suspected is real and persists through recalibration.

2. **Long-tenure customers are under-served, in the same direction as the archived estimate but not a like-for-like number.** The archived pass measured a 0.465 FN rate across a single 25+ month band; the current champion's finer quantile bins show 72.1% (39–61 months, 132 of 183 missed) and 92.8% (61–72 months, 64 of 69 missed) at the shipped threshold. The pattern — longer tenure, worse miss rate — is confirmed and the magnitude is starker, though the different binning means this isn't a direct replication. As with contract type, it's a threshold effect rather than a ranking collapse: the overlapping `tenure_cohort` bands (49–65m, 65+m) both clear their PR-AUC floor comfortably, so V1 does not fire here either.

3. **Non-churners sharing a risky profile are still over-flagged — confirmed on the champion, but the headline magnitude does not replicate.** The exploratory pass measured elevated false-positive rates on two single axes of that profile — 54.0 % for fiber-optic customers and 60.1 % for month-to-month customers — but never computed a joint rate for the three-way fiber-optic + month-to-month + no-add-on combination itself. At the shipped threshold (`t* = 0.3941`) the champion's overall sealed-test FP rate is far lower than either archived single-axis figure — 179 of 1,035 non-churners (17.3 %) — so neither archived rate carries over as a like-for-like comparison. What does replicate, and sharper than either archived single-axis figure, is the *structural* claim underneath them: the dev-OOF cohort scan resolves the false alarms into two distinct clusters rather than one — new sign-ups on flexible terms (51.2 % FPR at 0–6 months tenure, 40.1 % month-to-month, 42.4 % electronic check) and moderate-tenure premium subscribers who simply stay (30–51 % FPR in the high-charges bins, 36.1 % fiber optic, 27–37 % across the no-add-on features). Both clusters are the same underlying failure the archived pass named — no loyalty, satisfaction, or recency-of-service-change signal exists in the IBM feature set, so nothing distinguishes a contented customer with a risky profile from a genuine pre-churner — but the archived pass's two isolated 54–60 % single-axis rates understated how heterogeneous the false-alarm population actually is.

4. **Retired — the archived high-score calibration gap does not hold for the shipped champion.** The exploratory pass found the calibrated model under-predicting churn probability for high-scoring customers by up to ~10 pp above score ≈ 0.58, which would understate financial exposure exactly where it matters most. The dispute is settled: the champion's sealed-test **calibration slope is 0.992** (95 % CI **[0.891, 1.100]**), comfortably inside §0's [0.80, 1.25] guardrail band, and the reliability diagram's largest residuals — both in the same high-score region the archived pass flagged — point in *opposite* directions (mildly overconfident at predicted ≈0.62, observed 0.52; mildly underconfident at predicted ≈0.72, observed 0.77), cancelling rather than compounding into a one-directional under-prediction. The champion's high-score posteriors are honest enough for `t* = c/(r × LTV)` to remain the right threshold.

### Business & Economic Assumptions

*Inputs the expected-value calculation depends on that this dataset cannot itself supply.*

5. **Contact capacity is unknown, and until Phase 7 it was not even a parameter.** *(Structural.)* The expected-value calculation assumed **unlimited capacity**: `t*` flags a fraction of the base, the EV sums over all of them, and nothing asked whether the retention team can actually make that many calls. An assumption that is not written down cannot be checked, and this one was invisible. `configs/costs.yaml` now carries `contact_capacity` / `campaign_budget` explicitly — but the *value* is a placeholder, not a measured operational number, and it shares `r`'s status: a business input this dataset cannot supply. **The consequence is bounded, not fatal:** the EV-vs-K curve (§7) reports expected value at *every* contact volume, so a stakeholder with a real capacity reads their own number off it without re-deriving anything. What cannot be claimed is that the headline EV is achievable — only that it is the EV *if* the implied contact volume is affordable.

6. **Business cost parameters are illustrative.** *(Structural.)* Not Finance-validated. The shipped values live in `configs/costs.yaml` — three scenarios spanning `outreach_cost` $5–$50, `retention_rate` 0.20–0.40, and ARPU taken from churner `MonthlyCharges` quantiles at a 12-month horizon and 0.60 gross margin. **`r` is the load-bearing guess** (§0): it is the one parameter this dataset cannot supply, `t*` is inversely proportional to it, and it moves the operating point more than any realistic modelling improvement does. *(The `$68`/`$575` figures previously quoted here were from the exploratory pass and do not correspond to any shipped scenario.)*

### Methodology & Engineering Trade-offs

*Deliberate design choices in how features, hyperparameters, and guardrails were built — each with a known gap or an open validation question.*

7. **No uplift / persuadability modelling.** *(Structural.)* The model identifies *who will churn*, not *who will respond to a retention offer*. Without A/B test data it cannot separate persuadables from lost causes — and this is what the retention rate `r` (§0) papers over: the expected-value calculation assumes contacted churners are retainable at a benchmark rate rather than modelling who actually is. The EV is therefore a model of a model.

8. **Feature discovery redundancy screen has a mixed-type gap.** Screen 2 in the lap framework checks numeric-vs-numeric relationships (Pearson) and categorical-vs-categorical (Cramér's V) but has no branch for categorical-derived-from-numeric (e.g. `tenure_cohort` vs `tenure`). Screen 4 — permutation importance given all adopted features — is the empirical backstop: a feature that adds no marginal signal because the model already has the underlying numeric column is identified and rejected regardless of type. On the dev-partition run, `tenure_cohort` (Lap 5) was redundant enough to fail earlier, directly at Screen 3 (PR-AUC fell −0.0041) — Screen 2 raised no flag, since it has no cross-type branch, but the feature never reached Screen 4. `two_year_fiber` (Lap 1) demonstrates the Screen 4 backstop directly on this run: Screen 2 passed (max_corr 0.389), Screen 3 passed (+0.0017 PR-AUC), and Screen 4 correctly rejected it (importance 0.0001, below the 0.0054 floor) once measured against the full adopted context.

9. **Learning curve had not plateaued at Phase 5 Step 2c.** CV PR-AUC was still rising at the maximum training size available to any single CV fold in the Steps 2c/2d generative diagnostic loop (0.610 → 0.652 from 20%→100% of the dev-training folds) — more historical data would plausibly still improve ranking quality, and a fold's training partition is itself smaller than the eventual full-development refit. That same gap is why an early-stopped tree count measured on a smaller training partition (3,606 rows) needs scaling for a final fit trained on more rows (5,634): §4c derives and measures the correction (94 → 147 trees, +0.0015 CV PR-AUC on the two-count diagnostic). Not acted on with a new feature in Phase 5 (no feature was engineered in response); flagged as a Phase 10 retrain / data-acquisition consideration. **This curve is also why Phase 7 does not refit into the sealed test set.** On this project's own evidence, the extra 1,409 test rows would plausibly still buy some ranking quality — but the curve is already flattening, so the gain is judged modest, and weighed against permanently forfeiting the project's only holdout (on a dataset with no time axis to ever produce a fresh one), that trade was not taken. The evaluated model registers directly as champion instead (§8).

10. **The calibration-slope guardrail's band, `[0.80, 1.25]`, is asserted policy, not derived or validated.** *(Structural.)* The method itself — the Cox calibration slope, regress `y` on `logit(p)` — is standard practice in clinical prediction modelling and directly analogous cost-based decision-rule settings (credit risk, insurance underwriting), where a probability feeds a real decision the way `t*` does here. But the specific numeric width of the veto band has no citation and no project-specific derivation: nobody has checked, by simulation or otherwise, what degree of true miscalibration the band actually catches, or whether `[0.80, 1.25]` is well-matched to this dataset's scale. What **has** been validated is the *measurement* inside the band, not the band itself: `calibrate.py::calibration_slope`'s percentile-bootstrap CI has been independently cross-checked against a closed-form analytic (Wald) CI, confirmed correct via a finite-difference Hessian check in `tests/unit/test_calibrate.py`, and the two agree closely on this run's actual data (§2 "Bootstrap vs. analytic (Wald) CI cross-check", `notebooks/04-calibration-and-threshold.ipynb`) — so the number being compared against the band is trustworthy. Whether `0.80`/`1.25` are the right edges for that number remains open.

    **A related, sharper finding surfaced while building that cross-check.** The calibration slope alone can be a misleading summary: the *uncalibrated* dev-OOF probabilities already have a slope near 1.0 (0.98), even though the reliability diagram shows them clearly, one-directionally overconfident. The actual defect is in the **intercept** (≈ −0.98, vs. ≈0.00 after calibration) — the uncalibrated mean predicted probability (≈41%) sits far above the observed dev churn rate (≈26.5%), a "calibration-in-the-large" distortion the slope is largely blind to by construction (it measures relative spread, not overall level). `calibrate.py` now persists `uncalibrated_calibration_slope`, `mean_p_hat_calibrated`, `mean_p_hat_uncalibrated`, and `observed_churn_rate` alongside the existing `calibration_slope`, so this comparison is available every cycle rather than a one-off notebook observation. **The consequence for the guardrail:** a model whose intercept is badly off but whose slope happens to sit inside `[0.80, 1.25]` would clear this specific check despite being clearly miscalibrated in aggregate — the slope catches one specific failure mode (bad relative discrimination), not every way a model's probabilities can be dishonest.

    **The concrete follow-up, not yet done:** a coverage/sensitivity simulation — insert a known degree of miscalibration (varying both slope and intercept independently) into synthetic data shaped like this dataset, and check where the veto rule actually starts firing, and whether that threshold corresponds to a miscalibration severity that would meaningfully distort `t*` in practice.

### Data & Production-Readiness Constraints

*What the dataset's size and static nature limit — regardless of model quality or engineering effort.*

11. **The sealed test set is too small to support subgroup conclusions, so subgroup/fairness scrutiny (V1/V2/V2b) is read on development-set evidence and does not gate promotion.** Phase 7 reports disaggregated PR-AUC on the sealed test set — per contract type, tenure cohort, internet service, and the four protected/quasi-protected axes — because that is what a model card requires and what the published metrics of record should cover. But 1,409 test rows do not divide into seven axes and leave usable support: the two-year `contract_type` tier churns under 3 %, so it carries on the order of **ten churners**, and a PR-AUC estimated on ten positives has a CI wide enough to be uninformative. The consequence is stated plainly: **V1/V2/V2b are computed on the dev-OOF slices** (5,634 rows, roughly four times the churners per slice) for the power they need to say anything at all, and the test-set slices are *reported* alongside rather than acted on — but neither surface gates this decision (§0). Dev-OOF is evidence too noisy in provenance to bind a test-set-centered promotion decision, and the sealed test set is too noisy in sample size to bind one either; forcing either into a veto would trade one failure mode for another. What is therefore never verified as a promotion-time gate is whether the model's **subgroup** behaviour generalises — only its aggregate performance is, via V3 and the automated gate. This is a deliberate trade (a held-out estimate too noisy to trust, paired with a dev estimate that trusts the wrong data, is worth less than moving the decision to where both problems dissolve), not an oversight, and it is a limitation of the dataset's size, not of the method. It would dissolve on any realistically-sized production dataset, where every slice has thousands of positives and the question does not arise. **Downstream consequence:** Phase 10's `performance_monitor.py` compares realised per-segment performance — robustness *and* fairness axes, plus per-group calibration — against the *test-set* baselines (the published numbers), so a thin-support segment cannot support a tight degradation alert — its alert band must be derived from its baseline CI, not a fixed global threshold. This is where V1/V2/V2b's enforcement actually lives: continuously, against accumulating production volume, rather than once against a single undersized snapshot.

12. **Calibrated probabilities are calibrated *to development-set prevalence* — prevalence drift invalidates both the calibration and the threshold.** LightGBM trains with `class_weight='balanced'` (`models/train/common.py`), which systematically inflates scores toward the positive class. `CalibratedClassifierCV` corrects this because sklearn fits the calibrator on **unweighted** out-of-fold data, mapping the reweighted scores back to true posteriors against the real ~26.5 % dev prevalence. That is the right behaviour, and it is the mechanism that makes the closed-form threshold `t* = c/(r × LTV)` valid at all — `t*` is only Bayes-optimal against *honest* posteriors. The dependency runs both ways: if production churn prevalence shifts away from 26.5 %, the calibration map is stale, the posteriors are biased, and `t*` is being applied to numbers that no longer mean what it assumes. Neither a PR-AUC check nor a reliability diagram computed on old data will catch this. **This is the hook for Phase 13 drift monitoring:** track prevalence alongside feature PSI, and treat a sustained prevalence shift as a re-calibration trigger, not merely a retrain trigger — they are different remedies.

13. **The monitoring and continuous-training loops are *mechanism demonstrations*, not live signals — the dataset is a static snapshot with no customer feed.** A production ML monitoring stack is normally built on three assumptions: traffic keeps arriving (so drift can be observed), outcomes eventually mature (so accuracy can be checked against reality), and the dataset keeps growing (so a fresh holdout is always available for the next comparison). `customers_raw` violates all three by construction — it is a frozen 7,043-row Kaggle export, it does not grow, and no real customer is ever scored against it in production. (`customers_crm` — `serving/crm_data.py`'s seeded, deterministic nudge of `customers_raw` that the Lookup/Batch demo reads from — does not contradict this: it makes serving-time lookups honest about *not* being the training snapshot, but it is still a fixed derivation of the same static 7,043 rows, not a live feed. It simulates that a served row is a *different* moment for a *real* customer, never that new customers or genuinely fresh traffic exist.) Five consequences follow, and none of them are bugs — they are the honest boundary of what a monitoring stack built on a static, one-time snapshot can ever claim:
    - **Feature drift cannot occur.** The input distribution is fixed by construction, so PSI against the champion's `drift_reference.json` will sit at ~0 indefinitely. The reference, the Evidently report, the PSI threshold, the alert routing, and the version-scoped rollback fidelity are all real and correct; there is simply no drift for them to find. Phase 10's verification step injects synthetic rows with shifted `MonthlyCharges` precisely because that is the only way to exercise the path.
    - **The realised-performance loop has no labels to mature.** `performance_monitor.py` joins logged predictions to observed churn outcomes in `prediction_outcomes` — but with no live customers, no outcome ever arrives. The retention rate `r` (§0) is the parameter this most affects: it stays a benchmark guess forever, and the plan's claim that deployment turns it into a measured quantity is true of the *design*, not of this instance.
    - **The promotion gate's evaluation surface: what industry does, what this project substitutes for it, and why the substitute is a limitation.**

        *What industry does.* A continuous-training system evaluates by **walk-forward (rolling-origin) validation** — train on data up to time T, evaluate on the genuinely-later, never-seen slice T→T+Δ, then slide both windows every cycle so last cycle's evaluation window folds into next cycle's training data. There is no persistent "the test set," only "this cycle's window," always the freshest real data and never reused across cycles. The one exception is the very first model: cold start uses a classic one-shot held-out test set (often the most recent slice of history), touched once to clear an absolute bar, after which it simply becomes training data like everything else. A separately-kept frozen "golden" or regression set that persists across versions is a sanity surface, not a promotion gate — erosion does not apply to it because no statistical go/no-go decision rests on it.

        *What this project does.* Cold start (Phase 7) follows the norm exactly: `evaluate.py` touches the sealed test set once, v1 clears the absolute bars, and registers directly as champion. The **routine** retrain (`performance_check.py`-gated, `PROJECT_PLAN.md` Phase 10a-ii/10b) evaluates each candidate against the **`reserve` partition** — a 6-month carve-out from the original test set, delivered through the real serving path (`customers_crm → /predict → prediction_log → outcomes.py → prediction_outcomes`), split into monthly cohorts and folded forward one at a time — and **never touches the sealed test set at all**. This is the manufactured stand-in for walk-forward's rolling window. It also covers a drift-triggered cycle, routed here rather than to `evaluate.py` precisely because a drift-motivated comparison needs current data the frozen seal cannot give. The **rare** retrain (`evaluate.py`-gated: manual trigger or a `COMMITTED_FEATURES` mismatch — a genuinely new model generation) reuses the original sealed test set each time it fires.

        *Why it is a limitation.* The dataset is static — 7,043 rows, no calendar timestamp (`tenure` measures months-as-customer, not record date), and it never grows — while true walk-forward needs both a real time axis and an accumulating dataset. So the rolling window is manufactured: the `reserve` partition is a fixed carve-out on an invented release schedule, and it supports exactly **2 fold-forward cycles** before exhaustion (3 of its 6 months used), because nothing keeps that manufactured calendar advancing past what the reserve itself creates. The rare-cycle path is the residual exposure — it reuses one frozen holdout indefinitely, with no cap and no logged reuse count anywhere in the registry; each reuse makes the estimate quietly more optimistic (by then every modelling decision across every prior cycle has been implicitly tuned against it), mitigated only by "rare" being infrequent by convention, with nothing enforcing that. None of this special-casing would exist on a real growing dataset: the seal would become training data after v1, `training_pool` would accumulate real dated rows indefinitely, and the reserve mechanism would generalise into an ordinary rolling walk-forward split. The 2-cycle cap and the rare-cycle exposure are properties of *this dataset*, not of the method — and re-architecting the project's Phase 2-onward fixed `(dev, test)` split to retrofit true walk-forward onto a dataset that structurally cannot support it would be solving a problem the data has no room to pose.
    - **The champion alias flips on an *offline* gate alone — no shadow or canary stage validates the model on live traffic first, because there is no live traffic.** In a real deployment an offline gate earns a candidate a *trial*, not the throne: the standard sequence is **shadow** (the new model scores live requests in parallel, its outputs logged but never served, compared against the incumbent on real inputs) and/or **canary** (it serves a small traffic slice, watched on live metrics), and only then does the deployment pointer move. This project flips `champion` straight off the offline gate. The two-phase commit in `register.py` does *not* close this gap — it is a readiness probe (*does the artifact load and serve shaped-correctly?*), a different question from *are its predictions good on live traffic?*, which cannot be asked without a feed. **The sharper exposure is Phase 10, not Phase 7:** the cold-start flip has no incumbent to shadow against anyway, but the monthly retrain replaces a *serving* champion with a challenger on zero live validation — and there, shadow/canary is exactly the mechanism a production system would use and exactly what is absent. This is not synthesizable: faking the traffic fakes the evidence, and a canary whose green light you wired to always-green is worse than an absent one, for the same reason the drift stack cannot demonstrate *catching* something real. What is built instead is the **seam**: Phase 9's service resolves `champion` for serving but is structured so a `challenger` shadow/canary route plugs in without redesign, and Phase 10's promotion flow makes the alias flip a distinct final step behind a `require_traffic_validation` gate that no-ops on a static dataset. The honest claim is that the project demonstrates *offline* promotion machinery wired correctly and leaves an obvious home for the traffic-validation stage — not that it demonstrates traffic-validated promotion.

    - **The post-promotion bake-in window can route correctly, but can never watch anything real.** In a real deployment, a freshly-promoted champion is watched at heightened sensitivity for a defined window right after it starts serving — a genuine anomaly inside that window triggers an immediate revert to the previous champion, rather than waiting a full retrain cycle to catch up, since retraining might just reproduce the same mistake if the promotion itself was the fault. That routing distinction is real and tested here: a breach inside the bake-in window resolves to rollback, a breach in steady state resolves to a retrain trigger, and both paths share one `rollback_champion()` implementation rather than two rules that could quietly drift apart. What cannot exist is the anomaly to route on — with no live traffic, nothing is ever actually being watched during the window, so the mechanism is provably correct but never fires for a genuine reason. Faking an anomaly to demonstrate the auto-revert would be the same dishonesty as faking traffic for shadow/canary above: a green light wired to always-green proves nothing.

    What the project *does* demonstrate is that the machinery is correctly wired: the reference travels with the model version, the score baseline is out-of-sample, prevalence routes to re-calibration rather than retraining, and a rollback restores the matching baseline. Those are the properties that are hard to get right and easy to get wrong, and they are testable without a live feed. What it cannot demonstrate is the machinery *catching something real*. Any reading of this project that treats the monitoring stack as evidence of production drift detection is overclaiming; the correct claim is that the stack would work if a feed existed.

    **Phase 10c update (2026-08-29 — `PROJECT_PLAN.md` Phase 10c):** two of the sub-bullets above described stubs that Phase 10c partially activates. The `require_traffic_validation` gate is no longer a pure no-op — `canary_rollout.py` runs a staged ramp (`[0.01, 0.05, 0.25, 0.5, 1.0]`) against scripted traffic from `simulate_traffic.py`, and `outcomes.maturation_mode: compressed` produces a genuine online slice-vs-slice verdict inside the canary window (deliberately low-power — see item 20). The bake-in window's auto-revert is likewise exercised end to end: `guardrails.py` Tier-1 checks plus an induced fault (malformed rows from `simulate_traffic.py`) fire `rollback_champion` for real. The claims strengthen from *"leaves a home for the stage"* to *"runs the stage against scripted traffic"* — they still do not reach *"validated on organic live traffic"* or *"caught a real anomaly."* The seam is now built and run, not just left pluggable.

14. **Canary-routed and champion-routed customers compete for one shared contact-capacity pool, which can still bias what a canary measures even after eligibility is judged per-row.** `serving/contact_policy.py::select_contacts` judges each row's eligibility against whichever model actually served it (`serving/predict.py::ScoredBatch.served_threshold` — champion's `t*` for a champion-served row, challenger's own `t*` for a canary-routed one), rather than pinning every row to champion's threshold regardless of who scored it. That per-row judging is necessary — a pinned threshold would make a canary-routed customer's eligibility depend partly on an arbitrary hash-bucket assignment rather than on their risk under the model that actually scored them — but it is not sufficient. Downstream of eligibility, both cohorts still rank into *one* top-K cut against `configs/costs.yaml`'s `contact_capacity`/`campaign_budget`. If the challenger's threshold happens to be looser (admits more customers for the same cost inputs), canary-routed customers can crowd out champion-routed customers for the same limited slots — a textbook interference / SUTVA violation in constrained-allocation experimentation (the same problem ad-auction budget pacing and marketplace A/B tests contend with), and it would surface as an apparent champion capacity drop that has nothing to do with any individual customer's risk. The complete fix is to partition `contact_capacity`/`campaign_budget` into a dedicated slice per arm, sized to the canary fraction, rather than sharing one pool. Not implemented: this project's canary is a routing/logging mechanism demonstration (`docs/architecture.md`'s shadow/canary section), and — per item 13 above — there is no live traffic here for a capacity split to validate against anyway.

15. **Retired — canary bucketing is now salted per-challenger-version, so the same customer subset is no longer the canary population on every canary run.** Originally, `serving/predict.py::_canary_bucket_mask` bucketed purely on `sha256(customerid) % 10_000` — deliberately, so a given customer stays in the same bucket for the life of *one* canary rather than flapping call to call, but with no experiment identifier folded into the hash, that same reasoning had a side effect nothing corrected for: bucket assignment was identical across every canary this project would ever run, for every challenger, indefinitely — successive canary results would have been repeated measurements on the same ~fraction-sized subset rather than independent draws from the population. The fix: the hash key is now `sha256(f"{customerid}:{salt}")`, salted with the **challenger's own `model_version`** (`score_request`'s call site) rather than a manually-managed experiment id — no new config surface, and no operator discipline required, since the salt changes automatically the moment `challenger` moves to a different version, while staying fixed (preserving the original stability guarantee) for as long as the current version is under test. Verified in `tests/unit/test_predict.py`: the same customer set produces the same bucket assignment under a repeated salt, and a different salt reshuffles it.

16. **The expected-value ranking that decides who gets contacted always uses the champion's persisted cost scenario, even for challenger-scored rows, and nothing checks the two match.** `serving/app.py::predict_batch_endpoint` reads `scenario = bundle.scenarios[bundle.base_scenario_name]` once, from `runtime.champion`, and uses it to rank every row's expected value regardless of which model actually scored that row. Each model's `CostScenario` (arpu/ltv/cost/retention_rate) is not read live off `configs/costs.yaml` at request time — it is persisted at that *model's own* threshold-derivation cycle and loaded from that model's own MLflow run (`policy_config.py::load_threshold_payload`, docstring: "the threshold travels with the model version"), specifically so a rollback never leaves the API serving a threshold derived for a version it no longer runs. The side effect: if `costs.yaml` changes between the champion's and a later challenger's respective training cycles, their persisted "base" scenarios can genuinely hold different `arpu`/`ltv`/`cost`/`retention_rate` values, and challenger-routed rows would be ranked — and have their eligibility judged, since `t* = c/(r × LTV)` is derived from the same snapshot — against the champion's now-stale cost assumptions rather than the assumptions the challenger's own threshold was actually derived under. `register.py` already guards the analogous problem at promotion time with a `costs_config_hash` tag, precisely so a metric change can be attributed to the model or to the cost assumptions, never silently to both at once; no equivalent check exists at serving time between a live champion and a live challenger.

17. **`prediction_log` (Phase 10a-i) retains full customer PII in `feature_snapshot` indefinitely, with no retention or purge policy — a genuine production gap, not just a demonstration caveat like items 13–14 above.** Every prediction durably stores the exact feature row scored (gender, dependents, contract type, and the rest) with no expiry. A production system handling real customer data would enforce a retention window — commonly ~2 years for this kind of data — after which a scheduled job purges or anonymizes `feature_snapshot` while keeping the row's other columns (`prediction_id`, `model_version`, `probability`) intact for historical drift/performance metrics, which don't need the raw features to stay computable. Not built here: this project scores a static, publicly-available research dataset with no real customers behind it, so there is no compliance exposure to mitigate — but the gap is real and would need closing before this design could be pointed at actual customer data. Recorded as a deliberate, named simplification (`PROJECT_PLAN.md`'s Phase 10a-i), not an oversight.

### Progressive-Deployment Stack (Phase 10c)

*Deliberate simplifications introduced by the Phase 10c revision (2026-08-29) — the four-state registry lifecycle, canary ramp controller, guardrail auto-rollback, traffic simulator, and compressed label clock. Full design in `PROJECT_PLAN.md` Phase 10c and its Runtime Lifecycle Walkthrough.*

18. **`canary_rollout.py` is a scaled-down demonstration of progressive-delivery tooling.** ~150 lines of Prefect implementing the *shape* of a canary ramp — staged fractions, per-stage soak, guardrail-gated advance, automated Tier-1 rollback, human-escalated Tier-2 — where a production system would use Argo Rollouts / Flagger / Spinnaker with a service mesh for traffic splitting. The consistent-hash bucketing (`sha256(f"{customerid}:{challenger_version}") % 10_000`), the `canary_state` fraction poll, and the two guardrail tiers are real; absent are the battle-tested traffic-routing infrastructure and the operational maturity (metric backends, reusable analysis templates, automated rollback SLOs) those tools bring. A single-service project has no genuine consumer for a service mesh, so this is a scope choice, not a shortfall — but "we ran a canary" here means "we ran the canary control loop," not "we ran it on production-grade routing infrastructure."

19. **`simulate_traffic.py` resamples with replacement from the static 7,043-row `customers_crm`** (extends item 13). The arrival plumbing — Poisson schedule, `/predict` + `/predict/batch` routing, `prediction_log` variant tagging, champion/challenger dual scoring — is real and drives the shadow/canary metrics with genuine volume. What it is not is production traffic: the same customers recur, no new customers ever appear, and the canary A/B it feeds is deliberately low-power — a few hundred canary customers over the two available fold-forward cycles, wide CIs. It demonstrates the decision machinery, not model quality.

20. **Compressed-clock label maturation (`outcomes.maturation_mode: compressed`) is a simulation luxury.** It resolves `prediction_log` rows on an accelerated schedule (default 30 days → 1 hour) so the canary window can produce an online `churn_realized` verdict, letting the project demonstrate the fast-label *online-decides* regime alongside its native slow-label *offline-gate* regime. Two dependencies a real system does not have: an accelerated clock at all, and a clean untreated `churn_baseline` latent label kept alongside the treated `churn_realized` (real systems observe only the latter; training-label feedback contamination is its own unsolved problem — item 7). `churn_baseline` is structurally insulated from every offline surface (`performance_check.py`/`evaluate.py` gates, `training_pool` labels, the dummy-floor prevalence guardrail, `drift_reference.json` prevalence) so compression can never contaminate the promotion backtest — but `compressed`-mode verdicts carry the same low-power caveat as item 19 and are never claimed as model-quality evidence.

21. **`rollback_champion` (the recovery path) lacks verification the forward promotion path has — flagged, fixes scoped to Phase 10c.** `_flip_and_confirm` (forward) flips the alias, then loads through the alias and re-checks golden parity, reverting on failure. `rollback_champion` flips and returns with: no post-flip load/parity check on the reverted version; no environment-parity check against the container's installed dependencies (a rebuilt container can silently mismatch an older version's logged `numpy`/`scikit-learn`/`lightgbm`); and an auto-target chosen by `max(version number)` rather than `champion_history()` order — which can select a version that was promoted then immediately rolled back and never actually served. No flap guard rate-limits repeated automated rollbacks. The tag-based selection discipline itself is sound (never `N-1`, explicit target still validated for `promoted`, unset-and-page when no known-good exists). Fixes A–E specified in `PROJECT_PLAN.md` Phase 10c; until they land, an emergency rollback into a mismatched serving environment is a live risk.

22. **Orchestration notifications are Prefect-UI-only.** A suspended flow — human review, or a Tier-2 canary-guardrail escalation — surfaces as "waiting for input" in the Prefect dashboard with no external push (no Slack / email / PagerDuty / webhook). A human must know to check the UI rather than being paged. Acceptable for a self-hosted single-operator project; not a substitute for a real paging integration in an actual production deployment. Recorded in `PROJECT_PLAN.md` Phase 10b.

### Deployment Infrastructure (Phase 12a)

*The chosen deployment target and what it trades away — full design in `PROJECT_PLAN.md` Phase 12a, with the alternatives compared in `docs/architecture.md`.*

23. **v1 is deployed on a single EC2 `t3.micro` running docker-compose — a real production deployment, but not highly available.** The choice is driven by cost: App Runner and ECS-Fargate have no AWS free tier (~$9/mo per always-on task; the `api` + `ui` + a tracking server ≈ $20–27/mo), whereas EC2 `t3.micro` + RDS `db.t3.micro` are free-tier for 12 months, so the running cost is $0. A single VM with docker-compose is a legitimate production pattern for a service that does not need to scale — it is reachable, stays up without a laptop, has durable managed state (RDS), TLS, auth, CI/CD, and operational monitoring. What it trades away is a separate *resilience* axis, not "production vs. not": **(a)** single instance, no autoscaling — a traffic spike throttles the burstable CPU; **(b)** single point of failure, and free-tier RDS is single-AZ, so an instance reboot or AZ outage is downtime; **(c)** `docker compose up -d` briefly drops the container — no zero-downtime deploy; **(d)** the host OS is patched manually. Mitigations in place: containers `restart: unless-stopped`, a systemd unit brings the stack up on boot, RDS automated backups, a $1 billing alarm. The upgrade path is documented and cheap — the containers are identical, so Phase 12b's migration to App Runner / ECS-Fargate (autoscaling, multi-AZ, managed host) is a deploy-target swap, not a rebuild. This is a deliberate, documented cost/complexity trade for a portfolio project, not an oversight; a real production system with real customers and real traffic would start at the managed target.

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
7. **Leaner feature set via RFECV/embedded methods, if audit-surface or serving cost ever becomes a priority.** Already recorded in §4's "Tradeoff and future consideration" — not repeated here — since the current full-vs-reduced test can't isolate which 1-2 zero-signal features (`gender`, `phoneservice`) are individually safe to drop.

### Ongoing

8. **Deploy PSI monitoring** for score distribution drift. Trigger re-evaluation when PSI > 0.2.
9. **Recalibrate the threshold periodically** using updated OOF predictions as production data accumulates (Phase 10's retrain cycle).
10. **Replace illustrative business parameters** with Finance-validated figures before committing to P&L projections.

### Continuous-training & deployment follow-ups (from §9 items 18–22)

11. **Harden `rollback_champion` to the forward path's bar (§9 #21).** Add post-flip load/parity verification through the alias, an environment-parity check on the rollback target before the flip, a `champion_history()`-ordered auto-target, and a caller-side flap guard. Scoped as fixes A–E in `PROJECT_PLAN.md` Phase 10c; this is the highest-priority item here because an un-hardened emergency rollback can silently serve a broken or environment-mismatched model.
12. **Build a `recalibrate.py` flow genuinely distinct from `training_cycle.py` (§9 #12).** A prevalence shift should trigger recalibration, not a full retrain — but no lightweight flow exists yet that reuses `calibrate.py` against the frozen champion spec and re-derives the threshold's contact-rate figure without a `train.py` refit. Until it does, "recalibrate" and "retrain" resolve to the same expensive pipeline. Flagged for a Phase 10b design pass.
13. **If this design is ever pointed at real scale:** migrate the deployment from the single EC2 box to App Runner / ECS-Fargate for autoscaling and HA (§9 #23 — the Phase 12b target; a deploy-target swap, not a rebuild); replace `canary_rollout.py` with a progressive-delivery tool (Argo Rollouts / Flagger) on a service mesh (§9 #18); add an external paging integration for suspended-flow escalations (§9 #22); partition `contact_capacity`/`campaign_budget` per experiment arm so a canary cannot crowd the champion out of contact slots (§9 #14); and enforce a `feature_snapshot` retention/purge window (§9 #17).
14. **Keep the compressed-clock mode clearly fenced (§9 #20).** Its verdicts demonstrate the fast-label decision machinery only; never cite a `compressed`-mode A/B result as evidence of model quality, and keep `churn_baseline` insulated from every offline surface.
