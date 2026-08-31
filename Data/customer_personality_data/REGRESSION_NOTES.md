# EDA Notes — Customer Personality Analysis (Regression track: Income)

Narrative/discussion companion to `regression.ipynb`. The notebook keeps only short section headers; the "why," the exact method, and the numeric results for each section live here. Section headers below match the notebook's markdown headers 1:1.

Target: **`Income`** (yearly household income, a raw column) — the team's chosen regression target, paired with `Response` as the natural classification target already present in this one CSV. See `CLASSIFY_NOTES.md` for the `Response` track — both notebooks share the same "Import Lib + Load Data," "Data Cleaning," "EDA," and "Distribution Check" foundation sections, duplicated so each notebook runs standalone. See `MATRIX_METHODS.md` for the math behind every correlation/redundancy matrix used below.

This is a **backup dataset** (see `README.md`), scoped lighter than `banking_dataset/`.

---

## Data Cleaning

**Purpose / Method.** Same `ID`/`Z_CostContact`/`Z_Revenue` drop, `Dt_Customer` → `Customer_Tenure_Days`, `Year_Birth` → `Age` (age-outlier drop), and `Marital_Status` collapse as `CLASSIFY_NOTES.md` (2,240 → 2,237 rows after the 3-row age-outlier drop) — see that file for the step-by-step. One extra, regression-specific step: **drop rows with missing `Income`**, since imputing the target itself would be the wrong move.

**Result.** 24 of the 2,237 cleaned rows have missing `Income` → **2,213 rows** remain for this notebook. `Response` is unaffected by the `Income` drop — used both as a regression candidate below and in the "Income vs. Response" deep dive.

---

## Regression EDA — Primary Target: Income

**Purpose.** Same reasoning as `banking_dataset/REGRESSION_NOTES.md`'s `balance` section and `google_play_data/REGRESSION_NOTES.md`'s `Rating` section: rank all candidate features against `Income` first, so only threshold-clearing features get a detailed follow-up.

**Method.** Pearson `|r|` for 23 numeric candidates (`Income` obviously excluded from its own list; `AcceptedCmp1`-`5` included but flagged) plus `Response`, correlation ratio η for `Education`/`Marital_Status`. `Response` — the other track's target — is included as a candidate here rather than excluded by convention: score it like any other feature and let the ranking decide. See `MATRIX_METHODS.md`.

**Result.** Full ranking (relevance score, descending; signed direction noted separately below the table):

| feature | score | type |
|---|---|---|
| NumCatalogPurchases | 0.589 | numeric |
| MntMeatProducts | 0.584 | numeric |
| MntWines | 0.578 | numeric |
| NumWebVisitsMonth | 0.553 | numeric |
| NumStorePurchases | 0.530 | numeric |
| MntSweetProducts | 0.441 | numeric |
| MntFishProducts | 0.439 | numeric |
| MntFruits | 0.430 | numeric |
| Kidhome | 0.428 | numeric |
| NumWebPurchases | 0.388 | numeric |
| AcceptedCmp5 | 0.335 | numeric |
| MntGoldProds | 0.325 | numeric |
| AcceptedCmp1 | 0.277 | numeric |
| Education | 0.218 | categorical |
| AcceptedCmp4 | 0.185 | numeric |
| Age | 0.163 | numeric |
| Response | 0.133 | numeric |
| AcceptedCmp2 | 0.088 | numeric |
| NumDealsPurchases | 0.083 | numeric |
| Marital_Status | 0.046 | categorical |
| Complain | 0.025 | numeric |
| Teenhome | 0.019 | numeric |
| Customer_Tenure_Days | 0.018 | numeric |
| AcceptedCmp3 | 0.016 | numeric |
| Recency | 0.003 | numeric |

**This is by far the strongest bivariate signal set found across all three datasets explored for this project.** At threshold 0.1, **17 of 25** candidates clear it — more than double `google_play_data/Rating`'s 2-of-8 and well above `banking_dataset/balance`'s handful of borderline features. Several scores exceed 0.5, something neither prior regression track came close to. `Response` itself clears the threshold at 0.133 (identical to the point-biserial score `CLASSIFY_NOTES.md` reports for `Income` against `Response` — the same statistic, computed in the reverse direction) — but see the RF/MI results below for why it doesn't survive as a chosen feature once other candidates are modeled jointly.

**Signed direction (the table reports `|r|`; two features are notably negative):** `NumWebVisitsMonth` is **-0.553** — customers who browse the site more actually earn *less* and spend *less*, the reverse of the naive "more engagement = more valuable customer" assumption; plausibly because affluent customers shop through catalog/store channels instead of browsing the web repeatedly. `Kidhome` is **-0.428** — households with young children at home earn less, on average, than childless households in this sample. `NumDealsPurchases` is also negative (-0.083, doesn't clear threshold) — a weak echo of the same pattern (deal-seeking correlates with lower income).

**Caveat on `AcceptedCmp1`/`4`/`5`.** Not leakage, same conclusion as `CLASSIFY_NOTES.md`: these are behavioral flags, not literally derived from `Income`, but plausibly confounded with it (higher-income customers may be targeted differently or respond differently to campaigns). Included, not excluded, but not treated as a "clean" economic driver of `Income` the way spend/purchase-channel columns are.

**`Education` group means** (η = 0.218, the stronger of the two categorical scores):

| Education | mean | median | n |
|---|---|---|---|
| PhD | 56,088 | 55,185 | 480 |
| Master | 52,918 | 50,943 | 365 |
| Graduation | 52,720 | 52,029 | 1,116 |
| 2n Cycle | 47,625 | 46,805 | 198 |
| Basic | 20,306 | 20,744 | 54 |

`Basic` stands out sharply — mean income roughly **2.5x lower** than every other education level, all of which cluster within a ~9k band of each other (47.6k–56.1k). This one category is doing most of the work behind `Education`'s η score; the top four levels are only weakly separated by income.

---

### Discussion — Income

This is the **opposite pattern** from both prior regression tracks. `banking_dataset/REGRESSION_NOTES.md`'s `balance` and `google_play_data/REGRESSION_NOTES.md`'s `Rating` both landed on the same "no dominant predictor, weak numeric signal throughout" finding — every numeric `|r|` in both of those trees stayed below roughly 0.2, and the honest conclusion in both cases was that the target itself had limited variance a handful of features could explain. `Income` here breaks that pattern decisively: `NumCatalogPurchases`, `MntMeatProducts`, and `MntWines` all clear **0.5**, sixteen features clear the 0.1 threshold outright, and the Multicollinearity section below shows those top features aren't independent restatements of one signal but a genuinely rich, spend-and-channel-driven picture of what a household's income buys. This is not a coincidence of feature engineering — `Income` is a naturally continuous, economically-grounded quantity with plausible real linear drivers (people with more money spend more money, across every product category and every purchase channel), unlike `Rating` (a tightly-clustered, subjective 1–5 score) or `balance` (a bank-account snapshot with a lot of idiosyncratic, hard-to-explain variance). The takeaway generalizes the two prior datasets' finding rather than contradicting it: **weak-numeric-signal targets and strong-numeric-signal targets are both real, dataset- and target-dependent outcomes** — this project's three regression tracks now span both ends of that spectrum, which is itself a useful finding to report.

---

## Multicollinearity (numeric feature-to-feature)

**Purpose.** Same as `CLASSIFY_NOTES.md`'s version of this section — redundancy *among* features, independent of target relevance — but here `Income` is included as the 23rd column (it was excluded from `classify.ipynb`'s matrix only because it's a candidate feature there, not the target; here it's the target itself, included per the `banking_dataset`/`google_play_data` convention of folding the target into its own numeric-numeric matrix).

**Method.** Pearson correlation matrix over the 22 numeric candidates plus `Income`. See `MATRIX_METHODS.md`.

**Result.** Same dominant cluster as `CLASSIFY_NOTES.md`'s matrix (expected — it's nearly the same 2,213-vs-2,237-row data), with `Income` now an explicit member:

| pair | r |
|---|---|
| MntMeatProducts vs. NumCatalogPurchases | 0.734 |
| MntWines vs. NumStorePurchases | 0.640 |
| MntWines vs. NumCatalogPurchases | 0.634 |
| MntFruits vs. MntFishProducts | 0.593 |
| NumCatalogPurchases vs. Income | 0.589 |
| MntMeatProducts vs. Income | 0.584 |
| MntFishProducts vs. MntSweetProducts | 0.584 |
| MntWines vs. Income | 0.578 |
| MntWines vs. NumWebPurchases | 0.554 |
| NumWebVisitsMonth vs. Income | -0.553 |

**Implication.** The same "affluent, high-spend, catalog/store-shopping" cluster identified in `CLASSIFY_NOTES.md` is, unsurprisingly, also the cluster most correlated with `Income` itself — this is exactly the multicollinearity concern flagged in the Regression EDA section above, made concrete: `NumCatalogPurchases` (0.589 with `Income`) is itself 0.734-correlated with `MntMeatProducts` (0.584 with `Income`) and 0.634 with `MntWines` (0.578 with `Income`). A model using several of these features together would be leaning heavily on one underlying signal repeated across multiple columns, not three independent economic drivers — worth accounting for in feature selection (e.g. via regularization or picking one representative feature per cluster) rather than feeding all of them in raw.

---

## Deep Dive — Income vs Response

**Purpose.** `Response` is scored as a regular candidate above; this section is a qualitative companion view (boxplot/violin/group stats) of that same relationship, and mirrors `google_play_data/REGRESSION_NOTES.md`'s "Rating vs. hit" deep dive (a numeric regression target against the classification target).

**Method.** Boxplot and violin plot of `Income` split by `Response`, plus group medians/means/counts.

**Result.**

| Response | median | mean | count |
|---|---|---|---|
| 0 (no) | 50,150 | 50,824 | 1,880 |
| 1 (yes) | 64,090 | 60,210 | 333 |

Responders earn meaningfully more — **+13,940 on the median (+27.8%), +9,385 on the mean (+18.5%)**. This is the same relationship found from the opposite direction in `CLASSIFY_NOTES.md`'s relevance ranking (`Income`'s 0.133 point-biserial score against `Response`) — the two numbers describe the same underlying gap, `Income`-as-predictor-of-`Response` there, `Response`-as-grouping-variable-for-`Income` here. A plausible, non-circular economic story: higher-income customers have more disposable spend to respond to a marketing offer with, consistent with `Income`'s strong correlation with the spend columns found above.

---

## Categorical Redundancy (Education vs. Marital_Status)

**Purpose / Method.** Same shared foundation check as `CLASSIFY_NOTES.md`, duplicated so this notebook runs standalone. See `MATRIX_METHODS.md`.

**Result.** `Cramér's V(Education, Marital_Status) = 0.043` — essentially independent, same finding as the classify track.

---

## Model-Based Feature Importance (RF)

**Purpose.** Same rationale as `CLASSIFY_NOTES.md` — a model-based, interaction-aware complement to the bivariate relevance ranking above.

**Method.** Same `build_feature_matrix` approach as `classify.ipynb` (no NaN-fill actually needed here — every column is complete once the 24 missing-`Income` rows are dropped), fit `RandomForestRegressor(n_estimators=300, random_state=42)` on `Income` with all 25 candidates (including `Response`), importances aggregated back to parent columns.

**Result (top 12 of 25):**

| feature | importance |
|---|---|
| MntWines | 0.446 |
| MntMeatProducts | 0.145 |
| NumWebVisitsMonth | 0.079 |
| MntFruits | 0.075 |
| NumDealsPurchases | 0.059 |
| NumCatalogPurchases | 0.030 |
| MntSweetProducts | 0.026 |
| Recency | 0.024 |
| MntGoldProds | 0.018 |
| Age | 0.015 |
| Customer_Tenure_Days | 0.014 |
| NumWebPurchases | 0.014 |

**RF concentrates even more sharply than the bivariate ranking suggested — but on a different feature than expected.** `MntWines` becomes the dominant RF feature at **0.446** (3x the runner-up), despite `MntMeatProducts` scoring marginally *higher* in the bivariate ranking (0.584 vs. `MntWines`'s 0.578) and in the MI cross-check below. `NumCatalogPurchases` — the #1 bivariate feature (0.589) — collapses to 6th in RF (0.030). The likely mechanism: `NumCatalogPurchases`, `MntMeatProducts`, and `MntWines` are all mutually correlated (0.57–0.73, see Multicollinearity above), so once RF's greedy tree-building process picks `MntWines` for early splits, the *marginal* information the other two add on top is much smaller than their standalone bivariate correlation implies — a textbook multicollinearity effect on RF importance, the same kind of RF-demotes-a-strong-predictor pattern discussed for `AcceptedCmp*` in `CLASSIFY_NOTES.md` (neither case is leakage). `MntFishProducts` (13th, 0.012) is squeezed just out of the top 12 by `Response`'s addition shifting the tail slightly — not a meaningful change in its own right.

**`Response`, having cleared the bivariate threshold, all but disappears here: importance 0.0005 (22nd of 25).** Its 0.133 bivariate correlation with `Income` turns out to be almost entirely redundant with the spend/purchase-channel cluster already in the model — once those are available, `Response` adds essentially nothing further. This is the concrete evidence for why it isn't a chosen feature: not a low individual score, but a marginal contribution near zero once modeled jointly with everything else.

---

## Relevance Cross-Check — Mutual Information

**Purpose.** A third, non-parametric check on whether the RF's specific choice of `MntWines` over `MntMeatProducts` reflects a real relationship or an RF-specific (greedy, order-dependent) artifact.

**Method.** `sklearn.feature_selection.mutual_info_regression` — numeric columns (including `Response`) with `discrete_features=False`, categorical columns (label-encoded) with `discrete_features=True`.

**Result (top 12 of 25):**

| feature | mutual_info | type |
|---|---|---|
| MntMeatProducts | 0.719 | numeric |
| MntWines | 0.681 | numeric |
| NumCatalogPurchases | 0.573 | numeric |
| NumStorePurchases | 0.550 | numeric |
| MntFruits | 0.453 | numeric |
| MntSweetProducts | 0.424 | numeric |
| MntFishProducts | 0.408 | numeric |
| NumWebVisitsMonth | 0.399 | numeric |
| NumWebPurchases | 0.349 | numeric |
| MntGoldProds | 0.270 | numeric |
| NumDealsPurchases | 0.254 | numeric |
| Kidhome | 0.217 | numeric |

**MI agrees with the bivariate ranking's ordering, not RF's — and its top-12 membership/order is unchanged by adding `Response` to the candidate pool** (only third-decimal shifts, since one more column in the matrix barely perturbs the KNN-based estimator). `MntMeatProducts` (0.719) edges out `MntWines` (0.681) here, matching their bivariate order (0.584 vs. 0.578) — MI, computed independently per feature with no greedy "already explained" bookkeeping, doesn't produce the same lopsided 3x gap RF found. `NumCatalogPurchases` and `NumStorePurchases` also remain strong under MI (0.573, 0.550) despite RF demoting both — confirming the RF picture above is a genuine multicollinearity/greedy-splitting artifact specific to the tree-building process, not evidence that `NumCatalogPurchases` is actually a weak predictor of `Income` on its own. **Overall MI magnitudes here (up to 0.72) are an order of magnitude larger than `CLASSIFY_NOTES.md`'s (max 0.043)** — while MI scales aren't strictly comparable across a continuous vs. binary target, this is consistent with the Discussion above: `Income` genuinely has a much stronger, more recoverable signal in this feature set than `Response` does.

**`Response` scores 0.038 here (21st of 25)** — same story as RF: a real bivariate correlation that turns out to carry almost no information about `Income` once the spend/purchase-channel features already explain the underlying "engaged, affluent customer" signal it's a weaker proxy for. Bivariate, RF, and MI together give `Response` a clean verdict: **1 of 3 methods clears threshold, and the two model-based methods that see the full feature set both rank it near the bottom** — it does not belong in the chosen-feature set, and this is now a checked result, not a convention.

---

## Feature Selection — Combining the Three Methods (Final Cut)

**Purpose.** Same rationale as `CLASSIFY_NOTES.md`'s version of this section, duplicated so this notebook runs standalone. The three methods above measure different things — linear association (bivariate), tree-based interaction-aware importance (RF), and general statistical dependency (MI) — and each can be misled in its own way. This section states the actual rule used to collapse three rankings into one final feature-selection decision, and applies it to every one of the 25 candidates (24 original + `Response`).

**Method — the cutting rule, not an average.** The three scores live on incompatible scales (signed −1..+1 for bivariate; unsigned 0..1 summing to 1 for RF; unsigned 0..∞ for MI), so they aren't blended into one composite number. Each method instead casts an independent pass/fail vote:

- **Bivariate passes** if `|score| ≥ 0.10`.
- **RF passes** if the feature lands in RF's top 12 of 25.
- **MI passes** if the feature lands in MI's top 12 of 25.

Tiered by vote count: **3/3** → chosen, no caveat (unless separately flagged for correlated/historical signal — see below); **2/3** → generally cut — promoted to chosen only when there's a specific, named, independently-confirmed reason (see `NumStorePurchases`/`Kidhome` below); **1/3 or 0/3** → cut.

**Result — every candidate, all three votes:**

| feature | bivariate | RF (top 12?) | MI (top 12?) | votes | flagged? |
|---|---|---|---|---|---|
| NumCatalogPurchases | 0.589 ✓ | 0.030 ✓ (6th) | 0.573 ✓ (3rd) | **3/3** | — |
| MntMeatProducts | 0.584 ✓ | 0.145 ✓ (2nd) | 0.719 ✓ (1st) | **3/3** | correlated w/ Income |
| MntWines | 0.578 ✓ | 0.446 ✓ (1st) | 0.681 ✓ (2nd) | **3/3** | correlated w/ Income |
| NumWebVisitsMonth | −0.553 ✓ | 0.079 ✓ (3rd) | 0.399 ✓ (8th) | **3/3** | — |
| MntSweetProducts | 0.441 ✓ | 0.026 ✓ (7th) | 0.424 ✓ (6th) | **3/3** | correlated w/ Income |
| MntFruits | 0.430 ✓ | 0.075 ✓ (4th) | 0.453 ✓ (5th) | **3/3** | correlated w/ Income |
| NumWebPurchases | 0.388 ✓ | 0.014 ✓ (12th) | 0.349 ✓ (9th) | **3/3** | — |
| MntGoldProds | 0.325 ✓ | 0.018 ✓ (9th) | 0.270 ✓ (10th) | **3/3** | correlated w/ Income |
| NumStorePurchases | 0.530 ✓ | 0.008 ✗ (15th) | 0.550 ✓ (4th) | 2/3 | — |
| MntFishProducts | 0.439 ✓ | 0.012 ✗ (13th) | 0.408 ✓ (7th) | 2/3 | correlated w/ Income |
| Kidhome | −0.428 ✓ | 0.004 ✗ (19th) | 0.217 ✓ (12th) | 2/3 | — |
| Age | 0.163 ✓ | 0.015 ✓ (10th) | 0.130 ✗ (14th) | 2/3 | — |
| NumDealsPurchases | −0.083 ✗ | 0.059 ✓ (5th) | 0.254 ✓ (11th) | 2/3 | — |
| AcceptedCmp5 | 0.335 ✓ | 0.009 ✗ (14th) | 0.121 ✗ (16th) | 1/3 | prior-campaign history |
| AcceptedCmp1 | 0.277 ✓ | 0.001 ✗ (21st) | 0.074 ✗ (19th) | 1/3 | prior-campaign history |
| Education (cat) | 0.218 ✓ | 0.004 ✗ (18th) | 0.128 ✗ (15th) | 1/3 | — |
| AcceptedCmp4 | 0.185 ✓ | 0.0003 ✗ (23rd) | 0.037 ✗ (22nd) | 1/3 | prior-campaign history |
| Response | 0.133 ✓ | 0.0005 ✗ (22nd) | 0.038 ✗ (21st) | 1/3 | — |
| Customer_Tenure_Days | 0.018 ✗ | 0.014 ✓ (11th) | 0.105 ✗ (17th) | 1/3 | — |
| Recency | 0.003 ✗ | 0.024 ✓ (8th) | 0.100 ✗ (18th) | 1/3 | — |
| AcceptedCmp2 | 0.088 ✗ | 0.0001 ✗ (24th) | 0.010 ✗ (24th) | 0/3 | prior-campaign history |
| Marital_Status (cat) | 0.046 ✗ | 0.007 ✗ (17th) | 0.073 ✗ (20th) | 0/3 | — |
| Complain | 0.025 ✗ | 0.00007 ✗ (25th) | 0.004 ✗ (25th) | 0/3 | — |
| Teenhome | 0.019 ✗ | 0.008 ✗ (16th) | 0.176 ✗ (13th) | 0/3 | — |
| AcceptedCmp3 | 0.016 ✗ | 0.001 ✗ (20th) | 0.020 ✗ (23rd) | 0/3 | prior-campaign history |

**Final tally: 8 at 3/3, 5 at 2/3, 7 at 1/3, 5 at 0/3.** ("Flagged" here means not leakage — either correlated/non-independent with `Income` (`Mnt*`) or a prior-campaign-history feature (`AcceptedCmp*`); see the "Not Actually Leakage" discussion above.)

**The 3/3 tier splits into two groups:**
- **3 chosen features, no caveat:** `NumCatalogPurchases`, `NumWebVisitsMonth`, `NumWebPurchases`.
- **5 correlated-signal features, also full consensus:** `MntMeatProducts`, `MntWines`, `MntSweetProducts`, `MntFruits`, `MntGoldProds` — 5 of the 6 `Mnt*` spend columns (the 6th, `MntFishProducts`, falls to 2/3 — see below). Not leakage: the multicollinearity caveat from the Relevance section above still applies — spend scales with income almost by construction, a real correlation-direction ambiguity, but not a future-info or target-derived problem.

**Notable 2/3 cases:**
- `NumStorePurchases`, `MntFishProducts`, and `Kidhome` all **fail only RF** despite clearing bivariate and MI comfortably — the same greedy-splitting/multicollinearity artifact already documented for `NumCatalogPurchases`'s own RF collapse (7th → now 6th with `Response` added): RF spends its early splits on the dominant `MntWines`/`MntMeatProducts` pair and has little marginal signal left to attribute elsewhere, not evidence these three are weak.
- `Age` **fails only MI**, `NumDealsPurchases` **fails only bivariate** (−0.083, just short of threshold) yet clears both RF and MI — a case of real non-linear signal a purely linear correlation check misses.
- `Response` sits at exactly 1/3 (bivariate only) — see the dedicated discussion above for why that's not enough to be chosen despite the real underlying relationship.
