# EDA Notes — Customer Personality Analysis (Regression track: Income)

Narrative/discussion companion to `regression.ipynb`. The notebook keeps only short section headers; the "why," the exact method, and the numeric results for each section live here. Section headers below match the notebook's markdown headers 1:1.

Target: **`Income`** (yearly household income, a raw column) — the team's chosen regression target, paired with `Response` as the natural classification target already present in this one CSV. See `CLASSIFY_NOTES.md` for the `Response` track — both notebooks share the same "Import Lib + Load Data," "Data Cleaning," "EDA," and "Distribution Check" foundation sections, duplicated so each notebook runs standalone. See `MATRIX_METHODS.md` for the math behind every correlation/redundancy matrix used below.

This is a **backup dataset** (see `README.md`), scoped lighter than `banking_dataset/`.

---

## Data Cleaning

**Purpose / Method.** Same `ID`/`Z_CostContact`/`Z_Revenue` drop, `Dt_Customer` → `Customer_Tenure_Days`, `Year_Birth` → `Age` (age-outlier drop), and `Marital_Status` collapse as `CLASSIFY_NOTES.md` (2,240 → 2,237 rows after the 3-row age-outlier drop) — see that file for the step-by-step. One extra, regression-specific step: **drop rows with missing `Income`**, since imputing the target itself would be the wrong move.

**Result.** 24 of the 2,237 cleaned rows have missing `Income` → **2,213 rows** remain for this notebook. `Response` is still present here (unaffected by the `Income` drop) purely to support the `Income` vs. `Response` deep dive below — it is not a regression feature.

---

## Regression EDA — Primary Target: Income

**Purpose.** Same reasoning as `banking_dataset/REGRESSION_NOTES.md`'s `balance` section and `google_play_data/REGRESSION_NOTES.md`'s `Rating` section: rank all candidate features against `Income` first, so only threshold-clearing features get a detailed follow-up.

**Method.** Pearson `|r|` for the 22 numeric candidates (`Income` obviously excluded from its own list; `AcceptedCmp1`-`5` included but flagged; `Response` deliberately excluded here and explored separately below, mirroring `google_play_data/regression.ipynb` keeping `hit` out of the main ranking), correlation ratio η for `Education`/`Marital_Status`. See `MATRIX_METHODS.md`.

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
| AcceptedCmp2 | 0.088 | numeric |
| NumDealsPurchases | 0.083 | numeric |
| Marital_Status | 0.046 | categorical |
| Complain | 0.025 | numeric |
| Teenhome | 0.019 | numeric |
| Customer_Tenure_Days | 0.018 | numeric |
| AcceptedCmp3 | 0.016 | numeric |
| Recency | 0.003 | numeric |

**This is by far the strongest bivariate signal set found across all three datasets explored for this project.** At threshold 0.1, **16 of 24** candidates clear it — more than double `google_play_data/Rating`'s 2-of-8 and well above `banking_dataset/balance`'s handful of borderline features. Several scores exceed 0.5, something neither prior regression track came close to.

**Signed direction (the table reports `|r|`; two features are notably negative):** `NumWebVisitsMonth` is **-0.553** — customers who browse the site more actually earn *less* and spend *less*, the reverse of the naive "more engagement = more valuable customer" assumption; plausibly because affluent customers shop through catalog/store channels instead of browsing the web repeatedly. `Kidhome` is **-0.428** — households with young children at home earn less, on average, than childless households in this sample. `NumDealsPurchases` is also negative (-0.083, doesn't clear threshold) — a weak echo of the same pattern (deal-seeking correlates with lower income).

**Caveat on `AcceptedCmp1`/`4`/`5`.** Same leakage-adjacent logic as `CLASSIFY_NOTES.md`: these are behavioral flags, not literally derived from `Income`, but plausibly confounded with it (higher-income customers may be targeted differently or respond differently to campaigns). Included, not excluded, but not treated as a "clean" economic driver of `Income` the way spend/purchase-channel columns are.

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

**Implication.** The same "affluent, high-spend, catalog/store-shopping" cluster identified in `CLASSIFY_NOTES.md` is, unsurprisingly, also the cluster most correlated with `Income` itself — this is exactly the circularity risk flagged in the Regression EDA section above, made concrete: `NumCatalogPurchases` (0.589 with `Income`) is itself 0.734-correlated with `MntMeatProducts` (0.584 with `Income`) and 0.634 with `MntWines` (0.578 with `Income`). A model using several of these features together would be leaning heavily on one underlying signal repeated across multiple columns, not three independent economic drivers — worth accounting for in feature selection (e.g. via regularization or picking one representative feature per cluster) rather than feeding all of them in raw.

---

## Deep Dive — Income vs Response

**Purpose.** `Response` is the other track's target, built from the same cleaned data — checking whether it relates to `Income` links the two tracks together and mirrors `google_play_data/REGRESSION_NOTES.md`'s "Rating vs. hit" deep dive (a numeric regression target against the classification target).

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

**Method.** Same `build_feature_matrix` approach as `classify.ipynb` (no NaN-fill actually needed here — every column is complete once the 24 missing-`Income` rows are dropped), fit `RandomForestRegressor(n_estimators=300, random_state=42)` on `Income`, importances aggregated back to parent columns.

**Result (top 12 of 24):**

| feature | importance |
|---|---|
| MntWines | 0.444 |
| MntMeatProducts | 0.146 |
| NumWebVisitsMonth | 0.077 |
| MntFruits | 0.075 |
| NumDealsPurchases | 0.060 |
| Recency | 0.028 |
| NumCatalogPurchases | 0.027 |
| MntSweetProducts | 0.025 |
| MntGoldProds | 0.017 |
| NumWebPurchases | 0.015 |
| Customer_Tenure_Days | 0.014 |
| MntFishProducts | 0.013 |

**RF concentrates even more sharply than the bivariate ranking suggested — but on a different feature than expected.** `MntWines` becomes the dominant RF feature at **0.444** (3x the runner-up), despite `MntMeatProducts` scoring marginally *higher* in the bivariate ranking (0.584 vs. `MntWines`'s 0.578) and in the MI cross-check below. `NumCatalogPurchases` — the #1 bivariate feature (0.589) — collapses to 7th in RF (0.027). The likely mechanism: `NumCatalogPurchases`, `MntMeatProducts`, and `MntWines` are all mutually correlated (0.57–0.73, see Multicollinearity above), so once RF's greedy tree-building process picks `MntWines` for early splits, the *marginal* information the other two add on top is much smaller than their standalone bivariate correlation implies — a textbook multicollinearity effect on RF importance, distinct from the leakage-driven RF reordering discussed in `CLASSIFY_NOTES.md`.

---

## Relevance Cross-Check — Mutual Information

**Purpose.** A third, non-parametric check on whether the RF's specific choice of `MntWines` over `MntMeatProducts` reflects a real relationship or an RF-specific (greedy, order-dependent) artifact.

**Method.** `sklearn.feature_selection.mutual_info_regression` — numeric columns with `discrete_features=False`, categorical columns (label-encoded) with `discrete_features=True`.

**Result (top 12 of 24):**

| feature | mutual_info | type |
|---|---|---|
| MntMeatProducts | 0.723 | numeric |
| MntWines | 0.679 | numeric |
| NumCatalogPurchases | 0.572 | numeric |
| NumStorePurchases | 0.549 | numeric |
| MntFruits | 0.459 | numeric |
| MntSweetProducts | 0.420 | numeric |
| MntFishProducts | 0.407 | numeric |
| NumWebVisitsMonth | 0.397 | numeric |
| NumWebPurchases | 0.348 | numeric |
| MntGoldProds | 0.269 | numeric |
| NumDealsPurchases | 0.255 | numeric |
| Kidhome | 0.217 | numeric |

**MI agrees with the bivariate ranking's ordering, not RF's.** `MntMeatProducts` (0.723) edges out `MntWines` (0.679) here, matching their bivariate order (0.584 vs. 0.578) — MI, computed independently per feature with no greedy "already explained" bookkeeping, doesn't produce the same lopsided 3x gap RF found. `NumCatalogPurchases` and `NumStorePurchases` also remain strong under MI (0.572, 0.549) despite RF demoting both — confirming the RF picture above is a genuine multicollinearity/greedy-splitting artifact specific to the tree-building process, not evidence that `NumCatalogPurchases` is actually a weak predictor of `Income` on its own. **Overall MI magnitudes here (up to 0.72) are an order of magnitude larger than `CLASSIFY_NOTES.md`'s (max 0.043)** — while MI scales aren't strictly comparable across a continuous vs. binary target, this is consistent with the Discussion above: `Income` genuinely has a much stronger, more recoverable signal in this feature set than `Response` does.
