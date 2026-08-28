# EDA Notes — Google Play Store Apps (Regression track: Rating)

Narrative/discussion companion to `regression.ipynb`. The notebook keeps only short section headers; the "why," the exact method, and the numeric results for each section live here. Section headers below match the notebook's markdown headers 1:1.

Target: **`Rating`** (average user rating, 1.0–5.0) is the team's chosen regression target. See `CLASSIFY_NOTES.md` for the `hit` classification track — both notebooks share the same "Import Lib + Load Data," "Data Cleaning," "EDA," and "Distribution Check" foundation sections, duplicated so each notebook runs standalone. See `MATRIX_METHODS.md` for the math behind every correlation/redundancy matrix used below.

This is a **backup dataset** (see `README.md`), scoped lighter than `banking_dataset/`.

---

## Data Cleaning

**Purpose / Method.** Same corrupted-row-drop, numeric-string parsing, and exact-duplicate-drop as `CLASSIFY_NOTES.md` (10,841 → 10,840 → 10,357 rows) — see that file for the step-by-step. One extra, regression-specific step: **drop rows with missing `Rating`**, since imputing the target itself would be the wrong move.

**Result.** 1,465 of the 10,357 cleaned rows have missing `Rating` (slightly fewer than the raw file's 1,474 missing-`Rating` count, since a handful of those rows were also exact duplicates removed in the shared dedup step) → **8,892 rows** remain for this notebook. `hit` is still computed here (from `Installs_num`, unaffected by the `Rating` drop) purely to support the `Rating` vs. `hit` deep dive below — it is not a regression feature.

---

## Regression EDA — Primary Target: Rating

**Purpose.** Same reasoning as `banking_dataset/REGRESSION_NOTES.md`'s `balance` section: rank all candidate features against `Rating` first, so only threshold-clearing features get a detailed follow-up plot.

**Method.** Pearson `|r|` for numeric features (`Reviews_num, Size_num, Installs_num, Price_num`), correlation ratio η for categorical features (`Category, Type, Content Rating, Genres`). See `MATRIX_METHODS.md` for η's mechanics and its small-sample-inflation caveat.

**Result.** Full ranking (relevance score, descending):

| feature | score | type |
|---|---|---|
| Genres | 0.200 | categorical |
| Category | 0.170 | categorical |
| Size_num | 0.082 | numeric |
| Reviews_num | 0.069 | numeric |
| Content Rating | 0.051 | categorical |
| Installs_num | 0.051 | numeric |
| Type | 0.038 | categorical |
| Price_num | -0.022 | numeric |

At threshold 0.1, only **`Genres` and `Category`** clear it — every numeric feature is near-zero, even weaker than `balance`'s weakest numeric correlations in `banking_dataset`. `Rating` itself is extremely tightly clustered (mean 4.19, std 0.52, IQR 4.0–4.5, see `Distribution Check`) — there's very little variance to explain in the first place, independent of what predicts it.

**Caveat on `Genres`.** Flagged explicitly in the notebook: `Genres` has 115 categories on ~8,900 rows (119 in the full classify subset — a handful of rare `Genres` values only appear on rows with missing `Rating`, dropped here), and η is a known-biased statistic on sparse categories (small groups inflate between-group variance). `Category` (33 categories, min group size 42 — see the group table below) is the more trustworthy categorical signal; `Genres`'s extra 0.03 over `Category` is plausibly small-group noise rather than real added signal, especially given the 0.969 `Genres`–`Category` Cramér's V found in `CLASSIFY_NOTES.md`'s redundancy deep dive (same underlying dimension at two resolutions).

**Numeric detail (Pearson `r`, signed):** `Size_num` 0.082, `Reviews_num` 0.069, `Installs_num` 0.051, `Price_num` -0.022 — all essentially flat scatter clouds (see the 2x2 scatter grid in the notebook).

**`Category` group means** (8 highest / 8 lowest by mean `Rating`, group sizes in parentheses):

| top 8 | mean | n | | bottom 8 | mean | n |
|---|---|---|---|---|---|---|
| EVENTS | 4.436 | 45 | | FINANCE | 4.127 | 317 |
| EDUCATION | 4.376 | 129 | | BUSINESS | 4.103 | 270 |
| ART_AND_DESIGN | 4.358 | 62 | | LIFESTYLE | 4.096 | 305 |
| BOOKS_AND_REFERENCE | 4.347 | 177 | | TRAVEL_AND_LOCAL | 4.094 | 205 |
| PERSONALIZATION | 4.334 | 310 | | VIDEO_PLAYERS | 4.064 | 160 |
| PARENTING | 4.300 | 50 | | MAPS_AND_NAVIGATION | 4.052 | 124 |
| GAME | 4.281 | 1074 | | TOOLS | 4.047 | 734 |
| BEAUTY | 4.279 | 42 | | DATING | 3.972 | 159 |

Spread is real but modest — roughly 0.46 stars between the best- and worst-rated categories (`EVENTS` 4.436 vs. `DATING` 3.972), on a target whose overall std is only 0.52. All groups shown have reasonable sample sizes (42+), so unlike `Genres`, this spread is not primarily a small-sample artifact.

---

### Discussion — Rating

This is the *same* "no dominant predictor" pattern already found and documented for `balance` in `banking_dataset/REGRESSION_NOTES.md` — arguably slightly worse on the numeric side here (every numeric `|r|` is below 0.1, versus `balance` having a couple of borderline numeric features), with `Category` playing the role `banking_dataset`'s `month` played: a real, trustworthy categorical exception in an otherwise weak-signal target. This was the exact finding, generalized across two independently-explored datasets, that motivated documenting the weak-numeric-signal pattern as a tested, well-evidenced characteristic of this kind of EDA rather than a dataset-specific flaw (see `banking_dataset/REGRESSION_NOTES.md`'s outlier-hypothesis-ruled-out discussion). Building this dataset out did not "solve" the concern that prompted exploring it — it relocated the same shape of finding to a different domain, which is itself useful evidence that the pattern is about the *kind* of target (a tightly-clustered, subjectively-driven rating/behavioral outcome) rather than an artifact of one specific dataset.

---

## Multicollinearity (numeric feature-to-feature)

**Purpose.** Same as `CLASSIFY_NOTES.md`'s version of this section — redundancy *among* features, independent of target relevance — but here `Installs_num` is included (it was excluded in the classify notebook only because it's the literal basis of `hit`; no such circularity applies to `Rating`), and `Rating` itself is included as the 5th column, matching `banking_dataset`'s convention of including the target in its own numeric-numeric matrix.

**Method.** Pearson correlation matrix over `Reviews_num, Size_num, Installs_num, Price_num, Rating`. See `MATRIX_METHODS.md`.

**Result.** One clearly strong pair: **`Reviews_num` vs. `Installs_num`, r = 0.633** — substantially higher than any pair seen in the classify notebook's matrix (max there was 0.238), and expected: review count mechanically scales with install count (the same relationship flagged as a `hit`-leakage risk in `CLASSIFY_NOTES.md`, showing up here as ordinary — and much less concerning — multicollinearity, since neither `Reviews_num` nor `Installs_num` is circular with `Rating`). Next largest: `Size_num` vs. `Installs_num` (r = 0.167), `Size_num` vs. `Reviews_num` (r = 0.240). Every pair involving `Rating` is weak (|r| ≤ 0.082) — consistent with the relevance ranking above. `Price_num` is essentially uncorrelated with everything (|r| ≤ 0.027).

---

## Deep Dive — Rating vs hit

**Purpose.** `hit` (installs ≥ 1,000,000) is the other track's engineered target, built from the same cleaned data — checking whether it relates to `Rating` links the two tracks together and mirrors `banking_dataset/REGRESSION_NOTES.md`'s "balance vs. y" deep dive (a numeric regression target against the classification target, using the same boxplot/violin/group-stats pattern).

**Method.** Boxplot and violin plot of `Rating` split by `hit`, plus group medians/means/counts.

**Result.**

| hit | median | mean | count |
|---|---|---|---|
| 0 (flop) | 4.2 | 4.113 | 4,840 |
| 1 (hit) | 4.3 | 4.277 | 4,052 |

A real but modest gap (0.16 on the mean, 0.1 on the median) — hit apps skew slightly higher-rated, consistent with `Rating`'s 0.156 point-biserial score against `hit` found in `CLASSIFY_NOTES.md`'s relevance ranking (the two numbers describe the same relationship from opposite directions — `Rating`-as-predictor-of-`hit` there, `hit`-as-grouping-variable-for-`Rating` here). Plausibly a virtuous-cycle relationship rather than a one-directional causal one: popular apps accumulate more reviews (which tend to regress toward a moderately positive mean at volume) and get more visibility/recommendation, while well-rated apps are more likely to be installed and recommended — a correlation, not evidence of which side is driving the other.

---

## Model-Based Feature Importance (RF)

**Purpose.** Same rationale as `CLASSIFY_NOTES.md` — a model-based, interaction-aware complement to the bivariate relevance ranking above.

**Method.** Same `build_feature_matrix` approach as the classify notebook (median-fill for numeric NaNs, `pd.get_dummies` for categoricals), fit `RandomForestRegressor(n_estimators=300, random_state=42)` on `Rating`, importances aggregated back to parent columns.

**Result.**

| feature | importance |
|---|---|
| Reviews_num | 0.317 |
| Size_num | 0.238 |
| Genres | 0.170 |
| Category | 0.122 |
| Installs_num | 0.098 |
| Content Rating | 0.027 |
| Price_num | 0.021 |
| Type | 0.006 |

As with the classify track, RF materially reorders the bivariate ranking: `Reviews_num` — a near-zero bivariate correlation with `Rating` (r = 0.069, bottom half of the table above) — becomes the single largest RF importance. This is the same "correlation misses non-linear structure" pattern found in `CLASSIFY_NOTES.md`, here applied to a continuous target: a Random Forest can exploit a threshold-like or non-monotonic relationship (e.g. apps need some minimum review volume before their rating stabilizes; apps with a handful of reviews can swing to either extreme) that a linear correlation coefficient is blind to. `Category`/`Genres` remain meaningfully important (as they were in the bivariate ranking, with `Genres`'s small-sample caveat still applying), but no longer dominate outright once `Reviews_num` and `Size_num` are allowed nonlinear/interaction credit.

---

## Relevance Cross-Check — Mutual Information

**Purpose.** A third, non-parametric check on whether the RF reordering above reflects a real relationship or an RF-specific artifact — same role as in `CLASSIFY_NOTES.md`.

**Method.** `sklearn.feature_selection.mutual_info_regression` — numeric columns (median-filled, reusing the RF feature matrix) with `discrete_features=False`, categorical columns (label-encoded) with `discrete_features=True`.

**Result.**

| feature | mutual_info | type |
|---|---|---|
| Reviews_num | 0.299 | numeric |
| Installs_num | 0.181 | numeric |
| Genres | 0.105 | categorical |
| Category | 0.062 | categorical |
| Size_num | 0.043 | numeric |
| Price_num | 0.009 | numeric |
| Content Rating | 0.002 | categorical |
| Type | 0.000 | categorical |

MI confirms the RF finding: `Reviews_num` and `Installs_num` — both near-zero under Pearson `r` (0.069 and 0.051) — are the two strongest features by mutual information, well ahead of `Category`/`Genres`. `Size_num`, RF's second-place feature, drops to 5th under MI, suggesting some of its RF importance may reflect interaction effects (which RF can exploit but a single-feature MI score can't fully capture) rather than a strong marginal relationship on its own. Combined with the Discussion above, the honest summary for this track: **no single feature — linear or non-linear — explains much of `Rating`'s variance**, `Reviews_num`/`Installs_num`/`Category` are the closest things to a signal, and `Rating` itself is simply a tightly-clustered target with limited variance to explain in the first place.
