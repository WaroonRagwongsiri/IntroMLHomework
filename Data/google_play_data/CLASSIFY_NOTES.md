# EDA Notes — Google Play Store Apps (Classification track: hit)

Narrative/discussion companion to `classify.ipynb`. The notebook keeps only short section headers; the "why," the exact method, and the numeric results for each section live here. Section headers below match the notebook's markdown headers 1:1.

Target: **`hit`** (engineered: `Installs_num >= 1,000,000`) is the team's chosen classification target — not a raw column, built in the Data Cleaning section from the parsed `Installs` string. See `REGRESSION_NOTES.md` for the `Rating` regression track — both notebooks share the same "Import Lib + Load Data," "Data Cleaning," "EDA," and "Distribution Check" foundation sections, duplicated so each notebook runs standalone. See `MATRIX_METHODS.md` for the math behind every correlation/redundancy matrix used below.

This is a **backup dataset** (see `README.md`), scoped lighter than `banking_dataset/` — no Interaction Effects or Time/Seasonality deep dives here (no time dimension in this data, and both were already judged cuttable-if-needed even in the primary project); the sections kept are the ones that carried real weight in `banking_dataset/classify.ipynb`.

---

## Data Cleaning

**Purpose.** Unlike `banking_dataset/` (no missing values, no corrupted rows), this dataset needed real cleaning before any modeling — worth documenting as its own section rather than folding silently into "Import Lib + Load Data."

**Method / Result, in the order applied:**
1. **Drop the corrupted row** — index `10472` ("Life Made WI-Fi Touchscreen Photo Frame"), missing `Category`, every field from `Rating` onward shifted one column left (`Rating` reads as an impossible `19.0`). Confirmed by direct inspection before dropping. Shape: 10,841 → 10,840 rows.
2. **Parse messy numeric-as-string columns**: `Installs` (`"10,000+"` → int `Installs_num`), `Size` (`"19M"`/`"201k"` → float MB `Size_num`, `"Varies with device"` → explicit `NaN`, kept, not imputed — 1,695 rows), `Price` (`"$4.99"` → float `Price_num`), `Reviews` (string digits → float `Reviews_num`).
3. **Drop exact duplicates.** 1,181 duplicate `App` names found, but left alone (mostly legitimate — the same app listed more than once, e.g. under multiple categories). 483 **exact full-row duplicates** dropped. Shape: 10,840 → 10,357 rows.
4. **Define `hit`**: `hit = (Installs_num >= 1,000,000).astype(int)`. Result: **39.1% hit / 60.9% flop** — a much gentler imbalance than `banking_dataset`'s `y` target (11.7%/88.3%), so no aggressive class-weighting/resampling machinery is strictly necessary here.

---

## Relevance vs. Classification Target (hit) — Correlation First

**Purpose.** Same reasoning as `banking_dataset/CLASSIFY_NOTES.md`: with 8 candidate features, a single relevance ranking upfront avoids a wall of uninformative univariate plots and lets only threshold-clearing features get a follow-up plot.

**Method.** Two measures depending on feature type (see `MATRIX_METHODS.md` for the math and this dataset's NaN-handling wrinkle):
- Numeric (`Reviews_num, Size_num, Price_num, Rating`): point-biserial correlation with `hit` (0/1), masked to non-missing rows per column.
- Categorical (`Category, Type, Content Rating, Genres`): Cramér's V from the raw-label contingency table.

**`Installs_num` is deliberately excluded** — it's the literal basis of `hit` (`hit = Installs_num >= 1e6`), so any correlation with it would be circular, not predictive signal. This is a data-definition issue, distinct from the leakage case below.

**Result.** Full ranking (relevance score, descending):

| feature | score | type |
|---|---|---|
| Genres | 0.360 | categorical |
| Category | 0.331 | categorical |
| Size_num | 0.318 | numeric |
| Type | 0.208 | categorical |
| Reviews_num | 0.187 | numeric |
| Rating | 0.156 | numeric |
| Content Rating | 0.137 | categorical |
| Price_num | 0.050 | numeric |

At threshold 0.1 (same Cohen's-1988-small-effect convention used throughout, see `banking_dataset/CLASSIFY_NOTES.md` for the full justification), `top_features = ['Genres', 'Category', 'Size_num', 'Type', 'Reviews_num', 'Rating', 'Content Rating']` — 7 of 8 candidates clear it; only `Price_num` doesn't.

**Leakage note — `Reviews_num`.** Review count scales with install count by construction: a review can only be written after a download, so more installs mechanically means more potential reviewers. Since `hit` is defined directly from `Installs_num`, `Reviews_num` inherits that relationship without being a genuine causal driver of `hit` — the same leakage mechanism as `duration` in `banking_dataset` (a feature that's mechanically downstream of the outcome, not a usable predictor for deciding in advance). The gap is stark: median `Reviews_num` is **76 for flops vs. 91,378 for hits** (mean 1,565 vs. 1,034,556) — over 1,000x, far starker than any legitimate predictor's spread in this ranking. Unlike `duration`, `Reviews_num` doesn't even top the bivariate ranking here (0.187, mid-table) — but it dominates once nonlinear/interaction structure is captured (see RF/MI below), which is itself informative: a purely linear/bivariate view can under-state a leakage risk as easily as it can overstate one.

**Domain interpretation.**
- `Genres`/`Category` — certain categories (`GAME`, `COMMUNICATION`, `TOOLS`, all high-volume in the distribution check) plausibly reach mass-market audiences by nature of the category itself, independent of any one app's quality. `Genres` (119 categories) is largely a finer-grained restatement of `Category` (33 categories) — see the redundancy deep dive below — so its higher raw score partly reflects small-sample inflation (`MATRIX_METHODS.md`), not twice the independent signal.
- `Size_num` — the most surprising strong numeric feature (0.318, beating `Reviews_num`). A plausible mechanism: larger apps tend to be more full-featured, more actively maintained, or built by bigger publishers with bigger marketing/distribution budgets — all things that correlate with install volume independent of app quality per se.
- `Type` (Free vs. Paid) — free apps have a near-zero barrier to install, plausibly inflating install counts regardless of quality; consistent with `Type` and `Category`'s Cramér's V of 0.188 (categories skew differently free/paid) rather than being fully independent.
- `Rating` — hit apps skew slightly higher-rated (see `REGRESSION_NOTES.md`'s `Rating` vs. `hit` deep dive: median 4.3 vs. 4.2, mean 4.277 vs. 4.113) — plausibly a virtuous cycle (popular apps accumulate more, and often better, reviews; well-rated apps get recommended more), though the direction of causality isn't resolved by a correlation alone.
- `Content Rating` — modest signal; `Everyone`-rated apps have the broadest addressable audience by definition, a weak version of the same mass-market mechanism behind `Category`.

---

## Multicollinearity (numeric feature-to-feature)

**Purpose.** Same as `banking_dataset`: a separate check for redundancy *among* features, independent of the relevance ranking above. (Identical section/result also appears in `REGRESSION_NOTES.md`, minus `Rating` swapped for `Installs_num` — target-agnostic, duplicated as shared foundation rather than cross-referenced.)

**Method.** Pearson correlation matrix over `Reviews_num, Size_num, Price_num, Rating`. See `MATRIX_METHODS.md`.

**Result.** All pairwise correlations are weak. Largest: **`Reviews_num` vs. `Size_num`, r = 0.238** — larger apps tend to have more reviews, plausibly because both are proxies for "established, actively-maintained app." Next: `Size_num` vs. `Rating` (r = 0.082), `Reviews_num` vs. `Rating` (r = 0.069). `Price_num` is essentially uncorrelated with everything else (|r| ≤ 0.024). Conclusion: numeric features are largely non-redundant with each other — no pair approaches Cohen's "medium" threshold (0.3).

---

## Deep Dive — Categorical-Categorical Redundancy (hit)

**Purpose.** The relevance ranking treats `Category`, `Type`, `Content Rating`, and `Genres` as independent signals; this checks whether they're actually redundant with each other, which matters for feature selection regardless of target.

**Method.** Pairwise Cramér's V across `Category, Type, Content Rating` as a 3x3 heatmap. `Genres` (119 categories) is excluded from the matrix — unreadable at that cardinality and expensive to compute pairwise — and instead checked individually against the other three. See `MATRIX_METHODS.md`.

**Result.**

3x3 matrix: `Category`–`Content Rating` **0.338**, `Category`–`Type` **0.188**, `Type`–`Content Rating` 0.048.

`Genres` vs. the others: **`Genres`–`Category` = 0.969** — a near-maximal association, confirming `Genres` is largely a finer-grained restatement of `Category` rather than an independent dimension (a game's `Genres` value is almost always a sub-type of `GAME`, etc.). `Genres`–`Content Rating` = 0.388, `Genres`–`Type` = 0.268 — both moderate, roughly tracking `Category`'s own redundancy with those columns, consistent with `Genres` mostly just inheriting `Category`'s associations at higher resolution.

**Implication.** `Genres`'s strong relevance score (0.360, table above) should not be read as a second, independent strong predictor stacked on top of `Category` (0.331) — the two are measuring almost the same underlying signal at different granularities, and `Genres`'s extra 119-vs-33-category resolution comes with the small-sample-inflation caveat from `MATRIX_METHODS.md`.

---

## Model-Based Feature Importance (RF)

**Purpose.** Same rationale as `banking_dataset`: the relevance ranking and multicollinearity checks are bivariate; a Random Forest captures interactions and non-linearities a single-feature correlation can miss, and doubles as a sanity check against the bivariate ranking.

**Method.** Built a feature matrix with `pd.get_dummies` for `Category, Type, Content Rating, Genres` plus the numeric columns `Reviews_num, Size_num, Price_num, Rating` (median-filled for `RandomForestClassifier`, which needs a complete matrix — a modeling-only convenience not applied elsewhere in the notebook, since NaN handling in `Size_num`/`Rating` is otherwise left explicit throughout). Fit `RandomForestClassifier(n_estimators=300, random_state=42)` on `hit`. Dummy columns summed back to their parent column via a dummy→parent map, same approach as `banking_dataset`.

**Result.**

| feature | importance |
|---|---|
| Reviews_num | 0.690 |
| Size_num | 0.106 |
| Genres | 0.057 |
| Rating | 0.052 |
| Category | 0.045 |
| Type | 0.024 |
| Content Rating | 0.013 |
| Price_num | 0.012 |

**This is a materially different picture from the bivariate ranking.** `Reviews_num` — mid-table in the point-biserial/Cramér's V ranking (0.187, 5th of 8) — dominates RF importance by a wide margin (0.690, more than 6x the next feature). This is the leakage mechanism from the relevance section showing up much more starkly once the model can exploit `Reviews_num`'s nonlinear, threshold-like relationship with `hit` (the median gap — 76 vs. 91,378 — is enormous but not strongly *linear*, which is why the bivariate point-biserial score understated it). `Category` and `Genres`, the bivariate leaders, both drop well down the RF ranking — consistent with a lot of their apparent bivariate signal actually being redundant with `Reviews_num` and `Size_num` (categories that skew toward mass installs also skew toward high review counts and larger app sizes).

---

## Relevance Cross-Check — Mutual Information

**Purpose.** A third, non-parametric lens on relevance (no linearity or normality assumption), to see whether the RF-based re-ranking of `Reviews_num` above is an RF-specific artifact or a more general pattern.

**Method.** `sklearn.feature_selection.mutual_info_classif` — numeric columns (median-filled, reusing the RF feature matrix) with `discrete_features=False`, categorical columns (label-encoded via `pd.factorize`) with `discrete_features=True`.

**Result.**

| feature | mutual_info | type |
|---|---|---|
| Reviews_num | 0.539 | numeric |
| Size_num | 0.094 | numeric |
| Rating | 0.074 | numeric |
| Genres | 0.070 | categorical |
| Category | 0.058 | categorical |
| Type | 0.029 | categorical |
| Price_num | 0.026 | numeric |
| Content Rating | 0.009 | categorical |

MI agrees with RF, not with the bivariate ranking: `Reviews_num` is the clear top feature by a wide margin, confirming the RF finding isn't a model-specific artifact — it's a real nonlinear relationship that both a tree ensemble and a non-parametric information measure pick up, but that a linear (point-biserial) correlation understates. `Size_num` is a consistent second-place feature across both RF and MI. `Genres`/`Category` remain solid but no longer dominant once nonlinear structure is accounted for — same interpretation as above: their bivariate strength is partly a proxy for `Reviews_num`/`Size_num`, not fully independent signal.
