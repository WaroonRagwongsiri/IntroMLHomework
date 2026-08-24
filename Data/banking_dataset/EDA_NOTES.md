# EDA Notes — Bank Marketing Dataset

Narrative/discussion companion to `banking.ipynb`. The notebook keeps only short section headers; the "why," the exact method, and the numeric results for each section live here. Section headers below match the notebook's markdown headers 1:1.

Target decisions: **`y`** (subscribed to term deposit — classification) and **`balance`** (account balance — regression) are the team's chosen primary targets.

---

## Relevance vs. Classification Target (y) — Correlation First

**Purpose.** With 16 candidate features, plotting every one against `y` (as in the univariate pass) would produce a wall of plots, most uninformative. Instead we compute a single relevance ranking across all features first, so only features that clear a threshold get a detailed follow-up plot.

**Method.** Two measures depending on feature type, both landing on a comparable 0–1 scale, so they can be ranked together:
- Numeric (`age, balance, day, duration, campaign, pdays, previous`): point-biserial correlation with `y` (encoded 0/1 — only the target is mapped to numbers). Point-biserial is mathematically Pearson correlation for a continuous-vs-binary pair.
- Categorical (`job, marital, education, default, housing, loan, contact, month, poutcome`): Cramér's V from the raw-label contingency table (`pd.crosstab` → chi-square). No ordinal/one-hot encoding, so category count/order can't bias the score.

Leakage call-out: `duration` (call length) is only known *after* a call ends, so it cannot be used as a model input for deciding who to call, regardless of its ranking.

**Result.** Full ranking (relevance score, descending):

| feature | score | type |
|---|---|---|
| duration | 0.395 | numeric |
| poutcome | 0.312 | categorical |
| month | 0.260 | categorical |
| contact | 0.151 | categorical |
| housing | 0.139 | categorical |
| job | 0.136 | categorical |
| pdays | 0.104 | numeric |
| previous | 0.093 | numeric |
| campaign | 0.073 | numeric |
| education | 0.073 | categorical |
| loan | 0.068 | categorical |
| marital | 0.066 | categorical |
| balance | 0.053 | numeric |
| day | 0.028 | numeric |
| age | 0.025 | numeric |
| default | 0.022 | categorical |

At threshold 0.1, `top_features = ['duration', 'poutcome', 'month', 'contact', 'housing', 'job', 'pdays']` — these are the only features that got a follow-up boxplot/stacked-bar plot in the notebook.

---

## Multicollinearity (numeric feature-to-feature)

**Purpose.** The relevance section answers "which features relate to `y`"; this answers a different question — "which features overlap with *each other*." Two features can each look weak alone yet be correlated with each other (redundant signal), which matters for model feature selection independent of target relevance. Kept as its own heatmap over the numeric columns only, to avoid conflating the two questions.

**Method.** Pearson correlation matrix (`train[numeric_cols].corr()`) over `age, balance, day, duration, campaign, pdays, previous`, rendered as a heatmap.

**Result.** All pairwise correlations are weak (|r| < 0.1) except one clear pair: **`pdays` vs `previous`, r = 0.455** — expected, since both describe prior-campaign contact history (days since last contact vs. number of prior contacts). Next largest: `day` vs `campaign` (r = 0.162), `campaign` vs `pdays` (r = -0.089), `campaign` vs `duration` (r = -0.085). No other pair exceeds 0.1. Conclusion: numeric features are largely non-redundant with each other, aside from the expected `pdays`/`previous` link.

---

## Regression EDA — Primary Target: balance

**Purpose.** `balance` is the team's chosen regression target: predicting account balance is a real forecasting problem with clear business meaning, unlike `duration`/`campaign`, which are call-logistics metadata. Reuses the "correlate first, then plot only what matters" approach from the classification section for consistency.

**Method.** Pearson correlation of `balance` against the other 6 numeric columns, plus scatterplots for each; boxplots of `balance` by 5 categorical columns (`job, education, marital, housing, loan`), medians used to order the categories and outliers hidden for readability.

**Result.** Correlations with `balance` (Pearson r): `age` 0.098, `duration` 0.022, `previous` 0.017, `campaign` -0.015, `day` 0.005, `pdays` 0.003 — all weak, `age` is the strongest but still marginal. Median `balance` by group:
- `job`: retired 787, unknown 677, management 572, unemployed 529, self-employed 526, student 502, technician 421, housemaid 406, admin. 396, blue-collar 388, entrepreneur 352, services 340 (retired vs. services is a ~2.3x spread).
- `education`: tertiary 577, unknown 568, primary 403, secondary 392.
- `marital`: married 477, single 437, divorced 348.
- `housing`: no 507 vs. yes 412 (no-mortgage clients hold higher balances).
- `loan`: no 496 vs. yes 258 (clients with a personal loan hold roughly half the balance).

No single numeric feature is strongly predictive; categorical group differences (especially `job`, `loan`, `housing`) are more informative than any numeric correlation.

---

### Discussion — balance

**Purpose.** Consolidate the modeling implications of the balance-vs-numeric and balance-vs-categorical results above before moving to the secondary regression pass.

**Method.** N/A — synthesis of the results above plus the univariate `balance` histogram/boxplot from the earlier distribution-check section.

**Result.** No single numeric feature is strongly predictive on its own (max |r| = 0.098, `age`); a regression model will likely need to combine several weak numeric signals plus the categorical breakdowns (`housing`, `loan`, `job`, `education`) rather than lean on one dominant driver. `balance` itself is heavily right-skewed with a long tail of large positive values (max 102,127 in `train`) and a meaningful chunk of negative balances/overdrafts (min -8,019 in `train`; see the univariate histogram/boxplot earlier in the notebook, which flagged 4,729 IQR-outlier rows, 10.46% of data). Because of the negative values, a plain `log(balance)` transform won't work directly; a **signed-log** transform (`sign(x) * log1p(|x|)`) or a robust-scaling approach should be considered at modeling time.

---

## Regression EDA — Secondary Targets: duration, campaign

**Purpose.** The assignment lists `duration` and `campaign` as regression candidates alongside `balance`, so evidence is kept for each in case the target choice changes — but since `balance` is the stated primary target, this pass is intentionally lighter (correlations + a few key categorical breakdowns, not a full loop over every feature).

**Method.** Pearson correlation of each target against the other numeric columns; boxplots of `duration` by `job` and `month`, and `campaign` by `poutcome` and `contact` (outliers hidden).

**Result.**
- `duration` vs. other numeric (Pearson r): `campaign` -0.085, `day` -0.030, `balance` 0.022, `age` -0.005, `pdays` -0.002, `previous` 0.001 — all weak.
- `campaign` vs. other numeric (Pearson r): `day` 0.162, `pdays` -0.089, `duration` -0.085, `previous` -0.033, `balance` -0.015, `age` 0.005 — also all weak; `day` is the strongest link (more contacts tend to cluster on certain days of month).
- `campaign` is a discrete count (min = 1), right-skewed — a count-model approach (Poisson/negative-binomial-style) would be more appropriate than plain linear regression if it were the modeling target.

---

## Deep Dive — Time/Seasonality (y)

**Purpose.** `train.csv` is date-ordered (May 2008 – Nov 2010) but the `month` column has no year, so the per-month countplot from the univariate pass collapses all 3 years together and can't show drift over the campaign's actual timeline. Binning by row order (a chronological proxy, since the data is sorted by contact date) lets us see whether `y`-rate and call volume changed over the life of the campaign — directly relevant to whether a time-based train/validation split is appropriate.

**Method.** Row index split into `N_CHUNKS = 20` equal-sized sequential bins (`pd.cut` on `np.arange(len(train))`), then per-chunk `y`-rate (share of `yes`) and call volume (`n_calls`), plotted as two stacked line/bar charts.

**Result.** Clear upward drift, not flat: `y`-rate starts around 2–5% in the earliest chunks (chunk 0: 2.1%, chunk 1: 3.8%, chunk 4: 4.6%) and rises steadily through the middle chunks (chunk 8: 6.0%, chunk 12: 4.7%) before jumping sharply in the final chunks — chunk 13: 16.2%, chunk 17: 23.8%, chunk 18: 41.2%, chunk 19 (final, most recent): **52.9%**. Call volume per chunk is roughly even (~2,260 rows each, by construction of the equal-size bins). This is a ~25x increase in conversion rate from the earliest to latest chunk of the campaign — strong evidence of real temporal drift (later-stage campaign targeting/conditions were far more effective), which supports doing a time-based train/validation split rather than a random split, and suggests any model should account for a temporal signal even without an explicit year column.

---

## Deep Dive — Categorical-Categorical Redundancy (y)

**Purpose.** The multicollinearity section above only covers numeric-numeric redundancy. Categorical features can be just as redundant with each other (e.g. `contact` method and `month` might move together for operational reasons), and this matters for feature selection and for interpreting the relevance ranking. Also directly tests the suspected `poutcome`/`pdays` redundancy noted in the univariate pass (both driven by "was this client ever contacted before"), across two different dtypes (categorical `poutcome` vs. numeric `pdays`), which a same-dtype Cramér's V matrix can't capture on its own.

**Method.** Pairwise Cramér's V (reusing the `cramers_v` helper) across `job, marital, education, default, housing, loan, contact, month, poutcome`, rendered as a 9x9 heatmap. Separately, an explicit crosstab of `poutcome` vs. `train['pdays'] == -1` (boolean "never contacted" flag) to quantify the cross-dtype redundancy directly.

**Result.** Cramér's V matrix — highest off-diagonal pairs: `contact`–`month` **0.512**, `housing`–`month` **0.504**, `job`–`education` **0.458**; next tier: `month`–`poutcome` 0.214, `contact`–`poutcome` 0.207, `housing`–`contact` 0.214, `loan`–`month` 0.183. Most other pairs are below 0.15. The `poutcome` vs. `pdays == -1` crosstab shows near-total separation:

| poutcome | pdays != -1 (previously contacted) | pdays == -1 (never contacted) |
|---|---|---|
| failure | 4,901 | 0 |
| other | 1,840 | 0 |
| success | 1,511 | 0 |
| unknown | 5 | 36,954 |

Every row with `poutcome != 'unknown'` has `pdays != -1` (all 8,252 such rows), and 36,954/36,959 "never contacted" rows have `poutcome == 'unknown'` — confirming these two columns encode almost the same "was this client ever contacted before" information across different dtypes, even though it doesn't show up in a same-dtype-only correlation view. `poutcome`'s categorical detail (failure/other/success) is only meaningful for the ~18% of clients who were previously contacted; for the rest, it's a near-duplicate of `pdays == -1`.

---

## Deep Dive — Interaction Effects (y)

**Purpose.** The relevance ranking treats each feature independently, but two features can jointly predict `y` in a way neither does alone. Checking a couple of pairs from the `top_features` ranking — `contact`×`month` and `poutcome`×`previous` (binned) — surfaces whether specific combinations are unusually strong or weak, information that would be invisible in single-feature plots.

**Method.** `y`-rate heatmaps: `pd.crosstab(..., values=y_binary, aggfunc="mean")` for (1) `contact` × `month` (columns ordered `month_order`) and (2) `poutcome` × `previous` binned into `["0", "1-2", "3-5", "6+"]` via `pd.cut(train["previous"], bins=[-1,0,2,5,max])`.

**Result.**
`contact` × `month` y-rate (selected cells): `cellular`/mar 0.532, `cellular`/sep 0.517, `cellular`/dec 0.494, `cellular`/oct 0.447, `cellular`/jun 0.439 vs. `cellular`/may (the highest-volume month) only 0.119 and `cellular`/jul 0.096 — the low-volume months (mar, sep, oct, dec) convert far better than the high-volume months (may, jul, aug ~0.10-0.12), for cellular contact. `unknown` contact in apr shows 0.833 but on very small n; `unknown` contact in jun/jul/may is near-zero (0.033-0.045), suggesting `unknown` contact method combined with peak-campaign months performs worst.

`poutcome` × `previous`(binned) y-rate: `success` dominates regardless of contact count — 0.634 (1-2 prior contacts), 0.659 (3-5), 0.673 (6+) — roughly **6x** the rate of `failure` (0.116–0.150) or `other` (0.161–0.183) at the same `previous` bin. `poutcome == unknown` with 0 previous contacts sits at 0.092 (close to the base rate), confirming `poutcome` (specifically "success") is a strong, count-independent signal, while `failure`/`other` only mildly improve with more prior attempts.

---

## Deep Dive — balance vs. y

**Purpose.** Checks whether the regression target (`balance`) and the classification target (`y`) are linked — i.e. do converters have a meaningfully different balance profile? This informs whether `balance` could double as a classification feature and whether `y` should be considered as a segmenting variable for balance modeling.

**Method.** Boxplot and violinplot of `balance` by `y` (reusing `plot_numeric_vs_target`), plus median/mean `balance` per `y` group.

**Result.**

| y | median | mean | count |
|---|---|---|---|
| no | 417.0 | 1,303.7 | 39,922 |
| yes | 733.0 | 1,804.3 | 5,289 |

Converters (`y == yes`) have a **~76% higher median balance** (733 vs. 417) and ~38% higher mean balance (1,804 vs. 1,304) than non-converters. The gap is real but modest relative to the overall spread of `balance` (std ≈ 3,045 in the full dataset) — visible in the boxplot/violin as a shifted but heavily overlapping distribution, not a clean separation. `balance` alone would be a weak classifier feature, consistent with its low relevance score (0.053) from the ranking section, but it does carry some signal in combination with other features.

---

## Deep Dive — Age-Binned balance Trend

**Purpose.** The raw `age` vs. `balance` scatterplot (in the primary balance section) showed only a weak linear correlation (r = 0.098), which can hide a non-linear life-stage pattern (e.g. balance accumulating with career progression, dropping around major life expenses, rising again pre-retirement). Binning `age` into life-stage buckets tests for that non-linear structure directly.

**Method.** `pd.cut(train['age'], bins=[18, 25, 35, 45, 55, 65, 100])`, then median/mean `balance` per bucket, plotted as two line series.

**Result.**

| age bucket | median | mean | count |
|---|---|---|---|
| (18, 25] | 362.5 | 902.7 | 1,324 |
| (25, 35] | 373.0 | 1,147.2 | 15,571 |
| (35, 45] | 444.0 | 1,319.6 | 13,856 |
| (45, 55] | 506.0 | 1,466.1 | 9,548 |
| (55, 65] | 671.0 | 1,958.5 | 4,149 |
| (65, 100] | 1,413.0 | 2,822.0 | 751 |

Balance rises **monotonically** with age bucket — no dip or non-monotonic life-stage pattern — from a median of 362.5 for 18-25 year-olds to 1,413 for 65+ (roughly **3.9x**), and mean balance more than triples across the same range (902.7 → 2,822.0). This is a cleaner, more interpretable trend than the raw scatter's weak r = 0.098 suggested, because the linear correlation coefficient understates a relationship that's real but accelerates at the older end (the 65+ bucket's mean jump is the steepest step in the table) — i.e. the *shape* is closer to convex-increasing than strictly linear. `age` bucket is a stronger balance signal than the raw linear correlation implied and is worth keeping as a binned/non-linear feature (or with an age² term) rather than only as raw `age`.

---

## EDA Summary / Next Steps

- **`test.csv` is a random 10% subset of `train.csv`, not an independent holdout.** Per the UCI source documentation, `test.csv` rows are drawn from `train.csv` itself. This means the provided files cannot be used as a clean train/test split as-is (risk of leakage/duplicated rows if used naively) — the team needs its own split strategy.
- **A time-based split is now well-supported, not just plausible.** The seasonality deep-dive shows `y`-rate rising from ~2-5% in the earliest chronological chunks to 52.9% in the latest chunk — a ~25x drift. A random split would leak this temporal signal between train and validation; a chronological split (train on earlier contacts, validate/test on later ones) is the correct choice. Alternatively, dedupe `train.csv` against `test.csv` and draw a fresh random split, but that would still need to respect the same ordering caveat.
- **`duration` is a leakage feature for the classification target `y`.** Call duration is only known after a call ends, so despite ranking highest in the relevance analysis (0.395), it must be excluded (or used only for a separate "post-call" analysis) from any model meant to decide who to call *before* calling them.
- **Class imbalance in `y`:** ~11.7% "yes" — classification modeling will need to account for this (class weighting, resampling, or threshold tuning; not plain accuracy as the metric).
- **`unknown` sentinel categories** (no true missing values anywhere): `job` ~0.6%, `education` ~4.1%, `contact` ~28.8%, `poutcome` ~81.7% unknown. The categorical-redundancy deep-dive confirms `poutcome == unknown` is a near-duplicate of `pdays == -1` (36,954/36,959 "never contacted" rows have `poutcome == unknown`), so `poutcome`'s categorical detail is only informative for the ~18% of previously-contacted clients — and among those, `poutcome == success` is a very strong signal (y-rate 0.63-0.67 vs. 0.12-0.18 for failure/other at matching `previous` counts).
- **Classification (`y`) — most promising features:** `top_features = ['duration', 'poutcome', 'month', 'contact', 'housing', 'job', 'pdays']` from the relevance ranking. The interaction deep-dive adds nuance beyond single-feature relevance: `contact == cellular` combined with low-volume months (mar, sep, oct, dec; y-rate 0.44-0.53) outperforms the same contact method in high-volume months (may, jul; y-rate ~0.10-0.12) — campaign timing interacts with contact method, not just additive effects. Also watch `contact`–`month` (Cramér's V 0.512) and `housing`–`month` (0.504) redundancy — these categorical pairs move together and may not both be needed in a model.
- **Regression (`balance`) — most promising features:** no single numeric feature is strongly correlated with `balance` (max |r| = 0.098, `age`); `job`, `loan`, `housing`, `education` show clear group median differences (e.g. retired 787 vs. services 340; no-loan 496 vs. has-loan 258). The age-binned deep-dive shows a strong non-linear (convex-increasing) trend that raw linear correlation understates — median balance triples from the youngest to oldest age bucket. A signed-log or robust-scaling transform of `balance` should be considered before modeling given its skew (train max 102,127, min -8,019) and negative values.
- **`balance` vs. `y` link:** converters have a ~76% higher median balance (733 vs. 417) — a real but modest, heavily-overlapping difference. Not strong enough to use `balance` as a standalone classification feature, but worth including alongside stronger signals.
- **Regression (`duration`, `campaign`) — secondary:** kept as a lighter pass; `campaign`'s discrete, right-skewed, count-like nature (min = 1) is worth remembering if it's chosen as the eventual regression target (may call for a count-model-style approach rather than plain linear regression).
- **PCA on the numeric block was considered and dropped** as a secondary variance-structure check — it's target-agnostic and would mostly restate the existing numeric multicollinearity heatmap (only one notable pair, `pdays`/`previous` r=0.455) in a less direct, less interpretable form.
