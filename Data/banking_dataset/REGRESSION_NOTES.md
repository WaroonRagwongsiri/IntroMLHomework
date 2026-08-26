# EDA Notes — Bank Marketing Dataset (Regression track: balance)

Narrative/discussion companion to `regression.ipynb`. The notebook keeps only short section headers; the "why," the exact method, and the numeric results for each section live here. Section headers below match the notebook's markdown headers 1:1.

Target: **`balance`** (account balance — regression) is the team's chosen primary regression target; `duration`/`campaign` are kept as lighter secondary candidates. See `CLASSIFY_NOTES.md` for the `y` classification track — both notebooks share the same "Import Lib + Load Data," "EDA," "Train Distribution Check," and "Multicollinearity" foundation sections, duplicated so each notebook runs standalone. See `MATRIX_METHODS.md` for the math behind the correlation matrix used below.

---

## Multicollinearity (numeric feature-to-feature)

**Purpose.** Two features can each look weak alone yet be correlated with each other (redundant signal), which matters for model feature selection independent of target relevance. Kept as its own heatmap over the numeric columns only, ahead of the regression-specific sections below. (Identical section/result also appears in `CLASSIFY_NOTES.md` — the check is target-agnostic, so it's duplicated as shared foundation rather than cross-referenced.)

**Method.** Pearson correlation matrix (`train[numeric_cols].corr()`) over `age, balance, day, duration, campaign, pdays, previous`, rendered as a heatmap. See `MATRIX_METHODS.md` for how the matrix is computed and how to read it.

**Result.** All pairwise correlations are weak (|r| < 0.1) except one clear pair: **`pdays` vs `previous`, r = 0.455** — expected, since both describe prior-campaign contact history (days since last contact vs. number of prior contacts). Next largest: `day` vs `campaign` (r = 0.162), `campaign` vs `pdays` (r = -0.089), `campaign` vs `duration` (r = -0.085). No other pair exceeds 0.1. Conclusion: numeric features are largely non-redundant with each other, aside from the expected `pdays`/`previous` link.

---

## Regression EDA — Primary Target: balance

**Purpose.** `balance` is the team's chosen regression target: predicting account balance is a real forecasting problem with clear business meaning, unlike `duration`/`campaign`, which are call-logistics metadata. Reuses the "correlate first, then plot only what matters" approach from the classification track's relevance section, for consistency (see `CLASSIFY_NOTES.md`).

**Method.** Pearson correlation of `balance` against the other 6 numeric columns, plus scatterplots for each; boxplots of `balance` by 5 categorical columns (`job, education, marital, housing, loan`), medians used to order the categories and outliers hidden for readability.

**Result.** Correlations with `balance` (Pearson r): `age` 0.098, `duration` 0.022, `previous` 0.017, `campaign` -0.015, `day` 0.005, `pdays` 0.003 — all weak, `age` is the strongest but still marginal. Median `balance` by group:
- `job`: retired 787, unknown 677, management 572, unemployed 529, self-employed 526, student 502, technician 421, housemaid 406, admin. 396, blue-collar 388, entrepreneur 352, services 340 (retired vs. services is a ~2.3x spread).
- `education`: tertiary 577, unknown 568, primary 403, secondary 392.
- `marital`: married 477, single 437, divorced 348.
- `housing`: no 507 vs. yes 412 (no-mortgage clients hold higher balances).
- `loan`: no 496 vs. yes 258 (clients with a personal loan hold roughly half the balance).

No single numeric feature is strongly predictive; categorical group differences (especially `job`, `loan`, `housing`) are more informative than any numeric correlation.

**Domain interpretation.** `README.md`'s business framing (a term deposit locks cash into an interest-bearing product) gives a plausible mechanism for the categorical gaps above: clients already carrying a mortgage (`housing`) or personal loan (`loan`) likely have less spare cash, which tracks with their lower median balances. `job` reads as a standard income/stability proxy (retired/management/self-employed cluster high, matching typical income patterns) — the same reasoning used for `job`'s classification relevance in `CLASSIFY_NOTES.md`.

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

## Deep Dive — balance vs. y

**Purpose.** Checks whether the regression target (`balance`) and the classification target (`y`) are linked — i.e. do converters have a meaningfully different balance profile? This informs whether `balance` could double as a classification feature (see `CLASSIFY_NOTES.md`) and whether `y` should be considered as a segmenting variable for balance modeling.

**Method.** Boxplot and violinplot of `balance` by `y`, plus median/mean `balance` per `y` group.

**Result.**

| y | median | mean | count |
|---|---|---|---|
| no | 417.0 | 1,303.7 | 39,922 |
| yes | 733.0 | 1,804.3 | 5,289 |

Converters (`y == yes`) have a **~76% higher median balance** (733 vs. 417) and ~38% higher mean balance (1,804 vs. 1,304) than non-converters. The gap is real but modest relative to the overall spread of `balance` (std ≈ 3,045 in the full dataset) — visible in the boxplot/violin as a shifted but heavily overlapping distribution, not a clean separation. `balance` alone would be a weak classifier feature, consistent with its low relevance score (0.053) in the classification track, but it does carry some signal in combination with other features. Conversely, `y` is a plausible (if modest) segmenting feature for a `balance` model.

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

## Model-Based Feature Importance (RF)

**Purpose.** The correlation and categorical-boxplot checks above are all *bivariate* — each feature is scored against `balance` independently. The assignment's feature-importance hint (Medium article on "Feature Importance: How's and Why's") points specifically at *model-based* importance, which captures interactions and non-linearities a single-feature correlation can miss. Running a quick Random Forest gives a second, complementary view for `balance` and doubles as a sanity check: do the top RF features roughly agree with the correlation/boxplot findings above, or does the model surface something the bivariate pass missed? (The `y` classifier's RF importance is in `CLASSIFY_NOTES.md`.)

**Method.** Built a lightweight feature matrix with `pd.get_dummies` for the 9 categorical columns (`job, marital, education, default, housing, loan, contact, month, poutcome`) plus the numeric columns excluding `balance` itself (`age, day, duration, campaign, pdays, previous`), no sklearn pipeline (consistent with the notebook's light-weight style elsewhere). Fit a `RandomForestRegressor(n_estimators=300, random_state=42)` on `balance`. One-hot dummy columns are summed back to their parent column (`groupby` on a dummy→parent map) so a categorical feature's importance is comparable to a numeric feature's, rather than being split across its individual categories.

**Result.**

Regression (`balance`) — RF importance, aggregated per feature:

| feature | importance |
|---|---|
| duration | 0.266 |
| age | 0.142 |
| day | 0.106 |
| job | 0.104 |
| month | 0.079 |
| campaign | 0.077 |
| marital | 0.043 |
| education | 0.041 |
| pdays | 0.040 |
| contact | 0.034 |
| housing | 0.022 |
| poutcome | 0.017 |
| previous | 0.017 |
| loan | 0.008 |
| default | 0.003 |

**Comparison.** Confirms the correlation-section finding that `age` carries real signal (RF rank #2, 0.142) — consistent with the age-binned deep dive's clean monotonic trend. `job` also confirms as a meaningful categorical driver (RF rank #4, 0.104), matching the boxplot group-median spread (retired 787 vs. services 340). Two surprises: `duration` is RF's #1 feature for `balance` (0.266) despite having a near-zero Pearson correlation (0.022) — likely continuous-feature importance inflation (a known RF bias: continuous, high-cardinality numeric features tend to get inflated importance relative to low-cardinality categoricals, purely from having more possible split points) rather than a genuine relationship, since there's no obvious mechanism linking call length to account balance; and `housing`/`loan` — which showed clear median gaps in the categorical boxplots (507 vs. 412; 496 vs. 258) — rank quite low in RF (0.022, 0.008), suggesting their balance signal is smaller in a multivariate model than the univariate group-median comparison implied, or is being absorbed by correlated features (`housing`–`month` Cramér's V 0.504, from the classification track's categorical-redundancy deep dive). Net takeaway: RF **adds** `age` and `job` as more-confirmed balance drivers, but **tempers** confidence in `housing`/`loan` and treats `duration`'s high RF score with skepticism (probably an artifact of feature type, not a real driver).

**Domain interpretation.** The same business mechanisms noted in the Primary Target section's domain-interpretation paragraph (spare-cash constraints via `housing`/`loan`, income/stability proxy via `job`) apply here too — the RF view doesn't introduce a new business story, it just reweights confidence in the existing ones (see the comparison above for which features gain or lose credibility under RF).

---

## Relevance Cross-Check — Mutual Information

**Purpose.** Same rationale as the classification track (see `CLASSIFY_NOTES.md`): a model-free, non-linear cross-check that complements the linear Pearson-correlation/categorical-boxplot view above and doesn't inherit RF's known bias toward continuous, high-cardinality features. Particularly relevant here because the age-binned deep dive above already found a real, monotonic, convex-increasing `age`→`balance` relationship that raw Pearson `r` (0.098) understated — mutual information is exactly the tool to check whether that non-linear signal shows up in a method that isn't fitting a tree model at all.

**Method.** `sklearn.feature_selection.mutual_info_regression`, numeric columns (`age, day, duration, campaign, pdays, previous`, i.e. `other_numeric` — `balance` itself excluded) passed raw with `discrete_features=False`; categorical columns (`job, marital, education, default, housing, loan, contact, month, poutcome`) label-encoded via `pd.factorize` and passed with `discrete_features=True`. `random_state=42`. Same scale caveat as the classification track: MI values aren't on the same 0–1 scale as Pearson `r`, so read the comparison below as rank agreement, not matching numeric values.

**Result.**

| feature | mutual info | type |
|---|---|---|
| job | 0.1083 | categorical |
| age | 0.0672 | numeric |
| education | 0.0474 | categorical |
| marital | 0.0443 | categorical |
| month | 0.0362 | categorical |
| housing | 0.0327 | categorical |
| loan | 0.0282 | categorical |
| default | 0.0210 | categorical |
| day | 0.0146 | numeric |
| contact | 0.0126 | categorical |
| poutcome | 0.0105 | categorical |
| duration | 0.0092 | numeric |
| pdays | 0.0081 | numeric |
| previous | 0.0000 | numeric |
| campaign | 0.0000 | numeric |

**Comparison — does MI confirm `age`'s non-linear relevance?** Yes. `age` is MI's **2nd**-highest feature overall (0.067, behind only `job`), a far more prominent rank than the raw Pearson correlation table gave it — there, `age` r=0.098 was technically the strongest numeric correlation, but weak on an absolute 0–1 scale and easy to read as "nothing here." MI, computed independently of both the linear-correlation view and the RF model, agrees with RF (`age` ranked 2nd there too, importance 0.142) that `age` carries real, substantial signal — directly corroborating the age-binned deep dive's finding of a convex-increasing, non-linear `age`→`balance` trend (median balance triples from the 18-25 to 65+ bucket). Three independent methods (a monotonic binned trend, a fitted RF, and a model-free non-linear statistic) now agree on `age` — strong evidence this is a real effect, not an artifact of any one method.

A second useful divergence: **`duration` is RF's #1 feature for `balance` (0.266) but MI ranks it 12th of 15 (0.0092)**, near the bottom, matching its weak raw Pearson correlation (0.022). The RF Comparison above already flagged `duration`'s top rank as suspicious — "no obvious mechanism linking call length to account balance," likely continuous-feature importance inflation. MI's independent, model-free view agrees with that skepticism: `duration` shows essentially no relationship to `balance` outside of RF's split-point bias, exactly the sanity check this cross-check was meant to provide.

Third: `job` is MI's single highest feature (0.108), agreeing with RF (4th, 0.104) and the categorical-boxplot group-median spread (retired 787 vs. services 340) — solid three-way agreement `job` is a genuine `balance` driver. `housing` and `loan` — which showed clear median gaps in the boxplots (507 vs. 412; 496 vs. 258) but ranked low in RF (0.022, 0.008) — sit noticeably higher in MI's ranking (6th at 0.033, 7th at 0.028) than in RF. This supports the RF Comparison's hypothesis that RF was absorbing `housing`/`loan`'s signal into a correlated feature (`month`, Cramér's V 0.504) via greedy splitting rather than `housing`/`loan` genuinely lacking signal — MI, which scores each feature independently, still detects it.

---

## EDA Summary / Next Steps

- **`test.csv` is a random 10% subset of `train.csv`, not an independent holdout.** Per the UCI source documentation (see `README.md`), `test.csv` rows are drawn from `train.csv` itself. This means the provided files cannot be used as a clean train/test split as-is (risk of leakage/duplicated rows if used naively) — the team needs its own split strategy.
- **A time-based split is likely warranted here too.** The classification track's seasonality deep-dive (see `CLASSIFY_NOTES.md`) shows `y`-rate drifting ~25x over the campaign's chronological span. That deep-dive was run against `y`, not `balance`, but since the drift is about row ordering/campaign conditions over time rather than something specific to the classification target, a chronological train/validation split is the safer default for the `balance` model as well, pending a dedicated check against `balance` itself.
- **`unknown` sentinel categories** (no true missing values anywhere): `job` ~0.6%, `education` ~4.1%, `contact` ~28.8%, `poutcome` ~81.7% unknown. Only `job` and `education` are used as `balance` predictors here, so the sentinel-category caveat matters less than in the classification track, but the `unknown` group did show up with a distinct (non-imputable) median balance in both (`job` unknown: 677, `education` unknown: 568) — worth keeping as its own category rather than imputing.
- **Most promising features:** no single numeric feature is strongly correlated with `balance` (max |r| = 0.098, `age`); `job`, `loan`, `housing`, `education` show clear group median differences (e.g. retired 787 vs. services 340; no-loan 496 vs. has-loan 258). RF importance confirms `age` and `job` as real multivariate drivers, but tempers `housing`/`loan` (low RF importance despite the group-median gap) and flags `duration`'s top RF rank as a likely feature-type artifact rather than a genuine relationship. A third, model-free mutual-information cross-check agrees on all three points: `age` ranks 2nd overall by MI (confirming its non-linear signal independent of RF), `job` ranks 1st, and `duration` — RF's top feature — drops to 12th of 15 by MI, backing up the suspicion that its RF rank was a feature-type artifact rather than real signal; MI also restores some credibility to `housing`/`loan` (6th/7th) that RF had suppressed. The age-binned deep-dive shows a strong non-linear (convex-increasing) trend that raw linear correlation understates — median balance triples from the youngest to oldest age bucket — and both RF and MI corroborate that this is real, not a correlation-coefficient artifact. A signed-log or robust-scaling transform of `balance` should be considered before modeling given its skew (train max 102,127, min -8,019) and negative values.
- **`balance` vs. `y` link:** converters have a ~76% higher median balance (733 vs. 417) — a real but modest, heavily-overlapping difference; `y` could be included as a weak segmenting feature.
- **Secondary targets (`duration`, `campaign`):** kept as a lighter pass; `campaign`'s discrete, right-skewed, count-like nature (min = 1) is worth remembering if it's chosen as the eventual regression target (may call for a count-model-style approach rather than plain linear regression).
- **PCA on the numeric block was considered and dropped** as a secondary variance-structure check — it's target-agnostic and would mostly restate the existing numeric multicollinearity heatmap (only one notable pair, `pdays`/`previous` r=0.455) in a less direct, less interpretable form.
