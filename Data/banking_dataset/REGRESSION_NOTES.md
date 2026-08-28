# EDA Notes — Bank Marketing Dataset (Regression track: balance)

Narrative/discussion companion to `regression.ipynb`. The notebook keeps only short section headers; the "why," the exact method, and the numeric results for each section live here. Section headers below match the notebook's markdown headers 1:1.

Target: **`balance`** (account balance — regression) is the team's chosen primary regression target; `duration`/`campaign` are kept as lighter secondary candidates. See `CLASSIFY_NOTES.md` for the `y` classification track — both notebooks share the same "Import Lib + Load Data," "EDA," "Train Distribution Check," and "Multicollinearity" foundation sections, duplicated so each notebook runs standalone. See `MATRIX_METHODS.md` for the math behind the correlation matrix used below.

---

## Regression EDA — Primary Target: balance

**Purpose.** `balance` is the team's chosen regression target: predicting account balance is a real forecasting problem with clear business meaning, unlike `duration`/`campaign`, which are call-logistics metadata. Reuses the "score every candidate first, then plot only what matters" approach from the classification track's relevance section (see `CLASSIFY_NOTES.md`), now with one combined score for both feature types instead of a numeric-only correlation table plus separate unscored categorical boxplots.

**Method.** Two measures depending on feature type, both landing on a comparable 0–1 scale so they can be ranked together (see `MATRIX_METHODS.md` for background on the numeric side):
- Numeric (`age, day, duration, campaign, pdays, previous`): plain Pearson correlation with `balance` (`abs(train[col].corr(train["balance"]))`).
- Categorical (`job, marital, education, default, housing, loan, contact, month, poutcome, y`): **correlation ratio (η)** — `sqrt(SS_between / SS_total)` from a one-way ANOVA-style decomposition of `balance`'s variance by category. Same 0–1, unsigned scale as Cramér's V: 0 = no association, 1 = category membership fully determines the group mean. This is the direct regression-track analog of Cramér's V — it's what lets a categorical feature be compared against a numeric feature's Pearson r in one ranked table, the same way the classification track combines point-biserial r and Cramér's V.

`y` (the classification target) is included here as a candidate — mirroring how `duration` is included in the classification track's relevance/RF/MI ranking despite being unusable in a deployable model. Scoring it is an EDA question ("does it carry information about `balance`?"); whether it's *usable* is a separate modeling question, addressed below.

Same threshold convention as classification (0.1, Cohen's "small effect" for correlation-scale statistics) — with the same caveat that this convention was developed for Pearson r and is borrowed here across two different association measures (Pearson r and η), not derived specifically for this dataset.

**Result.** Full ranking (relevance score, descending):

| feature | score | type |
|---|---|---|
| month | 0.156 | categorical |
| job | 0.102 | categorical |
| age | 0.098 | numeric |
| education | 0.088 | categorical |
| loan | 0.084 | categorical |
| housing | 0.069 | categorical |
| default | 0.067 | categorical |
| y | 0.053 | categorical |
| contact | 0.049 | categorical |
| poutcome | 0.040 | categorical |
| marital | 0.028 | categorical |
| duration | 0.022 | numeric |
| previous | 0.017 | numeric |
| campaign | 0.015 | numeric |
| day | 0.005 | numeric |
| pdays | 0.003 | numeric |

At threshold 0.1, `top_features_reg = ['month', 'job']` — only these two clear the bar. `age` (0.098) falls just short of the threshold, which is worth flagging rather than accepting at face value: the Age-Binned deep dive below finds a real, monotonic, convex-increasing `age`→`balance` trend that a linear/ANOVA-style score understates, and both RF and Mutual Information (further down) independently rank `age` far higher than this threshold-only view suggests. So `age` is the one feature in this dataset where "didn't clear Step 1" turns out not to mean "no signal" — the same kind of finding the classification track saw with `pdays`/MI.

`y` scores 0.053 — identical to its point-biserial `r` in the classification track (0.053, `CLASSIFY_NOTES.md`'s relevance table), which is a useful sanity check: for a 2-category grouping, the correlation ratio and point-biserial `r` are mathematically the same statistic, so this pair of numbers being identical confirms the η implementation is correct, not a coincidence. `y` sits mid-table, well below the leaders and well below the threshold — real but modest, matching the qualitative "plausible weak segmenting feature" read from the earlier balance-vs-y deep dive, now backed by an actual score instead of just an impression from comparing two medians. Whether `y` is *usable*, though, is a separate question from whether it's *relevant* — see the caveat in the Model-Based Feature Importance section below.

Boxplots and scatterplots for each feature follow below for a visual read; medians by group:
- `job`: retired 787, unknown 677, management 572, unemployed 529, self-employed 526, student 502, technician 421, housemaid 406, admin. 396, blue-collar 388, entrepreneur 352, services 340 (retired vs. services is a ~2.3x spread).
- `education`: tertiary 577, unknown 568, primary 403, secondary 392.
- `marital`: married 477, single 437, divorced 348.
- `housing`: no 507 vs. yes 412 (no-mortgage clients hold higher balances).
- `loan`: no 496 vs. yes 258 (clients with a personal loan hold roughly half the balance).

**Domain interpretation.** `README.md`'s business framing (a term deposit locks cash into an interest-bearing product) gives a plausible mechanism for the categorical gaps above: clients already carrying a mortgage (`housing`) or personal loan (`loan`) likely have less spare cash, which tracks with their lower median balances. `job` reads as a standard income/stability proxy (retired/management/self-employed cluster high, matching typical income patterns) — the same reasoning used for `job`'s classification relevance in `CLASSIFY_NOTES.md`. `month` topping this ranking is a newer finding and less settled: it could reflect a genuine seasonal deposit pattern (payroll cycles, bonuses, tax-related deposits concentrated in certain months), but given the classification track's finding that `month` partly proxies chronological campaign drift rather than calendar seasonality (see `CLASSIFY_NOTES.md`'s time/seasonality deep-dive), the same caution likely applies here — this hasn't been separately verified for `balance` and would need its own chronological check before trusting `month` as a "real" balance driver rather than a proxy for *when in the campaign* a client happened to be contacted.

---

## Multicollinearity (numeric feature-to-feature)

**Purpose.** The relevance section above answers "which features relate to `balance`"; this section answers a different question — "which features overlap with *each other*." Two features can each look weak alone yet be correlated with each other (redundant signal), which matters for model feature selection independent of target relevance. (Identical section/result also appears in `CLASSIFY_NOTES.md` — the check is target-agnostic, so it's duplicated as shared foundation rather than cross-referenced.)

**Method.** Pearson correlation matrix (`train[numeric_cols].corr()`) over `age, balance, day, duration, campaign, pdays, previous`, rendered as a heatmap. See `MATRIX_METHODS.md` for how the matrix is computed and how to read it.

**Result.** All pairwise correlations are weak (|r| < 0.1) except one clear pair: **`pdays` vs `previous`, r = 0.455** — expected, since both describe prior-campaign contact history (days since last contact vs. number of prior contacts). Next largest: `day` vs `campaign` (r = 0.162), `campaign` vs `pdays` (r = -0.089), `campaign` vs `duration` (r = -0.085). No other pair exceeds 0.1. Conclusion: numeric features are largely non-redundant with each other, aside from the expected `pdays`/`previous` link.

---

### Discussion — balance

**Purpose.** Consolidate the modeling implications of the balance-vs-numeric and balance-vs-categorical results above before moving to the secondary regression pass.

**Method.** N/A — synthesis of the results above plus the univariate `balance` histogram/boxplot from the earlier distribution-check section.

**Result.** No single feature dominates the relevance ranking — only `month` (0.156) and `job` (0.102) clear the 0.1 threshold, `age` (0.098) falls just short despite carrying real signal (see the Age-Binned deep dive), and no numeric feature besides `age` comes close; a regression model will likely need to combine several moderate signals (`month`, `job`, `age`, plus the categorical breakdowns for `housing`/`loan`/`education`) rather than lean on one dominant driver. `balance` itself is heavily right-skewed with a long tail of large positive values (max 102,127 in `train`) and a meaningful chunk of negative balances/overdrafts (min -8,019 in `train`; see the univariate histogram/boxplot earlier in the notebook, which flagged 4,729 IQR-outlier rows, 10.46% of data). Because of the negative values, a plain `log(balance)` transform won't work directly; a **signed-log** transform (`sign(x) * log1p(|x|)`) or a robust-scaling approach should be considered at modeling time.

**Why is every relevance score weak? Tested, not just assumed.** The obvious hypothesis is that `balance`'s extreme skew (skew = 8.36) is dragging correlations toward zero via a handful of outlier rows. Tested directly: trimming `balance` to its 1st–99th percentile (dropping the most extreme 2% of rows, cutting skew to 2.80) barely moves any score — `age`'s r goes from 0.098 to 0.103, `month`'s η goes from 0.156 to 0.183, `job`'s η stays at ~0.10. **This rules out the outlier-artifact explanation**: the weak relationship holds throughout the bulk of the distribution, not just because a few extreme rows are suppressing it. The real explanation is a domain mismatch — every column in this dataset describes either the marketing campaign (`contact`, `day`, `month`, `duration`, `campaign`, `poutcome`) or coarse demographics (`job`, `age`, `education`, `marital`, binary `housing`/`loan` flags). None of these are direct financial variables — there's no income, spending history, years employed, or other account holdings recorded anywhere in this dataset. `balance` is a snapshot of a client's actual financial life, shaped by factors this dataset was never designed to capture; `job`/`housing`/`loan` are lossy proxies for "financial situation," not the thing itself. That's why no feature — numeric or categorical — gets close to a strong effect size: the data-generating process for `balance` mostly lives outside the columns available here.

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

**Method.** Built a lightweight feature matrix with `pd.get_dummies` for the 10 categorical columns (`job, marital, education, default, housing, loan, contact, month, poutcome, y`) plus the numeric columns excluding `balance` itself (`age, day, duration, campaign, pdays, previous`), no sklearn pipeline (consistent with the notebook's light-weight style elsewhere). Fit a `RandomForestRegressor(n_estimators=300, random_state=42)` on `balance`. One-hot dummy columns are summed back to their parent column (`groupby` on a dummy→parent map) so a categorical feature's importance is comparable to a numeric feature's, rather than being split across its individual categories.

**Result.**

Regression (`balance`) — RF importance, aggregated per feature:

| feature | importance |
|---|---|
| duration | 0.262 |
| age | 0.139 |
| job | 0.104 |
| day | 0.104 |
| month | 0.079 |
| campaign | 0.077 |
| marital | 0.043 |
| pdays | 0.040 |
| education | 0.040 |
| contact | 0.034 |
| housing | 0.022 |
| poutcome | 0.016 |
| previous | 0.016 |
| y | 0.012 |
| loan | 0.008 |
| default | 0.003 |

**Comparison.** `job` confirms as a meaningful driver in both views (relevance rank #2 at 0.102, RF rank #3 at 0.104) — good agreement. `age` is where RF adds real news: it falls just short of the Step 1 threshold (0.098) but RF ranks it #2 overall (0.139), consistent with the age-binned deep dive's clean monotonic trend — a case where the single-feature score understated a real relationship. The bigger surprise runs the other way for `month`: it's the **top** feature in the relevance ranking (0.156) but drops to 5th in RF (0.079). A plausible explanation, borrowing the target-agnostic categorical-redundancy finding from `CLASSIFY_NOTES.md`: `month` is heavily correlated with `contact` (Cramér's V 0.512) and `housing` (0.504), so RF's greedy splitting may credit those correlated features instead of "reusing" `month`'s signal — the same dynamic already seen suppressing `housing`/`contact` in the classification track's RF results. Two more surprises: `duration` is RF's #1 feature for `balance` (0.262) despite having a near-zero Pearson correlation (0.022) — likely continuous-feature importance inflation (a known RF bias: continuous, high-cardinality numeric features tend to get inflated importance relative to low-cardinality categoricals, purely from having more possible split points) rather than a genuine relationship, since there's no obvious mechanism linking call length to account balance; and `housing`/`loan` — which showed clear median gaps in the categorical boxplots (507 vs. 412; 496 vs. 258) — rank quite low in RF (0.022, 0.008), suggesting their balance signal is smaller in a multivariate model than the univariate group-median comparison implied, or is being absorbed by the same `month` redundancy. Net takeaway: RF **adds** `age` as a more-confirmed driver, **tempers** confidence in `month`/`housing`/`loan` relative to their univariate scores, and treats `duration`'s high RF score with skepticism (probably an artifact of feature type, not a real driver).

**`y` caveat — relevant but not usable.** `y` ranks 14th of 16 in RF (0.012) — weak, consistent with its modest Step 1 score (0.053, rank 8) and its MI score below. Even setting the low score aside: `y` cannot be used as an input for any `balance` model meant to run *before* a campaign call, for the same reason `duration` is excluded from the `y`-classification model — `y` (whether the client subscribed) is the outcome of that same call, not something known beforehand. If a future `balance` model is only ever used for post-call analysis (e.g. profiling clients after a campaign wraps), this restriction wouldn't apply — but treat `y` as off-limits by default, the same way `duration` is treated in the classification track.

**Domain interpretation.** The same business mechanisms noted in the Primary Target section's domain-interpretation paragraph (spare-cash constraints via `housing`/`loan`, income/stability proxy via `job`) apply here too — the RF view doesn't introduce a new business story, it just reweights confidence in the existing ones (see the comparison above for which features gain or lose credibility under RF).

---

## Relevance Cross-Check — Mutual Information

**Purpose.** Same rationale as the classification track (see `CLASSIFY_NOTES.md`): a model-free, non-linear cross-check that complements the Step 1 relevance ranking (Pearson r + correlation ratio) above and doesn't inherit RF's known bias toward continuous, high-cardinality features. Particularly relevant here because the age-binned deep dive above already found a real, monotonic, convex-increasing `age`→`balance` relationship that Step 1's linear/ANOVA-style score (0.098, just under threshold) understated — mutual information is exactly the tool to check whether that non-linear signal shows up in a method that isn't fitting a tree model at all.

**Method.** `sklearn.feature_selection.mutual_info_regression`, numeric columns (`age, day, duration, campaign, pdays, previous`, i.e. `other_numeric` — `balance` itself excluded) passed raw with `discrete_features=False`; categorical columns (`job, marital, education, default, housing, loan, contact, month, poutcome, y`) label-encoded via `pd.factorize` and passed with `discrete_features=True`. `random_state=42`. Same scale caveat as the classification track: MI values aren't on the same 0–1 scale as Pearson `r`, so read the comparison below as rank agreement, not matching numeric values.

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
| y | 0.0221 | categorical |
| default | 0.0210 | categorical |
| day | 0.0146 | numeric |
| contact | 0.0126 | categorical |
| poutcome | 0.0105 | categorical |
| duration | 0.0092 | numeric |
| pdays | 0.0081 | numeric |
| previous | 0.0000 | numeric |
| campaign | 0.0000 | numeric |

**Comparison — does MI confirm `age`'s non-linear relevance?** Yes. `age` is MI's **2nd**-highest feature overall (0.067, behind only `job`), a far more prominent rank than Step 1 gave it — there, `age` (0.098) was the strongest numeric score but still fell just short of the 0.1 threshold. MI, computed independently of both the Step 1 score and the RF model, agrees with RF (`age` ranked 2nd there too, importance 0.142) that `age` carries real, substantial signal — directly corroborating the age-binned deep dive's finding of a convex-increasing, non-linear `age`→`balance` trend (median balance triples from the 18-25 to 65+ bucket). Three independent methods (a monotonic binned trend, a fitted RF, and a model-free non-linear statistic) now agree on `age` — strong evidence this is a real effect, not an artifact of any one method, and a case where a feature that missed the Step 1 cutoff turned out to matter anyway.

A second useful divergence: **`duration` is RF's #1 feature for `balance` (0.262) but MI ranks it 13th of 16 (0.0092)**, near the bottom, matching its weak Step 1 score (0.022). The RF Comparison above already flagged `duration`'s top rank as suspicious — "no obvious mechanism linking call length to account balance," likely continuous-feature importance inflation. MI's independent, model-free view agrees with that skepticism: `duration` shows essentially no relationship to `balance` outside of RF's split-point bias, exactly the sanity check this cross-check was meant to provide.

Third: `job` is MI's single highest feature (0.108), agreeing with RF (3rd, 0.104) and Step 1 (2nd, 0.102) and the categorical-boxplot group-median spread (retired 787 vs. services 340) — solid three-way agreement `job` is a genuine `balance` driver. `month`, by contrast, tells a consistent demotion story across both cross-checks: it's Step 1's **top** feature (0.156) but drops to 5th in MI (0.036) as well as 5th in RF (0.079) — both independent methods agree it's weaker in a multivariate/non-linear sense than its raw score suggested, reinforcing the RF Comparison's redundancy hypothesis (`month` overlapping with `contact`/`housing`) rather than this being an RF-only quirk. `housing` and `loan` — which showed clear median gaps in the boxplots (507 vs. 412; 496 vs. 258) but ranked low in RF (0.022, 0.008) — sit noticeably higher in MI's ranking (6th at 0.033, 7th at 0.028) than in RF. This supports the hypothesis that RF was absorbing `housing`/`loan`'s signal into a correlated feature (`month`, Cramér's V 0.504) via greedy splitting rather than `housing`/`loan` genuinely lacking signal — MI, which scores each feature independently, still detects it.

`y` lands 8th of 16 by MI (0.022) — squarely in the middle, matching its equally middling Step 1 (rank 8) and RF (rank 14, though RF's greedy splitting tends to shortchange redundant/weaker categoricals generally) standing. All three methods agree: `y` carries real but modest information about `balance`, never approaching the top tier — consistent with the ~76% median gap already found in the standalone balance-vs-y comparison, now triangulated rather than resting on that one comparison alone. The usability caveat from the RF section still applies regardless of this score.

---

## EDA Summary / Next Steps

- **`test.csv` is a random 10% subset of `train.csv`, not an independent holdout.** Per the UCI source documentation (see `README.md`), `test.csv` rows are drawn from `train.csv` itself. This means the provided files cannot be used as a clean train/test split as-is (risk of leakage/duplicated rows if used naively) — the team needs its own split strategy.
- **A time-based split is likely warranted here too.** The classification track's seasonality deep-dive (see `CLASSIFY_NOTES.md`) shows `y`-rate drifting ~25x over the campaign's chronological span. That deep-dive was run against `y`, not `balance`, but since the drift is about row ordering/campaign conditions over time rather than something specific to the classification target, a chronological train/validation split is the safer default for the `balance` model as well, pending a dedicated check against `balance` itself.
- **`unknown` sentinel categories** (no true missing values anywhere): `job` ~0.6%, `education` ~4.1%, `contact` ~28.8%, `poutcome` ~81.7% unknown. Only `job` and `education` are used as `balance` predictors here, so the sentinel-category caveat matters less than in the classification track, but the `unknown` group did show up with a distinct (non-imputable) median balance in both (`job` unknown: 677, `education` unknown: 568) — worth keeping as its own category rather than imputing.
- **Final feature selection (6 features):** `job`, `age` (binned), `month`, `housing`, `loan`, `education`. Worth being explicit about scale here: none of these score well in absolute terms in *any* of the three methods — the best score in each is `month` 0.156 (Step 1), `age` 0.139 (RF), `job` 0.108 (MI), all still inside Cohen's "small effect" band, nowhere close to `duration`(0.395)/`poutcome`(0.312) in the classification track. This isn't a weakness of the selection — it's an honest fact about `balance`: no single feature is a strong predictor, so the model needs to combine several modest signals. `job`/`age`/`month` rank consistently across all three methods; `housing`/`loan`/`education` carry real signal (clear boxplot median gaps, moderate MI ranks) but their rank swings more, especially getting suppressed in RF via `month`'s redundancy. Clears the assignment's 5-feature minimum without padding — each of the 6 has independent evidence behind it, detailed below.
- **Most promising features:** `top_features_reg = ['month', 'job']` clear the Step 1 relevance threshold (Pearson r for numeric, correlation ratio η for categorical); `age` (0.098) falls just short but is the clearest case in this track of a feature the threshold underrates. RF and Mutual Information — two independent cross-checks — both **promote** `age` (RF 2nd at 0.139, MI 2nd at 0.067) and both **demote** `month` relative to its Step 1 rank (RF 5th at 0.079, MI 5th at 0.036), the same direction of disagreement in both methods, which points to `month`'s categorical redundancy with `contact`/`housing` (Cramér's V 0.51/0.50, from `CLASSIFY_NOTES.md`) rather than noise. `job` is the one feature all three methods agree on outright (Step 1 2nd, RF 3rd, MI 1st). `housing`/`loan` show clear group-median gaps (no-loan 496 vs. has-loan 258) but rank low in RF — MI restores some of that credibility (6th/7th), again consistent with `month` absorbing shared signal during RF's greedy splitting. `duration`'s RF top-rank (0.262) is not trustworthy — near-zero Step 1 score (0.022) and MI rank 13th of 16 both flag it as a feature-type artifact, not a real driver. The age-binned deep-dive shows a strong non-linear (convex-increasing) trend that Step 1's score understates — median balance triples from the youngest to oldest age bucket — and both RF and MI corroborate that this is real. A signed-log or robust-scaling transform of `balance` should be considered before modeling given its skew (train max 102,127, min -8,019) and negative values.
- **`y` as a `balance` predictor — relevant but off-limits by default.** `y` was scored alongside every other candidate (Step 1: 0.053, rank 8; RF: 0.012, rank 14; MI: 0.022, rank 8) — real but modest signal across all three methods, consistent with the ~76% higher median balance among converters (733 vs. 417) found in the standalone balance-vs-y comparison. But scoring it isn't the same as being able to use it: `y` is the outcome of the same campaign call that any pre-call `balance` model would need to run *before*, so it's excluded from a deployable model for the same reason `duration` is excluded from the `y`-classifier — this is a modeling restriction, not an EDA one.
- **Secondary targets (`duration`, `campaign`):** kept as a lighter pass; `campaign`'s discrete, right-skewed, count-like nature (min = 1) is worth remembering if it's chosen as the eventual regression target (may call for a count-model-style approach rather than plain linear regression).
- **PCA on the numeric block was considered and dropped** as a secondary variance-structure check — it's target-agnostic and would mostly restate the existing numeric multicollinearity heatmap (only one notable pair, `pdays`/`previous` r=0.455) in a less direct, less interpretable form.
