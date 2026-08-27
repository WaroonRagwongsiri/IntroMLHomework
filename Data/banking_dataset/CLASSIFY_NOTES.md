# EDA Notes — Bank Marketing Dataset (Classification track: y)

Narrative/discussion companion to `classify.ipynb`. The notebook keeps only short section headers; the "why," the exact method, and the numeric results for each section live here. Section headers below match the notebook's markdown headers 1:1.

Target: **`y`** (subscribed to term deposit — classification) is the team's chosen classification target. See `REGRESSION_NOTES.md` for the `balance` regression track — both notebooks share the same "Import Lib + Load Data," "EDA," "Train Distribution Check," and "Multicollinearity" foundation sections, duplicated so each notebook runs standalone. See `MATRIX_METHODS.md` for the math behind every correlation/redundancy matrix and crosstab used below.

---

## Relevance vs. Classification Target (y) — Correlation First

**Purpose.** With 16 candidate features, plotting every one against `y` (as in the univariate pass) would produce a wall of plots, most uninformative. Instead we compute a single relevance ranking across all features first, so only features that clear a threshold get a detailed follow-up plot.

**Method.** Two measures depending on feature type, both landing on a comparable 0–1 scale, so they can be ranked together (see `MATRIX_METHODS.md` for the underlying math of both):
- Numeric (`age, balance, day, duration, campaign, pdays, previous`): point-biserial correlation with `y` (encoded 0/1 — only the target is mapped to numbers). Point-biserial is mathematically Pearson correlation for a continuous-vs-binary pair.
- Categorical (`job, marital, education, default, housing, loan, contact, month, poutcome`): Cramér's V from the raw-label contingency table (`pd.crosstab` → chi-square). No ordinal/one-hot encoding, so category count/order can't bias the score.

Leakage call-out: `duration` (call length) is only known *after* a call ends, so it cannot be used as a model input for deciding who to call, regardless of its ranking.

**Why threshold = 0.1.** Point-biserial `r` and Cramér's V both land on a 0-1-ish scale comparable to Pearson `r`, so the standard **Cohen's (1988) "small effect" convention** for correlation-scale statistics applies: 0.1 = small, 0.3 = medium, 0.5 = large. Cutting at 0.1 isn't an arbitrary round number — it's "keep only features clearing at least a conventionally non-trivial effect size." Caveat: this convention was developed for Pearson `r` specifically and is borrowed here across two different association measures (point-biserial `r` and Cramér's V) rather than derived for this dataset, so it's a reasonable default, not a statistically proven optimum.

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

**Domain interpretation.** `README.md`'s column groupings (bank client data / last-contact fields / other campaign attributes) and business framing (a term deposit locks cash into an interest-bearing product for a fixed period) give plausible mechanisms behind the ranking, not just statistics:
- `duration` — call length is a symptom of engagement (a long call usually means the client is asking questions before saying yes; a short one is often an instant no or hang-up) — exactly why it's leakage rather than a usable predictor.
- `poutcome` — a client who already said yes to a past campaign is a warm lead; the strongest legitimate (non-leaky) behavioral signal available in the data.
- `month` — a "last contact" field that partly encodes the seasonality drift found in the time deep-dive below (bank-side targeting/strategy shifted over the campaign's life), not a calendar-season effect per se.
- `contact` — contact method; `unknown` skews toward the earlier, lower-quality-outreach period, so it's acting partly as another timing/data-quality proxy — consistent with its high Cramér's V with `month` (0.512, see the redundancy deep dive).
- `housing` / `loan` — clients already carrying a mortgage or personal loan plausibly have less spare cash to lock into a term deposit; also matches the lower `balance` medians found for `loan == yes` (258 vs. 496) and `housing == yes` (412 vs. 507) in the regression track (see `REGRESSION_NOTES.md`).
- `job` — a standard income/stability proxy, which is why it also ranks for `balance` in the regression track.

This doesn't change the ranking or threshold — it's a plausibility check that the top statistical features also have a believable business mechanism, strengthening (not substituting for) the quantitative case.

---

## Multicollinearity (numeric feature-to-feature)

**Purpose.** The relevance section answers "which features relate to `y`"; this answers a different question — "which features overlap with *each other*." Two features can each look weak alone yet be correlated with each other (redundant signal), which matters for model feature selection independent of target relevance. Kept as its own heatmap over the numeric columns only, to avoid conflating the two questions. (Identical section/result also appears in `REGRESSION_NOTES.md` — the check is target-agnostic, so it's duplicated as shared foundation rather than cross-referenced.)

**Method.** Pearson correlation matrix (`train[numeric_cols].corr()`) over `age, balance, day, duration, campaign, pdays, previous`, rendered as a heatmap. See `MATRIX_METHODS.md` for how the matrix is computed and how to read it.

**Result.** All pairwise correlations are weak (|r| < 0.1) except one clear pair: **`pdays` vs `previous`, r = 0.455** — expected, since both describe prior-campaign contact history (days since last contact vs. number of prior contacts). Next largest: `day` vs `campaign` (r = 0.162), `campaign` vs `pdays` (r = -0.089), `campaign` vs `duration` (r = -0.085). No other pair exceeds 0.1. Conclusion: numeric features are largely non-redundant with each other, aside from the expected `pdays`/`previous` link.

---

## Deep Dive — Categorical-Categorical Redundancy (y)

**Purpose.** The multicollinearity section above only covers numeric-numeric redundancy. Categorical features can be just as redundant with each other (e.g. `contact` method and `month` might move together for operational reasons), and this matters for feature selection and for interpreting the relevance ranking. Also directly tests the suspected `poutcome`/`pdays` redundancy noted in the univariate pass (both driven by "was this client ever contacted before"), across two different dtypes (categorical `poutcome` vs. numeric `pdays`), which a same-dtype Cramér's V matrix can't capture on its own.

**Method.** Pairwise Cramér's V (reusing the `cramers_v` helper) across `job, marital, education, default, housing, loan, contact, month, poutcome`, rendered as a 9x9 heatmap. Separately, an explicit crosstab of `poutcome` vs. `train['pdays'] == -1` (boolean "never contacted" flag) to quantify the cross-dtype redundancy directly. See `MATRIX_METHODS.md` for how the Cramér's V matrix and the follow-up frequency crosstab are computed.

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

**Method.** `y`-rate heatmaps: `pd.crosstab(..., values=y_binary, aggfunc="mean")` for (1) `contact` × `month` (columns ordered `month_order`) and (2) `poutcome` × `previous` binned into `["0", "1-2", "3-5", "6+"]` via `pd.cut(train["previous"], bins=[-1,0,2,5,max])`. See `MATRIX_METHODS.md` for how a `y`-rate crosstab differs from the association matrices above.

**Result.**
`contact` × `month` y-rate (selected cells): `cellular`/mar 0.532, `cellular`/sep 0.517, `cellular`/dec 0.494, `cellular`/oct 0.447, `cellular`/jun 0.439 vs. `cellular`/may (the highest-volume month) only 0.119 and `cellular`/jul 0.096 — the low-volume months (mar, sep, oct, dec) convert far better than the high-volume months (may, jul, aug ~0.10-0.12), for cellular contact. `unknown` contact in apr shows 0.833 but on very small n; `unknown` contact in jun/jul/may is near-zero (0.033-0.045), suggesting `unknown` contact method combined with peak-campaign months performs worst.

`poutcome` × `previous`(binned) y-rate: `success` dominates regardless of contact count — 0.634 (1-2 prior contacts), 0.659 (3-5), 0.673 (6+) — roughly **6x** the rate of `failure` (0.116–0.150) or `other` (0.161–0.183) at the same `previous` bin. `poutcome == unknown` with 0 previous contacts sits at 0.092 (close to the base rate), confirming `poutcome` (specifically "success") is a strong, count-independent signal, while `failure`/`other` only mildly improve with more prior attempts.

---

## Model-Based Feature Importance (RF)

**Purpose.** The relevance ranking (correlation / Cramér's V) and the multicollinearity checks are all *bivariate* — each feature is scored against the target (or against another feature) independently. The assignment's feature-importance hint (Medium article on "Feature Importance: How's and Why's") points specifically at *model-based* importance, which captures interactions and non-linearities a single-feature correlation can miss. Running a quick Random Forest gives a second, complementary view for `y` and doubles as a sanity check: do the top RF features roughly agree with the correlation-based ranking, or does the model surface something the bivariate pass missed? (The `balance` regressor's RF importance is in `REGRESSION_NOTES.md`.)

**Method.** Built a lightweight feature matrix with `pd.get_dummies` for the 9 categorical columns (`job, marital, education, default, housing, loan, contact, month, poutcome`) plus the numeric columns (`age, balance, day, duration, campaign, pdays, previous`), no sklearn pipeline (consistent with the notebook's light-weight style elsewhere). Fit a `RandomForestClassifier(n_estimators=300, random_state=42)` on `y_binary` (reusing the encoding from the relevance section). One-hot dummy columns are summed back to their parent column (`groupby` on a dummy→parent map) so a categorical feature's importance is comparable to a numeric feature's, rather than being split across its individual categories.

**Result.**

Classification (`y`) — RF importance, aggregated per feature:

| feature | importance |
|---|---|
| duration | 0.267 |
| month | 0.106 |
| balance | 0.095 |
| age | 0.092 |
| day | 0.081 |
| job | 0.070 |
| poutcome | 0.070 |
| campaign | 0.038 |
| pdays | 0.038 |
| education | 0.035 |
| marital | 0.027 |
| housing | 0.026 |
| contact | 0.022 |
| previous | 0.020 |
| loan | 0.012 |
| default | 0.002 |

**Comparison.** `duration` dominates both views (RF 0.267, correlation 0.395) — strong agreement, and a second confirmation of the leakage call-out above. `month` and `poutcome` both clear the correlation top-7 and rank in RF's top-7 too, so the two methods broadly agree on the headline features. Two notable disagreements: (1) `balance` and `age` rank 3rd/4th in RF (0.095, 0.092) despite being near the *bottom* of the correlation ranking (0.053, 0.025) — RF can pick up interaction/non-linear signal a single-feature point-biserial correlation misses, but this is also a known RF bias (continuous, high-cardinality numeric features tend to get inflated importance relative to low-cardinality categoricals, purely from having more possible split points), so this shouldn't be read as "balance/age are secretly strong predictors" without further checking; (2) `housing`, `contact`, and `pdays` rank noticeably lower in RF than in the correlation ranking — plausibly because these overlap heavily with `month` (Cramér's V 0.504 for `housing`–`month`, 0.512 for `contact`–`month`, from the categorical-redundancy deep dive), so RF's greedy splitting credits `month` and doesn't need to "reuse" the redundant signal in the others.

**Domain interpretation.** The same business mechanisms noted in the Relevance section's domain-interpretation paragraph (warm leads via `poutcome`, spare-cash constraints via `housing`/`loan`, income/stability proxy via `job`) apply here too — the RF view doesn't introduce a new business story, it just reweights confidence in the existing ones (see the comparison above for which features gain or lose credibility under RF).

---

## Relevance Cross-Check — Mutual Information

**Purpose.** Both prior feature-importance views have a gap: the correlation/Cramér's-V ranking only captures *linear* (numeric) or first-order categorical association, and the RF importance view — while it can pick up non-linearities and interactions — depends on a fitted model and is known to over-credit continuous, high-cardinality features (flagged in the RF Comparison above for `balance`/`age`). **Mutual information** closes both gaps: it's computed directly from the feature-vs-target joint distribution (no model fit involved, so it can't inherit a model's biases), and it captures non-linear dependency a linear correlation coefficient would miss. Included specifically to confirm — without touching a model — whether a feature's relationship to `y` is real.

**Method.** `sklearn.feature_selection.mutual_info_classif`, run in two passes to match each feature's dtype: numeric columns (`age, balance, day, duration, campaign, pdays, previous`) passed as raw values with `discrete_features=False`; categorical columns (`job, marital, education, default, housing, loan, contact, month, poutcome`) label-encoded to integer codes (`pd.factorize`) and passed with `discrete_features=True`, which tells sklearn to treat the codes as unordered categories rather than ordinal numbers — correct for label-encoded nominal data, and it avoids the one-hot "grouped shuffle" step the RF section needed, since MI naturally gives one score per original column. Both passes use `random_state=42`. **Scale caveat:** MI is measured in nats, not the same 0–1-ish scale as point-biserial `r` / Cramér's V — so the comparison below is about **rank agreement**, not matching numeric values.

**Result.**

| feature | mutual info | type |
|---|---|---|
| duration | 0.0709 | numeric |
| poutcome | 0.0294 | categorical |
| pdays | 0.0275 | numeric |
| month | 0.0244 | categorical |
| balance | 0.0223 | numeric |
| contact | 0.0136 | categorical |
| age | 0.0133 | numeric |
| previous | 0.0111 | numeric |
| housing | 0.0097 | categorical |
| job | 0.0083 | categorical |
| day | 0.0069 | numeric |
| campaign | 0.0066 | numeric |
| loan | 0.0026 | categorical |
| education | 0.0026 | categorical |
| marital | 0.0021 | categorical |
| default | 0.0003 | categorical |

**Comparison.** `duration` is #1 across all three methods (correlation 0.395, RF 0.267, MI 0.0709) — a third independent confirmation of the leakage call-out. `poutcome` and `month` also land in MI's top 4, agreeing with both the correlation `top_features` list and RF's top-7 — the headline non-leakage features are robust across a linear, a non-linear model-free, and a non-linear model-based method.

Two notable divergences, both informative rather than just noise:
- **`pdays` ranks much higher in MI (3rd) than in either the correlation ranking (7th, right at the 0.1 threshold) or RF (9th, 0.038).** `pdays` has step-like structure (`-1` = "never contacted" vs. an actual day count for previously-contacted clients — see the categorical-redundancy deep dive's near-total `poutcome`/`pdays==-1` overlap), which a linear point-biserial correlation understates and which RF may be discounting because `poutcome`/`previous` already carry overlapping signal via greedy splitting. MI, bivariate like correlation but non-linear, picks this up without either limitation — a case where the third method adds genuinely new information rather than just confirming the other two.
- **`balance` and `age` land in MI's top 7 (5th, 7th)**, echoing RF's elevation of the same two features (3rd, 4th) over their low correlation ranks (0.053, 0.025). Because MI is model-free, this is independent evidence that RF's high ranking for `balance`/`age` isn't *purely* the known continuous-feature/split-point bias — a second, differently-biased method finds real (if modest) non-linear signal here too. That said, both `balance` (MI 0.022) and `age` (MI 0.013) still sit well below the leakage/behavioral-signal cluster at the top (`duration`, `poutcome`, `pdays`, `month`), so this tempers rather than overturns the original correlation-based screening.
- At the bottom, `default` is the weakest feature in all three views (correlation 0.022, RF 0.002, MI 0.0003) — consistent agreement it carries essentially no signal.

---

## Deep Dive — Time/Seasonality (y)

**Purpose.** `train.csv` is date-ordered (May 2008 – Nov 2010) but the `month` column has no year, so the per-month countplot from the univariate pass collapses all 3 years together and can't show drift over the campaign's actual timeline. Binning by row order (a chronological proxy, since the data is sorted by contact date) lets us see whether `y`-rate and call volume changed over the life of the campaign — directly relevant to whether a time-based train/validation split is appropriate.

**Method.** Row index split into `N_CHUNKS = 20` equal-sized sequential bins (`pd.cut` on `np.arange(len(train))`), then per-chunk `y`-rate (share of `yes`) and call volume (`n_calls`), plotted as two stacked line/bar charts.

**Result.** Clear upward drift, not flat: `y`-rate starts around 2–5% in the earliest chunks (chunk 0: 2.1%, chunk 1: 3.8%, chunk 4: 4.6%) and rises steadily through the middle chunks (chunk 8: 6.0%, chunk 12: 4.7%) before jumping sharply in the final chunks — chunk 13: 16.2%, chunk 17: 23.8%, chunk 18: 41.2%, chunk 19 (final, most recent): **52.9%**. Call volume per chunk is roughly even (~2,260 rows each, by construction of the equal-size bins). This is a ~25x increase in conversion rate from the earliest to latest chunk of the campaign — strong evidence of real temporal drift (later-stage campaign targeting/conditions were far more effective), which supports doing a time-based train/validation split rather than a random split, and suggests any model should account for a temporal signal even without an explicit year column. This split recommendation applies to the `balance` regression track too (see `REGRESSION_NOTES.md`), since it's about row ordering, not the classification target specifically.

---

## EDA Summary / Next Steps

- **`test.csv` is a random 10% subset of `train.csv`, not an independent holdout.** Per the UCI source documentation (see `README.md`), `test.csv` rows are drawn from `train.csv` itself. This means the provided files cannot be used as a clean train/test split as-is (risk of leakage/duplicated rows if used naively) — the team needs its own split strategy. Applies equally to the `balance` regression track.
- **A time-based split is now well-supported, not just plausible.** The seasonality deep-dive shows `y`-rate rising from ~2-5% in the earliest chronological chunks to 52.9% in the latest chunk — a ~25x drift. A random split would leak this temporal signal between train and validation; a chronological split (train on earlier contacts, validate/test on later ones) is the correct choice. Alternatively, dedupe `train.csv` against `test.csv` and draw a fresh random split, but that would still need to respect the same ordering caveat.
- **`duration` is a leakage feature for the classification target `y`.** Call duration is only known after a call ends, so despite ranking highest in the relevance analysis (0.395) and the RF importance (0.267), it must be excluded (or used only for a separate "post-call" analysis) from any model meant to decide who to call *before* calling them.
- **Class imbalance in `y`:** ~11.7% "yes" — classification modeling will need to account for this (class weighting, resampling, or threshold tuning; not plain accuracy as the metric).
- **`unknown` sentinel categories** (no true missing values anywhere): `job` ~0.6%, `education` ~4.1%, `contact` ~28.8%, `poutcome` ~81.7% unknown. The categorical-redundancy deep-dive confirms `poutcome == unknown` is a near-duplicate of `pdays == -1` (36,954/36,959 "never contacted" rows have `poutcome == unknown`), so `poutcome`'s categorical detail is only informative for the ~18% of previously-contacted clients — and among those, `poutcome == success` is a very strong signal (y-rate 0.63-0.67 vs. 0.12-0.18 for failure/other at matching `previous` counts).
- **Most promising features:** `top_features = ['duration', 'poutcome', 'month', 'contact', 'housing', 'job', 'pdays']` from the relevance ranking, cross-checked with RF importance (`duration, month, balance, age, day, job, poutcome`) and now a third, model-free check via mutual information (`duration, poutcome, pdays, month, balance, contact, age`). All three methods agree `duration`, `poutcome`, and `month` are top-tier (with the `duration` leakage caveat still applying). RF and MI additionally agree with each other — independently of one another — that `balance`/`age` carry more real signal than the correlation-only ranking suggested, and MI specifically elevates `pdays` (rank 3rd by MI vs. 7th/9th by correlation/RF) thanks to its step-like never-contacted structure that linear correlation understates. The interaction deep-dive adds nuance beyond single-feature relevance: `contact == cellular` combined with low-volume months (mar, sep, oct, dec; y-rate 0.44-0.53) outperforms the same contact method in high-volume months (may, jul; y-rate ~0.10-0.12) — campaign timing interacts with contact method, not just additive effects. Also watch `contact`–`month` (Cramér's V 0.512) and `housing`–`month` (0.504) redundancy — these categorical pairs move together and may not both be needed in a model, which likely explains why both RF and MI rank `housing`/`contact` lower than the correlation-only ranking.
- **PCA on the numeric block was considered and dropped** as a secondary variance-structure check — it's target-agnostic and would mostly restate the existing numeric multicollinearity heatmap (only one notable pair, `pdays`/`previous` r=0.455) in a less direct, less interpretable form.
