# Matrix Methods — How Each One Works

Explains the mechanics behind every matrix/crosstab computed in `classify.ipynb` and `regression.ipynb`, so the "Method" lines in `CLASSIFY_NOTES.md` / `REGRESSION_NOTES.md` don't have to re-derive the math each time. Adapted from `google_play_data/MATRIX_METHODS.md` (itself adapted from `banking_dataset/MATRIX_METHODS.md`) — same four underlying statistics, applied to this dataset's columns. Referenced from both notes files wherever a matrix is used.

---

## Pearson Correlation Matrix (numeric feature-to-feature)

**Used in:** `Multicollinearity (numeric feature-to-feature)` — both notebooks (classify's `numeric_cols`, 23 columns; regression's `numeric_cols_full` = `numeric_cols` + `Income`, 23 columns).

**What it measures.** The strength and direction of the *linear* relationship between two continuous variables.

**Formula.** For two columns `X`, `Y`:

```
r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √( Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)² )
```

i.e. covariance of `X` and `Y`, normalized by the product of their standard deviations. This is exactly what `df[numeric_cols].corr()` computes for every column pair.

**Reading the numbers.**
- Range is **-1 to +1**. Sign = direction. Magnitude = strength, regardless of sign.
- `r = 0` doesn't mean "no relationship," only "no *linear* relationship."
- **How the matrix is built:** compute `r` for every pair of the ~23 numeric candidates (`Income`, `Age`, `Recency`, `Kidhome`, `Teenhome`, `Customer_Tenure_Days`, the six `Mnt*` spend columns, the five purchase-channel/engagement columns, `Complain`, and `AcceptedCmp1`-`5`) → a large NxN table, symmetric with 1.0 on the diagonal.

**Dataset-specific wrinkle — a genuinely larger matrix than either prior dataset.** `google_play_data` used 4-5 numeric columns, `banking_dataset` used 7 — this dataset has ~23 real numeric candidates (People/Products/Place/Promotion columns combined), so the heatmap is rendered larger (`figsize=(14, 12)` / `(15, 13)`) and is denser to read. Several genuinely strong pairs show up as a result (see `CLASSIFY_NOTES.md`/`REGRESSION_NOTES.md`'s Multicollinearity sections) that a smaller feature set wouldn't have surfaced.

**Caveat.** Only captures linear/monotonic-ish structure and says nothing about causation. The `Mnt*` columns are all right-skewed with a handful of high-spend outliers (see `Distribution Check`), which Pearson `r` is somewhat sensitive to.

---

## Point-Biserial Correlation (numeric feature vs. binary target)

**Used in:** `Relevance vs. Classification Target (Response) — Correlation First` (classify only) — the numeric half of the relevance ranking.

**What it is.** Mathematically identical to Pearson `r` above, just applied to a continuous variable and a binary variable (`Response` encoded 0/1). Not a matrix (one score per numeric feature vs. the target), but included here since it's the same underlying math as the correlation matrix above.

**Dataset-specific wrinkle.** `Income` carries 24 NaNs at this stage (not dropped in `classify.ipynb`, since `Response` doesn't depend on `Income`) — `scipy.stats.pointbiserialr` can't handle NaN, so the notebook's `point_biserial` helper masks to non-missing rows for that column before computing. Every other numeric candidate is complete, so only `Income`'s score is computed over a slightly smaller row count (2,213 of 2,237) than the rest.

---

## Correlation Ratio η (categorical feature vs. numeric target)

**Used in:** `Regression EDA — Primary Target: Income` (regression only) — the categorical half of the relevance ranking.

**What it measures.** How much of the total variance in a numeric target is explained by which category a row falls into — the categorical analogue of Pearson `r`, used because Pearson requires two numeric variables and categories have no numeric order.

**Formula.**

```
η = √( SS_between / SS_total )
```

where `SS_between` is the variance explained by group membership (each group's deviation from the grand mean, weighted by group size) and `SS_total` is the total variance in the target. Implemented as the `correlation_ratio` helper (`groupby("cat")["val"].agg(["mean", "count"])`, then the weighted-sum-of-squares ratio above).

**Reading the numbers.** Range **0 to 1**, always non-negative. 0 = group membership explains none of the target's variance, 1 = groups are perfectly separated.

**Dataset-specific note — no small-sample inflation risk here, unlike `google_play_data`.** `google_play_data/MATRIX_METHODS.md` flags η as biased on sparse categories (`Genres`, 115 categories, some single-digit group sizes). This dataset only has two categorical candidates, `Education` (5 categories, smallest group `Basic` at 54 rows) and `Marital_Status` (5 categories after cleaning, smallest group `Widow` at 77 rows) — every group is comfortably large, so both η scores here (`Education` = 0.218, `Marital_Status` = 0.046) are trustworthy at face value, not flagged.

---

## Cramér's V (categorical feature vs. binary target / categorical-categorical redundancy)

**Used in:** `Relevance vs. Classification Target (Response)` (classify, categorical half) and `Categorical Redundancy (Education vs. Marital_Status)` (both notebooks, shared foundation section).

**What it measures.** Association strength between two *categorical* variables — Pearson correlation doesn't apply since categories have no numeric order.

**How it's built, step by step:**
1. Build a **contingency table** — `pd.crosstab(colA, colB)` — counting how many rows fall into each combination of categories.
2. Run a **chi-square test of independence** on that table (`χ²`).
3. **Normalize** into a 0–1 scale:

```
V = √( (χ² / n) / min(r - 1, k - 1) )
```

where `n` = total row count, `r`/`k` = number of categories in each column.

**Reading the numbers.** Range **0 to 1**. 0 = no detectable association, 1 = perfect association.

**Dataset-specific wrinkle — no heatmap needed.** With only two categorical candidates (`Education`, `Marital_Status`), the "Categorical Redundancy" section that used a full pairwise heatmap in `google_play_data` (3x3) and `banking_dataset` (9x9) collapses to a single number here: `cramers_v(Education, Marital_Status) = 0.043`, computed identically in both notebooks (shared foundation, duplicated so each notebook runs standalone).

---

## Quick reference

| matrix / table | pair type | scale | symmetric? | tells you |
|---|---|---|---|---|
| Pearson correlation matrix | numeric × numeric | -1 to +1 | yes | linear relationship strength + direction |
| Point-biserial (relevance ranking) | numeric × binary target (`Response`) | -1 to +1 (reported as \|r\|) | n/a (vector, not matrix) | linear relationship strength |
| Correlation ratio η (relevance ranking) | categorical × numeric target (`Income`) | 0 to 1 | n/a (vector, not matrix) | share of target variance explained by group membership — not flagged here, both categories are large enough |
| Cramér's V | categorical × categorical, or categorical × binary target | 0 to 1 | yes (feature-to-feature) | association strength only (no direction) |
