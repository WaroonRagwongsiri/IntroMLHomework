# Matrix Methods — How Each One Works

Explains the mechanics behind every matrix/crosstab computed in `classify.ipynb` and `regression.ipynb`, so the "Method" lines in `CLASSIFY_NOTES.md` / `REGRESSION_NOTES.md` don't have to re-derive the math each time. Adapted from `banking_dataset/MATRIX_METHODS.md` — same underlying statistics, applied to this dataset's columns. Referenced from both notes files wherever a matrix is used.

---

## Pearson Correlation Matrix (numeric feature-to-feature)

**Used in:** `Multicollinearity (numeric feature-to-feature)` — identical section in both notebooks.

**What it measures.** The strength and direction of the *linear* relationship between two continuous variables.

**Formula.** For two columns `X`, `Y`:

```
r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √( Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)² )
```

i.e. covariance of `X` and `Y`, normalized by the product of their standard deviations. This is exactly what `df[numeric_cols].corr()` computes for every column pair.

**Reading the numbers.**
- Range is **-1 to +1**. Sign = direction. Magnitude = strength, regardless of sign.
- `r = 0` doesn't mean "no relationship," only "no *linear* relationship."
- **How the matrix is built:** compute `r` for every pair of numeric columns (`Reviews_num`, `Size_num`, `Price_num`, plus `Installs_num` in the regression notebook, plus `Rating` where relevant) → an NxN table, symmetric with 1.0 on the diagonal.

**Caveat.** Only captures linear/monotonic-ish structure, is sensitive to outliers (`Reviews_num` and `Installs_num` are both heavy-tailed — max values in the tens of millions against medians in the low thousands), and says nothing about causation.

---

## Point-Biserial Correlation (numeric feature vs. binary target)

**Used in:** `Relevance vs. Classification Target (hit) — Correlation First` (classify only) — the numeric half of the relevance ranking.

**What it is.** Mathematically identical to Pearson `r` above, just applied to a continuous variable and a binary variable (`hit` encoded 0/1). Not a matrix (one score per numeric feature vs. the target), but included here since it's the same underlying math as the correlation matrix above and is easy to confuse with Cramér's V (which handles the *categorical* features in the same ranking).

**Dataset-specific wrinkle.** `Size_num` and `Rating` both carry NaNs at this stage (`Rating` always; `Size_num` for `"Varies with device"` rows) — `scipy.stats.pointbiserialr` can't handle NaN, so the notebook's `point_biserial` helper masks to non-missing rows for that column before computing. This means each numeric feature's score is computed over a slightly different row count (see `relevance_df` — `Rating`'s score is over 8,892 rows, `Size_num`'s over 8,831, the rest over the full 10,357). A deviation from `banking_dataset`'s cleaner, NaN-free numeric columns.

---

## Correlation Ratio η (categorical feature vs. numeric target)

**Used in:** `Regression EDA — Primary Target: Rating` (regression only) — the categorical half of the relevance ranking.

**What it measures.** How much of the total variance in a numeric target is explained by which category a row falls into — the categorical analogue of Pearson `r`, used because Pearson requires two numeric variables and categories have no numeric order.

**Formula.**

```
η = √( SS_between / SS_total )
```

where `SS_between` is the variance explained by group membership (each group's deviation from the grand mean, weighted by group size) and `SS_total` is the total variance in the target. Implemented as the `correlation_ratio` helper (`groupby("cat")["val"].agg(["mean", "count"])`, then the weighted-sum-of-squares ratio above).

**Reading the numbers.** Range **0 to 1**, always non-negative. 0 = group membership explains none of the target's variance, 1 = groups are perfectly separated (every row in a group shares the exact same target value).

**Caveat — small-sample inflation.** η is a known-biased statistic on sparse categories: a category with very few rows can post an extreme group mean by chance alone, inflating `SS_between`. This is exactly why `Genres` (115 categories in the regression subset, some with single-digit row counts) scores higher than `Category` (33 categories, min group size 42 in the regression subset) despite `Genres` largely being a finer-grained restatement of `Category` (see the `Genres` vs. `Category` Cramér's V of 0.969 in the classify notebook's categorical redundancy check) — the extra score is partly small-group noise, not extra genuine signal. `Category`'s score is treated as the trustworthy one; `Genres`'s is flagged, not taken at face value.

---

## Cramér's V Matrix (categorical feature-to-feature / feature-to-target)

**Used in:** `Relevance vs. Classification Target (hit)` (classify, categorical half) and `Deep Dive — Categorical-Categorical Redundancy (hit)` (classify only).

**What it measures.** Association strength between two *categorical* variables — Pearson correlation doesn't apply since categories have no numeric order.

**How it's built, step by step:**
1. Build a **contingency table** — `pd.crosstab(colA, colB)` — counting how many rows fall into each combination of categories.
2. Run a **chi-square test of independence** on that table (`χ²`).
3. **Normalize** into a 0–1 scale:

```
V = √( (χ² / n) / min(r - 1, k - 1) )
```

where `n` = total row count, `r`/`k` = number of categories in each column.

**Reading the numbers.** Range **0 to 1**. 0 = no detectable association, 1 = perfect association. **How the matrix is built:** run pairwise over `Category`, `Type`, `Content Rating` → a 3x3 table, diagonal set to `1.0` by convention.

**Dataset-specific wrinkle — `Genres` excluded from the matrix.** At 119 categories, a full pairwise Cramér's V matrix including `Genres` would be both unreadable as a heatmap and slow to compute (each cell re-runs a chi-square test over a much larger contingency table). Instead, `Genres`'s redundancy with the other three categorical columns is checked individually as a small side-table — same statistic, just not folded into the square heatmap. This mirrors `banking_dataset/MATRIX_METHODS.md`'s note that a follow-up crosstab sometimes exists alongside the matrix rather than replacing it, just applied here for a cardinality reason rather than a cross-dtype reason.

**Caveat.** Cramér's V can be inflated on small samples or tables with many sparsely-filled categories — the same caveat that applies to η above, and part of why `Genres`'s very high score against `Category` (0.969) should be read as "these two columns are near-duplicates" rather than "these two columns are both independently important."

---

## Quick reference

| matrix / table | pair type | scale | symmetric? | tells you |
|---|---|---|---|---|
| Pearson correlation matrix | numeric × numeric | -1 to +1 | yes | linear relationship strength + direction |
| Point-biserial (relevance ranking) | numeric × binary target (`hit`) | -1 to +1 (reported as \|r\|) | n/a (vector, not matrix) | linear relationship strength |
| Correlation ratio η (relevance ranking) | categorical × numeric target (`Rating`) | 0 to 1 | n/a (vector, not matrix) | share of target variance explained by group membership — biased upward on sparse categories |
| Cramér's V matrix | categorical × categorical | 0 to 1 | yes | association strength only (no direction) |
