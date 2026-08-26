# Matrix Methods — How Each One Works

Explains the mechanics behind every matrix/crosstab computed in `classify.ipynb` and `regression.ipynb`, so the "Method" lines in `CLASSIFY_NOTES.md` / `REGRESSION_NOTES.md` don't have to re-derive the math each time. Referenced from both notes files wherever a matrix is used.

---

## Pearson Correlation Matrix (numeric feature-to-feature)

**Used in:** `Multicollinearity (numeric feature-to-feature)` — identical section in both notebooks.

**What it measures.** The strength and direction of the *linear* relationship between two continuous variables.

**Formula.** For two columns `X`, `Y`:

```
r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √( Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)² )
```

i.e. covariance of `X` and `Y`, normalized by the product of their standard deviations. This is exactly what `train[numeric_cols].corr()` computes for every column pair.

**Reading the numbers.**
- Range is **-1 to +1**. Sign = direction (positive → both rise together; negative → one rises as the other falls). Magnitude = strength, regardless of sign.
- `r = 0` doesn't mean "no relationship," only "no *linear* relationship" — a strong curved relationship can still score near 0 (this is exactly what happened with `age` vs. `balance`: raw `r = 0.098` looked weak, but the age-binned deep dive found a real, monotonic, convex-increasing trend underneath).
- **How the matrix is built:** compute `r` for every pair of the 7 numeric columns → a 7×7 table. It's symmetric (`corr(A,B) == corr(B,A)`) with 1.0 on the diagonal (every column perfectly correlates with itself), so really only the 21 off-diagonal cells above (or below) the diagonal carry information — the heatmap shows all of them for readability.

**Caveat.** Only captures linear/monotonic-ish structure, is sensitive to outliers (this dataset has heavy-tailed columns like `balance`, `duration`, `pdays`), and says nothing about causation — two features can correlate because they're both driven by a third factor (e.g. campaign timing) rather than one causing the other.

---

## Point-Biserial Correlation (numeric feature vs. binary target)

**Used in:** `Relevance vs. Classification Target (y)` (classify only) — the numeric half of the relevance ranking.

**What it is.** Mathematically identical to Pearson `r` above, just applied to a continuous variable and a binary variable (`y` encoded 0/1). It's not a new formula — "point-biserial" is just the name statisticians use when one of the two variables is binary. Not a matrix (it's one score per numeric feature vs. the target, i.e. a vector, not a feature-to-feature grid), but included here since it's the same underlying math as the correlation matrix above and is easy to confuse with Cramér's V (which handles the *categorical* features in the same ranking).

---

## Cramér's V Matrix (categorical feature-to-feature redundancy)

**Used in:** `Deep Dive — Categorical-Categorical Redundancy (y)` (classify only) and the categorical half of the `Relevance` ranking (classify only).

**What it measures.** Association strength between two *categorical* variables — Pearson correlation doesn't apply here since categories have no numeric order (e.g. `job` categories can't be meaningfully subtracted from each other).

**How it's built, step by step:**
1. Build a **contingency table** — `pd.crosstab(colA, colB)` — counting how many rows fall into each combination of categories.
2. Run a **chi-square test of independence** on that table. Chi-square (`χ²`) measures how far the *observed* cell counts are from the counts you'd *expect* if the two columns were completely independent (i.e. if the row category told you nothing about the column category). Bigger deviation → bigger `χ²` → stronger association.
3. **Normalize** `χ²` into a 0–1 scale so different-shaped tables (different numbers of categories, different sample sizes) become comparable:

```
V = √( (χ² / n) / min(r - 1, k - 1) )
```

where `n` = total row count, `r`/`k` = number of categories in each column (the table's row/column counts).

**Reading the numbers.**
- Range is **0 to 1**, always non-negative — unlike Pearson `r`, there's no "direction" to a categorical association (what would "negative" mean for `job` vs. `education`?), only strength.
- 0 = no detectable association, 1 = perfect association (knowing one column fully determines the other).
- **How the matrix is built:** run this pairwise for every pair among the 9 categorical columns → a 9×9 table, diagonal set to `1.0` by convention (a column is trivially "perfectly associated" with itself, computing `V` against itself would just be a degenerate case of the same formula).

**Caveat.** Cramér's V can be inflated on small samples or tables with many sparsely-filled categories (a known small-sample bias in the chi-square statistic it's built on) — worth treating high scores on rare categories (e.g. `default == yes`, only ~1.8% of rows) with a bit more skepticism than high scores on well-populated categories.

---

## y-rate Crosstab Heatmaps (interaction effects)

**Used in:** `Deep Dive — Interaction Effects (y)` (classify only) — `contact × month` and `poutcome × previous` (binned).

**What it is.** Not a feature-to-feature association matrix like the two above — this is a **pivot table**: rows = categories of one feature, columns = categories of another, and each **cell** = the *mean of `y_binary`* for all rows matching that (row-category, column-category) combination:

```python
pd.crosstab(train["contact"], train["month"], values=y_binary, aggfunc="mean")
```

Each cell is literally `P(y == "yes")` estimated within that specific slice of the data — e.g. the `cellular`/`mar` cell (0.532) means "of clients contacted by cellular in March, 53.2% subscribed." This is why it can reveal **interactions**: two features can each look weak alone (checked independently via the relevance ranking) yet have specific *combinations* that are unusually strong or weak — invisible to a single-feature view, but visible once you condition on both at once.

**Caveat.** Cells built from few rows (e.g. a rare contact-method/month combination) can show extreme rates purely from small-sample noise — the notes flag this explicitly where it happens (e.g. `unknown` contact in `apr`, 0.833 rate, called out as "small n").

---

## `poutcome` vs. `pdays == -1` Crosstab (plain frequency table)

**Used in:** `Deep Dive — Categorical-Categorical Redundancy (y)` (classify only), as a follow-up to the Cramér's V matrix.

**What it is.** The simplest kind of crosstab — just row/column **counts**, no aggregation function:

```python
pd.crosstab(train["poutcome"], train["pdays"] == -1)
```

Each cell is the number of rows with that combination of `poutcome` category and "never contacted" flag. Used here (rather than another Cramér's V score) because the goal was to show the *exact* overlap in plain numbers — "36,954 of 36,959 never-contacted rows have `poutcome == unknown`" is more concrete and checkable than a single 0–1 association score, even though the Cramér's V matrix above already flagged this pair as one of the highest (0.214 for `month`–`poutcome`, cross-dtype pairs like `poutcome`–`pdays` aren't even representable in a same-dtype Cramér's V matrix, which is exactly why this follow-up crosstab exists).

---

## Quick reference

| matrix / table | pair type | scale | symmetric? | tells you |
|---|---|---|---|---|
| Pearson correlation matrix | numeric × numeric | -1 to +1 | yes | linear relationship strength + direction |
| Point-biserial (relevance ranking) | numeric × binary target | -1 to +1 (reported as \|r\|) | n/a (vector, not matrix) | linear relationship strength |
| Cramér's V matrix | categorical × categorical | 0 to 1 | yes | association strength only (no direction) |
| y-rate crosstab heatmap | categorical × categorical (or binned numeric), aggregated | 0 to 1 (a rate) | no | conditional outcome rate per combination — surfaces interactions |
| plain frequency crosstab | categorical × boolean/categorical | raw counts | no | exact joint distribution, for spot-checking a suspected redundancy |
