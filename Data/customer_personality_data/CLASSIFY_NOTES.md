# EDA Notes — Customer Personality Analysis (Classification track: Response)

Narrative/discussion companion to `classify.ipynb`. The notebook keeps only short section headers; the "why," the exact method, and the numeric results for each section live here. Section headers below match the notebook's markdown headers 1:1.

Target: **`Response`** (accepted the last marketing campaign, 0/1) — a raw column, not engineered. This is the dataset's headline attraction: unlike `google_play_data` (needed an engineered `hit` target) and `banking_dataset` (raw `y`, but no natural regression target in the same file), this dataset ships **both** a natural classification target (`Response`) and a natural regression target (`Income`) in one CSV. See `REGRESSION_NOTES.md` for the `Income` track — both notebooks share the same "Import Lib + Load Data," "Data Cleaning," "EDA," and "Distribution Check" foundation sections, duplicated so each notebook runs standalone. See `MATRIX_METHODS.md` for the math behind every correlation/redundancy matrix used below.

This is a **backup dataset** (see `README.md`), scoped lighter than `banking_dataset/` — no Interaction Effects or Time/Seasonality deep dives here (`Customer_Tenure_Days` is a light nod to time, not a full seasonality track); the sections kept are the ones that carried real weight in the two prior EDA passes.

---

## Data Cleaning

**Purpose.** Unlike `banking_dataset` (no missing values, no corrupted rows) but similar to `google_play_data`, this dataset needs real cleaning before modeling — worth documenting as its own section.

**Method / Result, in the order applied:**
1. **Drop `ID`** — a unique identifier, not a feature. 0 duplicate IDs found (2,240 unique customers). Shape: 2,240 → 2,240 rows, 29 → 28 columns.
2. **Drop `Z_CostContact` and `Z_Revenue`** — confirmed constant (`Z_CostContact` always `3`, `Z_Revenue` always `11`, printed explicitly before dropping) — zero variance, no predictive content. Shape: 28 → 26 columns.
3. **Parse `Dt_Customer`** to datetime (`%d-%m-%Y`, confirmed range 2012-07-30 to 2014-06-29) and derive **`Customer_Tenure_Days`** = days between each row's enrollment date and the dataset's latest enrollment date (2014-06-29) — a simple derived feature in place of the raw date string, same "derived, not raw" pattern used for `Installs`/`Size`/`Price` in `google_play_data`. Mean tenure 353.6 days, range 0–699.
4. **Derive `Age`** = 2014 (the latest `Dt_Customer` year) − `Year_Birth`. Three rows carry an implausible `Year_Birth` (1893, 1899, 1900 → age 121, 115, 114) — genuine data-entry errors, not a real elderly-customer segment. Dropped outright (small n, no reasonable imputation target). Shape: 2,240 → **2,237 rows**.
5. **Collapse `Marital_Status` joke/near-empty categories.** Before: `Married` 864, `Together` 579, `Single` 479, `Divorced` 231, `Widow` 77, `Alone` 3, `Absurd` 2, `YOLO` 2. `Absurd` and `YOLO` are joke entries; `Alone` is a genuine status but semantically identical to `Single` and too small to model on its own. All three folded into `Single`. After: `Single` 479 → **486**, five categories total.
6. **`Education`** needed no cleaning: `Graduation` 1,127, `PhD` 485, `Master` 370, `2n Cycle` 201, `Basic` 54 — five clean categories, no missing values.
7. **`Income` missing (24 rows) is deliberately kept, not dropped, in this notebook** — `Response` does not depend on `Income`, so imputing or dropping is unnecessary here; the relevance ranking below masks to non-missing rows for `Income` only (see `MATRIX_METHODS.md`). Contrast with `regression.ipynb`, where these 24 rows are dropped because `Income` is the target itself.

**Result.** Final cleaned shape: **2,237 rows x 28 columns**. Of those 28: 1 is the target (`Response`), 2 are raw columns kept but not used directly (`Year_Birth`, `Dt_Customer` — superseded by the derived `Age`/`Customer_Tenure_Days`), leaving **25 candidate features** for the relevance ranking below. `Response` class balance: **14.9% positive / 85.1% negative** — meaningfully imbalanced, comparable in spirit to `banking_dataset`'s `y` (11.7%/88.3%) but a bit gentler, in the same range as `google_play_data`'s `hit` was gentler still (39.1%/60.9%).

---

## Relevance vs. Classification Target (Response) — Correlation First

**Purpose.** Same reasoning as both prior datasets: with 25 candidate features (23 numeric, 2 categorical — by far the largest candidate pool of the three datasets explored), a single relevance ranking upfront avoids a wall of uninformative univariate plots.

**Method.** Two measures depending on feature type (see `MATRIX_METHODS.md`):
- Numeric (`Income`, `Age`, `Recency`, `Kidhome`, `Teenhome`, `Customer_Tenure_Days`, the six `Mnt*` spend columns, the five purchase-channel/engagement columns, `Complain`, `AcceptedCmp1`-`5`): point-biserial correlation with `Response` (0/1), `Income` masked to non-missing rows.
- Categorical (`Education`, `Marital_Status`): Cramér's V from the raw-label contingency table.

**`AcceptedCmp1`-`5` are deliberately included, not excluded** — flagged explicitly below as a strong historical predictor worth a closer look, same treatment as `duration` (`banking_dataset`) and `Reviews_num` (`google_play_data`): shown in the ranking, not silently dropped.

**Result.** Full ranking (relevance score, descending; 25 candidates):

| feature | score | type |
|---|---|---|
| AcceptedCmp5 | 0.328 | numeric |
| AcceptedCmp1 | 0.294 | numeric |
| AcceptedCmp3 | 0.254 | numeric |
| MntWines | 0.247 | numeric |
| MntMeatProducts | 0.237 | numeric |
| NumCatalogPurchases | 0.221 | numeric |
| Recency | 0.199 | numeric |
| Customer_Tenure_Days | 0.194 | numeric |
| AcceptedCmp4 | 0.177 | numeric |
| AcceptedCmp2 | 0.169 | numeric |
| Teenhome | 0.155 | numeric |
| Marital_Status | 0.152 | categorical |
| NumWebPurchases | 0.148 | numeric |
| MntGoldProds | 0.141 | numeric |
| Income | 0.133 | numeric |
| MntFruits | 0.126 | numeric |
| MntSweetProducts | 0.117 | numeric |
| MntFishProducts | 0.111 | numeric |
| Education | 0.102 | categorical |
| Kidhome | 0.080 | numeric |
| NumStorePurchases | 0.039 | numeric |
| Age | 0.018 | numeric |
| NumWebVisitsMonth | 0.004 | numeric |
| NumDealsPurchases | 0.002 | numeric |
| Complain | 0.0002 | numeric |

At threshold 0.1, **19 of 25** candidates clear it — a far higher hit rate than either prior dataset's relevance ranking, consistent with this being the strongest bivariate signal set explored so far. Only `Kidhome`, `NumStorePurchases`, `Age`, `NumWebVisitsMonth`, `NumDealsPurchases`, and `Complain` don't clear it.

**Not leakage — `AcceptedCmp1`-`5`.** These flag acceptance of the five *earlier* campaigns. A customer who accepted a prior campaign is mechanically more likely to be a responsive, engaged customer — the same kind of customer likely to accept the *current* campaign (`Response`) too. This is not leakage: it isn't a feature mathematically/causally derived from `Response` (unlike `Installs_num` → `hit` in `google_play_data`, where the target is a deterministic function of the feature), nor does it draw on information from the future — the five earlier campaigns are, by definition, always resolved before `Response`'s campaign runs, so for an existing customer this history is genuinely available at prediction time. It's a real, strong behavioral proxy, with one caveat: a brand-new prospect with no campaign history wouldn't have this feature populated, a feature-availability/cold-start issue to check for at deployment, not a leakage concern. The three of the top four bivariate features are `AcceptedCmp5`/`1`/`3` — this ranking is dominated by this prior-campaign signal at the very top. Concretely: **Response rate given any prior campaign acceptance is 40.7% (n=462), vs. 8.2% given none (n=1,775)** — a 5x gap, the starkest split in the whole ranking.

**Domain interpretation.**
- **Spend columns (`MntWines`, `MntMeatProducts`, `MntGoldProds`, `MntFruits`, `MntSweetProducts`, `MntFishProducts`) and `NumCatalogPurchases`** all clear the threshold — customers who spend more, and who buy through the catalog channel specifically, are more likely to respond to a new campaign. Plausible: catalog buyers are already primed for marketing-driven purchases, unlike walk-in `NumStorePurchases` shoppers (which does *not* clear the threshold, 0.039).
- **`Recency`** (days since last purchase, 0.199) — customers who bought more recently are more likely to respond, a standard RFM-style marketing signal.
- **`Customer_Tenure_Days`** (0.194) — longer-tenured customers respond more, consistent with a loyalty/engagement story.
- **`Teenhome`** (0.155) clears the threshold but `Kidhome` (0.080) doesn't — households with teenagers respond more than households with young children, a plausible but non-obvious asymmetry worth flagging rather than assuming symmetric "has kids" effects.
- **`Income`** (0.133) clears the threshold on its own, distinct from its role as the other track's target — higher-income customers respond somewhat more.
- **`Marital_Status`** (0.152) clears the threshold, **`Education`** (0.102) barely clears it — both weaker than the strongest numeric signals.

---

## Multicollinearity (numeric feature-to-feature)

**Purpose.** Same as both prior datasets: a separate check for redundancy *among* features, independent of the relevance ranking above. (Near-identical section also appears in `REGRESSION_NOTES.md`, with `Income` swapped from a numeric candidate into the target — target-agnostic, duplicated as shared foundation.)

**Method.** Pearson correlation matrix over all 23 numeric candidates. See `MATRIX_METHODS.md` — the largest matrix of the three datasets explored (23x23 vs. `banking_dataset`'s 7x7 and `google_play_data`'s 4x4/5x5).

**Result.** Substantially more multicollinearity than either prior dataset. Strongest pairs:

| pair | r |
|---|---|
| MntMeatProducts vs. NumCatalogPurchases | 0.724 |
| MntWines vs. NumStorePurchases | 0.642 |
| MntWines vs. NumCatalogPurchases | 0.635 |
| MntFruits vs. MntFishProducts | 0.594 |
| Income vs. NumCatalogPurchases | 0.589 |
| Income vs. MntMeatProducts | 0.584 |
| MntFishProducts vs. MntSweetProducts | 0.580 |
| Income vs. MntWines | 0.578 |
| MntMeatProducts vs. MntFishProducts | 0.568 |
| Income vs. NumWebVisitsMonth | -0.553 |

**Implication.** `Income`, the six `Mnt*` spend columns, and the purchase-channel counts (especially `NumCatalogPurchases`, `NumStorePurchases`) largely move together — they all describe one underlying "affluent, high-spending, catalog/store-shopping" customer profile, not independent dimensions. `NumWebVisitsMonth` is the interesting exception: it's *negatively* correlated with `Income` and with spend (-0.553 with `Income`, -0.539 with `MntMeatProducts`) — high-income/high-spend customers browse the website less, plausibly because they buy through catalog or in-store channels instead. This redundancy is a real modeling consideration (RF/regularized-model feature selection would need to account for it) distinct from the relevance ranking above, which only measures each feature's relationship to `Response` in isolation.

---

## Categorical Redundancy (Education vs. Marital_Status)

**Purpose.** Same rationale as the categorical-categorical redundancy checks in both prior datasets — but with only two categorical candidates here (far fewer than `google_play_data`'s four or `banking_dataset`'s nine), a full heatmap is unnecessary; a single pairwise score suffices. See `MATRIX_METHODS.md`.

**Result.** `Cramér's V(Education, Marital_Status) = 0.043` — essentially independent. Unlike `google_play_data`'s `Genres`/`Category` pair (0.969, near-duplicate columns), there's no redundancy concern here: both categorical features can be treated as carrying separate signal.

---

## Model-Based Feature Importance (RF)

**Purpose.** Same rationale as both prior datasets: the relevance ranking and multicollinearity checks are bivariate; a Random Forest captures interactions and non-linearities a single-feature correlation can miss.

**Method.** Built a feature matrix with `pd.get_dummies` for `Education`/`Marital_Status` plus all 23 numeric columns (median-filled for `Income`, the only column with NaNs at this stage — a modeling-only convenience, not applied elsewhere). Fit `RandomForestClassifier(n_estimators=300, random_state=42)` on `Response`. Dummy columns summed back to their parent column.

**Result (top 15 of 25, aggregated importance):**

| feature | importance |
|---|---|
| Recency | 0.088 |
| Customer_Tenure_Days | 0.082 |
| Income | 0.072 |
| MntWines | 0.070 |
| MntMeatProducts | 0.069 |
| MntGoldProds | 0.050 |
| Marital_Status | 0.046 |
| Age | 0.045 |
| AcceptedCmp3 | 0.042 |
| MntSweetProducts | 0.042 |
| MntFishProducts | 0.040 |
| AcceptedCmp5 | 0.040 |
| MntFruits | 0.039 |
| NumStorePurchases | 0.039 |
| NumCatalogPurchases | 0.038 |

**A genuinely different pattern from both prior datasets — the strong historical-predictor features are *demoted*, not amplified, by RF.** In `google_play_data`, RF importance *concentrated* onto the actual leakage feature `Reviews_num` (0.690 of total, 6x the runner-up) relative to its mid-table bivariate rank — the nonlinear model exploited exactly the leakage relationship. Here the opposite happens: `AcceptedCmp5` — the #1 bivariate feature (0.328) — falls to 12th in RF importance (0.040), and `AcceptedCmp1`/`AcceptedCmp3`/`AcceptedCmp4` similarly drop out of the top ranks. Instead, RF elevates `Recency` and `Customer_Tenure_Days` (7th and 8th bivariate, now 1st and 2nd) to the top. A plausible reading: the `AcceptedCmp*` flags are highly correlated with each other and with `Response` in a fairly linear, additive way that point-biserial correlation already captures efficiently — RF has little nonlinear structure left to exploit there once the (many) spend/purchase/tenure features are available to substitute for the same underlying "engaged customer" signal. This is a useful contrast with `google_play_data`'s genuine leakage case: `AcceptedCmp*` is not leakage (it's real, temporally-valid prior-campaign history, always available before `Response`'s campaign), and RF's demotion here is a multicollinearity/redundancy effect, not evidence about whether the feature is legitimate.

---

## Relevance Cross-Check — Mutual Information

**Purpose.** A third, non-parametric lens on relevance, to see whether the RF-based re-ranking above is an RF-specific artifact or a broader pattern.

**Method.** `sklearn.feature_selection.mutual_info_classif` — numeric columns (median-filled, reusing the RF feature matrix) with `discrete_features=False`, categorical columns (label-encoded via `pd.factorize`) with `discrete_features=True`.

**Result (top 15 of 25):**

| feature | mutual_info | type |
|---|---|---|
| MntMeatProducts | 0.043 | numeric |
| AcceptedCmp5 | 0.042 | numeric |
| MntWines | 0.040 | numeric |
| Income | 0.037 | numeric |
| MntGoldProds | 0.034 | numeric |
| NumCatalogPurchases | 0.034 | numeric |
| Customer_Tenure_Days | 0.032 | numeric |
| NumWebPurchases | 0.032 | numeric |
| AcceptedCmp3 | 0.027 | numeric |
| Recency | 0.025 | numeric |
| AcceptedCmp1 | 0.024 | numeric |
| Age | 0.024 | numeric |
| AcceptedCmp2 | 0.022 | numeric |
| NumStorePurchases | 0.022 | numeric |
| MntFruits | 0.020 | numeric |

**A genuine three-way disagreement, reported honestly rather than resolved into one story.** MI puts `AcceptedCmp5` back in 2nd place (0.042, close behind `MntMeatProducts`'s 0.043) — a real signal MI still picks up strongly, unlike RF's 12th-place demotion of the same feature. `Recency`/`Customer_Tenure_Days` — RF's top two — rank only 7th/7th-adjacent here (0.025 and 0.032). `Marital_Status` (0.011) and `Education` (0.005), both comfortably above threshold bivariately, are near the bottom under MI. The honest summary: bivariate correlation, RF, and MI each tell a *different* story about how much weight the `AcceptedCmp*` prior-campaign features deserve relative to `Recency`/`Customer_Tenure_Days`/spend — no single method should be taken as the final word. These features aren't leakage (the history is genuinely available at prediction time for an existing customer), so a model built on this data can include `AcceptedCmp1`-`5`, with the one real caveat being availability for brand-new prospects with no campaign history, not any importance score.

---

## Feature Selection — Combining the Three Methods (Final Cut)

**Purpose.** The three methods above measure genuinely different things — linear/monotonic association (bivariate), tree-based interaction-aware importance (RF), and general statistical dependency (MI) — and each can be misled in its own way (small-sample noise for all three; RF specifically via greedy splitting on correlated features, see the RF section above). This section states the actual rule used to collapse three separate rankings into one final feature-selection decision, and applies it to every one of the 25 candidates.

**Method — the cutting rule, not an average.** The three scores live on incompatible scales: bivariate is signed, −1 to +1; RF importance is unsigned, 0 to 1, and sums to exactly 1 across all candidates; MI is unsigned, 0 to ∞ with no fixed ceiling. Averaging them would let whichever method happens to have the largest numeric range dominate a blended score for no principled reason. Instead, each method casts an independent pass/fail vote per feature:

- **Bivariate passes** if `|score| ≥ 0.10`.
- **RF passes** if the feature lands in RF's top 15 of 25.
- **MI passes** if the feature lands in MI's top 15 of 25.

Features are then tiered by vote count:

- **3/3 (full consensus) → chosen feature, no caveat** (unless separately flagged for a deployment-availability caveat — see below). A signal that survives three different statistical assumptions at once is very unlikely to be a fluke of one method or of this dataset's modest size (2,237 rows).
- **2/3 (lower-confidence signal) → generally cut.** In practice, across both tracks' 2/3 tiers (13 features total), only 3 ever get promoted into a chosen-feature set (`NumStorePurchases`/`Kidhome`/`MntFishProducts` on the regression track — see `REGRESSION_NOTES.md`) — and only because there's a specific, named, independently-confirmed reason (RF's greedy-splitting blind spot on a feature correlated with something RF already picked, with the correlation itself pointed to in the Multicollinearity section). Without a reason that concrete — a named already-selected feature and a correlation number tying it to the demoted one — a 2/3 vote is treated as cut, not as a softer version of chosen. The classify track's `AcceptedCmp1`/`AcceptedCmp2` fail this bar: "RF tends to demote the whole `AcceptedCmp*` family" is a real pattern but not a feature-specific substitution claim, so both stay cut (see "Notable 2/3 cases" below).
- **1/3 or 0/3 → cut.**

**Result — every candidate, all three votes:**

| feature | bivariate | RF (top 15?) | MI (top 15?) | votes | prior-campaign feature? |
|---|---|---|---|---|---|
| AcceptedCmp5 | 0.328 ✓ | 0.040 ✓ (12th) | 0.042 ✓ (2nd) | **3/3** | yes |
| AcceptedCmp3 | 0.254 ✓ | 0.042 ✓ (9th) | 0.027 ✓ (9th) | **3/3** | yes |
| MntWines | 0.247 ✓ | 0.070 ✓ (4th) | 0.040 ✓ (3rd) | **3/3** | — |
| MntMeatProducts | 0.237 ✓ | 0.069 ✓ (5th) | 0.043 ✓ (1st) | **3/3** | — |
| NumCatalogPurchases | 0.221 ✓ | 0.038 ✓ (15th) | 0.034 ✓ (6th) | **3/3** | — |
| Recency | 0.199 ✓ | 0.088 ✓ (1st) | 0.025 ✓ (10th) | **3/3** | — |
| Customer_Tenure_Days | 0.194 ✓ | 0.082 ✓ (2nd) | 0.032 ✓ (7th) | **3/3** | — |
| MntGoldProds | 0.141 ✓ | 0.050 ✓ (6th) | 0.034 ✓ (5th) | **3/3** | — |
| Income | 0.133 ✓ | 0.072 ✓ (3rd) | 0.037 ✓ (4th) | **3/3** | — |
| MntFruits | 0.126 ✓ | 0.039 ✓ (13th) | 0.020 ✓ (15th) | **3/3** | — |
| AcceptedCmp1 | 0.294 ✓ | 0.033 ✗ (17th) | 0.024 ✓ (11th) | 2/3 | yes |
| AcceptedCmp2 | 0.169 ✓ | 0.008 ✗ (23rd) | 0.022 ✓ (13th) | 2/3 | yes |
| Marital_Status (cat) | 0.152 ✓ | 0.046 ✓ (7th) | 0.011 ✗ (20th) | 2/3 | — |
| NumWebPurchases | 0.148 ✓ | 0.030 ✗ (19th) | 0.032 ✓ (8th) | 2/3 | — |
| MntSweetProducts | 0.117 ✓ | 0.042 ✓ (10th) | 0.017 ✗ (17th) | 2/3 | — |
| MntFishProducts | 0.111 ✓ | 0.040 ✓ (11th) | 0.006 ✗ (21st) | 2/3 | — |
| NumStorePurchases | 0.039 ✗ | 0.039 ✓ (14th) | 0.022 ✓ (14th) | 2/3 | — |
| Age | 0.018 ✗ | 0.045 ✓ (8th) | 0.024 ✓ (12th) | 2/3 | — |
| AcceptedCmp4 | 0.177 ✓ | 0.010 ✗ (22nd) | 0.015 ✗ (18th) | 1/3 | yes |
| Teenhome | 0.155 ✓ | 0.012 ✗ (21st) | 0.017 ✗ (16th) | 1/3 | — |
| Education (cat) | 0.102 ✓ | 0.031 ✗ (18th) | 0.005 ✗ (22nd) | 1/3 | — |
| Kidhome | 0.080 ✗ | 0.007 ✗ (24th) | 0.000 ✗ (24th) | 0/3 | — |
| NumWebVisitsMonth | 0.004 ✗ | 0.036 ✗ (16th) | 0.014 ✗ (19th) | 0/3 | — |
| NumDealsPurchases | 0.002 ✗ | 0.028 ✗ (20th) | 0.003 ✗ (23rd) | 0/3 | — |
| Complain | 0.0002 ✗ | 0.001 ✗ (25th) | 0.000 ✗ (25th) | 0/3 | — |

**Final tally: 10 at 3/3, 8 at 2/3, 3 at 1/3, 4 at 0/3.**

**The 3/3 tier splits into two groups:**
- **8 chosen features, no caveat:** `MntWines`, `MntMeatProducts`, `NumCatalogPurchases`, `Recency`, `Customer_Tenure_Days`, `MntGoldProds`, `Income`, `MntFruits`.
- **2 prior-campaign features, also full consensus:** `AcceptedCmp5`, `AcceptedCmp3`. All three methods agree these are relevant — not leakage, but the deployment-availability note from the Relevance section above still applies (they're populated for existing customers with prior-campaign history, less so for brand-new prospects).

**Notable 2/3 cases — cut, but for a specific, explainable reason rather than a blanket low score:**
- `NumStorePurchases` and `Age` both **fail only the bivariate threshold** (0.039 and 0.018) yet clear RF and MI comfortably — a case of real non-linear/interaction signal that a purely linear correlation check misses entirely.
- `MntSweetProducts` and `MntFishProducts` both **fail only MI** despite clearing bivariate and RF — MI's discrete-target estimator is noisier on binary targets than the continuous case (see the regression track's much larger MI magnitudes), so a near-miss here is less conclusive than an RF near-miss.
- `AcceptedCmp1` and `AcceptedCmp2` **fail only RF** (both prior-campaign features) — consistent with RF's broader pattern of demoting the `AcceptedCmp*` group relative to `Recency`/`Customer_Tenure_Days`, discussed in the RF section above.

## Feature Interaction

**Question:** does the effect of one feature on `Response` change depending on another feature? Correlation and Mutual Information above score each feature alone. This section scores every *pair*. No model is fitted.

**Method (all 25 candidate features, 296 pairs, 2,213 rows; the 24 rows with missing `Income` are left out):**
- Numeric features are cut into 4 quantile bins. Features with 8 or fewer distinct values (`Kidhome`, `Teenhome`, `AcceptedCmp*`, `Complain`, `Education`, `Marital_Status`) keep their own levels.
- For each pair (A, B): a likelihood-ratio test (G²) that the A–`Response` association is the same at every level of B. The null model is the log-linear model [AB][AY][BY], fitted by iterative proportional fitting.
- Effect size = (G² − df) / n. Zero means purely additive.
- p-values are corrected over all pairs with Benjamini-Hochberg. This is a screen, not a final claim. Four pairs are skipped because some A × B combinations never occur (`AcceptedCmp1`, `AcceptedCmp2` and `AcceptedCmp4` with `Complain`; `AcceptedCmp3` with `AcceptedCmp4`).

**What the figures show:**
- **Heatmap:** 33 of 296 pairs have q < 0.05. The strongest effects are small in absolute terms (largest 0.024). Interaction is a modest signal here, not a dominant one.
- **Top pairs:** `Teenhome` × `NumWebVisitsMonth` (0.0235), `NumDealsPurchases` × `NumStorePurchases` (0.0195), `Income` × `NumDealsPurchases` (0.0187), `NumDealsPurchases` × `NumWebVisitsMonth` (0.0179), `MntSweetProducts` × `NumStorePurchases` (0.0166).
- **Pattern:** `NumDealsPurchases` and `NumWebVisitsMonth` appear in most of the strong pairs. Both were cut from the chosen set (0/3 votes, see Feature Selection) because their *own* relation to `Response` is weak. They matter through combinations, which a one-feature-at-a-time ranking cannot see.
- **Clearest example (`Teenhome` × `NumWebVisitsMonth`):** with no teenager at home, customers who visit the web 3 or fewer times a month respond at 0.25 (n = 412). With one teenager the same group responds at 0.01 (n = 146). Heavy visitors (more than 7) respond at about 0.20 with or without a teenager.
- **Second example (`NumDealsPurchases` × `NumStorePurchases`):** the lines in the interaction plot cross. Response is not a simple function of either count alone.

**Hand-off to modelling (V5):**
- Of the strongest pairs, only `Income` × `NumStorePurchases`, `MntSweetProducts` × `NumStorePurchases` and a few more sit inside the current 22-feature model set.
- `Teenhome`, `NumDealsPurchases` and `NumWebVisitsMonth` are not in the classify model set. V5 tests adding them back together with their pair features.
- Pair scores are saved in `results/interaction_clf.csv` (columns: A, B, stat, df, effect, p, q).

<!-- V4-V6 begin (generated by results/make_experiments.py) -->

## V4 ReGLU

**Question (professor's Ask 2):** why does the FT-Transformer use ReGLU, and does it help our FT-Transformer-lite?

**What ReGLU is:** a gated feed-forward layer. The linear layer is split into a value half and a gate half, and the output is value × ReLU(gate), so the layer can switch parts of its signal on or off per input. Shazeer (2020) found it beat a plain ReLU feed-forward in Transformers, and FT-Transformer adopted it. We test two sizes: the same hidden width as the ReLU baseline (more parameters), and a width shrunk to match the ReLU parameter count.

**Result (15 folds, paired against ReLU):**

| model | mean F1 | std | diff_vs_reference | CI95 | p_corrected | p_wilcoxon | verdict |
|---|---|---|---|---|---|---|---|
| ReGLU, param-matched | 0.521 | 0.060 | -0.027 | [-0.052, -0.003] | 0.371 | 0.095 | no difference |
| ReGLU, same hidden | 0.544 | 0.050 | -0.003 | [-0.023, 0.018] | 0.912 | 0.847 | no difference |
| ReLU (baseline) | 0.547 | 0.037 | – | – | – | – | reference |

**Reading:** no ReGLU variant beats ReLU on this data. A gated layer needs enough rows to learn what to gate; with about 2,200 rows the extra flexibility is more likely to add variance than signal (a likely reason, not tested separately). The NumPy scratch twin matches PyTorch to about 1e-14 on identical weights, and its 1-repeat K-Fold results are saved in `weights_V4_1/`.

## V5 Interaction Features

**Question (Ask 1, modelling part):** does using the interactions found in the Feature Interaction section improve the models?

**Feature sets:** base 22; plus the 3 dropped features `Teenhome`, `NumDealsPurchases`, `NumWebVisitsMonth` (25); plus those 3 and 5 product features of the strongest pairs (30).

**Neural net result (paired against the base set):**

| model | mean F1 | std | diff_vs_reference | CI95 | p_corrected | p_wilcoxon | verdict |
|---|---|---|---|---|---|---|---|
| ReLU \| + 3 modifiers (25) | 0.553 | 0.044 | -0.003 | [-0.023, 0.016] | 0.883 | 0.762 | no difference |
| ReLU \| + 3 modifiers + 5 pairs | 0.558 | 0.042 | 0.002 | [-0.018, 0.023] | 0.936 | 0.730 | no difference |
| ReLU \| base (22) | 0.556 | 0.042 | – | – | – | – | reference |

**LightGBM on the same feature sets (fair comparison):**

| model | mean F1 | std | diff_vs_reference | CI95 | p_corrected | p_wilcoxon | verdict |
|---|---|---|---|---|---|---|---|
| + 3 modifiers (25) | 0.585 | 0.050 | 0.004 | [-0.009, 0.015] | 0.790 | 0.489 | no difference |
| + 3 modifiers + 5 pairs | 0.604 | 0.044 | 0.023 | [0.007, 0.039] | 0.227 | 0.030 | Wilcoxon p < 0.05 only (weak) |
| base (22) | 0.581 | 0.050 | – | – | – | – | reference |

**Reading:** the interaction EDA used all rows, so any gain here is slightly optimistic. Compare the corrected p-value (conservative) with Wilcoxon (optimistic); differences marked weak are suggestions, not findings. The scratch twin (`*_scratch_V5_0`) reproduces PyTorch to about 1e-14 on identical weights.

## V6 Curriculum Models

**Question (Ask 3):** is the extracurricular FT-Transformer-lite really better than the curriculum models, on the same data, folds and protocol?

**All models, 15 folds (top 12):**

| model | mean F1 | std | tier |
|---|---|---|---|
| LightGBM \| + 3 modifiers + 5 pairs | 0.604 | 0.044 | curriculum |
| LightGBM | 0.590 | 0.048 | curriculum |
| CatBoost | 0.587 | 0.047 | curriculum |
| LightGBM \| + 3 modifiers (25) | 0.585 | 0.050 | curriculum |
| XGBoost | 0.584 | 0.040 | curriculum |
| LightGBM \| base (22) | 0.581 | 0.050 | curriculum |
| HistGradientBoosting | 0.578 | 0.032 | curriculum |
| Stacking | 0.574 | 0.054 | curriculum |
| Gradient boosting | 0.567 | 0.050 | curriculum |
| Random forest | 0.565 | 0.051 | curriculum |
| PCA + Random forest | 0.562 | 0.035 | curriculum |
| SVM (RBF) | 0.558 | 0.050 | curriculum |

**Best of each group:** curriculum `LightGBM | + 3 modifiers + 5 pairs` (0.604); extracurricular `FT-Transformer-lite | ReLU | + 3 modifiers + 5 pairs` (0.558); best overall `LightGBM | + 3 modifiers + 5 pairs` (0.604).

**Decision-gate comparisons (curriculum minus extracurricular, paired):**

| curriculum | extracurricular | F1 cur. | F1 extra. | gap | CI95 | p_corrected | p_wilcoxon |
|---|---|---|---|---|---|---|---|
| LightGBM \| + 3 modifiers + 5 pairs | FT \| ReLU \| + 3 modifiers + 5 pairs | 0.604 | 0.558 | 0.046 | [0.022, 0.070] | 0.111 | 0.003 |
| LightGBM \| + 3 modifiers + 5 pairs | FT \| ReLU (baseline) | 0.604 | 0.547 | 0.057 | [0.041, 0.075] | 0.011 | 0.001 |
| LightGBM | FT \| ReLU \| + 3 modifiers + 5 pairs | 0.590 | 0.558 | 0.032 | [0.005, 0.058] | 0.320 | 0.048 |
| LightGBM | FT \| ReLU (baseline) | 0.590 | 0.547 | 0.042 | [0.025, 0.061] | 0.063 | 0.001 |
| CatBoost | FT \| ReLU \| + 3 modifiers + 5 pairs | 0.587 | 0.558 | 0.029 | [0.005, 0.053] | 0.330 | 0.074 |
| CatBoost | FT \| ReLU (baseline) | 0.587 | 0.547 | 0.039 | [0.021, 0.057] | 0.074 | 0.002 |
| XGBoost | FT \| ReLU \| + 3 modifiers + 5 pairs | 0.584 | 0.558 | 0.026 | [0.001, 0.050] | 0.383 | 0.095 |
| XGBoost | FT \| ReLU (baseline) | 0.584 | 0.547 | 0.036 | [0.016, 0.056] | 0.145 | 0.004 |
| HistGradientBoosting | FT \| ReLU \| + 3 modifiers + 5 pairs | 0.578 | 0.558 | 0.020 | [-0.002, 0.043] | 0.443 | 0.151 |
| HistGradientBoosting | FT \| ReLU (baseline) | 0.578 | 0.547 | 0.031 | [0.016, 0.047] | 0.104 | 0.001 |
| Stacking | FT \| ReLU \| + 3 modifiers + 5 pairs | 0.574 | 0.558 | 0.016 | [-0.009, 0.041] | 0.578 | 0.252 |
| Stacking | FT \| ReLU (baseline) | 0.574 | 0.547 | 0.027 | [0.010, 0.043] | 0.181 | 0.008 |
| Logistic regression | FT \| ReLU \| + 3 modifiers + 5 pairs | 0.539 | 0.558 | -0.018 | [-0.045, 0.008] | 0.553 | 0.256 |
| Logistic regression | FT \| ReLU (baseline) | 0.539 | 0.547 | -0.008 | [-0.033, 0.015] | 0.782 | 0.720 |

Full tables and caveats: `results/EXPERIMENTS.md`. Figure: `media/ranking_clf_all.png`.

<!-- V4-V6 end -->

## V7 Why the Gap

**Question:** where does the gap between FT-Transformer-lite and tree boosting come from: interactions, the way each feature's effect is fitted, or too little data?

**Interaction-depth ladder (15 folds, same 25 features, Optuna 30 trials each):**

| model | mean F1 | step vs previous | 95% CI | p_wilcoxon |
|---|---|---|---|---|
| FT-Transformer-lite + 3 modifiers (25) | 0.553 | - | - | - |
| LightGBM depth 1 (additive, no interactions) | 0.576 | +0.023 | [0.004, 0.043] | 0.095 |
| LightGBM depth 2 (pairs) | 0.588 | +0.012 | [-0.007, 0.031] | 0.21 |
| LightGBM full | 0.585 | -0.003 | [-0.021, 0.015] | 0.68 |

**Learning curve (5 folds of seed 42, + 3 modifiers + 5 pairs):** gap LightGBM minus FT = 0.068 (447 rows), 0.059 (894), 0.042 (1,341), 0.034 (1,789).

**Reading:** trees that cannot combine features at all already beat FT-Transformer-lite, so the classify gap is not about interactions. It is the way each feature's effect is fitted (trees cut at thresholds, the network must learn smooth curves) plus too little data (the gap halves as rows grow 4x). Outputs: `weights_V7_1/classification_lightgbm_kfold_metrics.json`, `weights_V7_1/classification_pytorch_learning_curve.json`, `media/gap_ladder.png`, `media/learning_curves.png`.

