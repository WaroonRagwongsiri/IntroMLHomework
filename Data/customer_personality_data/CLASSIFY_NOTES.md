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

**Result.** Final cleaned shape: **2,237 rows x 28 columns**. `Response` class balance: **14.9% positive / 85.1% negative** — meaningfully imbalanced, comparable in spirit to `banking_dataset`'s `y` (11.7%/88.3%) but a bit gentler, in the same range as `google_play_data`'s `hit` was gentler still (39.1%/60.9%).

---

## Relevance vs. Classification Target (Response) — Correlation First

**Purpose.** Same reasoning as both prior datasets: with 25 candidate features (23 numeric, 2 categorical — by far the largest candidate pool of the three datasets explored), a single relevance ranking upfront avoids a wall of uninformative univariate plots.

**Method.** Two measures depending on feature type (see `MATRIX_METHODS.md`):
- Numeric (`Income`, `Age`, `Recency`, `Kidhome`, `Teenhome`, `Customer_Tenure_Days`, the six `Mnt*` spend columns, the five purchase-channel/engagement columns, `Complain`, `AcceptedCmp1`-`5`): point-biserial correlation with `Response` (0/1), `Income` masked to non-missing rows.
- Categorical (`Education`, `Marital_Status`): Cramér's V from the raw-label contingency table.

**`AcceptedCmp1`-`5` are deliberately included, not excluded** — flagged explicitly as leakage-adjacent below, same treatment as `duration` (`banking_dataset`) and `Reviews_num` (`google_play_data`): shown in the ranking, not silently dropped.

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

**Leakage note — `AcceptedCmp1`-`5`.** These flag acceptance of the five *earlier* campaigns. A customer who accepted a prior campaign is mechanically more likely to be a responsive, engaged customer — the same kind of customer likely to accept the *current* campaign (`Response`) too. This is not circular by construction (unlike `Installs_num` → `hit` in `google_play_data`, where the target is a deterministic function of the feature), but it is a strong behavioral proxy that a real prospective-targeting model would not have available for a brand-new prospect with no campaign history. The three of the top four bivariate features are `AcceptedCmp5`/`1`/`3` — this ranking is dominated by leakage-adjacent signal at the very top. Concretely: **Response rate given any prior campaign acceptance is 40.7% (n=462), vs. 8.2% given none (n=1,775)** — a 5x gap, the starkest split in the whole ranking.

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

**A genuinely different pattern from both prior datasets — leakage features are *demoted*, not amplified, by RF.** In `google_play_data`, RF importance *concentrated* onto the leakage feature `Reviews_num` (0.690 of total, 6x the runner-up) relative to its mid-table bivariate rank — the nonlinear model exploited exactly the leakage relationship. Here the opposite happens: `AcceptedCmp5` — the #1 bivariate feature (0.328) — falls to 12th in RF importance (0.040), and `AcceptedCmp1`/`AcceptedCmp3`/`AcceptedCmp4` similarly drop out of the top ranks. Instead, RF elevates `Recency` and `Customer_Tenure_Days` (7th and 8th bivariate, now 1st and 2nd) to the top. A plausible reading: the `AcceptedCmp*` flags are highly correlated with each other and with `Response` in a fairly linear, additive way that point-biserial correlation already captures efficiently — RF has little nonlinear structure left to exploit there once the (many) spend/purchase/tenure features are available to substitute for the same underlying "engaged customer" signal. This is a useful, non-repeating counterexample to the "RF always inflates the leakage feature" pattern found in the two prior datasets — the direction of the RF/bivariate gap depends on *how* the leaked signal enters (a single dominant feature vs. several correlated ones spread across many columns).

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

**A genuine three-way disagreement, reported honestly rather than resolved into one story.** MI puts `AcceptedCmp5` back in 2nd place (0.042, close behind `MntMeatProducts`'s 0.043) — a real leakage signal MI still picks up strongly, unlike RF's 12th-place demotion of the same feature. `Recency`/`Customer_Tenure_Days` — RF's top two — rank only 7th/7th-adjacent here (0.025 and 0.032). `Marital_Status` (0.011) and `Education` (0.005), both comfortably above threshold bivariately, are near the bottom under MI. The honest summary: bivariate correlation, RF, and MI each tell a *different* story about how much weight the `AcceptedCmp*` leakage features deserve relative to `Recency`/`Customer_Tenure_Days`/spend — no single method should be taken as the final word, and a model built on this data should treat `AcceptedCmp1`-`5` as a deliberate inclusion/exclusion decision (does the deployment scenario have prior-campaign history for the customer, or not?) rather than trusting any one importance score to settle it.

---

## Feature Selection — Combining the Three Methods (Final Cut)

**Purpose.** The three methods above measure genuinely different things — linear/monotonic association (bivariate), tree-based interaction-aware importance (RF), and general statistical dependency (MI) — and each can be misled in its own way (small-sample noise for all three; RF specifically via greedy splitting on correlated features, see the RF section above). This section states the actual rule used to collapse three separate rankings into one final feature-selection decision, and applies it to every one of the 25 candidates.

**Method — the cutting rule, not an average.** The three scores live on incompatible scales: bivariate is signed, −1 to +1; RF importance is unsigned, 0 to 1, and sums to exactly 1 across all candidates; MI is unsigned, 0 to ∞ with no fixed ceiling. Averaging them would let whichever method happens to have the largest numeric range dominate a blended score for no principled reason. Instead, each method casts an independent pass/fail vote per feature:

- **Bivariate passes** if `|score| ≥ 0.10`.
- **RF passes** if the feature lands in RF's top 15 of 25.
- **MI passes** if the feature lands in MI's top 15 of 25.

Features are then tiered by vote count:

- **3/3 (full consensus) → chosen feature, no caveat** (unless separately leakage-flagged — see below). A signal that survives three different statistical assumptions at once is very unlikely to be a fluke of one method or of this dataset's modest size (2,237 rows).
- **2/3 (lower-confidence signal) → generally cut.** In practice, across both tracks' 2/3 tiers (13 features total), only 2 ever get promoted into a chosen-feature set (`NumStorePurchases`/`Kidhome` on the regression track — see `REGRESSION_NOTES.md`) — and only because there's a specific, named, independently-confirmed reason (RF's greedy-splitting blind spot on a feature correlated with something RF already picked). Without a reason that concrete, a 2/3 vote is treated as cut, not as a softer version of chosen.
- **1/3 or 0/3 → cut.**

**Result — every candidate, all three votes:**

| feature | bivariate | RF (top 15?) | MI (top 15?) | votes | leakage-adjacent? |
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
- **8 non-leakage chosen features, no caveat:** `MntWines`, `MntMeatProducts`, `NumCatalogPurchases`, `Recency`, `Customer_Tenure_Days`, `MntGoldProds`, `Income`, `MntFruits`.
- **2 leakage-adjacent features, also full consensus:** `AcceptedCmp5`, `AcceptedCmp3`. All three methods agree these are relevant — the leakage caveat from the Relevance section above still applies (include only if the deployment scenario genuinely has prior-campaign history for the customer at prediction time).

**Notable 2/3 cases — cut, but for a specific, explainable reason rather than a blanket low score:**
- `NumStorePurchases` and `Age` both **fail only the bivariate threshold** (0.039 and 0.018) yet clear RF and MI comfortably — a case of real non-linear/interaction signal that a purely linear correlation check misses entirely.
- `MntSweetProducts` and `MntFishProducts` both **fail only MI** despite clearing bivariate and RF — MI's discrete-target estimator is noisier on binary targets than the continuous case (see the regression track's much larger MI magnitudes), so a near-miss here is less conclusive than an RF near-miss.
- `AcceptedCmp1` and `AcceptedCmp2` **fail only RF** (both leakage-adjacent) — consistent with RF's broader pattern of demoting the `AcceptedCmp*` group relative to `Recency`/`Customer_Tenure_Days`, discussed in the RF section above.
