# Canva Deck Content — Step-by-Step EDA Walkthrough

Paste-ready content for the middle section of "01076641: ML Project" (design `DAHTNckkOHU`).

**How to use this file:** In Canva, open the design, go to the page where "Dataset Overview" currently sits, and use *Add page → Paste text to create slides* (or select all the text below one section at a time and paste — Canva splits on headings). Each `##` heading below becomes one slide's title; the bullets under it become the body content. Bullets marked **[metric]** are definition/method content folded into a combined slide — give those a light visual distinction (italics, a small icon, or a boxed callout) so they read as "how we measured this" rather than blending into the findings bullets around them, matching the deck's existing template. Slides tagged **[IMAGE PLACEHOLDER: ...]** need a gray placeholder box with that exact caption — drop your notebook screenshot in later.

**Title convention for Section B and Section C:** on the actual Canva slide, every content slide in these two sections keeps the same running title as its section header — `Classification EDA: Target `y`` for all of Section B, `Regression EDA: Target `balance`` for all of Section C — matching the house style already in use (title stays constant, the specific finding lives in the bullets/chart, not in the title). The `##` heading text for each slide below stays distinct and descriptive (`B3. Step 2: ...`, etc.) so this document itself is readable and easy to navigate/cross-reference — when you paste a slide into Canva, overwrite its auto-filled title with the running header above instead of leaving the descriptive `##` text as the visible title. Section A's slide titles are unaffected by this convention (each keeps its own distinct title, as-is, on both the doc and the slide).

**What stays untouched:** Title, Problem Setting Overview, Case Analysis & Data Prep, Next Steps, Thank You. This document only covers the replacement for the current 7 middle slides (Dataset Overview → Key EDA Findings), expanded into the full step-by-step walkthrough below. Delete the old 7 slides once these are in place.

**Slide-density note:** every step's method description and its metric explainer(s) are combined into a single slide, and any method slide with no unique metric of its own (multicollinearity recap, secondary passes, etc.) is folded straight into its result slide instead of standing alone with 2-3 bullets and no image. This keeps every slide either dense-with-purpose (method+metric together) or dense-with-evidence (result+image together) — never a slide that's just a few short lines and nothing else.

---

## SECTION A — Data Overview

---

## A1. Dataset Overview: Columns & What They Represent

- 45,211 rows, 17 columns (16 features + target `y`)
- **Bank client data:** age, job, marital, education, default, balance, housing, loan
- **Last-contact data:** contact, day, month, duration
- **Other campaign attributes:** campaign, pdays, previous, poutcome
- **Target:** y — did the client subscribe to a term deposit?
- "unknown" is a real category, not missing data: job (~0.6%), education (~4.1%), contact (~28.8%), poutcome (~81.7%)
- pdays == -1 is a sentinel meaning "never contacted before"
- Target y is imbalanced: ~11.7% yes / ~88.3% no

**[IMAGE PLACEHOLDER: Distribution of Data / Class Imbalance Chart — y class split, ~88% no / ~12% yes]** — source: `classify.ipynb` cell 27 (or `regression.ipynb` cell 27, identical)

---

## A2. train.csv vs. test.csv: Why We Drop test.csv

- train.csv: 45,211 rows, the full dataset, date-ordered May 2008 – Nov 2010
- test.csv: 4,521 rows — a random 10% subset drawn *from train.csv itself* (per UCI source docs), not an independent sample
- Decision: drop test.csv entirely; we run our own seeded train_test_split on train.csv at modeling time
- Why: using test.csv as a holdout risks leakage / duplicated rows since it isn't independent of train.csv

**[IMAGE PLACEHOLDER: train.csv vs test.csv Row-Count Comparison Chart]** — **GAP: no such chart in either notebook.** No cell plots train/test row counts; it's just two numbers (45,211 vs 4,521). Build by hand or replace with a stat call-out instead of an image.

---

## SECTION B — Classification EDA: Target `y`

---

## PHASE 1 — FIND THE CANDIDATES

---

## B1. Step 1: Relevance Ranking — Method & Metrics

- 16 candidate features — screen with one ranked score before plotting anything
- Numeric features → point-biserial correlation vs. y
- Categorical features → Cramér's V from the raw crosstab vs. y
- **[metric] Point-biserial correlation:** same math as Pearson r, applied to a continuous variable vs. a binary variable. Range -1 to +1 — sign = direction of the relationship, magnitude = strength.
- **[metric] Cramér's V:** built from a contingency table → chi-square test → normalized to 0–1. Formula: V = √((χ²/n) / min(r−1, k−1)). 0 = no association, 1 = perfect association — no direction, unlike Pearson r. Caveat: inflated on small or sparse samples.

**[IMAGE PLACEHOLDER: Point-Biserial Correlation illustration — duration & pdays by y, side-by-side boxplots]** — source: `classify.ipynb` cell 32 (id `999bf625`), a new cell added specifically for this slide right after the top-features loop, executed and confirmed working (zero errors). It's a single figure with two subplots — `duration` by `y` (r = 0.395, strong) next to `pdays` by `y` (r = 0.104, right at the threshold) — one image instead of stitching together two separate screenshots, and the pairing doubles as a visual for what the 0.1 threshold in B2 means.

**[IMAGE PLACEHOLDER: Cramér's V illustration — categorical association heatmap]** — source: `classify.ipynb` cell 38, the 9×9 categorical redundancy heatmap (this is the one you were remembering). Caveat: that heatmap's actual analytical purpose in this deck is the **B6 feature-vs-feature redundancy check**, not feature-vs-target relevance — it's still built from `cramers_v()`, so it's a legitimate illustration of "what a Cramér's V matrix looks like," but if you use it here too, caption it as a method illustration (not a relevance result) so it doesn't read as a duplicate finding when B6 shows the same image again with its real analytical purpose.

---

## B2. Step 1: Results & Threshold

- Threshold = 0.1 (Cohen 1988 "small effect" convention for correlation-scale statistics)
- duration 0.395 · poutcome 0.312 · month 0.260 · contact 0.151 · housing 0.139 · job 0.136 · pdays 0.104
- top_features = [duration, poutcome, month, contact, housing, job, pdays]
- Domain read: poutcome = warm lead signal; month = seasonality / strategy drift proxy; housing & loan = spare-cash constraint; job = income proxy

**[IMAGE PLACEHOLDER: Relevance Score Bar Chart — all 16 features ranked, threshold line at 0.1]** — source: `classify.ipynb` cell 30

---

## PHASE 2 — REASONS WHY (validate & explain the candidates)

---

## B3. Step 2: Multicollinearity Check — Method & Metric

- A different question than Step 1: feature-vs-feature overlap, not feature-vs-target
- Pearson correlation matrix over the 7 numeric columns
- **[metric] Pearson correlation matrix:** r = cov(X,Y) / (σx · σy) — range -1 to +1. Symmetric matrix, 1.0 on the diagonal. Caveat: only captures linear/monotonic structure, sensitive to outliers, no causation implied.

---

## B4. Step 2: Result

- All pairs weak (|r| < 0.1) except two: `pdays` × `previous` (**0.45**, moderate — both describe prior-contact history) and `day` × `campaign` (**0.16**, mild — timing within the month relates to how many times a client got contacted)
- Every other pair sits at or below the noise floor (e.g. `age`×`balance` 0.10, `duration`×`campaign` -0.08)
- Takeaway: numeric features are largely independent of each other — only one moderate and one mild redundancy to watch for

**[IMAGE PLACEHOLDER: Numeric Correlation Heatmap — 7x7 Pearson matrix]** — source: `classify.ipynb` cell 33

---

## B5. Step 3: Categorical Redundancy Deep Dive — Method & Metric

- Pairwise Cramér's V across the 9 categorical columns
- Plus an explicit crosstab of poutcome vs. pdays == -1
- **[metric] Plain frequency crosstab:** just row/column counts, no aggregation — used here to show exact overlap in checkable numbers rather than a single summary score

---

## B6. Step 3: Result

- Cramér's V — top tier: `contact`–`month` **0.51**, `housing`–`month` **0.50**, `job`–`education` **0.46**
- Second tier: `month`–`poutcome` 0.21, `contact`–`poutcome` 0.21, `housing`–`contact` 0.21, `loan`–`month` 0.18 — everything else below 0.15
- `month` is the connective hub — redundant with both `contact` and `housing`, so a lot of the categorical block's signal routes through campaign timing
- poutcome == unknown ≈ pdays == -1: near-total overlap, 36,954 / 36,959 rows
- Implication: poutcome's detail is only informative for the ~18% of clients who were previously contacted

**[IMAGE PLACEHOLDER: Categorical Redundancy Heatmap — 9x9 Cramér's V matrix]** — source: `classify.ipynb` cell 38 (the poutcome≈pdays overlap count comes from cell 39, a plain table not a chart — optional extra screenshot if you want the exact numbers on-slide)

---

## B7. Step 4: Model-Based Importance — Random Forest — Method & Metric

- Prior checks are all bivariate; a Random Forest captures interactions and non-linearities
- RandomForestClassifier(n_estimators=300, random_state=42) on one-hot encoded features, dummy columns summed back to their parent feature
- **[metric] RF feature importance:** measures average impurity reduction attributable to a feature, across all trees. Caveat: biased toward continuous / high-cardinality features (more possible split points).

---

## B8. Step 4: Result

- duration 0.267 · month 0.106 · balance 0.095 · age 0.092 · day 0.081 · job 0.070 · poutcome 0.070
- Agrees with Step 1 on duration / month / poutcome
- Disagrees on balance / age (much higher here) — flagged as possible RF continuous-feature bias

**[IMAGE PLACEHOLDER: RF Feature Importance Bar Chart (Classification)]** — source: `classify.ipynb` cell 45

---

## B9. Step 5: Mutual Information Cross-Check — Method & Metric

- Closes the gap in both prior methods: no model-fit bias (unlike RF), captures non-linear dependency (unlike Pearson / Cramér's V)
- mutual_info_classif — numeric columns raw, categorical columns factorized + discrete, random_state=42
- **[metric] Mutual information:** measures shared information between a feature and the target, directly from the joint distribution. Units: nats — not on the same 0–1 scale as r or Cramér's V, so compare by rank, not raw value.

---

## B10. Step 5: Result

- duration 0.0709 · poutcome 0.0294 · pdays 0.0275 · month 0.0244 · balance 0.0223
- duration / poutcome / month are robust across all 3 methods
- pdays jumps to 3rd here (vs. 7th/9th elsewhere) — its step-like "never contacted" structure is understated by linear correlation
- balance and age land in MI's top 7 too — independent (non-RF) evidence they carry real, if modest, signal

**[IMAGE PLACEHOLDER: Mutual Information Bar Chart (Classification)]** — source: `classify.ipynb` cell 48

---

## B11. Classification Summary

- **Features chosen:** `poutcome`, `month`, `contact`, `housing`, `job`, `pdays`
- **Why:** each cleared the Step 1 relevance threshold (0.1) and held up under two independent cross-checks (Random Forest, Mutual Information) — not a single method's quirk, agreement across 3 different kinds of test
- `duration` topped every method (correlation, RF, MI) but is excluded — it's only known after a call ends, so it can't be used to decide who to call before calling them
- **Class imbalance:** ~11.7% "yes" vs. ~88.3% "no" — needs class weighting, resampling, or threshold tuning; plain accuracy would be misleading

---

## SECTION C — Regression EDA: Target `balance`

---

## C1. Step 1: Primary Relevance Pass — Method & Metric

- 16 candidate features — screen with one ranked score before plotting anything (`y` included as a candidate here too, mirroring how `duration` is scored, not skipped, in the classification track despite being unusable in a deployable model)
- Numeric features → Pearson correlation vs. balance
- Categorical features → correlation ratio (η) from a one-way ANOVA vs. balance
- **[metric] Pearson correlation:** same math as point-biserial in the classification track, applied to two continuous variables here instead of continuous-vs-binary. Range -1 to +1 — sign = direction of the relationship, magnitude = strength.
- **[metric] Correlation ratio (η):** built from a one-way ANOVA decomposition (SS_between / SS_total) → normalized to 0–1. 0 = no association, 1 = category membership fully determines the group's average balance — no direction, unlike Pearson r. Caveat: inflated when a category has very few members (the η equivalent of Cramér's V's small-sample caveat).

---

## C2. Step 1: Results & Threshold

- Threshold = 0.1 (Cohen's 1988 "small effect" convention)
- 2 features clear it: `month`, `job`
- `age` (0.098) misses by a hair but isn't actually weak — RF/MI both rank it 2nd (see Step 5/6)

**[IMAGE PLACEHOLDER: Regression Relevance Score Bar Chart — all 16 features ranked, threshold line at 0.1]** — source: `regression.ipynb` cell 30 (new, mirrors the classification track's B2 chart)

**[IMAGE PLACEHOLDER: Balance-by-Category Boxplot Charts — job, education, marital, housing, loan]** — source: `regression.ipynb` cell 32 (one figure, 5 stacked boxplot subplots, covers all 5 categories). The numeric-correlation half of this slide's text comes from cell 31's scatter grid (6 subplots, balance vs. each other numeric column) — optional extra screenshot if you want the numeric side shown too.

---

## C3. Step 2: Multicollinearity Check — Method & Result

- Method: same method as the classification track (Pearson matrix, target-agnostic check) — brief recap since regression.ipynb runs standalone; metric already explained in slide B3, no new explainer needed here
- Result: same finding as classification — all pairs weak except pdays vs. previous, r = 0.455

**[IMAGE PLACEHOLDER: Numeric Correlation Heatmap (Regression track) — can reuse the classification-track image if preferred]** — source: `regression.ipynb` cell 35 (identical computation to `classify.ipynb` cell 33 / slide B4 — genuinely fine to reuse that screenshot)

---

## C4. Step 3: balance vs. y Deep Dive — Method & Result

- Method: checks whether the two tracks' targets are linked; boxplot/violin plot and medians of balance, split by y
- Result: y == yes: median 733 / mean 1,804 — y == no: median 417 / mean 1,304 — ~76% higher median for subscribers, but heavily overlapping distributions
- Explains why balance scored low (0.053) in the classification relevance ranking
- This gap is now triangulated, not just a two-median comparison: Step 1/RF/MI (slides C2/C6/C7) all score `y` as a real-but-modest `balance` predictor — but still unusable in a pre-call model, same as `duration` in classification

**[IMAGE PLACEHOLDER: Balance by y — Boxplot/Violin Chart]** — source: `regression.ipynb` cell 37 (produces both a boxplot and a violin plot as two separate images in the same cell — pick one, or screenshot both)

---

## C5. Step 4: Age-Binned Deep Dive — Method & Result

- Method: raw age–balance scatter showed only r = 0.098; binning age into life-stage buckets tests for hidden non-linear structure
- Result: monotonic, convex-increasing trend — median 362.5 (age 18–25) → 1,413 (age 65+), ~3.9x increase; mean nearly triples, 902.7 → 2,822.0
- age is a stronger signal than its raw linear correlation implied — worth using binned age (or age²) rather than raw age

**[IMAGE PLACEHOLDER: Age-Binned Balance Trend Line Chart]** — source: `regression.ipynb` cell 40

---

## C6. Step 5: Random Forest Importance — Method & Result

- Method: same method as classification (slide B7), refit on balance, now including `y` as a scored candidate; metric already explained there, no new explainer needed here
- Result: duration 0.262 · age 0.139 · job 0.104 · day 0.104 · month 0.079 · campaign 0.077 · ... · y 0.012 (rank 14 of 16)
- **Agrees with previous:**
  `job` — independent confirmation from a completely different method (Step 1 2nd at 0.102, RF 3rd at 0.104). `age` also confirms the age-binned deep dive's finding (RF 2nd at 0.139) that it carries real signal despite missing the Step 1 threshold.
- **Disagrees with previous:**
  `month` ranks much lower here (5th, 0.079) than its Step 1 top rank (1st, 0.156) — plausibly absorbed by month's redundant signal with contact/housing (see the classification track's B5/B6). `duration`'s #1 rank here (0.262) contradicts its near-zero Step 1 score (0.022) — no plausible mechanism linking call length to balance, flagged as RF continuous-feature bias (not yet resolved — see Step 6).
- `y` ranks low (0.012) — real but weak, and unusable regardless: it's the outcome of the same call a pre-call `balance` model would need to run before, same exclusion logic as `duration` in the `y`-classifier

**[IMAGE PLACEHOLDER: RF Feature Importance Bar Chart (Regression)]** — source: `regression.ipynb` cell 43

---

## C7. Step 6: Mutual Information Cross-Check — Method & Result

- **Mutual Information:** measures shared info between a feature and balance directly from the data. Captures non-linear dependency (unlike Pearson/correlation ratio η). Units: nats, so compare by rank, not raw value.
- Top 5: job 0.1083 · age 0.0672 · education 0.0474 · marital 0.0443 · month 0.0362
- `job` is robust across all 3 methods; `marital` jumps to 4th — a genuine promotion, its signal was understated by the univariate correlation-ratio score (10th in Step 1)
- `duration` drops to 13th of 16 — resolving the RF question from the last slide: its #1 RF rank was a continuous-feature artifact, not real signal
- `age` confirmed 2nd by a 3rd independent method — binned trend, RF, and MI all agree; `month` demoted to 5th here too, same direction as RF
- `y` lands mid-table (rank 8) — real, modest signal, but see the usability caveat on the previous slide

**[IMAGE PLACEHOLDER: Mutual Information Bar Chart (Regression)]** — source: `regression.ipynb` cell 46

---

## C8. Regression Summary

- **Features chosen (6, final):** `job`, `age` (binned), `month`, `housing`, `loan`, `education`
- **No feature scores well in absolute terms, in any method:** best Step 1 score is `month` (0.156), best RF score is `age` (0.139), best MI score is `job` (0.108) — all still in Cohen's "small effect" range, nowhere near classification's `duration`(0.395)/`poutcome`(0.312). `balance` needs several modest signals combined, not one dominant driver.
- **Why these 6:** `job`/`age`/`month` rank consistently across all three methods even though their raw scores stay modest; `housing`/`loan`/`education` carry real signal too (clear boxplot median gaps, e.g. no-loan 496 vs. has-loan 258, and moderate MI ranks) but swing more across methods — most notably suppressed in RF, plausibly because `month` absorbs their shared signal (see C3, and the classification track's B5/B6)
- `duration` topped RF (0.262) but that rank isn't trustworthy — near-zero Step 1 score (0.022) and 13th-of-16 by MI both flag it as a continuous-feature artifact, not a real driver
- `y` scores real but modest everywhere (rank 8 in both Step 1 and MI) yet is excluded outright — same logic as `duration` in the classification track: it's the outcome of the same call a pre-call `balance` model would need to run before
- **Class distribution:** `balance` is right-skewed with negative values present (min -8,019, max 102,127) — needs a signed-log or robust-scaling transform before modeling

**[IMAGE PLACEHOLDER: Final Feature Shortlist Checklist (regression) + balance Distribution/Skew Chart]** — split source: the skew/distribution half is `regression.ipynb` cell 16 (raw `balance` histogram, back in Train Distribution Check — shows right-skew and negative values). The checklist half is a **GAP: no notebook chart** — same as B14, build it by hand.

---

*End of new middle section — 23 slides (A1–A2, B1–B11 plus 2 phase-divider slides, C1–C8). Every method-only slide is now folded into its paired metric-explainer or result slide, so there are no more bare 2-3 bullet slides. Combined with the 5 unchanged bookend slides, final deck = 28 pages.*
