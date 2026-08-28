## Customer Personality Analysis
> https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

### Status: backup dataset, not primary
This dataset is a **second backup**, alongside `google_play_data/`, built out in case `banking_dataset/` (the primary dataset for this project) turns out to need replacing. `banking_dataset/` remains the primary dataset; `google_play_data/` remains untouched as the first backup. This folder holds a working, correctly-executed pair of notebooks and narrative notes (no presentation deck, no `PLAN.md` — that scope is only worth building if this dataset is ever actually promoted to primary).

**Why this dataset, specifically:** it is the only one of the three explored that ships **both a natural classification target (`Response`) and a natural regression target (`Income`) in the same raw CSV**, with no engineering needed for either — unlike `google_play_data`, which needed an engineered `hit` target, or `banking_dataset`, which has no regression target in the same file at all.

**Domain fit is the strongest of the three candidates.** This is retail/direct-marketing customer analytics data — a clean, unstretched fit for the project brief's "Customer/commercial analysis-related" suggested domain (see `_69-1-01076641_Project-description.pdf`), unlike `google_play_data`'s stretch-fit as mobile-app analytics (flagged with its own compliance caveat in `google_play_data/README.md`). No such caveat is needed here.

### Context
> Customer Personality Analysis is a detailed analysis of a company's ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors, and concerns of different types of customer segments. The dataset was originally built for a **clustering/customer-segmentation** exercise (the dataset's own stated "Target" is "perform clustering to summarize customer segments") — this project instead uses it for supervised classification (`Response`) and regression (`Income`), which the data supports natively but which is not the use case the dataset was originally designed around. Public notebooks referencing this file skew heavily toward unsupervised RFM/segmentation analysis; there is less external precedent to benchmark a supervised approach against than for `banking_dataset` or `google_play_data`.

### Content
> `marketing_campaign.csv`: **2,240 rows x 29 columns**, one row per customer, tab-separated (`sep="\t"`, not comma — confirmed by direct inspection). By far the smallest of the three datasets explored (~18x smaller than `banking_dataset`'s ~41k rows, smaller than `google_play_data`'s ~10k rows too).

### Detailed Column Descriptions
Grouped per the dataset's own documented structure (People / Products / Promotion / Place):

**People**
1. `ID`: unique customer identifier (dropped — not a feature)
2. `Year_Birth`: birth year (parsed into engineered `Age` = 2014 − `Year_Birth`; 3 rows with implausible values — 1893, 1899, 1900 → age 121, 115, 114 — dropped as data-entry errors)
3. `Education`: education level (categorical, 5 values: `Graduation`, `PhD`, `Master`, `2n Cycle`, `Basic`)
4. `Marital_Status`: marital status (categorical; raw file has 8 values including 2 `Absurd` and 2 `YOLO` joke entries and a 3-row `Alone` category — all three collapsed into `Single`, leaving 5 clean categories)
5. `Income`: yearly household income (numeric — **chosen regression target**; **24 of 2,240 rows (1.1%) missing**; 8 outliers above the IQR upper bound of $118,349, including one extreme value at $666,666, roughly 4x the next-highest)
6. `Kidhome` / `Teenhome`: number of children / teenagers in the household (numeric, 0–2)
7. `Dt_Customer`: enrollment date as a `DD-MM-YYYY` string (parsed to datetime; derived into engineered `Customer_Tenure_Days` = days since enrollment relative to the dataset's latest enrollment date, 2014-06-29)
8. `Recency`: days since the customer's last purchase (numeric)
9. `Complain`: 1 if the customer complained in the last 2 years, else 0 (binary, heavily imbalanced — 0.9% positive)

**Products** (amount spent, in the dataset's currency unit, over the last 2 years)
10–15. `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` — six numeric spend columns, all right-skewed with a meaningful outlier tail (9–11% of rows flagged by the 1.5x IQR rule per column, see `CLASSIFY_NOTES.md`'s Distribution Check)

**Promotion**
16. `NumDealsPurchases`: number of purchases made with a discount (numeric)
17–21. `AcceptedCmp1`–`AcceptedCmp5`: 1 if the customer accepted the offer in that numbered prior campaign, else 0 (binary; genuinely leakage-adjacent for `Response`, kept and flagged explicitly rather than dropped — see `CLASSIFY_NOTES.md`)
22. `Response`: 1 if the customer accepted the offer in the **most recent** campaign, else 0 (binary — **chosen classification target**, a raw column, not engineered)

**Place**
23–26. `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`: purchase counts by channel plus monthly website visit count (all numeric)

**Dropped entirely**
- `Z_CostContact`, `Z_Revenue`: confirmed constant across all 2,240 rows (`3` and `11` respectively, verified by direct inspection before dropping) — zero variance, no predictive content, present in the raw file but not part of the dataset's own documented column list above.

#### Target variables
- **Classification target: `Response`.** Raw column, no engineering needed. Confirmed split (on the 2,237-row cleaned classify dataset): **14.9% positive / 85.1% negative** — meaningfully imbalanced, comparable to `banking_dataset`'s `y` (11.7%/88.3%) but a bit gentler.
- **Regression target: `Income`.** Rows with missing `Income` are dropped in `regression.ipynb` (imputing the target itself would be the wrong move) — 2,213 of 2,237 cleaned rows remain (24 dropped). By far the strongest bivariate regression signal found across all three datasets explored (16 of 24 candidate features clear the 0.1 relevance threshold, several exceeding 0.5) — see `REGRESSION_NOTES.md`'s Discussion for the full contrast against `balance` and `Rating`'s "no dominant predictor" finding in the other two datasets.

#### Missing values (raw file)
Only `Income` has missing values: **24 of 2,240 rows (1.1%)**. Every other column is complete.

#### Data-quality issues found and handled (both notebooks, `## Data Cleaning`)
- **`Z_CostContact` / `Z_Revenue` constant columns** — confirmed zero variance (`3` and `11` respectively for every row), dropped explicitly with printed confirmation before dropping.
- **3 implausible `Year_Birth` values** (1893, 1899, 1900 → age 121, 115, 114 as of the dataset's 2014 reference year) — genuine data-entry errors, dropped outright (small n, no reasonable imputation target).
- **`Marital_Status` joke/near-empty categories** — `Absurd` (2 rows), `YOLO` (2 rows), and the genuine-but-tiny `Alone` (3 rows) all folded into `Single`.
- **`Income` missing (24 rows)** — kept (masked) in `classify.ipynb` since `Response` doesn't depend on it; dropped in `regression.ipynb` since it's the target itself.
- **Multicollinearity, not a data-quality defect but worth flagging here:** `Income`, the six `Mnt*` spend columns, and the purchase-channel counts (especially `NumCatalogPurchases`) are heavily inter-correlated (several pairs above r=0.55, one at r=0.73) — they largely describe one underlying "affluent, high-spending, catalog/store-shopping customer" profile rather than independent dimensions. See `CLASSIFY_NOTES.md`/`REGRESSION_NOTES.md`'s Multicollinearity sections.
- **Leakage-adjacent features, not dropped but flagged:** `AcceptedCmp1`–`5` for the `Response` classification target (customers who accepted an earlier campaign are ~5x more likely to accept the current one: 40.7% vs. 8.2%), and the `Mnt*` spend columns for the `Income` regression target (a real correlation-direction ambiguity, not the mechanical circularity found in `google_play_data`'s `Installs_num`/`hit` pair). Both are included as candidates and discussed explicitly, not silently excluded — see each notebook's relevance-ranking notes.

### Citation
> Please cite the original Kaggle dataset if used for further analysis:

> Akash Patel ("imakash3011"), "Customer Personality Analysis," Kaggle, 2021. https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

Licensed under **CC0: Public Domain** (confirmed directly via the Kaggle API's dataset metadata at download time — `"licenses": [{"name": "CC0-1.0"}]`) — no attribution legally required, though cited above as good practice.

**Provenance note — cite carefully, not as a single clean source.** The dataset's own Kaggle page acknowledges the underlying data was "provided by Dr. Omar Romero-Hernandez" (UC Berkeley); `imakash3011`'s upload is one of several near-identical mirrors of this same file circulating on Kaggle (e.g. `whenamancodes/customer-personality-analysis`), and secondary sources suggest an earlier `jackdaoud/marketing-data` Kaggle dataset may be an even earlier upload of the same underlying data. The citation above is to the specific upload actually used in this project, not a claim about ultimate original authorship.
