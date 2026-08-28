## Google Play Store Apps
> https://www.kaggle.com/datasets/lava18/google-play-store-apps

### Status: backup dataset, not primary
This dataset is a **backup**, built out in case `banking_dataset/` (the primary dataset for this project) turns out to need replacing — e.g. if the "Financial-related / Investment-related / Fraud-related / Customer-commercial-related" domain guidance in the project description turns out to be a hard requirement rather than a suggestion. `banking_dataset/` remains the primary dataset. This folder holds a working, correctly-executed pair of notebooks and narrative notes (no presentation deck — that's only worth building if this dataset is ever actually presented).

**⚠️ Before ever presenting this dataset as primary:** check the domain-scope question above against the class sign-up sheet or the instructor. Web apps/mobile-app analytics is a stretch fit at best against the four suggested domains (arguably reachable under "Customer/commercial analysis," but `Price` is the only genuinely financial column in the whole dataset).

### Context
> While many public datasets (on Kaggle and the like) provide Apple App Store data, there are not many counterpart datasets available for Google Play Store apps anywhere on the web. On digging deeper, most of the data sets are incomplete, with either the categorical genre, or the reviews, or the app rating being unavailable. This dataset was scraped from the Google Play Store and is one of the more complete Play Store datasets publicly available, combining basic app metadata (category, size, install count, price, content rating, genres) with app performance signals (rating, review count).

> A companion file, `googleplaystore_user_reviews.csv` (~64k rows of per-review sentiment data, joinable on `App`), is also present in this folder but is not used by either notebook — the core analysis here only needs `googleplaystore.csv`.

### Content
> `googleplaystore.csv`: **10,841 rows x 13 columns**, one row per app listing (not one row per unique app — see duplicates note below).

### Detailed Column Descriptions
1. `App`: application name (free text, not used as a model feature — a unique identifier, not a predictor)
2. `Category`: app's listed category (categorical, 33 values, e.g. `FAMILY`, `GAME`, `TOOLS`, `FINANCE`)
3. `Rating`: average user rating, 1.0-5.0 (numeric — **chosen regression target**; **1,474 of 10,841 rows (13.6%) missing**)
4. `Reviews`: number of user reviews (numeric, loaded as a string in the raw file, parsed to `Reviews_num`)
5. `Size`: app size (string, e.g. `"19M"`, `"201k"`, or the literal value `"Varies with device"` — 1,695 rows post-cleaning; parsed to `Size_num` in MB, `"Varies with device"` kept as an explicit `NaN`, not imputed or dropped)
6. `Installs`: install-count bucket as a string, e.g. `"10,000+"`, `"1,000,000+"` (parsed to `Installs_num`; **basis of the engineered `hit`/`flop` classification target** — `hit = Installs_num >= 1,000,000`)
7. `Type`: `Free` or `Paid` (categorical, 2 values; 1 row missing in the raw file, dropped along with the one corrupted row below)
8. `Price`: listed price as a string, e.g. `"0"`, `"$4.99"` (parsed to `Price_num`, a float)
9. `Content Rating`: age/content rating, e.g. `Everyone`, `Teen`, `Mature 17+` (categorical, 6 values; 1 row missing in the raw file)
10. `Genres`: sub-genre tag(s), often a finer-grained restatement of `Category` (categorical, **119 distinct values** post-cleaning — see the redundancy note below; some apps list multiple genres separated by `;`, treated here as a single compound category rather than split out)
11. `Last Updated`: last-update date as a string (not used as a model feature in either notebook — no time-series/seasonality track was built for this dataset, unlike `banking_dataset/`'s `month`-based deep dive)
12. `Current Ver`: current app version as a string (not used as a model feature; 8 rows missing in the raw file)
13. `Android Ver`: minimum required Android version as a string (not used as a model feature; 3 rows missing in the raw file)

#### Target variables
- **Regression target: `Rating`.** Rows with missing `Rating` are dropped in `regression.ipynb` (imputing the target itself would be the wrong move) — 8,892 of 10,357 cleaned rows remain.
- **Classification target: `hit`** (engineered, not a raw column). `hit = 1` if `Installs_num >= 1,000,000`, else `0`. Built in both notebooks; does not depend on `Rating`, so `classify.ipynb` keeps all non-corrupted, non-duplicate rows (10,357). Confirmed split: **39.1% hit / 60.9% flop** on the full cleaned classify dataset (45.6%/54.4% on the smaller Rating-complete regression subset, since dropping missing-`Rating` rows isn't a random sample of installs).

#### Missing values (raw file)
`Rating` 1,474, `Type` 1, `Content Rating` 1, `Current Ver` 8, `Android Ver` 3. `Size` and `Installs`/`Price` have no raw missing values, but `Size`'s `"Varies with device"` placeholder (1,695 rows) becomes an explicit `NaN` in `Size_num` after parsing.

#### Data-quality issues found and handled (both notebooks, `## Data Cleaning`)
- **One corrupted row**, index `10472` ("Life Made WI-Fi Touchscreen Photo Frame") — missing `Category`, so every field from `Rating` onward shifted one column left (`Rating` reads as an impossible `19.0`). Dropped explicitly, confirmed by inspection before dropping.
- **1,181 duplicate `App` names** — mostly legitimate (the same app listed more than once, e.g. under different categories or at different scrape times), not treated as errors.
- **483 exact full-row duplicates** — genuinely redundant rows, dropped.
- **`Genres` is largely a finer-grained restatement of `Category`** — Cramér's V between the two is **0.969** (see `CLASSIFY_NOTES.md`'s categorical redundancy section), close to the maximum of 1.0. Both are still kept in the analysis (with `Genres`'s relevance/importance scores explicitly flagged for small-sample inflation, given 119 categories vs. `Category`'s 33), but this redundancy is why `Genres` should not be read as an independent second strong feature on top of `Category`.

### Citation
> Please cite the original Kaggle dataset if used for further analysis:

> Lavanya Gupta, "Google Play Store Apps," Kaggle, 2019. https://www.kaggle.com/datasets/lava18/google-play-store-apps

Licensed under **CC BY 3.0 Unported** (see `license.txt`) — https://creativecommons.org/licenses/by/3.0/
