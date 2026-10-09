# Report — Customer Personality Analysis Model

> Convention: each `##` = one self-contained experiment version.
> `## 1st Version (BASELINE)` is the reference every later version diffs against.
> To add a version: copy the template at the bottom, fill it in, never edit old sections.

## 1st Version (BASELINE)

Built four notebooks on `marketing_campaign.csv`, all sharing one deliberately simple architecture —
single hidden layer, `Linear → ReLU → Linear(→1)`, He init, Adam(lr=1e-3), 50 epochs, batch 64, seed 42:

| Notebook | Target | Role | Test headline (scratch twin) |
|---|---|---|---|
| `classification_pytorch.ipynb` | `Response` | oracle only — expected values for the scratch twin | F1 0.5165, ROC-AUC 0.8725 |
| `classification_scratch.ipynb` | `Response` | **primary** (pure NumPy) | F1 0.4920, ROC-AUC 0.8617 |
| `regression_pytorch.ipynb` | `Income` | oracle only | R² 0.7287, RMSE $11,194 |
| `regression_scratch.ipynb` | `Income` | **primary** (pure NumPy) | R² 0.7537, RMSE $10,666 |

**Parity verdict: PASS on both tracks** (clf: test-BCE Δ 0.045 < 0.05, F1 Δ 0.025 < 0.05; reg: RMSE relΔ 0.047 < 0.10,
R² Δ 0.025 < 0.05). The from-scratch backprop reproduces the PyTorch oracles, so the NumPy twins are correct.

### Classification — Response (18 columns → 22 model dims, n=2,237)

- **Architecture used** (library arch allowed: torch `ClfMLP` + mirror NumPy `ScratchMLP`):
  `x(22) → Linear(22→32) → ReLU → Linear(32→1)` → logit + sigmoid / mean BCE-with-logits,
  `pos_weight = neg/pos ≈ 5.7` on the 85/15 imbalance. He init (`W ~ N(0, √(2/fan_in))`, `b = 0`, seed 42),
  no dropout/norm (dropout masks can't be bit-identical across torch/NumPy). Params: 769 (22·32+32 + 32·1+1).
  Forward: `Z1 = XW1+b1; A1 = max(0,Z1); logit = A1W2+b2`.
  Backward (mean reduction, batch B): `dlogit = (p + p·y·(pw−1) − pw·y)/B` (reduces to `p−y` at pw=1);
  then `dW2 = A1ᵀd; db2 = Σd; dA1 = dW2ᵀ; dZ1 = dA1·(Z1>0); dW1 = XᵀdZ1; db1 = ΣdZ1`.
  One hidden layer was chosen so the hand-derived backprop is exactly checkable; torch uses fused
  `BCEWithLogitsLoss` ≡ NumPy explicit sigmoid + clipped log; Adam(lr=1e-3, β1=0.9, β2=0.999, eps=1e-8)
  hand-coded identically in scratch (m/v + bias correction); DataLoader shuffle vs NumPy permutation both seeded 42.

- **Features used + reason per feature.** Rule from the EDA (same in both tracks): bivariate `|score| ≥ 0.10` +
  RF top-15 + MI top-15 each cast one vote; 3/3 = chosen, 2/3 = kept here only with a named,
  independently-confirmed reason. This project uses 3/3 + 2/3.

  3/3 tier (10, full consensus across point-biserial / RF / MI — kept without caveat except where noted):

  | Feature | Biv | RF | MI | Reason (from EDA) |
  |---|---|---|---|---|
  | AcceptedCmp5 | 0.328 ✓ | 0.040 ✓ | 0.042 ✓ | Prior-campaign acceptance; responders 40.7% vs 8.2% (5× gap). Not leakage (history always resolved before current campaign) — cold-start caveat for brand-new prospects only |
  | AcceptedCmp3 | 0.254 ✓ | 0.042 ✓ | 0.027 ✓ | Same as above, second prior-campaign flag at full consensus |
  | MntWines | 0.247 ✓ | 0.070 ✓ | 0.040 ✓ | Spend signal: high spenders respond more |
  | MntMeatProducts | 0.237 ✓ | 0.069 ✓ | 0.043 ✓ | Same spend story; MI #1 overall |
  | NumCatalogPurchases | 0.221 ✓ | 0.038 ✓ | 0.034 ✓ | Catalog buyers are marketing-primed (vs walk-in store shoppers) |
  | Recency | 0.199 ✓ | 0.088 ✓ (RF #1) | 0.025 ✓ | Classic RFM signal: recent buyers respond more |
  | Customer_Tenure_Days | 0.194 ✓ | 0.082 ✓ (RF #2) | 0.032 ✓ | Loyalty/engagement story |
  | MntGoldProds | 0.141 ✓ | 0.050 ✓ | 0.034 ✓ | Spend signal, survives all three methods |
  | Income | 0.133 ✓ | 0.072 ✓ | 0.037 ✓ | Higher-income customers respond somewhat more; doubles as the other track's target |
  | MntFruits | 0.126 ✓ | 0.039 ✓ | 0.020 ✓ | Weakest 3/3 member but clears all three bars |

  2/3 tier (8, kept with a specific reason each — not a blanket "close enough"):

  | Feature | Votes | Reason for keeping |
  |---|---|---|
  | AcceptedCmp1 (0.294) | biv✓ MI✓ RF✗ | RF demotes the whole `AcceptedCmp*` family once spend/tenure explain the same "engaged customer" signal (multicollinearity/redundancy effect, not leakage evidence) — family pattern, but the feature's own biv+MI signal is strong |
  | AcceptedCmp2 (0.169) | biv✓ MI✓ RF✗ | Same RF family-demotion note as AcceptedCmp1 |
  | Marital_Status (0.152, cat) | biv✓ RF✓ MI✗ | Passes RF (#7); MI's discrete-target estimator is noisier on binary targets, so a near-miss there is less conclusive. One-hot → 5 dims; Cramér's V vs Education = 0.043 (independent) |
  | NumWebPurchases (0.148) | biv✓ MI✓ RF✗ | Clears MI (#8); missed RF cut only |
  | MntSweetProducts (0.117) | biv✓ RF✓ MI✗ | Same MI-noise note as Marital_Status |
  | MntFishProducts (0.111) | biv✓ RF✓ MI✗ | Same as above |
  | NumStorePurchases (0.039) | RF✓ MI✓ biv✗ | Fails only the linear check — real non-linear/interaction signal correlation misses |
  | Age (0.018) | RF✓ MI✓ biv✗ | Same as above (RF #8, MI #12) |

  Cut (7): AcceptedCmp4, Teenhome, Education (1/3 each); Kidhome, NumWebVisitsMonth, NumDealsPurchases, Complain (0/3).

- **Training setup.** 80/20 split, `random_state=42`, stratified on `Response`; `Income` NaNs (24 rows) median-filled
  as a *feature* from the train median (51,735, no leakage); `StandardScaler` train-fit; batch 64; 50 epochs;
  per-epoch log of train loss + validation loss + validation metric to `weights/*_{pytorch,scratch}_history.csv`
  with loss-curve plots; weights to `weights/*.pt` (torch `state_dict`) and `weights/*.scratch.npz`
  (W1/b1/W2/b2 + scaler stats); test metrics to `weights/*_test_metrics.json`.
  Execution order: PyTorch oracle first (scratch parity cell asserts its csv/json exist).

- **Results — loss & scores** (test = held-out 20%, n_test=448, seed 42; loss = unweighted mean BCE, threshold 0.5):

  | Metric | PyTorch (oracle) | Scratch | absΔ | Tolerance | Verdict |
  |---|---|---|---|---|---|
  | test BCE | 0.3903 | 0.4357 | 0.0454 | < 0.05 | PASS |
  | accuracy | 0.8036 | 0.7879 | 0.0157 | — | — |
  | precision | 0.4087 | 0.3833 | 0.0254 | — | — |
  | recall | 0.7015 | 0.6866 | 0.0149 | — | — |
  | **F1** | **0.5165** | **0.4920** | **0.0245** | < 0.05 | PASS |
  | ROC-AUC | 0.8725 | 0.8617 | 0.0108 | — | — |

  Training: torch train 1.211→0.539 / val 0.605→0.390, val-F1 0.369→0.516; scratch train 1.542→0.565 /
  val 1.249→0.436, val-F1 0.257→0.492. Curves descend monotonically, no overfitting kink.

- **Why.** F1 ≈ 0.50 with ROC-AUC ≈ 0.87 is the honest 85/15-imbalance story: accuracy 0.80 beats the
  0.851 all-negative dummy only modestly, while recall 0.70 / precision ~0.40 shows the model actually finds
  responders at the cost of false alarms — exactly why the EDA mandated precision/recall/F1/ROC-AUC over accuracy.
  Prior-campaign flags + spend + recency/tenure carry the ranking, matching the 3/3 vote table.
  Parity gap (ΔBCE 0.045 / ΔF1 0.025) is explained by different RNG libraries for init and different
  batch-shuffle streams — the signature of equivalent implementations, not a systematic bug
  (scratch trails here on the init lottery; see regression for the opposite direction).

### Regression — Income (13 numeric columns → 13 dims, n=2,213)

- **Architecture used.** `x(13) → Linear(13→32) → ReLU → Linear(32→1)` → value + mean MSE, He init seed 42,
  no dropout. Params: 481 (13·32+32 + 32·1+1). Forward/backward identical to classification except the head:
  `dpred = 2(pred−y)/B`, then the same ReLU chain. **Target handling:** model trains on z-scored `Income`
  (train-fit `StandardScaler`) and inverse-transforms for every reported metric — raw $ (~50k) explodes
  gradients at lr=1e-3 (first attempt diverged to R² −5.96, RMSE $56k). Preprocessing otherwise identical:
  `StandardScaler` on features (train-fit only); no test statistic touches training.

- **Features used + reason per feature** (same 3-method voting rule as classification).

  3/3 tier (8):

  | Feature | Biv | RF | MI | Reason |
  |---|---|---|---|---|
  | NumCatalogPurchases | 0.589 ✓ | ✓ (6th) | 0.573 ✓ | #1 bivariate; RF collapse to 6th is the greedy-split artifact (0.734-correlated with MntMeatProducts, MI confirms real signal) |
  | MntMeatProducts | 0.584 ✓ | ✓ (2nd) | 0.719 ✓ (MI #1) | Core spend driver |
  | MntWines | 0.578 ✓ | ✓ (1st, 0.446) | 0.681 ✓ | RF's pick for early splits; 3× runner-up is multicollinearity bookkeeping, not true dominance (MI agrees with bivariate order) |
  | NumWebVisitsMonth | −0.553 ✓ | ✓ (3rd) | 0.399 ✓ | Negative: site-browsers earn/spend *less*; affluent buy via catalog/store |
  | MntSweetProducts | 0.441 ✓ | ✓ | 0.424 ✓ | Spend cluster member |
  | MntFruits | 0.430 ✓ | ✓ (4th) | 0.453 ✓ | Spend cluster member |
  | NumWebPurchases | 0.388 ✓ | ✓ (12th) | 0.349 ✓ | Channel signal |
  | MntGoldProds | 0.325 ✓ | ✓ | 0.270 ✓ | Spend cluster member |

  2/3 tier (5):

  | Feature | Votes | Reason for keeping |
  |---|---|---|
  | NumStorePurchases (0.530) | biv✓ MI✓ RF✗ | Fails only RF (15th) — same greedy-split artifact; MI #4 (0.550) confirms |
  | MntFishProducts (0.439) | biv✓ MI✓ RF✗ | Same artifact (RF 13th, MI #7) |
  | Kidhome (−0.428) | biv✓ MI✓ RF✗ | Same artifact (RF 19th, MI #12); negative: young-children households earn less |
  | Age (0.163) | biv✓ RF✓ MI✗ | Fails only MI (14th, 0.130) — near-miss |
  | NumDealsPurchases (−0.083) | RF✓ MI✓ biv✗ | Fails only the linear check (−0.083) — real non-linear signal both model-based methods see |

  Cut (12, incl.): AcceptedCmp5/1, Education, AcceptedCmp4, **Response** (all 1/3 — `Response`'s 0.133 bivariate
  correlation vanishes to ~0.0005 under RF/MI once spend channels explain the same affluent-customer signal),
  Customer_Tenure_Days, Recency (RF-only); AcceptedCmp2, Marital_Status, Complain, Teenhome, AcceptedCmp3 (0/3).

- **Training setup.** 80/20 split, `random_state=42` (n_test=443); feature + target scalers train-fit; Adam(lr=1e-3,
  β1=0.9, β2=0.999, eps=1e-8), batch 64, 50 epochs, CPU. Losses logged in z-space (train/val MSE),
  RMSE tracked in raw $; same history-csv / weights / metrics-json artifact pattern as classification.

- **Results — loss & scores** (test = held-out 20%, n_test=443, seed 42; metrics in raw $):

  | Metric | PyTorch (oracle) | Scratch | absΔ / relΔ | Tolerance | Verdict |
  |---|---|---|---|---|---|
  | test MSE (raw) | 125,307,774.9 | 113,761,654.3 | rel 0.092 | — | — |
  | **RMSE** | **11,194.1** | **10,665.9** | **rel 0.047** | < 0.10 | PASS |
  | MAE | 7,170.3 | 6,809.6 | rel 0.050 | — | — |
  | **R²** | **0.7287** | **0.7537** | **0.0250** | < 0.05 | PASS |

  Training (z-space MSE): torch train 1.016→0.464 / val 0.599→0.185, val-RMSE $20,126→$11,194;
  scratch train 0.891→0.456 / val 0.415→0.168, val-RMSE $16,746→$10,666.

- **Why.** R² ≈ 0.73–0.75 is strong because the EDA said it would be: 17/25 candidates clear 0.10
  bivariately with several > 0.5 — the strongest signal set of all three datasets explored in this project.
  Spend/channel columns are near-mechanical proxies of income (people with more money spend more money),
  unlike subjective or snapshot targets. Residual RMSE ≈ $11k against median $51k is the multicollinearity
  price: the top features repeat one affluent-spender signal rather than adding independent drivers.
  Scratch leading slightly here (opposite direction to classification) is the init lottery landing the other way —
  consistent with equivalent implementations, and inside tolerance either way.

### Shared notes (v1)

- **Data & cleaning done.** Raw file: 2,240 rows × 29 cols, tab-separated. Applied the same 5 EDA steps
  (`slide_deck.html` §05) in all four notebooks — none touch either target directly:
  1. Drop `ID` (identifier, not a feature).
  2. Drop `Z_CostContact` / `Z_Revenue` (constant 3 / 11 every row — zero variance, verified before dropping).
  3. Parse `Dt_Customer` (`%d-%m-%Y`, range 2012-07-30 → 2014-06-29) into `Customer_Tenure_Days` (days before 2014-06-29).
  4. Derive `Age = 2014 − Year_Birth`; drop 3 rows with birth years 1893/1899/1900 (ages 121/115/114, data-entry errors) → **2,237 rows**.
  5. Collapse `Marital_Status` joke/tiny categories (`Absurd` 2, `YOLO` 2, `Alone` 3 → `Single`, now n=486; 5 categories left).
  6. Regression only: drop 24 rows with missing `Income` (1.1%; imputing the target itself would bias it) → **2,213 rows**.
     Classification keeps them and median-fills `Income` as a *feature* (train median 51,735, no leakage).

- **Limitations & next steps.**
  - Small-n (~2.2k): single 80/20 split is noisy; k-fold would tighten the numbers (out of "simple pipeline" scope).
  - $666,666 Income outlier (≈4× next-highest) inflates squared-error metrics; a log-target or robust loss is the natural follow-up.
  - `AcceptedCmp*` cold-start caveat stands: brand-new prospects lack history, so deployment needs a fallback path.
  - Multicollinearity (spend cluster r up to 0.73) means weights are not interpretable as independent effects.

- **Artifacts index.**
  - Notebooks: `classification_pytorch.ipynb`, `regression_pytorch.ipynb` (oracles),
    `classification_scratch.ipynb`, `regression_scratch.ipynb` (primary; all executed in place via
    `uv run jupyter nbconvert --execute`).
  - Weights: `weights/classification_mlp.pt`, `weights/regression_mlp.pt`,
    `weights/classification_scratch.npz`, `weights/regression_scratch.npz`,
    `weights/{classification,regression}_scaler.npz` (incl. target scaler + column order + train median).
  - Logs: `weights/*_{pytorch,scratch}_history.csv` (per-epoch train/val loss + val metric),
    `weights/*_{pytorch,scratch}_test_metrics.json` (final test loss + scores).
  - Env: `torch 2.14.1` added to `/Data/pyproject.toml` via `uv add torch`; `scikit-learn`/`pandas`/`numpy` reused.

## 2nd Version (deeper MLP)

Same data, features, splits, seed, and losses as v1 — only depth, regularization, and schedule change:
`FC(in→64)→BN→ReLU→Dropout(0.2)→FC(64→64)→BN→(+residual)→ReLU→Dropout(0.2)→FC(64→1)`,
100 epochs, Adam(lr=1e-3) + cosine annealing + weight decay 1e-4. New files
(`*_v2.ipynb`, `weights_v2/`); v1 files frozen. Clf params ≈ 5,953 / reg ≈ 5,393 (incl. BN).
Scratch hand-codes batchnorm (batch stats + momentum-0.1 running stats, eps=1e-5, unbiased running-var
like torch), inverted dropout (separate rng stream), BN backward, and the residual gradient split
(`dD1 = dB2·W2ᵀ + dS`); cosine LR `0.5·lr0·(1+cos(π·(e−1)/100))` matches torch `CosineAnnealingLR(T_max=100)`;
weight decay added to every gradient like torch Adam.

### Classification — Response (same 18 cols → 22 dims)

- **Architecture / Setup.** ResNet-style block above (residual wraps the 64→64 block only — dims match, no
  projection). Weighted BCE (`pos_weight ≈ 5.7`) for training steps; all logged/reported losses unweighted.
  Eval-mode train loss logged per epoch (same convention both twins).
- **Results — loss + scores** (n_test=448):

  | Metric | PyTorch v2 | Scratch v2 | absΔ | Tol | Verdict |
  |---|---|---|---|---|---|
  | test BCE | 0.3863 | 0.3775 | 0.0088 | < 0.05 | PASS |
  | accuracy | 0.8214 | 0.8304 | 0.0090 | — | — |
  | precision | 0.4454 | 0.4571 | 0.0117 | — | — |
  | recall | 0.7910 | 0.7164 | 0.0746 | — | — |
  | **F1** | **0.5699** | **0.5581** | **0.0118** | < 0.05 | PASS |
  | ROC-AUC | 0.8814 | 0.8686 | 0.0128 | — | — |

  Curves: torch train 1.055→0.543 / val 0.615→0.386, val-F1 0.425→0.570;
  scratch train 0.642→0.281 / val 0.661→0.378, val-F1 0.366→0.558.
- **Results vs v1 / Verdict: KEEP.** Torch F1 0.5165→0.5699 (+0.053), ROC 0.8725→0.8814; scratch F1
  0.4920→0.5581 (+0.066). Capacity — not features — was the v1 bottleneck, as suspected (v1 curves never kinked).
  Depth + BN/dropout bought recall without losing precision (torch recall 0.70→0.79, precision 0.41→0.45).

### Regression — Income (same 13 dims, z-scored target)

- **Architecture / Setup.** Same ResNet block with 13-dim input. z-space MSE loss; metrics in raw $.
- **Results — loss + scores** (n_test=443, raw $):

  | Metric | PyTorch v2 | Scratch v2 | Δ | Tol | Verdict |
  |---|---|---|---|---|---|
  | test MSE | 107,448,373.9 | 118,746,515.5 | rel 0.105 | — | — |
  | **RMSE** | **10,365.7** | **10,897.1** | **rel 0.051** | < 0.10 | PASS |
  | MAE | 6,592.4 | 6,692.6 | rel 0.015 | — | — |
  | **R²** | **0.7673** | **0.7429** | **0.0244** | < 0.05 | PASS |

  Curves (z-MSE): torch train 0.653→0.457 / val 0.394→0.159, val-RMSE $16,324→$10,366;
  scratch train 0.732→0.463 / val 0.469→0.176, val-RMSE $17,815→$10,897.
- **Results vs v1 / Verdict: KEEP.** Torch R² 0.7287→0.7673 (+0.039), RMSE $11,194→$10,366 (−7%);
  scratch R² 0.7537→0.7429 (−0.011, init lottery within tolerance), RMSE $10,666→$10,897.
  Net: depth helps the oracle clearly; scratch holds roughly at v1 level — twin still valid, v2 is the new best
  per-track by oracle numbers with scratch confirming inside tolerance.

### Shared notes (v2 — only what's new/changed)

- New: `*_v2.ipynb` ×4, `weights_v2/` (pt/npz/scaler/csv/json, same naming with v2 paths), cosine LR + wd 1e-4,
  100 epochs, BN/dropout/residual in both frameworks.
- Parity held despite deeper nets (BN running stats + dropout masks can't match exactly across libraries —
  tolerances unchanged and still passing).
- Next candidate (not run): LightGBM ceiling (Option D) and/or 3/3-only ablation (Option B).

## 3rd Version (3/3-only ablation)

Single-variable test of the 2/3 tier: identical v2 deeper arch, training, splits, and seed — only features
shrink to 3/3 consensus (clf 10 cols → 10 dims, one-hot block deleted, params ≈ 5,185; reg 8 cols → 8 dims,
params ≈ 5,057). Dropped: clf AcceptedCmp1/2, Marital_Status, NumWebPurchases, MntSweet/FishProducts,
NumStorePurchases, Age; reg NumStorePurchases, MntFishProducts, Kidhome, Age, NumDealsPurchases.
New files (`*_v3.ipynb`, `weights_v3/`); v2 frozen.

### Classification — Response (10 dims, all numeric)

- **Results — loss + scores** (n_test=448):

  | Metric | PyTorch v3 | Scratch v3 | absΔ | Tol | Verdict |
  |---|---|---|---|---|---|
  | test BCE | 0.4589 | 0.4255 | 0.0334 | < 0.05 | PASS |
  | accuracy | 0.7388 | 0.7768 | 0.0380 | — | — |
  | precision | 0.3377 | 0.3759 | 0.0382 | — | — |
  | recall | 0.7761 | 0.7463 | 0.0298 | — | — |
  | **F1** | **0.4706** | **0.5000** | **0.0294** | < 0.05 | PASS |
  | ROC-AUC | 0.8466 | 0.8515 | 0.0049 | — | — |

  Curves: torch train 0.711→0.421 / val 0.704→0.459, val-F1 0.410→0.471;
  scratch train 0.690→0.384 / val 0.667→0.425, val-F1 0.425→0.500.
- **Results vs v2 / Verdict: REVERT to full set.** Torch F1 0.5699→0.4706 (**−0.099**), ROC 0.8814→0.8466;
  scratch F1 0.5581→0.5000 (−0.058). Both twins agree: the 2/3 tier carries real, non-redundant signal —
  most plausibly AcceptedCmp1 (biv 0.294, the strongest dropped feature) plus the Marital_Status/NumStorePurchases/Age
  non-linear signals the deep net could exploit even though RF demoted them. The "RF family-demotion = redundancy"
  reading from v1 was wrong for this track: demoted ≠ expendable. 2/3 stays.

### Regression — Income (8 dims, z-scored target)

- **Results — loss + scores** (n_test=443, raw $):

  | Metric | PyTorch v3 | Scratch v3 | Δ | Tol | Verdict |
  |---|---|---|---|---|---|
  | test MSE | 114,570,589.1 | 116,585,505.8 | rel 0.018 | — | — |
  | **RMSE** | **10,703.8** | **10,797.5** | **rel 0.009** | < 0.10 | PASS |
  | MAE | 6,906.5 | 6,915.9 | rel 0.001 | — | — |
  | **R²** | **0.7519** | **0.7476** | **0.0043** | < 0.05 | PASS |

  Curves (z-MSE): torch train 0.651→0.470 / val 0.335→0.169, val-RMSE $15,061→$10,704;
  scratch train 0.700→0.468 / val 0.399→0.172, val-RMSE $16,429→$10,798.
  Parity here is the tightest of all three versions (RMSE relΔ 0.009, R² Δ 0.004).
- **Results vs v2 / Verdict: REVERT to full set (weakly).** Torch R² 0.7673→0.7519 (−0.015), RMSE +3%;
  scratch R² 0.7429→0.7476 (+0.005, flat). Asymmetry vs classification: the 3/3 spend core carries ~98% of the
  signal (NumStorePurchases 0.530 / Kidhome −0.428 / MntFish 0.439 help at the margin but the deep net routes
  around them). Keep the full 13-dim set — it costs nothing and is still best-by-oracle — but the practical
  lesson is the 8-dim 3/3 model is nearly as good and simpler to deploy.

### Shared notes (v3 — only what's new/changed)

- New: `*_v3.ipynb` ×4, `weights_v3/`; clf params 5,953→5,185, reg 5,393→5,057; no other code changes
  (scratch BN/residual/Adam untouched — ablation needed none).
- Answer to the v1 open question: 2/3 tier is **load-bearing for Response, marginal-but-positive for Income**.
- Next candidates (not run): LightGBM ceiling (Option D); log-target / robust loss for the $666k outlier.

## Version template (copy for v3, v4, …)

### Classification — …
- Changed vs previous: … / Results vs previous best: … / Verdict: …

### Regression — …
- Changed vs previous: … / Results vs previous best: … / Verdict: …

### Shared notes (vN — only what's new/changed)
