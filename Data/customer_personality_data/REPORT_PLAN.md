# แผนการจัดทำรายงานฉบับจริง (Thai Report) — Customer Personality Analysis

Course 01076641 · ML Project for Case Analysis · Team: Waroon (68011008), Chanathip (68010203)
Sources: `REPORT.md`, `CLASSIFY_NOTES.md`, `REGRESSION_NOTES.md`, `MATRIX_METHODS.md`, `README.md`, project brief PDF.
(`SESSION_SUMMARY.md` is **stale** — old v1–v4 numbering, outlier kept, old scores. Do not use it as a source.)

---

## 0. Decisions needed before writing (3 conflicts in the source files)

| # | Issue | Evidence | Recommended resolution |
|---|---|---|---|
| D1 | **Row counts differ between EDA and models.** EDA notes/notebooks run on 2,237 rows (clf) / 2,213 rows (reg) with the `Income = $666,666` row still in (`describe()` max = 666,666). Every model version drops that row → 2,236 / 2,212. | `classify.ipynb`, `regression.ipynb` outputs vs `REPORT.md` "Data cleaning note" | Add the outlier drop to Data Cleaning step 4b in both EDA notebooks, re-run, refresh the tables in the notes. Pearson/η scores against `Income` are the most outlier-sensitive. If you do not re-run, the report must carry an explicit "EDA computed before the outlier drop" note. |
| D2 | **The EDA says "2/3 → generally cut"; the models use the 2/3 tier.** Notes name only 3 promoted regression features (`NumStorePurchases`, `Kidhome`, `MntFishProducts`) but the 13-feature set also contains `Age` and `NumDealsPurchases`; every classification 2/3 feature is also kept. `slide_deck.html` still says "2/3 — cut (8)". | `CLASSIFY_NOTES.md` vs `REPORT.md` V1.0 | Tell it as the real story: the vote rule gives a conservative **core (3/3)**; the **V1.2 ablation** then tested the rule empirically (dropping 2/3: clf torch F1 −0.089, scratch −0.051; reg torch R² −0.021, scratch −0.001) → the 2/3 tier is **kept**. Update the "Notable 2/3 cases" paragraphs in the notes and the cut slide in `slide_deck.html` so they do not contradict the report. |
| D3 | **GBDT (V3.0) is described as "not a taught-list submission".** But Gradient Boosting is on the brief's taught list (B: "Stacking, GB, XGBoost"). | brief p.2, `REPORT.md` V3.0 | In the report, V3.0 = the brief's *Gradient Boosting* item (library + from-scratch CART twin) **and** the ceiling check. The two *novel* models remain FT-Transformer-lite (clf) and NAM (reg). |

## 1. Coverage check against the brief (gaps found — fix or disclose)

| Brief requirement | Status in current files | Action in report |
|---|---|---|
| Problem setting, storytelling, feature insight (30 pts) | ✅ README, deck slide "Why This Analysis?" | Ch.1 + Ch.2 |
| ≥ 5 features | ✅ 18 (clf) / 13 (reg) | Ch.2.9 |
| Feature importance (hint) | ✅ RF, MI, LightGBM gain, CLS attention, NAM shape functions | Ch.2.7–2.8, 5.4 |
| Novel model ×2 with internal mechanism in detail | ⚠ architecture strings exist; the *derivation* (forward/backward, attention, NAM, Huber/log1p) is not written out | Ch.4.4 "Internal mechanism" — extract from `*_scratch_V2_0.ipynb` |
| Taught regression: Linear / Multiple / Polynomial | ❌ none in `REPORT.md` | **Decision:** add scratch baselines (closed-form OLS, degree-2/3 polynomial) — cheap and gives the "much better than taught models" comparison for extra points; otherwise disclose scope. |
| Taught classification list (Logistic, Tree, RF, SVM, kNN, NB, k-means, …) | ❌ only MLP + GB. RF appears in EDA only | Minimum: scratch Logistic + reuse the V3.0 scratch CART as Decision Tree/RF + kNN. Otherwise disclose. |
| Confusion matrix, Specificity/TNR | ❌ not reported | Reconstruct from reported precision/recall/accuracy and class counts (n_pos=67, n_neg=381) — exact, no retrain — then confirm by reloading saved weights. |
| ROC curve (not just AUC), accuracy plot | ❌ AUC value only; loss plots only | Reload `weights_V*` and run inference on the fixed test split → ROC overlay + accuracy/F1-vs-epoch from `*_history.csv`. |
| Save/load model demo | ✅ weights saved (`.pt`, `.npz`, scalers, metrics json) | Add one "reload and re-score" appendix table proving metrics reproduce. |
| Deliverables: report, slides, ipynb/py, other docs | report (this plan), `slide_deck2.html`, notebooks ✅ | Appendix file inventory |

## 2. Format and style

- Language: Thai body text; English technical terms and all identifiers kept in English (`Recency`, F1, BatchNorm). Numbers in Arabic numerals. Captions "รูปที่ 2.1", "ตารางที่ 2.1".
- Thai academic layout: TH Sarabun New, body 16 pt, chapter heading 20 pt bold; margins left 1.5", others 1"; page numbers from Chapter 1; front matter in Thai numerals/ก ข ค.
- Recommended file: **.docx** (TOC, list of figures/tables, equations, 40+ figures, Thai font control) with PDF export. Alternative: Claude Docs artifact, export to Word/PDF.
- Your standing preferences carried over: model-generated figures only (ONNX graphs, matplotlib), graphs over prose, atomic bullets, strict 4-column result tables, full text (no "same as V1.x").
- Length target: ~50–60 pages main body + appendices.

## 3. Report outline

### ส่วนนำ (front matter)
Cover · บทคัดย่อ (TH) + Abstract (EN) · กิตติกรรมประกาศ · สารบัญ · สารบัญรูป · สารบัญตาราง

### บทที่ 1 บทนำ — (brief criteria 1–2)
1.1 ที่มาและความสำคัญ: one customer database, two decisions — *who to target* (`Response`, only 14.9% respond) and *how much they're worth* (`Income`, sensitive/often missing)
1.2 วัตถุประสงค์: classification + regression; two novel models; robustness check
1.3 ชุดข้อมูล: Kaggle `imakash3011/customer-personality-analysis` (CC0), 2,240 × 29, tab-separated; note original purpose was clustering. (One line on dataset selection; drop the comparisons with `banking_dataset`/`google_play_data` that are scattered through the notes.)
1.4 ขอบเขตและตัวชี้วัด: F1 (main) / RMSE (main)
1.5 โครงสร้างรายงาน

### บทที่ 2 การวิเคราะห์ข้อมูลเชิงสำรวจ (EDA) — *first section, already done*
Source mapping (section headers match the notes 1:1):

| § | Content | Source | Figures / tables to include |
|---|---|---|---|
| 2.1 | Column dictionary: People / Products / Promotion / Place (+ 2 dropped) | README | Table: 29 columns grouped |
| 2.2 | Data cleaning, 7 steps: drop `ID`; drop `Z_CostContact`/`Z_Revenue`; derive `Customer_Tenure_Days`; derive `Age` + drop 3 impossible birth years; collapse `Absurd`/`YOLO`/`Alone` → `Single`; keep/drop 24 missing `Income`; **drop $666,666 outlier** (D1) | CLASSIFY/REGRESSION_NOTES, REPORT | Row-count waterfall: 2,240 → 2,237 → 2,236 (clf) / 2,212 (reg) |
| 2.3 | Target distributions: `Response` 14.94 / 85.06 %; `Income` mean $51,959, median $51,371, skew | REPORT tables | Class bar; Income histogram; train/test split table |
| 2.4 | Relevance vs target — clf: point-biserial + Cramér's V (19/25 clear 0.10); reg: Pearson + η (17/25 clear 0.10). Prior-campaign flags: *not leakage* (response 40.7% vs 8.2%) + cold-start caveat. `NumWebVisitsMonth` r = −0.553 and `Kidhome` −0.428 for Income. Income by `Response` deep dive (+27.8% median). Education η table (`Basic` ≈ 2.5× lower) | notes | Two ranked bar charts with 0.10 line |
| 2.5 | Multicollinearity: top pairs (`MntMeatProducts`–`NumCatalogPurchases` 0.72–0.73 …) → one "affluent catalogue/store shopper" factor | notes | `multicollinearity_heatmap.png` + top-10 pair table |
| 2.6 | Categorical redundancy: Cramér's V(Education, Marital_Status) = 0.043 | notes | one number |
| 2.7 | RF importance (clf top-15; reg top-12). clf: `Recency`, `Tenure` on top and `AcceptedCmp5` demoted to 12th; reg: `MntWines` 0.446 = 3× runner-up | notes | Two importance bar charts |
| 2.8 | MI cross-check; the honest three-way disagreement | notes | Two MI bar charts |
| 2.9 | **Feature selection — three-vote rule** (bivariate ≥ 0.10, RF top-N, MI top-N; N = 15 clf / 12 reg), **3/3 table and 2/3 table per track**, final sets (clf 18 → 22 dims; reg 13) with the D2 reconciliation | notes + REPORT V1.0/V1.2 | 4 tables (clf 3/3, clf 2/3, reg 3/3, reg 2/3) |
| 2.10 | EDA → modelling decisions: imbalance → weighted BCE + F1/ROC-AUC; multicollinearity → BN/dropout/weight decay; skewed `Income` → log1p + Huber (V2.0); ~2,200 rows → k-fold (V1.3/2.1/3.1) | synthesis | Decision table |
| App. A | Math of each matrix: Pearson, point-biserial, η, Cramér's V (equations already exist as PNG in the folder) | MATRIX_METHODS | 4 equation images |

### บทที่ 3 ระเบียบวิธีและการตั้งค่าการทดลอง
3.1 Common protocol: 80/20 split, `random_state=42` (stratified for clf), train-fit scalers, `Income` median-fill (clf feature, train median 51,651) vs drop 24 rows (reg target), Adam(1e-3), batch 64, test n = 448 (67 positives) / 443
3.2 **Oracle + from-scratch twin** convention (PyTorch/LightGBM library model vs NumPy implementation, identical seeds) — how it satisfies "built from scratch + ready-to-use"; parity checks (numeric grad-check, relative gaps); batch/epoch runs; save/load artifacts
3.3 Metrics with formulas: BCE, accuracy, precision, recall = sensitivity, specificity = TNR, F1, ROC-AUC, MSE/RMSE/MAE/R²; why F1 not accuracy (all-negative baseline = 0.851)
3.4 k-fold protocol: `StratifiedKFold` / `KFold`, k=5, shuffle, seed 42

### บทที่ 4 ลำดับการพัฒนาโมเดลและเหตุผลเบื้องหลัง — *second section, chain of thought by timeline*
**Use the real build order**, not the V-number order (REPORT.md "Development Timeline"):
V1.0 → V1.1 → V1.2 → V2.0 → V3.0 → V1.3 → V2.1 → V3.1.
Open with a timeline figure (generated from the metrics JSON): 8 nodes, decision arrows, F1/R² per node.

Every version uses the same five beats (this is the "chain of thought"):
**คำถาม** (what we wanted to find out) → **สิ่งที่เปลี่ยน** (architecture + setup) → **ผลลัพธ์** (4-column table + loss curve) → **การตีความ** (Why) → **ตัดสินใจต่อ** (What next → leads to the next version)

| Order | Version | Question | Key result (single split, oracle / scratch) | Decision it triggered |
|---|---|---|---|---|
| 1 | V1.0 baseline | What does the taught 1-hidden-layer MLP get? | clf F1 0.516 / 0.497, AUC 0.867 · reg R² 0.742 / 0.770, RMSE $11,065 / $10,463 | Curves never kink → suspect capacity |
| 2 | V1.1 deeper ResNet-MLP | Is capacity the bottleneck? | clf F1 0.546 / 0.516 · reg R² 0.777 / 0.760 | Yes → now test whether the 2/3 features matter |
| 3 | V1.2 ablation (3/3 only) | Does the 2/3 tier carry signal? | clf F1 0.457 / 0.465 (−0.089 / −0.051) · reg R² 0.756 / 0.759 | Yes → keep 2/3 tier (**resolves D2**) |
| 4 | V2.0 novel models | Brief: 1 new model per task | FT-Transformer-lite F1 0.604 / 0.520 · NAM R² 0.723 / 0.729 | Does anything simple beat these? → ceiling check |
| 5 | V3.0 GBDT / LightGBM | How much of the deep-net score is hard-earned? | clf F1 0.601 / 0.000 (degenerate checkpoint) · reg R² 0.821 / 0.821 | GBDT wins reg clearly; single-split noise is suspect → k-fold |
| 6 | V1.3 k-fold MLP | Was the V1.1 split lucky? | F1 0.551 ± 0.019 / 0.557 ± 0.034 · R² 0.755 / 0.757 | Not lucky → repeat for V2/V3 |
| 7 | V2.1 k-fold novel | Is the V2.0 parity failure real? | F1 0.585 / 0.572 · R² 0.740 / 0.744 | Gap was lucky-epoch noise |
| 8 | V3.1 k-fold GBDT | Is the F1 = 0 collapse systematic? | F1 0.594 / 0.577 · R² 0.826 / 0.823 | No fold collapses → closes the investigation |

4.1–4.8 one subsection per version in the order above (each: architecture ONNX graph, features table, training setup bullets, results table, loss curve, Why, What next)
4.4 additionally contains **"Internal mechanism" (brief-mandated)** for both novel models: FT-Transformer-lite (per-feature tokenizer, CLS token, pre-norm attention equations, residual, CLS-query attention as importance) and NAM (per-feature subnets, additive sum, log1p/Huber(δ=0.5)/expm1), plus the scratch backward-pass derivation and the grad-check numbers (relerr 8.9e-10 / 5.1e-11).
4.9 Honest-reporting boxes: V2.0 clf parity fail (Δ 0.084) and V3.0 scratch F1 = 0 explained and then closed by V2.1/V3.1.

### บทที่ 5 ผลการทดลองเปรียบเทียบและการอภิปราย (brief criteria: evaluation, 30 pts)
5.1 Consolidated scoreboard — single split (oracle and scratch) and 5-fold mean ± std (two tables, 4 columns each)
5.2 Brief's metric checklist per final model: loss, confusion matrix, accuracy, precision, sensitivity, specificity/TNR, F1, ROC + AUC curve, performance curves, R² (new figures from §1 gaps)
5.3 Oracle-vs-scratch parity summary
5.4 Feature importance across models: attention (`Recency` 0.107, `NumCatalogPurchases` 0.104, `Tenure` 0.102), LightGBM gain (`Recency`, `Tenure`; `MntWines`, `MntMeatProducts`), NAM shapes, EDA RF — they agree that recency/tenure/spend, not the raw `AcceptedCmp*` flags, drive clf
5.5 Discussion / honest claims:
- Regression: GBDT is clearly best (R² ≈ 0.826 vs ≈ 0.755 MLP, 0.740 NAM; spread ≈ 0.04) — supports "ceiling is GBDT on ~2,200 rows".
- Classification: GBDT 0.594 vs FT-T 0.585 vs MLP 0.551 — differences are about one fold-std (0.035–0.05); **do not claim a clear winner**.
- **Checkpoint-on-test caveat:** V2.0/V2.1 pick the best epoch on the held-out fold, so they are optimistic versus V1.3 (fixed 100 epochs) and V3.1 (inner validation slice). The GBDT numbers are the cleanest and still win.
- Extra-points claim #2 ("much better than taught models") is defensible for regression only, and only if the taught baselines from §1 are added.

### บทที่ 6 สรุปและข้อเสนอแนะ
Findings per track · limitations (n ≈ 2,200, 67 test positives, test-set checkpointing, `AcceptedCmp*` cold-start for new customers, EDA/outlier note) · future work (separate validation split, TabPFN as pretrained ceiling, threshold tuning for the 85/15 imbalance, business lift/cost analysis)

### บรรณานุกรม
Kaggle dataset (Akash Patel, 2021; provenance note in README) · FT-Transformer (Gorishniy et al., NeurIPS 2021) · NAM (Agarwal et al., NeurIPS 2021) · LightGBM (Ke et al., NeurIPS 2017) · Adam (Kingma & Ba, 2015) · BatchNorm (Ioffe & Szegedy, 2015) · Dropout (Srivastava et al., 2014) · Huber (1964) · SGDR/cosine annealing (Loshchilov & Hutter, 2017). *Verify each citation before submission.*

### ภาคผนวก
A. Matrix equations · B. Full hyper-parameter tables · C. Per-fold tables (V1.3/2.1/3.1) · D. Save/load reproduction table · E. File inventory (notebooks, `weights_V*`, `media/`) + how to re-run · F. TH–EN glossary

## 4. Figure inventory

Reuse (already model-generated, in `media/`): 8 ONNX graphs `arch_*` (V2.0 graphs are 9260 × 4288 / 5352 × 1523 px → place full-page landscape or show a crop plus pointer to the `.onnx`), 10 loss curves, 6 k-fold charts, attention bars, NAM shapes, 2 LightGBM importance charts.
Extract from notebook outputs: EDA ranking bars, RF/MI bars, histograms (`classify.ipynb`, `regression.ipynb`); heatmap and 4 equation PNGs already in the folder.
New (script, reads `weights_V*/` + `*_history.csv`): timeline figure; confusion matrices; ROC overlay (clf); accuracy/F1-vs-epoch; scoreboard bar chart.

## 5. Work order

1. Resolve D1–D3 (re-run EDA with outlier drop if chosen; edit notes + `slide_deck.html` 2/3 wording).
2. Close metric gaps: confusion matrix / specificity / ROC / reload table; optional taught baselines.
3. Extract EDA figures + generate new figures.
4. Draft Ch.2 (EDA) → Ch.4 (timeline) → Ch.3 → Ch.5 → Ch.1/6 → front matter, bibliography, appendices.
5. Check pass: every number against `REPORT.md` and `weights_V*/…metrics.json`; figure/table numbering; Thai typography; page count.
6. Export `.docx` + PDF; align with `slide_deck2.html` (same numbers, same timeline order).

## 6. `slide_deck2.html`

See `SLIDE_PLAN.md` (revised plan: 4-slide Part A, one slide per model in build order showing only architecture, features used and result).
