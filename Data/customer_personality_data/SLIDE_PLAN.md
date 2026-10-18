# Plan — `slide_deck2.html`

Course 01076641 · Customer Personality Analysis · Waroon (68011008), Chanathip (68010203)
Status: **plan only — deck not built yet.** Revised after review (Part A shortened; Part B limited to architecture, features, result).

## Principles

- Same look as `slide_deck.html` (scroll-snap slides, same fonts/colours; red = classification, teal = regression). English.
- Figures: only the existing `media/*.png` (loss curves, k-fold charts). **No ONNX graphs.**
- Numbers copied from `REPORT.md` (models) and `CLASSIFY_NOTES.md` / `REGRESSION_NOTES.md` (EDA). Nothing recomputed.
- Build order is V1.0 → V1.1 → V1.2 → V2.0 → V3.0 → V1.3 → V2.1 → V3.1. No divider or timeline slide; each model slide carries a small label such as "step 4 of 8 · V2.0".

## Part A — Task and data (4 slides)

| # | Slide | Content |
|---|---|---|
| 1 | Title | Course code, team names |
| 2 | Task | `Response` (classification) and `Income` (regression); main metrics F1 and RMSE |
| 3 | Data | Rows before/after cleaning (2,240 → 2,236 clf, 2,212 reg); class balance 85/15; train/test split; Income statistics |
| 4 | Features (lists only) | Classification: 3/3 list (10) and 2/3 list (8). Regression: 3/3 list (8) and 2/3 list (5). No score tables. |

Dropped: the feature-voting-rule slide and the separate 3/3 and 2/3 table slides.

## Part B — Models in build order (one slide per version)

Each slide has two columns (classification | regression) with exactly three blocks:

1. **Model architecture** — layer chain on one line, plus a short note on what each layer does.
2. **Features used** — chips for the 3/3 and 2/3 features that version used, with input size.
3. **Result** — 4-column table (metric | PyTorch or LightGBM | scratch | Δ) plus the loss curve. k-fold slides show mean ± std plus the fold chart.

Removed from every slide: ONNX graph, training setup bullets, sanity/grad-check notes, Question/Decision beats, long "Why" paragraphs.

| # | Version | Architecture shown | Features used |
|---|---|---|---|
| 5 | V1.0 | `Linear(→32) → ReLU → Linear(→1)` | 3/3 + 2/3 (22 clf dims, 13 reg) |
| 6 | V1.1 | ResNet-style MLP: `Linear → BN → ReLU → Dropout → Linear → BN → +residual → ReLU → Dropout → Linear` | 3/3 + 2/3 |
| 7 | V1.2 | Same chain as V1.1 | 3/3 only (10 clf dims, 8 reg) |
| 8 | V2.0 | FT-Transformer-lite (clf); NAM + log1p + Huber (reg). May split into two slides if crowded. | 3/3 + 2/3 |
| 9 | V3.0 | LightGBM and scratch GBM, main settings on one line. May split into two slides if crowded. | 3/3 + 2/3 |
| 10 | V1.3 | Deep MLP as V1.1, written out in full (5-fold) | 3/3 + 2/3 |
| 11 | V2.1 | FT-Transformer-lite and NAM as V2.0, written out in full (5-fold) | 3/3 + 2/3 |
| 12 | V3.1 | LightGBM and scratch GBM as V3.0, written out in full (5-fold) | 3/3 + 2/3 |

## Part C — Wrap-up (3 slides, pending confirmation)

13. Scoreboard: single-split and 5-fold tables
14. Take-aways and limits
15. Questions

Total: 15 slides (17 if V2.0 and V3.0 are each split in two).

## Open questions before building

1. Keep the three wrap-up slides, or end the deck after V3.1?
2. One-line takeaway under each Result block, or only the table and chart?
3. Loss curve / fold chart under each table, or tables only (lighter slides)?
4. Wording of the 2/3 features: the EDA notes and `slide_deck.html` call them "cut", but the models use them (kept after the V1.2 ablation). The new deck would say "kept after V1.2"; the old deck's cut slide would then contradict it.
5. EDA scores were computed on 2,237 / 2,213 rows (before the $666,666 row was dropped); the models use 2,236 / 2,212. Keep a footnote on the data slide, or re-run the EDA first?

## Source numbers (for checking)

Single split, oracle / scratch (n_test 448 clf, 443 reg):

| Version | clf F1 | clf ROC-AUC | reg R² | reg RMSE |
|---|---|---|---|---|
| V1.0 | 0.5161 / 0.4974 | 0.8669 / 0.8587 | 0.7423 / 0.7695 | $11,065 / $10,463 |
| V1.1 | 0.5455 / 0.5158 | 0.8818 / 0.8625 | 0.7774 / 0.7595 | $10,284 / $10,688 |
| V1.2 | 0.4566 / 0.4645 | 0.8452 / 0.8463 | 0.7562 / 0.7586 | $10,761 / $10,708 |
| V2.0 | 0.6038 / 0.5197 | 0.8657 / 0.8251 | 0.7233 / 0.7286 | $11,464 / $11,353 |
| V3.0 | 0.6014 / 0.0000 | 0.8898 / 0.8310 | 0.8212 / 0.8213 | $9,217 / $9,213 |

5-fold mean ± std, oracle / scratch:

| Version | clf F1 | reg R² | reg RMSE |
|---|---|---|---|
| V1.3 | 0.5508±0.0191 / 0.5565±0.0335 | 0.7553 / 0.7573 | $10,603 / $10,564 |
| V2.1 | 0.5845±0.0352 / 0.5720±0.0363 | 0.7395 / 0.7439 | $10,957 / $10,867 |
| V3.1 | 0.5944±0.0447 / 0.5771±0.0509 | 0.8260 / 0.8234 | $8,915 / $8,987 |
