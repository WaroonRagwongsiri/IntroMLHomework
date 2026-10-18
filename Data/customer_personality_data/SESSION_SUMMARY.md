# Session Summary (continuation note — 2026-10-10)

## Project context
IntroML course project on `marketing_campaign.csv` (2,237 rows) in `Data/customer_personality_data/`.
Two supervised tracks: **classification** (`Response`, 85/15 imbalance) and **regression**
(`Income`, right-skewed, $666,666 outlier). Standing convention: every version = PyTorch
oracle + from-scratch NumPy twin with parity checks. Brief mandates **1 novel regression +
1 novel classification model** (distinct, non-taught, built from scratch).

## What was accomplished (all executed, verified)

### 1. v4 models built + trained
- Classification `FT-Transformer-lite` (per-feature tokenizer + CLS, 2 PreNorm attention
  blocks, 18,625 params): F1 0.5464, ROC-AUC 0.8501.
- Regression `NAM` (13 per-feature subnets h=64 + log1p target + Huber δ=0.5, 2,510 params):
  R² 0.7232, RMSE $11,307. The $666k outlier is KEPT (robustness via log+Huber, not deletion).
- Scratch twins pass numeric grad-checks (clf relerr 3.8e-9, reg relerr 3.5e-12);
  reg parity relRMSE 0.003; clf F1 Δ 0.0084.
- Notebooks: `classification_pytorch_v4.ipynb`, `classification_scratch_v4.ipynb`,
  `regression_pytorch_v4.ipynb`, `regression_scratch_v4.ipynb` (+ v1–v3 pairs); weights in
  `weights/`, `weights_v2/`, `weights_v3/`, `weights_v4/`.

### 2. `media/` folder (26 files, all model-generated)
- 8 × `arch_*.onnx` (all pass `onnx.checker`; open in Netron to inspect).
- 8 × `arch_*.png` forward-flow graphs (input → … → output), rendered from the `.onnx` files.
- 8 × scratch loss curves (`loss_clf_v1–v4.png`, `loss_reg_v1–v4.png`).
- `attn_clf_v4.png` (CLS attention: Recency 0.118 / Tenure 0.092 on top),
  `shapes_reg_v4.png` (top-6 NAM shape functions).

### 3. `onnx_viz.py` (shared lib)
- One call per notebook: `save_graph(fresh, "arch_clf_v1", 22)` exports `.onnx` (opset 17)
  and renders the PNG. Also exposes `render_graph()` for label-only refreshes.
- `OP_LABELS = {"Gemm": "Linear"}` — ONNX has no `Linear` op; torch exports `nn.Linear`
  as `Gemm` (Y = A@B + C). Graphs read `Linear` like the code.
- Replaced the earlier torchviz approach (backward autograd graphs) per user request.

### 4. `REPORT.md` restructured (current format — keep following it)
- Title → 1-line goal per track → Metrics preamble (**F1** main / **RMSE** main).
- Per version: file list → `### Classification` / `### Regression` → Architecture
  (name item + `Linear()` chain item + ONNX graph) → feature tables
  (`| Feature (votes) | Description |`, Cut lines removed) → atomic one-fact-per-bullet
  Training lists → strict 4-column results tables → scratch loss graph → Why / What-next.
- No verdict/tolerance/shared-notes language anywhere.

## Scoreboard (oracle, test)
- clf F1: v1 0.5165 → v2 **0.5699** → v3 0.4706 → v4 0.5464.
- reg R²: v1 0.7287 → v2 **0.7673** → v3 0.7519 → v4 0.7232.
- v2 is raw-score best on both tracks; v4 fills the brief's novelty slots + interpretability story.

## Open / next steps
- Slide-deck figures for Oct 12 (attention heatmap, NAM shapes + Huber illustration into `slide_deck.html`) — offered, not yet requested.
- Optional cosmetic: `Relu → ReLU` label mapping (only `Gemm → Linear` applied so far).

## Standing user preferences (persist these)
- Model-generated artifacts only (no hand-drawn diagrams; torchviz rejected in favor of ONNX+Netron).
- One shared `.py` lib + one-line notebook calls over duplicated code.
- REPORT: exact full text everywhere (no "same as vX"), atomic bullets, graphs over prose, 4-column result tables only.
