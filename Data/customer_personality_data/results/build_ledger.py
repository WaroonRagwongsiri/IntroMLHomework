"""Build results/ledger.csv from every weights_V*/ metrics JSON.

Run from customer_personality_data/:  python results/build_ledger.py
One row per (version, task, framework, model, scheme, fold, repeat).
Source of truth stays in weights_V*/; this script only collects.
"""
import csv, glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- what each version is (edit here when a new version is added) -------------
# key: (version, task)  ->  (model, tier, selection)
#   tier      : curriculum | curriculum family | extracurricular
#   selection : how the epoch / round count was chosen, to keep the comparison honest
#               fixed (no selection) | inner validation | eval data (optimistic)
META = {
    ("V1_0", "classification"): ("MLP (1 hidden layer)", "curriculum", "fixed"),
    ("V1_1", "classification"): ("MLP (deep, 22 features)", "curriculum", "fixed"),
    ("V1_2", "classification"): ("MLP (deep, selected features)", "curriculum", "fixed"),
    ("V1_3", "classification"): ("MLP (deep, 22 features)", "curriculum", "fixed"),
    ("V1_0", "regression"): ("MLP (1 hidden layer)", "curriculum family", "fixed"),
    ("V1_1", "regression"): ("MLP (deep, 13 features)", "curriculum family", "fixed"),
    ("V1_2", "regression"): ("MLP (deep, selected features)", "curriculum family", "fixed"),
    ("V1_3", "regression"): ("MLP (deep, 13 features)", "curriculum family", "fixed"),
    ("V2_0", "classification"): ("FT-Transformer-lite", "extracurricular", "eval data (optimistic)"),
    ("V2_1", "classification"): ("FT-Transformer-lite", "extracurricular", "eval data (optimistic)"),
    ("V2_0", "regression"): ("NAM (log1p + Huber)", "extracurricular", "eval data (optimistic)"),
    ("V2_1", "regression"): ("NAM (log1p + Huber)", "extracurricular", "eval data (optimistic)"),
    ("V3_0", "classification"): ("LightGBM", "curriculum", "inner validation"),
    ("V3_1", "classification"): ("LightGBM", "curriculum", "inner validation"),
    ("V3_0", "regression"): ("LightGBM", "curriculum family", "inner validation"),
    ("V3_1", "regression"): ("LightGBM", "curriculum family", "inner validation"),
    # V4 ReGLU (Ask 2): ReLU baseline and two ReGLU sizes; each row also carries its own model name
    ("V4_0", "classification"): ("FT-Transformer-lite (ReLU / ReGLU)", "extracurricular", "inner validation"),
    ("V4_1", "classification"): ("FT-Transformer-lite (ReLU / ReGLU)", "extracurricular", "inner validation"),
    ("V4_0", "regression"): ("NAM (log1p + Huber) (ReLU / ReGLU)", "extracurricular", "inner validation"),
    ("V4_1", "regression"): ("NAM (log1p + Huber) (ReLU / ReGLU)", "extracurricular", "inner validation"),
    # V5 interaction features (Ask 1) and V6 curriculum models (Ask 3) mix several frameworks / models under one
    # version, so every row of those notebooks carries its own model / tier / selection and is read from the JSON.
}
IMPL = {"pytorch": "library", "scratch": "scratch", "lightgbm": "library",
        "sklearn": "library", "xgboost": "library", "catboost": "library"}

# metric name normalisation (older notebooks mix upper / lower case)
ALIAS = {"test_loss_BCE": "bce", "bce": "bce", "accuracy": "accuracy", "precision": "precision",
         "recall": "recall", "f1": "f1", "roc_auc": "roc_auc",
         "test_loss_MSE_raw": "mse", "test_MSE": "mse", "mse": "mse",
         "rmse": "rmse", "RMSE": "rmse", "mae": "mae", "MAE": "mae", "r2": "r2", "R2": "r2"}
METRICS = ["f1", "accuracy", "precision", "recall", "roc_auc", "bce", "r2", "rmse", "mae", "mse"]
COLS = (["version", "task", "framework", "impl", "model", "tier", "selection", "scheme",
         "repeat", "fold", "n_test", "best_iteration"] + METRICS + ["source"])


def norm(d):
    out = {}
    for k, v in d.items():
        if k in ALIAS and isinstance(v, (int, float)):
            out[ALIAS[k]] = v
    return out


def meta(version, task, model_hint=None):
    if (version, task) in META:
        return META[(version, task)]
    return (model_hint or "?", "?", "?")


rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "weights_V*", "*_metrics.json"))):
    version = re.search(r"weights_(V\d+_\d+)", path).group(1)
    base = os.path.basename(path)
    task, framework = base.split("_")[0], base.split("_")[1]
    d = json.load(open(path))
    if base.endswith("_test_metrics.json"):
        scheme = "single"
        units = ([dict(m, fold="", repeat="", n_test=d.get("n_test", "")) for m in d["models"]]
                 if "models" in d else [dict(d, fold="", repeat="")])
    elif base.endswith("_kfold_metrics.json"):
        scheme = "kfold"
        units = [dict(f, repeat=f.get("repeat", 1)) for f in d["folds"]]
    else:
        continue
    model, tier, selection = meta(version, task, d.get("model"))
    for u in units:
        row = {"version": version, "task": task, "framework": framework,
               "impl": IMPL.get(framework, "?"), "model": u.get("model", model),
               "tier": u.get("tier", tier), "selection": u.get("selection", selection),
               "scheme": scheme, "repeat": u.get("repeat", ""), "fold": u.get("fold", ""),
               "n_test": u.get("n_test", ""), "best_iteration": u.get("best_iteration", ""),
               "source": os.path.relpath(path, ROOT)}
        row.update(norm(u))
        rows.append(row)

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
out = os.path.join(ROOT, "results", "ledger.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}: {len(rows)} rows")
