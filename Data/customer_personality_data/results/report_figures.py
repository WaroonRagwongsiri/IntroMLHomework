"""Figures for the report, drawn from results/ledger.csv (15 shared outer folds).
Run from customer_personality_data/:  python results/report_figures.py"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A

ROOT = A.ROOT
COL = {"curriculum": "#4c78a8", "curriculum family": "#9aa5b1", "extracurricular": "#6a3d9a"}
SHORT = {"FT-Transformer-lite": "FT-Transformer", "NAM (log1p + Huber)": "NAM", " | ReLU": "", "(baseline)": "", " | ": " · ",
         "+ 3 modifiers + 5 pairs": "+3 features +5 pairs", "+ 3 modifiers (25)": "+3 features", "base (22)": "22 features",
         "+ Teenhome + 5 Teenhome pairs": "+Teenhome +5 pair nets", "13 features + 4 pair nets": "13 features +4 pair nets",
         "+ Teenhome (14)": "+Teenhome", "(cluster rate)": "", "LightGBM stumps (additive, depth 1)": "LightGBM depth 1",
         "LightGBM depth 2 (pairwise)": "LightGBM depth 2"}
def short(n):
    for a, b in SHORT.items():
        n = n.replace(a, b)
    return n.strip(" ·")

def table(task, metric):
    wide, tier = A.fold_table(task, metric)
    keep = [c for c in wide.columns if "[scratch]" not in c and "depth 1" not in c and "depth 2" not in c and wide[c].count() == 15]
    wide = wide[keep]
    s = A.summary(wide, tier.reindex(wide.columns))
    # drop exact duplicates (same model re-run in V5 on the same feature set gives identical folds)
    # V5 re-ran two baselines on the original feature set; keep the V4/V6 entry only
    s = s.drop(index=[n for n in ("NAM (log1p + Huber) | ReLU | 13 features", "LightGBM | 13 features") if n in s.index])
    return wide, s

def ranking(task, metric, label, path):
    wide, s = table(task, metric)
    s = s.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.2, 0.30 * len(s) + 1.3))
    best = s["mean"].max()
    ax.axvline(best, color="#4c78a8", ls=":", lw=1)
    for i, (n, r) in enumerate(s.iterrows()):
        ax.errorbar(r["mean"], i, xerr=r["std"], fmt="o", color=COL[r["tier"]], ms=5, capsize=2, lw=1)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([short(n) for n in s.index], fontsize=8.5)
    for t in ax.get_yticklabels():
        if s.loc[s.index[[short(n) for n in s.index].index(t.get_text())], "tier"] == "extracurricular":
            t.set_color(COL["extracurricular"]); t.set_fontweight("bold")
    ax.set_xlabel(f"{label} (mean ± std, 15 outer folds; dotted line = best model)")
    ax.grid(axis="x", alpha=0.3)
    hs = [plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in COL.items() if t in set(s["tier"])]
    ax.legend(handles=hs, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    return s

def ladder(path):
    steps = {
        "classification": ("f1", "F1", [("FT-Transformer-lite | ReLU | + 3 modifiers (25)", "FT-Transformer\n(can mix features)"),
                                        ("LightGBM stumps (additive, depth 1) | + 3 modifiers (25)", "trees, depth 1\n(no mixing)"),
                                        ("LightGBM depth 2 (pairwise) | + 3 modifiers (25)", "trees, depth 2\n(pairs)"),
                                        ("LightGBM | + 3 modifiers (25)", "trees, full\n(any mixing)")]),
        "regression": ("r2", "R²", [("NAM (log1p + Huber) | ReLU | + Teenhome (14)", "NAM\n(no mixing)"),
                                    ("NAM (log1p + Huber) | ReLU | + Teenhome + 5 Teenhome pairs", "NA²M\n(5 pairs)"),
                                    ("LightGBM stumps (additive, depth 1) | + Teenhome (14)", "trees, depth 1\n(no mixing)"),
                                    ("LightGBM depth 2 (pairwise) | + Teenhome (14)", "trees, depth 2\n(all pairs)"),
                                    ("LightGBM | + Teenhome (14)", "trees, full\n(any mixing)")]),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    rows = []
    for ax, (task, (metric, label, chain)) in zip(axes, steps.items()):
        wide, tier = A.fold_table(task, metric)
        means = [wide[m].mean() for m, _ in chain]
        cols = [COL["extracurricular"] if "LightGBM" not in m else COL["curriculum"] for m, _ in chain]
        x = np.arange(len(chain))
        ax.plot(x, means, color="#bbb", lw=1.2, zorder=1)
        ax.scatter(x, means, c=cols, s=90, zorder=2)
        lo, hi = min(means), max(means); pad = (hi - lo) * 0.35 + 0.01
        ax.set_ylim(lo - pad, hi + pad)
        for i, v in enumerate(means):
            ax.text(i, v - pad * 0.35, f"{v:.3f}", ha="center", va="top", fontsize=9)
        for i in range(1, len(chain)):
            r = A.paired(wide[chain[i][0]], wide[chain[i - 1][0]])
            rows.append(dict(task=task, step=f"{short(chain[i-1][0])} -> {short(chain[i][0])}", diff=r["mean_diff"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                             p_wilcoxon=r["p_wilcoxon"], p_nb=r["p_nb"]))
            ax.text(i - 0.5, (means[i] + means[i - 1]) / 2 + pad * 0.25, f"{r['mean_diff']:+.3f}", ha="center", fontsize=8.5, color="#444",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in chain], fontsize=8.5)
        ax.set_ylabel(f"{label} (mean of 15 folds)"); ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{'Classify (Response)' if task == 'classification' else 'Regression (Income)'}: where the gap comes from", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    pd.DataFrame(rows).round(5).to_csv(os.path.join(ROOT, "results", "gap_ladder.csv"), index=False)

def gain_share():
    """How much of the best model's improvement over a trivial baseline each model reaches."""
    rows = []
    for task, metric in (("classification", "f1"), ("regression", "r2")):
        wide, s = table(task, metric)
        base = 2 * 0.1494 / (1 + 0.1494) if task == "classification" else 0.0   # F1 of 'everyone responds' / R2 of the mean
        best = s.index[0]
        for n, r in s.iterrows():
            rows.append(dict(task=task, model=n, tier=r["tier"], mean=round(r["mean"], 4), rank=list(s.index).index(n) + 1, n_models=len(s),
                             share_of_best_gain=round((r["mean"] - base) / (s.loc[best, "mean"] - base), 4),
                             folds_beating_best=int((wide[n] > wide[best]).sum())))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(ROOT, "results", "gain_share.csv"), index=False)
    return out

if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)
    ranking("classification", "f1", "F1", os.path.join(ROOT, "media", "rank_clf_final.png"))
    ranking("regression", "r2", "R²", os.path.join(ROOT, "media", "rank_reg_final.png"))
    ladder(os.path.join(ROOT, "media", "gap_ladder.png"))
    g = gain_share()
    print(g[g.tier == "extracurricular"].to_string(index=False))
    print(pd.read_csv(os.path.join(ROOT, "results", "gap_ladder.csv")).to_string(index=False))

def learning_curves(path):
    import json
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for ax, (task, metric, label) in zip(axes, (("classification", "f1", "F1"), ("regression", "r2", "R²"))):
        d = json.load(open(os.path.join(ROOT, "weights_V7_1", f"{task}_pytorch_learning_curve.json")))
        r = pd.DataFrame(d["rows"])
        t = r.groupby(["model", "frac"]).agg(n=("n_train", "mean"), m=(metric, "mean"), s=(metric, "std")).reset_index()
        for m, g in t.groupby("model"):
            c = COL["curriculum"] if m.startswith("LightGBM") else COL["extracurricular"]
            ax.errorbar(g["n"], g["m"], yerr=g["s"], marker="o", capsize=3, color=c, label=short(m))
        gap = t.pivot(index="frac", columns="model", values="m")
        lg = [c for c in gap.columns if c.startswith("LightGBM")][0]; nn = [c for c in gap.columns if c != lg][0]
        for f, row in gap.iterrows():
            n = t[(t.frac == f)]["n"].iloc[0]
            ax.text(n, (row[lg] + row[nn]) / 2, f"gap {row[lg] - row[nn]:.3f}", fontsize=8, ha="left", va="center", color="#444",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        ax.set_xlabel("training rows per fold"); ax.set_ylabel(f"{label} (mean ± std, 5 folds)")
        ax.set_title(f"{'Classify (Response)' if task == 'classification' else 'Regression (Income)'}: score vs training size", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)

if __name__ == "__main__":
    learning_curves(os.path.join(ROOT, "media", "learning_curves.png"))
