"""Tables, paired tests and figures from results/ledger.csv.

Run from customer_personality_data/:  python results/build_ledger.py && python results/analyze.py
Only repeated K-fold rows (scheme == "kfold", selection == "inner validation") enter the comparison:
every model there was scored on the same 15 outer folds (5-fold x 3 repeats), so differences are paired.
"""
import os, json, itertools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
led = pd.read_csv(os.path.join(ROOT, "results", "ledger.csv"))
os.makedirs(os.path.join(ROOT, "media"), exist_ok=True)

TASKS = {"classification": ("f1", "F1"), "regression": ("r2", "R²")}
TIER_COLOR = {"curriculum": "#4c78a8", "curriculum family": "#9aa5b1", "extracurricular": "#6a3d9a"}
TIER_ORDER = ["curriculum", "curriculum family", "extracurricular"]
N_TEST_OVER_TRAIN = 1 / 4        # 5-fold: each outer test fold is 1/4 the size of its training part


def fold_table(task, metric, version_prefixes=None):
    d = led[(led.task == task) & (led.scheme == "kfold") & (led.selection == "inner validation")].copy()
    if version_prefixes:
        d = d[d.version.str.startswith(tuple(version_prefixes))]
    d = d[d.version != "V3_1"]        # LightGBM was re-run with the same protocol in V6_1; do not average two runs of one model
    d["key"] = d["repeat"].astype(int).astype(str) + "-" + d["fold"].astype(int).astype(str)
    d["name"] = d["model"] + "|" + d["framework"]
    # the same model name may come from the scratch and library twin: keep them apart
    d.loc[d["framework"] == "scratch", "name"] = d["model"] + " [scratch]"
    d.loc[d["framework"] != "scratch", "name"] = d["model"]
    wide = d.pivot_table(index="key", columns="name", values=metric, aggfunc="mean")
    tier = d.groupby("name")["tier"].first()
    return wide, tier


def summary(wide, tier):
    s = pd.DataFrame({"mean": wide.mean(), "std": wide.std(), "folds": wide.count()})
    s["tier"] = tier.reindex(s.index)
    return s.sort_values("mean", ascending=False)


def paired(a, b):
    """a, b: per-fold scores (same fold order). Returns mean diff and several p-values (H0: no difference)."""
    both = pd.concat([a, b], axis=1).dropna()
    d = (both.iloc[:, 0] - both.iloc[:, 1]).to_numpy()
    J = len(d)
    if J < 3 or np.allclose(d, 0):
        return dict(n=J, mean_diff=float(d.mean()) if J else np.nan, p_wilcoxon=np.nan, p_nb=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    t = d.mean() / np.sqrt((1 / J + N_TEST_OVER_TRAIN) * d.var(ddof=1))       # Nadeau-Bengio corrected resampled t
    p_nb = 2 * stats.t.sf(abs(t), J - 1)
    p_w = stats.wilcoxon(d).pvalue
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, J, replace=True).mean() for _ in range(5000)]
    return dict(n=J, mean_diff=float(d.mean()), p_wilcoxon=float(p_w), p_nb=float(p_nb),
                ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)))


def ranking_plot(task, s, label, path):
    s = s.copy().sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(s) + 1.6))
    for i, (name, r) in enumerate(s.iterrows()):
        ax.errorbar(r["mean"], i, xerr=r["std"], fmt="o", color=TIER_COLOR.get(r["tier"], "k"), capsize=3, ms=6)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=9)
    ax.set_xlabel(f"{label}: mean ± std over 15 outer folds")
    ax.grid(axis="x", alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in TIER_COLOR.items() if t in set(s["tier"])]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_title(f"{task.capitalize()}: all models, same folds")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def variant_tests(task, version, framework, ref_contains, tag):
    """Every other model of one version/framework against its reference model, paired on the same folds.
    Used for ReGLU vs ReLU (V4_1) and each interaction feature set vs its base set (V5_1)."""
    metric, _ = TASKS[task]
    d = led[(led.task == task) & (led.scheme == "kfold") & (led.version == version) & (led.framework == framework)].copy()
    if d.empty:
        return None
    d["key"] = d["repeat"].astype(int).astype(str) + "-" + d["fold"].astype(int).astype(str)
    wide = d.pivot_table(index="key", columns="model", values=metric, aggfunc="mean")
    ref = [c for c in wide.columns if c.endswith(ref_contains)]
    assert len(ref) == 1, (version, framework, ref_contains, list(wide.columns))
    ref = ref[0]
    rows = []
    for c in wide.columns:
        r = paired(wide[c], wide[ref]); r.update(model=c, reference=ref, mean=float(wide[c].mean()), std=float(wide[c].std()))
        rows.append(r)
    out = pd.DataFrame(rows).round(5)
    out.to_csv(os.path.join(ROOT, "results", f"{tag}_{task}.csv"), index=False)
    return out


def run(version_prefixes=None, tag="all"):
    out = {}
    for task, (metric, label) in TASKS.items():
        wide, tier = fold_table(task, metric, version_prefixes)
        if wide.empty:
            continue
        s = summary(wide, tier)
        s.round(4).to_csv(os.path.join(ROOT, "results", f"summary_{task}_{tag}.csv"))
        ranking_plot(task, s, label, os.path.join(ROOT, "media", f"ranking_{task[:3]}_{tag}.png"))
        full = s[~s.index.str.contains("scratch")]        # scratch twins cover 5 folds only; they are parity checks, not contenders
        best = {t: full[full.tier == t].index[0] for t in TIER_ORDER if (full.tier == t).any()}
        tests = []
        for t1, t2 in itertools.combinations(best, 2):
            r = paired(wide[best[t1]], wide[best[t2]]); r.update(a=best[t1], b=best[t2], tier_a=t1, tier_b=t2); tests.append(r)
        pd.DataFrame(tests).round(5).to_csv(os.path.join(ROOT, "results", f"tiers_{task}_{tag}.csv"), index=False)
        out[task] = (s, best, tests)
    return out


def gate(task, contenders, extras):
    """Decision-gate table: each contender (curriculum side) minus each extracurricular model, paired on the same 15 folds."""
    metric, _ = TASKS[task]
    wide, tier = fold_table(task, metric)
    rows = []
    for a in contenders:
        for b in extras:
            r = paired(wide[a], wide[b]); r.update(curriculum=a, extracurricular=b, mean_a=float(wide[a].mean()), mean_b=float(wide[b].mean())); rows.append(r)
    out = pd.DataFrame(rows).round(5)
    out.to_csv(os.path.join(ROOT, "results", f"gate_{task}.csv"), index=False)
    return out


GATE = {
    "classification": (["LightGBM | + 3 modifiers + 5 pairs", "LightGBM", "CatBoost", "XGBoost", "HistGradientBoosting", "Stacking", "Logistic regression"],
                       ["FT-Transformer-lite | ReLU | + 3 modifiers + 5 pairs", "FT-Transformer-lite | ReLU (baseline)"]),
    "regression": (["CatBoost", "LightGBM | + Teenhome (14)", "ExtraTrees", "Polynomial regression", "Multiple regression"],
                   ["NAM (log1p + Huber) | ReLU | + Teenhome + 5 Teenhome pairs", "NAM (log1p + Huber) | ReLU (baseline)"]),
}


def run_variants():
    out = {}
    for task in TASKS:
        for tag, version, fw, ref in [("reglu", "V4_1", "pytorch", "| ReLU (baseline)"),
                                      ("interaction_nn", "V5_1", "pytorch", "| base (22)" if task == "classification" else "| 13 features"),
                                      ("interaction_lgbm", "V5_1", "lightgbm", "| base (22)" if task == "classification" else "| 13 features")]:
            r = variant_tests(task, version, fw, ref, tag)
            if r is not None:
                out[(tag, task)] = r
    return out


if __name__ == "__main__":
    for (tag, task), r in run_variants().items():
        print(f"\n=== {tag} / {task} (difference vs reference, paired on the same folds) ===")
        print(r[["model", "mean", "std", "n", "mean_diff", "ci_lo", "ci_hi", "p_nb", "p_wilcoxon"]].to_string(index=False))
    for task, (cs, es) in GATE.items():
        g = gate(task, cs, es)
        print(f"\n=== gate / {task}: curriculum minus extracurricular (paired, same folds) ===")
        print(g[["curriculum", "extracurricular", "mean_a", "mean_b", "mean_diff", "ci_lo", "ci_hi", "p_nb", "p_wilcoxon"]].to_string(index=False))
    res = run()
    for task, (s, best, tests) in res.items():
        print(f"\n=== {task} ===")
        print(s.round(4).to_string())
        print("best per tier:", best)
        for r in tests:
            print(f"{r['tier_a']} ({r['a']}) minus {r['tier_b']} ({r['b']}): diff {r['mean_diff']:+.4f} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | p_NB {r['p_nb']:.4f} | p_Wilcoxon {r['p_wilcoxon']:.4f}")
