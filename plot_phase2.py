"""Phase 2, step 2: aggregate the run matrix, test the arms, and plot.

Reads  runs/<run_id>/{config,summary}.json and evals.jsonl
Writes results/phase2_runs.parquet     every evaluation of every run
       results/phase2_summary.csv      one row per run
       results/phase2_table.txt        the arms table for the paper
       results/phase2_stats.txt        arm-vs-baseline tests
       figs/fig7..fig10_*.pdf (+ .png)

Statistics
----------
Arms are compared **paired by seed**: for a given seed every arm starts from a
bit-identical initialization and consumes the same batches in the same order, so
the seed is a matched block and the per-seed difference removes the (large)
seed-to-seed variance. The test is an exact paired sign-flip permutation test on
the mean per-seed difference, which needs no distributional assumption.

With n seeds the smallest attainable two-sided p is 2 / 2^n (0.25 at n=3,
0.031 at n=6), so the number of seeds is reported next to every p-value and no
claim of "no difference" is made on the basis of a non-significant result alone.
Per-seed consistency (how many seeds show the same sign) and the effect size in
perplexity units are reported alongside.

    .venv/Scripts/python plot_phase2.py
    .venv/Scripts/python plot_phase2.py --steps 5000
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

RUNS_DIR = config.ROOT / "runs"

EXCLUDED_TAGS = {"smoke", "pilot"}

ARM_ORDER = ["baseline", "attn-first", "attn-second", "attn-first-lrm",
             "attn-second-lrm", "alternating", "attn-freeze"]
ARM_LABEL = {a: a for a in ARM_ORDER}
ARM_COLOR = {"baseline": "#444444", "attn-first": "#1f4e9c",
             "attn-second": "#d2601a", "attn-first-lrm": "#6b8fd6",
             "attn-second-lrm": "#f0a060", "alternating": "#2a7f62",
             "attn-freeze": "#8b2fa0"}
# Head-to-head direction tests: (first-named arm, mirror arm). The -lrm pair
# is the LR-mass-matched version of the original pair (see schedules.py).
DIRECTION_PAIRS = [("attn-first", "attn-second"),
                   ("attn-first-lrm", "attn-second-lrm")]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


SUFFIX = ""   # set from --label; keeps a demo run from clobbering real results


def _save(fig, name: str) -> None:
    name = name + SUFFIX
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(config.FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] figs/{name}.pdf")


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load_runs(steps: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev_rows, sum_rows = [], []
    for d in sorted(RUNS_DIR.glob("*/")):
        sp, cp, ep = d / "summary.json", d / "config.json", d / "evals.jsonl"
        if not (sp.exists() and cp.exists() and ep.exists()):
            continue
        s = json.loads(sp.read_text())
        c = json.loads(cp.read_text())
        rc = c["run_config"]
        # pilots and smoke runs never enter an analysis
        if rc.get("tag") in EXCLUDED_TAGS:
            continue
        if steps is not None and rc["max_steps"] != steps:
            continue
        probe = {}
        wp = d / "weight_probe.json"
        if wp.exists():
            probe = json.loads(wp.read_text()).get("per_group", {})
        sum_rows.append({
            "run_id": d.name, "arm": s["arm"], "seed": s["seed"],
            "max_steps": s["max_steps"], "final_val_ppl": s["final_val_ppl"],
            "final_val_loss": s["final_val_loss"], "best_val_ppl": s["best_val_ppl"],
            "train_wall_s": s["train_wall_s"], "total_wall_s": s["total_wall_s"],
            "peak_vram_gb": s["peak_vram_gb"], "n_params": s["n_params"],
            "tokens_per_step": s["tokens_per_step"],
            "total_tokens": s["total_tokens"],
            "init_signature": s["init_signature"],
            "disp_attn": probe.get("attn", {}).get("rel_displacement"),
            "disp_ffn": probe.get("ffn", {}).get("rel_displacement"),
            "disp_other": probe.get("other", {}).get("rel_displacement"),
        })
        for line in ep.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r.update(arm=s["arm"], seed=s["seed"], run_id=d.name,
                         max_steps=s["max_steps"])
                ev_rows.append(r)
    evals = pd.DataFrame(ev_rows)
    summ = pd.DataFrame(sum_rows)
    if len(summ):
        summ["arm"] = pd.Categorical(summ["arm"], ARM_ORDER, ordered=True)
        summ = summ.sort_values(["arm", "seed"]).reset_index(drop=True)
    return evals, summ


# --------------------------------------------------------------------------- #
# Steps / wall-clock to a target perplexity
# --------------------------------------------------------------------------- #

def to_threshold(g: pd.DataFrame, target: float, ycol: str = "val_ppl",
                 xcol: str = "step") -> float:
    """First x at which `ycol` first drops to `target`, linearly interpolated."""
    g = g.sort_values(xcol)
    y = g[ycol].to_numpy(dtype=float)
    x = g[xcol].to_numpy(dtype=float)
    for i in range(len(y)):
        if y[i] <= target:
            if i == 0:
                return float(x[0])
            y0, y1, x0, x1 = y[i - 1], y[i], x[i - 1], x[i]
            if y0 == y1:
                return float(x1)
            return float(x0 + (y0 - target) * (x1 - x0) / (y0 - y1))
    return math.nan


def pick_target(summ: pd.DataFrame, arms: list[str]) -> float:
    """A target perplexity every run in `arms` actually reaches.

    Chosen as the next round number above the worst final perplexity, so no run
    is excluded from the steps-to-threshold comparison. Fixed once and used for
    every arm.
    """
    sub = summ[summ["arm"].isin(arms)]
    worst = float(sub["final_val_ppl"].max())
    for t in [30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 150, 200, 300,
              500, 1000]:
        if t >= worst:
            return float(t)
    return float(math.ceil(worst / 100) * 100)


# --------------------------------------------------------------------------- #
# Paired sign-flip permutation test
# --------------------------------------------------------------------------- #

def paired_perm(diff: np.ndarray) -> dict:
    """Exact two-sided sign-flip permutation test on the mean of `diff`."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    n = len(d)
    out = {"n": n, "mean_diff": float(d.mean()) if n else math.nan,
           "median_diff": float(np.median(d)) if n else math.nan,
           "n_positive": int((d > 0).sum()), "p": math.nan,
           "min_attainable_p": (2.0 / 2 ** n) if n else math.nan}
    if n < 2:
        return out
    obs = abs(d.mean())
    null = [abs((d * np.array(s)).mean())
            for s in itertools.product([1, -1], repeat=n)]
    out["p"] = float(np.mean(np.array(null) >= obs - 1e-12))
    return out


def compare(summ: pd.DataFrame, per_run: pd.DataFrame, target: float,
            baseline: str = "baseline") -> tuple[str, pd.DataFrame]:
    """Every arm against the baseline, paired by seed."""
    metrics = [("steps_to_target", f"steps to val ppl <= {target:g}", "lower is better"),
               ("wall_to_target", f"train seconds to val ppl <= {target:g}", "lower is better"),
               ("final_val_ppl", "final val perplexity", "lower is better"),
               ("train_wall_s", "total train wall-clock (s)", "lower is better"),
               ("peak_vram_gb", "peak VRAM (GiB)", "lower is better")]
    base = per_run[per_run["arm"] == baseline].set_index("seed")
    lines = [f"Phase 2: arms vs `{baseline}`, paired by seed",
             f"Target perplexity for steps-to-threshold: {target:g}",
             "Exact two-sided sign-flip permutation test on the per-seed "
             "difference (arm - baseline).", "=" * 78, ""]
    recs = []
    for arm in ARM_ORDER:
        if arm == baseline or arm not in set(per_run["arm"]):
            continue
        a = per_run[per_run["arm"] == arm].set_index("seed")
        seeds = sorted(set(a.index) & set(base.index))
        if not seeds:
            continue
        lines.append(f"--- {arm}  (paired seeds: {seeds}) ---")
        for col, label, _ in metrics:
            if col not in a.columns:
                continue
            d = a.loc[seeds, col].to_numpy(dtype=float) - \
                base.loc[seeds, col].to_numpy(dtype=float)
            r = paired_perm(d)
            bm = float(base.loc[seeds, col].mean())
            am = float(a.loc[seeds, col].mean())
            rel = (am - bm) / bm * 100 if bm else math.nan
            lines.append(
                f"  {label:<42s} baseline {bm:>9.2f}   {arm} {am:>9.2f}   "
                f"diff {r['mean_diff']:+9.2f} ({rel:+6.1f}%)   "
                f"p={r['p']:.4g}{'  *' if np.isfinite(r['p']) and r['p'] < 0.05 else ''}"
                f"   worse in {r['n_positive']}/{r['n']} seeds"
                f"   [min attainable p = {r['min_attainable_p']:.3g}]")
            recs.append(dict(arm=arm, metric=col, baseline_mean=bm, arm_mean=am,
                             mean_diff=r["mean_diff"], rel_pct=rel, p=r["p"],
                             n_seeds=r["n"], n_worse=r["n_positive"],
                             min_p=r["min_attainable_p"]))
        lines.append("")

    # attn-first vs attn-second: the direction test (H9 in the plan)
    for first, second in DIRECTION_PAIRS:
        if not {first, second} <= set(per_run["arm"]):
            continue
        af = per_run[per_run["arm"] == first].set_index("seed")
        as_ = per_run[per_run["arm"] == second].set_index("seed")
        seeds = sorted(set(af.index) & set(as_.index))
        lines.append(f"--- direction test: {first} vs {second} "
                     f"(paired seeds: {seeds}) ---")
        for col, label, _ in metrics:
            if col not in af.columns:
                continue
            d = af.loc[seeds, col].to_numpy(dtype=float) - \
                as_.loc[seeds, col].to_numpy(dtype=float)
            r = paired_perm(d)
            lines.append(f"  {label:<42s} {first} {af.loc[seeds, col].mean():>9.2f}   "
                         f"{second} {as_.loc[seeds, col].mean():>9.2f}   "
                         f"diff {r['mean_diff']:+9.2f}   p={r['p']:.4g}"
                         f"   first worse in {r['n_positive']}/{r['n']} seeds")
            recs.append(dict(arm=f"{first}_vs_{second}", metric=col,
                             baseline_mean=float(as_.loc[seeds, col].mean()),
                             arm_mean=float(af.loc[seeds, col].mean()),
                             mean_diff=r["mean_diff"], rel_pct=math.nan, p=r["p"],
                             n_seeds=r["n"], n_worse=r["n_positive"],
                             min_p=r["min_attainable_p"]))
        lines.append("")
    return "\n".join(lines), pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig7_curves(evals: pd.DataFrame, target: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    # The baseline is drawn last, dashed, and on top: `alternating` lands almost
    # exactly on it, and hiding the control under another arm would misrepresent
    # the very result that makes the control informative.
    draw = [a for a in ARM_ORDER if a != "baseline"] + ["baseline"]
    for ax, (lo, hi, title) in zip(axes, [
            (0, None, "full run"), (0.35, None, "final 65% (zoom)")]):
        for arm in draw:
            g = evals[evals["arm"] == arm]
            if g.empty:
                continue
            piv = g.pivot_table(index="step", columns="seed", values="val_ppl")
            base = arm == "baseline"
            ax.plot(piv.index, piv.mean(axis=1), lw=2.0 if base else 1.5,
                    ls="--" if base else "-", color=ARM_COLOR[arm],
                    label=ARM_LABEL[arm], zorder=6 if base else 3)
            ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1),
                            color=ARM_COLOR[arm], alpha=0.16, lw=0, zorder=2)
        ax.axhline(target, color="0.55", lw=0.8, ls=":", zorder=1)
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_title(title)
        if lo:
            mx = evals["step"].max()
            ax.set_xlim(lo * mx, mx * 1.01)
            sub = evals[evals["step"] >= lo * mx]["val_ppl"]
            ax.set_ylim(sub.min() * 0.96, sub.max() * 1.06)
        ax.annotate(f"target ppl {target:g}", xy=(0.015, target),
                    xycoords=("axes fraction", "data"), fontsize=7,
                    color="0.35", va="bottom", ha="left")
    axes[0].set_ylabel("validation perplexity")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(ARM_LABEL[a]) for a in ARM_ORDER if ARM_LABEL[a] in labels]
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="upper right", framealpha=0.92)
    n_seeds = evals["seed"].nunique()
    fig.suptitle(f"validation perplexity vs step; line = mean over "
                 f"{n_seeds} seeds, band = min-max")
    _save(fig, "fig7_phase2_curves")


def fig8_bars(per_run: pd.DataFrame, target: float) -> None:
    metrics = [("steps_to_target", f"steps to ppl {target:g}"),
               ("wall_to_target", f"train seconds to ppl {target:g}"),
               ("final_val_ppl", "final val perplexity"),
               ("peak_vram_gb", "peak VRAM (GiB)")]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.9))
    arms = [a for a in ARM_ORDER if a in set(per_run["arm"])]
    rng = np.random.default_rng(0)
    for ax, (col, label) in zip(axes, metrics):
        means, errs = [], []
        for i, arm in enumerate(arms):
            v = per_run[per_run["arm"] == arm][col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            means.append(v.mean() if v.size else np.nan)
            errs.append(v.std(ddof=1) if v.size > 1 else 0.0)
            ax.scatter(np.full(v.size, i) + rng.normal(0, 0.055, v.size), v,
                       s=16, color="k", zorder=4, alpha=0.75, edgecolor="none")
        ax.bar(range(len(arms)), means, yerr=errs, capsize=3,
               color=[ARM_COLOR[a] for a in arms], alpha=0.8,
               error_kw=dict(lw=1.0, ecolor="0.25"))
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=35, ha="right")
        ax.set_title(label, fontsize=8.5)
        # An arm that never reaches the target has no bar; say so rather than
        # leaving a silent gap.
        for i, m in enumerate(means):
            if not np.isfinite(m):
                ax.text(i, ax.get_ylim()[1] * 0.04, "never\nreached",
                        ha="center", va="bottom", fontsize=6.5, color="0.3",
                        rotation=90)
        finite = [v for v in means if np.isfinite(v)]
        if col == "final_val_ppl" and finite:
            ax.set_ylim(min(finite) * 0.93, None)
    n = per_run.groupby("arm", observed=True)["seed"].nunique().max()
    fig.suptitle(f"Phase 2 summary; bars = mean +/- SD over seeds, "
                 f"dots = individual seeds (n up to {n})")
    _save(fig, "fig8_phase2_bars")


def fig9_paired(per_run: pd.DataFrame, target: float) -> None:
    """Per-seed differences against the baseline -- the paired view."""
    base = per_run[per_run["arm"] == "baseline"].set_index("seed")
    arms = [a for a in ARM_ORDER if a != "baseline" and a in set(per_run["arm"])]
    metrics = [("steps_to_target", f"steps to ppl {target:g}"),
               ("final_val_ppl", "final val perplexity"),
               ("train_wall_s", "train wall-clock (s)")]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.0))
    for ax, (col, label) in zip(axes, metrics):
        for i, arm in enumerate(arms):
            a = per_run[per_run["arm"] == arm].set_index("seed")
            seeds = sorted(set(a.index) & set(base.index))
            d = (a.loc[seeds, col].to_numpy(dtype=float)
                 - base.loc[seeds, col].to_numpy(dtype=float))
            d = d[np.isfinite(d)]
            ax.scatter(np.full(len(d), i), d, s=26, color=ARM_COLOR[arm],
                       zorder=4, edgecolor="none")
            if len(d):
                ax.plot([i - .25, i + .25], [np.mean(d)] * 2, lw=2.2, color="k",
                        zorder=5)
            else:
                ax.text(i, 0, "target\nnever\nreached", ha="center", va="center",
                        fontsize=6.5, color="0.35")
        ax.axhline(0, color="0.35", lw=1.0)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=35, ha="right")
        ax.set_title(label, fontsize=8.5)
        ax.set_xlim(-0.5, len(arms) - 0.5)
    axes[0].set_ylabel("arm - baseline (per seed)")
    fig.suptitle("paired per-seed differences from the baseline; "
                 "below 0 means the arm beat the baseline for that seed")
    _save(fig, "fig9_phase2_paired")


def fig10_displacement(per_run: pd.DataFrame) -> None:
    """Verification that the intervention changed which parameters moved."""
    arms = [a for a in ARM_ORDER if a in set(per_run["arm"])]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    w = 0.38
    for j, (col, lab, c) in enumerate([("disp_attn", "attention", "#1f4e9c"),
                                       ("disp_ffn", "FFN", "#d2601a")]):
        vals = [per_run[per_run["arm"] == a][col].mean() for a in arms]
        errs = [per_run[per_run["arm"] == a][col].std(ddof=1) for a in arms]
        ax.bar(np.arange(len(arms)) + (j - 0.5) * w, vals, w, yerr=errs,
               capsize=2.5, label=lab, color=c, alpha=0.85,
               error_kw=dict(lw=0.9, ecolor="0.25"))
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=35, ha="right")
    ax.set_ylabel(r"$\|W_{\mathrm{final}}-W_0\|_F\ /\ \|W_0\|_F$")
    ax.legend()
    ax.set_title("total weight displacement by group\n"
                 "(manipulation check: the arms moved different parameters)",
                 fontsize=8.5)
    _save(fig, "fig10_phase2_displacement")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=None,
                    help="only aggregate runs with this max_steps")
    ap.add_argument("--label", default="",
                    help="suffix for every output file, so a small/demo "
                         "analysis cannot overwrite the real results")
    ap.add_argument("--target", type=float, default=None,
                    help="target val perplexity (default: auto, see pick_target)")
    args = ap.parse_args()

    global SUFFIX
    SUFFIX = f"_{args.label}" if args.label else ""

    evals, summ = load_runs(args.steps)
    if summ.empty:
        raise SystemExit("no completed runs found in runs/ -- run run_matrix.sh first")
    print(f"[phase2] {len(summ)} runs, {summ['arm'].nunique()} arms, "
          f"seeds {sorted(summ['seed'].unique())}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    evals.to_parquet(config.RESULTS_DIR / f"phase2_runs{SUFFIX}.parquet", index=False)

    mandatory = ["baseline", "attn-first", "attn-second"]
    target = args.target or pick_target(summ, [a for a in mandatory
                                               if a in set(summ["arm"])])

    per_run = summ.copy()
    sts, wts = [], []
    for _, r in per_run.iterrows():
        g = evals[evals["run_id"] == r["run_id"]]
        sts.append(to_threshold(g, target, "val_ppl", "step"))
        wts.append(to_threshold(g, target, "val_ppl", "wall_s"))
    per_run["steps_to_target"] = sts
    per_run["wall_to_target"] = wts
    per_run["target_ppl"] = target
    per_run.to_csv(config.RESULTS_DIR / f"phase2_summary{SUFFIX}.csv", index=False)

    # --- reproducibility check: identical init per seed across arms
    bad = [s for s, g in per_run.groupby("seed")
           if g["init_signature"].nunique() != 1]
    init_msg = ("OK -- every arm at a given seed started from a bit-identical "
                "initialization" if not bad else
                f"FAILED for seeds {bad}: initializations differ across arms")
    print(f"[phase2] init check: {init_msg}")

    # --- table
    agg = (per_run.groupby("arm", observed=True)
           .agg(n_seeds=("seed", "nunique"),
                final_ppl_mean=("final_val_ppl", "mean"),
                final_ppl_sd=("final_val_ppl", "std"),
                steps_to_target_mean=("steps_to_target", "mean"),
                steps_to_target_sd=("steps_to_target", "std"),
                wall_to_target_mean=("wall_to_target", "mean"),
                train_wall_mean=("train_wall_s", "mean"),
                peak_vram_mean=("peak_vram_gb", "mean"),
                disp_attn=("disp_attn", "mean"),
                disp_ffn=("disp_ffn", "mean")).reset_index())
    tbl = [f"Phase 2 results  (target val ppl = {target:g})",
           f"Init check: {init_msg}",
           f"Token budget per run: {int(per_run['total_tokens'].iloc[0]):,} "
           f"({per_run['max_steps'].iloc[0]} steps x "
           f"{per_run['tokens_per_step'].iloc[0]:,} tokens)",
           "=" * 78, "",
           agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "",
           "Per-run detail:", "",
           per_run[["arm", "seed", "final_val_ppl", "steps_to_target",
                    "wall_to_target", "train_wall_s", "peak_vram_gb",
                    "disp_attn", "disp_ffn"]]
           .to_string(index=False, float_format=lambda v: f"{v:.3f}")]
    tbl_txt = "\n".join(tbl)
    (config.RESULTS_DIR / f"phase2_table{SUFFIX}.txt").write_text(tbl_txt + "\n")
    print("\n" + tbl_txt)

    stats_txt, stats_df = compare(summ, per_run, target)
    (config.RESULTS_DIR / f"phase2_stats{SUFFIX}.txt").write_text(stats_txt + "\n")
    stats_df.to_csv(config.RESULTS_DIR / f"phase2_stats{SUFFIX}.csv", index=False)
    print("\n" + stats_txt)

    fig7_curves(evals, target)
    fig8_bars(per_run, target)
    fig9_paired(per_run, target)
    fig10_displacement(per_run)


if __name__ == "__main__":
    main()
