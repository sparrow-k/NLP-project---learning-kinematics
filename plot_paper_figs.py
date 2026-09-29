"""Column-width versions of the Phase 2 figures, for the ACL paper.

The analysis figures in plot_phase2.py are sized for screen reading (7--9
inches wide). Dropped into a 3.3-inch ACL column they become unreadable, so
the paper uses these instead: same data, fewer panels, larger type, and no
in-figure title because the caption carries it.

    .venv/Scripts/python plot_paper_figs.py
"""

from __future__ import annotations

import itertools
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 8.5,
    "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ARMS = ["baseline", "attn-first", "attn-second", "attn-first-lrm",
        "attn-second-lrm", "alternating", "attn-freeze"]
SHORT = {"baseline": "base", "attn-first": "attn-1st",
         "attn-second": "ffn-1st", "attn-first-lrm": "attn-1st LR",
         "attn-second-lrm": "ffn-1st LR", "alternating": "alt",
         "attn-freeze": "freeze"}
COLOR = {"baseline": "#444444", "attn-first": "#1f4e9c",
         "attn-second": "#d2601a", "attn-first-lrm": "#6b8fd6",
         "attn-second-lrm": "#f0a060", "alternating": "#2a7f62",
         "attn-freeze": "#8b2fa0"}
# The paired panel omits attn-freeze: its +13 ppl difference would compress
# every other arm to a sliver. It is reported in Table 2 and the text.
PAIRED_EXCLUDE = {"attn-freeze"}
COL = 3.3   # ACL column width in inches


def save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(config.FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] figs/{name}.pdf")


def main() -> None:
    per = pd.read_csv(config.RESULTS_DIR / "phase2_summary.csv")
    ev = pd.read_parquet(config.RESULTS_DIR / "phase2_runs.parquet")
    target = float(per["target_ppl"].iloc[0])

    # ---------------------------------------------------------------- paired
    base = per[per["arm"] == "baseline"].set_index("seed")
    arms = [a for a in ARMS if a != "baseline" and a in set(per["arm"])
            and a not in PAIRED_EXCLUDE]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.0))
    for ax, (col, lab) in zip(axes, [("steps_to_target", f"steps to ppl {target:g}"),
                                     ("final_val_ppl", "final val ppl")]):
        for i, arm in enumerate(arms):
            a = per[per["arm"] == arm].set_index("seed")
            sd = sorted(set(a.index) & set(base.index))
            d = (a.loc[sd, col].to_numpy(float) - base.loc[sd, col].to_numpy(float))
            d = d[np.isfinite(d)]
            if len(d):
                ax.scatter(np.full(len(d), i), d, s=13, color=COLOR[arm],
                           zorder=4, edgecolor="none")
                ax.plot([i - .3, i + .3], [d.mean()] * 2, lw=1.8, color="k", zorder=5)
            else:
                ax.text(i, 0, "n/r", ha="center", va="center", fontsize=6.5,
                        color="0.4", rotation=90)
        ax.axhline(0, color="0.35", lw=0.9)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([SHORT[a] for a in arms], rotation=45, ha="right",
                           fontsize=7.5)
        ax.set_title(lab, fontsize=8)
        ax.set_xlim(-0.6, len(arms) - 0.4)
    axes[0].set_ylabel("arm $-$ baseline", fontsize=8)
    save(fig, "figp_paired")

    # ---------------------------------------------------------------- curves
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    mx = int(ev["step"].max())
    order = [a for a in ARMS if a != "baseline"] + ["baseline"]
    for arm in order:
        g = ev[ev["arm"] == arm]
        if g.empty:
            continue
        piv = g.pivot_table(index="step", columns="seed", values="val_ppl")
        keep = piv.index >= 0.33 * mx
        b = arm == "baseline"
        ax.plot(piv.index[keep], piv.mean(axis=1)[keep], lw=1.8 if b else 1.3,
                ls="--" if b else "-", color=COLOR[arm], label=SHORT[arm],
                zorder=6 if b else 3)
        ax.fill_between(piv.index[keep], piv.min(axis=1)[keep],
                        piv.max(axis=1)[keep], color=COLOR[arm], alpha=0.15,
                        lw=0, zorder=2)
    ax.axhline(target, color="0.55", lw=0.8, ls=":", zorder=1)
    ax.annotate(f"target {target:g}", xy=(0.02, target),
                xycoords=("axes fraction", "data"), fontsize=7, color="0.35",
                va="bottom")
    ax.set_yscale("log")
    sub = ev[ev["step"] >= 0.33 * mx]["val_ppl"]
    ax.set_ylim(sub.min() * 0.96, sub.max() * 1.05)
    ax.set_xlabel("training step", fontsize=8)
    ax.set_ylabel("validation perplexity", fontsize=8)
    ax.legend(ncol=2, fontsize=7, framealpha=0.92, handlelength=1.3,
              borderpad=0.3, labelspacing=0.25, columnspacing=1.0)
    save(fig, "figp_curves")


if __name__ == "__main__":
    main()
