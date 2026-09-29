"""Figure: per-matrix-type depth correlation at all three model scales.

The six-panel scale overlay is unreadable at ACL column width, and it is not
quite the claim we make anyway. It plots Spearman rho between layer index and
the stabilization step, one bar per model, grouped by matrix type, for both
scalars (t50 top, t90 bottom); * marks p < 0.05 after Holm correction across
the six matrix types. The two panels disagree for the FFN matrices, which is
itself a result.

    .venv/Scripts/python plot_depth_summary.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import hypotheses as hyp

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

MODELS = [("EleutherAI/pythia-70m", "70m (6L)", "#9ecae1"),
          ("EleutherAI/pythia-160m", "160m (12L)", "#4292c6"),
          ("EleutherAI/pythia-410m", "410m (24L)", "#08519c")]
MATS = ["q", "k", "v", "attn_out", "ffn_up", "ffn_down"]
LABEL = {"q": r"$W_Q$", "k": r"$W_K$", "v": r"$W_V$", "attn_out": r"$W_O$",
         "ffn_up": r"$W_{\mathrm{in}}$", "ffn_down": r"$W_{\mathrm{out}}$"}


def main() -> None:
    res = {}
    for model, _, _ in MODELS:
        jp = config.RESULTS_DIR / f"hypotheses_{config._slug(model)}.json"
        if jp.exists():
            res[model] = json.loads(jp.read_text())
    if len(res) < 2:
        print("[plot] need >=2 models"); return

    # Both scalars, stacked: the per-matrix depth trends are not the same under
    # t50 and t90, and showing only the primary scalar would hide that.
    fig, axes = plt.subplots(2, 1, figsize=(3.28, 3.9), sharex=True)
    x = np.arange(len(MATS))
    w = 0.26
    for ax, scalar in zip(axes, ("t50", "t90")):
        for i, (model, lab, col) in enumerate(MODELS):
            if model not in res:
                continue
            h1 = res[model][scalar]["H1"]
            vals = [h1["rho_by_matrix"].get(m, np.nan) for m in MATS]
            ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col,
                   edgecolor="white", linewidth=0.4)
            for j, m in enumerate(MATS):
                p = h1["p_holm_by_matrix"].get(m)
                if p is not None and p < 0.05:
                    v = vals[j]
                    ax.text(x[j] + (i - 1) * w, v + (0.03 if v >= 0 else -0.03),
                            "*", ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=9, color="0.1")
        ax.axhline(0, color="0.25", lw=0.9)
        ax.axvline(3.5, color="0.6", lw=0.8, ls=":")
        ax.set_ylim(-1.0, 1.12)
        ax.set_ylabel(rf"$\rho$(layer, $t_{{{scalar[1:]}}}$)")
    axes[0].text(1.5, 1.02, "attention", ha="center", va="bottom", fontsize=8,
                 color="0.25", transform=axes[0].get_xaxis_transform())
    axes[0].text(4.5, 1.02, "FFN", ha="center", va="bottom", fontsize=8,
                 color="0.25", transform=axes[0].get_xaxis_transform())
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABEL[m] for m in MATS])
    axes[0].legend(loc="lower left", framealpha=0.92, handlelength=1.1,
                   borderpad=0.3, labelspacing=0.25, fontsize=7)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(config.FIG_DIR / f"fig0_depth_summary.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    print("[plot] figs/fig0_depth_summary.pdf")


if __name__ == "__main__":
    main()
