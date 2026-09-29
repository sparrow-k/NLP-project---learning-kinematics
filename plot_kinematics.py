"""Phase 1, step 3: the kinematics figures.

Reads  data/kinematics_<model>.parquet  and  results/stabilization_<model>.csv
Writes figs/fig1..fig6_*.pdf (+ .png for convenience)

  Fig 1  convergence          1 - cos_to_final, and displacement rate, vs step
  Fig 2  progress curves      p_norm vs step with t50 / t90 markers
  Fig 3  kinematic wavefront  heatmap of displacement rate over (layer, matrix)
  Fig 4  stabilization vs depth   t50 and t90 vs layer index  [headline, H1]
  Fig 5  scale overlay        Fig 4 on relative depth, both models  [H5]
  Fig 6  matrix-type ordering per-layer t50 by matrix type  [H2 / H3]

    .venv/Scripts/python plot_kinematics.py --model EleutherAI/pythia-160m
    .venv/Scripts/python plot_kinematics.py --all
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import hypotheses as hyp

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,     # embed real fonts, not type-3 bitmaps
    "ps.fonttype": 42,
})

MATRIX_ORDER = ["q", "k", "v", "attn_out", "ffn_up", "ffn_down",
                "embed_in", "embed_out"]
MATRIX_LABEL = {"q": r"$W_Q$", "k": r"$W_K$", "v": r"$W_V$",
                "attn_out": r"$W_O$", "ffn_up": r"$W_{\mathrm{in}}$",
                "ffn_down": r"$W_{\mathrm{out}}$",
                "embed_in": "embed in", "embed_out": "embed out"}
MATRIX_COLOR = {"q": "#1f4e9c", "k": "#3f7fd0", "v": "#7fb3e8",
                "attn_out": "#0b2545",
                "ffn_up": "#d2601a", "ffn_down": "#f4a259",
                "embed_in": "#2a7f62", "embed_out": "#7fc29b"}
GROUP_COLOR = {"attn": "#1f4e9c", "ffn": "#d2601a", "embed": "#2a7f62"}
LAYER_MATS = hyp.LAYER_MATS


def _save(fig, name: str) -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(config.FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] figs/{name}.pdf")


def _xlog(ax, steps):
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, max(steps) * 1.15)


def _grid(n: int, w: float = 2.55, h: float = 2.0):
    ncols = 4 if n > 6 else min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                             squeeze=False, sharex=True, sharey=True)
    return fig, axes.ravel()


def _rate(steps: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Displacement per 1000 optimizer steps (matches compute_stabilization)."""
    out = np.full_like(vel, np.nan, dtype=float)
    for i in range(1, len(steps)):
        d = (steps[i] - steps[i - 1]) / 1000.0
        if d > 0:
            out[i] = vel[i] / d
    return out


def _series(df: pd.DataFrame, layer: int, mat: str) -> pd.DataFrame:
    g = df[(df["layer"] == layer) & (df["matrix"] == mat)]
    s = g.pivot_table(index="step", columns="metric", values="value").sort_index()
    steps = s.index.to_numpy(dtype=float)
    cos = s["cos_to_final"].to_numpy(dtype=float)
    vel = s["velocity"].to_numpy(dtype=float)
    p = np.clip((cos - cos[0]) / (1.0 - cos[0] + 1e-12), 0.0, None)
    return pd.DataFrame({"step": steps, "cos": cos, "p_norm": p,
                         "velocity": vel, "rate": _rate(steps, vel)})


# --------------------------------------------------------------------------- #
# Fig 1 -- convergence and displacement rate
# --------------------------------------------------------------------------- #

def fig1_convergence(df: pd.DataFrame, model: str) -> None:
    layers = sorted(df[df["layer"] >= 0]["layer"].unique())
    for field, ylabel, tag, logy in (
        ("cos", r"$1-\cos(W_t,\ W_{\mathrm{final}})$", "distance", True),
        ("rate", "displacement per 1000 steps", "rate", True),
    ):
        fig, axes = _grid(len(layers))
        for ax, layer in zip(axes, layers):
            for mat in LAYER_MATS:
                s = _series(df, layer, mat)
                y = (1 - s["cos"]) if field == "cos" else s["rate"]
                ax.plot(s["step"], y, lw=1.1, color=MATRIX_COLOR[mat],
                        label=MATRIX_LABEL[mat], marker="o", ms=1.8)
            if field == "rate":
                for tau in (0.02, 0.1):
                    ax.axhline(tau, color="0.55", lw=0.6, ls=":")
            if logy:
                ax.set_yscale("log")
            _xlog(ax, sorted(df["step"].unique()))
            ax.set_title(f"layer {layer}")
        for ax in axes[len(layers):]:
            ax.set_visible(False)
        axes[0].legend(ncol=2, loc="best", framealpha=0.85, handlelength=1.4)
        fig.supxlabel("training step")
        fig.supylabel(ylabel)
        fig.suptitle(f"{'directional distance to the final state' if field == 'cos' else 'displacement rate'}"
                     f"  ({model.split('/')[-1]})")
        _save(fig, f"fig1{'a' if field == 'cos' else 'b'}_{tag}_{config._slug(model)}")


# --------------------------------------------------------------------------- #
# Fig 2 -- progress curves
# --------------------------------------------------------------------------- #

def fig2_progress(df: pd.DataFrame, stab: pd.DataFrame, model: str) -> None:
    layers = sorted(df[df["layer"] >= 0]["layer"].unique())
    fig, axes = _grid(len(layers))
    for ax, layer in zip(axes, layers):
        for mat in LAYER_MATS:
            s = _series(df, layer, mat)
            ax.plot(s["step"], s["p_norm"], lw=1.1, color=MATRIX_COLOR[mat],
                    label=MATRIX_LABEL[mat])
            row = stab[(stab["layer"] == layer) & (stab["matrix"] == mat)]
            for col, lvl, mk in (("t50", 0.5, "o"), ("t90", 0.9, "^")):
                if not row.empty and np.isfinite(row[col].iloc[0]):
                    ax.plot(row[col].iloc[0], lvl, mk, ms=4.5,
                            color=MATRIX_COLOR[mat], mec="k", mew=0.4, zorder=5)
        ax.axhline(0.5, color="0.55", lw=0.6, ls=":")
        ax.axhline(0.9, color="0.55", lw=0.6, ls=":")
        _xlog(ax, sorted(df["step"].unique()))
        ax.set_ylim(-0.04, 1.06)
        ax.set_title(f"layer {layer}")
    for ax in axes[len(layers):]:
        ax.set_visible(False)
    axes[0].legend(ncol=2, loc="upper left", framealpha=0.85, handlelength=1.4)
    fig.supxlabel("training step")
    fig.supylabel(r"directional progress $p_{\mathrm{norm}}(t)$")
    fig.suptitle(f"directional progress; circles $t_{{50}}$, "
                 f"triangles $t_{{90}}$  ({model.split('/')[-1]})")
    _save(fig, f"fig2_progress_{config._slug(model)}")


# --------------------------------------------------------------------------- #
# Fig 3 -- kinematic wavefront
# --------------------------------------------------------------------------- #

def fig3_wavefront(df: pd.DataFrame, stab: pd.DataFrame, model: str) -> None:
    """One small heatmap per matrix type: layer (y) x step (x), colour = rate.

    The bottom-up wavefront is the rightward tilt of the t50 marker line as you
    move up a panel. Splitting by matrix type keeps the figure page-sized and
    lets H1 be read separately for each matrix.
    """
    steps = sorted(df["step"].unique())
    layers = sorted(df[df["layer"] >= 0]["layer"].unique())
    idx = {v: i for i, v in enumerate(steps)}
    xs = np.array([s if s > 0 else 0.5 for s in steps], float)

    grids = {}
    for mat in LAYER_MATS:
        g = np.full((len(layers), len(steps)), np.nan)
        for r, layer in enumerate(layers):
            s = _series(df, layer, mat)
            for st, rt in zip(s["step"], s["rate"]):
                g[r, idx[int(st)]] = rt
        # step 1 is byte-identical to step 0 for these checkpoints, so its rate
        # is exactly 0; log10 would send it to -inf and destroy the colour scale.
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.log10(np.where(g > 0, g, np.nan))
        grids[mat] = g
    allv = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    lo, hi = float(np.percentile(allv, 1)), float(np.percentile(allv, 99))

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4), sharex=True, sharey=True)
    for ax, mat in zip(axes.ravel(), LAYER_MATS):
        im = ax.imshow(grids[mat], aspect="auto", cmap="magma_r", origin="lower",
                       interpolation="nearest", vmin=lo, vmax=hi)
        for col, mk in (("t50", "o"), ("t90", "^")):
            xsc, ysc = [], []
            for r, layer in enumerate(layers):
                row = stab[(stab["layer"] == layer) & (stab["matrix"] == mat)]
                if row.empty or not np.isfinite(row[col].iloc[0]):
                    continue
                t = max(row[col].iloc[0], 0.5)
                xsc.append(np.interp(np.log10(t), np.log10(xs),
                                     np.arange(len(steps))))
                ysc.append(r)
            ax.plot(xsc, ysc, mk + "-", ms=3.0, lw=0.8, color="white", mec="k",
                    mew=0.4, label=rf"$t_{{{col[1:]}}}$")
        ax.set_title(MATRIX_LABEL[mat], pad=3)
        ax.grid(False)
        ticks = [i for i, s in enumerate(steps)
                 if s in (0, 16, 256, 2000, 16000, 64000, 143000)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{steps[i]:,}" for i in ticks], rotation=90, fontsize=7)
        ax.set_yticks(range(0, len(layers), 2))
        ax.set_yticklabels([str(layers[i]) for i in range(0, len(layers), 2)],
                           fontsize=7)
    axes[0, 0].legend(loc="upper left", fontsize=6.5, framealpha=0.9,
                      handlelength=1.2, borderpad=0.3)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015)
    cb.set_label(r"$\log_{10}$ displacement per 1000 steps", fontsize=8)
    fig.supxlabel("training step")
    fig.supylabel("layer index")
    fig.suptitle(f"kinematic wavefront ({model.split('/')[-1]}); "
                 f"markers: per-layer t50 and t90")
    _save(fig, f"fig3_wavefront_{config._slug(model)}")


# --------------------------------------------------------------------------- #
# Fig 4 -- stabilization vs depth (headline, H1)
# --------------------------------------------------------------------------- #

def fig4_depth(stab: pd.DataFrame, model: str, res: dict | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True)
    for ax, col in zip(axes, ("t50", "t90")):
        pl = hyp.per_layer_frame(stab, col)
        for grp, mats in (("attn", hyp.ATTN_MATS), ("ffn", hyp.FFN_MATS)):
            lo = stab[stab["matrix"].isin(mats) & (stab["layer"] >= 0)] \
                .groupby("layer")[col].quantile(0.25)
            hi = stab[stab["matrix"].isin(mats) & (stab["layer"] >= 0)] \
                .groupby("layer")[col].quantile(0.75)
            ax.fill_between(pl["layer"], lo.reindex(pl["layer"]).to_numpy(),
                            hi.reindex(pl["layer"]).to_numpy(),
                            color=GROUP_COLOR[grp], alpha=0.14, lw=0)
            ax.plot(pl["layer"], pl[grp], marker="o", ms=4.5, lw=1.5,
                    color=GROUP_COLOR[grp],
                    label=("attention" if grp == "attn" else "FFN"))
        for mat, mk in (("embed_in", "s"), ("embed_out", "D")):
            v = stab[stab["matrix"] == mat][col]
            if len(v) and np.isfinite(v.iloc[0]):
                ax.axhline(v.iloc[0], color=MATRIX_COLOR[mat], ls="--", lw=1.0,
                           label=MATRIX_LABEL[mat])
        ax.set_yscale("log")
        ax.set_xlabel("layer index (0 = closest to the input)")
        ax.set_title(f"${{\\it t}}_{{{col[1:]}}}$")
        r = (res or {}).get(col, {}).get("H1", {})
        if r:
            ax.text(0.03, 0.965,
                    rf"$\rho={r['rho_all']:+.2f}$, $p={r['p_all']:.3g}$",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(fc="white", ec="0.7", lw=0.5, pad=2.5))
    axes[0].set_ylabel("stabilization step (Pythia axis, log scale)")
    axes[0].legend(loc="lower right", framealpha=0.9, ncol=2)
    fig.suptitle(f"stabilization time vs depth  ({model.split('/')[-1]}); "
                 f"line = per-layer median, band = IQR across matrices")
    _save(fig, f"fig4_depth_{config._slug(model)}")


# --------------------------------------------------------------------------- #
# Fig 5 -- scale overlay (H5)
# --------------------------------------------------------------------------- #

def fig5_scale(tables: dict[str, pd.DataFrame], res: dict | None = None) -> None:
    """One panel per model: overlaying three models in one axis was unreadable."""
    if len(tables) < 2:
        print("[plot] fig5 skipped (needs >= 2 models)")
        return
    names = list(tables)
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(2.55 * n + 0.6, 4.8),
                             sharex=True, squeeze=False)
    for r, col in enumerate(("t50", "t90")):
        row = [t[col].dropna() for t in tables.values()]
        lo = min(float(v.min()) for v in row) * 0.9
        hi = max(float(v.max()) for v in row) * 1.1
        for c, name in enumerate(names):
            ax = axes[r][c]
            pl = hyp.per_layer_frame(tables[name], col)
            rel = pl["layer"] / max(len(pl) - 1, 1)
            for grp, lab in (("attn", "attention"), ("ffn", "FFN")):
                ax.plot(rel, pl[grp], marker="o", ms=3.5, lw=1.5,
                        color=GROUP_COLOR[grp], label=lab)
            ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            if r == 0:
                ax.set_title(f"{name.split('/')[-1]}  ({len(pl)} layers)")
            if r == 1:
                ax.set_xlabel("relative depth")
            if c == 0:
                ax.set_ylabel(f"${{\\it t}}_{{{col[1:]}}}$  (log scale)")
            rr = ((res or {}).get(col, {}).get("per_model", {})
                  .get(name, {}))
            if rr:
                ax.text(0.03, 0.965,
                        rf"H1 $\rho={rr['h1_rho']:+.2f}$"
                        + ("$^*$" if rr["h1_p"] < 0.05 else ""),
                        transform=ax.transAxes, va="top", fontsize=7.5,
                        bbox=dict(fc="white", ec="0.75", lw=0.5, pad=2))
    axes[0][0].legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.suptitle("stabilization vs relative depth, by model scale "
                 "($^*$ = $p<0.05$)")
    _save(fig, "fig5_scale_overlay")


# --------------------------------------------------------------------------- #
# Fig 6 -- matrix-type ordering (H2 / H3)
# --------------------------------------------------------------------------- #

def fig6_matrix_types(stab: pd.DataFrame, model: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    pl = hyp.per_layer_frame(stab, "t50")
    mats = hyp.ATTN_MATS + hyp.FFN_MATS

    ax = axes[0]
    for _, row in pl.iterrows():
        ax.plot(range(len(mats)), [row[m] for m in mats], lw=0.8, alpha=0.45,
                color="0.5", marker="o", ms=2.5)
    med = [np.nanmedian(pl[m].to_numpy(dtype=float)) for m in mats]
    ax.plot(range(len(mats)), med, lw=2.2, color="k", marker="o", ms=6,
            label="median over layers", zorder=5)
    for i, m in enumerate(mats):
        ax.plot([i], [med[i]], "o", ms=6, color=MATRIX_COLOR[m], zorder=6)
    ax.set_xticks(range(len(mats)))
    ax.set_xticklabels([MATRIX_LABEL[m] for m in mats])
    ax.set_yscale("log")
    ax.set_ylabel(r"$t_{50}$ (log scale)")
    ax.set_title("per-layer profile (grey = one layer)")
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axes[1]
    pairs = [("attention\nvs FFN", pl["attn"], pl["ffn"]),
             ("(Q,K) vs\n(V,$W_O$)", pl["qk"], pl["v_out"])]
    for i, (lab, a, b) in enumerate(pairs):
        d = np.log10(b.to_numpy(dtype=float) / a.to_numpy(dtype=float))
        d = d[np.isfinite(d)]
        ax.scatter(np.full(len(d), i) + np.random.default_rng(0).normal(0, .045, len(d)),
                   d, s=22, color="#1f4e9c", alpha=0.75, zorder=3, edgecolor="none")
        ax.plot([i - .22, i + .22], [np.median(d)] * 2, lw=2.4, color="k", zorder=4)
    ax.axhline(0, color="0.4", lw=0.9)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p[0] for p in pairs])
    ax.set_xlim(-0.5, len(pairs) - 0.5)
    ax.set_ylabel(r"$\log_{10}$ ratio of $t_{50}$")
    ax.set_title("one point per layer; >0 means the\nfirst group stabilizes earlier")
    fig.suptitle(f"which matrix types stabilize first  "
                 f"({model.split('/')[-1]})")
    _save(fig, f"fig6_matrix_types_{config._slug(model)}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def plot_model(model: str) -> None:
    import json
    kin = pd.read_parquet(config.kinematics_parquet(model))
    stab = pd.read_csv(config.stabilization_csv(model))
    res = None
    jp = config.RESULTS_DIR / f"hypotheses_{config._slug(model)}.json"
    if jp.exists():
        res = json.loads(jp.read_text())
    fig1_convergence(kin, model)
    fig2_progress(kin, stab, model)
    fig3_wavefront(kin, stab, model)
    fig4_depth(stab, model, res)
    fig6_matrix_types(stab, model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=config.PRIMARY_MODEL, choices=list(config.MODELS))
    ap.add_argument("--all", action="store_true",
                    help="plot every model that has a stabilization csv")
    args = ap.parse_args()

    models = [m for m in config.MODELS if config.stabilization_csv(m).exists()] \
        if args.all else [args.model]
    for m in models:
        plot_model(m)
    tables = {m: pd.read_csv(config.stabilization_csv(m)) for m in config.MODELS
              if config.stabilization_csv(m).exists()}
    import json
    h5p = config.RESULTS_DIR / "h5_scale.json"
    h5 = json.loads(h5p.read_text()) if h5p.exists() else None
    fig5_scale(tables, h5)


if __name__ == "__main__":
    main()
