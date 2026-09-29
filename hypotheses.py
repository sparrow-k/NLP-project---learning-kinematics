"""Statistical tests for the Phase 1 hypotheses H1-H5.

Design note on the **unit of analysis**. The tracked quantity is one
stabilization time per (layer, matrix) cell, so a 12-layer model yields 72
per-layer cells. The hypotheses, however, are statements about *depth* (H1),
about *matrix groups within a block* (H2, H3) and about *model scale* (H5).
Treating all 72 cells as independent observations of depth would be
pseudoreplication: the six matrices inside one block share that block's
training history. Every test below therefore reduces to one observation per
layer (or one per layer per matrix type, analysed as separate series with a
multiple-comparison correction) before any p-value is computed.

All tests are nonparametric, two-sided unless stated, and are reported with an
effect size and a descriptive summary -- not a p-value alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from scipy import stats

RNG_SEED = 0
N_PERM = 20000
N_BOOT = 10000

ATTN_MATS = ["q", "k", "v", "attn_out"]
FFN_MATS = ["ffn_up", "ffn_down"]
LAYER_MATS = ATTN_MATS + FFN_MATS
EMBED_MATS = ["embed_in", "embed_out"]


# --------------------------------------------------------------------------- #
# Small statistical helpers
# --------------------------------------------------------------------------- #

def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    clean = [p if np.isfinite(p) else 1.0 for p in pvals]
    order = np.argsort(clean)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * clean[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def spearman_perm(x, y, n_perm: int = N_PERM) -> tuple[float, float]:
    """Spearman rho with a permutation p-value (exact when n! <= n_perm).

    scipy's asymptotic p-value is unreliable at n = 6 layers, which is exactly
    the pythia-70m case, so the null is built by shuffling instead.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 3:
        return math.nan, math.nan
    rho = float(stats.spearmanr(x, y).statistic)
    if not np.isfinite(rho):
        return rho, math.nan
    if math.factorial(n) <= n_perm:
        null = np.array([stats.spearmanr(x, np.asarray(p)).statistic
                         for p in permutations(y)])
    else:
        rs = np.random.default_rng(RNG_SEED)
        null = np.array([stats.spearmanr(x, rs.permutation(y)).statistic
                         for _ in range(n_perm)])
    p = float((np.abs(null) >= abs(rho) - 1e-12).mean())
    return rho, p


def spearman_boot_ci(x, y, n_boot: int = N_BOOT, alpha: float = 0.05):
    """Percentile bootstrap CI for Spearman rho, resampling layers."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 4:
        return (math.nan, math.nan)
    rs = np.random.default_rng(RNG_SEED)
    vals = []
    for _ in range(n_boot):
        idx = rs.integers(0, n, n)
        if len(np.unique(x[idx])) < 3:
            continue
        r = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            vals.append(r)
    if len(vals) < 100:
        return (math.nan, math.nan)
    return (float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


def wilcoxon_paired(a, b) -> dict:
    """Wilcoxon signed-rank on paired samples + matched-pairs rank-biserial r.

    r = (W_plus - W_minus) / (W_plus + W_minus), in [-1, 1]. A negative value
    means `a` is smaller than `b` overall. scipy uses the exact null for
    n <= 25 pairs, which covers every model here (6 or 12 layers).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    both = np.isfinite(d)
    keep = both & (d != 0)
    n = int(keep.sum())
    out = dict(n_pairs=int(both.sum()), n_nonzero=n, W=math.nan, p=math.nan,
               r_rb=math.nan, n_a_lower=int(np.sum(d[both] < 0)))
    if n < 3:
        return out
    ranks = stats.rankdata(np.abs(d[keep]))
    w_pos = float(ranks[d[keep] > 0].sum())
    w_neg = float(ranks[d[keep] < 0].sum())
    res = stats.wilcoxon(a[both], b[both])
    out.update(W=float(res.statistic), p=float(res.pvalue),
               r_rb=(w_pos - w_neg) / (w_pos + w_neg))
    return out


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    return f"p={p:.4g}" + ("  *" if p < 0.05 else "")


def _med(s) -> float:
    v = np.asarray(s, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else math.nan


# --------------------------------------------------------------------------- #
# Per-layer reduction -- the anti-pseudoreplication step
# --------------------------------------------------------------------------- #

def per_layer_frame(stab: pd.DataFrame, col: str = "t90") -> pd.DataFrame:
    """One row per layer: median `col` overall, per group, and per matrix type."""
    lay = stab[stab["layer"] >= 0]
    rows = []
    for layer, g in lay.groupby("layer"):
        rec = {"layer": int(layer), "all": _med(g[col])}
        rec["attn"] = _med(g[g["matrix"].isin(ATTN_MATS)][col])
        rec["ffn"] = _med(g[g["matrix"].isin(FFN_MATS)][col])
        rec["qk"] = _med(g[g["matrix"].isin(["q", "k"])][col])
        rec["v_out"] = _med(g[g["matrix"].isin(["v", "attn_out"])][col])
        for m in LAYER_MATS:
            rec[m] = _med(g[g["matrix"] == m][col])
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# H1 -- depth ordering
# --------------------------------------------------------------------------- #

def h1_depth(stab: pd.DataFrame, col: str = "t90") -> tuple[str, dict]:
    pl = per_layer_frame(stab, col)
    out = [f"H1  Stabilization is bottom-up: {col} increases with layer index.",
           f"    Unit of analysis: one layer = one observation ({len(pl)} layers).",
           "    Spearman rho against a permutation null."]

    sub = pl.dropna(subset=["all"])
    rho, p = spearman_perm(sub["layer"].to_numpy(), sub["all"].to_numpy())
    lo, hi = spearman_boot_ci(sub["layer"].to_numpy(), sub["all"].to_numpy())
    out.append(f"    primary (median {col} per layer):  rho={rho:+.3f}  {fmt_p(p)}"
               f"   95% CI [{lo:+.2f}, {hi:+.2f}]   n={len(sub)}")
    res = {"rho_all": rho, "p_all": p, "ci_all": (lo, hi), "n_layers": int(len(sub))}

    out.append("    per matrix type (independent depth series, Holm-corrected "
               f"across {len(LAYER_MATS)} tests):")
    rhos, ps = {}, []
    for m in LAYER_MATS:
        s = pl.dropna(subset=[m])
        r, pp = spearman_perm(s["layer"].to_numpy(), s[m].to_numpy())
        rhos[m] = r
        ps.append(pp)
    ps_adj = holm(ps)
    for m, padj, praw in zip(LAYER_MATS, ps_adj, ps):
        out.append(f"      {m:9s} rho={rhos[m]:+.3f}   p_raw={praw:.4g}   "
                   f"p_holm={padj:.4g}" + ("  *" if padj < 0.05 else ""))
    res["rho_by_matrix"] = rhos
    res["p_holm_by_matrix"] = dict(zip(LAYER_MATS, ps_adj))
    pos = sum(1 for v in rhos.values() if np.isfinite(v) and v > 0)
    out.append(f"    direction: {pos}/{len(LAYER_MATS)} matrix types have rho > 0.")
    res["n_positive"] = pos
    return "\n".join(out), res


# --------------------------------------------------------------------------- #
# H2 -- attention vs FFN, paired within layer
# --------------------------------------------------------------------------- #

def h2_attn_vs_ffn(stab: pd.DataFrame, col: str = "t90") -> tuple[str, dict]:
    pl = per_layer_frame(stab, col)
    a, b = pl["attn"].to_numpy(dtype=float), pl["ffn"].to_numpy(dtype=float)
    w = wilcoxon_paired(a, b)
    ratio = np.log10(b / a)
    ratio = ratio[np.isfinite(ratio)]
    mlr = float(np.median(ratio)) if ratio.size else math.nan
    out = ["H2  Within a block, attention stabilizes before FFN.",
           f"    Unit of analysis: one layer = one paired observation "
           f"(median {col} over attn matrices vs over FFN matrices).",
           f"    layers paired: {w['n_pairs']}    median {col}: "
           f"attn={_med(a):.0f}  ffn={_med(b):.0f}",
           f"    Wilcoxon signed-rank: W={w['W']:.1f}  {fmt_p(w['p'])}   "
           f"rank-biserial r={w['r_rb']:+.3f}",
           f"    attention earlier in {w['n_a_lower']}/{w['n_pairs']} layers;  "
           f"median log10(t_ffn / t_attn) = {mlr:+.3f}  "
           "(positive means attention is earlier)"]
    return "\n".join(out), {**w, "median_log_ratio": mlr}


# --------------------------------------------------------------------------- #
# H3 -- Q/K vs V/attn-out inside the attention block
# --------------------------------------------------------------------------- #

def h3_attention_internals(stab: pd.DataFrame, col: str = "t90") -> tuple[str, dict]:
    pl = per_layer_frame(stab, col)
    out = ["H3  Routing geometry (Q, K) stabilizes before value transport (V, attn_out).",
           "    Unit of analysis: one layer = one block of four paired conditions."]
    med = {m: _med(pl[m]) for m in ATTN_MATS}
    out.append(f"    median {col} by matrix:  " +
               "  ".join(f"{m}={med[m]:.0f}" for m in ATTN_MATS))

    wide = pl[["layer"] + ATTN_MATS].dropna()
    res: dict = {"median_by_matrix": med, "n_layers": int(len(wide))}
    if len(wide) >= 3:
        chi, p = stats.friedmanchisquare(*[wide[m].to_numpy(dtype=float)
                                           for m in ATTN_MATS])
        kw = float(chi / (len(wide) * (len(ATTN_MATS) - 1)))  # Kendall's W
        out.append(f"    Friedman across q/k/v/attn_out: chi2={chi:.2f}  "
                   f"{fmt_p(p)}   Kendall's W={kw:.3f}")
        res.update(friedman_chi2=float(chi), friedman_p=float(p), kendall_w=kw)

        qk = wide[["q", "k"]].mean(axis=1).to_numpy(dtype=float)
        vo = wide[["v", "attn_out"]].mean(axis=1).to_numpy(dtype=float)
        w = wilcoxon_paired(qk, vo)
        out.append(f"    planned contrast (q,k) vs (v,attn_out): W={w['W']:.1f}  "
                   f"{fmt_p(w['p'])}   rank-biserial r={w['r_rb']:+.3f}   "
                   f"(q,k) earlier in {w['n_a_lower']}/{w['n_pairs']} layers")
        res["contrast"] = w

        pairs, ps, rs = [], [], []
        for m1, m2 in combinations(ATTN_MATS, 2):
            ww = wilcoxon_paired(wide[m1].to_numpy(dtype=float),
                                 wide[m2].to_numpy(dtype=float))
            pairs.append((m1, m2)); ps.append(ww["p"]); rs.append(ww["r_rb"])
        adj = holm(ps)
        out.append("    pairwise (Holm-corrected):")
        for (m1, m2), r, pa in zip(pairs, rs, adj):
            out.append(f"      {m1:8s} vs {m2:8s}  r={r:+.3f}  p_holm={pa:.4g}"
                       + ("  *" if pa < 0.05 else ""))
        res["pairwise"] = {f"{x}_vs_{y}": dict(r=r, p_holm=pa)
                           for (x, y), r, pa in zip(pairs, rs, adj)}
    else:
        out.append(f"    not enough complete layers ({len(wide)})")
    return "\n".join(out), res


# --------------------------------------------------------------------------- #
# H4 -- embeddings stay mobile longest
# --------------------------------------------------------------------------- #

def h4_embeddings(stab: pd.DataFrame, kin: pd.DataFrame, final_step: int,
                  col: str = "t90") -> tuple[str, dict]:
    out = ["H4  Embeddings keep moving longest (latest stabilization of all "
           "tracked matrices).",
           "    Two read-outs: (a) the rank of the embedding stabilization times "
           "among all tracked matrices with an exact combinatorial p; "
           "(b) directional progress at the last checkpoint before the reference."]
    lay = stab[stab["layer"] >= 0][col].to_numpy(dtype=float)
    emb = stab[stab["matrix"].isin(EMBED_MATS)][["matrix", col]]
    res: dict = {}

    emb_vals = emb[col].to_numpy(dtype=float)
    out.append("    embedding " + col + ":  " +
               "  ".join(f"{r.matrix}={getattr(r, col):.0f}" for r in emb.itertuples()))
    lay_f = lay[np.isfinite(lay)]
    if emb_vals.size and np.all(np.isfinite(emb_vals)) and lay_f.size:
        pool = np.concatenate([lay_f, emb_vals])
        n, k = len(pool), len(emb_vals)
        ranks = [int((pool > v).sum()) + 1 for v in emb_vals]  # 1 = latest
        out.append(f"    rank among all {n} tracked matrices "
                   f"(1 = latest-stabilizing): {ranks}")
        worst = max(ranks)
        p_obs = 1.0
        for i in range(k):
            p_obs *= (worst - i) / (n - i)
        out.append(f"    exact one-sided p (both embeddings ranking this late by "
                   f"chance) = {p_obs:.4g}" + ("  *" if p_obs < 0.05 else ""))
        res.update(ranks=ranks, p_exact=float(p_obs), n_pool=int(n))
    else:
        out.append(f"    embedding {col} is not finite for both embeddings "
                   "-- they never reach the threshold, which is itself "
                   "consistent with H4.")
        res.update(ranks=None, p_exact=math.nan,
                   never_reached=bool(not np.all(np.isfinite(emb_vals))))

    cos = kin[kin["metric"] == "cos_to_final"]
    pre = sorted(s for s in cos["step"].unique() if s < final_step)
    last = pre[-1] if pre else int(cos["step"].max())
    at = cos[cos["step"] == last]
    e = at[at["matrix"].isin(EMBED_MATS)]["value"]
    l = at[at["layer"] >= 0]["value"]
    out.append(f"    cos_to_final at step {last} (last checkpoint before the "
               f"reference):")
    out.append(f"      embeddings max={e.max():.4f}   layer matrices "
               f"min={l.min():.4f}  median={l.median():.4f}")
    verdict = ("HOLDS -- both embeddings are further from their final direction "
               "than every layer matrix"
               if e.max() < l.min() else
               "not strictly below every layer matrix")
    out.append(f"      verdict: {verdict}")
    res.update(last_pre_step=int(last), embed_cos_max=float(e.max()),
               layer_cos_min=float(l.min()), strict=bool(e.max() < l.min()))
    return "\n".join(out), res


# --------------------------------------------------------------------------- #
# H5 -- consistency across model scale
# --------------------------------------------------------------------------- #

def h5_scale(tables: dict[str, pd.DataFrame], col: str = "t90") -> tuple[str, dict]:
    """`tables` maps a model name to its stabilization table."""
    out = ["H5  The ordering is consistent across model sizes.",
           "    Depth is normalized to relative depth layer/(L-1) so models with "
           "different L are comparable. The step axis needs no rescaling: every "
           "Pythia model in this suite is trained for the same 143k steps on the "
           "same data order."]
    if len(tables) < 2:
        out.append(f"    only {len(tables)} model(s) available -- not testable.")
        return "\n".join(out), {"n_models": len(tables)}

    res: dict = {"n_models": len(tables), "per_model": {}}
    rows = []
    for name, stab in tables.items():
        pl = per_layer_frame(stab, col)
        L = len(pl)
        sub = pl.dropna(subset=["all"])
        rho, p = spearman_perm(sub["layer"].to_numpy(), sub["all"].to_numpy())
        w = wilcoxon_paired(pl["attn"].to_numpy(dtype=float),
                            pl["ffn"].to_numpy(dtype=float))
        lr = np.log10(pl["ffn"].to_numpy(dtype=float) /
                      pl["attn"].to_numpy(dtype=float))
        lr = lr[np.isfinite(lr)]
        qk_vo = wilcoxon_paired(pl["qk"].to_numpy(dtype=float),
                                pl["v_out"].to_numpy(dtype=float))
        rec = dict(model=name.split("/")[-1], L=L, h1_rho=rho, h1_p=p,
                   h2_r=w["r_rb"], h2_p=w["p"],
                   h2_log_ratio=float(np.median(lr)) if lr.size else math.nan,
                   h3_r=qk_vo["r_rb"], h3_p=qk_vo["p"])
        rows.append(rec)
        res["per_model"][name] = rec
    summ = pd.DataFrame(rows)
    out.append("    " + summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"
                                       ).replace("\n", "\n    "))

    def agree(key):
        signs = {np.sign(r[key]) for r in rows if np.isfinite(r[key])}
        return len(signs) == 1 and 0.0 not in signs

    a1, a2, a3 = agree("h1_rho"), agree("h2_r"), agree("h3_r")
    out.append(f"    sign agreement across models:  H1 {'yes' if a1 else 'NO'}   "
               f"H2 {'yes' if a2 else 'NO'}   H3 {'yes' if a3 else 'NO'}")
    res.update(h1_sign_agrees=bool(a1), h2_sign_agrees=bool(a2),
               h3_sign_agrees=bool(a3))

    # Pairwise agreement of the full depth profile, on a common relative-depth
    # grid so models with different layer counts are comparable.
    names = list(tables)
    if len(names) >= 2:
        grid = np.linspace(0, 1, 11)
        prof = {}
        for name in names:
            pl = per_layer_frame(tables[name], col)
            L = len(pl)
            rel = pl["layer"].to_numpy(dtype=float) / max(L - 1, 1)
            vec = []
            for m in LAYER_MATS:
                y = np.log10(pl[m].to_numpy(dtype=float))
                ok = np.isfinite(y)
                vec.append(np.interp(grid, rel[ok], y[ok]) if ok.sum() >= 2
                           else np.full_like(grid, np.nan))
            prof[name] = np.concatenate(vec)

        pairs = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = prof[names[i]], prof[names[j]]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= 5:
                    r = float(stats.spearmanr(a[ok], b[ok]).statistic)
                    pairs[f"{names[i].split('/')[-1]}__"
                          f"{names[j].split('/')[-1]}"] = r
        if pairs:
            out.append(f"    Spearman between models' log10({col}) profiles "
                       f"({len(LAYER_MATS)} matrix types x {len(grid)} "
                       f"relative-depth points):")
            for k, v in pairs.items():
                out.append(f"      {k.replace('__', ' vs '):34s} rho={v:+.3f}")
            res["profile_rho_pairs"] = pairs
            # headline figure: the mean pairwise agreement
            res["profile_rho"] = float(np.mean(list(pairs.values())))
            res["profile_rho_min"] = float(np.min(list(pairs.values())))
            if len(pairs) > 1:
                out.append(f"      {'mean over pairs':34s} "
                           f"rho={res['profile_rho']:+.3f}   "
                           f"(weakest pair {res['profile_rho_min']:+.3f})")
            out.append("    (descriptive only: the interpolated grid points are "
                       "not independent, so no p-value is quoted.)")
    return "\n".join(out), res


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

@dataclass
class Report:
    text: str
    results: dict = field(default_factory=dict)


def run_all(stab: pd.DataFrame, kin: pd.DataFrame, model: str, final_step: int,
            col: str = "t90") -> Report:
    parts = [f"Phase 1 hypothesis tests -- {model}",
             f"Stabilization scalar: {col}", "=" * 72, ""]
    res: dict = {}
    for fn, key in ((h1_depth, "H1"), (h2_attn_vs_ffn, "H2"),
                    (h3_attention_internals, "H3")):
        txt, r = fn(stab, col)
        parts += [txt, ""]
        res[key] = r
    txt, r = h4_embeddings(stab, kin, final_step, col)
    parts += [txt, ""]
    res["H4"] = r
    return Report("\n".join(parts), res)
