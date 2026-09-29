r"""Emit report/numbers.tex: every quantitative claim in the paper as a macro.

The paper never hard-codes a number. It says \HoneRhoBig, and this script writes
that macro from results/hypotheses_pythia160m.json and the Phase 2 outputs. Any
re-run of the pipeline therefore updates the paper, and a claim in the text
cannot silently drift away from the result it came from.

    .venv/Scripts/python make_report_numbers.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

import config

REPORT = config.ROOT / "report"
MACROS: dict[str, str] = {}


def mac(name: str, value) -> None:
    MACROS[name] = str(value)


def num(v, nd: int = 2, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    s = f"{v * 100:.{nd}f}" if pct else f"{v:.{nd}f}"
    return s


def pval(p) -> str:
    r"""A p-value macro carries its own relational operator.

    The paper writes "$p \Macro$", so this must expand to "= 0.0014" or
    "< 0.001". Returning a bare "<0.001" produced the nonsense "p =< 0.001".
    """
    if p is None or not math.isfinite(float(p)):
        return "= n/a"
    p = float(p)
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3g}"


def sgn(v, nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:+.{nd}f}"


def star(p) -> str:
    """LaTeX significance marker, so the table never hand-codes one."""
    try:
        return ("$^{*}$" if p is not None and math.isfinite(float(p))
                and float(p) < 0.05 else r"\phantom{$^{*}$}")
    except (TypeError, ValueError):
        return r"\phantom{$^{*}$}"


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #

def phase1() -> None:
    for model, tag in (("EleutherAI/pythia-160m", "Big"),
                       ("EleutherAI/pythia-70m", "Small"),
                       ("EleutherAI/pythia-410m", "Huge")):
        slug = config._slug(model)
        jp = config.RESULTS_DIR / f"hypotheses_{slug}.json"
        if not jp.exists():
            continue
        r = json.loads(jp.read_text())
        for scalar, stag in (("t50", ""), ("t90", "Ninety")):
            if scalar not in r:
                continue
            d = r[scalar]
            h1 = d["H1"]
            mac(f"HoneRho{tag}{stag}", sgn(h1["rho_all"]))
            mac(f"HoneP{tag}{stag}", pval(h1["p_all"]))
            mac(f"HoneSig{tag}{stag}", star(h1["p_all"]))
            ci = h1.get("ci_all") or [None, None]
            mac(f"HoneCI{tag}{stag}",
                f"[{sgn(ci[0])}, {sgn(ci[1])}]" if ci[0] is not None else "n/a")
            mac(f"HoneN{tag}{stag}", h1["n_layers"])
            mac(f"HonePos{tag}{stag}", h1["n_positive"])
            for m, nm in (("q", "Q"), ("k", "K"), ("v", "V"),
                          ("attn_out", "O"), ("ffn_up", "Fin"),
                          ("ffn_down", "Fout")):
                mac(f"HoneRho{nm}{tag}{stag}", sgn(h1["rho_by_matrix"].get(m)))
                mac(f"HoneP{nm}{tag}{stag}",
                    pval(h1["p_holm_by_matrix"].get(m)))
                mac(f"HoneSig{nm}{tag}{stag}",
                    star(h1["p_holm_by_matrix"].get(m)))
            h2 = d["H2"]
            mac(f"HtwoR{tag}{stag}", sgn(h2["r_rb"]))
            mac(f"HtwoP{tag}{stag}", pval(h2["p"]))
            mac(f"HtwoSig{tag}{stag}", star(h2["p"]))
            mac(f"HtwoLower{tag}{stag}", h2["n_a_lower"])
            mac(f"HtwoPairs{tag}{stag}", h2["n_pairs"])
            h3 = d["H3"]
            mac(f"HthreeChi{tag}{stag}", num(h3.get("friedman_chi2")))
            mac(f"HthreeP{tag}{stag}", pval(h3.get("friedman_p")))
            mac(f"HthreeW{tag}{stag}", num(h3.get("kendall_w"), 3))
            c = h3.get("contrast", {})
            mac(f"HthreeCR{tag}{stag}", sgn(c.get("r_rb"), 2))
            mac(f"HthreeCP{tag}{stag}", pval(c.get("p")))
            mac(f"HthreeCSig{tag}{stag}", star(c.get("p")))
            mac(f"HthreeCLower{tag}{stag}", c.get("n_a_lower"))
            mac(f"HthreeCPairs{tag}{stag}", c.get("n_pairs"))
            short = {"q": "Q", "k": "K", "v": "V", "attn_out": "O"}
            for key, pr in (h3.get("pairwise") or {}).items():
                m1, m2 = key.split("_vs_")
                nm = short[m1] + short[m2]
                mac(f"Hthree{nm}R{tag}{stag}", sgn(pr["r"], 2))
                mac(f"Hthree{nm}P{tag}{stag}", pval(pr["p_holm"]))
            for m, nm in (("q", "Q"), ("k", "K"), ("v", "V"),
                          ("attn_out", "O")):
                mac(f"Hthree{nm}Med{tag}{stag}",
                    f"{h3['median_by_matrix'][m]:,.0f}")
            h4 = d["H4"]
            mac(f"HfourRanks{tag}{stag}",
                ", ".join(str(x) for x in (h4.get("ranks") or [])))
            mac(f"HfourP{tag}{stag}", pval(h4.get("p_exact")))
            mac(f"HfourPool{tag}{stag}", h4.get("n_pool"))

        stab = pd.read_csv(config.stabilization_csv(model))
        mac(f"NMat{tag}", len(stab))
        mac(f"NLayer{tag}", config.MODELS[model]["n_layer"])
        for c, nm in (("t50", "Tfifty"), ("t90", "Tninety")):
            mac(f"{nm}Med{tag}", f"{stab[c].median():,.0f}")
            mac(f"{nm}Min{tag}", f"{stab[c].min():,.0f}")
            mac(f"{nm}Max{tag}", f"{stab[c].max():,.0f}")
        # Stationarity: matrices whose displacement rate falls, and stays,
        # below the smallest threshold of the sweep.
        tau = min(config.TAU_SWEEP)
        ts = stab[f"t_star@tau={tau}"]
        mac(f"TauMin", f"{tau:g}")
        mac(f"TauBelowN{tag}", int(ts.notna().sum()))
        mac(f"TauBelowMin{tag}", f"{ts.min():,.0f}")
        mac(f"TauBelowMax{tag}", f"{ts.max():,.0f}")
        mac(f"TauBelowMed{tag}", f"{ts.median():,.0f}")
        emb = stab[stab["matrix"] == "embed_out"]["t50"]
        if len(emb):
            mac(f"EmbOutTfifty{tag}", f"{emb.iloc[0]:,.0f}")
        embi = stab[stab["matrix"] == "embed_in"]["t50"]
        if len(embi):
            mac(f"EmbInTfifty{tag}", f"{embi.iloc[0]:,.0f}")

    h5p = config.RESULTS_DIR / "h5_scale.json"
    if h5p.exists():
        h5 = json.loads(h5p.read_text())
        for scalar, stag in (("t50", ""), ("t90", "Ninety")):
            if scalar in h5:
                mac(f"HfiveRho{stag}", sgn(h5[scalar].get("profile_rho"), 3))
                for k, nm in (("h1_sign_agrees", "Hone"),
                              ("h2_sign_agrees", "Htwo"),
                              ("h3_sign_agrees", "Hthree")):
                    mac(f"Hfive{nm}Agree{stag}",
                        "agree" if h5[scalar].get(k) else "disagree")

    mac("NCheckpoints", len(config.CHECKPOINT_STEPS))
    mac("FinalStep", f"{config.FINAL_STEP:,}")


# --------------------------------------------------------------------------- #
# Phase 2
# --------------------------------------------------------------------------- #

ARM_MACRO = {"baseline": "Base", "attn-first": "AF", "attn-second": "AS",
             "alternating": "Alt", "attn-freeze": "Frz",
             "attn-first-lrm": "AFL", "attn-second-lrm": "ASL",
             "attn-first_vs_attn-second": "AFvAS",
             "attn-first-lrm_vs_attn-second-lrm": "AFLvASL"}
METRIC_MACRO = {"steps_to_target": "Steps", "wall_to_target": "WallT",
                "final_val_ppl": "Ppl", "train_wall_s": "Wall",
                "peak_vram_gb": "Vram"}


def phase2() -> None:
    sp = config.RESULTS_DIR / "phase2_summary.csv"
    if not sp.exists():
        return
    per = pd.read_csv(sp)
    target = float(per["target_ppl"].iloc[0])
    mac("TargetPpl", f"{target:g}")
    mac("PhaseTwoSteps", f"{int(per['max_steps'].iloc[0]):,}")
    mac("PhaseTwoTokens", f"{int(per['total_tokens'].iloc[0]) / 1e6:.1f}")
    mac("PhaseTwoTokensPerStep", f"{int(per['tokens_per_step'].iloc[0]):,}")
    mac("PhaseTwoParams", f"{int(per['n_params'].iloc[0]) / 1e6:.1f}")
    mac("PhaseTwoRuns", len(per))

    # Overfit sanity check on the training corpus (train.py --overfit).
    so = config.ROOT / "runs" / "sanity_overfit_wikitext-103-raw-v1.json"
    if so.exists():
        o = json.loads(so.read_text())
        mac("OverfitStart", num(o["history"][0]["loss"], 2))
        mac("OverfitFinal", num(o["final_loss"], 3))
        mac("OverfitBatches", o["n_batches"])

    # Parameter share of each schedule group (identical in every run).
    run0 = config.ROOT / "runs" / per["run_id"].iloc[0] / "config.json"
    if run0.exists():
        groups = json.loads(run0.read_text())["optimizer_groups"]
        tot = sum(g["n_params"] for g in groups)
        for grp, nm in (("attn", "Attn"), ("ffn", "Ffn"), ("other", "Other")):
            n = sum(g["n_params"] for g in groups if g["sched_group"] == grp)
            mac(f"ParamPct{nm}", num(n / tot * 100, 0))
            mac(f"ParamM{nm}", num(n / 1e6, 1))

    # Integrated LR per group relative to the baseline, from the schedule the
    # runs actually used (trainer defaults; recorded in each config.json).
    import schedules
    import train as tr
    rc = tr.RunConfig()
    per_layer = {}
    for arm in ("attn-first", "attn-first-lrm"):
        sch = schedules.build(arm, int(per["max_steps"].iloc[0]), rc.phase1_model,
                              ratio=rc.emphasis_ratio, scalar=rc.phase1_scalar,
                              warmup=rc.warmup, min_ratio=rc.min_lr_ratio)
        b = sch.budget_check()
        a = ARM_MACRO[arm]
        mac(f"{a}LrAttn", num(b["attn_lr"], 3))
        mac(f"{a}LrFfn", num(b["ffn_lr"], 3))
        mac(f"{a}StepAvgAttn", num(b["attn"], 3))
        mac(f"{a}StepAvgFfn", num(b["ffn"], 3))
        if arm == "attn-first":
            g = sch.lr_shape
            gt = sum(g)
            vals = [sum(gi * sch.multiplier(t, "attn", l) for t, gi in enumerate(g)) / gt
                    for l in sorted(sch.sigma["attn"])]
            mac("BudgetEarlyPct", num((b["attn_lr"] - 1) * 100, 1))
            mac("BudgetLatePct", num((1 - b["ffn_lr"]) * 100, 1))
            mac("BudgetEarlyMaxPct", num((max(vals) - 1) * 100, 1))
            mac("BudgetEarlyMinPct", num((min(vals) - 1) * 100, 1))

    for arm, g in per.groupby("arm"):
        if arm not in ARM_MACRO:
            continue
        a = ARM_MACRO[arm]
        mac(f"{a}Seeds", g["seed"].nunique())
        mac(f"{a}Ppl", num(g["final_val_ppl"].mean()))
        mac(f"{a}PplSD", num(g["final_val_ppl"].std(ddof=1)))
        mac(f"{a}Steps", f"{g['steps_to_target'].mean():,.0f}")
        mac(f"{a}StepsSD", f"{g['steps_to_target'].std(ddof=1):,.0f}")
        mac(f"{a}Wall", f"{g['train_wall_s'].mean():,.0f}")
        mac(f"{a}Vram", num(g["peak_vram_gb"].mean()))
        mac(f"{a}DispAttn", num(g["disp_attn"].mean(), 3))
        mac(f"{a}DispFfn", num(g["disp_ffn"].mean(), 3))

    stp = config.RESULTS_DIR / "phase2_stats.csv"
    if stp.exists():
        st = pd.read_csv(stp)
        for _, r in st.iterrows():
            if r["arm"] not in ARM_MACRO or r["metric"] not in METRIC_MACRO:
                continue
            k = ARM_MACRO[r["arm"]] + METRIC_MACRO[r["metric"]]
            mac(f"{k}Diff", sgn(r["mean_diff"], 2))
            mac(f"{k}Rel", sgn(r["rel_pct"], 1) if math.isfinite(r["rel_pct"]) else "n/a")
            # Unsigned variants, for prose that already states the direction in
            # words ("4.9% fewer steps" rather than "-4.9% fewer steps").
            mac(f"{k}DiffAbs", num(abs(r["mean_diff"]), 2))
            # step counts are integers; 180.71 steps reads as false precision
            mac(f"{k}DiffAbsInt", f"{abs(r['mean_diff']):,.0f}")
            mac(f"{k}DiffInt", f"{r['mean_diff']:+,.0f}")
            mac(f"{k}RelAbs", num(abs(r["rel_pct"]), 1))
            mac(f"{k}P", pval(r["p"]))
            mac(f"{k}Nseeds", int(r["n_seeds"]))
            mac(f"{k}Worse", int(r["n_worse"]))
            mac(f"{k}MinP", pval(r["min_p"]))


ARM_TEX = {"baseline": "baseline", "attn-first": "attn-first",
           "attn-second": "attn-second",
           "attn-first-lrm": "attn-first-lrm", "attn-second-lrm": "attn-second-lrm",
           "alternating": "alternating", "attn-freeze": "attn-freeze"}


def phase2_table() -> None:
    """Write report/phase2_table.tex straight from the run summary."""
    sp = config.RESULTS_DIR / "phase2_summary.csv"
    if not sp.exists():
        return
    per = pd.read_csv(sp)
    target = float(per["target_ppl"].iloc[0])

    def cell(v, nd=2, comma=False):
        if v is None or not math.isfinite(v):
            return "---"
        return f"{v:,.0f}" if comma else f"{v:.{nd}f}"

    rows = []
    for arm in ARM_TEX:
        g = per[per["arm"] == arm]
        if g.empty:
            continue
        n = g["seed"].nunique()
        ppl, ppls = g["final_val_ppl"].mean(), g["final_val_ppl"].std(ddof=1)
        st, sts = g["steps_to_target"].mean(), g["steps_to_target"].std(ddof=1)
        wt = g["wall_to_target"].mean()
        wall, vram = g["train_wall_s"].mean(), g["peak_vram_gb"].mean()
        steps = (f"{cell(st, comma=True)} {{\\scriptsize$\\pm$ "
                 f"{cell(sts, comma=True)}}}" if math.isfinite(st) else "---")
        rows.append(
            f"{ARM_TEX[arm]} & {n} & "
            f"{cell(ppl)} {{\\scriptsize$\\pm$ {cell(ppls)}}} & "
            f"{steps} & "
            f"{cell(wt, 0)} & {cell(wall, 0)} & {cell(vram)} \\\\")

    tex = [
        r"\begin{table*}[t]", r"\centering\small",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"\textbf{arm} & \textbf{seeds} & \textbf{final val ppl} & "
        rf"\textbf{{steps to {target:g}}} & \textbf{{sec to {target:g}}} & "
        r"\textbf{train s} & \textbf{peak GiB} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Phase~2 results, mean $\pm$ SD over seeds. ``steps to "
        rf"{target:g}'' and ``sec to {target:g}'' are the training step and "
        rf"training seconds at which validation perplexity first reaches "
        rf"{target:g} (interpolated; --- if never reached). ``train s'' is total "
        rf"training time and ``peak GiB'' the peak GPU memory. At a given seed "
        rf"every arm shares the baseline's initialization and data order.}}",
        r"\label{tab:p2}", r"\end{table*}",
    ]
    (REPORT / "phase2_table.tex").write_text("\n".join(tex) + "\n")
    print("[report] wrote report/phase2_table.tex")


def copy_figures() -> None:
    """Mirror the figures the paper includes into report/figures/."""
    import shutil
    dst = REPORT / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(config.FIG_DIR.glob("*.pdf")):
        if f.stem.endswith("_demo"):   # demo-scale figures are not paper figures
            continue
        shutil.copy2(f, dst / f.name)
        n += 1
    print(f"[report] copied {n} PDF figures -> report/figures/")


def check_placeholders() -> None:
    """Fail loudly if the paper still contains an unfilled placeholder."""
    tex = REPORT / "report.tex"
    if not tex.exists():
        return
    body = tex.read_text(encoding="utf-8")
    left = [t for t in ("PHASETWOHEADLINE", "PHASETWORESULTS", "TODO", "XXX")
            if t in body]
    if left:
        print(f"[report] WARNING: report.tex still contains {left}")
    else:
        print("[report] no unfilled placeholders in report.tex")


def main() -> None:
    phase1()
    phase2()
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = ["% AUTO-GENERATED by make_report_numbers.py -- do not edit.",
             "% Every quantitative claim in main.tex resolves through one of these.",
             ""]
    for k in sorted(MACROS):
        lines.append(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}")
    (REPORT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"[report] wrote report/numbers.tex with {len(MACROS)} macros")

    # a plain-text mirror, handy for checking the paper against the results
    (REPORT / "numbers.txt").write_text(
        "\n".join(f"{k:32s} {MACROS[k]}" for k in sorted(MACROS)) + "\n")
    phase2_table()
    copy_figures()
    check_placeholders()


if __name__ == "__main__":
    main()
