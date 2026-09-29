"""Phase 1, step 2: turn the kinematics parquet into stabilization scalars + tests.

Reads  data/kinematics_<model>.parquet
Writes results/stabilization_<model>.csv     per matrix: t50/t90/t99, t_star@tau
       results/hypotheses_<model>.txt        H1-H4 test output
       results/hypotheses_<model>.json       the same numbers, machine-readable
       results/h5_scale.txt                  cross-model test (>= 2 models)

Derived quantities
------------------
  p_norm(t) = (cos_to_final(t) - cos_to_final(t_0)) / (1 - cos_to_final(t_0))
      Fraction of the final *direction* already reached at step t, in [0, 1].
      Threshold-free and scale-free.

  t_50 / t_90 / t_99
      First step where p_norm crosses 0.5 / 0.9 / 0.99, linearly interpolated
      between checkpoints in log10(step) space (step 0 is placed at
      log10(0.5)). NaN if the level is never reached.

      CAVEAT, and the reason t_90 rather than t_99 is the primary scalar:
      p_norm equals 1 at the final checkpoint *by construction*, because that
      checkpoint is the reference state. A crossing that happens only inside
      the last checkpoint interval is therefore partly definitional. The CSV
      carries a `t99_last_interval` / `t90_last_interval` flag so this can be
      audited, and the fraction of such crossings is printed.

  vel_rate(t) = velocity(t) / ((t - t_prev) / 1000)
      Fractional weight displacement per 1000 optimizer steps. The raw
      `velocity` column is a per-*interval* displacement, and the checkpoint
      grid spans intervals from 1 step to 16000 steps, so raw velocity is not
      comparable along the series and cannot be thresholded. Two candidate
      normalizations were measured on pythia-70m: per decade of log-step
      (median rises from 0.0 to 4.7 across training -- does not decay, so it
      cannot define a stabilization time) and per 1000 steps (median falls
      monotonically from 1.04 at step 128 to 0.015 at the end, a ~100x decay).
      The per-1000-step rate is used because it is the one that actually
      behaves like a velocity, and because it is the quantity a practitioner
      cares about: how much a matrix still moves per unit of optimizer work.

  t_star(tau) = earliest checkpoint t with vel_rate(t) < tau and
                vel_rate(t') < tau for every later checkpoint t'.
      Reported over a tau sweep; no single threshold is privileged.

    .venv/Scripts/python compute_stabilization.py --model EleutherAI/pythia-160m
    .venv/Scripts/python compute_stabilization.py --h5    # cross-model, after >=2 models
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

import config
import hypotheses as hyp

LOG0 = math.log10(0.5)  # sentinel x-position for step 0 in log-space interpolation


def _logx(step: float) -> float:
    return LOG0 if step <= 0 else math.log10(step)


def _cross(steps: np.ndarray, p: np.ndarray, level: float) -> tuple[float, bool]:
    """First step where p crosses `level`, interpolated in log10(step) space.

    Returns (step, crossed_in_final_interval).
    """
    for i in range(1, len(p)):
        if p[i] >= level:
            last = (i == len(p) - 1)
            if p[i - 1] >= level:
                return float(steps[i - 1]), False
            x0, x1 = _logx(float(steps[i - 1])), _logx(float(steps[i]))
            y0, y1 = p[i - 1], p[i]
            if y1 == y0:
                return float(steps[i]), last
            xc = x0 + (level - y0) * (x1 - x0) / (y1 - y0)
            return float(10 ** xc), last
    return math.nan, False


def _t_star(steps: np.ndarray, rate: np.ndarray, tau: float) -> float:
    """Earliest checkpoint whose velocity rate, and every later one, is < tau."""
    finite = np.isfinite(rate)
    for i in range(len(rate)):
        if not finite[i]:
            continue
        later = rate[i:][finite[i:]]
        if later.size and np.all(later < tau):
            return float(steps[i])
    return math.nan


def _vel_rate(steps: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Fractional displacement per 1000 optimizer steps."""
    out = np.full_like(vel, np.nan, dtype=float)
    for i in range(1, len(steps)):
        dx = (float(steps[i]) - float(steps[i - 1])) / 1000.0
        if dx > 0 and np.isfinite(vel[i]):
            out[i] = vel[i] / dx
    return out


# --------------------------------------------------------------------------- #
# Per-matrix table
# --------------------------------------------------------------------------- #

def stabilization_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["metric"].isin(("cos_to_final", "velocity"))]
    recs = []
    keys = ["model", "layer", "matrix", "group", "role"]
    for (model, layer, matrix, group, role), g in df.groupby(keys, sort=False):
        piv = g.pivot_table(index="step", columns="metric", values="value").sort_index()
        steps = piv.index.to_numpy()
        cos = piv["cos_to_final"].to_numpy(dtype=float)
        vel = piv["velocity"].to_numpy(dtype=float)

        c0 = cos[0]
        p = np.clip((cos - c0) / (1.0 - c0 + 1e-12), 0.0, None)
        rate = _vel_rate(steps, vel)

        rec = dict(model=model, layer=int(layer), matrix=matrix,
                   group=group, role=role,
                   cos_init=float(c0),
                   cos_pre_final=float(cos[-2]) if len(cos) > 1 else math.nan,
                   p_pre_final=float(p[-2]) if len(p) > 1 else math.nan)
        for lvl, name in zip(config.PROGRESS_LEVELS, ("t50", "t90", "t99")):
            t, last = _cross(steps, p, lvl)
            rec[name] = t
            rec[f"{name}_last_interval"] = bool(last)
        for tau in config.TAU_SWEEP:
            rec[f"t_star@tau={tau}"] = _t_star(steps, rate, tau)
        recs.append(rec)

    return pd.DataFrame(recs).sort_values(["model", "layer", "matrix"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Data integrity checks -- run before anything is believed
# --------------------------------------------------------------------------- #

def audit(kin: pd.DataFrame, stab: pd.DataFrame, model: str) -> str:
    spec = config.MODELS[model]
    L = spec["n_layer"]
    out = [f"Data audit -- {model}", "=" * 72]
    ok = True

    steps = sorted(kin["step"].unique())
    out.append(f"  checkpoints: {len(steps)}  {steps}")
    if steps != sorted(config.CHECKPOINT_STEPS):
        out.append("  !! checkpoint set differs from config.CHECKPOINT_STEPS"); ok = False

    n_mat = kin.groupby("step")["matrix"].count() / len(kin["metric"].unique())
    expect = L * 6 + 2
    out.append(f"  tracked matrices per checkpoint: {sorted(set(n_mat.astype(int)))}"
               f"  (expected {expect})")
    if set(n_mat.astype(int)) != {expect}:
        out.append("  !! a matrix is missing at some checkpoint"); ok = False

    n_null = int(kin["value"].isna().sum())
    exp_null = expect  # velocity is NaN at the first checkpoint only
    out.append(f"  NaN values: {n_null} (expected {exp_null}: velocity at step 0)")
    if n_null != exp_null:
        out.append("  !! unexpected NaNs"); ok = False

    cos = kin[kin["metric"] == "cos_to_final"]
    at_final = cos[cos["step"] == config.FINAL_STEP]["value"]
    out.append(f"  cos_to_final at step {config.FINAL_STEP}: "
               f"min={at_final.min():.8f} max={at_final.max():.8f}  (must be 1)")
    if not np.allclose(at_final, 1.0, atol=1e-6):
        out.append("  !! reference checkpoint is not self-identical"); ok = False

    at0 = cos[cos["step"] == 0]["value"]
    out.append(f"  cos_to_final at step 0: min={at0.min():+.4f} "
               f"median={at0.median():+.4f} max={at0.max():+.4f}")
    if at0.max() > 0.9:
        out.append("  !! step 0 already aligned with the final state"); ok = False

    # Byte-identical adjacent checkpoints show up as exactly-zero velocity.
    velp = (kin[kin["metric"] == "velocity"]
            .pivot_table(index="step", columns=["layer", "matrix"], values="value")
            .sort_index())
    dup = []
    all_steps = sorted(kin["step"].unique())
    for s in velp.index:
        row = velp.loc[s].to_numpy(dtype=float)
        if np.isfinite(row).all() and np.all(row == 0.0):
            i = all_steps.index(s)
            if i > 0:
                dup.append((int(all_steps[i - 1]), int(s)))
    out.append(f"  byte-identical adjacent revisions (velocity exactly 0): "
               f"{dup or 'none'}")
    unexpected = [d for d in dup if list(d) not in
                  [list(x) for x in config.KNOWN_DUPLICATE_STEPS]]
    if unexpected:
        out.append(f"  !! unexpected duplicate checkpoints {unexpected}"); ok = False
    elif dup:
        out.append("     (expected: Pythia's step0 and step1 tags both point at "
                   "the initial state -- documented in config.KNOWN_DUPLICATE_STEPS)")

    # q/k/v must be genuinely distinct logical matrices
    l0 = cos[(cos["layer"] == 0) & (cos["step"] == 1000)]
    qkv = {r.matrix: r.value for r in l0.itertuples() if r.matrix in ("q", "k", "v")}
    out.append(f"  layer-0 cos at step1000 q/k/v: " +
               "  ".join(f"{k}={v:+.5f}" for k, v in sorted(qkv.items())))
    if len(set(round(v, 8) for v in qkv.values())) != 3:
        out.append("  !! q/k/v are not distinct -- the fused split is wrong"); ok = False

    vel = kin[kin["metric"] == "velocity"].dropna(subset=["value"])
    out.append(f"  velocity range: [{vel['value'].min():.3e}, {vel['value'].max():.3e}]")
    rate_rows = []
    for (lay, mat), g in kin[kin["metric"] == "velocity"].groupby(["layer", "matrix"]):
        g = g.sort_values("step")
        r = _vel_rate(g["step"].to_numpy(), g["value"].to_numpy(dtype=float))
        rate_rows.append(pd.DataFrame({"step": g["step"].to_numpy(), "rate": r}))
    rates = pd.concat(rate_rows).dropna()
    bystep = rates.groupby("step")["rate"].median()
    peak_step, peak = int(bystep.idxmax()), float(bystep.max())
    tail = float(rates[rates["step"] >= 96000]["rate"].median())
    out.append(f"  displacement rate per 1000 steps: peaks at step {peak_step} "
               f"(median {peak:.4f}), tail median (step>=96000) = {tail:.4f}, "
               f"decay factor {peak / tail:.0f}x")
    if not peak > tail:
        out.append("  !! displacement rate does not decay"); ok = False
    n_tstar = {c: int(stab[c].notna().sum())
               for c in stab.columns if c.startswith("t_star@")}
    out.append(f"  t_star defined (out of {len(stab)}) per tau: {n_tstar}")

    fro = kin[kin["metric"] == "fro"]
    out.append(f"  Frobenius norm range: [{fro['value'].min():.2f}, "
               f"{fro['value'].max():.2f}]")

    n_nan90 = int(stab["t90"].isna().sum())
    out.append(f"  t90: {len(stab) - n_nan90}/{len(stab)} matrices reach p_norm=0.9; "
               f"{int(stab['t90_last_interval'].sum())} of those cross only inside "
               f"the final checkpoint interval (definitional -- see module docstring)")
    out.append(f"  t99: {int((~stab['t99'].isna()).sum())}/{len(stab)} reach 0.99; "
               f"{int(stab['t99_last_interval'].sum())} only in the final interval")
    for c in ("t50", "t90", "t99"):
        v = stab[c].dropna()
        out.append(f"  {c}: min={v.min():.1f}  median={v.median():.1f}  max={v.max():.1f}")
        if (v < 0).any() or (v > config.FINAL_STEP * 1.001).any():
            out.append(f"  !! {c} out of range"); ok = False

    out.append("")
    out.append("  AUDIT PASSED" if ok else "  AUDIT FAILED -- see !! lines above")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not math.isfinite(f) else f
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def run_one(model: str, parquet=None) -> pd.DataFrame:
    pq = config.kinematics_parquet(model) if parquet is None else parquet
    kin = pd.read_parquet(pq)

    stab = stabilization_table(kin)
    csv_path = config.stabilization_csv(model)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stab.to_csv(csv_path, index=False)
    print(f"[stab] wrote {len(stab)} rows -> {csv_path}\n")

    audit_txt = audit(kin, stab, model)
    print(audit_txt)
    (config.RESULTS_DIR / f"audit_{config._slug(model)}.txt").write_text(audit_txt + "\n")

    texts, results = [], {}
    for col in config.SCALARS:
        tag = "PRIMARY" if col == config.PRIMARY_SCALAR else "robustness check"
        rep = hyp.run_all(stab, kin, model, config.FINAL_STEP, col=col)
        header = f"\n{'#' * 72}\n# Scalar = {col}   ({tag})\n{'#' * 72}\n"
        texts.append(header + rep.text)
        results[col] = rep.results
    txt = "\n".join(texts)
    print("\n" + txt)
    slug = config._slug(model)
    (config.RESULTS_DIR / f"hypotheses_{slug}.txt").write_text(txt + "\n")
    (config.RESULTS_DIR / f"hypotheses_{slug}.json").write_text(
        json.dumps(_to_jsonable(results), indent=2) + "\n")
    print(f"\n[stab] wrote results/hypotheses_{slug}.txt/.json")
    return stab


def run_h5(models: list[str]) -> None:
    tables = {}
    for m in models:
        p = config.stabilization_csv(m)
        if p.exists():
            tables[m] = pd.read_csv(p)
    if len(tables) < 2:
        print(f"[h5] need >=2 stabilization tables, found {len(tables)} -- skipping")
        return
    texts, res = [], {}
    for col in config.SCALARS:
        t, r = hyp.h5_scale(tables, col=col)
        tag = "PRIMARY" if col == config.PRIMARY_SCALAR else "robustness check"
        texts.append(f"\n{'#' * 72}\n# Scalar = {col}   ({tag})\n{'#' * 72}\n" + t)
        res[col] = r
    txt = "\n".join(texts)
    print("\n" + txt)
    (config.RESULTS_DIR / "h5_scale.txt").write_text(txt + "\n")
    (config.RESULTS_DIR / "h5_scale.json").write_text(
        json.dumps(_to_jsonable(res), indent=2) + "\n")
    print("\n[h5] wrote results/h5_scale.txt/.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=config.PRIMARY_MODEL, choices=list(config.MODELS))
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--h5", action="store_true",
                    help="run only the cross-model H5 test over every model with a "
                         "stabilization csv")
    args = ap.parse_args()

    if args.h5:
        run_h5(list(config.MODELS))
        return

    run_one(args.model, args.parquet)
    run_h5(list(config.MODELS))


if __name__ == "__main__":
    main()
