"""Phase 1, step 1: walk the Pythia checkpoint series and record weight kinematics.

For each checkpoint `step` and each tracked matrix we log three metrics vs the
final state (step 143000) and vs the previous checkpoint:

    cos_to_final  = cos(vec(W_t), vec(W_final))          rises 0 -> ~1
    velocity      = ||W_t - W_prev||_F / ||W_t||_F       decays -> 0   (NaN at step 0)
    fro           = ||W_t||_F

Output: long-format parquet with columns
    model, step, layer, matrix, group, role, metric, value

Runtime is dominated by the HF download of each revision (~0.3-0.8 GB for
160m). Downloads are cached under .hf_cache/ so re-runs are fast.

    .venv/bin/python extract_kinematics.py --model EleutherAI/pythia-160m
    .venv/bin/python extract_kinematics.py --model EleutherAI/pythia-70m --steps 0,1,2,4,8
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import kinematics as kin


def _mid(m: kin.Matrix) -> tuple[str, int]:
    return (m.key, m.layer)


def extract(model: str, steps: list[int], out_path) -> pd.DataFrame:
    if config.FINAL_STEP not in steps:
        raise SystemExit(
            f"--steps must include the final step {config.FINAL_STEP} "
            f"(it is the W_final reference)."
        )
    steps = sorted(set(steps))

    print(f"[extract] {model}: {len(steps)} checkpoints -> {out_path}")

    # W_final once, kept for the whole run (fp32 to halve memory; upcast per matrix).
    t0 = time.time()
    sd_final = kin.load_state_dict(model, config.FINAL_STEP)
    w_final = {_mid(m): m.vec.astype(np.float32) for m in kin.iter_matrices(sd_final, model)}
    del sd_final
    gc.collect()
    print(f"[extract] W_final: {len(w_final)} matrices  ({time.time() - t0:.0f}s)")

    prev: dict[tuple[str, int], np.ndarray] = {}
    rows: list[dict] = []

    for step in steps:
        t0 = time.time()
        sd = kin.load_state_dict(model, step)
        n = 0
        for m in kin.iter_matrices(sd, model):
            mid = _mid(m)
            wt = m.vec  # float64
            wf = w_final[mid].astype(np.float64)

            vals = {
                "cos_to_final": kin.cos_to_final(wt, wf),
                "fro": kin.fro(wt),
                "velocity": (kin.velocity(wt, prev[mid].astype(np.float64))
                             if mid in prev else np.nan),
            }
            for metric, value in vals.items():
                rows.append(dict(
                    model=model, step=step, layer=m.layer, matrix=m.key,
                    group=m.group, role=m.role, metric=metric, value=value,
                ))
            prev[mid] = wt.astype(np.float32)
            n += 1

        del sd
        gc.collect()
        print(f"[extract] step{step:<7} {n} matrices  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df = df.astype({"step": "int64", "layer": "int64"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[extract] wrote {len(df):,} rows -> {out_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=config.PRIMARY_MODEL, choices=list(config.MODELS))
    ap.add_argument("--steps", default=None,
                    help="comma-separated step list (default: config.CHECKPOINT_STEPS)")
    ap.add_argument("--out", default=None, help="output parquet path")
    args = ap.parse_args()

    steps = ([int(s) for s in args.steps.split(",")] if args.steps
             else list(config.CHECKPOINT_STEPS))
    out_path = config.kinematics_parquet(args.model) if args.out is None else Path(args.out)

    extract(args.model, steps, out_path)


if __name__ == "__main__":
    main()
