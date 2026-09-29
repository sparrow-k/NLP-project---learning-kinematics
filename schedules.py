"""Phase 2: per-parameter-group learning-rate schedules derived from Phase 1.

How Phase 1 becomes a Phase 2 schedule
--------------------------------------
Phase 1 measures, for every (layer, matrix) of a *pretrained* Pythia model, the
step at which that matrix has covered half of its directional distance to its
final state (`t50`, see compute_stabilization.py). Those steps live on Pythia's
143 000-step axis; a Phase 2 run is much shorter. They are transferred by
**relative training progress**:

    sigma(layer, group) = t50(layer, group) / 143000        in [0, 1]
    switch_step(layer, group) = round(sigma * max_steps)

so a matrix that covered half its distance 20 % of the way through Pythia's run
switches 20 % of the way through ours. The schedule is therefore *layer-aware*:
layer 0's attention switches earlier than layer 5's if Phase 1 measured it that
way. Nothing is collapsed to a single global step.

The Phase 1 table is read from `results/stabilization_<model>.csv` for the
**same model size** that Phase 2 trains, so the timing is internally consistent
(70m Phase 1 -> 70m Phase 2). No hard-coded step numbers appear anywhere.

Budget control
--------------
Every arm uses the identical global LR schedule (linear warmup + cosine decay).
An arm only supplies a per-group *multiplier* m_g(t, layer), and that multiplier
is normalized so that **each group's time-averaged multiplier is exactly 1** --
the same as the baseline's. Concretely, a group that is emphasised on a fraction
sigma of training takes the two-level profile

    m = A  for t < sigma * T,      m = B  for t >= sigma * T
    A / B = ratio  (the emphasis ratio),   A*sigma + B*(1-sigma) = 1

which solves to  B = 1 / (ratio*sigma + 1 - sigma),  A = ratio * B. A group that
is emphasised *late* uses the mirrored profile with the same normalization.
`attn-first` and `attn-second` are exact mirror images of each other. In both,
*both* groups switch at attention's measured per-layer point; the FFN's own
Phase 1 time is computed but not used.

CAVEAT (found in the pre-submission audit): the normalization above equalizes
the *step-averaged multiplier*, not the integrated learning rate. The global
LR is not flat (warmup, then cosine decay), so a group emphasised early --
where the global LR is highest -- receives more total learning rate than the
baseline. On the 70m schedule the early-emphasised group gets +9.1% of the
baseline's integrated LR (+3% to +17.5% per layer) and the late-emphasised
group -4.6%. `attn-first` / `attn-second` therefore differ in per-group LR
*mass* as well as timing. The `-lrm` arms fix this: they solve for A, B
against the actual global-LR integral,

    A * G_early + B * G_late = G_early + G_late,     A / B = ratio,

where G_early / G_late are the sums of the global LR before / after the switch
step, so each group's integrated LR is exactly the baseline's. The original
arms are kept unchanged so that their recorded runs stay reproducible.

`attn-freeze` is deliberately not budget-matched: it sets the attention
multiplier to 0 after the measured switch point. It exists to test the freezing
claim (does removing updates hurt quality, and does it save wall-clock time?),
and its unequal budget is stated in the report rather than hidden.

Arms
----
  baseline      m = 1 for every group at every step. The control.
  attn-first    attention emphasised before its own per-layer switch point,
                de-emphasised after; FFN exactly mirrored.
  attn-second   the mirror of attn-first (FFN emphasised first).
  attn-first-lrm / attn-second-lrm
                the same two profiles, normalized on integrated LR (see the
                caveat above) instead of on the step-averaged multiplier.
  alternating  emphasis swaps between attention and FFN every `alt_period`
                steps; no Phase 1 input (a timing-agnostic control for
                "decoupling per se").
  attn-freeze   attention LR -> 0 after its per-layer switch point.

Groups
------
  attn   = every `attention.*` weight and bias in a block
  ffn    = every `mlp.*` weight and bias in a block
  other  = embeddings, LM head, all LayerNorms, final LN. Never modulated by any
           arm, so the arms differ only in the attention/FFN contrast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

import config

ARMS = ["baseline", "attn-first", "attn-second", "alternating", "attn-freeze",
        "attn-first-lrm", "attn-second-lrm"]
DIRECTED_ARMS = ("attn-first", "attn-second", "attn-first-lrm", "attn-second-lrm")
PHASE1_ARMS = DIRECTED_ARMS + ("attn-freeze",)
PHASE1_TOTAL_STEPS = config.FINAL_STEP  # the axis Phase 1's t50 lives on

ATTN_MATS = ["q", "k", "v", "attn_out"]
FFN_MATS = ["ffn_up", "ffn_down"]


# --------------------------------------------------------------------------- #
# Parameter grouping
# --------------------------------------------------------------------------- #

def group_of(param_name: str) -> str:
    """Map a GPT-NeoX parameter name to attn / ffn / other."""
    if ".attention." in param_name:
        return "attn"
    if ".mlp." in param_name:
        return "ffn"
    return "other"


def layer_of(param_name: str) -> int:
    """Block index for a per-layer parameter, else -1."""
    parts = param_name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1


# --------------------------------------------------------------------------- #
# Phase 1 -> switch points
# --------------------------------------------------------------------------- #

def switch_fractions(model: str, scalar: str | None = None) -> dict:
    """Read Phase 1 and return {group: {layer: sigma in [0,1]}} plus provenance.

    sigma is the fraction of training at which that (layer, group) had covered
    half of its directional distance in the Pythia run.
    """
    scalar = scalar or config.PRIMARY_SCALAR
    csv = config.stabilization_csv(model)
    if not csv.exists():
        raise SystemExit(
            f"missing {csv}. Run Phase 1 first:\n"
            f"  python extract_kinematics.py    --model {model}\n"
            f"  python compute_stabilization.py --model {model}")
    df = pd.read_csv(csv)
    df = df[df["layer"] >= 0]

    out: dict[str, dict[int, float]] = {"attn": {}, "ffn": {}}
    for grp, mats in (("attn", ATTN_MATS), ("ffn", FFN_MATS)):
        sub = df[df["matrix"].isin(mats)]
        for layer, g in sub.groupby("layer"):
            v = g[scalar].dropna()
            if len(v) == 0:
                continue
            out[grp][int(layer)] = float(min(max(v.median() / PHASE1_TOTAL_STEPS,
                                                 0.0), 1.0))
    prov = {"source_csv": str(csv), "phase1_model": model, "scalar": scalar,
            "phase1_total_steps": PHASE1_TOTAL_STEPS,
            "sigma": {g: {str(k): v for k, v in d.items()} for g, d in out.items()}}
    return {"sigma": out, "provenance": prov}


# --------------------------------------------------------------------------- #
# The schedule object
# --------------------------------------------------------------------------- #

@dataclass
class Schedule:
    """Produces a per-(group, layer) LR multiplier for a given step."""

    arm: str
    max_steps: int
    sigma: dict = field(default_factory=dict)   # {group: {layer: fraction}}
    ratio: float = 3.0                           # emphasis ratio A / B
    alt_period: int = 200
    default_sigma: float = 0.3                   # only if Phase 1 lacks a layer
    provenance: dict = field(default_factory=dict)
    lr_shape: list = field(default_factory=list)  # global LR per step (-lrm arms)

    def switch_fraction(self, group: str, layer: int) -> float:
        return float(self.sigma.get(group, {}).get(layer, self.default_sigma))

    def switch_step(self, group: str, layer: int) -> int:
        return int(round(self.switch_fraction(group, layer) * self.max_steps))

    def _levels(self, sigma: float) -> tuple[float, float]:
        """(A, B) for a profile that is high on the first `sigma` of training
        and whose time-average is exactly 1."""
        sigma = min(max(sigma, 1e-6), 1 - 1e-6)
        b = 1.0 / (self.ratio * sigma + (1.0 - sigma))
        return self.ratio * b, b

    def _lr_levels(self, switch: int) -> tuple[float, float, float, float]:
        """LR-mass-matched levels for a two-level profile switching at `switch`.

        Returns (hi_early, lo_early, hi_late, lo_late): the early-emphasis
        profile is hi_early before the switch and lo_early after; the
        late-emphasis profile is lo_late before and hi_late after. Both
        integrate, against the global LR shape, to exactly the baseline's LR
        mass.
        """
        cache = self.__dict__.setdefault("_lr_level_cache", {})
        if switch in cache:
            return cache[switch]
        if not self.lr_shape:
            raise ValueError(f"arm {self.arm!r} needs the global LR shape")
        g_early = float(sum(self.lr_shape[:switch]))
        g_late = float(sum(self.lr_shape[switch:]))
        tot = g_early + g_late
        lo_early = tot / (self.ratio * g_early + g_late)
        lo_late = tot / (g_early + self.ratio * g_late)
        cache[switch] = (self.ratio * lo_early, lo_early,
                         self.ratio * lo_late, lo_late)
        return cache[switch]

    def multiplier(self, step: int, group: str, layer: int) -> float:
        if group == "other" or self.arm == "baseline":
            return 1.0

        if self.arm == "alternating":
            # symmetric 50/50 emphasis; time-average is 1 for both groups
            a, b = self._levels(0.5)
            attn_hot = ((step // self.alt_period) % 2) == 0
            return (a if attn_hot else b) if group == "attn" else \
                   (b if attn_hot else a)

        if self.arm == "attn-freeze":
            # deliberately NOT budget-matched -- this arm tests the freezing
            # claim itself, and the unequal budget is reported as such.
            if group != "attn":
                return 1.0
            return 1.0 if step < self.switch_step("attn", layer) else 0.0

        # attn-first / attn-second (+ -lrm variants): mirrored profiles. Both
        # groups switch at *attention's* measured per-layer point; the FFN's own
        # Phase 1 time is not used by these arms.
        sigma = self.switch_fraction("attn", layer)
        switch = self.switch_step("attn", layer)
        early = step < switch
        if self.arm.endswith("-lrm"):
            hi_early, lo_early, hi_late, lo_late = self._lr_levels(switch)
        else:
            # normalized on the step-averaged multiplier (see module caveat)
            # early-emphasis profile: hi on [0, sigma), lo on [sigma, 1)
            hi_early, lo_early = self._levels(sigma)
            # late-emphasis profile: lo on [0, sigma), hi on [sigma, 1)
            hi_late, lo_late = self._levels(1.0 - sigma)
        early_val = hi_early if early else lo_early
        late_val = lo_late if early else hi_late
        attn_is_early = self.arm.startswith("attn-first")
        if group == "attn":
            return early_val if attn_is_early else late_val
        if group == "ffn":
            return late_val if attn_is_early else early_val
        return 1.0

    def frozen(self, step: int, group: str, layer: int) -> bool:
        return self.multiplier(step, group, layer) == 0.0

    def describe(self) -> dict:
        d = {"arm": self.arm, "max_steps": self.max_steps, "ratio": self.ratio,
             "alt_period": self.alt_period, "default_sigma": self.default_sigma,
             "provenance": self.provenance}
        if self.arm in PHASE1_ARMS:
            d["switch_steps"] = {
                f"layer{l}": self.switch_step("attn", l)
                for l in sorted(self.sigma.get("attn", {}))}
        return d

    def budget_check(self) -> dict:
        """Per-group budget relative to the baseline, averaged over layers.

        `attn` / `ffn`: step-averaged multiplier (what the original arms match).
        `attn_lr` / `ffn_lr`: integrated LR, i.e. the multiplier weighted by the
        global LR shape -- the quantity that actually sets how far a group can
        move. Both are exactly 1 for the baseline.
        """
        layers = sorted(set(self.sigma.get("attn", {})) | set(self.sigma.get("ffn", {})))
        layers = layers or [0]
        g = self.lr_shape or [1.0] * self.max_steps
        g_tot = float(sum(g))
        tot = {}
        for grp in ("attn", "ffn"):
            vals, lr_vals = [], []
            for l in layers:
                m = [self.multiplier(t, grp, l) for t in range(self.max_steps)]
                vals.append(sum(m) / self.max_steps)
                lr_vals.append(sum(a * b for a, b in zip(m, g)) / g_tot)
            tot[grp] = sum(vals) / len(vals)
            tot[f"{grp}_lr"] = sum(lr_vals) / len(lr_vals)
        tot["sum"] = tot["attn"] + tot["ffn"]
        return tot


def build(arm: str, max_steps: int, phase1_model: str | None,
          ratio: float = 3.0, alt_period: int = 200,
          scalar: str | None = None, warmup: int = 200,
          min_ratio: float = 0.1) -> Schedule:
    """`warmup` / `min_ratio` must match the trainer's global LR shape; they
    only matter for the -lrm arms and for the LR-weighted budget check."""
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; choose from {ARMS}")
    sigma, prov = {}, {}
    if arm in PHASE1_ARMS:
        got = switch_fractions(phase1_model, scalar)
        sigma, prov = got["sigma"], got["provenance"]
    # base_lr cancels out of every normalization, so the shape uses 1.0
    shape = [global_lr(t, 1.0, max_steps, warmup, min_ratio)
             for t in range(max_steps)]
    return Schedule(arm=arm, max_steps=max_steps, sigma=sigma, ratio=ratio,
                    alt_period=alt_period, provenance=prov, lr_shape=shape)


# --------------------------------------------------------------------------- #
# Global LR shape (identical for every arm)
# --------------------------------------------------------------------------- #

def global_lr(step: int, base_lr: float, max_steps: int, warmup: int,
              min_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay to `min_ratio * base_lr`."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(max_steps - warmup, 1)
    prog = min(max(prog, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cos)


if __name__ == "__main__":  # quick inspection
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.PRIMARY_MODEL)
    ap.add_argument("--max-steps", type=int, default=4000)
    args = ap.parse_args()
    for arm in ARMS:
        try:
            s = build(arm, args.max_steps, args.model)  # trainer defaults
        except SystemExit as e:
            print(f"{arm}: {e}"); continue
        print(f"\n=== {arm} ===")
        print(json.dumps(s.describe(), indent=2, default=str))
        print("integrated multiplier:", {k: round(v, 4)
                                         for k, v in s.budget_check().items()})
