"""Central configuration for the Kinematics-of-Learning project.

Phases 0-1 (this file's defaults) run on CPU: they only load pretrained Pythia
checkpoints and do linear algebra. Phase 2 training config lives in schedules.py
and train.py.
"""

from __future__ import annotations

import os
from pathlib import Path

# macOS/Homebrew: torch's bundled OpenMP collides with numpy's. Must be set
# before torch is imported anywhere in the process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

# Primary model for Phase 1. Secondary models are run with the same code by
# passing --model on the CLI.
PRIMARY_MODEL = "EleutherAI/pythia-160m"

MODELS = {
    "EleutherAI/pythia-70m": dict(n_layer=6, d_model=512, n_head=8),
    "EleutherAI/pythia-160m": dict(n_layer=12, d_model=768, n_head=12),
    "EleutherAI/pythia-410m": dict(n_layer=24, d_model=1024, n_head=16),
}

# --------------------------------------------------------------------------- #
# Checkpoints (HF `revision` tags)
# --------------------------------------------------------------------------- #

# 29-point grid. The original plan used a 19-point log grid; a first pass on
# pythia-70m showed that cos_to_final is still climbing at step 64000 (median
# ~0.86), so the directional-progress crossings t_90/t_99 all landed inside the
# single enormous 64000 -> 143000 interval and were interpolation artefacts
# rather than measurements. The 1000-143000 decade is therefore densified:
# Pythia stores a checkpoint every 1000 steps in that range, so this costs only
# extra downloads, not extra method.
CHECKPOINT_STEPS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000,
    32000, 48000, 64000, 80000, 96000, 112000, 128000, 136000, 143000,
]
FINAL_STEP = 143000  # W_final reference state

# Pythia's `step0` and `step1` revisions are byte-identical for these models
# (the step-1 tag points at the initial state), so velocity(step1) is exactly 0.
# This is a property of the released checkpoints, not of this code; the audit
# in compute_stabilization.py reports it rather than treating it as an error.
KNOWN_DUPLICATE_STEPS = [(0, 1)]


def revision(step: int) -> str:
    return f"step{step}"


# --------------------------------------------------------------------------- #
# Tracked matrices
# --------------------------------------------------------------------------- #

# Per-layer tensors, formatted with the layer index `i`. `.weight` only; biases
# are 1-D and excluded from the Frobenius/cosine metrics.
LAYER_MATRICES = {
    # key            : (state_dict name template,                          group, role)
    "q":        ("gpt_neox.layers.{i}.attention.query_key_value.weight",  "attn", "routing"),
    "k":        ("gpt_neox.layers.{i}.attention.query_key_value.weight",  "attn", "routing"),
    "v":        ("gpt_neox.layers.{i}.attention.query_key_value.weight",  "attn", "routing"),
    "attn_out": ("gpt_neox.layers.{i}.attention.dense.weight",            "attn", "routing"),
    "ffn_up":   ("gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight",          "ffn",  "memory"),
    "ffn_down": ("gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight",          "ffn",  "memory"),
}

# Non-layer tensors (context group).
GLOBAL_MATRICES = {
    "embed_in":  ("gpt_neox.embed_in.weight",  "embed", "context"),
    "embed_out": ("lm_head.weight",            "embed", "context"),  # was embed_out.* in older transformers
}

# IMPORTANT (Phase 1): the fused query_key_value weight is (3*d, d) but GPT-NeoX
# does NOT store it as [all-Q; all-K; all-V]. Rows are grouped per head as
# [h0_q, h0_k, h0_v, h1_q, h1_k, h1_v, ...]. To pull the Q/K/V slices:
#
#   W = sd["...query_key_value.weight"]              # (3*d, d)
#   W = W.view(n_head, 3, d // n_head, d)            # (n_head, {q,k,v}, head_dim, d)
#   W_q, W_k, W_v = W[:, 0], W[:, 1], W[:, 2]        # each (n_head, head_dim, d)
#
# then flatten for vec()/Frobenius. A naive dim-0 3-way split is wrong here.
QKV_SPLIT_ORDER = ("q", "k", "v")  # order within each head block

# --------------------------------------------------------------------------- #
# Stabilization thresholds
# --------------------------------------------------------------------------- #

# Thresholds on the per-decade displacement rate (see compute_stabilization.py).
# The first three are the conventional "small" thresholds from the original
# plan; the larger ones are included because the measured rate on Pythia never
# falls anywhere near 0.02, and reporting the whole sweep (including the values
# at which t_star is simply undefined) is more honest than picking one.
TAU_SWEEP = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]

PROGRESS_LEVELS = [0.5, 0.9, 0.99]     # p_norm crossings -> t_50, t_90, t_99

# t_50 is the primary stabilization scalar. p_norm reaches 1 at the final
# checkpoint by construction (it is the reference), so crossings close to 1 are
# partly definitional; t_50 sits in the densely sampled, fully measured part of
# the series. Every hypothesis is reported for both t_50 and t_90.
PRIMARY_SCALAR = "t50"
SCALARS = ["t50", "t90"]

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figs"
RESULTS_DIR = ROOT / "results"
HF_CACHE = ROOT / ".hf_cache"  # keep downloaded checkpoints out of ~/.cache

for _d in (DATA_DIR, FIG_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)


def kinematics_parquet(model: str) -> Path:
    return DATA_DIR / f"kinematics_{_slug(model)}.parquet"


def stabilization_csv(model: str) -> Path:
    return RESULTS_DIR / f"stabilization_{_slug(model)}.csv"


def _slug(model: str) -> str:
    return model.split("/")[-1].replace("-", "")
