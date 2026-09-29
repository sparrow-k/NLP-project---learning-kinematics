"""Shared Phase 1 helpers: checkpoint loading, tracked-matrix extraction, metrics.

Everything here is CPU-only linear algebra over pretrained Pythia checkpoints.
`extract_kinematics.py` is the only entry point that touches the network (HF
downloads); the derived-quantity math lives in `compute_stabilization.py`.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Iterator

import config  # sets KMP_DUPLICATE_LIB_OK before torch import

import numpy as np
import torch
from transformers import AutoModelForCausalLM

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #

def load_state_dict(model: str, step: int) -> dict[str, torch.Tensor]:
    """Load one Pythia checkpoint's weights as a plain {name: fp32 tensor} dict.

    The nn.Module is dropped before returning so the caller can gc between steps.
    """
    m = AutoModelForCausalLM.from_pretrained(
        model,
        revision=config.revision(step),
        cache_dir=str(config.HF_CACHE),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    sd = {k: v.detach().to(torch.float32).clone() for k, v in m.state_dict().items()}
    del m
    gc.collect()
    return sd


# --------------------------------------------------------------------------- #
# Tracked-matrix extraction
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Matrix:
    """One tracked weight matrix at one checkpoint, flattened to a vector."""

    key: str          # q / k / v / attn_out / ffn_up / ffn_down / embed_in / embed_out
    group: str        # attn / ffn / embed
    role: str         # routing / memory / context
    layer: int        # 0..L-1 for per-layer matrices, -1 for globals
    vec: np.ndarray   # float64, 1-D


def _qkv_slices(w: torch.Tensor, n_head: int) -> dict[str, torch.Tensor]:
    """Split the fused GPT-NeoX query_key_value weight (3d, d) into q/k/v.

    Rows are grouped per head as [h0_q, h0_k, h0_v, h1_q, ...], NOT as three
    contiguous blocks. See config.py QKV note.
    """
    three_d, d = w.shape
    assert three_d == 3 * d, w.shape
    head_dim = d // n_head
    wv = w.view(n_head, 3, head_dim, d)
    return {"q": wv[:, 0], "k": wv[:, 1], "v": wv[:, 2]}


def iter_matrices(sd: dict[str, torch.Tensor], model: str) -> Iterator[Matrix]:
    """Yield every tracked Matrix present in this state dict."""
    spec = config.MODELS[model]
    n_layer, n_head = spec["n_layer"], spec["n_head"]

    for layer in range(n_layer):
        # fused QKV -> three logical matrices
        qkv_name = f"gpt_neox.layers.{layer}.attention.query_key_value.weight"
        if qkv_name in sd:
            slices = _qkv_slices(sd[qkv_name], n_head)
            for key in ("q", "k", "v"):
                _, grp, role = config.LAYER_MATRICES[key]
                yield Matrix(key, grp, role, layer,
                             _to_vec(slices[key].reshape(-1)))

        for key in ("attn_out", "ffn_up", "ffn_down"):
            tmpl, grp, role = config.LAYER_MATRICES[key]
            name = tmpl.format(i=layer)
            if name in sd:
                yield Matrix(key, grp, role, layer, _to_vec(sd[name].reshape(-1)))

    for key, (name, grp, role) in config.GLOBAL_MATRICES.items():
        if name in sd:
            yield Matrix(key, grp, role, -1, _to_vec(sd[name].reshape(-1)))


def _to_vec(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().to(torch.float64).cpu().numpy().ravel()


# --------------------------------------------------------------------------- #
# Per-step metrics (weight space)
# --------------------------------------------------------------------------- #

def cos_to_final(w_t: np.ndarray, w_final: np.ndarray) -> float:
    """cos( vec(W_t), vec(W_final) ) -- directional convergence, rises 0 -> ~1."""
    denom = np.linalg.norm(w_t) * np.linalg.norm(w_final)
    return float(np.dot(w_t, w_final) / (denom + EPS))


def velocity(w_t: np.ndarray, w_prev: np.ndarray) -> float:
    """||W_t - W_prev||_F / ||W_t||_F -- step-to-step displacement, decays -> 0."""
    return float(np.linalg.norm(w_t - w_prev) / (np.linalg.norm(w_t) + EPS))


def fro(w_t: np.ndarray) -> float:
    """||W_t||_F."""
    return float(np.linalg.norm(w_t))


METRICS = ("cos_to_final", "velocity", "fro")
