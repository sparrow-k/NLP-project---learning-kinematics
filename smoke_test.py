"""Phase 0 smoke test.

Loads pythia-70m at step0 and step143000, prints the state-dict layout, and
checks every tracked matrix name in config.py actually resolves. Also verifies
the GPT-NeoX QKV split reproduces the fused tensor.

Run: .venv/bin/python smoke_test.py
"""

from __future__ import annotations

import config  # sets KMP_DUPLICATE_LIB_OK before torch import

import torch
from transformers import AutoModelForCausalLM

MODEL = "EleutherAI/pythia-70m"
SPEC = config.MODELS[MODEL]


def load_state_dict(revision: str) -> dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=revision,
        cache_dir=str(config.HF_CACHE),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    del model
    return sd


def check_names(sd: dict[str, torch.Tensor]) -> list[str]:
    missing = []
    for layer in range(SPEC["n_layer"]):
        for key, (tmpl, _grp, _role) in config.LAYER_MATRICES.items():
            name = tmpl.format(i=layer)
            if name not in sd:
                missing.append(name)
    for key, (name, _grp, _role) in config.GLOBAL_MATRICES.items():
        if name not in sd:
            missing.append(name)
    return missing


def check_qkv_split(sd: dict[str, torch.Tensor]) -> None:
    d = SPEC["d_model"]
    n_head = SPEC["n_head"]
    head_dim = d // n_head
    w = sd["gpt_neox.layers.0.attention.query_key_value.weight"]
    assert w.shape == (3 * d, d), w.shape
    wv = w.view(n_head, 3, head_dim, d)
    w_q, w_k, w_v = wv[:, 0], wv[:, 1], wv[:, 2]
    # round-trips back to the original fused tensor
    recon = torch.stack([w_q, w_k, w_v], dim=1).reshape(3 * d, d)
    assert torch.equal(recon, w), "QKV view/reshape is not a clean round-trip"
    print(f"  QKV split OK: fused {tuple(w.shape)} -> "
          f"q/k/v each {tuple(w_q.shape)} (n_head={n_head}, head_dim={head_dim})")


def main() -> None:
    print(f"Model: {MODEL}  {SPEC}")
    print(f"HF cache: {config.HF_CACHE}")

    sd0 = load_state_dict(config.revision(0))
    sdF = load_state_dict(config.revision(config.FINAL_STEP))

    print(f"\nstep0 state dict: {len(sd0)} tensors")
    for name, t in list(sd0.items())[:12]:
        print(f"  {name:55} {tuple(t.shape)}")
    print("  ...")

    missing = check_names(sd0)
    if missing:
        print(f"\nMISSING tracked names ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
        raise SystemExit(1)
    n_tracked = len(config.LAYER_MATRICES) * SPEC["n_layer"] + len(config.GLOBAL_MATRICES)
    print(f"\nAll {n_tracked} tracked matrix names resolve in the state dict.")

    print("\nQKV split check:")
    check_qkv_split(sd0)

    # sanity: weights actually moved between step0 and step143000
    w0 = sd0["gpt_neox.layers.0.attention.dense.weight"]
    wF = sdF["gpt_neox.layers.0.attention.dense.weight"]
    cos = torch.nn.functional.cosine_similarity(
        w0.flatten(), wF.flatten(), dim=0
    ).item()
    print(f"\nlayer0 attn.dense  cos(step0, step143000) = {cos:.4f} "
          f"(expect well below 1.0)")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
