"""Phase 2: train a small GPT-NeoX from scratch under one intervention arm.

Everything except the per-group learning-rate multiplier is held fixed across
arms: architecture, initialization, data order, batch size, sequence length,
token budget, optimizer family, base learning rate, global LR shape, evaluation
points and evaluation data. For a given `--seed`, the model is initialized from
the same RNG state and consumes the same sequence of batches in every arm, so a
difference between arms cannot come from a different initialization or a
different data order.

Outputs, per run, under `runs/<run_id>/`:
    config.json     every hyperparameter, the schedule, and the environment
    log.jsonl       one record per logged step (train loss, LR state, timing)
    evals.jsonl     one record per evaluation (val loss, val ppl, wall-clock)
    summary.json    final metrics + peak VRAM + steps-to-threshold inputs
    weight_probe.json   per-group weight displacement, used to verify that the
                    intervention actually changed which parameters moved

    .venv/Scripts/python train.py --arm baseline --seed 0
    .venv/Scripts/python train.py --arm attn-first --seed 0 --max-steps 4000
    .venv/Scripts/python train.py --smoke        # ~1 minute, wikitext-2
    .venv/Scripts/python train.py --overfit      # sanity: memorize 8 batches
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config  # sets KMP_DUPLICATE_LIB_OK before torch

import numpy as np
import torch
from torch.utils.data import DataLoader

import data_prep
import schedules

RUNS_DIR = config.ROOT / "runs"


# --------------------------------------------------------------------------- #
# Run configuration
# --------------------------------------------------------------------------- #

@dataclass
class RunConfig:
    arm: str = "baseline"
    seed: int = 0

    # architecture (pythia-70m scale, trained from scratch)
    n_layer: int = 6
    d_model: int = 512
    n_head: int = 8
    d_ff: int = 2048
    vocab_size: int = 50304
    seq_len: int = 512
    rotary_pct: float = 0.25

    # optimization
    max_steps: int = 4000
    batch_size: int = 16          # sequences per micro-batch, tuned for a 16 GB card
    grad_accum: int = 2           # -> tokens/step = batch*accum*seq_len
    base_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup: int = 200
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    # schedule
    emphasis_ratio: float = 3.0
    alt_period: int = 200
    phase1_model: str = "EleutherAI/pythia-70m"
    phase1_scalar: str = "t50"

    # data / eval
    dataset: str = "wikitext-103-raw-v1"
    eval_every: int = 200
    eval_batches: int = 24
    log_every: int = 20

    # runtime
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False
    run_id: str = ""
    tag: str = ""

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum * self.seq_len


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class PackedBlocks(torch.utils.data.Dataset):
    """Non-overlapping `seq_len + 1` windows over a packed uint16 token stream."""

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        s = i * self.seq_len
        chunk = np.asarray(self.tokens[s:s + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def make_loaders(cfg: RunConfig):
    train_tok = data_prep.load_split(cfg.dataset, "train")
    val_tok = data_prep.load_split(cfg.dataset, "validation")
    train_ds = PackedBlocks(train_tok, cfg.seq_len)
    val_ds = PackedBlocks(val_tok, cfg.seq_len)

    # Data order depends on the seed only -- never on the arm.
    g = torch.Generator().manual_seed(cfg.seed + 12345)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              generator=g, drop_last=True, num_workers=0,
                              pin_memory=True)
    # Validation is a fixed, unshuffled prefix: identical batches for every run.
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            drop_last=True, num_workers=0, pin_memory=True)
    return train_loader, val_loader, len(train_ds), len(val_ds)


def infinite(loader):
    while True:
        for b in loader:
            yield b


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(cfg: RunConfig):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    mc = GPTNeoXConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        num_hidden_layers=cfg.n_layer,
        num_attention_heads=cfg.n_head,
        intermediate_size=cfg.d_ff,
        max_position_embeddings=cfg.seq_len,
        rotary_pct=cfg.rotary_pct,
        use_parallel_residual=True,
        layer_norm_eps=1e-5,
        use_cache=False,
    )
    model = GPTNeoXForCausalLM(mc)
    return model, mc


def param_groups(model, cfg: RunConfig):
    """One optimizer group per (schedule-group, layer), plus decay/no-decay split.

    Splitting by layer is what lets the Phase 1 schedule be layer-aware. Biases
    and LayerNorm parameters get no weight decay, as usual.
    """
    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = schedules.group_of(name)
        l = schedules.layer_of(name)
        decay = p.dim() >= 2
        key = (g, l, decay)
        groups.setdefault(key, {"params": [], "sched_group": g, "layer": l,
                                "weight_decay": cfg.weight_decay if decay else 0.0,
                                "names": []})
        groups[key]["params"].append(p)
        groups[key]["names"].append(name)
    return [v for _, v in sorted(groups.items(), key=lambda kv: str(kv[0]))]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model, val_loader, cfg: RunConfig, device, amp_dtype) -> dict:
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for i, (x, y) in enumerate(val_loader):
        if i >= cfg.eval_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(input_ids=x).logits
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1),
            reduction="sum")
        tot_loss += loss.item()
        tot_tok += y.numel()
    model.train()
    mean = tot_loss / max(tot_tok, 1)
    return {"val_loss": mean, "val_ppl": float(math.exp(min(mean, 20.0))),
            "val_tokens": tot_tok}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(cfg: RunConfig, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": None}[cfg.dtype]
    if device.type == "cpu":
        amp_dtype = None

    # --- initialization: seeded before the model is built, so every arm at a
    # --- given seed starts from bit-identical weights.
    seed_everything(cfg.seed)
    model, mc = build_model(cfg)
    init_sig = float(sum(p.detach().double().sum().item()
                         for p in model.parameters()))
    init_snapshot = {n: p.detach().clone().float().cpu()
                     for n, p in model.named_parameters()}
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    # transformers 5.x renamed the output embedding to lm_head (see config.py)
    head = getattr(model, "lm_head", None) or getattr(model, "embed_out")
    n_emb = model.gpt_neox.embed_in.weight.numel() + head.weight.numel()
    sched = schedules.build(cfg.arm, cfg.max_steps,
                            cfg.phase1_model if cfg.arm in schedules.PHASE1_ARMS
                            else None,
                            ratio=cfg.emphasis_ratio, alt_period=cfg.alt_period,
                            scalar=cfg.phase1_scalar, warmup=cfg.warmup,
                            min_ratio=cfg.min_lr_ratio)

    pgs = param_groups(model, cfg)
    opt = torch.optim.AdamW(pgs, lr=cfg.base_lr, betas=tuple(cfg.betas),
                            weight_decay=cfg.weight_decay, eps=1e-8)

    train_loader, val_loader, n_train_blocks, n_val_blocks = make_loaders(cfg)
    batches = infinite(train_loader)

    meta = {
        "run_config": asdict(cfg),
        "tokens_per_step": cfg.tokens_per_step,
        "total_tokens": cfg.tokens_per_step * cfg.max_steps,
        "n_params": n_params,
        "n_params_non_embedding": n_params - n_emb,
        "n_train_blocks": n_train_blocks,
        "n_val_blocks": n_val_blocks,
        "epochs_over_train": cfg.tokens_per_step * cfg.max_steps
                             / (n_train_blocks * cfg.seq_len),
        "model_config": mc.to_dict(),
        "schedule": sched.describe(),
        "schedule_budget": sched.budget_check(),
        "optimizer_groups": [{"sched_group": g["sched_group"], "layer": g["layer"],
                              "weight_decay": g["weight_decay"],
                              "n_params": sum(p.numel() for p in g["params"]),
                              "names": g["names"]} for g in pgs],
        "init_signature": init_sig,
        "env": env_info(),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    log_f = (out_dir / "log.jsonl").open("w")
    eval_f = (out_dir / "evals.jsonl").open("w")
    evals: list[dict] = []
    t_start = time.perf_counter()
    t_eval_total = 0.0
    frozen_steps = {}

    model.train()
    for step in range(cfg.max_steps):
        lr_global = schedules.global_lr(step, cfg.base_lr, cfg.max_steps,
                                        cfg.warmup, cfg.min_lr_ratio)
        # per-group LR = global shape x arm multiplier
        for g in opt.param_groups:
            m = sched.multiplier(step, g["sched_group"], g["layer"])
            g["lr"] = lr_global * m
            g["_mult"] = m
            need_grad = m != 0.0
            if cfg.arm == "attn-freeze":
                for p in g["params"]:
                    if p.requires_grad != need_grad:
                        p.requires_grad_(need_grad)
                if not need_grad:
                    frozen_steps[f"{g['sched_group']}_L{g['layer']}"] = \
                        frozen_steps.get(f"{g['sched_group']}_L{g['layer']}", 0) + 1

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(cfg.grad_accum):
            x, y = next(batches)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                logits = model(input_ids=x).logits
            loss = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.size(-1)), y.reshape(-1))
            (loss / cfg.grad_accum).backward()
            loss_acc += loss.item() / cfg.grad_accum

        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            rec = {"step": step, "train_loss": loss_acc,
                   "lr_global": lr_global, "grad_norm": float(gnorm),
                   "tokens": (step + 1) * cfg.tokens_per_step,
                   "wall_s": time.perf_counter() - t_start - t_eval_total,
                   "mult": {f"{g['sched_group']}_L{g['layer']}": g["_mult"]
                            for g in opt.param_groups if g["sched_group"] != "other"}}
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()

        if (step + 1) % cfg.eval_every == 0 or step == cfg.max_steps - 1:
            t0 = time.perf_counter()
            ev = evaluate(model, val_loader, cfg, device, amp_dtype)
            t_eval_total += time.perf_counter() - t0
            ev.update(step=step + 1, train_loss=loss_acc,
                      tokens=(step + 1) * cfg.tokens_per_step,
                      wall_s=time.perf_counter() - t_start - t_eval_total)
            if device.type == "cuda":
                ev["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            evals.append(ev)
            eval_f.write(json.dumps(ev) + "\n"); eval_f.flush()
            print(f"[{cfg.arm} s{cfg.seed}] step {step + 1:>5}/{cfg.max_steps}  "
                  f"train {loss_acc:.3f}  val {ev['val_loss']:.4f}  "
                  f"ppl {ev['val_ppl']:.2f}  {ev['wall_s']:.0f}s", flush=True)

    log_f.close(); eval_f.close()
    train_wall = time.perf_counter() - t_start - t_eval_total

    # --- verification probe: did the intervention actually move different
    # --- parameters? Total relative displacement per schedule group.
    probe: dict[str, dict] = {}
    for name, p in model.named_parameters():
        g, l = schedules.group_of(name), schedules.layer_of(name)
        key = f"{g}_L{l}"
        w0 = init_snapshot[name]
        d = (p.detach().float().cpu() - w0)
        e = probe.setdefault(key, {"sq_disp": 0.0, "sq_init": 0.0, "n": 0})
        e["sq_disp"] += float((d ** 2).sum())
        e["sq_init"] += float((w0 ** 2).sum())
        e["n"] += p.numel()
    for k, e in probe.items():
        e["rel_displacement"] = math.sqrt(e["sq_disp"]) / (math.sqrt(e["sq_init"]) + 1e-12)
    by_group = {}
    for k, e in probe.items():
        g = k.split("_L")[0]
        a = by_group.setdefault(g, {"sq_disp": 0.0, "sq_init": 0.0})
        a["sq_disp"] += e["sq_disp"]; a["sq_init"] += e["sq_init"]
    for g, a in by_group.items():
        a["rel_displacement"] = math.sqrt(a["sq_disp"]) / (math.sqrt(a["sq_init"]) + 1e-12)
    (out_dir / "weight_probe.json").write_text(
        json.dumps({"per_group_layer": probe, "per_group": by_group,
                    "frozen_steps": frozen_steps}, indent=2) + "\n")

    summary = {
        "run_id": cfg.run_id, "arm": cfg.arm, "seed": cfg.seed,
        "max_steps": cfg.max_steps, "tokens_per_step": cfg.tokens_per_step,
        "total_tokens": cfg.tokens_per_step * cfg.max_steps,
        "final_val_loss": evals[-1]["val_loss"] if evals else None,
        "final_val_ppl": evals[-1]["val_ppl"] if evals else None,
        "best_val_ppl": min((e["val_ppl"] for e in evals), default=None),
        "train_wall_s": train_wall,
        "eval_wall_s": t_eval_total,
        "total_wall_s": time.perf_counter() - t_start,
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 2 ** 30
                         if device.type == "cuda" else None),
        "n_params": n_params,
        "init_signature": init_sig,
        "weight_displacement_by_group": {g: a["rel_displacement"]
                                         for g, a in by_group.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{cfg.arm} s{cfg.seed}] done: final ppl "
          f"{summary['final_val_ppl']:.2f}  train {train_wall:.0f}s  "
          f"peak VRAM {summary['peak_vram_gb']:.2f} GB" if summary["peak_vram_gb"]
          else f"[{cfg.arm} s{cfg.seed}] done", flush=True)
    return summary


# --------------------------------------------------------------------------- #
# Sanity experiments (the professor asks for these before the big runs)
# --------------------------------------------------------------------------- #

def overfit_check(cfg: RunConfig, n_batches: int = 8, steps: int = 800) -> dict:
    """Can the model memorize a handful of batches? If not, there is a bug.

    Success criterion: train loss on the fixed subset falls below 0.5 (from
    ~10.8 at random init, i.e. ln(50304)).
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    seed_everything(cfg.seed)
    model, _ = build_model(cfg)
    model.to(device).train()

    tok = data_prep.load_split(cfg.dataset, "train")
    ds = PackedBlocks(tok, cfg.seq_len)
    fixed = [ds[i] for i in range(n_batches * cfg.batch_size)]
    xs = torch.stack([f[0] for f in fixed]).to(device)
    ys = torch.stack([f[1] for f in fixed]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    hist = []
    for step in range(steps):
        i = (step % n_batches) * cfg.batch_size
        x, y = xs[i:i + cfg.batch_size], ys[i:i + cfg.batch_size]
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(input_ids=x).logits
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 50 == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(loss)})
            print(f"  overfit step {step:>4}  loss {float(loss):.4f}", flush=True)

    final = hist[-1]["loss"]
    ok = final < 0.5
    res = {"n_batches": n_batches, "steps": steps, "final_loss": final,
           "random_init_loss": math.log(cfg.vocab_size), "passed": ok,
           "history": hist}
    RUNS_DIR.mkdir(exist_ok=True)
    # One file per dataset: demo.sh runs this on wikitext-2 and must not
    # overwrite the wikitext-103 result the paper reports.
    res["dataset"] = cfg.dataset
    (RUNS_DIR / f"sanity_overfit_{cfg.dataset}.json").write_text(
        json.dumps(res, indent=2) + "\n")
    print(f"\n  overfit sanity: final loss {final:.4f} "
          f"(random init {res['random_init_loss']:.2f}) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return res


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    d = RunConfig()
    ap.add_argument("--arm", default=d.arm, choices=schedules.ARMS)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--max-steps", type=int, default=d.max_steps)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--grad-accum", type=int, default=d.grad_accum)
    ap.add_argument("--seq-len", type=int, default=d.seq_len)
    ap.add_argument("--base-lr", type=float, default=d.base_lr)
    ap.add_argument("--warmup", type=int, default=d.warmup)
    ap.add_argument("--emphasis-ratio", type=float, default=d.emphasis_ratio)
    ap.add_argument("--alt-period", type=int, default=d.alt_period)
    ap.add_argument("--dataset", default=d.dataset)
    ap.add_argument("--eval-every", type=int, default=d.eval_every)
    ap.add_argument("--eval-batches", type=int, default=d.eval_batches)
    ap.add_argument("--phase1-model", default=d.phase1_model)
    ap.add_argument("--phase1-scalar", default=d.phase1_scalar)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None, help="explicit run directory")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run on wikitext-2 to prove the path works")
    ap.add_argument("--overfit", action="store_true",
                    help="sanity check: memorize a few batches")
    args = ap.parse_args()

    cfg = RunConfig(
        arm=args.arm, seed=args.seed, max_steps=args.max_steps,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        seq_len=args.seq_len, base_lr=args.base_lr, warmup=args.warmup,
        emphasis_ratio=args.emphasis_ratio, alt_period=args.alt_period,
        dataset=args.dataset,
        eval_every=args.eval_every, eval_batches=args.eval_batches,
        phase1_model=args.phase1_model, phase1_scalar=args.phase1_scalar,
        tag=args.tag)

    if args.smoke:
        cfg.dataset = "wikitext-2-raw-v1"
        cfg.max_steps = 60
        cfg.warmup = 10
        cfg.eval_every = 30
        cfg.eval_batches = 8
        cfg.tag = cfg.tag or "smoke"

    if args.overfit:
        overfit_check(cfg)
        return

    cfg.run_id = (f"{cfg.tag + '_' if cfg.tag else ''}{cfg.arm}_seed{cfg.seed}"
                  f"_s{cfg.max_steps}")
    # `--out` is always a path, taken as given. It used to be treated as a path
    # only when it contained os.path.sep, which on Windows is a backslash, so a
    # forward-slash path fell through and was appended to runs/ a second time.
    out_dir = Path(args.out) if args.out else RUNS_DIR / cfg.run_id
    if args.out:
        cfg.run_id = out_dir.name
    train(cfg, out_dir)


if __name__ == "__main__":
    main()
