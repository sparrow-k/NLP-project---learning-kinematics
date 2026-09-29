"""Phase 2, step 0: tokenize and pack WikiText-103 into flat uint16 token arrays.

WikiText-103 ships with official train / validation / test splits, so the
train/val separation is the corpus's own -- no re-splitting and no leakage. The
test split is tokenized but never touched by any experiment in this project.

Tokenizer: the GPT-NeoX tokenizer shipped with Pythia (vocab 50277, padded to
50304 in the model config). Documents are joined with the EOS token and packed
into a single contiguous stream, which is then read as non-overlapping blocks
of `seq_len` tokens. Packing (rather than padding per document) keeps every
token in every batch a real training token, so the token budget quoted in the
report is the true budget.

    .venv/Scripts/python data_prep.py
    .venv/Scripts/python data_prep.py --dataset wikitext-2-raw-v1   # fast smoke path
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import config

TOKENIZER = "EleutherAI/pythia-70m"
HF_DATASET = "Salesforce/wikitext"
SPLITS = {"train": "train", "validation": "validation", "test": "test"}


def token_path(dataset: str, split: str):
    return config.DATA_DIR / f"{dataset}_{split}_uint16.npy"


def meta_path(dataset: str):
    return config.DATA_DIR / f"{dataset}_meta.json"


def prepare(dataset: str = "wikitext-103-raw-v1", batch_size: int = 1000) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, cache_dir=str(config.HF_CACHE))
    eos = tok.eos_token_id
    meta = {"dataset": dataset, "tokenizer": TOKENIZER, "eos_token_id": int(eos),
            "vocab_size": int(tok.vocab_size), "splits": {}}

    for split, hf_split in SPLITS.items():
        out = token_path(dataset, split)
        if out.exists():
            arr = np.load(out, mmap_mode="r")
            print(f"[data] {split}: cached, {len(arr):,} tokens -> {out.name}")
            meta["splits"][split] = {"n_tokens": int(len(arr)), "path": out.name}
            continue

        t0 = time.time()
        ds = load_dataset(HF_DATASET, dataset, split=hf_split,
                          cache_dir=str(config.HF_CACHE))
        chunks: list[np.ndarray] = []
        n_docs = 0
        buf: list[str] = []
        for i, text in enumerate(ds["text"]):
            if not text.strip():
                continue
            buf.append(text)
            n_docs += 1
            if len(buf) >= batch_size:
                ids = tok(buf, add_special_tokens=False)["input_ids"]
                chunks.append(np.concatenate(
                    [np.asarray(x + [eos], dtype=np.uint16) for x in ids]))
                buf = []
        if buf:
            ids = tok(buf, add_special_tokens=False)["input_ids"]
            chunks.append(np.concatenate(
                [np.asarray(x + [eos], dtype=np.uint16) for x in ids]))

        arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
        assert arr.max() < 2 ** 16, "token id overflows uint16"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, arr)
        print(f"[data] {split}: {n_docs:,} non-empty lines -> {len(arr):,} tokens "
              f"({time.time() - t0:.0f}s) -> {out.name}")
        meta["splits"][split] = {"n_tokens": int(len(arr)), "n_lines": n_docs,
                                 "path": out.name}

    meta_path(dataset).write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[data] wrote {meta_path(dataset).name}")
    return meta


def load_split(dataset: str, split: str) -> np.ndarray:
    p = token_path(dataset, split)
    if not p.exists():
        raise SystemExit(f"missing {p} -- run `python data_prep.py --dataset {dataset}`")
    return np.load(p, mmap_mode="r")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="wikitext-103-raw-v1",
                    choices=["wikitext-103-raw-v1", "wikitext-2-raw-v1"])
    args = ap.parse_args()
    prepare(args.dataset)


if __name__ == "__main__":
    main()
