#!/usr/bin/env python3
"""Extract tokenizer vocab + input embedding matrix from local HF-format models.

Only reads the safetensors shard that contains ``embed_tokens.weight`` (usually
the first shard); never loads the full model. Output layout is:

    <out_root>/<model_name>/
        vocab.json                     token -> id (utf-8)
        special_tokens_meta.json       [{id, token, special}]
        special_tokens_vectors_f32.npy special token rows only
        embed_tokens_f32.npy           full input matrix, float32
        embed_tokens_bf16_bits.npy     full input matrix, bf16 bit-cast uint16
        manifest.json                  provenance/shape/dtype summary
        tokenizer.json etc.            copies of tokenizer/config files
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import safetensors
import torch
from tokenizers import Tokenizer


EMBED_RE = re.compile(r"(?:^|\.)embed_tokens\.weight$")


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_embedding_shard(model_dir: Path) -> tuple[str, str]:
    index = load_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map", {})
    key = next((k for k in weight_map if EMBED_RE.search(k)), None)
    if key is None:
        raise RuntimeError(f"no embed_tokens.weight in {model_dir}/model.safetensors.index.json")
    return key, weight_map[key]


def extract_model(model_dir: Path, out_dir: Path, save_f32: bool) -> dict:
    model_dir = model_dir.resolve()
    model_name = model_dir.parent.parent.name  # <org>--<name> -> keep short name part
    short_name = model_name.split("--", 1)[-1]
    out_dir = out_dir / short_name
    out_dir.mkdir(parents=True, exist_ok=True)

    key, shard = find_embedding_shard(model_dir)
    shard_path = model_dir / shard
    print(f"[{short_name}] reading {shard} for {key}")

    with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
        emb = f.get_tensor(key)
    src_dtype = str(emb.dtype).replace("torch.", "")
    emb32 = emb.to(torch.float32)
    if emb.dtype == torch.bfloat16 or emb.dtype == torch.float16:
        bits = emb.view(torch.uint16)
    else:
        bits = emb.to(torch.bfloat16).view(torch.uint16)

    np.save(out_dir / "embed_tokens_f32.npy", emb32.numpy()) if save_f32 else None
    np.save(out_dir / "embed_tokens_bf16_bits.npy", bits.numpy())

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    vocab = tok.get_vocab()
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False)

    added = tok.get_added_tokens_decoder()
    special = [
        {"id": i, "token": str(at), "special": bool(at.special)}
        for i, at in sorted(added.items())
    ]
    with open(out_dir / "special_tokens_meta.json", "w", encoding="utf-8") as fh:
        json.dump(special, fh, ensure_ascii=False, indent=2)

    if special:
        ids = np.asarray([x["id"] for x in special], dtype=np.int64)
        np.save(out_dir / "special_tokens_vectors_f32.npy", emb32[ids].numpy())

    cfg = load_json(model_dir / "config.json") if (model_dir / "config.json").exists() else {}
    index = load_json(model_dir / "model.safetensors.index.json")
    manifest = {
        "model_dir": str(model_dir),
        "model_name": model_name,
        "embedding_key": key,
        "embedding_shape": list(emb.shape),
        "source_dtype": src_dtype,
        "vocab_size_from_tokenizer": len(vocab),
        "num_special_tokens": len(special),
        "tie_word_embeddings": cfg.get("tie_word_embeddings"),
        "has_lm_head": any("lm_head.weight" in k for k in index.get("weight_map", {})),
        "config_vocab_size": cfg.get("vocab_size") or (cfg.get("text_config") or {}).get("vocab_size"),
        "config_hidden_size": cfg.get("hidden_size") or (cfg.get("text_config") or {}).get("hidden_size"),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "config.json",
        "added_tokens.json",
        "special_tokens_map.json",
    ):
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    total_gb = sum(
        p.stat().st_size for p in out_dir.glob("*.npy") if p.exists()
    ) / 1e9
    print(f"[{short_name}] done: {emb.shape[0]} tokens x {emb.shape[1]} dim, "
          f"dtype {src_dtype}, {len(special)} special tokens, npy {total_gb:.2f} GB")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--no-f32", action="store_true", help="skip float32 copy to save space")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for d in args.model_dirs:
        extract_model(Path(d), out_root, save_f32=not args.no_f32)
    print("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
