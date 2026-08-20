#!/usr/bin/env python3
"""General MetaBreak-style analysis for an extracted model directory.

Reads the artifacts produced by ``extract_input_embeddings.py`` (or the older
``qwen35_embed``/``qwen25_embed`` layouts) and computes, for every special
token in the tokenizer:

- L2 nearest regular-token substitutes, scored as the paper does:
      sim% = 100 / (1 + d),  d = ||v_special - v_regular||_2
- cosine top-5 (direction-only) for comparison
- structure-preserving pair search (triple) for the assistant/system headers

Outputs written next to the inputs:
  metabreak_neighbors.json / .md
  metabreak_triple_search.json / .md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

K_L2 = 20
K_COS = 5
POOL = 500


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_DECODER = {ord(v): k for k, v in bytes_to_unicode().items()}


def decode_token(s: str) -> str:
    bs = bytearray()
    for ch in s:
        o = ord(ch)
        if o in _BYTE_DECODER:
            bs.append(_BYTE_DECODER[o])
        else:
            bs.extend(ch.encode("utf-8"))
    return bs.decode("utf-8", errors="replace")


def load_special(data_dir: Path) -> list[dict]:
    meta = data_dir / "special_tokens_meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    tj = json.loads((data_dir / "tokenizer.json").read_text(encoding="utf-8"))
    return [
        {"id": int(t["id"]), "token": t["content"], "special": bool(t["special"])}
        for t in tj.get("added_tokens", [])
    ]


def load_embeddings(data_dir: Path) -> np.ndarray:
    f32 = data_dir / "embed_tokens_f32.npy"
    if f32.exists():
        return np.load(f32, mmap_mode="r")
    bf = data_dir / "embed_tokens_bf16_bits.npy"
    if bf.exists():
        raw = np.load(bf, mmap_mode="r")
        return (raw.astype(np.uint32) << 16).view(np.float32)
    raise FileNotFoundError(f"no embedding npy in {data_dir}")


def analyze(data_dir: Path, out_dir: Path) -> None:
    data_dir = data_dir.resolve()
    out_dir = out_dir or data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    display_path = data_dir / "id2display.json"
    if display_path.exists():
        id2display = {
            int(k): v
            for k, v in json.loads(display_path.read_text(encoding="utf-8")).items()
        }
        id2tok = None
        vocab = None
    else:
        vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
        id2tok = {int(v): k for k, v in vocab.items()}
        id2display = None
    special = load_special(data_dir)
    special_id_set = {t["id"] for t in special}
    if id2display is not None:
        regular_ids = np.array(
            sorted(i for i in id2display if i not in special_id_set), dtype=np.int64
        )
    else:
        regular_ids = np.array(
            sorted(i for i in id2tok if i not in special_id_set), dtype=np.int64
        )
    n_regular = len(regular_ids)
    special_ids = np.asarray([t["id"] for t in special], dtype=np.int64)

    E = load_embeddings(data_dir)
    print(f"embeddings: {E.shape}, special={len(special)}, regular={n_regular}")
    R = np.asarray(E[regular_ids], dtype=np.float32)
    S = np.asarray(E[special_ids], dtype=np.float32)

    s_norm2 = np.einsum("ij,ij->i", S, S).astype(np.float64)
    r_norm2 = np.einsum("ij,ij->i", R, R).astype(np.float64)
    s_norm = np.sqrt(s_norm2)
    r_norm = np.sqrt(r_norm2)
    print(
        f"norm stats | regular: mean={r_norm.mean():.4f} min={r_norm.min():.6f} "
        f"max={r_norm.max():.4f} | special: mean={s_norm.mean():.4f} "
        f"min={s_norm.min():.6f} max={s_norm.max():.4f}"
    )
    dots = (S.astype(np.float32) @ R.astype(np.float32).T).astype(np.float64)
    dist2 = np.clip(s_norm2[:, None] + r_norm2[None, :] - 2.0 * dots, 0.0, None)
    cos = dots / (s_norm[:, None] * r_norm[None, :] + 1e-12)

    results = []
    md = [
        f"# MetaBreak substitutes for {data_dir.name}",
        "",
        "Distance metric: L2 norm of vector difference. sim% = 100 / (1 + d); d=0 -> 100%, d -> +inf -> 0%.",
        f"Top {K_L2} L2 neighbors shown; cosine top {K_COS} in JSON only.",
        "",
        "| special token | id | norm | top L2 substitutes (token: d, sim) |",
        "|---|---|---|---|",
    ]
    for i, meta in enumerate(special):
        idx = meta["id"]
        order_l2 = np.argsort(dist2[i], kind="stable")[:K_L2]
        order_cos = np.argsort(-cos[i], kind="stable")[:K_COS]
        l2_rows = []
        for j in order_l2:
            j = int(j)
            actual = int(regular_ids[j])
            if id2display is not None:
                text = id2display.get(actual, "<unmapped>")
            else:
                text = decode_token(id2tok.get(actual, "<unmapped>"))
            l2_rows.append(
                {
                    "id": actual,
                    "token": text,
                    "dist": round(float(math.sqrt(dist2[i, j])), 6),
                    "cos": round(float(cos[i, j]), 6),
                    "norm": round(float(r_norm[j]), 6),
                    "sim_pct": round(100.0 / (1.0 + math.sqrt(dist2[i, j])), 6),
                }
            )
        cos_rows = [
            {
                "id": int(regular_ids[int(j)]),
                "token": (
                    id2display.get(int(regular_ids[int(j)]), "<unmapped>")
                    if id2display is not None
                    else decode_token(id2tok.get(int(regular_ids[int(j)]), "<unmapped>"))
                ),
                "dist": round(float(math.sqrt(dist2[i, int(j)])), 6),
                "cos": round(float(cos[i, int(j)]), 6),
                "norm": round(float(r_norm[int(j)]), 6),
            }
            for j in order_cos
        ]
        results.append(
            {
                "token": meta["token"],
                "id": int(idx),
                "is_special_meta": bool(meta["special"]),
                "norm": round(float(s_norm[i]), 6),
                "top_l2": l2_rows,
                "top_cosine": cos_rows,
            }
        )
        md_top = ", ".join(
            f"{r['token']} (d={r['dist']}, sim={r['sim_pct']}%)" for r in l2_rows[:8]
        )
        md.append(f"| {meta['token']} | {idx} | {round(float(s_norm[i]), 6)} | {md_top} |")

    (out_dir / "metabreak_neighbors.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "metabreak_neighbors.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'metabreak_neighbors.json'} and .md")

    tokenizer_path = data_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print("skip triple: no tokenizer.json (tiktoken-style vocab)")
        (out_dir / "metabreak_triple_search.json").write_text(
            json.dumps({"triple": []}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return
    tok = Tokenizer.from_file(str(tokenizer_path))
    special_index = {t["id"]: i for i, t in enumerate(special)}
    triples = []
    for role in ("assistant", "system"):
        text = f"<|im_end|>\n<|im_start|>{role}\n"
        tids = tok.encode(text, add_special_tokens=False).ids
        positions = [p for p, t in enumerate(tids) if t in special_id_set]
        if len(positions) != 2:
            print(f"skip triple {role}: unexpected structure {tids}")
            continue
        pools = []
        for p in positions:
            tid = tids[p]
            row = dist2[special_index[tid]]
            order = np.argsort(row, kind="stable")[:POOL]
            pool = []
            for j in order:
                j = int(j)
                actual = int(regular_ids[j])
                s = tok.decode([actual])
                if tok.encode(s, add_special_tokens=False).ids == [actual]:
                    pool.append((actual, s, float(math.sqrt(row[j]))))
            pools.append(pool)
        pairs = []
        for ia, ta, da in pools[0]:
            for ib, tb, db in pools[1]:
                pairs.append((ia, ta, da, ib, tb, db))
        texts = [ta + "\n" + tb + f"{role}\n" for _, ta, _, _, tb, _ in pairs]
        valid = []
        for start in range(0, len(texts), 20000):
            encs = tok.encode_batch(texts[start : start + 20000], add_special_tokens=False)
            for off, enc in enumerate(encs):
                i = start + off
                ia, ta, da, ib, tb, db = pairs[i]
                expected = list(tids)
                expected[positions[0]] = ia
                expected[positions[1]] = ib
                if enc.ids == expected:
                    valid.append((da + db, ia, ta, da, ib, tb, db))
        valid.sort(key=lambda x: x[0])
        rows = [
            {
                "total_dist": round(float(v[0]), 6),
                "im_end": {"id": v[1], "token": v[2], "dist": round(float(v[3]), 6)},
                "im_start": {"id": v[4], "token": v[5], "dist": round(float(v[6]), 6)},
            }
            for v in valid[:20]
        ]
        triples.append(
            {
                "name": f"{role}_header",
                "target_ids": tids,
                "pairs_tried": len(pairs),
                "pairs_kept": len(valid),
                "top": rows,
            }
        )
        print(f"{role}_header: {len(pairs)} pairs tried, {len(valid)} kept structure")

    (out_dir / "metabreak_triple_search.json").write_text(
        json.dumps({"triple": triples}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_t = ["# MetaBreak triple search", ""]
    for t in triples:
        md_t.append(f"## {t['name']}\n")
        md_t.append(
            f"target ids: {t['target_ids']} | {t['pairs_tried']} pairs tried, "
            f"{t['pairs_kept']} kept structure\n"
        )
        md_t.append("| total L2 | <|im_end|> substitute (dist) | <|im_start|> substitute (dist) |")
        md_t.append("|---|---|---|")
        for r in t["top"][:10]:
            md_t.append(
                f"| {r['total_dist']} | {r['im_end']['token']} ({r['im_end']['dist']}) | "
                f"{r['im_start']['token']} ({r['im_start']['dist']}) |"
            )
        md_t.append("")
    (out_dir / "metabreak_triple_search.md").write_text("\n".join(md_t) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'metabreak_triple_search.json'} and .md")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    analyze(Path(args.model_dir), Path(args.out_dir) if args.out_dir else None)
    return 0


if __name__ == "__main__":
    main()
