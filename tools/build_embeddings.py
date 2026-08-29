"""Precompute BGE embeddings for the frozen catalog.

The evaluator constructs ``Agent(catalog_path)`` fresh for every run, so
embedding 50k products at start-up is not viable. This writes the matrix once;
the agent memory-maps it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starter.agent import _text
from starter.dense import DIMENSION, MODEL_NAME, Encoder

CHAR_BUDGET = 1600      # roughly the 512-token window, before tokenizer truncation


def passage(product: dict, recipe: str) -> str:
    """Product text to embed, most identifying field first."""
    parts = [_text(product.get("title"))]
    if recipe != "title":
        parts.append(_text(product.get("categories")))
        features = product.get("features") or []
        parts.extend(str(value) for value in features[:5])
        parts.append(_text(product.get("details")))
    text = ". ".join(part.strip() for part in parts if part and part.strip())
    return text[:CHAR_BUDGET]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dense catalog index")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--model", default="models/bge-small-en-v1.5")
    parser.add_argument("--out", default="data/bge_embeddings")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recipe", default="full", choices=("full", "title"))
    args = parser.parse_args()

    ids: list[str] = []
    passages: list[str] = []
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            ids.append(str(product["parent_asin"]))
            passages.append(passage(product, args.recipe))
    print(f"{len(ids)} products; median passage {sorted(len(p) for p in passages)[len(ids)//2]} chars")

    encoder = Encoder(args.model, max_tokens=args.max_tokens)
    started = time.time()
    chunks: list[np.ndarray] = []
    for start in range(0, len(passages), 2048):
        chunks.append(encoder.encode(passages[start:start + 2048], batch_size=args.batch_size))
        done = min(start + 2048, len(passages))
        rate = done / (time.time() - started)
        print(f"  {done}/{len(passages)}  {rate:.0f}/s  eta {(len(passages)-done)/rate/60:.1f} min", flush=True)
    matrix = np.vstack(chunks) if chunks else np.zeros((0, DIMENSION), dtype=np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".npy"), matrix)
    out.with_suffix(".json").write_text(json.dumps({
        "ids": ids,
        "meta": {"model": MODEL_NAME, "max_tokens": args.max_tokens,
                 "recipe": args.recipe, "count": len(ids), "dimension": int(matrix.shape[1])},
    }), encoding="utf-8")
    size = out.with_suffix(".npy").stat().st_size / 1e6
    print(f"wrote {out.with_suffix('.npy')} {matrix.shape} {size:.0f} MB "
          f"in {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
