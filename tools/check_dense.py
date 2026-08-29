"""Correctness checks for the dense encoder.

Ranking scores cannot tell a correct encoder from a subtly wrong one, so
check the embedding itself before trusting any measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starter.dense import DenseIndex, Encoder

MODEL = "models/bge-small-en-v1.5"


def main() -> None:
    encoder = Encoder(MODEL)
    probes = ["a black leather belt", "shoes with a grippy rubber bottom", "silver pendant necklace"]
    vectors = encoder.encode(probes)

    norms = np.linalg.norm(vectors, axis=1)
    print(f"1. L2 norms          : {np.round(norms, 6).tolist()}  "
          f"{'PASS' if np.allclose(norms, 1.0, atol=1e-4) else 'FAIL'}")

    # CLS must differ from mean pooling, or we silently pooled the wrong way.
    encodings = encoder.tokenizer.encode_batch(probes)
    feed = {
        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
    }
    hidden = encoder.session.run(None, feed)[0]
    mask = feed["attention_mask"][..., None]
    mean = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1)
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    agreement = float(np.mean(np.sum(vectors * mean, axis=1)))
    print(f"2. CLS vs mean cosine: {agreement:.4f}  "
          f"{'PASS (distinct)' if agreement < 0.99 else 'FAIL (identical - pooling may be wrong)'}")

    prefix = Path("data/bge_embeddings")
    if not prefix.with_suffix(".npy").exists():
        print("3-4. skipped: build data/bge_embeddings first")
        return
    index = DenseIndex.load(prefix)
    print(f"   index: {index.matrix.shape} {index.meta}")
    catalog = {}
    with open("data/catalog.jsonl", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            catalog[str(item["parent_asin"])] = item

    seed = index.ids[0]
    print(f"\n3. neighbours of: {catalog[seed]['title'][:64]}")
    for pid in index.search(np.asarray(index.matrix[0]), 4)[1:]:
        print(f"     {catalog[pid]['title'][:64]}")

    print("\n4. paraphrase queries (no literal term overlap with the catalog wording)")
    for query in ["shoes with a grippy rubber bottom",
                  "something warm to wear hiking in winter",
                  "a delicate silver chain for my mum"]:
        hits = index.search(encoder.encode_query(query), 3)
        print(f"   {query!r}")
        for pid in hits:
            print(f"     -> {catalog[pid]['title'][:62]}")


if __name__ == "__main__":
    main()
