"""Precompute the intent prototype embeddings (tiny: ~36 short sentences)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starter.dense import Encoder
from starter.intent import CLASSES, PROTOTYPES, _fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description="Build intent prototype embeddings")
    parser.add_argument("--model", default="models/bge-small-en-v1.5")
    parser.add_argument("--out", default="data/intent_prototypes")
    args = parser.parse_args()

    sentences, labels = [], []
    for name in CLASSES:
        for sentence in PROTOTYPES[name]:
            sentences.append(sentence)
            labels.append(name)
    matrix = Encoder(args.model).encode(sentences)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".npy"), matrix)
    out.with_suffix(".json").write_text(
        json.dumps({"labels": labels, "prototypes": _fingerprint()}), encoding="utf-8")
    print(f"wrote {out.with_suffix('.npy')} {matrix.shape} for {len(set(labels))} classes")


if __name__ == "__main__":
    main()
