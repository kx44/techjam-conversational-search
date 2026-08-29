"""Dense retrieval with BAAI/bge-small-en-v1.5 under onnxruntime.

The ONNX graph exposes only ``last_hidden_state``, so pooling and
normalisation happen here. BGE pools the **CLS token** - not the mean - and
then L2-normalises (``1_Pooling/config.json`` sets ``pooling_mode_cls_token``,
``modules.json`` appends a Normalize module). Mean pooling this model yields
vectors that look reasonable and rank badly, so the choice is explicit.

Nothing in this module imports at agent start-up unless the artifacts exist;
callers are expected to treat a failure as "no dense retrieval" rather than an
error, because the evaluator turns an agent exception into an empty response.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSION = 384
MAX_TOKENS = 512
# BGE v1.5 asks for an instruction on the query side only; passages go in bare.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder:
    """Tokenise, run the transformer, take CLS, normalise."""

    def __init__(self, model_dir: str | Path, max_tokens: int = MAX_TOKENS) -> None:
        directory = Path(model_dir)
        self.max_tokens = max_tokens
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._accepts = {value.name for value in self.session.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [text or " " for text in texts[start:start + batch_size]]
            encodings = self.tokenizer.encode_batch(batch)
            feed = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
            }
            feed = {name: value for name, value in feed.items() if name in self._accepts}
            hidden = self.session.run(None, feed)[0]
            cls = hidden[:, 0, :]                                   # CLS pooling
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            vectors.append((cls / np.maximum(norms, 1e-12)).astype(np.float32))
        if not vectors:
            return np.zeros((0, DIMENSION), dtype=np.float32)
        return np.vstack(vectors)

    def encode_query(self, text: str, prefix: bool = True) -> np.ndarray:
        return self.encode([QUERY_PREFIX + text if prefix else text])[0]


class DenseIndex:
    """Brute-force cosine search over the precomputed catalog matrix.

    50k x 384 float32 is ~77 MB and one matrix-vector product per query, so
    this needs no vector database - which also keeps it inside the
    specification's exclusion of infrastructure-heavy vector stores.
    """

    def __init__(self, matrix: np.ndarray, ids: list[str], meta: dict) -> None:
        self.matrix = matrix
        self.ids = ids
        self.meta = meta

    @classmethod
    def load(cls, prefix: str | Path) -> "DenseIndex":
        prefix = Path(prefix)
        matrix = np.load(prefix.with_suffix(".npy"), mmap_mode="r")
        payload = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
        ids = payload["ids"]
        if len(ids) != matrix.shape[0]:
            raise ValueError(f"embedding matrix has {matrix.shape[0]} rows for {len(ids)} ids")
        return cls(matrix, ids, payload.get("meta", {}))

    def search(self, vector: np.ndarray, limit: int) -> list[str]:
        scores = self.matrix @ vector
        limit = min(limit, scores.shape[0])
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [self.ids[i] for i in top]
