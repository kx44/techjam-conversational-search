from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
from pathlib import Path

# Ensure package path resolution works from any execution context
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from starter.stemmer import stem
except ModuleNotFoundError:
    from stemmer import stem


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# Retrieval weights and depth limits
BM25_WEIGHTS = "0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0"
RRF_K = 60
FEEDBACK_DOCS = 10
EXPANSION_TERMS = 10
MIN_FEEDBACK_DOCS = 6
RETRIEVE = 500
DIGITS = re.compile(r"\d")

# Dense embedding model configuration
MODEL_DIR = "models/bge-small-en-v1.5"
EMBEDDINGS = "data/bge_embeddings"
USE_DENSE = False
QUERY_INSTRUCTION = True
DENSE_WEIGHT = 1.0
DENSE_LIMIT = RETRIEVE

# Mistral Generative Reranker Configuration
MISTRAL_MODEL_ID = os.environ.get("MISTRAL_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
USE_MISTRAL_RERANK = os.environ.get("MISTRAL_RERANK", "1").lower() not in {"0", "false", "no", "off"}
MISTRAL_RERANK_DEPTH = 10  # Capped at top 10 for maximum inference speed

USE_OVERRIDE = True

RERANK_DEPTH = 50
RERANK_WEIGHTS = {
    "phrase": 0.8,
    "popularity": 0.2,
    "price": 0.3,
}

PRICE_HINT = re.compile(r"\$\s*(\d+(?:\.\d+)?)|\b(?:under|below|around|about|up to)\s+(\d+(?:\.\d+)?)", re.I)
USE_EXPANSION = False

ATTRIBUTE_ORDER = ("feature", "use_case", "style", "material", "color",
                   "size", "budget", "brand", "category")

PROTOTYPES = "data/intent_prototypes"
ATTRIBUTE_DECAY = 0.55
FRESH_COST = 0.25
DECLINE_PENALTY = 3.0
QUESTIONS = {
    "material": "What material would you prefer?",
    "color": "Any particular colour you have in mind?",
    "budget": "Roughly what budget are you working with?",
    "style": "What sort of style or fit are you after?",
    "size": "What size do you need?",
    "use_case": "What will you mainly be using it for?",
    "feature": "Is there a specific feature it has to have?",
    "brand": "Is there a brand you tend to go for?",
    "category": "What kind of item are we talking about exactly?",
}

# Module-level singleton across session resets
_CACHED_MISTRAL_MODEL = None
_CACHED_MISTRAL_TOKENIZER = None


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _stemmed(text: str) -> list[str]:
    return [stem(token) for token in _terms(text)]


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit - 3].rstrip() + "..."


class Agent:
    """Hybrid search agent combining BM25, RRF fusion, and local 4-bit Mistral-7B reranking."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._doc: dict[str, str] = {}
        self._summary: dict[str, str] = {}
        self._pop: dict[str, float] = {}
        self._price: dict[str, float] = {}
        self._df: dict[str, int] = {}
        self._cache: dict[tuple, list[str]] = {}
        self._vectors: dict[str, object] = {}
        self._encoder = None
        self._index = None
        self._intent = None
        self._intents: dict[str, bool] = {}
        self._overrides: dict[str, bool] = {}
        
        self._mistral_model = None
        self._mistral_tokenizer = None
        self._mistral_failed = False
        
        self._build_index()
        self._load_model()
        if USE_MISTRAL_RERANK:
            self._load_mistral()

    def _load_model(self) -> None:
        try:
            from starter.dense import Encoder
            self._encoder = Encoder(MODEL_DIR)
        except Exception:
            self._encoder = None
            return

        if USE_DENSE:
            try:
                from starter.dense import DenseIndex
                index = DenseIndex.load(EMBEDDINGS)
                if index.meta.get("count") not in (None, len(index.ids)):
                    raise ValueError("embedding metadata does not match the id list")
                self._index = index
            except Exception:
                self._index = None

        try:
            from starter.intent import IntentDetector
            self._intent = IntentDetector.load(PROTOTYPES)
        except Exception:
            self._intent = None

    def _load_mistral(self) -> None:
        global _CACHED_MISTRAL_MODEL, _CACHED_MISTRAL_TOKENIZER
        if _CACHED_MISTRAL_MODEL is not None and _CACHED_MISTRAL_TOKENIZER is not None:
            self._mistral_model = _CACHED_MISTRAL_MODEL
            self._mistral_tokenizer = _CACHED_MISTRAL_TOKENIZER
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self._mistral_tokenizer = AutoTokenizer.from_pretrained(
                MISTRAL_MODEL_ID,
                local_files_only=True
            )
            self._mistral_model = AutoModelForCausalLM.from_pretrained(
                MISTRAL_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            _CACHED_MISTRAL_MODEL = self._mistral_model
            _CACHED_MISTRAL_TOKENIZER = self._mistral_tokenizer
        except Exception:
            self._mistral_model = None
            self._mistral_tokenizer = None
            self._mistral_failed = True

    def _declined(self, message: str) -> bool:
        if self._intent is None or self._encoder is None:
            return False
        cached = self._intents.get(message)
        if cached is None:
            from starter.intent import NO_PREFERENCE
            try:
                cached = self._intent.classify_message(message, self._encoder)[0] == NO_PREFERENCE
            except Exception:
                cached = False
            self._intents[message] = cached
        return cached

    def _dense_ranking(self, text: str, limit: int) -> list[str]:
        if self._encoder is None or self._index is None or not text.strip():
            return []
        try:
            vector = self._vectors.get(text)
            if vector is None:
                vector = self._encoder.encode_query(text, prefix=QUERY_INSTRUCTION)
                self._vectors[text] = vector
            return self._index.search(vector, limit)
        except Exception:
            return []

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        for table in ("products", "products_stem"):
            cursor.execute(
                f"CREATE VIRTUAL TABLE {table} USING fts5("
                "parent_asin UNINDEXED, title, categories, features, details, store, description, "
                "tokenize='unicode61 remove_diacritics 2')"
            )
        raw: list[tuple] = []
        stemmed: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                pid = str(product["parent_asin"])
                fields = [
                    _text(product.get("title")), _text(product.get("categories")),
                    _text(product.get("features")), _text(product.get("details")),
                    _text(product.get("store")), _text(product.get("description")),
                ]
                raw.append((pid, *fields))
                stems = [" ".join(_stemmed(field)) for field in fields]
                stemmed.append((pid, *stems))
                self._doc[pid] = " " + " ".join(stems) + " "
                self._summary[pid] = self._summarize(product)
                number = product.get("rating_number")
                self._pop[pid] = math.log1p(number) if isinstance(number, int) else 0.0
                price = product.get("price")
                self._price[pid] = float(price) if isinstance(price, (int, float)) and price > 0 else 0.0
                if len(raw) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", raw)
                    cursor.executemany("INSERT INTO products_stem VALUES (?,?,?,?,?,?,?)", stemmed)
                    raw.clear()
                    stemmed.clear()
        if raw:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", raw)
            cursor.executemany("INSERT INTO products_stem VALUES (?,?,?,?,?,?,?)", stemmed)
        self.connection.commit()
        self._docs = len(self._doc)
        self._pop_max = max(self._pop.values()) or 1.0

    def _search(self, table: str, terms: list[str], limit: int) -> list[str]:
        unique = list(dict.fromkeys(terms))[:40]
        if not unique:
            return []
        expression = " OR ".join(f'"{t}"' for t in unique)
        try:
            rows = self.connection.execute(
                f"SELECT parent_asin FROM {table} WHERE {table} MATCH ? "
                f"ORDER BY bm25({table}, {BM25_WEIGHTS}) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row[0]) for row in rows]

    def _idf(self, term: str) -> float:
        cached = self._df.get(term)
        if cached is None:
            try:
                row = self.connection.execute(
                    "SELECT count(*) FROM products_stem WHERE products_stem MATCH ?", (f'"{term}"',)
                ).fetchone()
                cached = int(row[0]) if row else 0
            except sqlite3.OperationalError:
                cached = 0
            self._df[term] = cached
        return math.log((self._docs + 1.0) / (cached + 1.0))

    def _expand(self, feedback: list[str], query: list[str]) -> list[str]:
        if not feedback:
            return []
        known = set(query)
        weights: dict[str, float] = {}
        appearances: dict[str, int] = {}
        for rank, pid in enumerate(feedback):
            decay = 1.0 / (1.0 + rank)
            counts: dict[str, int] = {}
            for token in self._doc[pid].split():
                if len(token) > 2 and token not in known and not DIGITS.search(token):
                    counts[token] = counts.get(token, 0) + 1
            length = sum(counts.values()) or 1
            for token, count in counts.items():
                weights[token] = weights.get(token, 0.0) + decay * (count / length)
                appearances[token] = appearances.get(token, 0) + 1
        candidates = {t: w for t, w in weights.items() if appearances[t] >= MIN_FEEDBACK_DOCS}
        scored = sorted(candidates.items(), key=lambda kv: -kv[1] * self._idf(kv[0]))
        return [token for token, _ in scored[:EXPANSION_TERMS]]

    @staticmethod
    def _summarize(product: dict) -> str:
        parts = [
            f"title: {_clip(_text(product.get('title')), 80)}",
            f"category: {_clip(_text(product.get('categories')), 40)}",
            f"features: {_clip(_text(product.get('features')), 100)}",
        ]
        price = product.get("price")
        if isinstance(price, (int, float)) and price > 0:
            parts.append(f"price: ${price:.2f}")
        return "; ".join(parts)

    def _rerank(self, state: dict, fused: list[tuple[str, float]], top_k: int) -> list[str]:
        head = fused[:RERANK_DEPTH]
        if len(head) < 2:
            return [pid for pid, _ in fused[:top_k]]
        phrases = sorted(state["phrases"])
        budget = state.get("budget")
        weights = RERANK_WEIGHTS
        scored: list[tuple[float, str]] = []
        for pid, _ in head:
            document = self._doc[pid]
            phrase = sum(1 for p in phrases if p in document) / len(phrases) if phrases else 0.0
            score = (weights["phrase"] * phrase
                     + weights["popularity"] * (self._pop[pid] / self._pop_max))
            if budget and self._price[pid] > 0:
                score += weights["price"] * max(0.0, 1.0 - abs(self._price[pid] - budget) / budget)
            scored.append((-score, pid))
        scored.sort()
        reordered = [pid for _, pid in scored]
        ranked = reordered + [pid for pid, _ in fused[RERANK_DEPTH:]]
        return self._mistral_rerank(state, ranked, top_k)

    def _mistral_prompt(self, state: dict, pool: list[str], top_k: int) -> str:
        history = _clip(" ".join(state["text"]), 800)
        terms = ", ".join(self._query(state["plain"])[:20])
        budget = state.get("budget")
        candidates = "\n".join(
            f"[{idx}] {self._summary.get(pid, '')}"
            for idx, pid in enumerate(pool, 1)
        )
        budget_text = f" | Budget: ${budget:.2f}" if budget else ""
        return (
            "Rank the items from most to least relevant based on user requirements.\n"
            f"User History: {history}\n"
            f"Extracted Needs: {terms}{budget_text}\n\n"
            f"Items:\n{candidates}\n\n"
            f"Output JSON only formatted as {{\"order\": [1, 2, ...]}} ordering the best {top_k} indices:"
        )

    @staticmethod
    def _parse_mistral_order(output: str, pool: list[str]) -> list[str]:
        output = output.strip()
        values: list[object] = []
        try:
            match = re.search(r"\{.*?\}", output, re.DOTALL)
            if match:
                payload = json.loads(match.group(0))
                if isinstance(payload, dict):
                    raw = payload.get("order", [])
                else:
                    raw = payload
                if isinstance(raw, list):
                    values = raw
        except Exception:
            pass

        if not values:
            values = [int(m) for m in re.findall(r"(?<![A-Z0-9])#?(\d{1,2})(?![A-Z0-9])", output, re.I)]

        ordered: list[str] = []
        seen: set[str] = set()
        pool_set = set(pool)
        for value in values:
            pid = None
            if isinstance(value, int) and 1 <= value <= len(pool):
                pid = pool[value - 1]
            elif isinstance(value, str):
                s = value.strip()
                if s.isdigit() and 1 <= int(s) <= len(pool):
                    pid = pool[int(s) - 1]
                elif s in pool_set:
                    pid = s
            if pid and pid not in seen:
                ordered.append(pid)
                seen.add(pid)
        return ordered

    def _run_mistral(self, prompt: str) -> str | None:
        if self._mistral_failed or self._mistral_model is None or self._mistral_tokenizer is None:
            return None
        try:
            import torch
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self._mistral_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._mistral_tokenizer(formatted_prompt, return_tensors="pt").to(self._mistral_model.device)
            with torch.no_grad():
                outputs = self._mistral_model.generate(
                    **inputs,
                    max_new_tokens=32,  # Short token budget for immediate JSON completion
                    do_sample=False,
                    pad_token_id=self._mistral_tokenizer.eos_token_id
                )
            generated = outputs[0][inputs.input_ids.shape[1]:]
            return self._mistral_tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception:
            self._mistral_failed = True
            return None

    def _mistral_rerank(self, state: dict, ranked: list[str], top_k: int) -> list[str]:
        # Skip LLM on early turns where algorithmic BM25/phrase heuristic is already sufficient
        if not USE_MISTRAL_RERANK or len(ranked) < 2 or self._mistral_failed or len(state["text"]) < 2:
            return ranked[:top_k]
        
        pool = ranked[:max(top_k, MISTRAL_RERANK_DEPTH)]
        prompt = self._mistral_prompt(state, pool, top_k)
        output = self._run_mistral(prompt)
        if not output:
            return ranked[:top_k]
        ordered = self._parse_mistral_order(output, pool)
        if not ordered:
            return ranked[:top_k]
        seen = set(ordered)
        reranked = ordered + [pid for pid in pool if pid not in seen]
        return (reranked + ranked[len(pool):])[:top_k]

    @staticmethod
    def _fuse(rankings: list[tuple[list[str], float]], top_k: int) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for ranking, weight in rankings:
            for rank, pid in enumerate(ranking):
                scores[pid] = scores.get(pid, 0.0) + weight / (RRF_K + rank + 1)
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ordered[:top_k]

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "seen": set(), "plain": {}, "stems": {},
            "asked": set(), "retired": set(), "last_ask": None, "size": 0,
            "text": [], "budget": None, "suppress": {}, "phrases": set(),
            "values": {},
        }

    def _override(self, message: str) -> bool:
        if self._intent is None or self._encoder is None:
            return False
        cached = self._overrides.get(message)
        if cached is None:
            from starter.intent import OVERRIDE
            try:
                cached = self._intent.classify_message(message, self._encoder)[0] == OVERRIDE
            except Exception:
                cached = False
            self._overrides[message] = cached
        return cached

    def _drop_terms(self, state: dict, values: set[str]) -> None:
        if not values:
            return
        stems = {stem(value) for value in values}
        for value in values:
            state["plain"].pop(value, None)
        for token in stems:
            state["stems"].pop(token, None)
        state["phrases"] = {phrase for phrase in state["phrases"]
                            if not any(f" {token} " in phrase for token in stems)}
        pattern = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(v) for v in values), re.I)
        state["text"] = [pattern.sub("", entry) for entry in state["text"]]

    def _retract(self, state: dict, message: str) -> None:
        from starter.intent import extract_values
        stale = set()
        for attribute, value in extract_values(message).items():
            stale |= {old for old in state["values"].get(attribute, set())
                      if old != value}
        self._drop_terms(state, stale)

    def _accumulate(self, state: dict, message: str) -> None:
        repeated = message in state["seen"]
        state["seen"].add(message)
        if self._intent is not None:
            suppress = state["suppress"]
            for attribute in suppress:
                suppress[attribute] *= ATTRIBUTE_DECAY
            if state["last_ask"]:
                suppress[state["last_ask"]] = max(
                    suppress.get(state["last_ask"], 0.0),
                    DECLINE_PENALTY if self._declined(message) else 1.0,
                )
        if repeated:
            return
        if USE_OVERRIDE and self._override(message):
            self._retract(state, message)
        self._absorb(state, message)
        from starter.intent import extract_values
        for attribute, value in extract_values(message).items():
            state["values"].setdefault(attribute, set()).add(value)

    def _absorb(self, state: dict, message: str) -> None:
        state["text"].append(message)
        hint = PRICE_HINT.search(message)
        if hint:
            try:
                state["budget"] = float(hint.group(1) or hint.group(2))
            except (TypeError, ValueError):
                pass
        for term in _terms(message):
            state["plain"][term] = state["plain"].get(term, 0) + 1
        sequence = _stemmed(message)
        for term in sequence:
            state["stems"][term] = state["stems"].get(term, 0) + 1
        state["phrases"].update(f" {a} {b} " for a, b in zip(sequence, sequence[1:]))
        state["phrases"].update(f" {a} {b} {c} "
                                for a, b, c in zip(sequence, sequence[1:], sequence[2:]))

    @staticmethod
    def _query(counts: dict[str, int]) -> list[str]:
        return sorted(counts, key=lambda term: (-counts[term], term))

    def _choose(self, state: dict) -> str:
        if self._intent is not None:
            suppress = state["suppress"]
            return min(ATTRIBUTE_ORDER,
                       key=lambda a: (suppress[a] if a in suppress else FRESH_COST,
                                      ATTRIBUTE_ORDER.index(a)))
        for attribute in ATTRIBUTE_ORDER:
            if attribute in state["retired"] or attribute in state["asked"]:
                continue
            return attribute
        for attribute in ATTRIBUTE_ORDER:
            if attribute not in state["retired"]:
                return attribute
        return "feature"

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        before = len(state["plain"])
        self._accumulate(state, user_message)
        if state["last_ask"] and len(state["plain"]) == before:
            state["retired"].add(state["last_ask"])
        state["size"] = len(state["plain"])
        plain = self._query(state["plain"])
        stems = self._query(state["stems"])
        
        # When not running the generative LLM reranker, dynamic cache avoids repeat calls
        if not USE_MISTRAL_RERANK:
            key = (tuple(plain), top_k)
            if key in self._cache:
                ranked = self._cache[key]
                return self._reply(state, ranked)
                
        stemmed_ranking = self._search("products_stem", stems, RETRIEVE)
        rankings = [
            (self._search("products", plain, RETRIEVE), 1.0),
            (stemmed_ranking, 1.0),
            (self._dense_ranking(" ".join(state["text"]), DENSE_LIMIT) if USE_DENSE else [],
             DENSE_WEIGHT),
        ]
        if USE_EXPANSION:
            rankings.append((self._search(
                "products_stem",
                stems + self._expand(stemmed_ranking[:FEEDBACK_DOCS], stems),
                RETRIEVE,
            ), 1.0))
        fused = self._fuse([r for r in rankings if r[0]], max(top_k, RERANK_DEPTH))
        ranked = self._rerank(state, fused, top_k)
        return self._reply(state, ranked)

    def _reply(self, state: dict, ranked: list[str]) -> dict:
        attribute = self._choose(state)
        state["asked"].add(attribute)
        state["last_ask"] = attribute
        return {
            "message": QUESTIONS.get(attribute, QUESTIONS["feature"]),
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": pid} for pid in ranked],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }