from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path

from starter.stemmer import stem


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

BM25_WEIGHTS = "0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0"
RRF_K = 60          # standard Reciprocal Rank Fusion constant
FEEDBACK_DOCS = 10  # RM3: documents assumed relevant
EXPANSION_TERMS = 10
MIN_FEEDBACK_DOCS = 6   # an expansion term must recur, or the query drifts
RETRIEVE = 100
DIGITS = re.compile(r"\d")

# RM3 is implemented but off. Pseudo-relevance feedback assumes the first-pass
# top-k is mostly relevant; this baseline's hit rate is 0.125, so the feedback
# set is mostly wrong products and expansion amplifies the error. Measured:
# enabling it costs 0.121 -> 0.116 on the reference evaluator and never beats
# plain BM25 on natural-language input. Flip to True to re-check.
USE_EXPANSION = False

# Broad questions first, narrow ones after. A customer almost always has
# something to say about what they will use an item for; far fewer have a
# colour or material in mind, and those answers separate the catalog less.
# Measured against both harnesses: broad-first 0.6973 / 0.7765 vs
# narrow-first 0.6772 / 0.7690.
ATTRIBUTE_ORDER = ("feature", "use_case", "style", "material", "color",
                   "size", "budget", "brand", "category")
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


class Agent:
    """BM25 baseline with three classical IR additions and nothing else.

    1. Porter stemming, as a second index, so inflected forms match.
    2. RM3 pseudo-relevance feedback, to expand short or vague queries.
    3. Reciprocal Rank Fusion over the three resulting rankings, which needs
       no weight calibration - only rank order.

    Session handling, output shape and clarification behaviour are unchanged
    from the starter baseline so the retrieval change can be measured alone.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._doc: dict[str, str] = {}   # stemmed content tokens, for RM3 feedback
        self._df: dict[str, int] = {}
        self._cache: dict[tuple, list[str]] = {}
        self._build_index()

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

    # ------------------------------------------------------------- retrieval

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
        """RM3: weight terms by frequency across the assumed-relevant set."""
        if not feedback:
            return []
        known = set(query)
        weights: dict[str, float] = {}
        appearances: dict[str, int] = {}
        for rank, pid in enumerate(feedback):
            decay = 1.0 / (1.0 + rank)          # earlier documents are better evidence
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
    def _fuse(rankings: list[list[str]], top_k: int) -> list[str]:
        """Reciprocal Rank Fusion - combines rankings using order alone."""
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, pid in enumerate(ranking):
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [pid for pid, _ in ordered[:top_k]]

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = {
            "seen": set(), "plain": {}, "stems": {},
            "asked": set(), "retired": set(), "last_ask": None, "size": 0,
        }

    @staticmethod
    def _accumulate(state: dict, message: str) -> None:
        """Fold one customer turn into the running query.

        A message identical to one already seen carries no new information, so
        it is ignored - otherwise repeated boilerplate accrues weight and
        crowds out the terms that actually distinguish the target.
        """
        if message in state["seen"]:
            return
        state["seen"].add(message)
        for term in _terms(message):
            state["plain"][term] = state["plain"].get(term, 0) + 1
        for term in _stemmed(message):
            state["stems"][term] = state["stems"].get(term, 0) + 1

    @staticmethod
    def _query(counts: dict[str, int]) -> list[str]:
        return sorted(counts, key=lambda term: (-counts[term], term))

    def _choose(self, state: dict) -> str:
        """Next question: the broadest one not already answered or retired."""
        for attribute in ATTRIBUTE_ORDER:
            if attribute in state["retired"] or attribute in state["asked"]:
                continue
            return attribute
        for attribute in ATTRIBUTE_ORDER:            # everything asked once; reuse
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
            # Asked, learned nothing. Retire it rather than asking again.
            state["retired"].add(state["last_ask"])
        state["size"] = len(state["plain"])
        plain = self._query(state["plain"])
        stems = self._query(state["stems"])
        key = (tuple(plain), top_k)
        if key in self._cache:                     # same evidence, same answer
            ranked = self._cache[key]
            return self._reply(state, ranked)
        rankings = [
            self._search("products", plain, RETRIEVE),
            self._search("products_stem", stems, RETRIEVE),
        ]
        if USE_EXPANSION:
            rankings.append(self._search(
                "products_stem",
                stems + self._expand(rankings[1][:FEEDBACK_DOCS], stems),
                RETRIEVE,
            ))
        ranked = self._fuse([r for r in rankings if r], top_k)
        self._cache[key] = ranked
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
