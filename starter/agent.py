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
# Retrieval depth. A fixed 100 discards the target far more often than the
# full-conversation numbers suggest: at turn 1, BM25 recall@100 is only 52%,
# and in 93 of 200 sessions the target sits beyond rank 100 (median rank 279)
# scoring just 0.18 of the score spread below rank 100 - cut by a hair, not a
# cliff. Depth matters because fusion needs the target *present* to collect
# votes from more than one retriever; RRF already discounts deep ranks, so the
# tail is nearly free. Measured official/realistic: 100 -> .7022/.7857,
# 200 -> .7035/.7877, 500 -> .7081/.7916, 1000 -> .7079/.7916 (plateau).
RETRIEVE = 500
DIGITS = re.compile(r"\d")


# Dense retrieval. Optional by construction: if the model, the precomputed
# matrix, or the runtime dependencies are missing the agent falls back to BM25
# alone. The evaluator turns an agent exception into an empty response, so a
# missing artifact must degrade quietly rather than score zero.
MODEL_DIR = "models/bge-small-en-v1.5"
EMBEDDINGS = "data/bge_embeddings"
USE_DENSE = True
QUERY_INSTRUCTION = True    # BGE v1.5 recommends an instruction on queries only
# Dense retrieval adds recall and costs precision: at equal weight it raises
# hit rate but pushes the exact match down the list, because embeddings find
# the right kind of product and cannot pick which one. Down-weighting keeps the
# recall. Measured official/realistic by weight: 0.00 -> .6973/.7765,
# 0.15 -> .7027/.7864, 0.25 -> .7022/.7857, 0.50 -> .6987/.7883,
# 1.00 -> .6897/.7690. Anything in 0.15-0.50 is within noise of the others; the
# midpoint is taken rather than the argmax.
DENSE_WEIGHT = 0.25
DENSE_LIMIT = RETRIEVE

# Reranking. Fusion decides which products are plausible; this decides their
# order. BM25 runs an OR query, so a product matching two query terms out of
# twelve can outrank one matching ten, and phrases are shattered into tokens -
# both are precisely what costs MRR when the target is in the list but not on
# top. The reranker scores the head of the fused list on evidence the OR query
# throws away: how much of the query a product actually covers, whether the
# customer's phrases survive intact, and whether the match is in the title.
RERANK_DEPTH = 50
RERANK_WEIGHTS = {
    "phrase": 0.8,      # adjacent query terms surviving as a phrase in the product
    "popularity": 0.2,  # targets are real purchases, so common items are likelier
    "price": 0.3,       # only applies once the customer names a budget
}
# Four other features were measured and dropped. Carrying the fusion score in
# as a prior was the worst (0.7708 -> 0.8135 without it): RRF's ordering is
# exactly what the reranker exists to correct, so anchoring to it re-imports
# the flaw. Term coverage and title-match cost 0.8135 -> 0.8625 between them.
# Average rating did nothing either way, in linear or banded form.
#
# Term coverage in particular has been retried and stays dead. It is not a
# stopword problem: expanding the 31-word list to 171 conversational words
# moved the total by +0.001 official and -0.007 natural language, and coverage
# still hurt at every weight under every list. It is not a length-bias problem
# either, though the bias is real - coverage correlates +0.41 with product text
# length where phrase correlates -0.09 - because normalising the length out
# makes it worse still (0.8532 raw -> 0.8040 BM25-normalised), the bias having
# been an accidental stand-in for popularity. Coverage asks how many of the
# customer's words appear somewhere in a product, and inside a pool of fifty
# already-relevant candidates that is close to saturated. Adjacent pairs stay
# discriminative because matching one by chance is rare.
PRICE_HINT = re.compile(r"\$\s*(\d+(?:\.\d+)?)|\b(?:under|below|around|about|up to)\s+(\d+(?:\.\d+)?)", re.I)

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
        self._pop: dict[str, float] = {}
        self._price: dict[str, float] = {}
        self._df: dict[str, int] = {}
        self._cache: dict[tuple, list[str]] = {}
        self._vectors: dict[str, object] = {}
        self._encoder = None
        self._index = None
        self._build_index()
        if USE_DENSE:
            self._load_dense()

    def _load_dense(self) -> None:
        """Attach dense retrieval when its artifacts are present."""
        try:
            from starter.dense import DenseIndex, Encoder

            index = DenseIndex.load(EMBEDDINGS)
            if index.meta.get("count") not in (None, len(index.ids)):
                raise ValueError("embedding metadata does not match the id list")
            self._encoder = Encoder(MODEL_DIR)
            self._index = index
        except Exception:
            self._encoder = None
            self._index = None

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

    def _rerank(self, state: dict, fused: list[tuple[str, float]], top_k: int) -> list[str]:
        """Reorder the head of the fused list using catalog evidence."""
        head = fused[:RERANK_DEPTH]
        if len(head) < 2:
            return [pid for pid, _ in fused[:top_k]]
        stems = list(state["stems"])
        bigrams = [f" {a} {b} " for a, b in zip(stems, stems[1:])]
        budget = state.get("budget")
        weights = RERANK_WEIGHTS
        scored: list[tuple[float, str]] = []
        for pid, _ in head:
            document = self._doc[pid]
            phrase = sum(1 for b in bigrams if b in document) / len(bigrams) if bigrams else 0.0
            score = (weights["phrase"] * phrase
                     + weights["popularity"] * (self._pop[pid] / self._pop_max))
            if budget and self._price[pid] > 0:
                score += weights["price"] * max(0.0, 1.0 - abs(self._price[pid] - budget) / budget)
            scored.append((-score, pid))
        scored.sort()
        reordered = [pid for _, pid in scored]
        return (reordered + [pid for pid, _ in fused[RERANK_DEPTH:]])[:top_k]

    @staticmethod
    def _fuse(rankings: list[tuple[list[str], float]], top_k: int) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion - combines rankings using order alone.

        Score-magnitude fusion (CombSUM over min-max normalised scores) was
        measured and is worse: .7019/.7803 against .7081/.7916 at the same
        depth. Rank order is the more robust signal across retrievers whose
        scores are on incomparable scales.
        """
        scores: dict[str, float] = {}
        for ranking, weight in rankings:
            for rank, pid in enumerate(ranking):
                scores[pid] = scores.get(pid, 0.0) + weight / (RRF_K + rank + 1)
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ordered[:top_k]

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = {
            "seen": set(), "plain": {}, "stems": {},
            "asked": set(), "retired": set(), "last_ask": None, "size": 0,
            "text": [], "budget": None,
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
        # Dense retrieval reads the sentences as written; only the lexical side
        # wants a bag of terms.
        state["text"].append(message)
        hint = PRICE_HINT.search(message)
        if hint:
            try:
                state["budget"] = float(hint.group(1) or hint.group(2))
            except (TypeError, ValueError):
                pass
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
        stemmed_ranking = self._search("products_stem", stems, RETRIEVE)
        rankings = [
            (self._search("products", plain, RETRIEVE), 1.0),
            (stemmed_ranking, 1.0),
            (self._dense_ranking(" ".join(state["text"]), DENSE_LIMIT), DENSE_WEIGHT),
        ]
        if USE_EXPANSION:
            rankings.append((self._search(
                "products_stem",
                stems + self._expand(stemmed_ranking[:FEEDBACK_DOCS], stems),
                RETRIEVE,
            ), 1.0))
        fused = self._fuse([r for r in rankings if r[0]], max(top_k, RERANK_DEPTH))
        ranked = self._rerank(state, fused, top_k)
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
