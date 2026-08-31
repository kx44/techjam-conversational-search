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

# Per-column BM25 weights, positional against the CREATE VIRTUAL TABLE order
# (the UNINDEXED id takes slot 1): title, categories, features, details, store,
# description. Inherited from the starter kit and swept rather than trusted -
# ten configurations span only 0.026, and flat 1.0 across every field loses
# just 0.022, so the weighting barely matters now that reranking decides order
# and BM25 only has to supply recall. Boosting features and details, where the
# constraint text the customer quotes actually lives, is worse (0.8602 and
# 0.8534 against 0.8638); raising categories looks better at 0.8670 but the
# curve is non-monotone - 6.0 scores below both 4.0 and 8.0 - so that is one
# session of noise, not signal. Left as found.
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
# Dense product retrieval, on. It was off until tools/shopper_sim.py existed,
# because the two harnesses available before it both had customers who quote
# catalog text close to verbatim - the regime where lexical matching is
# strongest and embeddings have least to add. On a shopper who names things in
# their own words it is the only change that has ever moved hit rate off its
# ceiling, 0.935 -> 0.985.
#
#   weight   reference  realistic_sim  shopper_sim
#     0.00      0.8922         0.9389       0.8673
#     0.25      0.8864         0.9358       0.8778
#     1.00      0.8864         0.9346       0.9006   <- shipped
#     2.50      0.8512         0.9236       0.8790
#
# The cost is binary rather than proportional: nearly all of it is the step
# from 0.00 to 0.25, and the reference score is flat from 0.25 to 1.00 while
# the realistic shopper gains 0.023. So a quarter-weight is strictly dominated
# - if it runs at all it should run at full voice.
#
# Requires data/bge_embeddings.npy (tools/build_embeddings.py, ~18 min). Absent,
# the agent runs BM25-only and says nothing; check agent._index is not None.
USE_DENSE = True
QUERY_INSTRUCTION = True    # BGE v1.5 recommends an instruction on queries only
# Dense retrieval adds recall and costs precision: at equal weight it raises
# hit rate but pushes the exact match down the list, because embeddings find
# the right kind of product and cannot pick which one. Down-weighting keeps the
# recall. Measured official/realistic by weight: 0.00 -> .6973/.7765,
# 0.15 -> .7027/.7864, 0.25 -> .7022/.7857, 0.50 -> .6987/.7883,
# 1.00 -> .6897/.7690. Anything in 0.15-0.50 is within noise of the others; the
# midpoint is taken rather than the argmax.
DENSE_WEIGHT = 1.0
# Dense is a complement, not a final authority. When lexical routes agree, BGE
# is down-weighted so exact catalog matches stay in charge. When the user gives
# a short or paraphrased request, BGE keeps full weight to backfill candidates
# BM25 may miss.
DENSE_LEXICAL_WEIGHT = 0.15
DENSE_PARAPHRASE_WEIGHT = 1.0
DENSE_LEXICAL_OVERLAP = 0.25
DENSE_LIMIT = RETRIEVE
PREFERENCE_TERM_BOOST = 2
PREFERENCE_DENSE_BOOST = 2

# Intent override. The OVERRIDE class was recognised from the start but never
# acted on, so a retraction left every superseded term in the query. The newly
# stated value now supersedes the earlier value of the SAME attribute: "I'd
# like leather" then "I need canvas instead" drops leather, keeps canvas, and
# keeps the category.
#
# Same-attribute is the only link available - the message names the new value
# and never the old one ("ignore my earlier preference" has no referent), so
# attribute identity is what makes the target identifiable at all. The cost of
# that safety is that a retraction ACROSS attributes ("buckle closure ...
# actually leather") is a real retraction and is not honoured.
#
# MERGE NOTE. On its own branch this was implemented as `_drop_terms`, which
# reached into `plain`, `stems`, `phrases` and `text` and deleted the stale
# value in place. That cannot survive here: the preference layer rebuilds all
# four of those from `positive_clauses` on every turn, so an in-place deletion
# is erased by the next rebuild and the retraction silently stops working.
# Superseded values are therefore recorded as state and excised *during* the
# rebuild, which is idempotent and gives the same semantics.
#
# Bit-identical to off on the reference evaluator. That evaluator draws old and
# new from different slices of one candidate list, so they land on different
# attributes (18/30) or on the same value (1/30), and 11/30 state a value
# outside the known vocabulary - there is never a same-attribute conflict.
USE_OVERRIDE = True

# Reranking. Fusion decides which products are plausible; this decides their
# order. BM25 runs an OR query, so a product matching two query terms out of
# twelve can outrank one matching ten, and phrases are shattered into tokens -
# both are precisely what costs MRR when the target is in the list but not on
# top. The reranker scores the head of the fused list on evidence the OR query
# throws away: how much of the query a product actually covers, whether the
# customer's phrases survive intact, and whether the match is in the title.
RERANK_DEPTH = 100
RERANK_WEIGHTS = {
    "phrase": 0.8,      # adjacent query terms surviving as a phrase in the product
    "popularity": 0.2,  # targets are real purchases, so common items are likelier
    "price": 0.3,       # only applies once the customer names a budget
}
# If all retrieval routes strongly agree on a candidate, but the phrase-heavy
# reranker pushes it out of sight, rescue it into the visible list. This keeps
# semantic help narrow: dense can confirm a lexical match, but cannot overrule
# BM25 on its own.
SEMANTIC_RESCUE_INSERT = 5      # zero-based; rescue to visible rank 6
SEMANTIC_RESCUE_FUSED_MAX = 2   # zero-based; fused top 3 only
SEMANTIC_RESCUE_LEXICAL_MAX = 2 # zero-based; raw+stem top 3 only
SEMANTIC_RESCUE_DENSE_MAX = 4   # zero-based; dense top 5 only
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
EXPLORE_ORDER = ATTRIBUTE_ORDER
CONSTRAIN_ORDER = ("material", "color", "size", "budget", "feature",
                   "use_case", "style", "brand", "category")
VERIFY_ORDER = ("material", "color", "size", "budget", "style", "feature",
                "use_case", "brand", "category")
FLAT_ORDER = ("feature", "style", "material", "color", "use_case", "size",
              "budget", "brand", "category")
OVERGENERAL_TERMS = 4
LOW_ROUTE_OVERLAP = 0.12
TOP_OVERLAP = 20
# Flat fused scores mean the top candidates are hard to distinguish. In that
# situation retrieval is usually waiting for one more discriminating detail, so
# the dialogue policy switches into `flat` mode and asks sharper questions.
FLAT_POOL_SPREAD = 0.27
FLAT_POOL_VARIANCE = 0.000007

# A declined question is not an answered one. The reference customer may decline
# for reasons unrelated to preference - a Boundary session refuses the first
# attribute asked, whatever it is - and treating that as settled retires the
# question permanently. In one measured session that consumed "feature", which
# was the only attribute unlocking three of its four constraints.
#
# Asking an attribute suppresses it; declining suppresses it harder. Suppression
# decays each turn, so a declined attribute drifts back into contention later
# rather than being re-asked at once. Re-asking immediately costs a turn and
# cancels the gain; deferring it does not - MTTC improves, 2.98 -> 2.91.
# Measured over decay 0.45-0.70 and fresh cost 0.15-0.50: every setting in
# 0.45-0.60 x 0.2-0.3 beats the previous behaviour, so this is a plateau rather
# than a tuned point, and the midpoint is taken.
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
    """Deterministic conversational search pipeline.

    The agent keeps an explicit constraint state, masks rejected values before
    retrieval, combines raw BM25, stemmed BM25 and optional BGE dense retrieval,
    then reranks the fused pool with lightweight catalog evidence.

    Everything is local and deterministic after model embeddings are loaded:
    no LLM call is made at response time, and the output shape stays compatible
    with the evaluator contract.
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
        self._intent = None
        self._intents: dict[str, bool] = {}
        self._overrides: dict[str, bool] = {}
        self._clause_classifier = None
        self._mention_signals: dict[tuple[str, str, str], object] = {}
        self._build_index()
        self._load_model()

    def _load_model(self) -> None:
        """Attach whichever model-backed parts have their artifacts present.

        The two consumers need different files and are loaded independently:
        product retrieval needs the 73 MB catalog matrix, while state and
        preference detection can build its small prototype matrix directly from
        sentences in starter.intent. Coupling them cost the detector whenever
        the catalog matrix was absent, which is the cheaper artifact to skip.
        """
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

            try:
                self._intent = IntentDetector.load(PROTOTYPES)
            except Exception:
                self._intent = IntentDetector.build(self._encoder)
        except Exception:
            self._intent = None
        try:
            from starter.intent import ClausePreferenceClassifier

            self._clause_classifier = ClausePreferenceClassifier.build(self._encoder)
        except Exception:
            self._clause_classifier = None

    def _declined(self, message: str) -> bool:
        """Did the customer decline the question rather than answer it?

        Catalog statistics cannot tell a decline from a reveal - the rarest new
        term has median IDF 1.34 against 1.15 - so this needs the embedding
        detector, and is simply unavailable without it.
        """
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
        """Dense route: retrieve semantically similar products using BGE.

        This is allowed to fail closed. If the local model/index is missing,
        the rest of the pipeline still works with the two lexical routes.
        """
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

    def _score_mention(self, clause: str, mention):
        """Classify one clause/value pair as approve, reject, or neutral."""
        if self._encoder is None or self._clause_classifier is None:
            from starter.intent import NEUTRAL, PreferenceSignal

            return PreferenceSignal(mention.attribute, mention.value, NEUTRAL, 0.0, 0.0, {})
        key = (clause, mention.attribute, mention.value)
        cached = self._mention_signals.get(key)
        if cached is None:
            try:
                cached = self._clause_classifier.score_mention(clause, mention, self._encoder)
            except Exception:
                from starter.intent import NEUTRAL, PreferenceSignal

                cached = PreferenceSignal(mention.attribute, mention.value, NEUTRAL, 0.0, 0.0, {})
            self._mention_signals[key] = cached
        return cached

    def _build_index(self) -> None:
        """Build raw and stemmed FTS indexes plus small reranking side tables."""
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
        """Run one lexical route over either the raw or stemmed FTS table."""
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
        route_ranks = state.get("route_ranks", {})
        fused_ranks = {pid: rank for rank, (pid, _) in enumerate(head)}
        rescues: list[tuple[int, str]] = []
        # Rescue only candidates with simultaneous lexical+dense agreement.
        # This fixes cases like "Chinese New Year" where fusion is confident
        # but phrase reranking is overly harsh, while avoiding broad semantic
        # promotion of merely similar products.
        for pid in reordered[top_k:]:
            ranks = route_ranks.get(pid, {})
            if (
                fused_ranks.get(pid, RERANK_DEPTH) <= SEMANTIC_RESCUE_FUSED_MAX
                and ranks.get("raw", RERANK_DEPTH) <= SEMANTIC_RESCUE_LEXICAL_MAX
                and ranks.get("stem", RERANK_DEPTH) <= SEMANTIC_RESCUE_LEXICAL_MAX
                and ranks.get("dense", RERANK_DEPTH) <= SEMANTIC_RESCUE_DENSE_MAX
            ):
                rescues.append((fused_ranks[pid], pid))
        for _, pid in reversed(sorted(rescues)):
            reordered.remove(pid)
            reordered.insert(min(SEMANTIC_RESCUE_INSERT, top_k - 1, len(reordered)), pid)
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
            # plain/stems/text are rebuilt after every turn from accepted
            # evidence only. Rejected and superseded values remain in state,
            # but never leak into positive retrieval terms.
            "text": [], "budget": None, "suppress": {}, "phrases": set(),
            # Constraint state. Three maps, three jobs, all attribute -> values:
            #   rejected    the customer ruled it out            (negation)
            #   superseded  a later turn replaced it             (override)
            #   values      everything stated, so `_retract` knows what to
            #               supersede when a new value arrives for an attribute
            "positive_clauses": [], "rejected": {}, "preferences": {},
            "values": {}, "superseded": {},
            "diagnostics": {}, "mode": "explore",
        }

    def _override(self, message: str) -> bool:
        """Did the customer retract a preference rather than add one?"""
        if self._intent is None or self._encoder is None:
            return False
        cached = self._overrides.get(message)
        if cached is None:
            from starter.intent import OVERRIDE

            try:
                cached = self._intent.classify_message(
                    message, self._encoder)[0] == OVERRIDE
            except Exception:
                cached = False
            self._overrides[message] = cached
        return cached

    def _retract(self, state: dict, message: str) -> None:
        """Record that this turn's value supersedes the attribute's earlier one.

        Nothing is deleted here. The rebuild is the single place that decides
        what reaches retrieval, so a retraction has to be a fact it consults,
        not an edit it will overwrite.
        """
        from starter.intent import extract_values

        for attribute, value in extract_values(message).items():
            for old in state["values"].get(attribute, set()):
                if old != value:
                    self._remember(state["superseded"], attribute, old)
            # A value stated now is wanted now, whatever was said before.
            self._forget(state["superseded"], attribute, value)

    def _add_positive_clause(self, state: dict, clause: str) -> None:
        state["positive_clauses"].append(clause)

    @staticmethod
    def _remember(mapping: dict, attribute: str, value: str) -> None:
        mapping.setdefault(attribute, set()).add(value)

    @staticmethod
    def _forget(mapping: dict, attribute: str, value: str) -> None:
        values = mapping.get(attribute)
        if not values:
            return
        values.discard(value)
        if not values:
            mapping.pop(attribute, None)

    @staticmethod
    def _strip(state: dict, clause: str) -> str:
        """One clause with every superseded value removed.

        Excision, not exclusion: dropping the whole clause would take the
        category anchor with it, and that anchor is worth more than the stale
        value costs.
        """
        stale = {v for values in state["superseded"].values() for v in values}
        if not stale:
            return clause
        pattern = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(v) for v in sorted(stale)), re.I)
        return pattern.sub(" ", clause)

    def _rebuild_positive_state(self, state: dict) -> None:
        """Materialize retrieval inputs from the current constraint state."""
        state["plain"].clear()
        state["stems"].clear()
        state["phrases"].clear()
        positive_clauses = []
        for clause in state["positive_clauses"]:
            try:
                from starter.intent import extract_mentions

                mentions = extract_mentions(clause)
            except Exception:
                mentions = []
            if any(mention.value in state["rejected"].get(mention.attribute, set())
                   for mention in mentions):
                continue
            positive_clauses.append(clause)
        positive_clauses = [self._strip(state, clause) for clause in positive_clauses]
        dense_clauses = list(positive_clauses)
        state["budget"] = None
        for clause in positive_clauses:
            hint = PRICE_HINT.search(clause)
            if hint:
                try:
                    state["budget"] = float(hint.group(1) or hint.group(2))
                except (TypeError, ValueError):
                    pass
            for term in _terms(clause):
                state["plain"][term] = state["plain"].get(term, 0) + 1
            sequence = _stemmed(clause)
            for term in sequence:
                state["stems"][term] = state["stems"].get(term, 0) + 1
            # Phrases come from adjacency *within one accepted evidence clause*.
            # Rejected clauses must not contribute phrase features either.
            state["phrases"].update(f" {a} {b} " for a, b in zip(sequence, sequence[1:]))
            state["phrases"].update(f" {a} {b} {c} "
                                    for a, b, c in zip(sequence, sequence[1:], sequence[2:]))
        for (attribute, value), signal in state["preferences"].items():
            if signal.weight <= 0 or value in state["rejected"].get(attribute, set()):
                continue
            if value in state["superseded"].get(attribute, set()):
                continue
            repeats = max(1, round(signal.weight * PREFERENCE_TERM_BOOST))
            for term in _terms(value):
                state["plain"][term] = state["plain"].get(term, 0) + repeats
            for term in _stemmed(value):
                state["stems"][term] = state["stems"].get(term, 0) + repeats
            dense_repeats = max(1, round(signal.weight * PREFERENCE_DENSE_BOOST))
            dense_clauses.extend(f"{attribute} {value}" for _ in range(dense_repeats))
        state["text"] = dense_clauses

    def _accumulate(self, state: dict, message: str) -> None:
        """Fold one customer turn into the running query.

        A message identical to one already seen carries no new information, so
        it is ignored - otherwise repeated boilerplate accrues weight and
        crowds out the terms that actually distinguish the target.
        """
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
            # Nothing new to add to the query, but the question policy still
            # advances - a customer repeating themselves verbatim should not
            # freeze it.
            return
        if USE_OVERRIDE and self._override(message):
            # Safe on the opening turn without a guard: nothing has been
            # recorded yet, so no value can be superseded.
            self._retract(state, message)
        from starter.intent import extract_values

        for attribute, value in extract_values(message).items():
            state["values"].setdefault(attribute, set()).add(value)
        if self._clause_classifier is None or self._encoder is None:
            self._add_positive_clause(state, message)
            self._rebuild_positive_state(state)
            return
        from starter.intent import (HARD_APPROVE, HARD_REJECT, NEUTRAL, SOFT_APPROVE,
                                    extract_mentions, split_clauses)

        for clause in split_clauses(message):
            mentions = extract_mentions(clause)
            if mentions:
                # Score each mention once. The verdict is read twice - a
                # rejection anywhere disqualifies the whole clause as evidence,
                # and only a clause that survives that can approve anything -
                # but scoring twice invited the two passes to drift apart.
                signals = [(m, self._score_mention(clause, m)) for m in mentions]
                for mention, signal in signals:
                    state["preferences"][(mention.attribute, mention.value)] = signal
                if any(signal.label == HARD_REJECT for _, signal in signals):
                    for mention, signal in signals:
                        if signal.label == HARD_REJECT:
                            self._remember(state["rejected"],
                                           mention.attribute, mention.value)
                    continue
                for mention, signal in signals:
                    if signal.label in (HARD_APPROVE, SOFT_APPROVE, NEUTRAL):
                        # Stating a value plainly takes back an earlier ban on
                        # it, the same way it takes back a supersession.
                        self._forget(state["rejected"], mention.attribute, mention.value)
                self._add_positive_clause(state, clause)
                continue
            if self._declined(clause):
                continue
            self._add_positive_clause(state, clause)
        self._rebuild_positive_state(state)

    @staticmethod
    def _query(counts: dict[str, int]) -> list[str]:
        return sorted(counts, key=lambda term: (-counts[term], term))

    @staticmethod
    def _overlap(left: list[str], right: list[str], depth: int = TOP_OVERLAP) -> float:
        a = set(left[:depth])
        b = set(right[:depth])
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _dense_weight(self, state: dict, raw_ranking: list[str], stemmed_ranking: list[str]) -> float:
        """Let dense help paraphrases, but keep lexical agreement in charge."""
        if len(state["plain"]) < OVERGENERAL_TERMS:
            return DENSE_PARAPHRASE_WEIGHT
        lexical_overlap = self._overlap(raw_ranking, stemmed_ranking)
        if lexical_overlap >= DENSE_LEXICAL_OVERLAP:
            return DENSE_LEXICAL_WEIGHT
        return DENSE_PARAPHRASE_WEIGHT

    def _diagnose(self, state: dict, rankings: list[tuple[list[str], float]],
                  fused: list[tuple[str, float]]) -> dict:
        """Cheap uncertainty signals for question policy, not ranking.

        `route_overlap` captures disagreement across retrieval routes.
        `flat_pool` captures a diffuse candidate set where many top products
        have nearly equal fused scores. Both signals help decide whether to
        ask broad exploratory questions or sharper constraint questions.
        """
        active = [ranking for ranking, _ in rankings if ranking]
        overlaps = []
        for index, left in enumerate(active):
            for right in active[index + 1:]:
                overlaps.append(self._overlap(left, right))
        route_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        fused_scores = [score for _, score in fused[:TOP_OVERLAP]]
        if len(fused_scores) > 1 and fused_scores[0] > 0:
            score_spread = (fused_scores[0] - fused_scores[-1]) / fused_scores[0]
            mean = sum(fused_scores) / len(fused_scores)
            score_variance = sum((score - mean) ** 2 for score in fused_scores) / len(fused_scores)
        else:
            score_spread = 1.0
            score_variance = 0.0
        flat_pool = (
            score_spread < FLAT_POOL_SPREAD
            or score_variance < FLAT_POOL_VARIANCE
        )
        preference_count = sum(
            1 for signal in state["preferences"].values()
            if getattr(signal, "weight", 0.0) > 0
        )
        rejected_count = sum(len(values) for values in state["rejected"].values())
        superseded_count = sum(len(values) for values in state["superseded"].values())
        query_terms = len(state["plain"])
        over_general = (
            query_terms < OVERGENERAL_TERMS
            or (preference_count == 0 and route_overlap < LOW_ROUTE_OVERLAP)
        )
        needs_constraints = (
            not over_general
            and preference_count < 2
            and len(state["asked"]) >= 2
        )
        if rejected_count or superseded_count:
            mode = "verify"
        elif over_general:
            mode = "explore"
        elif flat_pool and (needs_constraints or len(state["asked"]) >= 1):
            mode = "flat"
        elif needs_constraints:
            mode = "constrain"
        else:
            mode = "normal"
        return {
            "query_terms": query_terms,
            "route_overlap": round(route_overlap, 4),
            "score_spread": round(score_spread, 6),
            "score_variance": round(score_variance, 9),
            "flat_pool": flat_pool,
            "dense_weight": state.get("dense_weight", DENSE_WEIGHT),
            "candidate_count": len(fused),
            "preference_count": preference_count,
            "rejected_count": rejected_count,
            "superseded_count": superseded_count,
            "over_general": over_general,
            "needs_constraints": needs_constraints,
            "mode": mode,
        }

    def _choose(self, state: dict) -> str:
        """Next question: least-suppressed, or the broadest untried one."""
        mode = state.get("mode", "normal")
        order = {
            "explore": EXPLORE_ORDER,
            "constrain": CONSTRAIN_ORDER,
            "verify": VERIFY_ORDER,
            "flat": FLAT_ORDER,
        }.get(mode, ATTRIBUTE_ORDER)
        if mode == "verify":
            front = tuple(
                attribute for attribute in VERIFY_ORDER
                if attribute in state["rejected"] or attribute in state["superseded"]
            )
            order = front + tuple(attribute for attribute in order if attribute not in front)
        if self._intent is not None:
            suppress = state["suppress"]
            return min(order,
                       key=lambda a: (suppress[a] if a in suppress else FRESH_COST,
                                      order.index(a)))
        for attribute in order:
            if attribute in state["retired"] or attribute in state["asked"]:
                continue
            return attribute
        for attribute in order:                       # everything asked once; reuse
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
        """Main turn entry point used by the evaluator/API."""
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
        raw_ranking = self._search("products", plain, RETRIEVE)
        dense_ranking = self._dense_ranking(" ".join(state["text"]), DENSE_LIMIT) if USE_DENSE else []
        dense_weight = self._dense_weight(state, raw_ranking, stemmed_ranking) if dense_ranking else 0.0
        state["dense_weight"] = dense_weight
        # Three active routes: exact lexical, stemmed lexical, and semantic.
        # RRF uses only rank order, so the route weights matter only by ratio.
        rankings = [
            (raw_ranking, 1.0),
            (stemmed_ranking, 1.0),
            (dense_ranking, dense_weight),
        ]
        if USE_EXPANSION:
            expansion_ranking = self._search(
                "products_stem",
                stems + self._expand(stemmed_ranking[:FEEDBACK_DOCS], stems),
                RETRIEVE,
            )
            rankings.append((expansion_ranking, 1.0))
        route_ranks: dict[str, dict[str, int]] = {}
        # Keep per-route positions for diagnostics and the guarded rescue.
        for name, ranking in (("raw", raw_ranking), ("stem", stemmed_ranking),
                              ("dense", dense_ranking)):
            for rank, pid in enumerate(ranking[:RERANK_DEPTH]):
                route_ranks.setdefault(pid, {})[name] = rank
        state["route_ranks"] = route_ranks
        fused = self._fuse([r for r in rankings if r[0]], max(top_k, RERANK_DEPTH))
        diagnostics = self._diagnose(state, rankings, fused)
        state["diagnostics"] = diagnostics
        state["mode"] = diagnostics["mode"]
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
