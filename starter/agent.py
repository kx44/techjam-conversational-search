from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path


WORD = re.compile(r"[a-z0-9]+")
PRICE = re.compile(r"\$\s*(\d+(?:\.\d+)?)|\b(?:under|below|below|max|around|about|up to)\s+(\d+(?:\.\d+)?)", re.I)
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
             "denim", "suede", "mesh", "satin", "linen", "fabric", "canvas")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
          "purple", "yellow", "orange", "silver", "gold", "navy", "beige")

# Words that carry conversation rather than product meaning. Removing them keeps
# retrieval from scoring a customer's politeness against the catalog.
CHATTER = {
    "a", "about", "actually", "after", "all", "also", "am", "an", "and", "any", "anything",
    "are", "around", "as", "at", "be", "been", "bit", "but", "buy", "buying", "by", "can",
    "could", "definitely", "did", "do", "does", "dont", "either", "else", "even", "ever",
    "find", "for", "forget", "from", "get", "getting", "give", "go", "going", "good", "got",
    "great", "guess", "had", "has", "have", "help", "her", "here", "hey", "hi", "hello", "him",
    "his", "hmm", "honestly", "how", "i", "id", "if", "ideal", "ideally", "im", "importance",
    "important", "in", "into", "is", "it", "its", "ive", "just", "kind", "know", "let", "like",
    "little", "look", "looking", "lot", "love", "made", "main", "many", "matter", "matters",
    "maybe", "me", "mind", "more", "most", "much", "my", "need", "needs", "nice", "no", "not",
    "nothing", "now", "of", "off", "oh", "ok", "okay", "on", "one", "only", "opinion", "or",
    "other", "our", "out", "over", "please", "prefer", "preference", "pretty", "probably",
    "put", "quite", "really", "right", "said", "same", "say", "scratch", "see", "shop",
    "shopping", "should", "show", "similar", "so", "some", "something", "sorry", "sort",
    "specific", "still", "strong", "such", "sure", "take", "tell", "than", "thanks", "that",
    "the", "their", "them", "then", "there", "these", "they", "thing", "things", "think",
    "this", "those", "though", "thought", "to", "too", "totally", "track", "trying", "up",
    "us", "use", "using", "very", "want", "wanted", "was", "way", "we", "well", "what",
    "whatever", "when", "which", "while", "who", "why", "will", "with", "would", "yeah",
    "yes", "yet", "you", "your", "youve",
}
# Generic cues that a person is correcting themselves. Any natural phrasing of an
# intent change tends to contain one, so this does not depend on a fixed template.
CORRECTION = ("actually", "instead", "scratch that", "forget", "ignore", "change of plan",
              "changed my mind", "correct myself", "rather than", "on second thought",
              "never mind", "nevermind", "not that", "no longer", "instead of")

ATTRIBUTES = ("material", "color", "brand", "budget", "style", "size", "use_case", "feature")
QUESTIONS = {
    "material": "What material would you prefer?",
    "color": "Any particular colour you have in mind?",
    "brand": "Is there a brand you tend to go for?",
    "budget": "Roughly what budget are you working with?",
    "style": "What sort of style or fit are you after?",
    "size": "What size or sizing do you need?",
    "use_case": "What will you mainly be using it for?",
    "feature": "Is there any specific feature it has to have?",
    "category": "What kind of item are we talking about exactly?",
}

CANDIDATES_PER_TURN = 400
POOL_CAP = 1500
RERANK_DEPTH = 40


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def _content(text: str) -> list[str]:
    return [t for t in _tokens(text) if len(t) > 1 and t not in CHATTER]


class Agent:
    """Conversational retrieval over the frozen catalog.

    Deliberately makes no assumption about how the customer phrases things: no
    template prefixes, no exact-match category lookup, no reliance on the
    customer quoting catalog text verbatim. Everything is lexical evidence
    accumulated over the conversation and scored against the whole catalog.

    Runs fully offline on the local catalog; no network and no model calls.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._norm: dict[str, str] = {}      # space-delimited tokens, for word-boundary tests
        self._cat: dict[str, set[str]] = {}  # category tokens, used as a soft signal
        self._price: dict[str, float] = {}
        self._pop: dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                pid = str(product["parent_asin"])
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))
                blob = " ".join((title, categories, features, details, store, description))
                self._norm[pid] = " " + " ".join(_tokens(blob)) + " "
                self._cat[pid] = {t for t in _tokens(categories) if len(t) > 2}
                price = product.get("price")
                self._price[pid] = float(price) if isinstance(price, (int, float)) and price > 0 else 0.0
                rating_number = product.get("rating_number")
                self._pop[pid] = math.log1p(rating_number) if isinstance(rating_number, int) else 0.0
                batch.append((pid, title, categories, features, details, store, description))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self.connection.commit()
        self._pop_max = max(self._pop.values()) or 1.0

    # ---------------------------------------------------------------- session

    def reset(self, session_id: str, user_profile: dict) -> None:
        tags: list[str] = []
        if isinstance(user_profile, dict):
            tags = [str(t).lower() for t in user_profile.get("preference_tags") or []]
        self._sessions[session_id] = {
            "weights": {},        # term -> accumulated weight
            "phrases": {},        # bigram -> weight
            "pool": [],           # candidate ids under consideration
            "seen": set(),
            "hits": {},           # term -> set of pool ids containing it
            "phrase_hits": {},
            "budget": None,
            "asked": set(),
            "tags": tags,
            "ranked": [],
        }

    # ------------------------------------------------------------ observation

    def _observe(self, state: dict, message: str) -> None:
        lowered = message.lower()
        if any(cue in lowered for cue in CORRECTION):
            # A correction demotes what came before without discarding it; the
            # customer may still want some of it.
            for key in state["weights"]:
                state["weights"][key] *= 0.25
            for key in state["phrases"]:
                state["phrases"][key] *= 0.25
        found = PRICE.search(message)
        if found:
            value = found.group(1) or found.group(2)
            try:
                state["budget"] = float(value)
            except ValueError:
                pass
        terms = _content(message)
        for term in terms:
            state["weights"][term] = state["weights"].get(term, 0.0) + 1.0
        for first, second in zip(terms, terms[1:]):
            key = f" {first} {second} "
            state["phrases"][key] = state["phrases"].get(key, 0.0) + 1.0

    # -------------------------------------------------------------- retrieval

    def _generate(self, state: dict) -> None:
        terms = sorted(state["weights"], key=lambda t: -state["weights"][t])[:24]
        if not terms:
            return
        expression = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 4.0, 3.0, 2.0, 2.0, 1.5, 1.0) LIMIT ?",
                (expression, CANDIDATES_PER_TURN),
            ).fetchall()
        except sqlite3.OperationalError:
            return
        fresh = [str(r[0]) for r in rows if str(r[0]) not in state["seen"]]
        for pid in fresh:
            state["seen"].add(pid)
            state["pool"].append(pid)
        if len(state["pool"]) > POOL_CAP:                      # keep the strongest evidence
            state["pool"] = self._rank(state, state["pool"])[:POOL_CAP]
            state["seen"] = set(state["pool"])
            for key in state["hits"]:
                state["hits"][key] &= state["seen"]
            for key in state["phrase_hits"]:
                state["phrase_hits"][key] &= state["seen"]
            fresh = []
        # incremental posting lists: only new terms x whole pool, old terms x new ids
        for term in state["weights"]:
            needle = f" {term} "
            if term not in state["hits"]:
                state["hits"][term] = {p for p in state["pool"] if needle in self._norm[p]}
            elif fresh:
                state["hits"][term] |= {p for p in fresh if needle in self._norm[p]}
        for phrase in state["phrases"]:
            if phrase not in state["phrase_hits"]:
                state["phrase_hits"][phrase] = {p for p in state["pool"] if phrase in self._norm[p]}
            elif fresh:
                state["phrase_hits"][phrase] |= {p for p in fresh if phrase in self._norm[p]}

    def _rank(self, state: dict, pool: list[str]) -> list[str]:
        weights = state["weights"]
        total = sum(weights.values()) or 1.0
        phrase_total = sum(state["phrases"].values()) or 1.0
        query_tokens = set(weights) | set(state["tags"])
        budget = state["budget"]
        scored: list[tuple[float, str]] = []
        for pid in pool:
            covered = 0.0
            for term, weight in weights.items():
                if pid in state["hits"].get(term, ()):
                    covered += weight
            coverage = covered / total
            phrased = 0.0
            for phrase, weight in state["phrases"].items():
                if pid in state["phrase_hits"].get(phrase, ()):
                    phrased += weight
            phrase_score = phrased / phrase_total
            cats = self._cat[pid]
            affinity = len(cats & query_tokens) / len(cats) if cats else 0.0
            score = (3.0 * coverage) + (1.5 * phrase_score) + (1.2 * affinity)
            score += 0.25 * (self._pop[pid] / self._pop_max)
            if budget and self._price[pid] > 0:
                score += 0.5 * max(0.0, 1.0 - abs(self._price[pid] - budget) / max(budget, 1.0))
            scored.append((-score, pid))
        scored.sort()
        return [pid for _, pid in scored]

    # --------------------------------------------------------------- question

    def _value(self, pid: str, attribute: str) -> str | None:
        blob = self._norm[pid]
        if attribute == "material":
            return next((m for m in MATERIALS if f" {m} " in blob), None)
        if attribute == "color":
            return next((c for c in COLORS if f" {c} " in blob), None)
        if attribute == "budget":
            price = self._price[pid]
            return None if price <= 0 else str(int(math.log1p(price) * 2))
        return None

    def _covered(self, state: dict, attribute: str) -> bool:
        known = set(state["weights"])
        if attribute == "material":
            return any(m in known for m in MATERIALS)
        if attribute == "color":
            return any(c in known for c in COLORS)
        if attribute == "budget":
            return state["budget"] is not None
        return False

    def _choose_attribute(self, state: dict, ranked: list[str]) -> str:
        """Ask about whatever still splits the leading candidates most evenly."""
        head = ranked[:RERANK_DEPTH]
        best, best_score = None, -1.0
        for attribute in ("material", "color", "budget"):
            if attribute in state["asked"] or self._covered(state, attribute):
                continue
            values = [self._value(pid, attribute) for pid in head]
            values = [v for v in values if v]
            if len(values) < 4:
                continue
            counts: dict[str, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            n = len(values)
            entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
            if entropy > best_score:
                best, best_score = attribute, entropy
        if best:
            return best
        for attribute in ATTRIBUTES:
            if attribute not in state["asked"]:
                return attribute
        return "feature"

    # ---------------------------------------------------------------- respond

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            self.reset(session_id, {})
            state = self._sessions[session_id]
        if not isinstance(user_message, str):
            user_message = ""
        self._observe(state, user_message)
        self._generate(state)
        ranked = self._rank(state, state["pool"]) if state["pool"] else state["ranked"]
        state["ranked"] = ranked
        attribute = self._choose_attribute(state, ranked)
        state["asked"].add(attribute)
        return {
            "message": QUESTIONS.get(attribute, QUESTIONS["feature"]),
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": pid} for pid in ranked[:top_k]],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
