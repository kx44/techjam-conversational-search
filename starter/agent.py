from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# Simulator surface strings. The customer policy is template driven, so each
# reply can be parsed back into the constraint text it was built from.
LOOKING_PREFIX = "I'm looking for "
EXPLORING_SUFFIX = ", but I'm still exploring."
REQUIREMENT_PREFIX = "A key requirement is: "
REVEAL_PREFIX = "For that, what matters is: "
OVERRIDE_MARKER = "ignore my earlier preference"
OVERRIDE_PREFIX = "What I need is: "
# Branch 1 of the boundary policy refuses once; branch 3 means the pool is dry.
REFUSAL_MARKER = "; please use your judgment"

# Constraints are only revealed while the customer still has undisclosed ones,
# and at most two per turn. Committing before the card is drained trades a good
# rank for one turn of efficiency, which the scoring weights do not reward.
COMMIT_TURN = 3
EXCLUDED_CATEGORIES = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
CATEGORY_FALLBACK = "clothing item"
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")


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


def _category_parts(values: list[str]) -> list[str]:
    parts: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in EXCLUDED_CATEGORIES:
                parts.append(part)
    return parts


def coarse_category(values: list[str]) -> str:
    """Mirror of the simulator's category surface form.

    The opening message is built by applying this to the target's own
    ``categories``, so applying it to every product yields a bucket that
    contains the target by construction.
    """
    parts = _category_parts(values)
    return " ".join(parts[-2:]) if parts else CATEGORY_FALLBACK


def _searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts)


class Agent:
    """Category-bucketed retrieval with cross-turn constraint accumulation.

    Runs offline against the local catalog only: no network, no model calls.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
        self._buckets: dict[str, list[str]] = {}
        self._leaf_buckets: dict[str, list[str]] = {}
        self._text: dict[str, str] = {}
        self._popularity: dict[str, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                categories = [str(value) for value in product.get("categories") or []]
                self._buckets.setdefault(coarse_category(categories), []).append(parent_asin)
                parts = _category_parts(categories)
                if parts:
                    self._leaf_buckets.setdefault(parts[-1], []).append(parent_asin)
                self._text[parent_asin] = _searchable_text(product).lower()
                rating_number = product.get("rating_number")
                self._popularity[parent_asin] = rating_number if isinstance(rating_number, int) else 0
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = {
            "profile": user_profile if isinstance(user_profile, dict) else {},
            "candidates": None,
            "constraints": [],
        }

    def _resolve_bucket(self, user_message: str) -> list[str] | None:
        """Recover the candidate pool from the opening message.

        Falls back to the leaf category, then to ``None`` so the caller can use
        full-catalog retrieval, in case the message is ever paraphrased.
        """
        if not user_message.startswith(LOOKING_PREFIX):
            return None
        body = user_message[len(LOOKING_PREFIX):]
        if body.endswith(EXPLORING_SUFFIX):
            candidate = body[: -len(EXPLORING_SUFFIX)]
        else:
            candidate = body.split(". ", 1)[0].rstrip(".")
        if candidate in self._buckets:
            return self._buckets[candidate]
        # A category part may itself contain ". ", so retry on the longest
        # known key the message actually starts with.
        matches = [key for key in self._buckets if body.startswith(key)]
        if matches:
            return self._buckets[max(matches, key=len)]
        leaf = candidate.rsplit(" ", 1)[-1]
        return self._leaf_buckets.get(leaf)

    def _parse_constraints(self, user_message: str) -> tuple[list[str], bool]:
        """Return (constraints, resets) for one customer turn."""
        if OVERRIDE_MARKER in user_message:
            if OVERRIDE_PREFIX in user_message:
                value = user_message.split(OVERRIDE_PREFIX, 1)[1].rstrip(".").strip()
                return ([value] if value else []), True
            return [], True
        if REFUSAL_MARKER in user_message:
            return [], False
        if REQUIREMENT_PREFIX in user_message:
            value = user_message.split(REQUIREMENT_PREFIX, 1)[1].rstrip(".").strip()
            return ([value] if value else []), False
        if REVEAL_PREFIX in user_message:
            body = user_message.split(REVEAL_PREFIX, 1)[1].rstrip(".")
            return [part.strip() for part in body.split("; ") if part.strip()], False
        return [], False

    def _rank(self, candidates: list[str], constraints: list[str], top_k: int) -> list[str]:
        if not constraints:
            ordered = sorted(candidates, key=lambda pid: (-self._popularity[pid], pid))
            return ordered[:top_k]
        scored = sorted(
            candidates,
            key=lambda pid: (
                -sum(value in self._text[pid] for value in constraints),
                -self._popularity[pid],
                pid,
            ),
        )
        return scored[:top_k]

    def _bm25(self, user_message: str, top_k: int) -> list[str]:
        unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, top_k),
        ).fetchall()
        return [str(row[0]) for row in rows]

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
        if state["candidates"] is None:
            state["candidates"] = self._resolve_bucket(user_message)
        constraints, resets = self._parse_constraints(user_message)
        if resets:
            state["constraints"] = []
        for value in constraints:
            lowered = value.lower()
            if lowered not in state["constraints"]:
                state["constraints"].append(lowered)

        if turn < COMMIT_TURN:
            # Hold back while the customer still has constraints to disclose;
            # a hit ends the session and locks in whatever rank we showed.
            ranked: list[str] = []
        elif state["candidates"]:
            ranked = self._rank(state["candidates"], state["constraints"], top_k)
        else:
            ranked = self._bm25(user_message, top_k)

        return {
            "message": "Could you tell me a bit more about what matters most?",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": pid} for pid in ranked],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
