"""Semantic intent detection over the same BGE encoder used for retrieval.

Cue-word matching ("actually", "instead") only catches corrections a customer
phrases the expected way. Embedding the message and comparing it against a
handful of prototype sentences per class generalises to phrasings nobody
enumerated, and costs one extra encode per turn on a model already loaded for
product retrieval - no second model, no generative call, no network.

Classification is deliberately conservative: when the best match is weak, or
two classes are nearly tied, the result is UNKNOWN and the caller changes no
state. A misfired OVERRIDE discards evidence the customer never retracted.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

NORMAL = "NORMAL"
OVERRIDE = "OVERRIDE"
NO_PREFERENCE = "NO_PREFERENCE"
UNKNOWN = "UNKNOWN"
ACCEPT = "ACCEPT"
REJECT = "REJECT"
NEUTRAL = "NEUTRAL"
HARD_REJECT = "HARD_REJECT"
SOFT_REJECT = "SOFT_REJECT"
SOFT_APPROVE = "SOFT_APPROVE"
HARD_APPROVE = "HARD_APPROVE"

# Varied phrasings per class - short, first person, the way a shopper writes.
PROTOTYPES: dict[str, tuple[str, ...]] = {
    NORMAL: (
        "I need something made of leather.",
        "It should be black.",
        "For that, what matters is a rubber sole.",
        "I'm looking for a pair of running shoes.",
        "Cotton would be ideal.",
        "Something warm for the winter.",
        "My budget is around thirty dollars.",
        "It has to be waterproof.",
        "I'd prefer a slim fit.",
        "Size medium, please.",
        "A key requirement is that it is machine washable.",
        "I want something for hiking.",
    ),
    OVERRIDE: (
        "Actually, ignore my earlier preference.",
        "Scratch that, what I really need is something else.",
        "Let me correct myself, forget what I said before.",
        "Sorry, change of plan, ignore that.",
        "On second thought, I would rather have something different.",
        "Never mind what I said earlier.",
        "I've changed my mind about that.",
        "Instead of that, I want something else.",
        "Disregard my previous request.",
        "Actually no, not that one.",
        "Forget the last thing I asked for.",
        "I take that back, I need something different.",
    ),
    NO_PREFERENCE: (
        "I don't have a preference for that.",
        "No strong opinion on that one.",
        "Honestly that doesn't matter much to me.",
        "I don't mind either way, you pick.",
        "Please use your judgment on that.",
        "No preference there.",
        "I'm not fussed about that.",
        "That doesn't matter to me.",
        "I don't have an additional preference.",
        "Whatever you think is best.",
        "Either is fine with me.",
        "I really don't mind.",
    ),
}
CLASSES = tuple(PROTOTYPES)

# A weak best match, or two classes nearly tied, means UNKNOWN.
MIN_SIMILARITY = 0.60
MIN_MARGIN = 0.02

# Attribute values readable from a message, so a conflicting value can be
# detected as an override even when the customer states it flatly.
ATTRIBUTE_VALUES: dict[str, tuple[str, ...]] = {
    "material": ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
                 "rayon", "denim", "suede", "mesh", "satin", "linen", "canvas", "fleece"),
    "color": ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
              "purple", "yellow", "orange", "silver", "gold", "navy", "beige"),
}
PRICE = re.compile(r"\$\s*(\d+(?:\.\d+)?)|\b(?:under|below|around|about|up to|max)\s+(\d+(?:\.\d+)?)", re.I)
WORD = re.compile(r"[a-z0-9]+")
# Split on sentence and clause boundaries. A correction is usually one clause
# of a longer turn - "Actually, ignore that. What I need is leather." - and
# embedding the whole turn averages the retraction away against the new
# constraint, which is how the reference simulator phrases every override.
SENTENCE = re.compile(r"(?<=[.!?;])\s+|\s+--\s+")
# A non-NORMAL reading of any one clause decides the turn.
PRIORITY = (OVERRIDE, NO_PREFERENCE, NORMAL)

# Shopper preference clauses are shorter and more local than whole-turn intent.
# Classify the relation around one masked attribute value at a time so the
# material word itself cannot dominate the BGE comparison.
CLAUSE_PROTOTYPES: dict[str, tuple[str, ...]] = {
    ACCEPT: (
        'Clause: "I like [VALUE]."\nPair: color=[VALUE]\nDoes the user want this value? yes, the user wants this value.',
        'Clause: "[VALUE] works for me."\nPair: color=[VALUE]\nDoes the user want this value? yes, this value is acceptable.',
        'Clause: "[VALUE] would be good."\nPair: material=[VALUE]\nDoes the user want this value? yes, the user prefers this value.',
        'Clause: "I prefer [VALUE]."\nPair: material=[VALUE]\nDoes the user want this value? yes, the user prefers this value.',
        'Clause: "A key requirement is: [VALUE]."\nPair: material=[VALUE]\nDoes the user want this value? yes, this value is required.',
        'Clause: "For that, what matters is: [VALUE]."\nPair: material=[VALUE]\nDoes the user want this value? yes, this value matters.',
        'Clause: "Something in [VALUE] is perfect."\nPair: color=[VALUE]\nDoes the user want this value? yes, this value is wanted.',
        'Clause: "Yes, [VALUE] is fine."\nPair: color=[VALUE]\nDoes the user want this value? yes, this value is fine.',
    ),
    REJECT: (
        'Clause: "I do not want [VALUE]."\nPair: material=[VALUE]\nDoes the user want this value? no, the user rejects this value.',
        'Clause: "[VALUE] is not for me."\nPair: material=[VALUE]\nDoes the user want this value? no, this value is not acceptable.',
        'Clause: "No [VALUE] please."\nPair: material=[VALUE]\nDoes the user want this value? no, the user does not want this value.',
        'Clause: "Avoid [VALUE]."\nPair: color=[VALUE]\nDoes the user want this value? no, the user wants to avoid this value.',
        'Clause: "Anything but [VALUE]."\nPair: color=[VALUE]\nDoes the user want this value? no, any value except this one.',
        'Clause: "Definitely not [VALUE]."\nPair: color=[VALUE]\nDoes the user want this value? no, this value is strongly rejected.',
        'Clause: "Not [VALUE] lah."\nPair: color=[VALUE]\nDoes the user want this value? no, the user rejects this value.',
        'Clause: "I would prefer something other than [VALUE]."\nPair: material=[VALUE]\nDoes the user want this value? no, the user wants something else.',
        'Clause: "Not too keen on [VALUE]."\nPair: material=[VALUE]\nDoes the user want this value? no, this value is not preferred.',
        'Clause: "[VALUE] is a dealbreaker."\nPair: material=[VALUE]\nDoes the user want this value? no, this value is unacceptable.',
        'Clause: "I am not looking for [VALUE]."\nPair: material=[VALUE]\nDoes the user want this value? no, this value is rejected.',
        'Clause: "Without [VALUE]."\nPair: material=[VALUE]\nDoes the user want this value? no, this value should be excluded.',
        'Clause: "I dislike [VALUE]."\nPair: color=[VALUE]\nDoes the user want this value? no, the user dislikes this value.',
        'Clause: "[VALUE] will not work."\nPair: color=[VALUE]\nDoes the user want this value? no, this value will not work.',
        'Clause: "I would rather not have [VALUE]."\nPair: material=[VALUE]\nDoes the user want this value? no, the user would rather avoid this value.',
    ),
    NEUTRAL: (
        'Clause: "100% [VALUE]."\nPair: material=[VALUE]\nDoes the user want this value? neutral, this only describes a material.',
        'Clause: "[VALUE] lining."\nPair: material=[VALUE]\nDoes the user want this value? neutral, this is a descriptive fragment.',
        'Clause: "95% [VALUE], 5% spandex."\nPair: material=[VALUE]\nDoes the user want this value? neutral, this is product composition text.',
        'Clause: "Body: [VALUE]."\nPair: material=[VALUE]\nDoes the user want this value? neutral, this only mentions the value.',
        'Clause: "I do not have a preference."\nPair: material=[VALUE]\nDoes the user want this value? neutral, no preference is expressed.',
        'Clause: "Either is fine with me."\nPair: color=[VALUE]\nDoes the user want this value? neutral, either option is fine.',
        'Clause: "I am not sure yet."\nPair: material=[VALUE]\nDoes the user want this value? neutral, the user is undecided.',
        'Clause: "Show me some options."\nPair: color=[VALUE]\nDoes the user want this value? neutral, the user is asking to browse.',
    ),
}
CLAUSE_CLASSES = (ACCEPT, REJECT, NEUTRAL)
CLAUSE_MIN_SIMILARITY = 0.50
CLAUSE_MIN_MARGIN = 0.025
CLAUSE_SPLIT = re.compile(
    r"(?<=[.!?;])\s+|\s+--\s+"
    r"|\s*,\s+(?=(?:definitely|defintely|not|no|avoid|without|never|dont|don't|cant|can't|cannot|skip)\b)"
    r"|\s*,?\s+\b(?<!anything\s)(?:but|however|though|although|except)\b\s+",
    re.I,
)


@dataclass(frozen=True)
class AttributeMention:
    attribute: str
    value: str


@dataclass(frozen=True)
class PreferenceSignal:
    attribute: str
    value: str
    label: str
    weight: float
    confidence: float
    scores: dict[str, float]


def extract_values(message: str) -> dict[str, str]:
    """Attribute values stated in one message."""
    tokens = set(WORD.findall(message.lower()))
    found: dict[str, str] = {}
    for attribute, vocabulary in ATTRIBUTE_VALUES.items():
        hit = next((value for value in vocabulary if value in tokens), None)
        if hit:
            found[attribute] = hit
    price = PRICE.search(message)
    if price:
        found["budget"] = price.group(1) or price.group(2)
    return found


def split_clauses(message: str) -> list[str]:
    """Small, deterministic clause splitter tuned for preference turns."""
    return [clause.strip(" ,") for clause in CLAUSE_SPLIT.split(message) if len(clause.strip(" ,")) > 1]


def extract_mentions(clause: str) -> list[AttributeMention]:
    """Known product attribute values stated inside one clause."""
    tokens = set(WORD.findall(clause.lower()))
    mentions: list[AttributeMention] = []
    for attribute, vocabulary in ATTRIBUTE_VALUES.items():
        for value in vocabulary:
            if value in tokens:
                mentions.append(AttributeMention(attribute, value))
    price = PRICE.search(clause)
    if price:
        mentions.append(AttributeMention("budget", price.group(1) or price.group(2)))
    return mentions


def relation_text(clause: str, mention: AttributeMention) -> str:
    """Prompt-shaped text for relation classification around one value."""
    value_pattern = re.compile(rf"\b{re.escape(mention.value)}\b", re.I)
    masked = value_pattern.sub("[VALUE]", clause)
    return (
        f'Clause: "{masked}"\n'
        f"Pair: {mention.attribute}=[VALUE]\n"
        "Does the user want this value?"
    )


class IntentDetector:
    """Cosine nearest-prototype classifier over the retrieval encoder."""

    def __init__(self, matrix, labels: list[str]) -> None:
        self.matrix = matrix          # (n_prototypes, dim), L2-normalised
        self.labels = labels

    @classmethod
    def build(cls, encoder) -> "IntentDetector":
        sentences: list[str] = []
        labels: list[str] = []
        for name in CLASSES:
            for sentence in PROTOTYPES[name]:
                sentences.append(sentence)
                labels.append(name)
        return cls(encoder.encode(sentences), labels)

    @classmethod
    def load(cls, path, encoder=None) -> "IntentDetector":
        import json
        from pathlib import Path

        import numpy as np

        path = Path(path)
        payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if payload.get("prototypes") != _fingerprint():
            raise ValueError("prototype set changed since the artifact was built")
        return cls(np.load(path.with_suffix(".npy")), payload["labels"])

    def classify_message(self, message: str, encoder) -> tuple[str, float]:
        """Classify a turn clause by clause, strongest non-NORMAL reading wins."""
        clauses = [c.strip() for c in SENTENCE.split(message) if len(c.strip()) > 2]
        if not clauses:
            return UNKNOWN, 0.0
        vectors = encoder.encode(clauses)
        verdicts = [self.classify(vector) for vector in vectors]
        for wanted in PRIORITY:
            hits = [(label, score) for label, score in verdicts if label == wanted]
            if hits:
                return max(hits, key=lambda pair: pair[1])
        return UNKNOWN, max((score for _, score in verdicts), default=0.0)

    def classify(self, vector) -> tuple[str, float]:
        """Best class for an already-encoded clause, or UNKNOWN."""
        scores = self.matrix @ vector
        best: dict[str, float] = {}
        for label, score in zip(self.labels, scores):
            value = float(score)
            if value > best.get(label, -1.0):
                best[label] = value
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        if not ranked or ranked[0][1] < MIN_SIMILARITY:
            return UNKNOWN, ranked[0][1] if ranked else 0.0
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < MIN_MARGIN:
            return UNKNOWN, ranked[0][1]
        return ranked[0][0], ranked[0][1]


class ClausePreferenceClassifier:
    """Nearest-prototype preference classifier over BGE clause embeddings."""

    def __init__(self, matrix, labels: list[str]) -> None:
        self.matrix = matrix
        self.labels = labels

    @classmethod
    def build(cls, encoder) -> "ClausePreferenceClassifier":
        sentences: list[str] = []
        labels: list[str] = []
        for name in CLAUSE_CLASSES:
            for sentence in CLAUSE_PROTOTYPES[name]:
                sentences.append(sentence)
                labels.append(name)
        return cls(encoder.encode(sentences), labels)

    def classify_clause(self, clause: str, encoder) -> tuple[str, float]:
        vector = encoder.encode([clause])[0]
        return self.classify(vector)

    def classify_mention(self, clause: str, mention: AttributeMention, encoder) -> tuple[str, float]:
        vector = encoder.encode([relation_text(clause, mention)])[0]
        return self.classify(vector)

    def score_mention(self, clause: str, mention: AttributeMention, encoder) -> PreferenceSignal:
        vector = encoder.encode([relation_text(clause, mention)])[0]
        scores = self.class_scores(vector)
        accept = scores.get(ACCEPT, 0.0)
        reject = scores.get(REJECT, 0.0)
        neutral = scores.get(NEUTRAL, 0.0)
        polarity = accept - reject
        strength = max(accept, reject) - neutral
        confidence = max(0.0, min(1.0, abs(polarity) + max(0.0, strength)))
        if abs(polarity) < CLAUSE_MIN_MARGIN or max(accept, reject) < CLAUSE_MIN_SIMILARITY:
            label = NEUTRAL
            weight = 0.0
        elif polarity < 0:
            if strength >= CLAUSE_MIN_MARGIN:
                label = HARD_REJECT
                weight = -1.0
            else:
                label = SOFT_REJECT
                weight = -0.5
        elif strength >= CLAUSE_MIN_MARGIN:
            label = HARD_APPROVE
            weight = 1.0
        else:
            label = SOFT_APPROVE
            weight = 0.5
        return PreferenceSignal(
            mention.attribute,
            mention.value,
            label,
            weight,
            confidence,
            scores,
        )

    def classify(self, vector) -> tuple[str, float]:
        ranked = sorted(self.class_scores(vector).items(), key=lambda kv: -kv[1])
        if not ranked or ranked[0][1] < CLAUSE_MIN_SIMILARITY:
            return NEUTRAL, ranked[0][1] if ranked else 0.0
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < CLAUSE_MIN_MARGIN:
            return NEUTRAL, ranked[0][1]
        return ranked[0][0], ranked[0][1]

    def class_scores(self, vector) -> dict[str, float]:
        scores = self.matrix @ vector
        best: dict[str, float] = {}
        for label, score in zip(self.labels, scores):
            value = float(score)
            if value > best.get(label, -1.0):
                best[label] = value
        return best


def _fingerprint() -> str:
    """Stable across processes - hash() is salted per run and cannot be used."""
    digest = hashlib.sha256()
    for name in CLASSES:
        digest.update(name.encode("utf-8"))
        for sentence in PROTOTYPES[name]:
            digest.update(sentence.encode("utf-8"))
    return digest.hexdigest()[:16]
