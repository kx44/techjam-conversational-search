"""Porter stemmer (Porter, 1980), pure stdlib.

Conflates inflected forms so that a customer saying "necklaces" matches a
catalog entry saying "necklace". Vendored rather than pulled from nltk to keep
the agent dependency-free and offline.
"""
from __future__ import annotations

VOWELS = "aeiou"

STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ("logi", "log"),
)
STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
)
STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)


def _consonant(word: str, index: int) -> bool:
    letter = word[index]
    if letter in VOWELS:
        return False
    if letter == "y":
        return index == 0 or not _consonant(word, index - 1)
    return True


def _measure(word: str) -> int:
    """Count the VC repetitions in [C](VC)^m[V]."""
    count = 0
    index = 0
    length = len(word)
    while index < length and _consonant(word, index):
        index += 1
    while index < length:
        while index < length and not _consonant(word, index):
            index += 1
        if index >= length:
            return count
        count += 1
        while index < length and _consonant(word, index):
            index += 1
    return count


def _has_vowel(word: str) -> bool:
    return any(not _consonant(word, i) for i in range(len(word)))


def _double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _consonant(word, len(word) - 1)


def _cvc(word: str) -> bool:
    """Consonant-vowel-consonant where the final letter is not w, x or y."""
    if len(word) < 3:
        return False
    if not (_consonant(word, len(word) - 3)
            and not _consonant(word, len(word) - 2)
            and _consonant(word, len(word) - 1)):
        return False
    return word[-1] not in "wxy"


def _step1a(word: str) -> str:
    if word.endswith("sses") or word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        return word[:-1] if _measure(word[:-3]) > 0 else word
    stem = None
    if word.endswith("ed") and _has_vowel(word[:-2]):
        stem = word[:-2]
    elif word.endswith("ing") and _has_vowel(word[:-3]):
        stem = word[:-3]
    if stem is None:
        return word
    if stem.endswith(("at", "bl", "iz")):
        return stem + "e"
    if _double_consonant(stem) and not stem.endswith(("l", "s", "z")):
        return stem[:-1]
    if _measure(stem) == 1 and _cvc(stem):
        return stem + "e"
    return stem


def _step1c(word: str) -> str:
    if word.endswith("y") and _has_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _replace(word: str, rules, minimum: int) -> str:
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            return stem + replacement if _measure(stem) > minimum else word
    return word


def _step4(word: str) -> str:
    for suffix in STEP4:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if suffix == "ion" and not stem.endswith(("s", "t")):
                continue
            return stem if _measure(stem) > 1 else word
    return word


def _step5(word: str) -> str:
    if word.endswith("e"):
        stem = word[:-1]
        measure = _measure(stem)
        if measure > 1 or (measure == 1 and not _cvc(stem)):
            word = stem
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


def stem(word: str) -> str:
    if len(word) <= 2:
        return word
    word = _step1c(_step1b(_step1a(word)))
    word = _replace(word, STEP2, 0)
    word = _replace(word, STEP3, 0)
    word = _step4(word)
    return _step5(word)
