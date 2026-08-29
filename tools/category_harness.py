"""How much of the score depends on the customer naming the catalog's own category?

The reference evaluator builds its opening message with
``coarse_category(target.categories)`` - the target's own category string,
verbatim - so BM25 receives an exact-match anchor lifted straight from the
document it is meant to find. Real shoppers do not talk that way. This harness
degrades that anchor in graduated steps and reports the curve.

Levels, from the evaluator's own wording down to nothing:

  L0 exact        "Shoes Fashion Sneakers"          as the evaluator gives it
  L1 natural      "shoes fashion sneakers"          same words, spoken
  L2 head noun    "sneaker"                         only the last word, singular
  L3 synonym      "trainer"                         an everyday word for it
  L4 functional   "something to wear on my feet"    described, not named
  L5 absent       "something"                       no category at all

Only L3 and L4 need a vocabulary, and their coverage is reported rather than
assumed; uncovered categories fall back to the level above.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORD = re.compile(r"[A-Za-z&']+")

# Everyday words for the same thing - the vocabulary mismatch a real customer
# creates without trying. Keyed on the singular head noun.
SYNONYM = {
    "sneaker": "trainer", "pant": "trouser", "sweater": "jumper", "jumper": "sweater",
    "brief": "underwear", "undershirt": "vest", "sweatshirt": "sweater",
    "tunic": "long top", "bodysuit": "one piece", "legging": "tight",
    "clog": "slip on shoe", "flat": "flat shoe", "slide": "sliders",
    "anorak": "rain jacket", "cap": "hat", "beanie": "woolly hat",
    "wallet": "billfold", "purse": "handbag", "bag": "handbag",
    "sleepshirt": "nightshirt", "short": "shorts", "jean": "denims",
    "tee": "t shirt", "shirt": "top", "blouse": "top", "dress": "frock",
    "sock": "socks", "bra": "bralette", "slipper": "house shoe",
    "watch": "wristwatch", "belt": "waist belt", "boot": "ankle boot",
    "sandal": "open shoe", "necklace": "chain", "earring": "ear stud",
    "bikini": "swimsuit", "warmer": "warmers", "sleeve": "arm sleeve",
}

# Described by what it does, never named. Keyed on a substring of the category.
FUNCTIONAL = {
    "necklace": "something to wear round my neck", "earring": "something for my ears",
    "bracelet": "something for my wrist", "ring": "something for my finger",
    "watch": "something to tell the time", "wallet": "something to keep cards in",
    "bag": "something to carry things in", "purse": "something to carry things in",
    "shoe": "something to wear on my feet", "sneaker": "something to wear on my feet",
    "boot": "something for my feet in bad weather", "sandal": "something open for my feet",
    "slipper": "something to wear indoors", "clog": "something easy to slip on",
    "sock": "something to go under my shoes", "belt": "something to hold my trousers up",
    "hat": "something for my head", "cap": "something for my head",
    "glove": "something for my hands", "scarf": "something for my neck",
    "bra": "an undergarment", "brief": "an undergarment", "undershirt": "an undergarment",
    "pant": "something for my legs", "jean": "something for my legs",
    "legging": "something for my legs", "short": "something for my legs in summer",
    "shirt": "something for my top half", "tee": "something for my top half",
    "blouse": "something for my top half", "tunic": "something for my top half",
    "top": "something for my top half", "dress": "something to wear to an occasion",
    "jacket": "an outer layer", "coat": "an outer layer", "anorak": "an outer layer",
    "hoodi": "something warm and casual", "sweatshirt": "something warm and casual",
    "sweater": "something warm", "sleep": "something to sleep in",
    "swim": "something to swim in", "bikini": "something to swim in",
}


def singular(word: str) -> str:
    lower = word.lower()
    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if lower.endswith(("ses", "xes", "zes", "ches", "shes")):
        return lower[:-2]
    if lower.endswith("s") and not lower.endswith("ss"):
        return lower[:-1]
    return lower


def head_noun(category: str) -> str:
    words = WORD.findall(category)
    return singular(words[-1]) if words else "item"


def express(category: str, level: int) -> tuple[str, bool]:
    """The customer's wording for a category, and whether a vocabulary covered it."""
    if level <= 0:
        return category, True
    if level == 1:
        return category.lower(), True
    head = head_noun(category)
    if level == 2:
        return head, True
    if level == 3:
        return (SYNONYM[head], True) if head in SYNONYM else (head, False)
    if level == 4:
        low = category.lower()
        for key, phrase in FUNCTIONAL.items():
            if key in low:
                return phrase, True
        return "something", False
    return "something", True


def main() -> None:
    from evaluator import local_evaluator as E

    parser = argparse.ArgumentParser(description="Category-dependency harness")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--dense", default="1")
    args = parser.parse_args()

    import starter.agent as A
    A.USE_DENSE = args.dense == "1"

    samples = E.load_jsonl(args.dataset)
    catalog_ids, categories, products = E.catalog_index(args.catalog)
    base = A.Agent(args.catalog)
    print(f"dense attached: {base._index is not None}", flush=True)
    original = E.initial_message

    def make(level: int):
        def rewritten(sample, category, disclosed):
            said, _ = express(category, level)
            return original(sample, category, disclosed).replace(category, said, 1)
        return rewritten

    names = {0: "L0 exact (as the evaluator gives it)", 1: "L1 same words, spoken",
             2: "L2 head noun only", 3: "L3 everyday synonym",
             4: "L4 described, never named", 5: "L5 no category at all"}
    print(f"{'category wording':<40}{'score':>9}{'hit':>8}{'MRR':>8}{'covered':>9}", flush=True)
    for level in range(6):
        covered = sum(express(E.coarse_category(categories[str(s['ground_truth']['parent_asin'])]),
                              level)[1] for s in samples)
        E.initial_message = make(level)
        agent = A.Agent.__new__(A.Agent)
        agent.__dict__.update(base.__dict__)
        agent._cache, agent._sessions = {}, {}
        agent._vectors = dict(base._vectors)
        result = E.evaluate(agent, samples, catalog_ids, categories, products)
        print(f"{names[level]:<40}{result['recommended_technical_score']:>9.4f}"
              f"{result['hit_rate_at_10']:>8.3f}{result['mrr']:>8.3f}"
              f"{covered/len(samples):>8.0%}", flush=True)
    E.initial_message = original


if __name__ == "__main__":
    main()
