"""Mine a synonym/association table from the catalog itself.

Motivation: the reference evaluator hands the customer the target's own
category string verbatim, and ``tools/category_harness.py`` measures that this
free anchor is worth about 0.22 of the score. A real shopper says "trainer"
where the catalog says "sneaker". This mines term associations from the catalog
so a query can be expanded toward the vocabulary the catalog actually uses -
no external model, no download, no network.

MEASURED RESULT: it does not work. Query expansion using this table scores
negative or flat at every level of the category harness, including the synonym
level it was built for:

    category wording      baseline  +expansion    delta
    L0 exact                0.8598      0.8566   -0.0031
    L2 head noun            0.7736      0.7520   -0.0216
    L3 synonym              0.7035      0.7015   -0.0020
    L4 described            0.6349      0.6354   +0.0005
    L5 none                 0.6231      0.6063   -0.0168

Two reasons, both visible with ``--inspect``:

1. Polysemy. In a clothing catalog "trainer" means waist trainer, so the top
   associations are cincher, shaper, corset, shapewear. Co-occurrence counts
   cannot know the customer meant footwear; in this corpus the dominant sense
   genuinely is shapewear, and expansion injects the wrong one.
2. The correct associations are lateral, not synonymous. handbag -> satchel,
   hobo, clutch, tote are sibling product types. Adding them widens the query
   into neighbouring categories, which is harmful when the task is to rank one
   specific item first.

Kept as a documented negative result, and because the table itself is sound -
it may be useful for something other than query expansion.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starter.agent import _stemmed, _text

MIN_DF = 20             # rarer than this and the statistics are noise
MAX_DF_SHARE = 0.20     # commoner than this and the term discriminates nothing
MIN_PAIR = 10
TERMS_PER_PRODUCT = 24  # caps the pair explosion; titles are short anyway
KEEP_PER_TERM = 6
PROBES = ("trainer", "sneaker", "chain", "top", "jumper", "handbag", "trouser", "necklac")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine catalog term associations")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--out", default="data/associations.json")
    parser.add_argument("--inspect", action="store_true", help="show probe terms and stop")
    args = parser.parse_args()

    # Category vocabulary lives in titles and categories; mining the full text
    # would drown product nouns in marketing copy.
    documents: list[set[str]] = []
    frequency: collections.Counter = collections.Counter()
    in_store: collections.Counter = collections.Counter()
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            terms = {t for t in _stemmed(_text(product.get("title")) + " "
                                         + _text(product.get("categories"))) if len(t) > 2}
            documents.append(terms)
            frequency.update(terms)
            in_store.update(set(_stemmed(_text(product.get("store")))))

    total = len(documents)
    # A term appearing mostly inside store names is a brand. Brands have very
    # high PMI with their product type - they co-occur perfectly - but are
    # useless as synonyms, so they are excluded rather than surfaced.
    brands = {t for t in frequency if in_store[t] >= 0.5 * frequency[t] and in_store[t] >= 5}
    vocabulary = {t for t, c in frequency.items()
                  if MIN_DF <= c <= total * MAX_DF_SHARE} - brands
    print(f"{total} products | {len(vocabulary)} terms kept | {len(brands)} brand-like terms dropped")

    pairs: collections.Counter = collections.Counter()
    for terms in documents:
        selected = sorted(terms & vocabulary)[:TERMS_PER_PRODUCT]
        pairs.update(itertools.combinations(selected, 2))

    associations: dict[str, list[tuple[float, str]]] = collections.defaultdict(list)
    for (left, right), count in pairs.items():
        if count < MIN_PAIR:
            continue
        pmi = math.log((count / total) / ((frequency[left] / total) * (frequency[right] / total)))
        if pmi > 0:
            associations[left].append((pmi, right))
            associations[right].append((pmi, left))
    table = {term: [w for _, w in sorted(values, reverse=True)[:KEEP_PER_TERM]]
             for term, values in associations.items()}

    print(f"terms with associations: {len(table)}")
    for probe in PROBES:
        print(f"   {probe:<10} -> {table.get(probe, ['(absent)'])}")
    if args.inspect:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
