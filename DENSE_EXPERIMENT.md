# Dense retrieval — an open question

This branch turns dense retrieval **on**. It is off on `pipeline`. The question
is whether our reason for turning it off survives contact with a harness we
did not write.

## What it does

A third retriever alongside the two BM25 ones. The whole accumulated
conversation is embedded with BGE-small and cosined against a precomputed
50,000 × 384 matrix of product vectors; the resulting ranking joins reciprocal
rank fusion at weight 0.25, against 1.0 for each BM25 retriever.

The embedding model is loaded either way — the intent classifier needs it. What
this branch adds is the **catalog matrix** and dense retrieval's vote in fusion.

## What we measured, and why that may not settle it

```
harness                          dense on   dense off     delta
reference evaluator                0.8864      0.8922   -0.0058
realistic_sim                      0.9358      0.9389   -0.0031
```

Dense retrieval was worth **+0.005 before the reranker existed** and is worth
**−0.004 after it**. Two things are going on:

**The reranker absorbed what it contributed.** Fusion picks plausible
candidates; the reranker decides their order. Dense was helping with the first
job at a time when nothing was doing the second.

**Its unique finds cannot reach the output.** Dense returns 500 candidates of
which **61% are ones BM25 never sees**, and it is the *only* retriever that
finds the target in **10 of 200 sessions**. But **zero** of those unique finds
reach the reranked top 50 — at weight 0.25 an item found by dense alone scores
`0.25/(60+rank)` against `2/(60+1)` for anything both BM25 retrievers found. It
is outvoted every time. Raising the weight is monotonically worse on our
harnesses (0.8638 at 0.25 → 0.8484 at 1.5): louder dense promotes
semantically-similar-but-wrong products into the reranked head.

**The caveat worth testing.** Both harnesses above have customers who quote
catalog text close to verbatim — the reference evaluator literally interpolates
the target's own feature bullets, and `realistic_sim` reuses the target's
vocabulary while varying the wording. That is the regime where exact lexical
matching is strongest and semantic matching has least to add. A customer who
paraphrases properly is the case dense retrieval exists for, and neither of our
harnesses produces one.

## Setup

Needs the catalog matrix, which is not in git:

```bash
python3 tools/build_embeddings.py --catalog data/catalog.jsonl   # ~18 min, 73 MB
```

Also needs `models/bge-small-en-v1.5/` — the same three files the intent
classifier uses, so if that already works you have them. **If the matrix is
missing the agent runs BM25-only and says nothing**, so check before trusting a
null result:

```bash
python3 -c "import sys; sys.path.insert(0,'.'); import starter.agent as A
print('dense:', A.Agent('data/catalog.jsonl')._index is not None)"
```

## Running the comparison

`tools/compare_dense.py` runs the same agent twice, dense on and off, holding
everything else constant:

```bash
python3 tools/compare_dense.py                          # our two harnesses
python3 tools/compare_dense.py --harness mysim          # plus yours
python3 tools/compare_dense.py --harness mysim --weights 0,.25,.5,1,2
```

To plug in your own simulator, expose one callable:

```python
def run(agent, samples, products, catalog_ids) -> dict:
    # call agent.reset(session_id, profile) then agent.respond(...) per turn
    # return {"score": float}   # "hit_rate_at_10" and "mrr" are shown if present
```

`tools/realistic_sim.py` is a worked example of that contract.

## What would be worth knowing

- Does dense still lose on a customer who **paraphrases rather than quotes**?
- Is **0.25 the wrong weight** for such a customer? Our sweep found it optimal,
  but only where lexical matching was already winning.
- Do those **10 sessions where dense is the sole finder** convert on your
  harness? If they do, the fusion weight is the thing to change, not the
  retriever.

If dense wins anywhere, that is a real result and this branch should merge. Our
measurement says it does not — on two harnesses that share an assumption we did
not choose deliberately.
