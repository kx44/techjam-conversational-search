# How the agent got here

Onboarding notes. What we changed, what worked, what didn't, and what we learned
about the benchmark along the way. Read the first two sections to be useful; the
rest is reference.

**Current: `rerank` branch, 0.8922 on the public set**, from a 0.1067 starter
baseline. Run it with `python3 -m evaluator.local_evaluator`. Setup is in
`submission/README.md`; method and limitations in `submission/REPORT.md`.

## The pipeline in one screen

```
customer turn
  ├─ CLASSIFY     answered or declined? each clause matched against 36
  │               embedded prototype phrasings (BGE-small)
  ├─ ACCUMULATE   fold into a running query: terms, and phrases from words
  │               said next to each other in this message
  ├─ RETRIEVE     BM25 raw terms, BM25 Porter-stemmed terms, 500 each
  ├─ FUSE         weighted Reciprocal Rank Fusion, k=60 -> top 50
  ├─ RERANK       phrase 0.8, popularity 0.2, price 0.3
  └─ RESPOND      top 10, plus one question
```

Everything is per-session state; nothing is discarded between turns. Runs offline
on CPU, no network, no generative model, no credentials.

## The two things that mattered

Out of roughly twenty-five ideas measured, two produced most of the score.

**Asking a question (+0.477).** The starter baseline returned
`ask_attribute: None`, so the simulated customer had no reason to disclose
anything after its opening message and just repeated a prompt. Asking anything
at all turns a one-sentence brief into a multi-turn one.

**Reranking the fused head (+0.156).** Retrieval decides which products are
plausible; nothing decided their order. BM25 runs a disjunctive query, so a
product matching 2 of 12 query terms can outrank one matching 10, and phrases
are shattered into tokens. A second pass over the top 50 fixes rank, not recall.

Everything else combined is worth about +0.05.

## What worked, in order

| # | change | score | Δ |
|---|---|---|---|
| 0 | starter baseline — stateless OR-BM25 | 0.1067 | |
| 1 | Porter stemming + RRF fusion | 0.1210 | +0.014 |
| 2 | accumulate the whole conversation | 0.2201 | +0.099 |
| 3 | ask a clarifying question each turn | 0.6973 | +0.477 |
| 4 | BGE-small dense retrieval | 0.7022 | +0.005 |
| 5 | retrieval depth 100 → 500 | 0.7081 | +0.006 |
| 6 | rerank the fused head | 0.8638 | +0.156 |
| 7 | a declined question decays back into contention | 0.8760 | +0.012 |
| 8 | model used for intent only, not retrieval | 0.8802 | +0.004 |
| 9 | phrases from within-message adjacency | 0.8922 | +0.012 |

Notes on the less obvious ones:

**(5) Depth.** The fixed top-100 was justified on the wrong measurement. With the
full conversation BM25 has 100% recall at rank 100, but hits happen mid
conversation, and at turn 1 recall@100 is only 52% — in 93 of 200 sessions the
target sits beyond rank 100, median rank 279, scoring just 0.18 of the spread
below rank 100. It was cut by a hair.

**(7) Declined questions.** A question the customer declined was being treated as
one they had answered, and retired permanently. In `public_0035` that retired
`feature`, the only attribute unlocking three of that session's four constraints;
the remaining turns asked about attributes that unlocked nothing. Asking an
attribute now suppresses it, declining suppresses it harder, and suppression
decays 0.55 per turn so it returns later. Re-asking *immediately* is a wash —
it costs a turn and cancels its own gain. Deferring improves MTTC.

**(9) Phrases.** The reranker's dominant feature matched query bigrams against
product text, but the bigrams came from the accumulated term dictionary —
consecutive distinct stems in first-seen order. That loses real adjacency and
invents pairs spanning message boundaries. Taking bi- and trigrams from each
message as it arrives is simpler and what the feature always meant.

## What didn't work

Recorded because the negative results shaped the design more than the wins did,
and because several look obviously good on paper.

| idea | outcome and why |
|---|---|
| **fusion score as a rerank feature** | **−0.043**, the worst. The fused ordering is exactly what the reranker exists to correct, so passing it forward reimports the error |
| **term coverage** | −0.011. Retried three ways and stays dead. Not a stopword problem: expanding the list 31 → 171 words moved the total +0.001/−0.007. Not a length-bias problem either, though the bias is real (+0.41 correlation with product text length, vs phrase's −0.09) — normalising it out makes things worse, because the bias was standing in for popularity. It is saturated: among 50 already-relevant candidates almost all contain most of the common words |
| **RM3 pseudo-relevance feedback** | −0.013. Needs a mostly-relevant first pass; at 0.125 hit rate the feedback set is the wrong products and expansion amplifies the error. Re-tested after the first pass improved — still flat |
| **entropy-based question choice** | −0.125 against a fixed broad-to-narrow order. It optimises for separating *catalog values*, not for what a customer can answer, so it spends early turns on colour and material |
| **catalog-mined synonym table (PMI)** | −0.022. Polysemy: in a clothing catalog "trainer" means waist trainer, so it expands to cincher/shaper/corset. And the correct associations are lateral — handbag → satchel, clutch — which widens the query into neighbouring categories |
| **spaCy dependency parsing** | −0.026 on natural language. 82% accurate at deciding which noun is negated, but a misattached negation fires on the *wrong* noun: 40% of the time it discards the customer's real requirement against 34% correct, and the costs are asymmetric. Fully written up in `submission/REPORT.md` |
| **average rating** | −0.001 to −0.111, linear or banded. Real signal in aggregate (a random rival out-rates the target only 35% of the time) but useless at this resolution: within the top-50 pool the gap is 0.11 against a spread of 0.31 |
| **adaptive score-gap cutoffs** | −0.004 to −0.019 across relative-threshold, largest-gap and z-score variants. Plain deeper retrieval beat every one — RRF already discounts deep ranks, so carrying the tail is nearly free and there is no cut to decide |
| **CombSUM over normalised scores** | −0.006. Rank order is more robust than magnitude across retrievers whose scores are on incomparable scales |
| **dense retrieval at full weight** | −0.014. Embeddings find the right *kind* of product and cannot pick which one |
| **acting on intent override** | −0.044 blanket decay, 0.000 attribute retirement, −0.017 term suppression. Classification is fine (93% accuracy, override at 100% precision and recall) — the actions are wrong, because this evaluator's "earlier preference" is a decoy that was never true of the target |
| **title match, and title-phrase** | −0.004 to −0.019. Double-counts what the phrase feature already reads, and biases toward products whose title happens to contain the phrase |
| **BM25 field weights** | no change worth making. Ten configurations span 0.026 and flat 1.0 everywhere loses only 0.022. Boosting `features`/`details`, where the quoted constraint text lives, is *worse* |
| **categories-only retriever** | −0.001. Correlated with a column BM25 already weights at 4.0 |

A recurring theme: **a feature having information is not sufficient.** It must add
information the stronger features lack, at the resolution where the ranking
decision is actually made.

## What we learned about the benchmark

**The evaluator hands you the answer's category, verbatim.** The opening message
is built with `coarse_category(target.categories)` — the target's own category
string. `tools/category_harness.py` degrades it in steps:

| how the customer names the category | score |
|---|---|
| exactly as the evaluator gives it | 0.8922 |
| head noun only — "sneaker" | 0.7993 |
| an everyday word — "trainer" | 0.7410 |
| described, never named | 0.6811 |
| no category at all | 0.6595 |

That anchor is worth **0.22** — as much as the reranker and the question policy
combined. The private set is generated by the same code so it will be there too;
this measures the gap between the benchmark and a real shopper. Nothing we tried
closed it.

**Optimising against the reference evaluator alone is dangerous.** The
`bucket-filter` branch scores **0.9364** — higher than what we ship — by
exact-matching the evaluator's category key and template prefixes. It scores
**0.4388** on natural language, below the untouched baseline's 0.6655, and
**0.3079** under paraphrase. Every number since has been measured on two
harnesses for this reason.

**`ask_attribute: "other"` is a wildcard** in this simulator — it matches any
undisclosed constraint, and asking it every turn scores 0.7503 against our
0.6973 at the time. We do not use it: it exploits a quirk of that matching rule
and means nothing to a person.

**Hit rate is pinned.** It has sat at 0.94–0.97 across roughly twenty
configurations. Nothing added moves recall; everything only moves MRR.

## Test tooling

| tool | what it is for |
|---|---|
| `tools/realistic_sim.py` | a customer simulator written from scratch — no shared code, templates or vocabulary with the reference evaluator. Paraphrases metadata instead of quoting it, answers whichever topic was raised, volunteers information, corrects itself. **Use this on anything you change** |
| `tools/category_harness.py` | degrades the category anchor in six graduated steps and reports the curve |
| `tools/build_associations.py` | the PMI synonym miner, kept as a documented negative result; `--inspect` reproduces both failure modes |
| `tools/build_intent_prototypes.py` | builds the 56 KB prototype file the classifier needs |
| `tools/build_embeddings.py` | catalog embedding matrix — only needed if you set `USE_DENSE = True` |
| `tools/check_dense.py` | verifies the encoder itself: unit norms, CLS-vs-mean pooling, paraphrase neighbours |

## Branch map

| branch | what it is |
|---|---|
| **`rerank`** | **current. 0.8922 / 0.9389** |
| `main` | untouched starter kit, 0.1067 |
| `bucket-filter` | fitted to the reference simulator. 0.9364 official, 0.4388 natural. **Do not ship** — kept as the cautionary case |
| `robust-agent` | an independent rebuild assuming an unknown evaluator, 0.8104 / 0.8654 |
| `ir-classical`, `dense-bge`, `adaptive-cutoff`, `boundary-retry` | intermediate points in the current lineage |
| `intent-detector` | WIP. Classification works; no action on it helped |

## Open

- **Team contributions** for `submission/REPORT.md` — not written
- **Configuration choice**: ship with the model (0.8922, 128 MB, Python 3.12) or
  stdlib-only (0.8598, runs anywhere). Both documented, fallback is automatic
- **Boundary** is the weakest scenario at hit 0.800 against 0.965 overall, on 10
  sessions
- Remaining headroom is ~0.02–0.04 and the last several ideas all measured
  negative, which usually means a local optimum
