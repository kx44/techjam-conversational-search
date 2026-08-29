# Method, results and limitations

**Public-set score 0.8802**, from a 0.1067 baseline. No generative model, no
network, no credentials, no reported token usage. Median 38 ms per turn.

## Architecture

```
customer message
  │
  ├─ ACCUMULATE   skip if identical to an earlier turn; add terms to the
  │               running query; keep the raw sentences for the dense side
  │
  ├─ RETRIEVE     three views of one catalog, 500 candidates each
  │                 BM25 over raw tokens                weight 1.00
  │                 BM25 over Porter-stemmed tokens     weight 1.00
  │                 BGE-small cosine over 384-d vectors weight 0.25
  │
  ├─ FUSE         weighted Reciprocal Rank Fusion (k=60) -> top 50
  │
  ├─ RERANK       score those 50 on catalog evidence the OR query discards
  │                 phrase      0.8   query bigrams surviving in product text
  │                 popularity  0.2   log1p(rating_number)
  │                 price       0.3   fit against a stated budget
  │
  └─ RESPOND      top 10, plus one clarifying question
                  (broad first: feature, use_case, style, then material,
                   colour, size, budget; an attribute yielding nothing is retired)
```

Two design choices carry most of the result. **Conversation accumulation** — the
starter baseline re-queried from the latest message and discarded everything
prior. **Reranking** — fusion selects plausible products but never judges them,
and BM25's disjunctive query lets a product matching 2 of 12 terms outrank one
matching 10.

A third matters more than its size suggests. A declined question was being
treated as an answered one, so an attribute the customer waved away was retired
permanently. In one measured session that consumed `feature`, the only
attribute unlocking three of that session's four constraints, and the remaining
turns asked about attributes that unlocked nothing.

## Results

Reference evaluator, 200 public sessions:

```
hit@10 0.960    MRR 0.798    MTTC 2.96    efficiency 0.804    score 0.8802
```

| increment | score | Δ |
|---|---|---|
| starter baseline (stateless OR-BM25) | 0.1067 | |
| + Porter stemming, RRF fusion | 0.1210 | +0.014 |
| + conversation accumulation | 0.2201 | +0.099 |
| + clarifying question each turn | 0.6973 | +0.477 |
| + BGE-small dense retrieval | 0.7022 | +0.005 |
| + retrieval depth 100 → 500 | 0.7081 | +0.006 |
| + catalog reranking of the fused head | 0.8638 | +0.156 |
| + declined questions decay back into contention | 0.8760 | +0.012 |
| + embedding model used for intent only, not retrieval | **0.8802** | +0.004 |

By scenario:

| scenario | n | hit@10 | MRR | MTTC |
|---|---|---|---|---|
| browsing | 80 | 0.988 | 0.812 | 2.71 |
| buying | 80 | 0.950 | 0.783 | 2.42 |
| intent override | 30 | 0.967 | 0.831 | 4.03 |
| boundary | 10 | 0.800 | 0.716 | 4.60 |

## Models and cost

| | |
|---|---|
| generative model | **none** |
| embedding model | BAAI/bge-small-en-v1.5, 33M params, ONNX on CPU — **intent detection only** |
| API calls | **zero** |
| reported token usage | 0 prompt, 0 completion |
| estimated model cost | **$0.00** |
| median latency | 38 ms/turn (p95 91 ms), 113 ms/session |
| index build | 18.2 s, of which 17.7 s is the two FTS5 indexes |
| query encoding | 2 ms/turn |
| agent memory | 328 MB stdlib-only, 577 MB with the model |
| local assets | 128 MB model + 56 KB prototypes |

The model was bought for retrieval and earns its place in classification
instead. One encoder serves two consumers needing different artifacts:

| use | artifact | worth |
|---|---|---|
| dense product retrieval | 73 MB catalog matrix | **−0.004** |
| decline detection | 56 KB prototypes | **+0.020** |

Encoding the conversation and cosining it against 50,000 product vectors stopped
paying once reranking existed — the reranker absorbed what it contributed, and
its residual effect is promoting semantically-similar-but-wrong products into
the reranked head. Encoding each customer clause against 36 prototype sentences
pays, because nothing else distinguishes a decline from an answer: catalog
statistics cannot, the rarest new term in a decline having median IDF 1.34
against a reveal's 1.15, with no usable threshold.

So dense retrieval is off and its matrix is not built, which also removes an
18-minute precompute. It costs memory, not time: 0.5 s of start-up and 2 ms per
turn. It runs under `onnxruntime` on CPU with no PyTorch dependency; the
exported graph returns `last_hidden_state`, so CLS pooling and L2 normalisation
are done explicitly (BGE pools CLS, not mean — mean pooling this model produces
plausible-looking vectors that rank badly). `README.md` documents the
stdlib-only configuration, which drops the model for 0.020 of score.

## Validation beyond the reference evaluator

Every number was measured on **two independent harnesses**. `tools/realistic_sim.py`
is written from scratch — no shared code, templates or vocabulary with
`evaluator/local_evaluator.py`. Its customer paraphrases metadata instead of
quoting it, varies sentence structure, answers whichever topic the agent raised
(from `ask_attribute` *or* the message text), volunteers information unasked,
and corrects itself naturally.

This mattered. An earlier branch (`bucket-filter`) reached **0.9364** on the
reference evaluator by exact-matching its category key and template prefixes,
and scored **0.4388** on natural language — below the untouched baseline's
0.6655. The submitted design is the one that survives both.

Degradation under paraphrase:

| harness | score |
|---|---|
| reference evaluator | 0.8802 |
| conversational frame reworded | 0.8307 |
| `bucket-filter`, same treatment | 0.9364 → **0.3079** |

## Limitations

**The score depends heavily on one sentence.** The evaluator builds its opening
message with `coarse_category(target.categories)` — the target's own category
string, verbatim — giving BM25 an exact-match anchor lifted from the document it
must find. `tools/category_harness.py` degrades that anchor in steps:

| customer's wording | score |
|---|---|
| exact, as the evaluator gives it | 0.8802 |
| head noun only ("sneaker") | 0.7993 |
| everyday synonym ("trainer") | 0.7410 |
| described, never named | 0.6811 |
| no category at all | 0.6595 |

That anchor is worth **0.22**, comparable to the reranker and the clarifying
question combined. The private set is generated by the same code, so it will be
present there too — this measures the gap between the benchmark and a real
deployment, not a scoring risk. We could not close it: dense retrieval is worth
−0.003 at the synonym level, and a catalog-mined synonym table is worse still.

**Boundary is still the weakest scenario** — hit 0.800 against 0.960 overall,
on 10 sessions, though the decay policy recovered 4 of the 6 it was missing.

**Efficiency is near its floor.** Override sessions cannot convert before turn
3 by construction, capping MTTC around 1.375 and efficiency around 0.96.

**Estimates for the private set.** Held-out sets of 800 targets drawn from the
same catalog, sampled to match the public set's popularity distribution, scored
0.907–0.917 against 0.936 public on the branch measured at the time — roughly
0.02 of optimism from sampling alone.

## What we tried that did not work

Recorded because the negative results shaped the design.

| idea | outcome |
|---|---|
| RM3 pseudo-relevance feedback | −0.013; needs a good first pass, and at 0.125 hit rate the feedback set is the wrong products |
| term coverage in the reranker | −0.011; saturated among 50 already-relevant candidates. Not a stopword problem (31→171 words changed +0.001) and not a length-bias problem (correcting the +0.41 length correlation made it worse) |
| title match, and title-phrase | −0.004 to −0.019; double-counts what `phrase` already reads |
| average rating | −0.001 to −0.111; real signal in aggregate but the target-vs-pool gap is 0.11 against a spread of 0.31 |
| fusion score as a rerank prior | −0.043; RRF's ordering is what the reranker exists to correct |
| dense retrieval at full weight | −0.014; finds the right kind of product, cannot pick which one |
| adaptive score-gap cutoffs | −0.004 to −0.019; plain deeper retrieval beat every variant |
| CombSUM over normalised scores | −0.006; rank order is more robust across incomparable score scales |
| semantic intent detection, most actions | classification works (93% accuracy, override at 100% precision and recall) but only one action on it helped. Blanket override decay −0.044, attribute retirement 0.000, term suppression −0.017 message-level and −0.013 clause-level. Letting a *declined* attribute decay back into contention is the exception, at +0.012 |
| re-asking a declined attribute at once | −0.002; it recovers the same boundary sessions but spends a turn doing it, and MTTC rises 2.98 → 3.21, cancelling the gain. Deferring the re-ask instead improves MTTC |
| BM25 field weights | unchanged; ten configurations span 0.026 and flat 1.0 everywhere loses only 0.022. Boosting features and details, where the quoted constraint text lives, is worse |
| catalog-mined synonym expansion | −0.022; polysemy — in a clothing catalog "trainer" means waist trainer |
| categories-only retriever | −0.001; correlated with a column BM25 already weights |

A consistent theme: a feature having information is not sufficient. It must add
information the stronger features lack, at the resolution where the ranking
decision is actually made.
