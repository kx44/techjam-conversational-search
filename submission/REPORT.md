# Method, results and limitations

**Public-set score 0.8864**, from a 0.1067 baseline. No generative model, no
network, no credentials, no reported token usage. Median 38 ms per turn.

## Architecture

```
customer message
  │
  ├─ CLASSIFY     did the customer answer the question, or decline it?
  │               each clause embedded and matched to prototype sentences
  │
  ├─ ACCUMULATE   skip if identical to an earlier turn; add terms to the
  │               running query
  │
  ├─ RETRIEVE     three views of one catalog, 500 candidates each
  │                 BM25 over raw tokens                weight 1.00
  │                 BM25 over Porter-stemmed tokens     weight 1.00
  │                 BGE-small cosine over 384-d vectors weight 1.00
  │
  ├─ FUSE         weighted Reciprocal Rank Fusion (k=60) -> top 50
  │
  ├─ RERANK       score those 50 on catalog evidence the OR query discards
  │                 phrase      0.8   query bigrams surviving in product text
  │                 popularity  0.2   log1p(rating_number)
  │                 price       0.3   fit against a stated budget
  │
  └─ RESPOND      top 10, plus one clarifying question
                  broad first: feature, use_case, style, then material,
                  colour, size, budget. Asking an attribute suppresses it,
                  declining suppresses it harder, and suppression decays each
                  turn - so a declined question returns to contention later.
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
hit@10 0.965    MRR 0.798    MTTC 2.78    efficiency 0.822    score 0.8864
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
| + embedding model used for intent only, not retrieval | 0.8802 | +0.004 |
| + phrases taken from within-message adjacency | 0.8922 | +0.012 |
| + dense retrieval re-enabled at equal weight | **0.8864** | −0.006 |

By scenario:

| scenario | n | hit@10 | MRR | MTTC |
|---|---|---|---|---|
| browsing | 80 | 0.988 | 0.824 | 2.59 |
| buying | 80 | 0.950 | 0.752 | 2.34 |
| intent override | 30 | 0.967 | 0.868 | 3.60 |
| boundary | 10 | 0.900 | 0.821 | 3.70 |

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

| use | artifact | public set | realistic shopper |
|---|---|---|---|
| dense product retrieval | 73 MB catalog matrix | **−0.006** | **+0.033** |
| decline detection | 56 KB prototypes | **+0.020** | +0.020 |

Encoding the conversation and cosining it against 50,000 product vectors stopped
paying once reranking existed — the reranker absorbed what it contributed, and
its residual effect is promoting semantically-similar-but-wrong products into
the reranked head. Encoding each customer clause against 36 prototype sentences
pays, because nothing else distinguishes a decline from an answer: catalog
statistics cannot, the rarest new term in a decline having median IDF 1.34
against a reveal's 1.15, with no usable threshold.

Dense retrieval was off until a third harness existed, and that reversal is
worth stating plainly. `tools/realistic_sim.py` and the reference evaluator both
have customers who quote catalog text close to verbatim - the regime where
lexical matching is strongest and embeddings add least - so both measured dense
retrieval as slightly negative. `tools/shopper_sim.py` models a shopper who
names things in their own words, and there dense retrieval is the only change
that has ever moved hit rate off its ceiling: 0.935 to 0.985.

| dense weight | reference | realistic_sim | shopper_sim |
|---|---|---|---|
| 0.00 | 0.8922 | 0.9389 | 0.8673 |
| 0.25 | 0.8864 | 0.9358 | 0.8778 |
| 1.00 | 0.8864 | 0.9346 | **0.9006** |
| 2.50 | 0.8512 | 0.9236 | 0.8790 |

The cost is binary rather than proportional: nearly all of it is the step from
0.00 to 0.25, and the reference score is flat from 0.25 to 1.00 while the
realistic shopper gains 0.023. A quarter weight is therefore strictly dominated.
We ship it at 1.00, paying 0.006 on the public set - roughly one session in 200,
inside the noise band we decline to chase elsewhere - for 0.033 on the harness
that best models a real customer, and for the robustness that buys if the
private harness paraphrases at all. It costs memory, not time: 0.5 s of start-up and 2 ms per
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
| reference evaluator | 0.8922 |
| conversational frame reworded | 0.8307 |
| `bucket-filter`, same treatment | 0.9364 → **0.3079** |

## Limitations

**The score depends heavily on one sentence.** The evaluator builds its opening
message with `coarse_category(target.categories)` — the target's own category
string, verbatim — giving BM25 an exact-match anchor lifted from the document it
must find. `tools/category_harness.py` degrades that anchor in steps:

| customer's wording | score |
|---|---|
| exact, as the evaluator gives it | 0.8922 |
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
| spaCy for negation, modality and attribute structure | see below — evaluated at length, not adopted |

A consistent theme: a feature having information is not sufficient. It must add
information the stronger features lack, at the resolution where the ranking
decision is actually made.


## Why there is no linguistic parser in the pipeline

A bag of terms cannot express polarity or modality, and the failure is silent
and in the wrong direction. Asked for `Men's Shoes, must be leather, not
suede`, the agent puts `suede` in the query as positive evidence and returns
*PUMA Men's Suede Striped* and *FRYE Men's Phillip Suede Oxford* in the top
five. The customer's exclusion becomes a recommendation.

spaCy was evaluated for exactly this, plus modality ("must" vs "would be
nice"), no-preference detection, intent override and attribute extraction,
inserted before retrieval. It was built, debugged over several iterations,
measured, and removed.

**It is genuinely good at the core task.** Isolating negation *attachment* -
which noun does the parser say is negated - gives 82% on single-clause
utterances, with four of six constructions perfect:

| construction | correct |
|---|---|
| `Not suede - I need leather instead.` | 25/25 |
| `No suede, please.` | 50/50 |
| `...but avoid suede` | 25/25 |
| `...without any suede` | 25/25 |
| `must be leather, not suede` | 79/100 |
| `Anything but suede` | 0/25 |

**Reading clause by clause fixed the integration.** Parsing whole messages let
negation leak across coordination - in "not suede, and ideally black", `black`
is a conjunct of `suede` and was wrongly marked negative - and discarded turns
that decline one attribute while volunteering another. Per-clause reading fixed
both, and made the layer harmless on the shipping harnesses where the
whole-message version had cost −0.005:

| | official | natural language |
|---|---|---|
| shipped | 0.8922 | 0.9389 |
| + clause-scoped negation | 0.8922 | 0.9385 |

**It still fails, because the errors are asymmetric.** Across a naturalness
gradient holding the conveyed facts constant and varying only the wording:

| customer language | control hit/MRR | with parsing | Δ |
|---|---|---|---|
| reference template | 0.435 / 0.326 | 0.435 / 0.326 | +0.000 |
| reworded frame | 0.461 / 0.331 | 0.461 / 0.331 | +0.000 |
| plain natural | 0.391 / 0.296 | 0.391 / 0.296 | +0.000 |
| natural, with polarity | 0.426 / 0.305 | 0.409 / 0.288 | **−0.017** |
| very natural | 0.443 / 0.325 | 0.417 / 0.318 | **−0.026** |

It gets *worse* as language gets more natural - the opposite of the intended
effect. On those richer sentences the parse degrades, and when it misattaches
it does not merely fail to fire, it fires on the wrong noun:

| what the layer does on a negation | share |
|---|---|
| drops the true value | **40%** |
| drops the decoy (correct) | 34% |
| drops neither | 25% |

The costs are not symmetric. Keeping a decoy adds one noisy term to a query of
twenty; dropping the true value removes the term that identifies the target. A
34/40 split is therefore well net negative, which is what the gradient shows.

**A bigger model does not fix it.** `en_core_web_md` scores 76% attachment
against `sm`'s 82% - it differs only in word vectors, not parser architecture.
Only `en_core_web_trf` would plausibly clear the bar, at ~400 MB plus PyTorch,
which exceeds the entire current asset budget for a capability neither harness
exercises.

**Where a parser would earn its place**: mixed polarity in one utterance,
mixed modality ("must be waterproof, ideally black, definitely not leather"),
comparatives, quantities beyond price, and correcting one attribute among
several. All are compositional - the meaning depends on how the words relate,
not which are present. The reference simulator emits either a verbatim catalog
string or one of six fixed templates, and produces none of them: every content
word it generates is a positive, equal-weight assertion, which is precisely the
regime where a bag of terms is adequate.

So the representation is matched to this benchmark's language, and would
degrade silently rather than loudly on language the benchmark does not produce.
If the private harness paraphrases into anything more conversational, negation
is the first thing that would bite - and on this evidence a parser is not yet
the fix.
