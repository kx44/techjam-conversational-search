# Pipeline context

Paste-in context for a fresh session. Dense by design. Companion docs:
`ITERATIONS.md` (narrative, for humans), `submission/REPORT.md` (method and
limitations), `submission/README.md` (setup).

## 1. Task and scoring

Multi-turn agent finds one hidden product in a frozen 50,000-item
`Clothing, Shoes & Jewelry` catalog, ≤10 turns, returning ≤10 `parent_asin`
per turn. Session ends the moment the target appears in the returned list.

```
Efficiency     = clip((11 - MTTC) / 10, 0, 1)      misses count as turn 11
TechnicalScore = 0.50*HitRate@10 + 0.30*MRR + 0.20*Efficiency
```

Hits are **exact string equality** on `parent_asin`. No LLM judge. The
`message` field is never read by the reference evaluator; `usage` tokens are
reported but excluded from scoring.

**Current: branch `constraint-state`, 0.8770 reference.** Baseline was 0.1067.

This branch is `override-retraction` merged with `origin/negation`. Both lines
maintain a structured model of what the customer currently wants, so they are
one subsystem, not two - see §11. **The negation logic itself is not settled**;
what is settled is where it plugs in.

## 2. Interface (fixed by `docs/agent_api_contract.json`)

```python
class Agent:
    def __init__(self, catalog_path="data/catalog.jsonl") -> None
    def reset(self, session_id: str, user_profile: dict) -> None
    def respond(self, session_id, user_message, turn, top_k) -> dict
        # {"message": str,                     <- must be str or the whole
        #  "ask_attribute": <enum|None>,        #   response is discarded
        #  "recommendations": [{"parent_asin": str}, ...],
        #  "usage": {"prompt_tokens": int, "completion_tokens": int}}
```

`ask_attribute` enum: `category, material, color, size, style, brand, budget,
feature, use_case, other`, or `null`. Anything else is coerced to `"other"`.

**The evaluator swallows agent exceptions into an empty response**
(`local_evaluator.py:241`). A crashing agent scores zero silently. Always wrap
`respond` and count exceptions when measuring.

## 3. How the reference customer works — read this before changing anything

`evaluator/local_evaluator.py` is a deterministic template machine. It never
reads your `message`; only `ask_attribute` drives it.

**Intent cards are generated at runtime** from the target's own metadata
(`materialize_hidden_fields` → `intent_card`): flattens `features` + `details`,
prepends a regex-matched material and `color: X`, appends `budget around $price`,
dedups, truncates each to 180 chars. First two → `hard_constraints`, next two →
`soft_preferences`. **≤4 constraints exist per session.**

**Opening message** (`initial_message`) interpolates
`coarse_category(target.categories)` — the target's own category string,
verbatim. Four scenario shapes:

| scenario | share | opener |
|---|---|---|
| buying | 40% | `I'm looking for {cat}. A key requirement is: {hard[0]}.` |
| browsing | 40% | `I'm looking for {cat}, but I'm still exploring.` |
| intent_override | 15% | `I'm looking for {cat}. {soft[-1]}` — the tail is a **decoy**, never true of the target; hits do not count until the override fires at turn 3 or 4 |
| boundary | 5% | same as browsing; **refuses the first attribute asked**, whatever it is |

**Replies** (`customer_reply`), in precedence order:
1. boundary refusal, once per session
2. `ask_attribute` falsy → a nag, zero information
3. up to **2** undisclosed constraints whose `classify_constraint(value)` equals
   the asked attribute → `For that, what matters is: a; b.`
4. otherwise `I don't have an additional preference for {attr}.`

`classify_constraint` is a keyword cascade returning only
`budget → material → color → size → style → use_case → feature`.
**`category` and `brand` are never produced** — asking them always yields
nothing.

## 4. Architecture

```
starter/agent.py      the agent
starter/stemmer.py    vendored Porter stemmer, stdlib
starter/intent.py     TWO prototype classifiers over one encoder:
                        IntentDetector            NORMAL/OVERRIDE/NO_PREFERENCE
                                                  whole message, drives override
                                                  and decline detection
                        ClausePreferenceClassifier ACCEPT/REJECT/NEUTRAL per
                                                  (attribute, value) mention,
                                                  drives negation
starter/dense.py      ONNX encoder + optional dense index
```

Per turn, `respond()`:

```
_accumulate(state, msg)     dedupe on exact text; suppression update;
                            OVERRIDE? → _retract: record superseded values;
                            record stated values in state["values"];
                            split into clauses; per clause:
                              mentions? → score each (attribute, value) once;
                                          HARD_REJECT anywhere disqualifies the
                                          clause and bans the value;
                                          otherwise the clause is evidence
                              no mentions and declined? → drop it
                            _rebuild_positive_state: THE single place that
                            decides what reaches retrieval - clears and rebuilds
                            plain / stems / phrases / text / budget from
                            positive_clauses, dropping rejected clauses and
                            excising superseded values
_search("products", ...)    BM25 raw,     RETRIEVE=500
_search("products_stem",..) BM25 stemmed, RETRIEVE=500
_fuse(...)                  weighted RRF, RRF_K=60 → list[(pid, score)]
_rerank(state, fused, k)    top RERANK_DEPTH=50 rescored, tail appended
_reply(state, ranked)       top_k + _choose(state)
```

Constants that matter:

```python
RETRIEVE        = 500        # not 100 — see §6
RRF_K           = 60
RERANK_DEPTH    = 50
RERANK_WEIGHTS  = {"phrase": 0.8, "popularity": 0.2, "price": 0.3}
USE_DENSE       = False      # dense *retrieval* off; model still loads
USE_EXPANSION   = False      # RM3, measured worse twice
ATTRIBUTE_DECAY = 0.55
FRESH_COST      = 0.25
DECLINE_PENALTY = 3.0
BM25_WEIGHTS    = "0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0"   # swept, left as found
```

Session state keys: `seen, plain, stems, phrases, text, asked, retired,
last_ask, size, budget, suppress`, plus the constraint state added by this
merge: `positive_clauses, rejected, superseded, values, preferences`.

**`plain`, `stems`, `phrases` and `text` are derived, not accumulated.** They
are cleared and rebuilt from `positive_clauses` on every turn. Anything that
edits them directly is erased by the next rebuild - see §11.

Index state: `_doc` (stemmed text per product, space-padded for word-boundary
`" term "` tests), `_pop` (`log1p(rating_number)`), `_price`, `_df` (IDF cache),
`_cache` (per-agent result cache keyed on the query term tuple), `_intents`.

**Reranker** scores `fused[:50]` on:
`0.8*phrase + 0.2*popularity [+ 0.3*price_fit if budget known]`, where
`phrase = |query phrases present in _doc[pid]| / |query phrases|`. The fusion
score is deliberately **not** used — see §7.

**Question policy** `_choose`: `min(ATTRIBUTE_ORDER, key=(suppress.get(a,
FRESH_COST), order_index))`. Asking sets suppression 1.0, a detected decline
sets 3.0, everything decays ×0.55 per turn. Order is broad→narrow:
`feature, use_case, style, material, color, size, budget, brand, category`.
**`"other"` is never emitted** — see §8.

**Intent classification** (`starter/intent.py`): 36 prototype sentences (12
each for NORMAL / OVERRIDE / NO_PREFERENCE) embedded with BGE-small; each
customer clause is embedded and matched by cosine to the nearest prototype;
`< 0.60` similarity or `< 0.02` margin → UNKNOWN. **Only `NO_PREFERENCE` has
behaviour**; OVERRIDE and UNKNOWN are treated as NORMAL. 93% accuracy on 486
labelled harness messages, OVERRIDE at 100% precision and recall.

## 5. Running and measuring

```bash
.venv/bin/python -m evaluator.local_evaluator      # reference, 200 sessions
.venv/bin/python tools/realistic_sim.py            # independent harness
.venv/bin/python tools/category_harness.py         # category-dependency curve
```

`tools/realistic_sim.py` shares no code, templates or vocabulary with the
reference evaluator. **Measure every change on both.** Python 3.12 venv
(`onnxruntime, tokenizers, numpy`); the stdlib-only fallback runs on 3.9+.

## 6. What worked

| change | score | Δ |
|---|---|---|
| baseline (stateless OR-BM25) | 0.1067 | |
| Porter stemming + RRF | 0.1210 | +0.014 |
| accumulate the conversation | 0.2201 | +0.099 |
| ask a clarifying question | 0.6973 | **+0.477** |
| dense retrieval | 0.7022 | +0.005 |
| depth 100 → 500 | 0.7081 | +0.006 |
| rerank the fused head | 0.8638 | **+0.156** |
| declined question decays back | 0.8760 | +0.012 |
| model for intent only | 0.8802 | +0.004 |
| phrases from within-message adjacency | 0.8922 | +0.012 |

Depth: turn-1 BM25 recall@100 is only 52% (93/200 targets beyond rank 100,
median 279, scoring 0.18 of the spread below rank 100). Recall@500 ≈ 86%.

## 7. Do not retry — all measured, all negative

| idea | Δ | why it fails |
|---|---|---|
| fusion score as a rerank feature | −0.043 | RRF's ordering is what the reranker corrects; feeding it back reimports the error |
| term coverage | −0.011 | saturated among 50 already-relevant candidates. Not stopwords (31→171 words: +0.001/−0.007). Not length bias (+0.41 corr. with doc length; normalising it out is *worse*, the bias proxied popularity) |
| entropy-based question choice | −0.125 | optimises separation of catalog values, not answerability |
| RM3 pseudo-relevance feedback | −0.013 | needs a relevant first pass; retested after retrieval improved, still flat |
| PMI synonym expansion from catalog | −0.022 | polysemy ("trainer" → cincher/corset in a clothing catalog); correct associations are lateral siblings, not synonyms |
| spaCy dependency parsing | −0.026 | 82% negation-attachment accuracy, but misattachment fires on the *wrong* noun: 40% drops the true requirement vs 34% correct, and costs are asymmetric |
| average rating (linear or banded) | −0.001..−0.111 | target-vs-pool gap 0.11 against spread 0.31 |
| adaptive score-gap cutoffs (3 variants) | −0.004..−0.019 | RRF already discounts deep ranks; plain depth beats every cutoff rule |
| CombSUM over normalised scores | −0.006 | rank order more robust across incomparable score scales |
| dense retrieval at weight ≥0.5 | −0.014 | finds the right *kind* of product, cannot pick which |
| acting on OVERRIDE, as blanket decay / attribute retirement / term suppression | −0.044, 0.000, −0.017 | this evaluator's "earlier preference" is a decoy; decaying evidence discards genuine constraints. **Superseded by the same-attribute rule now shipped** — see §11 |
| title match / title-phrase | −0.004..−0.019 | double-counts what `phrase` already reads (`_doc` includes the title) |
| categories-only retriever | −0.001 | correlated with a column BM25 already weights 4.0 |
| BM25 field-weight retuning | 0.000 | 10 configs span 0.026; flat 1.0 loses only 0.022; boosting `features`/`details` is *worse* |
| suppressing terms on a decline | −0.013..−0.187 | the boilerplate was inert; message-level suppression also discards volunteered content |

**Recurring principle:** a feature having information is insufficient. It must
add information the stronger features lack, at the resolution where the ranking
decision is made.

## 8. Benchmark properties

**The category anchor is worth 0.22.** The opener contains
`coarse_category(target.categories)` verbatim. Degrading it:
`exact 0.8922 → head noun 0.7993 → synonym 0.7410 → described 0.6811 → absent
0.6595`. The private set is generated by the same code, so it will be present;
this measures benchmark-vs-reality, not scoring risk. Four approaches failed to
close it.

**`ask_attribute: "other"` is a wildcard** — `customer_reply` matches it against
*any* undisclosed constraint. Asking it every turn scored 0.7503 vs 0.6973 at
the time. **Deliberately unused**: it exploits that matching rule and is
meaningless to a person.

**Over-fitting is measurable and severe.** Branch `bucket-filter` scores 0.9364
public — higher than what we ship — by exact-matching `coarse_category` output
and template prefixes. It scores 0.4388 on natural language (below the 0.6655
baseline) and 0.3079 under paraphrase. This is why everything is dual-measured.

**Hit rate is pinned** at 0.94–0.97 across ~20 configurations. Remaining
headroom ≈ 0.02–0.04, mostly MRR at ranks 2–9.

## 9. Invariants and gotchas

- `_accumulate` dedupes on exact message text but **still advances suppression**
  before returning; moving that check breaks the question policy on repeats.
- `_doc[pid]` is space-padded so `f" {term} "` is a word-boundary test. Losing
  the padding silently enables substring matches (`red` in `requi**red**`).
- FTS5 `bm25()` returns **negative** scores; ascending `ORDER BY` is best-first.
  Adding `DESC` inverts the ranking silently.
- BGE pools the **CLS token**, not the mean, then L2-normalises. Mean pooling
  produces plausible-looking vectors that rank badly.
- Model/artifact absence must degrade, never raise: `_load_model` catches
  everything and the agent scores 0.8598 with no model at all.
- Every parameter is tuned on 200 sessions. Where a sweep is flat we ship the
  midpoint, not the argmax.

## 10. Open

- team contributions for `submission/REPORT.md` — unwritten
- config choice: model (0.8922, 128 MB, Python 3.12) vs stdlib-only (0.8598,
  runs anywhere); both documented, fallback automatic
- boundary is the weakest scenario, hit 0.800 vs 0.965 overall, n=10
- capabilities that are real but invisible to this benchmark: negation
  handling, genuine intent-override handling, conversation quality

## 11. Constraint state (the merge)

`constraint-state` = `override-retraction` + `origin/negation`. Negation,
override and retraction are the same subsystem: writes to one structured model
of what the customer currently wants.

```
state["values"]      attribute -> everything stated       (so _retract knows
                                                           what to supersede)
state["rejected"]    attribute -> ruled out               (negation)
state["superseded"]  attribute -> replaced by a later turn (override)
state["preferences"] (attribute, value) -> PreferenceSignal, boosts the query
state["positive_clauses"]  the evidence the query is rebuilt from
```

**The one invariant that matters.** `_rebuild_positive_state` is the single
place that decides what reaches retrieval. It clears `plain`, `stems`,
`phrases` and `text` and rebuilds them from `positive_clauses` every turn. So
**a constraint must be state that the rebuild consults, never an edit applied
to the query.**

This was the entire difficulty of the merge. `override-retraction` originally
implemented retraction as `_drop_terms`, deleting the stale value out of those
four structures in place. Against a rebuild that is erased on the next turn:
the retraction appears to work on the turn it happens and silently stops
afterwards. It now records `state["superseded"]` and `_strip()` excises those
values *during* the rebuild, which is idempotent.

Excision rather than dropping the clause, because the clause carries the
category anchor and that anchor is worth 0.22 (§8) - far more than a stale
value costs.

### Not yet decided: the negation logic

The classifier merged from `origin/negation` abstains on most real negations.
`nothing in silver` scores REJECT 0.924 against ACCEPT 0.910 - the right class
wins, but by 0.0143 against `CLAUSE_MIN_MARGIN = 0.025`, so the verdict is
UNKNOWN and the value stays in the query.

The cause is `relation_text()`. Every prototype and every query share the same
scaffold (`Clause: "..."\nPair: color=[VALUE]\nDoes the user want this
value?`), plus the prototypes carry a trailing answer the query never has. The
boilerplate dominates the embedding, all similarities pile up at 0.89-0.99, and
the discriminative signal is worth ~0.01-0.05 of margin.

| option | cost |
|---|---|
| `CLAUSE_MIN_MARGIN = 0.010` | one character; unblocks the common cases, leaves the signal compressed |
| drop the scaffold, mask into the natural clause | prototypes and queries become ordinary English; separates at 0.85-1.00 vs 0.65 |
| port the two-layer parser from `backupnegation` | cue-and-scope + pair-anchored relation, open- and closed-vocabulary both covered |

**The plug-in point is settled and none of these change it**: whatever decides
a rejection writes `state["rejected"]`, and the rebuild does the rest.
