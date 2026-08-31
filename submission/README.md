# Submission — setup and how to run

Conversational retrieval agent for the TechJam Conversational E-Commerce Search
Challenge. **No generative model, no network, no credentials.** See `REPORT.md`
for method, results and limitations.

## What constitutes the submission

| file | role |
|---|---|
| `starter/agent.py` | entry file, exports `Agent` |
| `starter/stemmer.py` | vendored Porter stemmer (stdlib) |
| `starter/intent.py` | intent detection over the embedding model |
| `starter/dense.py` | ONNX encoder and vector search |
| `requirements.txt` | dependency manifest |
| `tools/` | test harnesses, not used at inference |

The agent is left at `starter/agent.py` because `evaluator/local_evaluator.py`
imports `from starter.agent import Agent`; moving it would break the official
harness.

## Two supported configurations

Selected by one module flag. **Both are offline.** Configuration A is the
default and the recommendation; B is a genuine fallback rather than a
degradation path, and needs no install step of any kind.

|  | A: with the embedding model | B: stdlib only |
|---|---|---|
| public-set score | **0.8894** | 0.8598 |
| natural-language harness | **0.9346** | 0.9297 |
| boundary sessions | **0.80** hit | 0.60 hit |
| Python | **verified on 3.13.5**; 3.12+ recommended | **3.9+, any** |
| dependencies | onnxruntime, tokenizers, numpy | **none — stdlib only** |
| local assets | 128 MB model + 73 MB matrix | **none** |
| agent memory | 577 MB | **328 MB** |
| setup | download the model and build the catalog matrix (~18 min) | **none** |

The model does two jobs: **intent/state detection** — no preference, override,
and clause-level accept/reject signals — and **dense product retrieval**. Dense
retrieval is conditional: it keeps full weight for short/paraphrased requests
and is down-weighted when the raw and stemmed lexical routes already agree. See
`REPORT.md` for the measured trade-off. `USE_DENSE = False` disables product
dense retrieval while keeping BGE state detection, which drops the 73 MB matrix
and its precompute.

## Configuration A — with the embedding model (default)

**Python 3.13.5 verified locally.** Python 3.12+ is recommended because the
BGE path depends on `onnxruntime`; Python 3.9 works only for the stdlib fallback
unless compatible wheels are available on the judging machine.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

# model: files from BAAI/bge-small-en-v1.5 into models/bge-small-en-v1.5/
#   onnx/model.onnx (127 MB), tokenizer.json

.venv/bin/python tools/build_embeddings.py               # ~18 min, writes 73 MB
.venv/bin/python -m evaluator.local_evaluator --output results.json
```

If the catalog matrix is absent the agent runs BM25-only and reports nothing —
check `agent._index is not None` before trusting a null result.

If the model is absent the agent falls back to configuration B automatically —
it does not fail. Verified: with `models/` removed it scores exactly 0.8598
with zero exceptions.

## Configuration B — stdlib only

Requires only a Python interpreter. No `pip install`, no assets, no network.

```bash
# in starter/agent.py set:  MODEL_DIR = ""      (or simply delete models/)
python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl \
                                     --dataset data/public_set.jsonl \
                                     --output results.json
```

Python 3.9 or newer. Verified on 3.9.6 and 3.12.13.

## Environment variables

**None.** No credentials, no API keys, no configuration.

## Network access

**Not required at any point during scoring.** The only network use is the
one-off model download for configuration A, done in advance. Configuration B
never touches the network.

## Reproducing the reported numbers

```bash
python3 -m evaluator.local_evaluator            # reference evaluator
python3 tools/realistic_sim.py                  # independent natural-language harness
python3 tools/shopper_sim.py                    # a shopper who talks around the product
python3 tools/category_harness.py               # category-dependency curve
```

`tools/realistic_sim.py` and `tools/shopper_sim.py` share no code, templates or
vocabulary with the reference evaluator. The robustness figures in `REPORT.md`
are reproducible from these commands.
