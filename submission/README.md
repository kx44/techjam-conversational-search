# Submission — setup and how to run

Conversational retrieval agent for the TechJam Conversational E-Commerce Search
Challenge. **No LLM, no network, no credentials.** See `REPORT.md` for method,
results and limitations.

## What constitutes the submission

| file | role |
|---|---|
| `starter/agent.py` | entry file, exports `Agent` |
| `starter/stemmer.py` | vendored Porter stemmer (stdlib) |
| `starter/dense.py` | optional dense retrieval (see below) |
| `requirements.txt` | dependency manifest |
| `tools/` | test harnesses, not used at inference |

The agent is left at `starter/agent.py` because `evaluator/local_evaluator.py`
imports `from starter.agent import Agent`; moving it would break the official
harness.

## Two supported configurations

The agent runs in either of two modes, selected by one module flag in
`starter/agent.py`. **Both are offline.** Configuration B is our recommendation
unless the organiser confirms the extra assets are acceptable.

|  | A: with dense retrieval | B: BM25 only |
|---|---|---|
| `USE_DENSE` | `True` (default) | `False` |
| public-set score | **0.8638** | 0.8598 |
| natural-language harness | 0.9262 | **0.9297** |
| Python | 3.12 | **3.9+, any** |
| dependencies | onnxruntime, tokenizers, numpy | **none — stdlib only** |
| local assets | 128 MB model + 73 MB matrix | **none** |
| setup | download model, 18 min precompute | **none** |

The 0.004 difference between them is within run-to-run noise, and B is ahead on
the natural-language harness. B needs no install step of any kind.

## Configuration B — BM25 only (recommended)

Requires only a Python interpreter. No `pip install`, no assets, no network.

```bash
# in starter/agent.py set:  USE_DENSE = False
python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl \
                                     --dataset data/public_set.jsonl \
                                     --output results.json
```

Python 3.9 or newer. Verified on 3.9.6 and 3.12.13.

## Configuration A — with dense retrieval

**Python 3.12 required** (onnxruntime provides no wheel for the system 3.9 on
this platform).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# model: 3 files from BAAI/bge-small-en-v1.5 into models/bge-small-en-v1.5/
#   onnx/model.onnx (127 MB), tokenizer.json, config.json

.venv/bin/python tools/build_embeddings.py --catalog data/catalog.jsonl   # ~18 min
.venv/bin/python -m evaluator.local_evaluator --output results.json
```

If the model or the embedding matrix is absent, the agent logs nothing and
falls back to configuration B automatically — it does not fail. This is
verified: with the artifacts deleted it scores exactly 0.8598 with zero
exceptions.

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
python3 tools/category_harness.py               # category-dependency curve
python3 tools/build_associations.py --inspect   # a documented negative result
```

`tools/realistic_sim.py` shares no code, templates or vocabulary with the
reference evaluator. Every figure in `REPORT.md` is reproducible from these
four commands.
