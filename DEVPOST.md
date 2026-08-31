#  ShopPal
## Your lightweight, reliable shopping kaki

**0.8894 TechnicalScore · 0.985 Hit@10 · 0.759 MRR · 38 ms median latency · 91 ms p95 · CPU-only · no inference-time API calls**

### The Problem
Conventional shopping copilots rely on heavy data-consuming architecture to process user input and understand user intent. For example, Amazon's copilot 'Rufus' relies on custom-trained LLMs on top of multiple foundation models routed dynamically. Such infrastructure design demands significant resources: AI tokens, electricity, time (latency)...


### Why Our Approach?
Our design prioritises minimalism: All stages in our algorithm is intentional and executes its intended role with no redundancy. This lightweight design makes our copilot low-latency, effective, and readily scalable.

#### 1. Technical Execution

The architecture is intentionally compact - each component has a measured role that fits uniquely into the logical flow of the pipeline. As the pipeline was designed, any components that made the system worse were immediately removed to optimise efficiency.

The system combines:

- clause-level semantic classification;
- explicit `NORMAL` / `NO_PREFERENCE` / `OVERRIDE` state transitions;
- raw BM25;
- Porter-stemmed BM25;
- dense cosine retrieval;
- conditional dense weighting;
- Reciprocal Rank Fusion;
- deterministic phrase/popularity/budget reranking;
- guarded semantic rescue;
- diagnostics-driven question selection.

Three independent retrieval routes each return 500 candidates. These candidates are combined and reranked using RRF, after which the top 100 fused candidates are passed to the deterministic reranker:

```text
RRF(score) = Σ weight / (60 + rank + 1)

weight = 1.0 raw BM25, 1.0 stemmed BM25,
         0.15 or 1.0 dense (conditional)

0.8 × phrase coverage
+ 0.2 × log1p(review count)
+ 0.3 × budget fit
```

The final architecture was hence selected through ablation rather than by adding techniques which would accumulate complexity.

| Measurement | Result |
|---|---:|
| Hit@10 | **0.985** |
| MRR | **0.759** |
| MTTC | **2.54** |
| Efficiency | **0.846** |
| TechnicalScore | **0.8894** |
| Median / turn | **38 ms** |
| p95 / turn | **91 ms** |

#### 2. Innovation & Problem Insight

When analysing the problem, we recognised that prioritising a high TechnicalScore using the deterministic template-driven evaluator provided to us could predispose us to designing a copilot overfit to an unrealistic environment.

This would create a system that may look good on paper but fail to actually tackle the problem statement and deliver meaningful value.

Therefore, we built our own customer simulators (realistic_sim.py and shopper_sim.py) to produce phrases in a less predictable manner than the deterministic evaluator. Our design was then developed after this implementation so that the copilot actually tackles the problem statement directly.

Our copilot actually performs better in the more realistic simulations, yielding better results than when run with the deterministic evaluator.

realistic_sim.py:
| Measurement | Result |
|---|---:|
| TechnicalScore | **0.935** |

shopper_sim.py:
| Measurement | Result |
|---|---:|
| TechnicalScore | **0.901** |


#### 3. Impact & Relevance

Mentioned in the previous section, the additional simulators were created to model more natural human language by real-world shoppers. To address ambiguity and complexity in real-world language, our system passes keywords through three different scoring systems:

(1) raw BM25: This scores the keyword for the case of an exact match

(2) Porter-stemmed BM25: This scores the keyword should the root word match (e.g run > running)

(3) dense cosine retrieval: This scores the keyword for semantic similarities, if there is a low score (match) on the first two scoring systems

This more flexible scoring system can help pick out less explicit keyword matches common in real-world human dialogue. This system is therefore also designed to be adapted for real-world usage instead of overfit to a simulated environment.

#### 4. Feasibility & Practicality

The implementation is designed to be deployable without a hosted inference stack. The entire shipped inference pipeline runs locally, with no hosted model or inference-time API dependency:

- **CPU-only**
- **38 ms median latency**
- **91 ms p95**
- **0 inference-time API calls**
- **no hosted vector database**
- **no generative model**
- in-memory catalog retrieval
- hard **10-turn** limit

The local embedding configuration uses `BAAI/bge-small-en-v1.5` through ONNX Runtime. This model is used for semantic classification and dense retrieval; it is not used to generate text.

There is also a genuine stdlib-only configuration with **0.8598 TechnicalScore** and 328 MB agent memory.

---

# Results

### Public development set — 200 sessions

The headline result is **0.8894 TechnicalScore**, achieved with a CPU-only local pipeline and zero inference-time API calls. The independent natural-language harness reaches **0.9346** for the embedding configuration and **0.9297** for the stdlib-only configuration, providing a second view of robustness beyond the official evaluator.

```text
Hit Rate@10     0.985
MRR             0.759
MTTC            2.54
Efficiency      0.846
TechnicalScore  0.8894
```

### Scenario Breakdown

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Browsing | 80 | 1.000 | 0.718 | 2.35 |
| Buying | 80 | 0.975 | 0.753 | 1.90 |
| Intent override | 30 | 1.000 | 0.881 | 4.00 |
| Boundary | 10 | 0.900 | 0.767 | 4.80 |

---

# Future Work

The current reranking parameters (see last part of (1)) are optimised for the one anonymised `user_profile` in this simulation.

In a real world setting, different shoppers will exhibit different buyer-behaviours. For each shopper, their buying history can be logged and updated dynamically to optimise their own reranking parameters over time.

This personalises the copilot over time to more accurately retrieve products based on their unique buying style.

---

# Tools and Data

**Dataset:** frozen 50,000-product Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` catalog plus 200 labeled public sessions.

**Development:** Python 3.13.5, VS Code, SQLite.

**Libraries:** SQLite FTS5, ONNX Runtime, Hugging Face `tokenizers`, NumPy, standard library.

**Models:** `BAAI/bge-small-en-v1.5`; no generative model.

**APIs:** none during inference; no hosted model or external API is required for scoring.

---

# Links

**GitHub:** https://github.com/kx44/techjam-conversational-search

**Demo:** https://youtu.be/BQ3X-T_hrKY
