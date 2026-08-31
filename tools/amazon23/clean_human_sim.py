"""800 sessions in which the customer speaks in real buyers' own sentences.

Every other harness in this repo builds the customer's words out of the target
product's own listing. `evaluator/local_evaluator.py` hands back `features` and
`details` verbatim; `realistic_sim.py` keeps the listing's vocabulary and only
reorders it ("same vocabulary, different surface"); `wild_customer_sim.py`
roughens the form with a hand-written synonym table. In all three the customer
says words that, by construction, already sit in the document being retrieved -
98.6% of them, measured. A lexical retriever is handed most of the answer.

Here the customer's turns are **sentences real buyers wrote**, taken from the
review that person left for the product they actually bought last (Amazon
Reviews 2023, McAuley Lab; last-out Same-Distribution protocol inside the frozen
catalogue). Nothing is paraphrased and nothing is generated.

The brief excluded typos and acronyms, which real reviews are full of, so each
candidate sentence must pass:

  length      4-26 words - long enough to carry a constraint, short enough to
              be something a person types into a chat box
  acronym     no all-caps token whose lowercase form is absent from the
              dictionary. TTS, DD and IMO are out; emphatic JUST is not an
              acronym and stays; XS-4XL are sizes and stay
  spelling    every alphabetic token known to /usr/share/dict/words, the
              catalogue's own vocabulary, or the contraction list - so "isn't"
              passes and "badonadonk" does not

A sentence that fails falls back to the short mined form ("size small"), which
is clean by construction. Filtering the *source* rather than rewriting it keeps
the register human: the words are still the buyer's, just the subset of their
sentences that a spell-checker would leave alone.
"""
from __future__ import annotations

import argparse, json, random, re, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MAX_TURNS, TOP_K = 10, 10
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
CAPS = re.compile(r"\b[A-Z]{2,}\b")
SIZES = {"XS", "S", "M", "L", "XL", "XXL", "XXXL", "2XL", "3XL", "4XL"}
CONTRACTIONS = {
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "can't",
    "won't", "wouldn't", "couldn't", "shouldn't", "haven't", "hasn't", "hadn't",
    "it's", "that's", "there's", "he's", "she's", "let's", "what's", "here's",
    "i'm", "i've", "i'd", "i'll", "you're", "you've", "you'd", "you'll",
    "we're", "we've", "we'd", "we'll", "they're", "they've", "they'd", "they'll",
}
OPEN_BUYING = ("Hi, I'm looking for {cat}. {c}",
               "I need {cat}. {c}",
               "Looking for {cat} - {c}",
               "Hi, after some {cat}. {c}")
OPEN_BROWSING = ("Hi, I'm browsing for {cat}.",
                 "Looking for {cat}, not sure exactly what yet.",
                 "Thinking about {cat}, still deciding.",
                 "I need {cat} at some point, just looking for now.")
DECLINE = ("No preference there.", "I don't really mind.", "Either way is fine.",
           "That's up to you.", "It doesn't matter to me.")
NUDGE = ("None of those look right.", "Not really what I had in mind.",
         "Those aren't it, anything else?", "Close, but not quite.")


def build_vocabulary(catalog: str) -> set[str]:
    words = set()
    try:
        with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as fh:
            words = {w.strip().lower() for w in fh if w.strip()}
    except OSError:
        pass
    with open(catalog, encoding="utf-8") as fh:
        for line in fh:
            words.update(WORD.findall(line.lower()))
    return words | CONTRACTIONS


def is_clean(text: str, known: set[str]) -> bool:
    """A sentence a person wrote, minus the ones with typos or acronyms."""
    tokens = WORD.findall(text)
    if not (4 <= len(tokens) <= 26):
        return False
    for caps in CAPS.findall(text):
        if caps in SIZES:
            continue
        if caps.lower() not in known:      # genuine acronym, not emphasis
            return False
    for token in tokens:
        low = token.lower()
        if low in known or low.replace("'", "") in known:
            continue
        if low.rstrip("s") in known or (low.endswith("ed") and low[:-2] in known):
            continue
        return False
    return True


class CleanHumanShopper:
    """Discloses one real buyer's constraints, in that buyer's own sentences."""

    def __init__(self, sample: dict, known: set[str], rng: random.Random):
        self.rng, self.known = rng, known
        intent = sample["real_intent"]
        self.category = sample["intent_card"]["target_category"]
        self.scenario = sample["scenario_type"]
        self.pool = {a: list(e) for a, e in intent["disclosures"].items() if e}
        self.told: set[str] = set()
        self.turn = 0
        self.declined = 0
        self.override = (sample.get("behavior") or {}).get("override") or {}
        self.override_done = self.scenario != "intent_override"
        self.used_quote = 0
        self.used_short = 0
        # One sentence can back two different attributes; a buyer would not
        # repeat it word for word two turns apart.
        self.spoken: set[str] = set()

    @staticmethod
    def _normalise(text: str) -> str:
        return (text.replace("\u2019", "'").replace("\u2018", "'")
                    .replace("\u201c", '"').replace("\u201d", '"')
                    .replace("\u2013", "-").replace("\u2014", "-"))

    def _say(self, entry: dict) -> str:
        quote = self._normalise((entry.get("quote") or "").strip())
        if (quote and not entry.get("negated") and quote not in self.spoken
                and is_clean(quote, self.known)):
            self.spoken.add(quote)
            self.used_quote += 1
            return quote if quote[-1] in ".!?" else quote + "."
        self.used_short += 1
        short = self._normalise(entry.get("spoken") or entry["said"])
        return short[0].upper() + short[1:] + "."

    def _take(self, attribute: str) -> str | None:
        for entry in list(self.pool.get(attribute) or []):
            key = f"{attribute}:{entry['value']}".lower()
            self.pool[attribute].remove(entry)
            if key in self.told:
                continue
            self.told.add(key)
            return self._say(entry)
        return None

    def opening(self) -> str:
        if self.scenario == "intent_override" and self.override:
            return f"I'm looking for {self.category}. {self.override['old_value']}"
        if self.scenario == "buying":
            for attribute in ("use_case", "material", "color", "feature", "style"):
                lead = self._take(attribute)
                if lead:
                    return self.rng.choice(OPEN_BUYING).format(cat=self.category, c=lead)
        return self.rng.choice(OPEN_BROWSING).format(cat=self.category)

    def reply(self, asked: object) -> str:
        self.turn += 1
        if not self.override_done and self.turn + 1 >= int(self.override.get("turn", 3)):
            self.override_done = True
            return f"Actually, forget that. What I need is {self.override['new_value']}."
        attribute = asked if isinstance(asked, str) else None
        if not attribute:
            return self.rng.choice(NUDGE)
        if self.scenario == "boundary" and self.declined == 0:
            self.declined += 1
            return self.rng.choice(DECLINE)
        said = self._take(attribute)
        if said is None:                       # indirection: answer what is on their mind
            for other in ("use_case", "feature", "material", "color", "size", "style", "other"):
                said = self._take(other)
                if said:
                    break
        if said is None:
            self.declined += 1
            return self.rng.choice(DECLINE)
        return said


def run(agent, samples, catalog_ids, known, seed):
    from evaluator.local_evaluator import normalize_recommendations
    rows, exceptions, quotes, shorts = [], 0, 0, 0
    for index, sample in enumerate(samples):
        rng = random.Random(f"{seed}:{sample['sample_id']}")
        shopper = CleanHumanShopper(sample, known, rng)
        sid = f"clean_{index:04d}"
        agent.reset(sid, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        message, hit_turn, best_rank = shopper.opening(), None, None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(sid, message, turn, TOP_K)
                if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                    raise ValueError("malformed")
            except Exception:
                exceptions += 1
                response = {"ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if target in ranked:
                hit_turn, best_rank = turn, ranked.index(target) + 1
                break
            if turn == MAX_TURNS:
                break
            message = shopper.reply(response.get("ask_attribute"))
        quotes += shopper.used_quote
        shorts += shopper.used_short
        rows.append({"sample_id": sample["sample_id"], "scenario_type": sample["scenario_type"],
                     "hit": hit_turn is not None, "first_hit_turn": hit_turn,
                     "best_rank": best_rank,
                     "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank})
    return rows, exceptions, quotes, shorts


def summarise(rows):
    n = len(rows)
    hit = sum(r["hit"] for r in rows) / n
    mrr = statistics.fmean(r["reciprocal_rank"] for r in rows)
    mttc = statistics.fmean(r["first_hit_turn"] or MAX_TURNS + 1 for r in rows)
    eff = max(0.0, min(1.0, (11 - mttc) / 10))
    return {"n": n, "hit_rate_at_10": round(hit, 6), "mrr": round(mrr, 6),
            "mttc": round(mttc, 4), "efficiency": round(eff, 6),
            "technical_score": round(0.5 * hit + 0.3 * mrr + 0.2 * eff, 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--set", default="data/amazon23/real_set_800.jsonl")
    ap.add_argument("--sessions", type=int, default=800)
    ap.add_argument("--seed", type=int, default=800)
    ap.add_argument("--output", default="")
    ap.add_argument("--show", type=int, default=0, help="print N transcripts and exit")
    args = ap.parse_args()

    from evaluator.local_evaluator import catalog_index
    rows = [json.loads(l) for l in open(args.set, encoding="utf-8") if l.strip()]
    known = build_vocabulary(args.catalog)

    # Trim to the requested count, keeping the public set's scenario mix as
    # closely as the available pool allows.
    want = {"buying": .40, "browsing": .40, "intent_override": .15, "boundary": .05}
    pools = {k: [r for r in rows if r["scenario_type"] == k] for k in want}
    rng = random.Random(args.seed)
    for pool in pools.values():
        rng.shuffle(pool)
    picked, short = [], 0
    for name, share in want.items():
        take = min(round(args.sessions * share), len(pools[name]))
        picked += pools[name][:take]
        pools[name] = pools[name][take:]
        short += round(args.sessions * share) - take
    leftovers = [r for pool in pools.values() for r in pool]
    rng.shuffle(leftovers)
    picked += leftovers[:args.sessions - len(picked)]
    rng.shuffle(picked)

    if args.show:
        for sample in picked[:args.show]:
            shopper = CleanHumanShopper(sample, known, random.Random(sample["sample_id"]))
            print(f"\n--- {sample['sample_id']}  {sample['scenario_type']}  "
                  f"target={sample['ground_truth']['parent_asin']}")
            print(f"  C1: {shopper.opening()}")
            for attribute in ("material", "color", "size", "feature", "style"):
                line = shopper.reply(attribute)
                print(f"  A(ask={attribute}) -> C: {line}")
        return

    catalog_ids, _, _ = catalog_index(args.catalog)
    from starter.agent import Agent
    started = time.time()
    agent = Agent(args.catalog)
    result, exceptions, quotes, shorts = run(agent, picked, catalog_ids, known, args.seed)
    overall = summarise(result)
    mix = {k: sum(1 for r in picked if r["scenario_type"] == k) for k in want}
    print(f"\n  sessions {overall['n']}  unique targets "
          f"{len({r['ground_truth']['parent_asin'] for r in picked})}  "
          f"exceptions {exceptions}  ({time.time()-started:.0f}s)")
    print(f"  scenario mix {mix}")
    print(f"  turns spoken as a real review sentence: {quotes}  "
          f"({quotes/(quotes+shorts):.0%})   as the short mined form: {shorts}\n")
    print(f"  {'':<10} {'n':>4} {'HR@10':>8} {'MRR':>9} {'MTTC':>7} {'SCORE':>9}")
    print(f"  {'OVERALL':<10} {overall['n']:>4} {overall['hit_rate_at_10']:>8.4f} "
          f"{overall['mrr']:>9.6f} {overall['mttc']:>7.3f} {overall['technical_score']:>9.6f}")
    for name in want:
        sub = [r for r in result if r["scenario_type"] == name]
        if sub:
            s = summarise(sub)
            print(f"  {name:<10} {s['n']:>4} {s['hit_rate_at_10']:>8.4f} "
                  f"{s['mrr']:>9.6f} {s['mttc']:>7.3f} {s['technical_score']:>9.6f}")
    if args.output:
        Path(args.output).write_text(json.dumps(
            {"overall": overall, "scenario_mix": mix, "exceptions": exceptions,
             "quote_turns": quotes, "short_turns": shorts,
             "by_scenario": {k: summarise([r for r in result if r["scenario_type"] == k])
                             for k in want if any(r["scenario_type"] == k for r in result)},
             "sessions": result}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
