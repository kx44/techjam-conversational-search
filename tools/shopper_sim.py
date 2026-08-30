"""A shopper who talks around the product, not about its listing.

Both existing harnesses share three assumptions that neither was chosen
deliberately: the customer names the category in the catalog's own words, every
turn carries exactly one clean positive constraint, and nothing is ever said
that does not narrow the search. Real conversations violate all three.

This one models a shopper with an occasion and a recipient who mentions
requirements unevenly, pads them with true-but-useless context, occasionally
negates, sometimes repeats, and sometimes asks a question back. The underlying
facts still come from the target's metadata - otherwise the task is unsolvable
by anything - but they arrive wrapped in the way a person actually types.

Scored with the official metric definitions so numbers stay comparable.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

MAX_TURNS = 10
TOP_K = 10
WORD = re.compile(r"[a-z0-9]+")
MATERIAL = re.compile(r"\b(cotton|polyester|nylon|leather|wool|silk|denim|suede|mesh|linen|fleece|satin)\b", re.I)
COLOR = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|navy|beige|silver|gold|purple)\b", re.I)
STOP = {"the","a","an","and","or","of","for","with","to","in","on","is","are","it","its","this","that",
        "you","your","our","we","they","be","been","will","can","not","no","if","as","at","by","from",
        "very","more","most","all","any","each","other","than","then","so","such","just","also","made"}

OCCASIONS = ["a wedding next month", "work", "the gym", "a holiday", "everyday wear",
             "a birthday present", "hiking trips", "the office", "travelling"]
RECIPIENTS = ["me", "my sister", "my husband", "my mum", "a friend", "my daughter", "my partner"]
# True of the shopper, useless for narrowing 50,000 products.
NOISE = ["I'm not in a huge rush, but sooner is better.",
         "Last time I bought online the sizing was all over the place.",
         "It's a present, so it should look decent when it arrives.",
         "I've been putting this off for weeks honestly.",
         "Budget is flexible-ish, within reason.",
         "I usually shop in store but nothing nearby had anything."]
ASK_BACK = ["What would you suggest?", "Anything you'd recommend?",
            "Does that narrow it down at all?", "Which of those is most popular?"]


def words(text: str, limit: int) -> list[str]:
    out: list[str] = []
    for token in WORD.findall(text.lower()):
        # Catalog copy is full of CJK brackets and bullet glyphs that tokenise
        # to bare digits; a shopper would not read those out.
        if token.isdigit() or any(c.isdigit() for c in token):
            continue
        if len(token) > 2 and token not in STOP and token not in out:
            out.append(token)
        if len(out) >= limit:
            break
    return out


def leaf_noun(categories: list[str]) -> str:
    parts: list[str] = []
    for value in categories:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece and piece.lower() not in {"clothing", "clothing shoes & jewelry"}:
                parts.append(piece)
    return (parts[-1] if parts else "something").lower()


def facts(product: dict, rng: random.Random) -> list[str]:
    """What the shopper knows they want, phrased as they would say it."""
    blob = " ".join([str(product.get("title") or "")]
                    + [str(v) for v in (product.get("features") or [])]
                    + [f"{k} {v}" for k, v in (product.get("details") or {}).items()])
    out: list[str] = []
    material = MATERIAL.search(blob)
    if material:
        out.append(rng.choice(["it should be {v}", "{v} if possible", "something in {v}",
                               "{v} rather than anything synthetic"]).format(v=material.group(1).lower()))
    color = COLOR.search(blob)
    if color:
        out.append(rng.choice(["ideally {v}", "{v} would go with everything",
                               "something {v}", "{v}, or close to it"]).format(v=color.group(1).lower()))
    price = product.get("price")
    if isinstance(price, (int, float)) and price > 0:
        out.append(rng.choice(["I'd rather not go much over ${v:.0f}",
                               "somewhere around ${v:.0f}", "under ${v:.0f} ideally"]).format(v=price))
    for feature in (product.get("features") or [])[:5]:
        picked = words(str(feature), 3)
        if picked:
            out.append(rng.choice(["{v} matters to me", "it needs {v}", "something with {v}",
                                   "{v} is the main thing"]).format(v=" ".join(picked)))
    rng.shuffle(out)
    return out[:6]


class Shopper:
    def __init__(self, product: dict, scenario: str, seed: str) -> None:
        self.rng = random.Random(seed)
        self.scenario = scenario
        self.noun = leaf_noun([str(v) for v in product.get("categories") or []])
        self.facts = facts(product, self.rng)
        self.said: list[str] = []
        self.occasion = self.rng.choice(OCCASIONS)
        self.recipient = self.rng.choice(RECIPIENTS)
        self.refused = False
        self.override_turn = self.rng.choice([3, 4])
        self.override_done = scenario != "intent_override"

    def _next_fact(self) -> str | None:
        remaining = [f for f in self.facts if f not in self.said]
        if not remaining:
            return None
        self.said.append(remaining[0])
        return remaining[0]

    def opening(self) -> str:
        # The category is named the way a person would, not as the catalog spells it.
        lead = self.rng.choice([
            f"Hi — I'm after {self.noun} for {self.occasion}, for {self.recipient}.",
            f"Hello! Looking for {self.noun}. It's for {self.occasion}.",
            f"Hey, I need {self.noun} — {self.occasion}, if that helps.",
        ])
        if self.scenario == "buying":
            fact = self._next_fact()
            if fact:
                return f"{lead} {fact.capitalize()}."
        return lead

    def reply(self, ask_attribute, agent_message, turn: int) -> str:
        if not self.override_done and turn >= self.override_turn:
            self.override_done = True
            fact = self._next_fact()
            return (f"Actually, forget what I said before — {fact}." if fact
                    else "Actually, ignore my last message.")
        if self.scenario == "boundary" and not self.refused:
            self.refused = True
            return self.rng.choice(["No strong view on that, you pick.",
                                    "I'll leave that one up to you.",
                                    "Not fussed about that, honestly."])
        roll = self.rng.random()
        if roll < 0.14:                                  # pure context, no signal
            return self.rng.choice(NOISE)
        if roll < 0.22 and self.said:                    # restate something already said
            return f"Like I said, {self.rng.choice(self.said)}."
        if roll < 0.30:                                  # turn it back on the agent
            return self.rng.choice(ASK_BACK)
        first = self._next_fact()
        if first is None:
            return self.rng.choice(["That's about all I've got.", "Nothing else springs to mind.",
                                    "I think that covers it."])
        if roll < 0.55:                                  # two requirements at once
            second = self._next_fact()
            if second:
                return f"{first.capitalize()}, and {second}."
        if roll < 0.62:                                  # one requirement plus an exclusion
            return f"{first.capitalize()}. Nothing too flashy though."
        return f"{first.capitalize()}."


def normalize(payload, catalog_ids):
    if not isinstance(payload, list):
        return []
    out, seen = [], set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        pid = str(value).strip()
        if pid and pid not in seen and pid in catalog_ids:
            seen.add(pid)
            out.append(pid)
        if len(out) >= TOP_K:
            break
    return out


def run(agent, samples, products, catalog_ids) -> dict:
    rows = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        shopper = Shopper(products[target], sample["scenario_type"], sample["sample_id"])
        session = f"shop_{sample['sample_id']}"
        agent.reset(session, sample.get("user_profile", {}))
        message = shopper.opening()
        hit_turn = rank = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session, message, turn, TOP_K)
            except Exception:
                response = {}
            ranked = normalize((response or {}).get("recommendations"), catalog_ids)
            if shopper.override_done and target in ranked:
                rank, hit_turn = ranked.index(target) + 1, turn
                break
            if turn == MAX_TURNS:
                break
            message = shopper.reply((response or {}).get("ask_attribute"),
                                    (response or {}).get("message"), turn + 1)
        rows.append({"scenario": sample["scenario_type"], "hit": hit_turn is not None,
                     "turn": hit_turn, "rr": 0.0 if rank is None else 1.0 / rank})
    hit = sum(r["hit"] for r in rows) / len(rows)
    mrr = statistics.fmean(r["rr"] for r in rows)
    mttc = statistics.fmean(r["turn"] if r["turn"] else MAX_TURNS + 1 for r in rows)
    eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    by = {}
    for name in sorted({r["scenario"] for r in rows}):
        group = [r for r in rows if r["scenario"] == name]
        by[name] = {"n": len(group), "hit": round(sum(g["hit"] for g in group) / len(group), 4),
                    "mrr": round(statistics.fmean(g["rr"] for g in group), 4)}
    return {"hit_rate_at_10": round(hit, 4), "mrr": round(mrr, 4), "mttc": round(mttc, 3),
            "efficiency": round(eff, 4), "score": round(0.5 * hit + 0.3 * mrr + 0.2 * eff, 4),
            "scenario": by}


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from starter.agent import Agent

    parser = argparse.ArgumentParser(description="Conversational shopper harness")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()
    products = {}
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            products[str(item["parent_asin"])] = item
    samples = [json.loads(l) for l in Path(args.dataset).open(encoding="utf-8") if l.strip()]
    print(json.dumps(run(Agent(args.catalog), samples, products, set(products)), indent=2))


if __name__ == "__main__":
    main()
