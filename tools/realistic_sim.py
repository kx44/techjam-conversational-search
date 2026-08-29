"""An independently written customer simulator, for robustness testing only.

The point is to NOT share code, templates, or assumptions with
``evaluator/local_evaluator.py``. A customer here speaks in varied natural
language, paraphrases the product's own wording instead of quoting it, answers
whichever topic the agent raised, and sometimes volunteers something unasked.
Scores are computed with the official metric definitions so numbers stay
comparable.
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
WORD = re.compile(r"[a-z0-9$]+")
MATERIAL = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|denim|suede|mesh|satin|linen)\b", re.I)
COLOR = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|silver|gold|navy|beige)\b", re.I)
FILLER = {
    "the","a","an","and","or","of","for","with","to","in","on","is","are","it","its","this","that",
    "you","your","our","we","they","be","been","will","can","not","no","if","as","at","by","from",
    "very","more","most","all","any","each","other","than","then","so","such","just","also","made",
}


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def content_words(text: str, limit: int) -> list[str]:
    seen: list[str] = []
    for token in words(text):
        if len(token) > 2 and token not in FILLER and token not in seen:
            seen.append(token)
        if len(seen) >= limit:
            break
    return seen


def naturalize(text: str, rng: random.Random, limit: int = 5) -> str:
    """Express a metadata string the way a person would: same vocabulary, different surface."""
    picked = content_words(text, limit)
    if not picked:
        return ""
    if len(picked) > 2 and rng.random() < 0.6:
        head, tail = picked[:-1], picked[-1]
        return " ".join(head) + " and " + tail
    return " ".join(picked)


def category_phrase(categories: list[str], rng: random.Random) -> str:
    parts: list[str] = []
    for value in categories:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece and piece.lower() not in {"clothing", "clothing shoes & jewelry"}:
                parts.append(piece)
    if not parts:
        return "something"
    tail = parts[-2:] if len(parts) > 1 else parts[-1:]
    phrase = " ".join(tail).lower()
    if rng.random() < 0.35 and len(tail) > 1:      # people often say just the leaf
        phrase = tail[-1].lower()
    return phrase


def build_intent(product: dict, rng: random.Random) -> list[tuple[str, str]]:
    """A list of (topic, natural phrasing) the customer can disclose."""
    blob = " ".join(
        [str(product.get("title") or "")]
        + [str(v) for v in (product.get("features") or [])]
        + [f"{k} {v}" for k, v in (product.get("details") or {}).items()]
    )
    slots: list[tuple[str, str]] = []
    material = MATERIAL.search(blob)
    if material:
        slots.append(("material", material.group(1).lower()))
    color = COLOR.search(blob)
    if color:
        slots.append(("color", color.group(1).lower()))
    store = str(product.get("store") or "").strip()
    if store and rng.random() < 0.3:
        slots.append(("brand", store.lower()))
    price = product.get("price")
    if isinstance(price, (int, float)) and price > 0:
        slots.append(("budget", f"around ${price:.0f}"))
    # Deliberately NOT the title: a shopper describes what they want, they do
    # not recite the product's name back at you.
    for feature in (product.get("features") or [])[:6]:
        phrase = naturalize(str(feature), rng, limit=3)
        if phrase:
            slots.append(("feature", phrase))
    rng.shuffle(slots)
    return slots[:6]


OPEN_BUY = [
    "Hi, I'm looking for {cat}. It really needs to be {v}.",
    "Hey there - after {cat} if you can help. Main thing is {v}.",
    "I'm shopping for {cat}, and {v} matters a lot to me.",
    "Trying to track down {cat}. Ideally {v}.",
]
OPEN_BROWSE = [
    "Hi! Just browsing for {cat} at the moment, nothing specific yet.",
    "Not totally sure what I want - something along the lines of {cat}?",
    "Hey, thinking about {cat}. Show me what you've got?",
]
ANSWER = [
    "For that, {v} would be ideal.",
    "{v}, I'd say.",
    "Hmm, probably {v}.",
    "It should be {v} really.",
    "Yeah - {v}.",
]
VOLUNTEER = [
    "No strong feelings there, though {v} would be good.",
    "Not fussed about that. But {v} is important.",
    "Don't mind either way - what I do care about is {v}.",
]
NOTHING = [
    "I don't really have a preference there.",
    "No strong opinion on that one, sorry.",
    "Honestly that doesn't matter much to me.",
]
REFUSE = [
    "I genuinely don't mind about that - use your judgment.",
    "No preference there, you pick.",
]
OVERRIDE = [
    "Actually, scratch that earlier bit - what I really need is {v}.",
    "Hmm, let me correct myself. Forget what I said before; {v} is the priority.",
    "Sorry, change of plan - ignore that. It has to be {v}.",
]


class Customer:
    def __init__(self, product: dict, scenario: str, seed: str) -> None:
        self.rng = random.Random(seed)
        self.scenario = scenario
        self.cat = category_phrase([str(v) for v in product.get("categories") or []], self.rng)
        self.slots = build_intent(product, self.rng)
        self.said: set[int] = set()
        self.refused = False
        self.override_turn = self.rng.choice([3, 4])
        self.override_done = scenario != "intent_override"

    def _take(self, topic: str | None) -> tuple[int, str] | None:
        pool = [i for i in range(len(self.slots)) if i not in self.said]
        if not pool:
            return None
        if topic:
            match = [i for i in pool if self.slots[i][0] == topic]
            if match:
                return match[0], self.slots[match[0]][1]
        return pool[0], self.slots[pool[0]][1]

    def opening(self) -> str:
        if self.scenario in ("browsing", "boundary"):
            return self.rng.choice(OPEN_BROWSE).format(cat=self.cat)
        got = self._take(None)
        if self.scenario == "intent_override":
            # opens with a preference that is later revoked; never a real constraint
            return f"Hi, I'm after {self.cat}. Something fairly ordinary, nothing fancy."
        if not got:
            return self.rng.choice(OPEN_BROWSE).format(cat=self.cat)
        idx, value = got
        self.said.add(idx)
        return self.rng.choice(OPEN_BUY).format(cat=self.cat, v=value)

    def reply(self, ask_attribute: object, agent_message: object, turn: int) -> str:
        if not self.override_done and turn >= self.override_turn:
            self.override_done = True
            got = self._take(None)
            if got:
                idx, value = got
                self.said.add(idx)
                return self.rng.choice(OVERRIDE).format(v=value)
        topic = ask_attribute if isinstance(ask_attribute, str) else None
        if topic is None and isinstance(agent_message, str):
            # a real person answers the question that was actually asked
            for name in ("material", "colour", "color", "size", "brand", "budget", "price", "style"):
                if name in agent_message.lower():
                    topic = "color" if name == "colour" else ("budget" if name == "price" else name)
                    break
        if self.scenario == "boundary" and not self.refused and topic:
            self.refused = True
            return self.rng.choice(REFUSE)
        got = self._take(topic)
        if not got:
            return self.rng.choice(NOTHING)
        idx, value = got
        self.said.add(idx)
        matched = topic is not None and self.slots[idx][0] == topic
        template = ANSWER if (matched or topic is None) else VOLUNTEER
        return self.rng.choice(template).format(v=value)


def normalize(payload: object, catalog_ids: set[str]) -> list[str]:
    if not isinstance(payload, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        pid = str(value).strip()
        if pid and pid not in seen and pid in catalog_ids:
            seen.add(pid)
            out.append(pid)
        if len(out) >= TOP_K:
            break
    return out


def run(agent, samples: list[dict], products: dict[str, dict], catalog_ids: set[str]) -> dict:
    rows = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        scenario = sample["scenario_type"]
        customer = Customer(products[target], scenario, sample["sample_id"])
        session_id = f"sim_{sample['sample_id']}"
        agent.reset(session_id, sample.get("user_profile", {}))
        message = customer.opening()
        hit_turn = None
        rank = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, message, turn, TOP_K)
            except Exception:
                response = {}
            if not isinstance(response, dict):
                response = {}
            ranked = normalize(response.get("recommendations"), catalog_ids)
            if customer.override_done and target in ranked:
                rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            message = customer.reply(response.get("ask_attribute"), response.get("message"), turn + 1)
        rows.append({"scenario": scenario, "hit": hit_turn is not None,
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

    parser = argparse.ArgumentParser(description="Realistic-conversation robustness harness")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()
    products = {}
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            products[str(item["parent_asin"])] = item
    samples = [json.loads(line) for line in Path(args.dataset).open(encoding="utf-8") if line.strip()]
    print(json.dumps(run(Agent(args.catalog), samples, products, set(products)), indent=2))


if __name__ == "__main__":
    main()
