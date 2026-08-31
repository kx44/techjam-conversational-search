from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MAX_TURNS = 10
TOP_K = 10
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
ROOT_CATEGORIES = {
    "clothing",
    "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
}
COLORS = (
    "rose gold", "navy blue", "light blue", "dark blue", "sky blue",
    "hot pink", "light pink", "dark green", "olive green", "black",
    "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "gold", "silver", "beige", "khaki",
    "navy", "teal", "cream", "ivory", "burgundy", "maroon",
)
MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex",
    "silk", "rayon", "linen", "denim", "satin", "fleece", "suede",
    "stainless steel", "sterling silver", "gold plated",
)
USE_CASES = (
    "running", "walking", "hiking", "trail", "gym", "workout", "yoga",
    "training", "cycling", "swimming", "winter", "outdoor", "work",
    "office", "wedding", "party", "formal", "casual", "halloween",
    "costume", "travel", "beach", "dance",
)
SLEEVE_PATTERNS = (
    ("long sleeve", r"\blong[- ]sleeve(?:d)?\b"),
    ("short sleeve", r"\bshort[- ]sleeve(?:d)?\b"),
    ("sleeveless", r"\bsleeveless\b"),
)
TYPE_RULES = (
    ("running shoes", ("running shoe", "running shoes")),
    ("hiking boots", ("hiking boot", "hiking boots")),
    ("sneakers", ("sneaker", "sneakers", "tennis shoe", "athletic shoe")),
    ("sandals", ("sandal", "sandals", "slides", "flip flop")),
    ("boots", ("boot", "boots")),
    ("heels", ("heel", "heels", "pump", "pumps")),
    ("dress shoes", ("dress shoe", "oxford shoe", "loafer", "loafers")),
    ("bra", ("sports bra", "bra", "bralette")),
    ("swimsuit", ("swimsuit", "bikini", "swimwear", "one-piece")),
    ("t-shirt", ("t-shirt", "t shirt", "tee shirt", "tees")),
    ("shirt", ("shirt", "blouse", "top")),
    ("dress", ("dress", "gown")),
    ("jacket", ("jacket", "coat", "parka")),
    ("hoodie", ("hoodie", "hooded sweatshirt")),
    ("sweater", ("sweater", "cardigan")),
    ("pants", ("pants", "trousers", "slacks")),
    ("jeans", ("jeans", "denim pants")),
    ("shorts", ("shorts",)),
    ("leggings", ("leggings", "yoga pants")),
    ("skirt", ("skirt",)),
    ("socks", ("sock", "socks")),
    ("underwear", ("underwear", "briefs", "boxers", "panties")),
    ("costume", ("costume", "cosplay")),
    ("earrings", ("earring", "earrings")),
    ("necklace", ("necklace", "pendant")),
    ("bracelet", ("bracelet", "bangle")),
    ("ring", ("ring", "rings")),
    ("watch", ("watch", "watches")),
    ("wallet", ("wallet",)),
    ("belt", ("belt",)),
    ("bag", ("handbag", "purse", "backpack", "tote", "bag")),
)
ROUTES = (
    "direct_buying",
    "terse_keywords",
    "browse_then_narrow",
    "attribute_accumulation",
    "intent_override",
    "negative_constraint",
    "boundary_broad_start",
    "title_fragment",
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "to", "with", "without", "women",
    "womens", "woman", "men", "mens", "man", "girls", "boys", "unisex",
}


@dataclass(frozen=True)
class Facts:
    parent_asin: str
    title: str
    categories: tuple[str, ...]
    department: str
    product_type: str
    colors: tuple[str, ...]
    materials: tuple[str, ...]
    sleeves: tuple[str, ...]
    use_cases: tuple[str, ...]
    brand: str
    price: float | None
    title_fragment: str


def flatten(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from flatten(item)
    elif isinstance(value, list):
        for item in value:
            yield from flatten(item)
    elif value not in (None, ""):
        yield str(value)


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if value not in (None, "", []):
            parts.extend(flatten(value))
    return " ".join(parts)


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" -;,.\t\n")


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'&+-]*", text)


def unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = clean_space(value.lower())
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return tuple(result)


def contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(phrase).replace(r"\ ", r"[- ]") + r"(?![a-z0-9])"
    return re.search(pattern, text, re.I) is not None


def detect_department(text: str, categories: tuple[str, ...]) -> str:
    joined_categories = " ".join(categories).lower()
    haystack = f"{joined_categories} {text.lower()}"
    if re.search(r"\b(women|womens|woman|female|ladies|girls?)\b", haystack):
        if re.search(r"\b(girl|girls)\b", haystack):
            return "girls"
        return "women"
    if re.search(r"\b(men|mens|man|male|boys?)\b", haystack):
        if re.search(r"\b(boy|boys)\b", haystack):
            return "boys"
        return "men"
    if "unisex" in haystack:
        return "unisex"
    return ""


def detect_product_type(text: str, categories: tuple[str, ...]) -> str:
    lowered = text.lower()
    for label, phrases in TYPE_RULES:
        if any(contains_phrase(lowered, phrase) for phrase in phrases):
            return label
    for category in reversed(categories):
        cleaned = clean_space(category)
        if cleaned and cleaned.lower() not in ROOT_CATEGORIES:
            return cleaned.lower()
    return "item"


def title_fragment(title: str, limit: int = 6) -> str:
    picked: list[str] = []
    for token in words(title):
        lowered = token.lower().strip("'")
        if len(lowered) < 3 or lowered in STOPWORDS:
            continue
        if lowered not in picked:
            picked.append(lowered)
        if len(picked) >= limit:
            break
    return " ".join(picked)


def as_price(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def extract_facts(product: dict) -> Facts:
    title = clean_space(str(product.get("title") or ""))
    categories = tuple(str(item) for item in product.get("categories") or [])
    text = searchable_text(product)
    lowered = text.lower()
    colors = unique(color for color in COLORS if contains_phrase(lowered, color))
    materials = unique(material for material in MATERIALS if contains_phrase(lowered, material))
    sleeves = unique(label for label, pattern in SLEEVE_PATTERNS if re.search(pattern, lowered, re.I))
    uses = unique(use for use in USE_CASES if contains_phrase(lowered, use))
    return Facts(
        parent_asin=str(product["parent_asin"]),
        title=title,
        categories=categories,
        department=detect_department(text, categories),
        product_type=detect_product_type(text, categories),
        colors=colors,
        materials=materials,
        sleeves=sleeves,
        use_cases=uses,
        brand=clean_space(str(product.get("store") or "")),
        price=as_price(product.get("price")),
        title_fragment=title_fragment(title),
    )


def department_phrase(department: str) -> str:
    return {
        "men": "men's",
        "women": "women's",
        "boys": "boys'",
        "girls": "girls'",
        "unisex": "unisex",
    }.get(department, "")


def target_label(facts: Facts) -> str:
    dept = department_phrase(facts.department)
    if dept and dept.lower().rstrip("'s") not in facts.product_type:
        return clean_space(f"{dept} {facts.product_type}")
    return facts.product_type or "item"


def compact_join(values: Iterable[str], limit: int = 3) -> str:
    clean = [value for value in values if value]
    clean = clean[:limit]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def requirements(
    facts: Facts,
    limit: int = 3,
    include_brand: bool = False,
    include_title_hint: bool = True,
) -> list[str]:
    values: list[str] = []
    values.extend(facts.sleeves[:1])
    values.extend(facts.colors[:1])
    values.extend(facts.materials[:1])
    values.extend(facts.use_cases[:1])
    if include_brand and facts.brand:
        values.append(f"from {facts.brand}")
    if include_title_hint and facts.title_fragment and len(values) < limit:
        values.append(facts.title_fragment)
    return list(unique(values))[:limit]


def budget_phrase(facts: Facts) -> str:
    if facts.price is None:
        return ""
    rounded = round(facts.price)
    if rounded <= 0:
        return ""
    return f"around ${rounded}"


def opposite_department(department: str) -> str:
    return {
        "men": "women's",
        "women": "men's",
        "boys": "girls'",
        "girls": "boys'",
    }.get(department, "")


def opposite_product_type(product_type: str) -> str:
    if "shoe" in product_type or "sneaker" in product_type:
        return "sandals"
    if "sandal" in product_type:
        return "boots"
    if "shirt" in product_type or "t-shirt" in product_type:
        return "pants"
    if "pants" in product_type or "jeans" in product_type:
        return "shirt"
    if "earring" in product_type:
        return "necklace"
    if "necklace" in product_type:
        return "earrings"
    if "dress" in product_type:
        return "jacket"
    return "shoes"


def opposite_constraint(facts: Facts) -> str:
    dept = opposite_department(facts.department)
    if dept:
        return dept
    if facts.sleeves:
        return "short sleeve" if facts.sleeves[0] == "long sleeve" else "long sleeve"
    if facts.colors:
        return "white" if facts.colors[0] == "black" else "black"
    return opposite_product_type(facts.product_type)


def profile_for(facts: Facts, route: str, rng: random.Random) -> dict:
    profile = {
        "style": rng.choice(["practical", "casual", "value-conscious", "comfort-first"]),
        "route": route,
    }
    if facts.department:
        profile["department"] = facts.department
    if facts.colors:
        profile["favorite_color"] = facts.colors[0]
    if facts.price is not None:
        profile["budget_hint"] = budget_phrase(facts)
    return profile


def route_messages(facts: Facts, route: str, rng: random.Random) -> list[str]:
    label = target_label(facts)
    reqs = requirements(facts, include_brand=False)
    req_text = compact_join(reqs)
    title_hint = facts.title_fragment or label
    use = facts.use_cases[0] if facts.use_cases else rng.choice(["everyday wear", "a gift", "daily use"])
    budget = budget_phrase(facts)

    if route == "direct_buying":
        suffix = f" with {req_text}" if req_text else ""
        if budget and rng.random() < 0.35:
            suffix = f"{suffix} {budget}".strip()
        return [clean_space(f"I am looking for {label}{suffix}.")]

    if route == "terse_keywords":
        tokens = [department_phrase(facts.department), facts.product_type]
        tokens.extend(reqs)
        if budget and rng.random() < 0.25:
            tokens.append(budget)
        return [clean_space(" ".join(token for token in tokens if token))]

    if route == "browse_then_narrow":
        return [
            f"I need something for {use}, but I am still browsing.",
            f"Narrow it to {label}.",
            f"The details that matter are {req_text or title_hint}.",
        ]

    if route == "attribute_accumulation":
        messages = [f"I am shopping for {facts.product_type or 'an item'}."]
        if facts.department:
            messages.append(f"It is for {department_phrase(facts.department)}.")
        for req in reqs[:2]:
            messages.append(f"Please prioritize {req}.")
        if len(messages) < 3:
            messages.append(f"A useful clue is {title_hint}.")
        return messages

    if route == "intent_override":
        wrong_dept = opposite_department(facts.department)
        wrong_type = opposite_product_type(facts.product_type)
        wrong_label = clean_space(f"{wrong_dept} {wrong_type}") or wrong_type
        return [
            f"I am looking for {wrong_label}.",
            f"Actually, ignore that earlier choice. I need {label}.",
            f"Please match {req_text or title_hint}.",
        ]

    if route == "negative_constraint":
        wrong = opposite_constraint(facts)
        return [
            f"I need {label}, not {wrong}.",
            f"What matters most is {req_text or title_hint}.",
        ]

    if route == "boundary_broad_start":
        return [
            "Show me clothing and accessories.",
            f"Make it {label}.",
            f"Use these clues: {req_text or title_hint}.",
        ]

    if route == "title_fragment":
        second = f"It should be {label}"
        if req_text:
            second = f"{second} with {req_text}"
        return [
            f"Do you have something like {title_hint}?",
            f"{second}.",
        ]

    return [f"I am looking for {label}. {req_text}"]


def value_for_attribute(facts: Facts, attribute: object) -> str:
    attr = attribute if isinstance(attribute, str) else "feature"
    if attr == "category":
        return target_label(facts)
    if attr == "material" and facts.materials:
        return facts.materials[0]
    if attr == "color" and facts.colors:
        return facts.colors[0]
    if attr == "style":
        return facts.sleeves[0] if facts.sleeves else (facts.use_cases[0] if facts.use_cases else "")
    if attr == "brand" and facts.brand:
        return facts.brand
    if attr == "budget":
        return budget_phrase(facts)
    if attr == "use_case" and facts.use_cases:
        return facts.use_cases[0]
    if attr == "size":
        return ""
    reqs = requirements(facts, include_brand=False)
    if reqs:
        return reqs[0]
    return facts.title_fragment or target_label(facts)


def followup_for_attribute(facts: Facts, attribute: object) -> str:
    value = value_for_attribute(facts, attribute)
    if value:
        return f"For that, please use {value}."
    if attribute == "budget":
        return "I do not have a strict budget; focus on the closest product match."
    if attribute == "size":
        return "I do not have a size preference; focus on the product details."
    return "I do not have another preference; pick the closest match."


def load_catalog(path: Path) -> tuple[list[dict], set[str], int]:
    products: list[dict] = []
    ids: set[str] = set()
    row_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row_count += 1
            product = json.loads(line)
            parent_asin = str(product.get("parent_asin") or "").strip()
            title = str(product.get("title") or "").strip()
            if not parent_asin or not title or parent_asin in ids:
                continue
            ids.add(parent_asin)
            products.append(product)
    return products, ids, row_count


def raw_bucket_for(product: dict) -> str:
    categories = [str(item) for item in product.get("categories") or []]
    category = categories[1] if len(categories) > 1 else ""
    return clean_space(category.lower() or "other")


def sample_targets(products: list[dict], count: int, seed: int) -> list[tuple[dict, Facts]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        buckets[raw_bucket_for(product)].append(product)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = list(buckets)
    rng.shuffle(keys)
    selected: list[dict] = []
    while len(selected) < count and any(buckets.values()):
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                if len(selected) >= count:
                    break
    return [(product, extract_facts(product)) for product in selected]


def normalize_recommendations(payload: object, catalog_ids: set[str], top_k: int) -> list[str]:
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        parent_asin = str(value).strip()
        if not parent_asin or parent_asin in seen or parent_asin not in catalog_ids:
            continue
        seen.add(parent_asin)
        result.append(parent_asin)
        if len(result) >= top_k:
            break
    return result


def load_agent_class():
    from starter.agent import Agent

    return Agent


def run_session(agent, facts: Facts, route: str, catalog_ids: set[str],
                top_k: int, max_turns: int, seed: int) -> dict:
    rng = random.Random(f"{seed}:{facts.parent_asin}:{route}")
    session_id = f"synthetic_{uuid.uuid4().hex}"
    agent.reset(session_id, profile_for(facts, route, rng))
    scripted = route_messages(facts, route, rng)
    messages: list[str] = []
    first_hit_turn: int | None = None
    first_hit_rank: int | None = None
    best_rank: int | None = None
    final_ranked: list[str] = []
    response: dict = {}

    for turn in range(1, max_turns + 1):
        if turn <= len(scripted):
            user_message = scripted[turn - 1]
        else:
            user_message = followup_for_attribute(facts, response.get("ask_attribute"))
        messages.append(user_message)
        try:
            response = agent.respond(session_id, user_message, turn, top_k)
        except Exception as exc:
            response = {
                "message": f"agent_error: {type(exc).__name__}",
                "ask_attribute": None,
                "recommendations": [],
            }
        final_ranked = normalize_recommendations(response.get("recommendations"), catalog_ids, top_k)
        if facts.parent_asin in final_ranked:
            rank = final_ranked.index(facts.parent_asin) + 1
            best_rank = rank if best_rank is None else min(best_rank, rank)
            if first_hit_turn is None:
                first_hit_turn = turn
                first_hit_rank = rank
                break

    return {
        "parent_asin": facts.parent_asin,
        "title": facts.title,
        "route": route,
        "messages": messages,
        "hit": first_hit_turn is not None,
        "first_hit_turn": first_hit_turn,
        "first_hit_rank": first_hit_rank,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if first_hit_rank is None else 1.0 / first_hit_rank,
        "final_top_10": final_ranked,
        "facts": {
            "department": facts.department,
            "product_type": facts.product_type,
            "colors": list(facts.colors),
            "materials": list(facts.materials),
            "sleeves": list(facts.sleeves),
            "use_cases": list(facts.use_cases),
            "brand": facts.brand,
            "price": facts.price,
        },
    }


def metric_summary(rows: list[dict], max_turns: int) -> dict:
    if not rows:
        return {
            "sample_count": 0,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": None,
            "efficiency": 0.0,
            "technical_score": 0.0,
        }
    hit = sum(1 for row in rows if row["hit"]) / len(rows)
    mrr = statistics.fmean(row["reciprocal_rank"] for row in rows)
    mttc = statistics.fmean(
        row["first_hit_turn"] if row["first_hit_turn"] is not None else max_turns + 1
        for row in rows
    )
    efficiency = max(0.0, min(1.0, ((max_turns + 1) - mttc) / max_turns))
    technical_score = 0.5 * hit + 0.3 * mrr + 0.2 * efficiency
    return {
        "sample_count": len(rows),
        "hit_rate_at_10": round(hit, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(technical_score, 6),
    }


def grouped_summary(rows: list[dict], max_turns: int) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["route"]].append(row)
    return {
        route: metric_summary(items, max_turns)
        for route, items in sorted(grouped.items())
    }


def print_table(overall: dict, by_route: dict, catalog_rows: int, target_pool: int) -> None:
    print("\nSynthetic robustness eval")
    print("These targets are sampled from data/catalog.jsonl; only shopper phrasing is synthetic.\n")
    print(f"Catalog rows read: {catalog_rows}; usable target pool: {target_pool}.\n")
    print(f"{'slice':<26} {'n':>5} {'hit@10':>8} {'mrr':>9} {'mttc':>7} {'score':>9}")
    print("-" * 70)
    print(
        f"{'overall':<26} {overall['sample_count']:>5} "
        f"{overall['hit_rate_at_10']:>8.3f} {overall['mrr']:>9.6f} "
        f"{overall['mttc']:>7.3f} {overall['technical_score']:>9.6f}"
    )
    for route, summary in by_route.items():
        print(
            f"{route:<26} {summary['sample_count']:>5} "
            f"{summary['hit_rate_at_10']:>8.3f} {summary['mrr']:>9.6f} "
            f"{summary['mttc']:>7.3f} {summary['technical_score']:>9.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic robustness evaluator for the shopping copilot. "
            "Samples real products from the 50,000-item catalog and varies only "
            "the shopper phrasing/routes."
        )
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--products", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--variants-per-product", type=int, default=1)
    parser.add_argument("--output", default="", help="optional JSON output path")
    parser.add_argument("--show", type=int, default=0, help="print N generated sessions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_path = Path(args.catalog)
    products, catalog_ids, catalog_rows = load_catalog(catalog_path)
    targets = sample_targets(products, min(args.products, len(products)), args.seed)
    routes: list[tuple[dict, Facts, str]] = []
    for index, (product, facts) in enumerate(targets):
        for variant in range(max(1, args.variants_per_product)):
            route = ROUTES[(index + variant) % len(ROUTES)]
            routes.append((product, facts, route))

    Agent = load_agent_class()
    agent = Agent(catalog_path)
    rows: list[dict] = []
    for index, (_, facts, route) in enumerate(routes, 1):
        row = run_session(agent, facts, route, catalog_ids, args.top_k, args.max_turns, args.seed)
        row["sample_id"] = f"synthetic_{index:04d}"
        rows.append(row)

    overall = metric_summary(rows, args.max_turns)
    by_route = grouped_summary(rows, args.max_turns)
    print_table(overall, by_route, catalog_rows, len(products))

    if args.show:
        print("\nExamples")
        print("-" * 70)
        for row in rows[:args.show]:
            print(f"{row['sample_id']} {row['route']} target={row['parent_asin']}")
            print(f"title: {row['title'][:140]}")
            for turn, message in enumerate(row["messages"], 1):
                print(f"  turn {turn}: {message}")
            print(f"  hit={row['hit']} rank={row['first_hit_rank']}\n")

    if args.output:
        payload = {
            "config": {
                "catalog": str(catalog_path),
                "catalog_row_count": catalog_rows,
                "usable_target_pool": len(products),
                "products": len(targets),
                "sessions": len(rows),
                "seed": args.seed,
                "top_k": args.top_k,
                "max_turns": args.max_turns,
                "variants_per_product": args.variants_per_product,
            },
            "overall": overall,
            "by_route": by_route,
            "failures": [row for row in rows if not row["hit"]][:25],
            "rows": rows,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
