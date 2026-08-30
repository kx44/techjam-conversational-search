"""Smoke checks for clause-level preference handling.

These are deliberately small and deterministic: the agent may change ranking
strategy, but rejected attribute values must never enter positive retrieval
state as lexical terms, phrase features, or dense text.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.agent import Agent
from starter.intent import HARD_APPROVE, HARD_REJECT, NEUTRAL, SOFT_APPROVE
from starter.stemmer import stem


CASES = (
    (
        "I like black, but leather isn't for me",
        ("color", "black"),
        ("material", "leather"),
    ),
    (
        "Cotton would be ideal; no polyester please",
        ("material", "cotton"),
        ("material", "polyester"),
    ),
    (
        "A blue option works, however avoid suede",
        ("color", "blue"),
        ("material", "suede"),
    ),
)


def main() -> None:
    agent = Agent()
    failures: list[str] = []
    for index, (message, accepted, rejected) in enumerate(CASES, start=1):
        state_id = f"negation-{index}"
        agent.reset(state_id, {})
        state = agent._sessions[state_id]
        agent._accumulate(state, message)

        accepted_attr, accepted_value = accepted
        rejected_attr, rejected_value = rejected
        if accepted_value not in state["plain"]:
            failures.append(f"{message!r}: missing accepted term {accepted_value!r}")
        if accepted_value not in state["accepted"].get(accepted_attr, set()):
            failures.append(f"{message!r}: missing accepted pair {accepted_attr}={accepted_value}")
        if rejected_value not in state["rejected"].get(rejected_attr, set()):
            failures.append(f"{message!r}: missing rejected pair {rejected_attr}={rejected_value}")
        if rejected_value in state["plain"]:
            failures.append(f"{message!r}: rejected raw term leaked into plain query")
        if stem(rejected_value) in state["stems"]:
            failures.append(f"{message!r}: rejected term leaked into stem query")
        if rejected_value in " ".join(state["text"]).lower():
            failures.append(f"{message!r}: rejected term leaked into dense text")
        signal = state["preferences"].get((rejected_attr, rejected_value))
        if signal is None or signal.label != HARD_REJECT or signal.weight >= 0:
            failures.append(f"{message!r}: rejected pair lacks a hard negative preference signal")
        signal = state["preferences"].get((accepted_attr, accepted_value))
        if signal is None or signal.label == HARD_REJECT or signal.weight < 0:
            failures.append(f"{message!r}: accepted pair has a negative preference signal")

    agent.reset("flip", {})
    state = agent._sessions["flip"]
    agent._accumulate(state, "I like leather.")
    agent._accumulate(state, "Actually, leather isn't for me.")
    if "leather" not in state["rejected"].get("material", set()):
        failures.append("change-of-mind: leather was not marked rejected")
    if "leather" in state["accepted"].get("material", set()):
        failures.append("change-of-mind: leather stayed accepted after rejection")
    if "leather" in state["plain"]:
        failures.append("change-of-mind: leather stayed in positive plain query")

    agent.reset("ambiguous", {})
    state = agent._sessions["ambiguous"]
    agent._accumulate(state, "A key requirement is: polyester.")
    if "polyester" not in state["plain"]:
        failures.append("ambiguous positive: polyester was dropped from positive retrieval")
    if "polyester" in state["rejected"].get("material", set()):
        failures.append("ambiguous positive: polyester was marked rejected")
    signal = state["preferences"].get(("material", "polyester"))
    if signal is None or signal.label != HARD_APPROVE or signal.weight <= 0:
        failures.append("ambiguous positive: polyester lacks a positive preference signal")

    agent.reset("unpaired", {})
    state = agent._sessions["unpaired"]
    agent._accumulate(state, "Wrap closure.")
    if "wrap" not in state["plain"] or "closure" not in state["plain"]:
        failures.append("unpaired reject: useful clause text was dropped")

    agent.reset("neutral", {})
    state = agent._sessions["neutral"]
    agent._accumulate(state, "100% Polyester.")
    signal = state["preferences"].get(("material", "polyester"))
    if signal is None or signal.label != NEUTRAL or signal.weight != 0:
        failures.append("neutral fragment: polyester composition did not produce a neutral signal")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)
    print("PASS: rejected attribute values stayed out of positive retrieval state")


if __name__ == "__main__":
    main()
