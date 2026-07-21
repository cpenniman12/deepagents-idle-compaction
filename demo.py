"""Watch IdleCompactionMiddleware fire -- offline, no API key, no cost.

This drives the middleware's ``before_model`` hook directly with synthetic
state, using a fake chat model for the summary. It runs three cases:

  A. large + stale  -> COMPACTS   (idle 60m, ~400k tokens)
  B. large + fresh  -> no-op       (idle  5m, ~400k tokens)  size ok, but active
  C. small + stale  -> no-op       (idle 60m,  ~5k tokens)   stale, but too small

Run:  python demo.py
"""

from __future__ import annotations

import time

from langchain.agents.middleware.summarization import count_tokens_approximately
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from idle_compaction import IdleCompactionMiddleware

FIXED_NOW = 1_000_000_000.0  # frozen clock so the demo is deterministic


def make_messages(n: int, chars_each: int) -> list:
    """Build an alternating Human/AI conversation of a given rough size."""
    msgs: list = []
    for i in range(n):
        body = f"turn {i}: " + ("x" * chars_each)
        msgs.append(HumanMessage(body) if i % 2 == 0 else AIMessage(body))
    return msgs


def make_realistic_big() -> list:
    """A huge stale prefix followed by normal-sized recent turns.

    Mirrors the real case: the old context is enormous, but the last handful of
    turns (the ones ``keep`` preserves) are ordinary. So compaction collapses the
    thread to ~a few thousand tokens, not ~120k.
    """
    stale_prefix = make_messages(28, 55_000)  # ~1.5M chars -> ~380k tokens
    recent = make_messages(12, 300)           # normal recent turns
    return stale_prefix + recent


def run_case(name: str, messages: list, idle_seconds: float) -> None:
    fake_summary = "<COMPACTED SUMMARY of the stale prefix>"
    model = FakeListChatModel(responses=[fake_summary])

    mw = IdleCompactionMiddleware(
        model,
        token_threshold=300_000,
        idle_seconds=59 * 60,
        keep=("messages", 12),
        time_fn=lambda: FIXED_NOW,
    )

    state = {
        "messages": messages,
        # stamp the previous turn `idle_seconds` in the past
        "last_turn_at": FIXED_NOW - idle_seconds,
    }

    before_n = len(messages)
    before_tok = count_tokens_approximately(messages)

    patch = mw.before_model(state, runtime=None)

    compacted = "messages" in (patch or {})
    if compacted:
        after = patch["messages"]
        # RemoveMessage(ALL) + summary/system + preserved
        real = [m for m in after if not m.__class__.__name__.startswith("Remove")]
        after_n = len(real)
        after_tok = count_tokens_approximately(real)
    else:
        after_n, after_tok = before_n, before_tok

    print(f"\n=== {name} ===")
    print(f"  idle: {idle_seconds/60:.0f} min   tokens: ~{before_tok:,}")
    verdict = "COMPACTED" if compacted else "no-op"
    print(f"  -> {verdict}")
    print(f"     messages: {before_n} -> {after_n}")
    print(f"     tokens:   ~{before_tok:,} -> ~{after_tok:,}")
    print(f"     last_turn_at re-stamped: {patch.get('last_turn_at') == FIXED_NOW}")
    return compacted


def main() -> None:
    big = make_realistic_big()        # huge stale prefix + normal recent turns
    small = make_messages(20, 800)    # ~16k chars  -> a few thousand tokens

    a = run_case("A. large + stale (60m)", big, idle_seconds=60 * 60)
    b = run_case("B. large + fresh (5m)", big, idle_seconds=5 * 60)
    c = run_case("C. small + stale (60m)", small, idle_seconds=60 * 60)

    print("\n--- expected: A compacts, B and C do not ---")
    assert a is True, "A should compact (large + stale)"
    assert b is False, "B should NOT compact (active session)"
    assert c is False, "C should NOT compact (below size threshold)"
    print("all assertions passed ✅")


if __name__ == "__main__":
    main()
