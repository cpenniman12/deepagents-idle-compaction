"""Prove the sweeper compacts an idle thread IN PLACE -- offline, no API key.

Simulates the real at-59-min flow without waiting an hour or calling a real model:

  1. Build a deep agent backed by a persistent (sqlite) checkpointer.
  2. Seed three threads directly in the store:
       - big + stale   (idle 90m, ~380k tokens)
       - big + fresh    (idle  2m, ~380k tokens)
       - small + stale  (idle 90m,   ~4k tokens)
  3. Run the sweeper (as cron would).
  4. Assert ONLY the big+stale thread was compacted in place.

Run:  python sweeper_demo.py
"""

from __future__ import annotations

import time

from langchain.agents.middleware.summarization import count_tokens_approximately
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from deepagents import create_deep_agent
from idle_compaction import IdleCompactionMiddleware
from idle_compaction.sweeper import sweep

NOW = 2_000_000_000.0


def msgs(n: int, chars: int) -> list:
    out: list = []
    for i in range(n):
        body = f"turn {i}: " + ("x" * chars)
        out.append(HumanMessage(body) if i % 2 == 0 else AIMessage(body))
    return out


def big_thread() -> list:
    return msgs(28, 55_000) + msgs(12, 300)  # huge stale prefix + normal recent


def seed(graph, thread_id: str, messages: list, last_turn_at: float) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {"messages": messages, "last_turn_at": last_turn_at})


def tokens_in(graph, thread_id: str) -> int:
    config = {"configurable": {"thread_id": thread_id}}
    return count_tokens_approximately(graph.get_state(config).values["messages"])


def main() -> None:
    fake = FakeListChatModel(responses=["<COMPACTED SUMMARY>"] * 10)
    mw = IdleCompactionMiddleware(
        fake, token_threshold=300_000, idle_seconds=59 * 60, keep=("messages", 12)
    )

    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph = create_deep_agent(model=fake, tools=[], middleware=[mw], checkpointer=saver)

        seed(graph, "big-stale", big_thread(), last_turn_at=NOW - 90 * 60)
        seed(graph, "big-fresh", big_thread(), last_turn_at=NOW - 2 * 60)
        seed(graph, "small-stale", msgs(20, 800), last_turn_at=NOW - 90 * 60)

        before = {t: tokens_in(graph, t) for t in ["big-stale", "big-fresh", "small-stale"]}

        # This is the line cron runs every few minutes:
        compacted = sweep(graph, saver, mw, now=NOW)

        after = {t: tokens_in(graph, t) for t in ["big-stale", "big-fresh", "small-stale"]}

    print("compacted threads:", compacted)
    for t in ["big-stale", "big-fresh", "small-stale"]:
        mark = "COMPACTED" if t in compacted else "untouched"
        print(f"  {t:12s} {before[t]:>8,} -> {after[t]:>8,} tokens   [{mark}]")

    assert compacted == ["big-stale"], f"expected only big-stale, got {compacted}"
    assert after["big-stale"] < 10_000, "big-stale should be tiny after compaction"
    assert after["big-fresh"] == before["big-fresh"], "fresh thread must be untouched"
    assert after["small-stale"] == before["small-stale"], "small thread must be untouched"
    print("\nsweeper fired during idle, in place, on the right thread ✅")


if __name__ == "__main__":
    main()
