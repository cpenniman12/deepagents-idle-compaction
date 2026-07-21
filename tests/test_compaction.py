"""Durability + correctness suite.

Covers the idle/size gate, the resume-turn stamping, the sqlite deadlock
regression (with a timeout so a regression FAILS instead of hanging), and
cross-process persistence (agent process writes, sweep process compacts, a third
process sees it).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from langchain.agents.middleware.summarization import count_tokens_approximately
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from deepagents import create_deep_agent
from idle_compaction import IdleCompactionMiddleware, sweep

NOW = 1_700_000_000.0
IDLE = 59 * 60
BIG_THRESHOLD = 20_000  # small threshold keeps tests fast; production uses 300k


def msgs(n: int, c: int) -> list:
    return [(HumanMessage if i % 2 == 0 else AIMessage)(f"t{i}:" + ("x" * c)) for i in range(n)]


def make_mw() -> IdleCompactionMiddleware:
    fake = FakeListChatModel(responses=["<SUMMARY>"] * 40)
    return IdleCompactionMiddleware(
        fake, token_threshold=BIG_THRESHOLD, idle_seconds=IDLE, keep=("messages", 6)
    )


def build(saver):
    fake = FakeListChatModel(responses=["<SUMMARY>"] * 40)
    mw = IdleCompactionMiddleware(
        fake, token_threshold=BIG_THRESHOLD, idle_seconds=IDLE, keep=("messages", 6)
    )
    graph = create_deep_agent(model=fake, tools=[], middleware=[mw], checkpointer=saver)
    return graph, mw


# --- gate -------------------------------------------------------------------

def test_idle_and_large_compacts():
    patch = make_mw().maybe_compact(
        {"messages": msgs(20, 6000), "last_turn_at": NOW - 90 * 60}, now=NOW
    )
    assert patch and "messages" in patch


def test_large_but_fresh_skips():
    assert make_mw().maybe_compact(
        {"messages": msgs(20, 6000), "last_turn_at": NOW - 60}, now=NOW
    ) is None


def test_stale_but_small_skips():
    assert make_mw().maybe_compact(
        {"messages": msgs(10, 300), "last_turn_at": NOW - 90 * 60}, now=NOW
    ) is None


def test_missing_timestamp_skips():
    assert make_mw().maybe_compact({"messages": msgs(20, 6000)}, now=NOW) is None


def test_before_model_always_stamps():
    patch = make_mw().before_model(
        {"messages": msgs(4, 10), "last_turn_at": NOW - 10}, runtime=None
    )
    assert patch["last_turn_at"] is not None


# --- tool-call pairing (deep agents are tool-heavy) -------------------------

def _tool_turn(i: int, big: int) -> list:
    cid = f"call_{i}"
    return [
        HumanMessage(f"user asks {i}"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": i}, "id": cid}]),
        ToolMessage(content="RESULT " + ("x" * big), tool_call_id=cid),
        AIMessage(f"answer {i}"),
    ]


def test_tool_call_pairs_not_split():
    """A ToolMessage kept without its AIMessage would 400 on resume. Must not happen."""
    conv: list = []
    for i in range(20):
        conv += _tool_turn(i, 6000)   # big, tool-heavy prefix
    for i in range(100, 104):
        conv += _tool_turn(i, 20)     # small recent turns

    patch = make_mw().maybe_compact({"messages": conv, "last_turn_at": NOW - 90 * 60}, now=NOW)
    assert patch and "messages" in patch
    out = [m for m in patch["messages"] if not m.__class__.__name__.startswith("Remove")]

    open_ids: set = set()
    for m in out:
        if isinstance(m, AIMessage):
            open_ids.update(tc["id"] for tc in (m.tool_calls or []))
        elif isinstance(m, ToolMessage):
            assert m.tool_call_id in open_ids, "orphan ToolMessage -> would 400 on resume"


# --- sweep (in-process) -- @timeout catches a deadlock regression -----------

@pytest.mark.timeout(30)
def test_sweep_compacts_only_eligible_in_place():
    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph, mw = build(saver)
        graph.update_state({"configurable": {"thread_id": "big-stale"}},
                           {"messages": msgs(20, 6000), "last_turn_at": NOW - 90 * 60})
        graph.update_state({"configurable": {"thread_id": "big-fresh"}},
                           {"messages": msgs(20, 6000), "last_turn_at": NOW - 60})
        graph.update_state({"configurable": {"thread_id": "small-stale"}},
                           {"messages": msgs(10, 300), "last_turn_at": NOW - 90 * 60})

        compacted = sweep(graph, saver, mw, now=NOW)

        def toks(t):
            return count_tokens_approximately(
                graph.get_state({"configurable": {"thread_id": t}}).values["messages"]
            )

        assert compacted == ["big-stale"]
        assert toks("big-stale") < 10_000
        assert toks("big-fresh") > 25_000
        assert toks("small-stale") < 2_000


@pytest.mark.timeout(30)
def test_sweep_is_idempotent():
    """A second sweep immediately after must NOT re-compact (last_turn_at bumped)."""
    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph, mw = build(saver)
        graph.update_state({"configurable": {"thread_id": "t"}},
                           {"messages": msgs(20, 6000), "last_turn_at": NOW - 90 * 60})
        assert sweep(graph, saver, mw, now=NOW) == ["t"]
        assert sweep(graph, saver, mw, now=NOW) == []  # now fresh -> skipped


# --- cross-process durability ----------------------------------------------

@pytest.mark.timeout(90)
def test_cross_process_durability(tmp_path):
    db = str(tmp_path / "x.sqlite")
    last = None
    for phase in ("seed", "sweep", "verify"):
        last = subprocess.run(
            [sys.executable, "-m", "tests.xproc", phase, db],
            capture_output=True, text=True,
        )
        assert last.returncode == 0, f"{phase} failed:\n{last.stderr}"
    assert "CROSS-PROCESS DURABILITY OK" in last.stdout


@pytest.mark.timeout(120)
def test_console_command_against_full_agent(tmp_path):
    """The installed `idle-compaction-sweep` command compacts a FULL deep agent's
    thread across processes, preserving non-message channels (files)."""
    root = os.path.dirname(os.path.dirname(__file__))
    env = {**os.environ, "SWEEP_DB": str(tmp_path / "full.sqlite"), "PYTHONPATH": root}

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=root, env=env)
        assert r.returncode == 0, f"{cmd} failed:\n{r.stderr}"
        return r

    run([sys.executable, "-m", "tests.full_phases", "seed"])
    # the console command's entry point == idle_compaction.cli:main
    run([sys.executable, "-m", "idle_compaction.cli",
         "--factory", "tests.factory_full:build_for_sweep"])
    out = run([sys.executable, "-m", "tests.full_phases", "verify"]).stdout
    assert "FULL-AGENT SWEEP OK" in out
