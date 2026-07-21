"""Template: the one factory function the `idle-compaction-sweep` command needs.

Copy this into your app (e.g. myapp/agent.py), edit the two marked spots, then
schedule the installed console command -- no wrapper script to maintain:

    */5 * * * * cd /path/to/app && /path/to/.venv/bin/idle-compaction-sweep \
        --factory myapp.agent:build_for_sweep >> /tmp/sweep.log 2>&1

Requirements for at-59-min compaction:
  1. Your app's agent uses the SAME persistent checkpointer + DB as this factory.
  2. Your app's agent has IdleCompactionMiddleware attached (stamps last_turn_at).
"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from idle_compaction import IdleCompactionMiddleware

# --- EDIT 1: same model your agent runs on (used for the summaries) ------------
MODEL = "anthropic:claude-opus-4-8"

# --- EDIT 2: same persistent DB your app writes to -----------------------------
DB_PATH = "/path/to/app/agent_state.sqlite"


def build_for_sweep():
    """Return (graph, checkpointer, middleware) -- built exactly like your app.

    The console command calls this, runs one sweep, and exits.
    """
    from deepagents import create_deep_agent

    saver = SqliteSaver(sqlite3.connect(DB_PATH, check_same_thread=False))
    mw = IdleCompactionMiddleware(
        MODEL,
        token_threshold=300_000,
        idle_seconds=59 * 60,
        keep=("messages", 12),
    )
    graph = create_deep_agent(
        model=MODEL,
        tools=[],  # <- your real tools
        middleware=[mw],  # <- plus your other middleware
        checkpointer=saver,
    )
    return graph, saver, mw
