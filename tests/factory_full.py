"""A FULL deep agent (real tool + default middleware) behind a sweep factory.

Used to prove the console command compacts threads written by a complete agent,
and that other state channels (files/todos) survive compaction.
"""
import os, sqlite3
from langchain_core.language_models import FakeListChatModel
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from deepagents import create_deep_agent
from idle_compaction import IdleCompactionMiddleware

@tool
def search(q: str) -> str:
    """Pretend search tool."""
    return f"results for {q}"

def _model():
    return FakeListChatModel(responses=["<COMPACTED SUMMARY>"] * 40)

def build_for_sweep():
    db = os.environ["SWEEP_DB"]  # read lazily so import order doesn't matter
    saver = SqliteSaver(sqlite3.connect(db, check_same_thread=False))
    mw = IdleCompactionMiddleware(_model(), token_threshold=20_000, idle_seconds=59*60, keep=("messages", 6))
    graph = create_deep_agent(model=_model(), tools=[search], middleware=[mw], checkpointer=saver)
    return graph, saver, mw
