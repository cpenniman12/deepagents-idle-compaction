"""Shared agent factory for the cross-process durability test."""
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from deepagents import create_deep_agent
from idle_compaction import IdleCompactionMiddleware

NOW = 2_000_000_000.0  # fixed clock so phases in different processes agree

def msgs(n, c):
    return [(HumanMessage if i % 2 == 0 else AIMessage)(f"t{i}:" + ("x" * c)) for i in range(n)]

def make(saver):
    fake = FakeListChatModel(responses=["<COMPACTED SUMMARY>"] * 20)
    mw = IdleCompactionMiddleware(fake, token_threshold=20_000, idle_seconds=59 * 60, keep=("messages", 6))
    graph = create_deep_agent(model=fake, tools=[], middleware=[mw], checkpointer=saver)
    return graph, mw
