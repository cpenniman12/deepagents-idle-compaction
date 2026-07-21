"""Idle + size triggered conversation compaction for LangChain / deepagents."""

from idle_compaction.middleware import (
    DEFAULT_IDLE_SECONDS,
    DEFAULT_KEEP,
    DEFAULT_TOKEN_THRESHOLD,
    CompactionState,
    IdleCompactionMiddleware,
)
from idle_compaction.sweeper import iter_thread_ids, sweep

__all__ = [
    "IdleCompactionMiddleware",
    "CompactionState",
    "sweep",
    "iter_thread_ids",
    "DEFAULT_TOKEN_THRESHOLD",
    "DEFAULT_IDLE_SECONDS",
    "DEFAULT_KEEP",
]
