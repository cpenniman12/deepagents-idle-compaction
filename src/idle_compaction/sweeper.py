"""Out-of-band compaction sweeper -- fires during idle, no LangGraph Platform.

Run this on a schedule (e.g. OS cron every few minutes). For each thread in the
checkpointer it checks the SAME gate the middleware uses -- idle >= idle_seconds
AND tokens >= token_threshold -- and, when both hold, rewrites that thread's
persisted state to the compacted version *in place*. The user then resumes on an
already-small context; the big block is never re-written into cache on resume.

This is the piece that makes compaction fire AT the idle mark instead of on the
next resume: something has to be running during idle, and this sweeper is it.

Requires a persistent checkpointer (e.g. SqliteSaver / PostgresSaver) so state
survives between the user's turn and the sweep.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

from idle_compaction.middleware import IdleCompactionMiddleware


def iter_thread_ids(checkpointer: Any) -> Iterator[str]:
    """Yield each distinct thread_id known to the checkpointer, most-recent first."""
    seen: set[str] = set()
    for ct in checkpointer.list(None):
        tid = ct.config.get("configurable", {}).get("thread_id")
        if tid and tid not in seen:
            seen.add(tid)
            yield tid


def sweep(
    graph: Any,
    checkpointer: Any,
    middleware: IdleCompactionMiddleware,
    *,
    now: float | None = None,
) -> list[str]:
    """Compact every idle+large thread in place. Returns the ids compacted.

    Args:
        graph: The compiled agent (used for ``get_state`` / ``update_state``).
        checkpointer: The persistent checkpointer (used to enumerate threads).
        middleware: The configured ``IdleCompactionMiddleware`` -- its gate and
            summarizer are reused so the sweep matches in-session behavior exactly.
        now: Clock override for testing.
    """
    now = time.time() if now is None else now
    compacted: list[str] = []

    # Drain the checkpointer's read cursor FULLY before writing anything back.
    # SqliteSaver.list() streams over an open cursor on the same connection;
    # calling update_state() (a write) while that cursor is still open
    # deadlocks. Materializing the ids first closes the read before any write.
    thread_ids = list(iter_thread_ids(checkpointer))

    for tid in thread_ids:
        config = {"configurable": {"thread_id": tid}}
        state = graph.get_state(config).values

        patch = middleware.maybe_compact(state, now=now)
        if not patch:
            # Not idle enough or not large enough. Crucially, do NOT touch
            # last_turn_at here -- that would reset the idle clock and starve
            # the thread of ever being compacted.
            continue

        # We did compact -> stamp so the freshly-summarized state isn't seen as
        # stale again on the next sweep.
        graph.update_state(config, {**patch, "last_turn_at": now})
        compacted.append(tid)

    return compacted
