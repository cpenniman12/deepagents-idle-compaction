"""Idle + size triggered conversation compaction for LangChain / deepagents.

Standard summarization middleware fires on token count alone. This subclass adds
the *idle-time* half of the trigger: it only compacts when the conversation is
BOTH large (>= ``token_threshold`` tokens) AND stale (idle >= ``idle_seconds``).

Why: the expensive case for prompt caching is a big context that sits idle past
the cache TTL and is then resumed -- the whole block gets re-read/re-cached at
full price. Collapsing that block to a short summary while it's stale makes the
resume (and every turn after) cheap. Compaction is a no-op during active use,
which protects the live cache and recent-turn fidelity.

The size half is delegated to the parent ``SummarizationMiddleware`` (its token
trigger). This class contributes the idle half and stamps ``last_turn_at`` into
state every turn so idle can be measured -- LangGraph messages are not
wall-clock stamped.
"""

from __future__ import annotations

import time
from typing import Any, Callable, NotRequired

from langchain.agents.middleware import AgentState, Runtime, SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_TOKEN_THRESHOLD = 300_000
DEFAULT_IDLE_SECONDS = 59 * 60  # 59 minutes -- just under the 1h cache TTL
DEFAULT_KEEP: tuple[str, int] = ("messages", 12)


class CompactionState(AgentState):
    """Agent state extended with a wall-clock stamp of the last model turn.

    ``last_turn_at`` is a Unix timestamp written on every ``before_model`` pass.
    On the next pass we compare it to now to measure how long the thread sat idle.
    """

    last_turn_at: NotRequired[float]


class IdleCompactionMiddleware(SummarizationMiddleware):
    """Compact the conversation only when it is both large and stale.

    Fires partial summarization when ``idle >= idle_seconds`` AND
    ``tokens >= token_threshold``. Summarization runs on the same model the agent
    uses and keeps the most recent ``keep`` messages verbatim (partial
    compaction), so only the stale prefix is collapsed.

    Args:
        model: The chat model (or model id) used for summarization -- pass the
            same model the agent runs on.
        token_threshold: Minimum token count before compaction is eligible.
        idle_seconds: Minimum idle time (since the previous turn) before
            compaction is eligible. Defaults to 59 minutes.
        keep: How many recent messages to preserve verbatim, e.g.
            ``("messages", 12)``. Forwarded to ``SummarizationMiddleware``.
        time_fn: Clock function, injectable for testing. Defaults to
            ``time.time``.
        **summarization_kwargs: Forwarded to ``SummarizationMiddleware``
            (e.g. ``summary_prompt``, ``token_counter``).

    Note:
        In library mode nothing runs during idle, so compaction is detected and
        performed on the *resume* turn (the summarization read is cold). That
        still makes the resume's continuation and every future resume cheap. For
        compaction *during* idle while the cache is still warm, drive the same
        trigger from a LangGraph Platform cron instead.
    """

    state_schema = CompactionState

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        token_threshold: int = DEFAULT_TOKEN_THRESHOLD,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        keep: tuple[str, int] = DEFAULT_KEEP,
        time_fn: Callable[[], float] = time.time,
        **summarization_kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            trigger=("tokens", token_threshold),
            keep=keep,
            **summarization_kwargs,
        )
        self.token_threshold = token_threshold
        self.idle_seconds = idle_seconds
        self._time_fn = time_fn

    def maybe_compact(
        self, state: CompactionState, *, now: float | None = None
    ) -> dict[str, Any] | None:
        """Return a compacted-messages patch if the thread is idle AND large.

        Applies the idle gate here and delegates the size gate + summarization to
        the parent. Returns ``None`` (no messages key) when either gate fails.

        This deliberately does NOT stamp ``last_turn_at`` -- so it is safe to call
        from an out-of-band sweeper without resetting a thread's idle clock. The
        caller decides whether/when to re-stamp.
        """
        now = self._time_fn() if now is None else now
        last = state.get("last_turn_at")

        # No prior stamp -> idle is undefined. Never idle enough yet.
        if last is None or now - last < self.idle_seconds:
            return None

        # Idle gate passed. Parent applies the size gate (its token trigger) and
        # returns None when the thread is below token_threshold.
        return super().before_model(state, runtime=None)  # type: ignore[arg-type]

    def before_model(
        self, state: CompactionState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        now = self._time_fn()
        patch: dict[str, Any] = {"last_turn_at": now}

        # A turn is happening now, so stamping is correct here (unlike the sweeper).
        compaction = self.maybe_compact(state, now=now)
        if compaction:
            patch.update(compaction)
        return patch
