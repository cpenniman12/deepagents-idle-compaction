# deepagents-idle-compaction

One of the most expensive operations your agent will run is writing a large context
back into the cache.

Anthropic models charge 2× to write into a 1-hour cache, then 0.1× to read from it.
So if you resume a session with 500k tokens after an hour+ of inactivity, the cache
has expired — and writing that context back in costs about **$5** on Opus, just to
pick up where you left off.

One solution I landed on is **time-based compaction**. A cron job kicks off compaction
right before the cache expires. It doesn't stop the cache from expiring an hour later —
but when you resume, you're writing a small compacted summary back into the cache
instead of the full 500k.

The time before compaction is configurable, and so is a token threshold — I set mine to
300k, so this doesn't fire on every conversation that happens to be sitting idle.

Right now this middleware is only compatible with **LangGraph / Deep Agents**.

Trigger: `idle >= idle_seconds` **AND** `tokens >= token_threshold` (defaults: 59 min,
300k tokens). Summaries run on the **same model** the agent uses; the most recent turns
are kept verbatim (partial compaction), and compaction is a no-op during active use — so
the live cache and recent-turn fidelity stay untouched.

## The two parts (both needed for at-N-min compaction)

Something has to run *during* idle to compact at the N-minute mark — a library
alone can't, because none of its code is running while the thread is idle. So:

1. **Middleware** on your agent — stamps `last_turn_at` every turn and carries the
   state. Also compacts on resume as a fallback.
2. **A scheduled sweep job** (plain OS cron) — the piece that actually fires during
   idle. It reads each thread from a **persistent checkpointer** and compacts the
   idle+large ones in place.

> Without the sweep job you still get compaction, but only **on the next resume**
> (cold read). With it, compaction happens at the idle mark and the resume lands
> instantly on the small context. No LangGraph Platform required.

## Install

```bash
pip install deepagents-idle-compaction
```

## Setup

**1 — attach the middleware and use a persistent checkpointer:**

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from idle_compaction import IdleCompactionMiddleware

model = "anthropic:claude-opus-4-8"

with SqliteSaver.from_conn_string("agent_state.sqlite") as saver:
    agent = create_deep_agent(
        model=model,
        tools=[...],
        middleware=[
            IdleCompactionMiddleware(
                model,                    # same model the agent runs on
                token_threshold=300_000,  # only compact large contexts
                idle_seconds=59 * 60,     # only compact stale contexts
                keep=("messages", 12),    # preserve the last 12 turns verbatim
            ),
        ],
        checkpointer=saver,               # REQUIRED: state must persist
    )
```

**2 — schedule the sweep.** `pip install` also installs the `idle-compaction-sweep`
command. You give it a one-function factory (see `sweep_job.example.py`) that
rebuilds your agent the same way, then add one crontab line (`crontab -e`):

```cron
*/5 * * * * cd /path/to/app && /path/to/.venv/bin/idle-compaction-sweep \
    --factory myapp.agent:build_for_sweep >> /tmp/sweep.log 2>&1
```

```python
# myapp/agent.py
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from idle_compaction import IdleCompactionMiddleware

def build_for_sweep():
    """Return (graph, checkpointer, middleware) -- same construction as your app."""
    from deepagents import create_deep_agent
    saver = SqliteSaver(sqlite3.connect("agent_state.sqlite", check_same_thread=False))
    mw = IdleCompactionMiddleware("anthropic:claude-opus-4-8",
                                  token_threshold=300_000, idle_seconds=59 * 60)
    graph = create_deep_agent(model="anthropic:claude-opus-4-8", tools=[...],
                              middleware=[mw], checkpointer=saver)
    return graph, saver, mw
```

That's it. Idle threads get compacted within ~5 minutes of crossing 59 minutes.

> **Why a factory and not zero-config?** The sweep runs in a separate process from
> your app and has to rebuild your agent (your tools/model/checkpointer) to read and
> write thread state — a generic command can't infer that. The command ships with the
> package; you supply one function. Verified across processes against a full deep
> agent (tools + default middleware), preserving non-message state like `files`.

## What you actually save

- **Eliminated: the 2× re-cache of the big block on resume.** The dominant cost.
  After compaction the thread is a few thousand tokens, so the resume never
  re-writes 300k+ tokens into cache, and every subsequent turn reads a small
  context instead of a huge one. This holds in both modes.
- **Instant resume (sweep mode).** Compaction already happened during idle, so the
  user's resume turn isn't blocked on summarizing.
- **One summary read, unavoidable either way.** Summarizing the block requires
  reading it once. Whether that read is warm (~0.1×) depends on the summarization
  call reusing the cached prefix; don't count on it — the guaranteed win is the
  re-cache elimination above.

## Modes

| | fires | summary read | resume |
| --- | --- | --- | --- |
| middleware only | on next resume | cold | pays the summary read, then cheap |
| middleware + sweep | at the idle mark | during idle | instant, already small |

## Config

| arg | default | meaning |
| --- | --- | --- |
| `token_threshold` | `300_000` | minimum tokens before compaction is eligible |
| `idle_seconds` | `59 * 60` | minimum idle since the previous turn |
| `keep` | `("messages", 12)` | recent messages kept verbatim (partial compaction) |
| `time_fn` | `time.time` | clock, injectable for tests |
| `**summarization_kwargs` | — | forwarded to `SummarizationMiddleware` (e.g. `summary_prompt`) |

## How it works

`IdleCompactionMiddleware` subclasses the stock `SummarizationMiddleware`, which
already does same-model partial summarization gated on a token count. This subclass
adds the **idle gate** and exposes `maybe_compact(state)` — the gate + summarize
step *without* stamping `last_turn_at`, so the out-of-band sweeper can reuse the
exact in-session logic without resetting a thread's idle clock. `sweep()` enumerates
threads from the checkpointer and applies `maybe_compact` to each, writing the
compacted state back via `graph.update_state`.

## Durability notes

- **Sweep drains the checkpointer's read cursor before writing.** SqliteSaver's
  `list()` streams over an open cursor on the same connection; writing through it
  deadlocks. `sweep()` materializes thread ids first. (Regression-tested.)
- **The idle gate protects against races.** A thread being actively used has a
  recent `last_turn_at`, so the sweep skips it — the window for a sweep to collide
  with a live turn is tiny.
- **Threads without `last_turn_at` are skipped**, so pre-existing threads (created
  before installing the middleware) are never touched until they take a turn.

## Try it (offline, no API key)

```bash
pip install -e ".[demo]"
python demo.py           # in-session middleware: 386k -> ~1k tokens
python sweeper_demo.py   # idle/cron sweep, in place, on the right thread
pytest                   # gating + deadlock regression + cross-process durability
```
