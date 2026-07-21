"""``idle-compaction-sweep`` -- the console command cron runs.

Installed with the package, so scheduling the sweep is: write one factory
function, then one crontab line. The command can't rebuild your agent for you
(it doesn't know your tools/model/checkpointer), so you point it at a factory:

    def build_for_sweep():
        \"\"\"Return (graph, checkpointer, IdleCompactionMiddleware).\"\"\"
        ...
        return graph, saver, mw

Then:

    idle-compaction-sweep --factory myapp.agent:build_for_sweep

Crontab (every 5 min):

    */5 * * * * cd /app && /app/.venv/bin/idle-compaction-sweep \\
        --factory myapp.agent:build_for_sweep >> /tmp/sweep.log 2>&1
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from typing import Any

from idle_compaction.sweeper import sweep


def _load_factory(spec: str) -> Any:
    """Import a ``module.path:callable`` factory reference."""
    if ":" not in spec:
        raise SystemExit(
            f"--factory must be 'module.path:callable', got {spec!r}"
        )
    module_path, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:  # noqa: BLE001
        raise SystemExit(f"could not import {module_path!r}: {e}") from e
    try:
        return getattr(module, attr)
    except AttributeError as e:  # noqa: BLE001
        raise SystemExit(f"{module_path!r} has no attribute {attr!r}") from e


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="idle-compaction-sweep")
    parser.add_argument(
        "--factory",
        required=True,
        help="'module.path:callable' returning (graph, checkpointer, middleware)",
    )
    args = parser.parse_args(argv)

    factory = _load_factory(args.factory)
    built = factory()
    try:
        graph, checkpointer, middleware = built
    except (TypeError, ValueError) as e:  # noqa: BLE001
        raise SystemExit(
            "factory must return (graph, checkpointer, middleware); "
            f"got {type(built).__name__}"
        ) from e

    started = time.time()
    compacted = sweep(graph, checkpointer, middleware)
    elapsed = time.time() - started

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if compacted:
        print(f"[{stamp}] idle-compaction: compacted {len(compacted)} thread(s) "
              f"in {elapsed:.1f}s: {compacted}")
    else:
        print(f"[{stamp}] idle-compaction: nothing eligible ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
