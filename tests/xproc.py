"""One phase of the cross-process test. Each invocation is a fresh process."""
import sys
from langchain.agents.middleware.summarization import count_tokens_approximately
from langgraph.checkpoint.sqlite import SqliteSaver
from tests._shared import NOW, msgs, make
from idle_compaction.sweeper import sweep

phase, db = sys.argv[1], sys.argv[2]

def toks(graph, t):
    return count_tokens_approximately(graph.get_state({"configurable": {"thread_id": t}}).values["messages"])

with SqliteSaver.from_conn_string(db) as saver:
    graph, mw = make(saver)
    if phase == "seed":
        graph.update_state({"configurable": {"thread_id": "big-stale"}}, {"messages": msgs(20, 6000), "last_turn_at": NOW - 90*60})
        graph.update_state({"configurable": {"thread_id": "big-fresh"}}, {"messages": msgs(20, 6000), "last_turn_at": NOW - 120})
        graph.update_state({"configurable": {"thread_id": "small-stale"}}, {"messages": msgs(10, 300), "last_turn_at": NOW - 90*60})
        print("SEED  pid", __import__("os").getpid(), "-> big-stale:", toks(graph,"big-stale"), "big-fresh:", toks(graph,"big-fresh"), "small-stale:", toks(graph,"small-stale"))
    elif phase == "sweep":
        comp = sweep(graph, saver, mw, now=NOW)
        print("SWEEP pid", __import__("os").getpid(), "-> compacted:", comp)
    elif phase == "verify":
        res = {t: toks(graph, t) for t in ["big-stale","big-fresh","small-stale"]}
        print("VERIFY pid", __import__("os").getpid(), "->", res)
        assert res["big-stale"] < 10_000, "big-stale not compacted across processes"
        assert res["big-fresh"] > 25_000, "big-fresh wrongly touched"
        assert res["small-stale"] < 2_000, "small-stale wrongly touched"
        print("CROSS-PROCESS DURABILITY OK")
