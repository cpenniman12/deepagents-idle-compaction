"""Seed / verify phases for the full-agent console-command test (separate procs)."""
import sys
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.middleware.summarization import count_tokens_approximately
from tests.factory_full import build_for_sweep

NOW = 1_700_000_000.0

def msgs(n, c):
    return [(HumanMessage if i % 2 == 0 else AIMessage)(f"t{i}:" + ("x" * c)) for i in range(n)]

phase = sys.argv[1]
graph, saver, mw = build_for_sweep()
cfg = {"configurable": {"thread_id": "big-stale"}}

if phase == "seed":
    graph.update_state(cfg, {"messages": msgs(20, 6000), "last_turn_at": NOW - 90*60,
                             "files": {"notes.txt": "keep me"}})
    st = graph.get_state(cfg).values
    print("SEED tokens:", count_tokens_approximately(st["messages"]), "files:", st.get("files"))
elif phase == "verify":
    st = graph.get_state(cfg).values
    tok = count_tokens_approximately(st["messages"])
    print("VERIFY tokens:", tok, "files:", st.get("files"))
    assert tok < 10_000, "messages not compacted"
    assert st.get("files") == {"notes.txt": "keep me"}, "non-message channel lost!"
    print("FULL-AGENT SWEEP OK")
