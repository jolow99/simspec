"""Can LangGraph host a turn-based game loop with per-step checkpoints, exact replay-by-readback,
and counterfactual forks? Build a 2-player alternating game as a StateGraph."""
import time, random
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class S(TypedDict):
    step: int
    history: list
    rng_state: object
    score: dict

def _t(x):
    return tuple(_t(i) for i in x) if isinstance(x, list) else x

def act(player):
    def node(state: S) -> S:
        rng = random.Random(); rng.setstate(_t(state["rng_state"]))
        a = rng.choice(["COOP", "DEFECT"])
        hist = state["history"] + [(player, a)]
        score = dict(state["score"]); score[player] += 1 if a == "COOP" else 2
        return {"step": state["step"] + 1, "history": hist, "rng_state": rng.getstate(), "score": score}
    return node

def route(state: S):
    return END if state["step"] >= 6 else ("p1" if state["step"] % 2 == 0 else "p2")

g = StateGraph(S)
g.add_node("p1", act("p1")); g.add_node("p2", act("p2"))
g.add_conditional_edges(START, route)
g.add_conditional_edges("p1", route); g.add_conditional_edges("p2", route)
saver = InMemorySaver()
app = g.compile(checkpointer=saver)
cfg = {"configurable": {"thread_id": "ep0"}}
rng = random.Random(0)
t0 = time.time()
out = app.invoke({"step": 0, "history": [], "rng_state": rng.getstate(), "score": {"p1": 0, "p2": 0}}, cfg)
print("run:", out["history"], out["score"], f"{time.time()-t0:.3f}s")
hist = list(app.get_state_history(cfg))
print("checkpoints:", len(hist), [(h.metadata.get("step"), h.metadata.get("source"), h.next) for h in hist][:8])
# time travel: fork from step 3 with a modified history (counterfactual)
cp = [h for h in hist if h.metadata.get("step") == 3][0]
new_cfg = app.update_state(cp.config, {"score": {"p1": 100, "p2": 0}})
out2 = app.invoke(None, new_cfg)
print("fork:", out2["history"], out2["score"])
print("original thread still:", app.get_state(cfg).values["score"])
# serialization of a checkpoint
import json
ck = saver.get_tuple(cfg)
print("checkpoint keys:", list(ck.checkpoint.keys()), "channel_values keys:", list(ck.checkpoint["channel_values"].keys()))
