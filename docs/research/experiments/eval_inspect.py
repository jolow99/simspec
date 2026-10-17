"""Can Inspect AI's log/transcript be used as the trajectory log of a multi-agent game loop
that is NOT an Inspect Task (scripted agents, no LLM), and can episodes be read back for
offline metrics? Probe the public API."""
import inspect_ai, inspect
from inspect_ai import log as L
print("inspect_ai", inspect_ai.__version__)
names = [n for n in dir(L) if not n.startswith("_")]
print("inspect_ai.log exports:", names)

from inspect_ai.log import EvalLog, EvalSample, EvalSpec, EvalDataset, write_eval_log, read_eval_log
import inspect_ai.log as logmod
# Event types available?
ev = [n for n in names if n.endswith("Event")]
print("event types:", ev)
print("EvalSample fields:", list(EvalSample.model_fields))
print("EvalSpec fields:", list(EvalSpec.model_fields))
try:
    from inspect_ai.log import InfoEvent, StateEvent, ModelEvent, ScoreEvent, SampleInitEvent
    print("InfoEvent fields:", list(InfoEvent.model_fields))
    print("StateEvent fields:", list(StateEvent.model_fields))
    print("ModelEvent fields:", list(ModelEvent.model_fields))
except Exception as e:
    print("event import failed:", e)

# Try to construct a minimal EvalLog with one sample whose events are a hand-built game transcript.
import datetime, tempfile, os
from inspect_ai.log import EvalResults, EvalStats
from inspect_ai.log import InfoEvent
from inspect_ai.model import ModelOutput, ChatMessageUser, ChatMessageAssistant
events = []
for step in range(3):
    events.append(InfoEvent(source="game", data={"step": step, "actor": f"p{step%2}", "action": "COOP", "obs": {"round": step}}))
spec = EvalSpec(
    task="contractsim_catering", task_id="t1", run_id="r1", created=datetime.datetime.now().isoformat(),
    dataset=EvalDataset(name="episodes", samples=1), model="none/none",
    config={}, task_attribs={}, task_args={}, model_args={}, packages={}, metadata={"harness_card": {"game": "x"}},
)
sample = EvalSample(id="ep0", epoch=1, input="episode 0", target="", messages=[], events=events,
                    metadata={"seed": 0}, scores={})
log = EvalLog(version=2, status="success", eval=spec, samples=[sample],
              stats=EvalStats(started_at="", completed_at=""))
d = tempfile.mkdtemp()
path = os.path.join(d, "ep.eval")
write_eval_log(log, path)
back = read_eval_log(path)
print("roundtrip ok:", back.eval.task, len(back.samples), [type(e).__name__ for e in back.samples[0].events])
print("custom metadata:", back.eval.metadata)
# header-only read
hdr = read_eval_log(path, header_only=True)
print("header_only samples:", hdr.samples)
# Also test json format
path2 = os.path.join(d, "ep.json")
write_eval_log(log, path2)
print("json size", os.path.getsize(path2), "eval size", os.path.getsize(path))
