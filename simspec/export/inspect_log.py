"""Export a run directory as an Inspect AI EvalLog (one sample per episode, custom InfoEvents for
game events, ModelEvent-like records for model calls, the Harness Card in metadata).

Requires `inspect-ai` (not a dependency of simspec). Verified against inspect-ai 0.3.281
(docs/research/experiments/eval_inspect.py)."""
from __future__ import annotations

import datetime
import os

from ..log import Trajectory


def export_run(run_dir: str, out_path: str | None = None):
    try:
        from inspect_ai.log import EvalLog, EvalSample, EvalSpec, EvalDataset, EvalStats, write_eval_log
        from inspect_ai.event import InfoEvent
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install inspect-ai to export logs") from e
    traj = Trajectory.load(run_dir)
    card = traj.card or {}
    events = []
    for e in traj.events:
        events.append(InfoEvent(source=f"simspec.{e['kind']}", data={k: v for k, v in e.items() if k != "kind"}))
    for c in traj.calls:
        events.append(InfoEvent(source="simspec.model_call", data=c))
    game = card.get("game", {})
    spec = EvalSpec(task=game.get("game_id", "simspec"), task_id=game.get("spec_hash", "spec"), run_id=card.get("run_id", "run"),
                    created=datetime.datetime.now().isoformat(), dataset=EvalDataset(name="episodes", samples=1),
                    model="simspec/multi-agent", config={}, task_attribs={}, task_args={}, model_args={}, packages={},
                    metadata={"harness_card": card})
    sample = EvalSample(id=card.get("run_id", "ep"), epoch=1, input=f"{game.get('game_id')} episode", target="", messages=[], events=events,
                        metadata={"payoffs": traj.payoffs()}, scores={})
    log = EvalLog(version=2, status="success", eval=spec, samples=[sample], stats=EvalStats(started_at="", completed_at=""))
    out_path = out_path or os.path.join(run_dir, "episode.eval")
    write_eval_log(log, out_path)
    return out_path
