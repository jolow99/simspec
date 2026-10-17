"""Typed JSONL trajectory log.

One directory per run:
  card.json     Harness Card
  events.jsonl  typed events, in order (the source of truth)
  calls.jsonl   raw model requests/responses (content-addressed by call_ref)
  scores.json   written by offline metrics, never by the run
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator


EVENT_KINDS = ("episode_start", "phase_start", "observation", "agent_call", "action", "chance",
               "transition", "rule", "message", "payoff", "error", "episode_end", "info")


class RunWriter:
    def __init__(self, run_dir: str | None):
        self.run_dir = run_dir
        self.events: list[dict] = []
        self.calls: list[dict] = []
        self._ev_fh = None
        self._call_fh = None
        if run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            self._ev_fh = open(os.path.join(run_dir, "events.jsonl"), "w")
            self._call_fh = open(os.path.join(run_dir, "calls.jsonl"), "w")

    def event(self, kind: str, **data: Any) -> dict:
        assert kind in EVENT_KINDS, kind
        rec = {"i": len(self.events), "kind": kind, **data}
        self.events.append(rec)
        if self._ev_fh is not None:
            self._ev_fh.write(json.dumps(rec, default=_default) + "\n")
            self._ev_fh.flush()
        return rec

    def call(self, request: Any, response: Any, meta: dict | None = None) -> str:
        blob = json.dumps({"request": request, "response": response}, sort_keys=True, default=_default)
        ref = hashlib.sha256(blob.encode()).hexdigest()[:16]
        rec = {"ref": ref, "request": request, "response": response, "meta": meta or {}}
        self.calls.append(rec)
        if self._call_fh is not None:
            self._call_fh.write(json.dumps(rec, default=_default) + "\n")
            self._call_fh.flush()
        return ref

    def write_card(self, card: dict) -> None:
        if self.run_dir is not None:
            with open(os.path.join(self.run_dir, "card.json"), "w") as fh:
                json.dump(card, fh, indent=2, default=_default, sort_keys=True)

    def close(self) -> None:
        for fh in (self._ev_fh, self._call_fh):
            if fh is not None:
                fh.close()


def _default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "model_dump"):
        return o.model_dump()
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return repr(o)


@dataclass
class Trajectory:
    """In-memory view of one episode's events with query helpers. Built from a RunWriter or a run dir."""
    events: list[dict]
    calls: list[dict] = field(default_factory=list)
    card: dict | None = None

    @classmethod
    def load(cls, run_dir: str) -> "Trajectory":
        with open(os.path.join(run_dir, "events.jsonl")) as fh:
            events = [json.loads(l) for l in fh if l.strip()]
        calls = []
        cp = os.path.join(run_dir, "calls.jsonl")
        if os.path.exists(cp):
            with open(cp) as fh:
                calls = [json.loads(l) for l in fh if l.strip()]
        card = None
        cj = os.path.join(run_dir, "card.json")
        if os.path.exists(cj):
            with open(cj) as fh:
                card = json.load(fh)
        return cls(events, calls, card)

    def of(self, kind: str) -> Iterator[dict]:
        return (e for e in self.events if e["kind"] == kind)

    def states(self) -> list[dict]:
        return [e["state"] for e in self.of("transition")]

    def initial_state(self) -> dict | None:
        e = next(self.of("episode_start"), None)
        return e["state"] if e else None

    def actions(self, role: str | None = None) -> list[dict]:
        return [e for e in self.of("action") if role is None or e["role"] == role]

    def rule_events(self, role: str | None = None, rule_id: str | None = None) -> list[dict]:
        return [e for e in self.of("rule") if (role is None or e["role"] == role) and (rule_id is None or e["rule_id"] == rule_id)]

    def chance_draws(self) -> list[dict]:
        return [{"name": e["name"], "value": e["value"]} for e in self.of("chance")]

    def payoffs(self) -> dict[str, float] | None:
        e = next(self.of("episode_end"), None)
        return e["payoffs"] if e else None

    def final_state(self) -> dict | None:
        st = self.states()
        return st[-1] if st else self.initial_state()

    def steps(self) -> int:
        return len(self.states())

    def agent_calls(self, role: str | None = None) -> list[dict]:
        return [e for e in self.of("agent_call") if role is None or e["role"] == role]

    def messages(self, channel: str | None = None) -> list[dict]:
        return [e for e in self.of("message") if channel is None or e["channel"] == channel]
