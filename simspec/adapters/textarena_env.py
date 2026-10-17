"""TextArena-style adapter (sketch, runnable without textarena installed).

TextArena's contract is `env.reset(num_players)`, `pid, observation = env.get_observation()`,
`done, info = env.step(action: str)`, `rewards = env.close()`, with observations as text. A GameSpec
maps onto it by rendering the acting role's Observation with the game's renderer and parsing the
returned string through the same parser the LLM scaffold uses. The sequential/protocol phases map
directly; a simultaneous phase is emulated by collecting one string per player and resolving the
round once all are in (which is also how TextArena implements IPD).

What is cheap: everything above (this file). What is hard: TextArena has no observation structure
beyond per-player inboxes and no rules, so nothing of the spec's enforcement survives the export;
rules would only be enforced by our engine underneath, invisibly to TextArena agents.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from ..spec import GameSpec, Observation, ActionSpec
from ..runtime import Engine, Vault
from ..agents.base import ActionDistribution, BaseAgent
from ..agents.llm import _extract_json


class _Relay(BaseAgent):
    def __init__(self, role):
        self.role = role
        self.pending: str | None = None

    def act(self, obs, spec, ctx):
        text = self.pending or ""
        self.pending = None
        if spec.kind == "text":
            return ActionDistribution.of_text(text, source="scripted")
        obj = _extract_json(text)
        if spec.kind == "choice":
            key = (obj or {}).get("action", text.strip())
            keys = obs.legal or spec.keys()
            return ActionDistribution.point(key if key in keys else keys[0], source="scripted")
        val = (obj or {}).get("action", obj) if obj else spec.default
        return ActionDistribution.point(spec.key_of(val if spec.validate(val) else spec.default), source="scripted")

    def describe(self, seat):
        from ..card import ScaffoldSection
        return ScaffoldSection(seat=seat, role=self.role, agent_class="textarena-relay", distribution_source="scripted")


class TextArenaGameSpecEnv:
    """Sequential/protocol phases map directly; simultaneous phases collect one action per player and resolve when all are in."""

    def __init__(self, spec: GameSpec, renderers: dict[str, Callable[[Observation], str]], seed: int = 0):
        self.spec = spec
        self.renderers = renderers
        self.seed = seed
        self.relays = {r: _Relay(r) for r in spec.roles}
        self.pid_of = {r: i for i, r in enumerate(spec.roles)}

    def reset(self, num_players: int | None = None, seed: int | None = None):
        self.eng = Engine(self.spec, dict(self.relays), seed=self.seed if seed is None else seed)
        eng = self.eng
        state = self.spec.initial(self.spec.params, eng.chance)
        self.vault = Vault(state)
        eng._flush_chance()
        self.phase = self.spec.phases[0]
        self.round = 0
        self.queue = list(self.phase.actors_in(state))
        self.pending_joint: dict[str, Any] = {}
        self.done = False

    def get_observation(self) -> tuple[int, str]:
        role = self.queue[0]
        aspec = self.phase.action_for(role)
        obs = self.spec.observe(self.vault.state, role, self.eng.step, self.phase.name, self.eng.messages, aspec.legal_keys(self.vault.state, role))
        return self.pid_of[role], self.renderers.get(role, lambda o: json.dumps(o.to_dict()))(obs)

    def step(self, action: str) -> tuple[bool, dict]:
        role = self.queue.pop(0)
        self.relays[role].pending = action
        key, value = self.eng._elicit(self.vault, self.phase, role, self.round)
        if self.phase.kind == "simultaneous":
            self.pending_joint[role] = value
            if self.queue:
                return False, {"step": self.eng.step, "waiting": list(self.queue)}
            self.eng._apply(self.vault, self.phase, dict(self.pending_joint), self.round, actor=None)
            self.pending_joint = {}
        else:
            self.eng._apply(self.vault, self.phase, {role: value}, self.round, actor=role)
        s = self.vault.state
        if self.spec.terminal(s):
            self.done = True
        elif not self.queue:
            self.round += 1
            if (self.phase.until and self.phase.until(s)) or (self.phase.max_rounds and self.round >= self.phase.max_rounds):
                nxt = self.phase.next_phase(s)
                if nxt is None:
                    self.done = True
                else:
                    self.phase = self.spec.phase(nxt)
                    self.round = 0
            if not self.done:
                self.queue = list(self.phase.actors_in(s))
        return self.done, {"step": self.eng.step}

    def close(self) -> dict[int, float]:
        from ..log import Trajectory
        traj = Trajectory(self.eng.writer.events, self.eng.writer.calls, None)
        pay = self.spec.payoff(traj) if self.spec.payoff else {}
        return {self.pid_of[r]: v for r, v in pay.items()}
