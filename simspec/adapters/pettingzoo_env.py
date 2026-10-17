"""PettingZoo ParallelEnv adapter for a single-phase simultaneous GameSpec with finite choice actions.

Cheap: the engine already has PettingZoo's shape (possible_agents, observe(agent), step(dict),
action masks). The adapter drives `simspec.runtime.Engine` one round at a time and exposes
observations as dicts (not Gymnasium spaces; wrap with `gymnasium.spaces.Dict` if a trainer needs
them). Rules, chance logging and the Harness Card keep working underneath.

Limits: one simultaneous phase only (sequential/protocol phases would need the AEC API); record and
text actions are not representable as a Discrete space.
"""
from __future__ import annotations

from typing import Any

from ..spec import GameSpec, ActionSpec
from ..runtime import Engine
from ..agents.base import ActionDistribution, ActContext, BaseAgent


class _Relay(BaseAgent):
    """An agent whose action is injected from outside (the PettingZoo caller)."""

    def __init__(self, role: str):
        self.role = role
        self.next_key: str | None = None
        self.last_obs = None

    def act(self, obs, spec, ctx):
        self.last_obs = obs
        key = self.next_key if self.next_key is not None else (obs.legal[0] if obs.legal else spec.keys()[0])
        return ActionDistribution.point(key, source="scripted")

    def describe(self, seat):
        from ..card import ScaffoldSection
        return ScaffoldSection(seat=seat, role=self.role, agent_class="pettingzoo-relay", distribution_source="scripted")


class ParallelGameSpecEnv:
    metadata = {"name": "simspec_parallel"}

    def __init__(self, spec: GameSpec, seed: int = 0, run_dir: str | None = None):
        phases = spec.phases
        if len(phases) != 1 or phases[0].kind != "simultaneous" or not isinstance(phases[0].action, ActionSpec) or phases[0].action.kind != "choice":
            raise ValueError("ParallelGameSpecEnv supports a single simultaneous phase with a finite choice action")
        self.spec = spec
        self.phase = phases[0]
        self.aspec = self.phase.action
        self.possible_agents = list(spec.roles)
        self.agents = list(spec.roles)
        self.action_keys = self.aspec.keys()
        self.seed = seed
        self.run_dir = run_dir
        self._relays = {r: _Relay(r) for r in self.possible_agents}
        self._engine: Engine | None = None
        self._vault = None
        self._round = 0

    # -- PettingZoo-ish API --------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        from ..runtime import Vault
        self.seed = self.seed if seed is None else seed
        self._engine = Engine(self.spec, dict(self._relays), seed=self.seed, run_dir=self.run_dir)
        eng = self._engine
        card = eng.build_card()
        eng.writer.write_card(card.to_json())
        state = self.spec.initial(self.spec.params, eng.chance)
        self.spec.state.validate_state(state)
        self._vault = Vault(state)
        eng._flush_chance()
        eng.writer.event("episode_start", seed=self.seed, card_hash=card.card_hash, run_id=eng.run_id, state=state.to_dict(), rules=[r.id for r in self.spec.rules])
        self.agents = list(self.possible_agents)
        self._round = 0
        return self._observations(), {a: {} for a in self.agents}

    def _observations(self) -> dict[str, dict]:
        obs = {}
        for r in self.agents:
            legal = self.aspec.legal_keys(self._vault.state, r)
            o = self.spec.observe(self._vault.state, r, self._engine.step, self.phase.name, self._engine.messages, legal)
            obs[r] = {"observation": dict(o), "action_mask": [1 if (legal is None or k in legal) else 0 for k in self.action_keys]}
        return obs

    def step(self, actions: dict[str, int | str]):
        eng = self._engine
        for r, a in actions.items():
            self._relays[r].next_key = self.action_keys[a] if isinstance(a, int) else a
        eng.writer.event("phase_start", step=eng.step, phase=self.phase.name, round=self._round)
        joint = {}
        for r in self.agents:
            key, value = eng._elicit(self._vault, self.phase, r, self._round)
            joint[r] = value
        eng._apply(self._vault, self.phase, joint, self._round, actor=None)
        self._round += 1
        last = eng.writer.events[-1]
        rewards = {r: float(last.get("rewards", {}).get(r, 0.0)) for r in self.agents}
        done = self.spec.terminal(self._vault.state) or (self.phase.max_rounds is not None and self._round >= self.phase.max_rounds)
        terms = {r: done for r in self.agents}
        truncs = {r: False for r in self.agents}
        obs = self._observations()
        if done:
            payoffs = self.spec.payoff(self._traj()) if self.spec.payoff else {}
            eng.writer.event("episode_end", step=eng.step, payoffs=payoffs, rules=eng.rules.snapshot())
            eng.writer.close()
            self.agents = []
        return obs, rewards, terms, truncs, {r: {} for r in self.possible_agents}

    def _traj(self):
        from ..log import Trajectory
        return Trajectory(self._engine.writer.events, self._engine.writer.calls, self._engine.build_card().to_json())

    def close(self):
        if self._engine is not None:
            self._engine.writer.close()
