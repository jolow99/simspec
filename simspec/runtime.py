"""The thin engine: phases, vault-locked state, chance log, rule engine, JSONL events, replay, checker.

The loop is deliberately small. What matters is at the boundaries: agents only ever receive an
Observation built by ``GameSpec.observe``; the Vault is locked while an agent acts; every random
draw is logged; every sampled action is logged with its distribution.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .chance import Chance
from .spec import (GameSpec, State, Observation, ObservationLeak, ActionSpec, Phase, StepContext,
                   TransitionResult)
from .rules import RuleEngine, Rule, RuleEvent
from .log import RunWriter, Trajectory
from .card import HarnessCard, GameSection, ScaffoldSection, runtime_section
from .agents.base import Agent, ActionDistribution, ActContext, ReplayAgent, DryRunAttempted


class Vault:
    """Holds the state. Locked while an agent is acting: any read then is an ObservationLeak."""

    def __init__(self, state: State):
        self._state = state
        self._locked = False
        self._holder: str | None = None

    @property
    def state(self) -> State:
        if self._locked:
            raise ObservationLeak(f"state read while locked for agent {self._holder!r}: agents may only use their Observation")
        return self._state

    def set(self, state: State) -> None:
        if self._locked:
            raise ObservationLeak(f"state write while locked for agent {self._holder!r}")
        self._state = state

    def lock(self, holder: str) -> None:
        self._locked, self._holder = True, holder

    def unlock(self) -> None:
        self._locked, self._holder = False, None


@dataclass
class EpisodeResult:
    trajectory: Trajectory
    payoffs: dict[str, float]
    final_state: State
    card: HarnessCard
    run_dir: str | None


class Engine:
    def __init__(self, spec: GameSpec, agents: dict[str, Agent], seed: int = 0, run_dir: str | None = None,
                 seats: dict[str, str] | None = None, replay_chance: list[dict] | None = None,
                 dry_run: bool = False, max_steps: int = 10_000, run_id: str | None = None,
                 replay_mode: str = "live", action_hooks: list[Callable] | None = None):
        self.spec = spec
        self.agents = agents                       # seat -> agent  (seat == role unless seats given)
        self.seats = seats or {s: getattr(a, "role", s) for s, a in agents.items()}
        self.agent_of_role = {self.seats[s]: a for s, a in agents.items()}
        for r in spec.roles:
            if r not in self.agent_of_role:
                raise ValueError(f"no agent for role {r}")
        self.seed = seed
        self.dry_run = dry_run
        self.max_steps = max_steps
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.run_dir = run_dir
        self.writer = RunWriter(run_dir)
        self.chance = Chance(seed, replay=replay_chance)
        self.rules = RuleEngine(list(spec.rules))
        self.messages: list[dict] = []
        self.step = 0
        self.replay_mode = replay_mode
        self.action_hooks = action_hooks or []

    # -- card -----------------------------------------------------------------------------------
    def build_card(self) -> HarnessCard:
        scaffolds = {seat: ag.describe(seat) for seat, ag in self.agents.items()}
        game = GameSection(game_id=self.spec.id, game_version=self.spec.version, spec_hash=self.spec.spec_hash(),
                           spec=self.spec.describe(), seats=dict(self.seats),
                           instance={"params": _jsonable(self.spec.params)},
                           illegal_action_policy={p.name: (p.action.on_illegal if isinstance(p.action, ActionSpec)
                                                            else {r: a.on_illegal for r, a in p.action.items()}) for p in self.spec.phases})
        card = HarnessCard(run_id=self.run_id, game=game, scaffolds=scaffolds,
                           runtime=runtime_section(self.seed, self.replay_mode))
        return card.sealed()

    # -- main loop ------------------------------------------------------------------------------
    def run(self) -> EpisodeResult:
        spec = self.spec
        card = self.build_card()
        self.writer.write_card(card.to_json())
        state = spec.initial(spec.params, self.chance)
        spec.state.validate_state(state)
        vault = Vault(state)
        self._flush_chance()
        self.writer.event("episode_start", seed=self.seed, card_hash=card.card_hash, run_id=self.run_id,
                          state=state.to_dict(), rules=[r.id for r in spec.rules])
        if spec.dynamic_rules is not None:
            for r in spec.dynamic_rules(state):
                self.rules.add(r)
        phase: Phase | None = spec.phases[0]
        round_no = 0
        traj_feats: dict[str, dict[str, list]] = {r: {} for r in spec.roles}
        while phase is not None and self.step < self.max_steps:
            self.writer.event("phase_start", step=self.step, phase=phase.name, round=round_no)
            actors = phase.actors_in(vault.state)
            if phase.kind == "simultaneous":
                joint: dict[str, Any] = {}
                for role in actors:
                    key, action = self._elicit(vault, phase, role, round_no)
                    joint[role] = action
                self._apply(vault, phase, joint, round_no, actor=None)
            else:  # sequential / protocol: apply per actor
                for role in actors:
                    key, action = self._elicit(vault, phase, role, round_no)
                    self._apply(vault, phase, {role: action}, round_no, actor=role)
                    if spec.terminal(vault.state) or (phase.until is not None and phase.until(vault.state)):
                        break
            round_no += 1
            if spec.terminal(vault.state):
                break
            if (phase.until is not None and phase.until(vault.state)) or (phase.max_rounds is not None and round_no >= phase.max_rounds):
                nxt = phase.next_phase(vault.state)
                phase = spec.phase(nxt) if nxt else None
                round_no = 0
                if phase is not None and spec.dynamic_rules is not None:
                    for r in spec.dynamic_rules(vault.state):
                        if r.id not in self.rules.rules:
                            self.rules.add(r)
                            self.writer.event("rule", step=self.step, rule_id=r.id, role=",".join(r.scope), status="added", detail={"source": r.source, "text": r.text})
        traj = Trajectory(self.writer.events, self.writer.calls, card.to_json())
        payoffs = spec.payoff(traj) if spec.payoff is not None else {}
        for role, t in self.rules.transfers.items():
            payoffs[role] = payoffs.get(role, 0.0) + t
        self.writer.event("episode_end", step=self.step, payoffs=payoffs, rules=self.rules.snapshot())
        for ag in self.agents.values():
            if hasattr(ag, "end_episode"):
                ag.end_episode(payoffs)
        self.writer.close()
        traj = Trajectory(self.writer.events, self.writer.calls, card.to_json())
        if self.run_dir is not None:
            with open(os.path.join(self.run_dir, "payoffs.json"), "w") as fh:
                json.dump(payoffs, fh, indent=2)
        return EpisodeResult(traj, payoffs, vault.state, card, self.run_dir)

    # -- one decision ---------------------------------------------------------------------------
    def _elicit(self, vault: Vault, phase: Phase, role: str, round_no: int) -> tuple[str, Any]:
        spec = self.spec
        aspec = phase.action_for(role)
        state = vault.state
        legal = aspec.legal_keys(state, role)
        masked = self.rules.masked_classes(role, self.step)
        if legal is not None and masked:
            legal = [k for k in legal if _action_class(aspec, k) not in masked]
        obs = spec.observe(state, role, self.step, phase.name, self.messages, legal)
        self.writer.event("observation", step=self.step, role=role, phase=phase.name, obs=obs.to_dict())
        agent = self.agent_of_role[role]
        ctx = ActContext(step=self.step, phase=phase.name, round=round_no, seed=self.seed, dry_run=self.dry_run,
                         log_call=self.writer.call, log_info=self._log_info)
        vault.lock(role)
        t0 = time.perf_counter()
        try:
            dist = agent.act(obs, aspec, ctx)
        finally:
            vault.unlock()
        latency = (time.perf_counter() - t0) * 1000
        dist = dist.normalized()
        self.writer.event("agent_call", step=self.step, role=role, phase=phase.name, source=dist.source,
                          probs=dist.probs if len(dist.probs) <= 512 else {"_n": len(dist.probs)},
                          text=dist.text, meta=dist.meta, latency_ms=round(latency, 3))
        # sample
        if aspec.kind == "text":
            key = dist.text if dist.text is not None else self.chance.categorical(f"sample:{role}", dist.probs)
            value = key
        else:
            key = self.chance.categorical(f"sample:{role}", dist.probs)
            value = aspec.value_of(key)
        valid = aspec.validate(value) and (legal is None or key in legal)
        if valid and aspec.valid is not None:
            valid = bool(aspec.valid(state, role, value))
        retried = False
        if not valid:
            policy = aspec.on_illegal
            if policy == "error":
                raise ValueError(f"illegal action {key!r} by {role} at step {self.step}")
            if policy == "resample" and legal:
                sub = {k: v for k, v in dist.probs.items() if k in legal}
                if sub:
                    key = self.chance.categorical(f"resample:{role}", sub)
                else:
                    key = self.chance.choice(f"resample_uniform:{role}", legal)
                value = aspec.value_of(key)
                retried = True
            elif policy == "forfeit":
                key, value = "__forfeit__", None
            else:  # default
                if aspec.default is not None:
                    value = aspec.default
                    key = aspec.key_of(value) if aspec.kind != "text" else value
                elif legal:
                    key = legal[0]
                    value = aspec.value_of(key)
                else:
                    key, value = "__forfeit__", None
        self._flush_chance()
        self.writer.event("action", step=self.step, role=role, phase=phase.name, key=key, value=value,
                          valid=valid, replaced=not valid, retried=retried, action=aspec.name)
        for h in self.action_hooks:
            h(role, key, value, obs)
        return key, value

    # -- transition + rules ---------------------------------------------------------------------
    def _apply(self, vault: Vault, phase: Phase, joint: dict[str, Any], round_no: int, actor: str | None) -> None:
        spec = self.spec
        ctx = StepContext(step=self.step, phase=phase.name, round=round_no, actor=actor, rules=self.rules)
        before = vault.state
        res = spec.transition(before, joint, self.chance, ctx)
        if not isinstance(res, TransitionResult):
            raise TypeError("transition must return TransitionResult")
        self._flush_chance()
        spec.state.validate_state(res.state)
        vault.set(res.state)
        for m in res.messages:
            m = dict(m)
            m["step"] = self.step
            self.messages.append(m)
            self.writer.event("message", **m)
        # rules: prohibitions on the actors' actions, obligations and constraints for every bound role
        rule_events: list[RuleEvent] = []
        for role in spec.roles:
            if not self.rules.rules_for(role):
                continue
            feats = dict(spec.observe(res.state, role, self.step, phase.name, self.messages, strict=False))
            if role in joint:
                aspec = phase.action_for(role)
                key = aspec.key_of(joint[role]) if aspec.kind != "text" else str(joint[role])
                rule_events += self.rules.check_prohibitions(self.step, role, _action_class(aspec, key), feats)
            rule_events += self.rules.step_obligations(self.step, role, feats)
            hist = {"role": role, "step": self.step, "features": feats, "state": res.state.to_dict(),
                    "joint": {r: _jsonable(a) for r, a in joint.items()}, "events": list(res.events)}
            rule_events += self.rules.check_constraints(self.step, role, hist)
        for ev in rule_events:
            self.writer.event("rule", **ev.to_dict())
        self.writer.event("transition", step=self.step, phase=phase.name, round=round_no, actor=actor,
                          joint={r: _jsonable(a) for r, a in joint.items()}, state=res.state.to_dict(),
                          events=res.events, rewards=res.rewards)
        # non-acting agents may observe (e.g. norm learners watch others)
        for role, agent in self.agent_of_role.items():
            if hasattr(agent, "observe"):
                obs = spec.observe(res.state, role, self.step, phase.name, self.messages)
                vault.lock(role)
                try:
                    agent.observe(obs, ActContext(step=self.step, phase=phase.name, round=round_no, seed=self.seed, dry_run=self.dry_run,
                                                  log_info=self._log_info,
                                                  extra={"joint": {r: _jsonable(a) for r, a in joint.items()}, "events": list(res.events)}))
                finally:
                    vault.unlock()
        self.step += 1

    def _log_info(self, source: str, data: dict) -> None:
        self.writer.event("info", step=self.step, source=source, data=_jsonable(data))

    def _flush_chance(self) -> None:
        for d in self.chance.drain():
            self.writer.event("chance", step=self.step, name=d["name"], value=d["value"])


def _action_class(aspec: ActionSpec, key: str) -> str | None:
    """The 'action class' used by prohibitions. For choice actions it is the key; records use their 'class' field if any."""
    if aspec.classes is not None:
        return aspec.classes.get(key, key)
    if aspec.kind == "choice":
        return key
    try:
        v = json.loads(key)
        if isinstance(v, dict) and "class" in v:
            return v["class"]
    except (ValueError, TypeError):
        pass
    return aspec.name


from .spec import _jsonable  # noqa: E402  (dataclasses, pydantic models, containers)


# ----------------------------------------------------------------------------- entry points

def run_episode(spec: GameSpec, agents: dict[str, Agent], seed: int = 0, run_dir: str | None = None, **kw) -> EpisodeResult:
    return Engine(spec, agents, seed=seed, run_dir=run_dir, **kw).run()


def check_spec(spec: GameSpec, agents: dict[str, Agent] | None = None, seed: int = 0, rollout_steps: int = 50) -> dict[str, Any]:
    """The spec checker. Static: every view only lists fields visible to its role, derived features
    only read declared fields. Dynamic: a dry-run episode with model transports stubbed, the vault
    locked during every agent call. Raises ObservationLeak (or DryRunAttempted) before any model call."""
    report: dict[str, Any] = {"static": [], "dynamic": None}
    # static
    for role, view in spec.views.items():
        for f in view.fields:
            fld = spec.state[f]
            if not fld.visible_to(role):
                raise ObservationLeak(f"view for {role!r} includes field {f!r} which is {fld.visibility}"
                                      + (f" to {fld.owner}" if fld.owner else ""))
        for d in view.derived:
            for f in d.reads:
                fld = spec.state[f]
                if not fld.visible_to(role):
                    raise ObservationLeak(f"derived feature {d.name!r} of {role!r} declares a read of {f!r} which is {fld.visibility}"
                                          + (f" to {fld.owner}" if fld.owner else ""))
        report["static"].append(role)
    # dynamic: evaluate derived features on the initial state (tracing proxy)
    chance = Chance(seed)
    s0 = spec.initial(spec.params, chance)
    spec.state.validate_state(s0)
    for role in spec.roles:
        spec.observe(s0, role, 0, spec.phases[0].name, [], None)
    # dry run with the given agents (or scripted fallbacks)
    if agents is None:
        from .agents.base import ScriptedAgent
        agents = {}
        for role in spec.roles:
            def pol(obs, aspec, _r=role):
                if aspec.kind == "text":
                    return ""
                legal = obs.legal or aspec.keys()
                if legal:
                    return legal[0]
                return aspec.default
            agents[role] = ScriptedAgent(role, pol, name="check-fallback")
    for ag in agents.values():
        if hasattr(ag, "enter_dry_run"):
            ag.enter_dry_run()
    try:
        res = Engine(spec, agents, seed=seed, run_dir=None, dry_run=True, max_steps=rollout_steps, replay_mode="live").run()
    finally:
        for ag in agents.values():
            if hasattr(ag, "exit_dry_run"):
                ag.exit_dry_run()
    report["dynamic"] = {"steps": res.trajectory.steps(), "ok": True}
    return report


def replay_episode(spec_factory: Callable[[dict], GameSpec], run_dir: str, verify: bool = True) -> EpisodeResult:
    """Card test: rebuild the spec from the card, replay logged actions and chance draws, and
    verify every state snapshot matches the log."""
    traj = Trajectory.load(run_dir)
    card = HarnessCard.from_json(traj.card)
    spec = spec_factory(card.game.instance.get("params") or {})
    if spec.spec_hash() != card.game.spec_hash:
        raise ValueError(f"spec hash mismatch: card {card.game.spec_hash} vs rebuilt {spec.spec_hash()}")
    agents = {seat: ReplayAgent(role, traj.actions()) for seat, role in card.game.seats.items()}
    eng = Engine(spec, agents, seed=card.runtime.seed, run_dir=None, seats=dict(card.game.seats),
                 replay_chance=traj.chance_draws(), run_id=card.run_id, replay_mode="readback")
    res = eng.run()
    if verify:
        a, b = traj.states(), res.trajectory.states()
        if len(a) != len(b):
            raise AssertionError(f"replay produced {len(b)} transitions, log has {len(a)}")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                raise AssertionError(f"replay diverged at transition {i}")
        if traj.payoffs() != res.payoffs:
            raise AssertionError(f"replay payoffs {res.payoffs} differ from logged {traj.payoffs()}")
    return res
