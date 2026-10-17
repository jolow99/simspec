"""The one agent interface: Observation (+ ActionSpec) -> ActionDistribution.

Agents never see State. The runtime samples from the distribution with the episode RNG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ..spec import Observation, ActionSpec
from ..card import ScaffoldSection, DistributionSource


class ActionDistribution(BaseModel):
    """Probabilities over action keys (sum to 1). For text actions, ``text`` holds the sampled text and
    ``probs`` is ``{text: 1.0}``. ``source`` records how the distribution was obtained."""
    model_config = ConfigDict(frozen=True)
    probs: dict[str, float]
    source: DistributionSource
    text: str | None = None
    meta: dict[str, Any] = {}

    @classmethod
    def point(cls, key: str, source: DistributionSource = "scripted", **meta: Any) -> "ActionDistribution":
        return cls(probs={key: 1.0}, source=source, meta=meta)

    @classmethod
    def of_text(cls, text: str, source: DistributionSource = "degenerate", **meta: Any) -> "ActionDistribution":
        return cls(probs={text: 1.0}, source=source, text=text, meta=meta)

    def normalized(self) -> "ActionDistribution":
        s = sum(self.probs.values())
        if s <= 0:
            raise ValueError("distribution has no mass")
        return self.model_copy(update={"probs": {k: v / s for k, v in self.probs.items()}})

    def restricted(self, legal: list[str] | None) -> "ActionDistribution":
        if legal is None:
            return self
        keep = {k: v for k, v in self.probs.items() if k in legal}
        if not keep or sum(keep.values()) <= 0:
            raise ValueError("no mass on legal actions")
        return self.model_copy(update={"probs": keep}).normalized()

    def argmax(self) -> str:
        return max(self.probs, key=self.probs.get)


@dataclass
class ActContext:
    """Runtime context handed to agents: step, phase, round, episode seed, logging sink, dry-run flag."""
    step: int
    phase: str
    round: int
    seed: int
    dry_run: bool = False
    log_call: Callable[[Any, Any, dict | None], str] | None = None
    log_info: Callable[[str, dict], None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    role: str

    def act(self, obs: Observation, spec: ActionSpec, ctx: ActContext) -> ActionDistribution: ...

    def describe(self, seat: str) -> ScaffoldSection: ...


class BaseAgent:
    """Optional conveniences for agents: no-op observe/get_state/set_state."""
    role: str = ""

    def observe(self, obs: Observation, ctx: ActContext) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, s: dict[str, Any]) -> None:
        pass

    def end_episode(self, payoffs: dict[str, float]) -> None:
        pass


class ScriptedAgent(BaseAgent):
    """A policy given as a callable obs -> ActionDistribution | key | value, or a fixed key."""

    def __init__(self, role: str, policy: Callable[[Observation, ActionSpec], Any] | str, name: str = "scripted"):
        self.role = role
        self.policy = policy
        self.name = name

    def act(self, obs: Observation, spec: ActionSpec, ctx: ActContext) -> ActionDistribution:
        out = self.policy(obs, spec) if callable(self.policy) else self.policy
        if isinstance(out, ActionDistribution):
            return out
        if isinstance(out, dict) and spec.kind != "record":
            return ActionDistribution(probs=out, source="scripted").normalized()
        if spec.kind == "text":
            return ActionDistribution.of_text(str(out), source="scripted")
        key = out if isinstance(out, str) and spec.kind == "choice" else spec.key_of(out)
        return ActionDistribution.point(key, source="scripted")

    def describe(self, seat: str) -> ScaffoldSection:
        from ..spec import _fn_hash
        return ScaffoldSection(seat=seat, role=self.role, agent_class=f"scripted:{self.name}",
                               distribution_source="scripted", solver={"policy": _fn_hash(self.policy) if callable(self.policy) else str(self.policy)})


class ReplayAgent(BaseAgent):
    """Replays the actions logged for a role, in order. Used by replay-by-readback and the card test."""

    def __init__(self, role: str, actions: list[dict]):
        self.role = role
        self._actions = [a for a in actions if a["role"] == role]
        self._pos = 0

    def act(self, obs: Observation, spec: ActionSpec, ctx: ActContext) -> ActionDistribution:
        if self._pos >= len(self._actions):
            raise RuntimeError(f"replay exhausted for {self.role} at step {ctx.step}")
        rec = self._actions[self._pos]
        self._pos += 1
        if rec["step"] != ctx.step:
            raise RuntimeError(f"replay misaligned for {self.role}: logged step {rec['step']} vs {ctx.step}")
        key = rec["key"]
        if spec.kind == "text":
            return ActionDistribution.of_text(key, source="replay")
        return ActionDistribution.point(key, source="replay")

    def describe(self, seat: str) -> ScaffoldSection:
        return ScaffoldSection(seat=seat, role=self.role, agent_class="replay", distribution_source="replay")


class DryRunAttempted(Exception):
    """A model transport was invoked during a dry run."""


class DryRunTransport:
    """Transport stub used by spec.check(): raises if any model call is attempted."""

    def __getattr__(self, name):
        def _fail(*a, **k):
            raise DryRunAttempted(f"model call {name!r} attempted during dry run")
        return _fail
