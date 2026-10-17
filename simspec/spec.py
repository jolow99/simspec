"""The game spec: state schema, actions, observation structure (views), phases, GameSpec.

Observation structure is the enforced part. See ``View`` and ``ObservationLeak``.
"""
from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal

from pydantic import BaseModel, ConfigDict

from .domains import AnyType, Domain, Text, Table, Product, Enum, IntRange, Grid, Bool


class ObservationLeak(Exception):
    """Raised whenever something reads state it is not declared to observe."""


# ----------------------------------------------------------------------------- state

Visibility = Literal["public", "private", "hidden"]


class Field(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    type: AnyType
    visibility: Visibility = "hidden"
    owner: str | None = None          # role, for private fields
    doc: str = ""

    def visible_to(self, role: str) -> bool:
        if self.visibility == "public":
            return True
        if self.visibility == "private":
            return self.owner == role
        return False


class StateSchema(BaseModel):
    model_config = ConfigDict(frozen=True)
    fields: list[Field]

    def __getitem__(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def names(self) -> list[str]:
        return [f.name for f in self.fields]

    def validate_state(self, state: "State") -> None:
        for f in self.fields:
            if f.name not in state._values:
                raise ValueError(f"state missing field {f.name}")
            v = state._values[f.name]
            if not f.type.contains(v):
                raise ValueError(f"state field {f.name}={v!r} not in {f.type.name}")
        extra = set(state._values) - set(self.names())
        if extra:
            raise ValueError(f"state has undeclared fields {sorted(extra)}")


class State(Mapping):
    """Immutable record of field values. Values are JSON-native (int/float/str/bool/list/dict)."""
    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]):
        object.__setattr__(self, "_values", dict(values))

    def __getitem__(self, k: str) -> Any:
        return self._values[k]

    def __getattr__(self, k: str) -> Any:
        try:
            return self._values[k]
        except KeyError:
            raise AttributeError(k) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __setattr__(self, k, v):
        raise AttributeError("State is immutable; use .update(**kw)")

    def update(self, **kw: Any) -> "State":
        new = dict(self._values)
        new.update(kw)
        return State(new)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def __repr__(self) -> str:
        return f"State({self._values!r})"


class TracingState(Mapping):
    """A State proxy that records every field read and raises on undeclared reads."""

    def __init__(self, state: State, allowed: set[str] | None, who: str):
        self._state = state
        self._allowed = allowed
        self._who = who
        self.reads: set[str] = set()

    def _check(self, k: str) -> None:
        self.reads.add(k)
        if self._allowed is not None and k not in self._allowed:
            raise ObservationLeak(
                f"{self._who} read state field {k!r}, which it is not declared to observe "
                f"(declared: {sorted(self._allowed)})")

    def __getitem__(self, k: str) -> Any:
        self._check(k)
        return self._state[k]

    def __getattr__(self, k: str) -> Any:
        if k.startswith("_") or k == "reads":
            raise AttributeError(k)
        self._check(k)
        return self._state[k]

    def __iter__(self):
        raise ObservationLeak(f"{self._who} tried to enumerate the whole state")

    def __len__(self) -> int:
        return len(self._state)

    def to_dict(self):
        raise ObservationLeak(f"{self._who} tried to copy the whole state")


# ----------------------------------------------------------------------------- actions

ActionKind = Literal["choice", "record", "text"]


class ActionSpec(BaseModel):
    """What an actor may do in a phase. ``choice`` over a finite Domain, a ``record`` of typed
    sub-fields (a Product, possibly with Text fields handled separately), or a pure ``text`` slot."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    kind: ActionKind
    domain: Domain | None = None
    record: dict[str, AnyType] | None = None     # for kind=record
    text: Text | None = None                       # for kind=text
    call_to_action: str = ""
    on_illegal: Literal["resample", "default", "forfeit", "error"] = "default"
    default: Any = None
    legal: Callable[[State, str], Any] | None = None   # (state, role) -> iterable of legal keys, or None
    valid: Callable[[State, str, Any], bool] | None = None   # (state, role, value) -> ok, for non-enumerable actions
    classes: dict[str, str] | None = None                      # action key -> action class (for prohibitions)

    def keys(self) -> list[str] | None:
        """All action keys for finite choice actions; None for record/text."""
        if self.kind == "choice" and self.domain is not None:
            return [self.key_of(v) for v in self.domain.values()]
        return None

    @staticmethod
    def key_of(value: Any) -> str:
        import json
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def value_of(self, key: str) -> Any:
        import json
        if self.kind == "choice" and self.domain is not None:
            if isinstance(self.domain, Enum):
                return key
            return json.loads(key)
        if self.kind == "text":
            return key
        return json.loads(key)

    def validate(self, value: Any) -> bool:
        if self.kind == "choice":
            return self.domain is not None and self.domain.contains(value)
        if self.kind == "text":
            return self.text is not None and self.text.contains(value)
        if self.kind == "record":
            if not isinstance(value, dict) or self.record is None:
                return False
            if set(value) != set(self.record):
                return False
            return all(t.contains(value[k]) for k, t in self.record.items())
        return False

    def legal_keys(self, state: State, role: str) -> list[str] | None:
        if self.legal is None:
            return self.keys()
        out = self.legal(state, role)
        if out is None:
            return self.keys()
        return [self.key_of(v) if not isinstance(v, str) else v for v in out]

    def describe(self) -> dict[str, Any]:
        d = {"name": self.name, "kind": self.kind, "on_illegal": self.on_illegal, "call_to_action": self.call_to_action}
        if self.domain is not None:
            d["domain"] = self.domain.model_dump()
        if self.record is not None:
            d["record"] = {k: t.model_dump() for k, t in self.record.items()}
        if self.text is not None:
            d["text"] = self.text.model_dump()
        d["legal_fn"] = _fn_hash(self.legal)
        d["valid_fn"] = _fn_hash(self.valid)
        d["classes"] = self.classes
        return d


# ----------------------------------------------------------------------------- views

class Derived(BaseModel):
    """A named feature computed from the state, with the fields it is allowed to read declared."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    reads: list[str]
    fn: Callable[[Any], Any]
    type: AnyType | None = None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "reads": list(self.reads), "fn": _fn_hash(self.fn),
                "type": self.type.model_dump() if self.type is not None else None}


class View(BaseModel):
    """What one role observes: declared state fields, derived features, and message channels."""
    model_config = ConfigDict(frozen=True)
    role: str
    fields: list[str] = []
    derived: list[Derived] = []
    channels: list[str] = []
    history: Literal["none", "own_actions", "visible_actions"] = "none"

    def allowed_fields(self) -> set[str]:
        return set(self.fields) | {f for d in self.derived for f in d.reads}

    def describe(self) -> dict[str, Any]:
        return {"role": self.role, "fields": list(self.fields), "derived": [d.describe() for d in self.derived],
                "channels": list(self.channels), "history": self.history}


class Observation(Mapping):
    """Immutable, role-scoped observation. The only thing agents ever receive."""
    __slots__ = ("_d", "role", "step", "phase", "messages", "legal")

    def __init__(self, role: str, step: int, phase: str, values: Mapping[str, Any],
                 messages: list[dict] | None = None, legal: list[str] | None = None):
        object.__setattr__(self, "_d", dict(values))
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "messages", list(messages or []))
        object.__setattr__(self, "legal", list(legal) if legal is not None else None)

    def __getitem__(self, k):
        return self._d[k]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __setattr__(self, k, v):
        raise AttributeError("Observation is immutable")

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "step": self.step, "phase": self.phase, "values": dict(self._d),
                "messages": list(self.messages), "legal": self.legal}

    def __repr__(self):
        return f"Observation({self.role}@{self.step}:{self.phase} {self._d!r})"


# ----------------------------------------------------------------------------- phases

PhaseKind = Literal["simultaneous", "sequential", "protocol"]


class Phase(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    kind: PhaseKind
    actors: list[str] | Callable[[State], list[str]]
    action: ActionSpec | dict[str, ActionSpec]
    max_rounds: int | None = None
    until: Callable[[State], bool] | None = None          # end the phase early when true
    next: str | None | Callable[[State], str | None] = None

    def actors_in(self, state: State) -> list[str]:
        return list(self.actors(state)) if callable(self.actors) else list(self.actors)

    def action_for(self, role: str) -> ActionSpec:
        if isinstance(self.action, dict):
            return self.action[role]
        return self.action

    def next_phase(self, state: State) -> str | None:
        return self.next(state) if callable(self.next) else self.next

    def describe(self) -> dict[str, Any]:
        act = ({r: a.describe() for r, a in self.action.items()} if isinstance(self.action, dict)
               else self.action.describe())
        return {"name": self.name, "kind": self.kind,
                "actors": self.actors if not callable(self.actors) else _fn_hash(self.actors),
                "action": act, "max_rounds": self.max_rounds, "until": _fn_hash(self.until),
                "next": self.next if not callable(self.next) else _fn_hash(self.next)}


# ----------------------------------------------------------------------------- game spec

@dataclass
class TransitionResult:
    state: State
    events: list[dict] = field(default_factory=list)       # free-form game events for the log
    messages: list[dict] = field(default_factory=list)     # {channel, from, to, text}
    rewards: dict[str, float] = field(default_factory=dict)


@dataclass
class GameSpec:
    """A complete game. Pure functions only; the runtime owns randomness, time and logging."""
    id: str
    version: str
    roles: list[str]
    state: StateSchema
    phases: list[Phase]
    views: dict[str, View]
    initial: Callable[[Any, "Chance"], State]                     # (params, chance) -> State
    transition: Callable[[State, dict[str, Any], "Chance", "StepContext"], TransitionResult]
    terminal: Callable[[State], bool]
    payoff: Callable[["Trajectory"], dict[str, float]] | None = None   # trajectory-level
    rules: list["Rule"] = field(default_factory=list)                   # static rules
    dynamic_rules: Callable[[State], list["Rule"]] | None = None         # rules created at run time (contracts)
    params: Any = None
    channels: list[str] = field(default_factory=list)
    doc: str = ""
    forward: Callable[[State, str, Any], State] | None = None            # single-agent forward model for planners
    renderers: dict[str, Callable[[Observation], str]] = field(default_factory=dict)

    def __post_init__(self):
        names = set(self.state.names())
        for v in self.views.values():
            for f in v.fields:
                if f not in names:
                    raise ValueError(f"view for {v.role} lists unknown field {f}")
        for r in self.roles:
            if r not in self.views:
                raise ValueError(f"no view declared for role {r}")

    def phase(self, name: str) -> Phase:
        for p in self.phases:
            if p.name == name:
                return p
        raise KeyError(name)

    # -- observation construction (the enforced boundary) ------------------------------------
    def observe(self, state: State, role: str, step: int, phase: str, messages: list[dict],
                legal: list[str] | None = None, strict: bool = True) -> Observation:
        view = self.views[role]
        values: dict[str, Any] = {}
        for f in view.fields:
            fld = self.state[f]
            if strict and not fld.visible_to(role):
                raise ObservationLeak(f"view for {role} includes field {f!r} which is {fld.visibility}"
                                      + (f" to {fld.owner}" if fld.owner else ""))
            values[f] = state[f]
        for d in view.derived:
            proxy = TracingState(state, set(d.reads), f"derived feature {d.name!r} of {role}")
            values[d.name] = d.fn(proxy)
        msgs = [m for m in messages if m.get("channel") in view.channels
                and (m.get("to") in (None, "all", role) or m.get("from") == role)]
        return Observation(role, step, phase, values, msgs, legal)

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id, "version": self.version, "roles": list(self.roles),
            "state": [f.model_dump() for f in self.state.fields],
            "phases": [p.describe() for p in self.phases],
            "views": {r: v.describe() for r, v in self.views.items()},
            "rules": [r.model_dump() for r in self.rules],
            "channels": list(self.channels),
            "fns": {"initial": _fn_hash(self.initial), "transition": _fn_hash(self.transition),
                    "terminal": _fn_hash(self.terminal), "payoff": _fn_hash(self.payoff),
                    "dynamic_rules": _fn_hash(self.dynamic_rules)},
            "params": _jsonable(self.params),
        }

    def spec_hash(self) -> str:
        import json
        blob = json.dumps(self.describe(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class StepContext:
    step: int
    phase: str
    round: int
    actor: str | None          # the acting role in sequential phases, None for simultaneous
    rules: "RuleEngine | None" = None


def _fn_hash(fn: Callable | None) -> str | None:
    if fn is None:
        return None
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, BaseModel):
        return x.model_dump()
    if hasattr(x, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return repr(x)
