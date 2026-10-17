"""Rules: prohibitions, obligations and constraints as first-class, serialisable objects, with a
compliance engine and a hypothesis-space enumerator (for norm induction).

Semantics (generalising Oldenburg & Zhi-Xuan 2024 and Sajja et al. 2026):
- prohibition  <action_class, post>: violated at step t by role r if r's action is in the class and
  post holds on r's observation of the next state.
- obligation   <pre, post, deadline>: triggered when pre holds on r's observation; discharged if post
  holds within `deadline` steps (deadline=0 means the same step); violated otherwise.
- constraint   <classifier over the role's history>: violated when the classifier returns True.
Predicates are evaluated on observation feature dicts, never on raw state, so rules respect the
declared observation structure.
"""
from __future__ import annotations

import hashlib
import inspect
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field as PField


# ----------------------------------------------------------------------------- predicates

class Cmp(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["cmp"] = "cmp"
    feature: str
    cmp: Literal["==", "!=", "<", "<=", ">", ">="]
    value: Any

    def eval(self, feats: dict[str, Any], action: str | None = None) -> bool:
        x = feats[self.feature]
        v = self.value
        return {"==": x == v, "!=": x != v, "<": x < v, "<=": x <= v, ">": x > v, ">=": x >= v}[self.cmp]

    def text(self) -> str:
        return f"{self.feature} {self.cmp} {self.value!r}"


class IsTrue(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["true"] = "true"
    feature: str

    def eval(self, feats, action=None) -> bool:
        return bool(feats[self.feature])

    def text(self) -> str:
        return self.feature


class ActionIs(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["action"] = "action"
    action_class: str

    def eval(self, feats, action=None) -> bool:
        return action is not None and action == self.action_class

    def text(self) -> str:
        return f"action is {self.action_class}"


class And(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["and"] = "and"
    args: list["Predicate"]

    def eval(self, feats, action=None) -> bool:
        return all(a.eval(feats, action) for a in self.args)

    def text(self) -> str:
        return "(" + " and ".join(a.text() for a in self.args) + ")"


class Or(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["or"] = "or"
    args: list["Predicate"]

    def eval(self, feats, action=None) -> bool:
        return any(a.eval(feats, action) for a in self.args)

    def text(self) -> str:
        return "(" + " or ".join(a.text() for a in self.args) + ")"


class Not(BaseModel):
    model_config = ConfigDict(frozen=True)
    op: Literal["not"] = "not"
    arg: "Predicate"

    def eval(self, feats, action=None) -> bool:
        return not self.arg.eval(feats, action)

    def text(self) -> str:
        return f"not {self.arg.text()}"


class PyPred(BaseModel):
    """Escape hatch: an arbitrary Python predicate over (features, action). Serialises as its source hash."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    op: Literal["py"] = "py"
    name: str
    fn: Callable[..., bool] = PField(exclude=True)
    source_hash: str = ""

    def model_post_init(self, __ctx):
        if not self.source_hash:
            object.__setattr__(self, "source_hash", _fn_hash(self.fn))

    def eval(self, feats, action=None) -> bool:
        try:
            return bool(self.fn(feats, action))
        except TypeError:
            return bool(self.fn(feats))

    def text(self) -> str:
        return f"py:{self.name}"


Predicate = Union[Cmp, IsTrue, ActionIs, And, Or, Not, PyPred]
And.model_rebuild()
Or.model_rebuild()
Not.model_rebuild()


# ----------------------------------------------------------------------------- rules

class Sanction(BaseModel):
    """What happens on violation. ``transfer``: move ``amount`` from violator to ``to`` (or burn);
    ``mask``: forbid ``action_class`` for the violator for ``steps``; ``release``: release ``to``
    from all obligations toward the violator; ``custom``: handled by the game's transition via events."""
    model_config = ConfigDict(frozen=True)
    kind: Literal["transfer", "mask", "release", "custom"]
    amount: float | None = None
    to: str | None = None
    action_class: str | None = None
    steps: int | None = None
    detail: dict[str, Any] = {}


class Rule(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    kind: Literal["prohibition", "obligation", "constraint"]
    scope: list[str]                       # roles bound by the rule
    pre: Predicate | None = None           # obligation trigger
    post: Predicate | None = None          # prohibition effect / obligation discharge
    action_class: str | None = None        # prohibition
    deadline: int | None = None            # obligation
    classifier: PyPred | None = None       # constraint: (history_features, None) -> violated?
    sanction: Sanction | None = None
    tolerance: float = 0.0
    text: str | None = None                # the natural-language source, if any (declared text slot)
    source: str = "spec"                   # spec | contract | induced

    def signature(self) -> str:
        """Semantic description without the id (for matching induced rules to true rules)."""
        return self.describe().split("] ", 1)[1]

    def describe(self) -> str:
        if self.kind == "prohibition":
            return f"[{self.id}] {','.join(self.scope)}: if {self.post.text()} after {self.action_class} -> forbidden"
        if self.kind == "obligation":
            return f"[{self.id}] {','.join(self.scope)}: when {self.pre.text()} -> achieve {self.post.text()} within {self.deadline}"
        return f"[{self.id}] {','.join(self.scope)}: constraint {self.classifier.text() if self.classifier else '?'}"


@dataclass
class RuleEvent:
    step: int
    rule_id: str
    role: str
    status: Literal["triggered", "discharged", "violated", "sanctioned", "released"]
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "rule_id": self.rule_id, "role": self.role, "status": self.status, "detail": self.detail}


@dataclass
class _Active:
    rule_id: str
    role: str
    since: int


class RuleEngine:
    """Tracks obligation status, checks prohibitions and constraints, records violations and sanctions."""

    def __init__(self, rules: list[Rule]):
        self.rules: dict[str, Rule] = {}
        self.active: list[_Active] = []
        self.violations: list[RuleEvent] = []
        self.events: list[RuleEvent] = []
        self.masks: dict[str, dict[str, int]] = {}      # role -> {action_class: until_step}
        self.transfers: dict[str, float] = {}           # role -> net transfer applied to payoff
        self.released: set[tuple[str, str]] = set()     # (role, rule_id) released obligations
        self.breached: set[str] = set()                 # roles with any violation (absorbing)
        self._pending: list[RuleEvent] = []             # sanction/release events produced by _violate
        for r in rules:
            self.add(r)

    def add(self, rule: Rule) -> None:
        self.rules[rule.id] = rule

    def remove(self, rule_id: str) -> None:
        self.rules.pop(rule_id, None)
        self.active = [a for a in self.active if a.rule_id != rule_id]

    def rules_for(self, role: str) -> list[Rule]:
        return [r for r in self.rules.values() if role in r.scope and (role, r.id) not in self.released]

    # -- checks -------------------------------------------------------------------------------
    def check_prohibitions(self, step: int, role: str, action_class: str | None, next_feats: dict[str, Any]) -> list[RuleEvent]:
        out = []
        for r in self.rules_for(role):
            if r.kind != "prohibition" or action_class is None:
                continue
            if r.action_class != action_class:
                continue
            if r.post is not None and _safe_eval(r.post, next_feats, action_class):
                ev = RuleEvent(step, r.id, role, "violated", {"action_class": action_class})
                self._violate(ev, r)
                out.append(ev)
        out += self._drain()
        self.events.extend(out)
        return out

    def forbidden_action_classes(self, role: str, feats_by_action: dict[str, dict[str, Any]], class_of: Callable[[str], str]) -> set[str]:
        """Which candidate actions (keyed) would violate a prohibition given their predicted next features."""
        bad = set()
        for key, nf in feats_by_action.items():
            ac = class_of(key)
            for r in self.rules_for(role):
                if r.kind == "prohibition" and r.action_class == ac and r.post is not None and _safe_eval(r.post, nf, ac):
                    bad.add(key)
                    break
        return bad

    def step_obligations(self, step: int, role: str, feats: dict[str, Any]) -> list[RuleEvent]:
        out = []
        for r in self.rules_for(role):
            if r.kind != "obligation":
                continue
            act = next((a for a in self.active if a.rule_id == r.id and a.role == role), None)
            if act is None:
                if r.pre is not None and _safe_eval(r.pre, feats):
                    act = _Active(r.id, role, step)
                    self.active.append(act)
                    out.append(RuleEvent(step, r.id, role, "triggered"))
                else:
                    continue
            # active: discharged?
            if r.post is not None and _safe_eval(r.post, feats):
                self.active.remove(act)
                out.append(RuleEvent(step, r.id, role, "discharged", {"since": act.since}))
            elif r.deadline is not None and step - act.since > r.deadline:
                self.active.remove(act)
                ev = RuleEvent(step, r.id, role, "violated", {"since": act.since, "deadline": r.deadline})
                self._violate(ev, r)
                out.append(ev)
        out += self._drain()
        self.events.extend(out)
        return out

    def check_constraints(self, step: int, role: str, history_feats: dict[str, Any]) -> list[RuleEvent]:
        out = []
        for r in self.rules_for(role):
            if r.kind != "constraint" or r.classifier is None:
                continue
            if r.classifier.eval(history_feats, None):
                if any(v.rule_id == r.id and v.role == role for v in self.violations):
                    continue  # absorbing: report once
                ev = RuleEvent(step, r.id, role, "violated")
                self._violate(ev, r)
                out.append(ev)
        out += self._drain()
        self.events.extend(out)
        return out

    def _drain(self) -> list[RuleEvent]:
        out, self._pending = self._pending, []
        return out

    def _violate(self, ev: RuleEvent, rule: Rule) -> None:
        self.violations.append(ev)
        self.breached.add(ev.role)
        s = rule.sanction
        if s is None:
            return
        if s.kind == "transfer" and s.amount:
            self.transfers[ev.role] = self.transfers.get(ev.role, 0.0) - s.amount
            if s.to:
                self.transfers[s.to] = self.transfers.get(s.to, 0.0) + s.amount
            self._pending.append(RuleEvent(ev.step, rule.id, ev.role, "sanctioned", {"transfer": s.amount, "to": s.to}))
        elif s.kind == "mask" and s.action_class:
            self.masks.setdefault(ev.role, {})[s.action_class] = ev.step + (s.steps or 1)
            self._pending.append(RuleEvent(ev.step, rule.id, ev.role, "sanctioned", {"mask": s.action_class, "steps": s.steps}))
        elif s.kind == "release" and s.to:
            victim = s.to
            for r in self.rules.values():
                if victim in r.scope and r.kind in ("obligation", "constraint"):
                    self.released.add((victim, r.id))
            self.active = [a for a in self.active if a.role != victim]
            self._pending.append(RuleEvent(ev.step, rule.id, victim, "released", {"because": ev.role}))
        else:
            self._pending.append(RuleEvent(ev.step, rule.id, ev.role, "sanctioned", {"custom": s.detail}))

    def masked_classes(self, role: str, step: int) -> set[str]:
        return {ac for ac, until in self.masks.get(role, {}).items() if step < until}

    def is_compliant(self, role: str) -> bool:
        return role not in self.breached

    def snapshot(self) -> dict[str, Any]:
        return {"active": [(a.rule_id, a.role, a.since) for a in self.active],
                "breached": sorted(self.breached), "transfers": dict(self.transfers),
                "released": sorted(self.released), "rules": sorted(self.rules)}


def _safe_eval(p: Any, feats: dict[str, Any], action: str | None = None) -> bool:
    try:
        return bool(p.eval(feats, action))
    except KeyError:
        return False


# ----------------------------------------------------------------------------- rule spaces

class FeatureSpec(BaseModel):
    """One observation feature usable by generated rules."""
    model_config = ConfigDict(frozen=True)
    name: str
    kind: Literal["bool", "threshold", "categorical"]
    cmp: Literal["<", ">", "==", "!="] | None = None
    thresholds: list[Any] = []
    goal: Cmp | IsTrue | None = None        # discharge predicate induced by this feature (obligations)
    group: str = ""                         # feasibility grouping (cell / env / agent)


class RuleSpace(BaseModel):
    """Enumerates a finite hypothesis space of rules from feature grids, as in the norms paper."""
    model_config = ConfigDict(frozen=True)
    features: list[FeatureSpec]
    action_classes: list[str]
    roles: list[str] = []                   # obligations are role-conditioned on these (via feature `role`)
    role_feature: str | None = None
    max_len: int = 3
    prohibition_filter: Callable[[list[FeatureSpec]], bool] | None = PField(default=None, exclude=True)
    obligation_filter: Callable[[list[FeatureSpec]], bool] | None = PField(default=None, exclude=True)
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    def _atoms(self, f: FeatureSpec) -> list[Predicate]:
        if f.kind == "bool":
            return [IsTrue(feature=f.name)]
        if f.kind == "threshold":
            return [Cmp(feature=f.name, cmp=f.cmp or ">", value=t) for t in f.thresholds]
        return [Cmp(feature=f.name, cmp="==", value=t) for t in f.thresholds]

    def enumerate(self) -> list[Rule]:
        rules: list[Rule] = []
        seen: set[str] = set()
        feats = list(self.features)
        for n in range(1, self.max_len + 1):
            for combo in itertools.combinations(feats, n):
                combo = list(combo)
                # prohibitions
                if self.prohibition_filter is None or self.prohibition_filter(combo):
                    for atoms in itertools.product(*[self._atoms(f) for f in combo]):
                        post = atoms[0] if len(atoms) == 1 else And(args=list(atoms))
                        for ac in self.action_classes:
                            r = Rule(id="", kind="prohibition", scope=list(self.roles) or ["*"], post=post, action_class=ac, source="induced")
                            key = r.describe()
                            if key not in seen:
                                seen.add(key)
                                rules.append(r.model_copy(update={"id": f"P{len(rules)}"}))
                # obligations: one feature with a goal, optionally role-conditioned
                if self.obligation_filter is None or self.obligation_filter(combo):
                    goal_feats = [f for f in combo if f.goal is not None]
                    if len(goal_feats) == 1 and len(combo) <= 2:
                        gf = goal_feats[0]
                        others = [f for f in combo if f is not gf]
                        for atoms in itertools.product(*[self._atoms(f) for f in combo]):
                            pre = atoms[0] if len(atoms) == 1 else And(args=list(atoms))
                            r = Rule(id="", kind="obligation", scope=list(self.roles) or ["*"], pre=pre, post=gf.goal, deadline=None, source="induced")
                            key = r.describe()
                            if key not in seen:
                                seen.add(key)
                                rules.append(r.model_copy(update={"id": f"O{len(rules)}"}))
        return rules


def _fn_hash(fn: Callable | None) -> str:
    if fn is None:
        return ""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    return hashlib.sha256(src.encode()).hexdigest()[:16]
