"""The norms gridworld as a simspec: one simultaneous phase, public state, per-agent derived
features (the rule vocabulary), default norms as Rules, and the hypothesis space generator."""
from __future__ import annotations

from typing import Any

from ...domains import Enum, IntRange, Bool, Table, Grid
from ...spec import Field, StateSchema, State, ActionSpec, View, Derived, Phase, GameSpec, TransitionResult, StepContext
from ...chance import Chance
from ...rules import Rule, Cmp, IsTrue, And, RuleSpace, FeatureSpec
from .world import (Layout, PAPER_MAP, SMALL_MAP, ACTIONS, ACTION_CLASS, ROLES, LOOK, initial_world, apply_action, env_step,
                    dirt_fraction, apples_around, desiccated_ratio)

Act = Enum(name="act", labels=ACTIONS)
DirtState = Enum(name="dirt_state", labels=["none", "potential", "dirt"])


def schema(layout: Layout, n_agents: int, horizon: int) -> StateSchema:
    Cell = IntRange(name="cell", lo=0, hi=layout.n - 1)
    Ag = IntRange(name="agent", lo=0, hi=n_agents - 1)
    Pair = IntRange(name="pair", lo=0, hi=n_agents * n_agents - 1)
    F = Field
    pub = "public"
    return StateSchema(fields=[
        F(name="t", type=IntRange(name="t", lo=0, hi=horizon + 1), visibility=pub),
        F(name="apple", type=Table(name="apple", index=Cell, cell=Bool(name="b")), visibility=pub),
        F(name="desiccated", type=Table(name="des", index=Cell, cell=Bool(name="b")), visibility=pub),
        F(name="dirt", type=Table(name="dirt", index=Cell, cell=DirtState), visibility=pub),
        F(name="property", type=Table(name="prop", index=Cell, cell=IntRange(name="owner", lo=0, hi=n_agents)), visibility=pub),
        F(name="pos", type=Table(name="pos", index=Ag, cell=Cell), visibility=pub),
        F(name="orient", type=Table(name="orient", index=Ag, cell=IntRange(name="o", lo=0, hi=3)), visibility=pub),
        F(name="inventory", type=Table(name="inv", index=Ag, cell=IntRange(name="n", lo=0, hi=10_000)), visibility=pub),
        F(name="since_cleaned", type=Table(name="sc", index=Ag, cell=IntRange(name="n", lo=-1, hi=100_000)), visibility=pub),
        F(name="since_paid", type=Table(name="sp", index=Ag, cell=IntRange(name="n", lo=-1, hi=100_000)), visibility=pub),
        F(name="frozen", type=Table(name="fr", index=Ag, cell=IntRange(name="n", lo=0, hi=10)), visibility=pub),
        F(name="look", type=Table(name="look", index=Ag, cell=IntRange(name="l", lo=0, hi=2)), visibility=pub),
        F(name="stolen", type=Table(name="stolen", index=Pair, cell=IntRange(name="n", lo=0, hi=10_000)), visibility=pub),
        F(name="harvested", type=Table(name="h", index=Ag, cell=Bool(name="b")), visibility=pub),
        F(name="harvest_apples_around", type=Table(name="ha", index=Ag, cell=IntRange(name="n", lo=0, hi=8)), visibility=pub),
        F(name="harvested_foreign", type=Table(name="hf", index=Ag, cell=Bool(name="b")), visibility=pub),
        F(name="ate", type=Table(name="ate", index=Ag, cell=Bool(name="b")), visibility=pub),
        F(name="zapped", type=Table(name="z", index=Ag, cell=Bool(name="b")), visibility=pub),
        F(name="cleaned", type=Table(name="cl", index=Ag, cell=Bool(name="b")), visibility=pub),
        F(name="paid", type=Table(name="pd", index=Ag, cell=Bool(name="b")), visibility=pub),
    ])


# ----------------------------------------------------------------------------- features (the rule vocabulary)

FEATURE_NAMES = ["DIRT_FRACTION", "NUM_APPLES_AROUND", "CUR_CELL_HAS_APPLE", "CUR_CELL_IS_FOREIGN_PROPERTY",
                 "SINCE_AGENT_LAST_PAID", "SINCE_AGENT_LAST_CLEANED", "AGENT_LOOK", "ORIENTATION", "POSITION",
                 "HARVESTED", "HARVEST_APPLES_AROUND", "HARVESTED_FOREIGN", "INVENTORY", "FROZEN"]


def features_of(layout: Layout, w: dict[str, Any], i: int) -> dict[str, Any]:
    """Rule-relevant features of agent i in world dict w (public information only)."""
    c = w["pos"][i]
    return {
        "DIRT_FRACTION": dirt_fraction(layout, w["dirt"]),
        "NUM_APPLES_AROUND": apples_around(layout, w["apple"], c),
        "CUR_CELL_HAS_APPLE": bool(w["apple"][c]),
        "CUR_CELL_IS_FOREIGN_PROPERTY": w["property"][c] not in (0, i + 1),
        "SINCE_AGENT_LAST_PAID": w["since_paid"][i],
        "SINCE_AGENT_LAST_CLEANED": w["since_cleaned"][i],
        "AGENT_LOOK": w["look"][i],
        "ORIENTATION": w["orient"][i],
        "POSITION": c,
        "HARVESTED": bool(w["harvested"][i]),
        "HARVEST_APPLES_AROUND": w["harvest_apples_around"][i],
        "HARVESTED_FOREIGN": bool(w["harvested_foreign"][i]),
        "INVENTORY": w["inventory"][i],
        "FROZEN": w["frozen"][i],
    }


STATE_FIELDS = ["t", "apple", "desiccated", "dirt", "property", "pos", "orient", "inventory", "since_cleaned", "since_paid",
                "frozen", "look", "stolen", "harvested", "harvest_apples_around", "harvested_foreign", "ate", "zapped", "cleaned", "paid"]


def world_of(obs_or_state) -> dict[str, Any]:
    return {k: obs_or_state[k] for k in STATE_FIELDS}


def make_views(layout: Layout, roles: list[str]) -> dict[str, View]:
    views = {}
    reads = ["dirt", "apple", "pos", "property", "since_paid", "since_cleaned", "look", "orient", "harvested",
             "harvest_apples_around", "harvested_foreign", "inventory", "frozen"]
    for i, name in enumerate(agent_names(roles)):
        def mk(fname, idx=i):
            return Derived(name=fname, reads=reads, fn=lambda s, _f=fname, _i=idx: features_of(layout, {k: s[k] for k in reads}, _i)[_f])
        views[name] = View(role=name, fields=STATE_FIELDS, derived=[mk(f) for f in FEATURE_NAMES], history="none")
    return views


def agent_names(roles: list[str]) -> list[str]:
    return [f"{r}{i}" for i, r in enumerate(roles)]


# ----------------------------------------------------------------------------- rules

def default_rules(roles: list[str]) -> list[Rule]:
    names = agent_names(roles)
    by_look = {look: [n for n, r in zip(names, roles) if r == look] for look in ROLES}
    rules = [
        Rule(id="P1_no_empty_orchard", kind="prohibition", scope=names, action_class="MOVE",
             post=And(args=[IsTrue(feature="HARVESTED"), Cmp(feature="HARVEST_APPLES_AROUND", cmp="<", value=3)]),
             text="Don't pick up an apple if there are fewer than 3 apples around it."),
        Rule(id="P2_no_stealing", kind="prohibition", scope=names, action_class="MOVE", post=IsTrue(feature="HARVESTED_FOREIGN"),
             text="Don't harvest apples on someone else's property."),
    ]
    if by_look["farmer"]:
        rules.append(Rule(id="O1_farmer_pays", kind="obligation", scope=by_look["farmer"], pre=Cmp(feature="SINCE_AGENT_LAST_PAID", cmp=">", value=30),
                          post=Cmp(feature="SINCE_AGENT_LAST_CLEANED", cmp="<", value=-100) if False else Cmp(feature="SINCE_AGENT_LAST_PAID", cmp="==", value=0),
                          deadline=20, text="A farmer who has not paid a cleaner for 30 steps must pay one within 20 steps."))
    if by_look["cleaner"]:
        rules.append(Rule(id="O2_cleaner_cleans", kind="obligation", scope=by_look["cleaner"], pre=Cmp(feature="DIRT_FRACTION", cmp=">", value=0.3),
                          post=Cmp(feature="SINCE_AGENT_LAST_CLEANED", cmp="==", value=0), deadline=20, text="A cleaner must clean when the river is more than 30% dirty."))
    if by_look["free"]:
        rules.append(Rule(id="O3_free_cleans", kind="obligation", scope=by_look["free"], pre=Cmp(feature="DIRT_FRACTION", cmp=">", value=0.45),
                          post=Cmp(feature="SINCE_AGENT_LAST_CLEANED", cmp="==", value=0), deadline=20, text="An egalitarian must clean when the river is more than 45% dirty."))
    return rules


def rule_space(roles: list[str], max_len: int = 2) -> list[Rule]:
    """The hypothesis space, generated from the paper's feature grid (docs/research/norms.md, rule_generation.py)."""
    names = agent_names(roles)
    feats = [
        FeatureSpec(name="DIRT_FRACTION", kind="threshold", cmp=">", thresholds=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
                    goal=Cmp(feature="SINCE_AGENT_LAST_CLEANED", cmp="==", value=0), group="env"),
        FeatureSpec(name="SINCE_AGENT_LAST_PAID", kind="threshold", cmp=">", thresholds=[10, 15, 20, 25, 30],
                    goal=Cmp(feature="SINCE_AGENT_LAST_PAID", cmp="==", value=0), group="agent"),
        FeatureSpec(name="HARVESTED", kind="bool", group="cell"),
        FeatureSpec(name="HARVEST_APPLES_AROUND", kind="threshold", cmp="<", thresholds=[1, 2, 3, 4, 5, 6, 7, 8], group="cell"),
        FeatureSpec(name="HARVESTED_FOREIGN", kind="bool", group="cell"),
        FeatureSpec(name="ORIENTATION", kind="categorical", thresholds=[0, 1, 2, 3], group="agent"),
    ]

    def prohib_ok(fs):
        n = {f.name for f in fs}
        if "SINCE_AGENT_LAST_PAID" in n:
            return False
        if len({f.group for f in fs}) != 1:
            return False
        if "HARVEST_APPLES_AROUND" in n and "HARVESTED" not in n:
            return False
        if "HARVESTED_FOREIGN" in n and "HARVESTED" in n and len(n) == 2:
            return False            # redundant: HARVESTED_FOREIGN implies HARVESTED
        return True

    def oblig_ok(fs):
        return len(fs) == 1 and fs[0].goal is not None

    space = RuleSpace(features=feats, action_classes=["MOVE"], roles=names, max_len=max_len,
                      prohibition_filter=prohib_ok, obligation_filter=oblig_ok)
    base = space.enumerate()
    out: list[Rule] = []
    k = 0
    for r in base:
        if r.kind == "prohibition":
            out.append(r.model_copy(update={"id": f"P{k}", "deadline": None}))
            k += 1
        else:
            # obligations are role-conditioned, as in the paper (one copy per look)
            for look in ROLES:
                scope = [n for n, rr in zip(names, roles) if rr == look]
                if not scope:
                    continue
                out.append(r.model_copy(update={"id": f"O{k}_{look}", "scope": scope, "deadline": 20, "text": f"[{look}] {r.describe()}"}))
                k += 1
    return out


# ----------------------------------------------------------------------------- spec

def make_spec(roles: list[str] | None = None, ascii_map: str = PAPER_MAP, horizon: int = 300, rules: list[Rule] | None = None,
              leak: str | None = None) -> GameSpec:
    roles = list(roles or ["cleaner", "farmer", "free", "free"])
    layout = Layout(ascii_map)
    names = agent_names(roles)
    n = len(roles)
    sch = schema(layout, n, horizon)
    views = make_views(layout, roles)
    if leak == "derived":
        # a planted leak: a derived feature that reads a field it has not declared
        views[names[0]] = views[names[0]].model_copy(update={"derived": views[names[0]].derived + [Derived(name="peek", reads=["t"], fn=lambda s: s.inventory)]})

    def initial(params, chance: Chance) -> State:
        return State(initial_world(layout, roles, chance))

    act = ActionSpec(name="act", kind="choice", domain=Act, on_illegal="default", default="NOOP", classes=ACTION_CLASS,
                     call_to_action="Choose your action.")

    def transition(s: State, joint: dict[str, Any], chance: Chance, ctx: StepContext) -> TransitionResult:
        w = world_of(s)
        rewards = {}
        order = list(range(n))
        for i in order:
            w, r = apply_action(layout, w, i, joint.get(names[i], "NOOP"))
            rewards[names[i]] = r
        for j in range(n):
            if w["zapped"][j]:
                rewards[names[j]] = rewards.get(names[j], 0.0) - ZAP_PENALTY_VALUE
        w = env_step(layout, w, chance)
        for j in range(n):
            w["zapped"][j] = False
        ev = [{"kind": "step", "t": w["t"], "dirt_fraction": round(dirt_fraction(layout, w["dirt"]), 4),
               "desiccated_ratio": round(desiccated_ratio(layout, w), 4)}]
        return TransitionResult(State(w), events=ev, rewards=rewards)

    def forward(s, role: str, action: Any) -> State:
        w = world_of(s)
        i = names.index(role)
        w, _ = apply_action(layout, w, i, action)
        return State(w)

    def terminal(s: State) -> bool:
        return s["t"] >= horizon

    def payoff(traj) -> dict[str, float]:
        tot = {nm: 0.0 for nm in names}
        for e in traj.of("transition"):
            for k, v in e.get("rewards", {}).items():
                tot[k] += v
        return tot

    phase = Phase(name="step", kind="simultaneous", actors=names, action=act, max_rounds=horizon, next=None)
    spec = GameSpec(id="norms_harvest", version="1.0", roles=names, state=sch, phases=[phase], views=views, initial=initial,
                    transition=transition, terminal=terminal, payoff=payoff, rules=list(rules) if rules is not None else default_rules(roles),
                    params={"roles": roles, "map": "paper" if ascii_map == PAPER_MAP else ("small" if ascii_map == SMALL_MAP else "custom"), "horizon": horizon},
                    forward=forward, doc="Norm-augmented Markov game on the rule_obeying_harvest gridworld (Oldenburg & Zhi-Xuan 2024).")
    spec.layout = layout          # type: ignore[attr-defined]  (used by planners; derived from public map only)
    return spec


ZAP_PENALTY_VALUE = 15.0
