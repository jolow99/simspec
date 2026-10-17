"""A tiny two-player game used by the core tests: a 3-round prisoner's dilemma where player 2 has a
hidden type, with one prohibition rule and one obligation rule, plus a text 'chat' phase."""
from __future__ import annotations

from simspec import (Enum, IntRange, Bool, Text, Field, StateSchema, State, ActionSpec, View, Derived, Phase,
                      GameSpec, Rule, Cmp, IsTrue, And, Sanction)
from simspec.domains import Bool as BoolDom
from simspec.spec import TransitionResult

Act = Enum(name="act", labels=["COOP", "DEFECT"])
Ty = Enum(name="type", labels=["NICE", "MEAN"])
Rounds = IntRange(name="round", lo=0, hi=3)
Score = IntRange(name="score", lo=0, hi=100)

schema = StateSchema(fields=[
    Field(name="round", type=Rounds, visibility="public"),
    Field(name="phase_done", type=BoolDom(name="b"), visibility="public"),
    Field(name="type2", type=Ty, visibility="private", owner="p2"),
    Field(name="secret", type=IntRange(name="secret", lo=0, hi=9), visibility="hidden"),
    Field(name="last_p1", type=Enum(name="last", labels=["NONE", "COOP", "DEFECT"]), visibility="public"),
    Field(name="last_p2", type=Enum(name="last", labels=["NONE", "COOP", "DEFECT"]), visibility="public"),
    Field(name="score_p1", type=Score, visibility="private", owner="p1"),
    Field(name="score_p2", type=Score, visibility="private", owner="p2"),
])

PAY = {("COOP", "COOP"): (3, 3), ("COOP", "DEFECT"): (0, 5), ("DEFECT", "COOP"): (5, 0), ("DEFECT", "DEFECT"): (1, 1)}


def initial(params, chance):
    ty = chance.choice("type2", ["NICE", "MEAN"])
    sec = chance.integers("secret", 0, 9)
    return State(dict(round=0, phase_done=False, type2=ty, secret=sec, last_p1="NONE", last_p2="NONE", score_p1=0, score_p2=0))


def transition(state, joint, chance, ctx):
    if ctx.phase == "chat":
        msgs = [{"channel": "chat", "from": r, "to": "all", "text": t} for r, t in joint.items()]
        return TransitionResult(state.update(phase_done=True), messages=msgs)
    a1, a2 = joint["p1"], joint["p2"]
    g1, g2 = PAY[(a1, a2)]
    noise = chance.integers("noise", 0, 1)
    return TransitionResult(state.update(round=state.round + 1, last_p1=a1, last_p2=a2,
                                         score_p1=state.score_p1 + g1 + noise, score_p2=state.score_p2 + g2),
                            rewards={"p1": g1 + noise, "p2": g2})


def terminal(state):
    return state.round >= 3


def payoff(traj):
    s = traj.final_state()
    return {"p1": float(s["score_p1"]), "p2": float(s["score_p2"])}


def make_spec(params=None, leak: str | None = None):
    """leak = 'view' plants a hidden field in a view; 'derived' plants a derived feature that reads one."""
    p1_fields = ["round", "last_p1", "last_p2", "score_p1"]
    p1_derived = [Derived(name="opp_defected", reads=["last_p2"], fn=lambda s: s.last_p2 == "DEFECT")]
    if leak == "view":
        p1_fields = p1_fields + ["type2"]
    if leak == "derived":
        p1_derived = p1_derived + [Derived(name="sneaky", reads=["last_p2"], fn=lambda s: s.type2 == "MEAN")]
    views = {
        "p1": View(role="p1", fields=p1_fields, derived=p1_derived, channels=["chat"]),
        "p2": View(role="p2", fields=["round", "last_p1", "last_p2", "score_p2", "type2"], channels=["chat"]),
    }
    phases = [
        Phase(name="chat", kind="sequential", actors=["p1", "p2"],
              action=ActionSpec(name="say", kind="text", text=Text(name="msg", max_chars=200)),
              max_rounds=1, next="play"),
        Phase(name="play", kind="simultaneous", actors=["p1", "p2"],
              action=ActionSpec(name="move", kind="choice", domain=Act, on_illegal="default", default="COOP"),
              max_rounds=3, next=None),
    ]
    rules = [
        Rule(id="no_defect_after_coop", kind="prohibition", scope=["p1"], action_class="DEFECT",
             post=Cmp(feature="last_p2", cmp="==", value="COOP"), sanction=Sanction(kind="transfer", amount=1.0, to="p2")),
        Rule(id="reciprocate", kind="obligation", scope=["p2"], pre=Cmp(feature="last_p1", cmp="==", value="COOP"),
             post=Cmp(feature="last_p2", cmp="==", value="COOP"), deadline=1),
    ]
    return GameSpec(id="toy_pd", version="1", roles=["p1", "p2"], state=schema, phases=phases, views=views,
                    initial=initial, transition=transition, terminal=terminal, payoff=payoff, rules=rules,
                    params=params or {}, channels=["chat"])
