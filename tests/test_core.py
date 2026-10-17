import json
import os

import pytest

from simspec import (Enum, IntRange, Grid, Product, ScriptedAgent, run_episode, check_spec, replay_episode,
                      ObservationLeak, Trajectory, HarnessCard)
from simspec.domains import Bool
from simspec.rules import Rule, Cmp, IsTrue, And, Not, RuleEngine, RuleSpace, FeatureSpec
from simspec.chance import Chance
from simspec.agents.base import ActionDistribution, DryRunTransport, DryRunAttempted
from tests.toy import make_spec


# ----------------------------------------------------------------------------- domains

def test_domains_roundtrip():
    e = Enum(name="e", labels=["a", "b", "c"])
    assert [e.decode(e.encode(v)) for v in e.values()] == ["a", "b", "c"]
    r = IntRange(name="r", lo=-2, hi=3)
    assert r.size() == 6 and r.encode(3) == 5 and r.decode(0) == -2
    g = Grid(name="g", lo=0.0, hi=1.0, n=5)
    assert g.encode(0.25) == 1 and g.decode(4) == 1.0 and not g.contains(0.3)
    p = Product(name="p", fields={"x": IntRange(name="x", lo=0, hi=2), "y": e, "z": Bool(name="z")})
    assert p.size() == 18
    for i in range(p.size()):
        assert p.encode(p.decode(i)) == i
    assert "domain(x=3, y=3, z=2)" in p.to_memo()
    assert "IntEnum" in e.to_memo()


def test_product_matches_memo_domain():
    from memo import domain
    md = domain(x=3, y=4, z=2)
    p = Product(name="p", fields={"x": IntRange(name="x", lo=0, hi=2), "y": IntRange(name="y", lo=0, hi=3), "z": IntRange(name="z", lo=0, hi=1)})
    for i in range(p.size()):
        d = p.decode(i)
        assert md(d["x"], d["y"], d["z"]) == i
        assert tuple(md._tuple(i)) == (d["x"], d["y"], d["z"])


# ----------------------------------------------------------------------------- chance

def test_chance_logs_and_replays():
    c = Chance(1)
    a = c.integers("a", 0, 100); b = c.choice("b", ["x", "y", "z"]); u = c.uniform("u")
    log = c.drain()
    c2 = Chance(999, replay=log)
    assert c2.integers("a", 0, 100) == a and c2.choice("b", ["x", "y", "z"]) == b and c2.uniform("u") == u
    with pytest.raises(ValueError):
        Chance(0, replay=log).integers("wrong", 0, 1)


# ----------------------------------------------------------------------------- rules

def test_predicates_serialize_and_eval():
    p = And(args=[Cmp(feature="dirt", cmp=">", value=0.3), Not(arg=IsTrue(feature="done"))])
    assert p.eval({"dirt": 0.5, "done": False}) and not p.eval({"dirt": 0.1, "done": False})
    d = p.model_dump()
    assert d["op"] == "and" and d["args"][1]["op"] == "not"
    r = Rule(id="r", kind="obligation", scope=["a"], pre=p, post=IsTrue(feature="done"), deadline=2)
    assert Rule.model_validate(r.model_dump()) == r


def test_rule_engine_obligation_lifecycle():
    r = Rule(id="o", kind="obligation", scope=["a"], pre=Cmp(feature="dirty", cmp=">", value=0.3),
             post=IsTrue(feature="cleaned"), deadline=2)
    eng = RuleEngine([r])
    ev = eng.step_obligations(0, "a", {"dirty": 0.5, "cleaned": False})
    assert [e.status for e in ev] == ["triggered"]
    assert eng.step_obligations(1, "a", {"dirty": 0.5, "cleaned": False}) == []
    assert eng.step_obligations(2, "a", {"dirty": 0.5, "cleaned": False}) == []
    ev = eng.step_obligations(3, "a", {"dirty": 0.5, "cleaned": False})
    assert [e.status for e in ev] == ["violated"] and not eng.is_compliant("a")
    eng2 = RuleEngine([r])
    eng2.step_obligations(0, "a", {"dirty": 0.5, "cleaned": False})
    assert [e.status for e in eng2.step_obligations(1, "a", {"dirty": 0.5, "cleaned": True})] == ["discharged"]


def test_rule_engine_prohibition_and_sanction():
    r = Rule(id="p", kind="prohibition", scope=["a"], action_class="MOVE", post=IsTrue(feature="on_apple"),
             sanction=__import__("simspec").rules.Sanction(kind="transfer", amount=2.0, to="b"))
    eng = RuleEngine([r])
    assert eng.check_prohibitions(0, "a", "MOVE", {"on_apple": False}) == []
    ev = eng.check_prohibitions(1, "a", "MOVE", {"on_apple": True})
    assert ev[0].status == "violated" and eng.transfers == {"a": -2.0, "b": 2.0}
    assert eng.forbidden_action_classes("a", {"up": {"on_apple": True}, "down": {"on_apple": False}}, lambda k: "MOVE") == {"up"}


def test_rule_space_enumerates():
    space = RuleSpace(features=[
        FeatureSpec(name="dirt", kind="threshold", cmp=">", thresholds=[0.3, 0.4], goal=Cmp(feature="since_cleaned", cmp="==", value=0), group="env"),
        FeatureSpec(name="on_apple", kind="bool", group="cell"),
        FeatureSpec(name="apples_around", kind="threshold", cmp="<", thresholds=[1, 2], group="cell"),
    ], action_classes=["MOVE"], roles=["*"], max_len=2,
        prohibition_filter=lambda fs: len({f.group for f in fs}) == 1,
        obligation_filter=lambda fs: all(f.group == "env" for f in fs))
    rules = space.enumerate()
    kinds = [r.kind for r in rules]
    assert kinds.count("obligation") == 2          # dirt>0.3, dirt>0.4 -> clean
    assert kinds.count("prohibition") == 2 + 1 + 2 + 1 * 2   # dirt(2) + on_apple(1) + around(2) + on_apple&around(2)
    assert len({r.id for r in rules}) == len(rules)


# ----------------------------------------------------------------------------- engine, log, card, replay

def tit_for_tat(obs, spec):
    return "COOP" if obs["last_p2"] in ("NONE", "COOP") else "DEFECT"


def mean_or_nice(obs, spec):
    return "DEFECT" if obs["type2"] == "MEAN" else "COOP"


def agents():
    return {"p1": ScriptedAgent("p1", lambda o, s: "hi" if s.kind == "text" else tit_for_tat(o, s), "tft"),
            "p2": ScriptedAgent("p2", lambda o, s: "yo" if s.kind == "text" else mean_or_nice(o, s), "typed")}


def test_episode_runs_and_logs(tmp_path):
    spec = make_spec()
    res = run_episode(spec, agents(), seed=3, run_dir=str(tmp_path / "run"))
    traj = res.trajectory
    assert traj.steps() == 2 + 3                    # 2 chat turns + 3 simultaneous rounds
    assert set(res.payoffs) == {"p1", "p2"}
    kinds = {e["kind"] for e in traj.events}
    assert {"episode_start", "phase_start", "observation", "agent_call", "action", "chance", "transition", "message", "episode_end"} <= kinds
    assert all(e["source"] == "scripted" for e in traj.of("agent_call"))
    assert len(traj.messages("chat")) == 2
    assert os.path.exists(tmp_path / "run" / "events.jsonl") and os.path.exists(tmp_path / "run" / "card.json")
    card = HarnessCard.from_json(traj.card)
    assert card.card_hash == card.compute_hash() and card.game.spec_hash == spec.spec_hash()
    assert set(card.scaffolds) == {"p1", "p2"}
    # observations never contain hidden or foreign private fields
    for e in traj.of("observation"):
        v = e["obs"]["values"]
        assert "secret" not in v
        if e["role"] == "p1":
            assert "type2" not in v and "score_p2" not in v
            assert "opp_defected" in v


def test_rules_fire_in_episode(tmp_path):
    spec = make_spec()
    # p1 defects after p2 cooperates -> prohibition; p2 (NICE) cooperates after p1 coop -> obligation discharged
    ag = {"p1": ScriptedAgent("p1", lambda o, s: "x" if s.kind == "text" else "DEFECT"),
          "p2": ScriptedAgent("p2", lambda o, s: "y" if s.kind == "text" else "COOP")}
    res = run_episode(spec, ag, seed=0)
    statuses = [(e["rule_id"], e["status"]) for e in res.trajectory.rule_events()]
    assert ("no_defect_after_coop", "violated") in statuses
    assert ("no_defect_after_coop", "sanctioned") in statuses
    # sanction transfer applied to payoffs: p1 loses 1 per violation, p2 gains
    base = res.trajectory.final_state()
    n_viol = sum(1 for s in statuses if s == ("no_defect_after_coop", "violated"))
    assert res.payoffs["p1"] == base["score_p1"] - n_viol and res.payoffs["p2"] == base["score_p2"] + n_viol


def test_replay_by_readback_matches(tmp_path):
    spec = make_spec()
    run_dir = str(tmp_path / "run")
    res = run_episode(spec, agents(), seed=7, run_dir=run_dir)
    res2 = replay_episode(lambda params: make_spec(params), run_dir, verify=True)
    assert res2.payoffs == res.payoffs
    assert res2.trajectory.states() == res.trajectory.states()


def test_distribution_sampling_is_logged_and_seeded():
    spec = make_spec()
    mixed = {"p1": ScriptedAgent("p1", lambda o, s: "m" if s.kind == "text" else ActionDistribution(probs={"COOP": 0.5, "DEFECT": 0.5}, source="scripted")),
             "p2": ScriptedAgent("p2", lambda o, s: "n" if s.kind == "text" else "COOP")}
    a = run_episode(spec, mixed, seed=11).trajectory.actions("p1")
    b = run_episode(spec, mixed, seed=11).trajectory.actions("p1")
    c = run_episode(spec, mixed, seed=12).trajectory.actions("p1")
    assert [x["key"] for x in a] == [x["key"] for x in b]
    assert any(e["probs"] == {"COOP": 0.5, "DEFECT": 0.5} for e in run_episode(spec, mixed, seed=11).trajectory.agent_calls("p1"))
    assert len(c) == len(a)


def test_illegal_action_policy_default():
    spec = make_spec()
    bad = {"p1": ScriptedAgent("p1", lambda o, s: "m" if s.kind == "text" else ActionDistribution(probs={"NUKE": 1.0}, source="scripted")),
           "p2": ScriptedAgent("p2", lambda o, s: "n" if s.kind == "text" else "COOP")}
    res = run_episode(spec, bad, seed=0)
    acts = res.trajectory.actions("p1")
    moves = [a for a in acts if a["action"] == "move"]
    assert all(a["replaced"] and a["key"] == "COOP" for a in moves)


# ----------------------------------------------------------------------------- the checker (leak tests live in test_leaks.py)

def test_check_spec_passes_on_clean_spec():
    rep = check_spec(make_spec(), agents())
    assert rep["dynamic"]["ok"] and rep["dynamic"]["steps"] == 5


def test_dry_run_transport_raises():
    t = DryRunTransport()
    with pytest.raises(DryRunAttempted):
        t.complete([], {})


def test_replay_works_with_non_degenerate_distributions(tmp_path):
    spec = make_spec()
    mixed = {"p1": ScriptedAgent("p1", lambda o, s: "m" if s.kind == "text" else ActionDistribution(probs={"COOP": 0.5, "DEFECT": 0.5}, source="scripted")),
             "p2": ScriptedAgent("p2", lambda o, s: "n" if s.kind == "text" else ActionDistribution(probs={"COOP": 0.2, "DEFECT": 0.8}, source="scripted"))}
    run_dir = str(tmp_path / "mixed")
    res = run_episode(spec, mixed, seed=5, run_dir=run_dir)
    res2 = replay_episode(lambda params: make_spec(params), run_dir, verify=True)
    assert res2.payoffs == res.payoffs
