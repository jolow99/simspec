import json

import pytest

from simspec import run_episode, check_spec, replay_episode, ObservationLeak
from simspec.games.contractsim import make_spec, env, Contract, PaperBaselineAgent, FixedContractNegotiator, validate_contract, PRODUCTS, PAY_WEEKS, PROD_WEEKS


def simple_contract(prices=(11, 6, 3), qty=(2, 2, 1), clauses=()):
    sched = {w: {p: q for p, q in zip(PRODUCTS, qty)} for w in PROD_WEEKS}
    weekly = sum(p * q for p, q in zip(prices, qty))
    pays = {w: weekly for w in PAY_WEEKS[:-1]}
    pays[11] = 0
    return Contract(prices=dict(zip(PRODUCTS, prices)), schedule=sched, payments=pays, clauses=list(clauses))


def agents(params, cust="rc", supp="rc", contract=None):
    neg = FixedContractNegotiator(contract or simple_contract())
    return {"customer": PaperBaselineAgent("customer", params, cust, neg), "supplier": PaperBaselineAgent("supplier", params, supp, neg)}


def test_contract_validates_and_roundtrips():
    c = simple_contract()
    assert c.is_complete()
    v = validate_contract(c, env(1))
    assert v.ok, v.problems
    from simspec.games.contractsim.contract import Translator
    t = Translator()
    text = FixedContractNegotiator(c).text
    c2 = t(text)
    assert c2 is not None and c2.prices == c.prices and c2.schedule == c.schedule and c2.payments == c.payments
    assert t("blah blah") is None


def test_rc_vs_rc_deterministic_env_full_compliance():
    p = env(1)
    spec = make_spec(p)
    res = run_episode(spec, agents(p), seed=0)
    t = res.trajectory
    s = t.final_state()
    assert s["agreed"] and s["week"] == 11 and s["stage"] == "done"
    assert s["violations_customer"] == 0 and s["violations_supplier"] == 0
    assert all(e["status"] == "added" for e in t.rule_events())
    # payoffs: U_cust = B + sum v q' - sum M ; U_supp = sum M - sum o.p
    c = simple_contract()
    deliveries = sum(sum(c.schedule[w].values()) for w in PROD_WEEKS)
    assert sum(s["cum_delivered"]) == deliveries
    assert res.payoffs["customer"] == 200 + sum(v * q for v, q in zip(p.values, s["cum_delivered"])) - s["total_paid"]
    assert res.payoffs["supplier"] == s["total_paid"] - s["cum_ordered_cost"]
    assert res.payoffs["supplier"] > 0 and res.payoffs["customer"] > 200


def test_re_customer_triggers_supplier_rcc_grim():
    p = env(1)
    spec = make_spec(p)
    c = simple_contract(clauses=("grim_trigger",))
    res = run_episode(spec, agents(p, cust="re", supp="rcc", contract=c), seed=1)
    t = res.trajectory
    s = t.final_state()
    assert s["violations_customer"] >= 1
    statuses = [(e["rule_id"], e["status"]) for e in t.rule_events()]
    assert ("C_customer", "violated") in statuses and ("C_customer", "released") in statuses
    # the released supplier stops delivering after the breach
    deliveries = [e for e in t.events if e["kind"] == "transition" and e["phase"] == "produce"]
    assert sum(deliveries[-1]["state"]["delivered_this_week"]) == 0
    assert res.payoffs["customer"] <= 200


def test_no_agreement_gives_disagreement_payoffs():
    p = env(2)
    spec = make_spec(p)
    from simspec import ScriptedAgent
    term = {"customer": ScriptedAgent("customer", lambda o, s: {"response": "no", "contract": "", "accept": False, "terminate": True}),
            "supplier": ScriptedAgent("supplier", lambda o, s: s.default)}
    res = run_episode(spec, term, seed=0)
    assert res.payoffs == {"customer": 200.0, "supplier": 0.0}
    assert not res.final_state["agreed"]


@pytest.mark.parametrize("env_id", [3, 5])
def test_stochastic_envs_run_and_replay(tmp_path, env_id):
    p = env(env_id)
    spec = make_spec(p)
    res = run_episode(spec, agents(p), seed=env_id, run_dir=str(tmp_path / f"run{env_id}"))
    assert res.trajectory.steps() > 10
    draws = res.trajectory.chance_draws()
    assert any(d["name"].startswith("receipt:") for d in draws) and any(d["name"].startswith("spoilage:") for d in draws)
    res2 = replay_episode(lambda params: make_spec(params), str(tmp_path / f"run{env_id}"))
    assert res2.payoffs == res.payoffs


def test_customer_cannot_see_supplier_private_state():
    p = env(1)
    spec = make_spec(p)
    res = run_episode(spec, agents(p), seed=0)
    for e in res.trajectory.of("observation"):
        v = e["obs"]["values"]
        if e["role"] == "customer":
            assert "inventory" not in v and "prices" not in v and "balance" not in v and "laws" not in v
        else:
            assert "values" not in v and "budget_remaining" not in v and "laws" in v


def test_planted_leaks_are_caught_before_model_calls():
    p = env(1)
    with pytest.raises(ObservationLeak):
        check_spec(make_spec(p, leak="view"), agents(p))
    with pytest.raises(ObservationLeak):
        check_spec(make_spec(p, leak="derived"), agents(p))
    rep = check_spec(make_spec(p), agents(p), rollout_steps=100)
    assert rep["dynamic"]["ok"]
