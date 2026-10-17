import pytest

from simspec import run_episode
from simspec.games.contractsim import make_spec, env, Contract, PaperBaselineAgent, FixedContractNegotiator, PRODUCTS, PAY_WEEKS, PROD_WEEKS
from simspec.games.contractsim.memo_baseline import MemoTrustAgent, TrustGameTables, solve
from simspec.games.contractsim.metrics import performance_metrics, week_compliance, contract_value, summarize
from tests.test_contractsim import simple_contract


def contract_with_final_payment():
    c = simple_contract()
    pays = dict(c.payments)
    pays[1], pays[11] = 0, 37
    return Contract(prices=c.prices, schedule=c.schedule, payments=pays)


def test_memo_tables_shape_and_last_round_logic():
    p = env(1)
    c = simple_contract()
    t = TrustGameTables("supplier", p, c, 0.0, 1.0)
    Q = solve(t)
    assert Q.shape == (5, 21, 2, 2)
    # last production week has no payment after it (payment 11 = 0): defecting dominates complying
    assert Q[4, -1, 0, 1] > Q[4, -1, 0, 0]
    # first week with a trusting belief: complying dominates
    assert Q[0, -1, 0, 0] > Q[0, -1, 0, 1]
    # once the supplier has defected, a complier customer stops paying: complying is pure cost
    assert Q[0, -1, 1, 1] >= Q[0, -1, 1, 0]


def test_memo_supplier_complies_when_payments_are_enforceable():
    p = env(1)
    c = contract_with_final_payment()
    neg = FixedContractNegotiator(c)
    supp = MemoTrustAgent("supplier", p, c, neg, prior_complier=0.9)
    res = run_episode(make_spec(p), {"customer": PaperBaselineAgent("customer", p, "rcc", neg), "supplier": supp}, seed=0)
    s = res.final_state
    assert s["violations_supplier"] == 0 and s["violations_customer"] == 0
    assert all(e["source"] == "exact" for e in res.trajectory.agent_calls("supplier") if e["phase"] in ("order", "produce"))
    assert res.card.scaffolds["supplier"].agent_class == "memo:trust_pomdp"


def test_memo_supplier_stops_after_customer_breach():
    p = env(1)
    c = contract_with_final_payment()
    neg = FixedContractNegotiator(c)
    res = run_episode(make_spec(p), {"customer": PaperBaselineAgent("customer", p, "re", neg), "supplier": MemoTrustAgent("supplier", p, c, neg, prior_complier=0.9)}, seed=0)
    m = performance_metrics(res.trajectory, "supplier")
    assert m["reciprocal_defection"] and not m["unilateral_defection"]
    assert m["conditional_compliance"] == 1.0 and m["exploited_compliance"] == 0.0


def test_memo_customer_pays_when_trusting_and_surplus_exists():
    p = env(2)      # higher values: 20/10/5 vs prices 11/6/3 -> real surplus
    c = simple_contract()
    neg = FixedContractNegotiator(c)
    cust = MemoTrustAgent("customer", p, c, neg, prior_complier=0.9)
    res = run_episode(make_spec(p), {"customer": cust, "supplier": PaperBaselineAgent("supplier", p, "rcc", neg)}, seed=0)
    s = res.final_state
    # pays every week with a delivery after it; the final week-11 payment is 0 in this contract anyway
    assert s["violations_customer"] == 0 and s["violations_supplier"] == 0
    assert res.payoffs["customer"] > 200


def test_metrics_and_contract_value():
    p = env(1)
    c = simple_contract()
    neg = FixedContractNegotiator(c)
    res = run_episode(make_spec(p), {"customer": PaperBaselineAgent("customer", p, "rc", neg), "supplier": PaperBaselineAgent("supplier", p, "rc", neg)}, seed=0)
    comp = week_compliance(res.trajectory)
    assert set(comp["customer"]) == set(PAY_WEEKS) and set(comp["supplier"]) == set(PROD_WEEKS)
    m = performance_metrics(res.trajectory, "customer")
    assert m["compliance"] == 1.0 and not m["defection"]
    cv = contract_value(p, c, make_spec, n=3)
    assert cv["p_sat"] == 1.0 and cv["mutual_benefit"]
    summ = summarize([res.trajectory, res.trajectory], "supplier")
    assert summ["n"] == 2 and summ["defection"] == 0.0
