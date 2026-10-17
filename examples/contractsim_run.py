"""Run ContractSim (Catering) end to end with a chosen agent pairing and print the metrics.

Examples
  python examples/contractsim_run.py --env 1 --customer rcc --supplier memo --seeds 3
  python examples/contractsim_run.py --env 5 --customer llm --supplier rcc --model gpt-4.1-mini --provider openai
  python examples/contractsim_run.py --env 5 --customer llm --supplier memo --model fake      # scripted fake LLM, no key needed

Agents: rc | re | rcc (paper baselines), memo (memo trust-POMDP baseline), llm (LLM scaffold; uses a
cassette under runs/cassettes so re-runs replay recorded responses), decision (decision-model API).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simspec import run_episode, check_spec, replay_episode
from simspec.agents.llm import LLMAgent, FakeTransport, CassetteTransport, make_transport
from simspec.games.contractsim import make_spec, env, Contract, PaperBaselineAgent, FixedContractNegotiator, PRODUCTS, PAY_WEEKS, PROD_WEEKS
from simspec.games.contractsim.memo_baseline import MemoTrustAgent
from simspec.games.contractsim.prompts import make_renderer, scaffold
from simspec.games.contractsim.metrics import performance_metrics, contract_value, summarize


def default_contract(params) -> Contract:
    """A plausible negotiated contract for the environment: prices just under the customer's values,
    2/2/1 per production week, payments spread over weeks 1..11 (the final payment makes the last
    delivery enforceable)."""
    v = params.values
    prices = {"pizza": v[0] - 1, "pasta": v[1] - 1, "soup": max(v[2] - 1, 1)}
    qty = {"pizza": 2, "pasta": 2, "soup": 1}
    if 5 * sum(prices[p] * qty[p] for p in PRODUCTS) > params.budget:
        qty = {"pizza": 1, "pasta": 1, "soup": 1}
    sched = {w: dict(qty) for w in PROD_WEEKS}
    weekly = sum(prices[p] * qty[p] for p in PRODUCTS)
    pays = {1: weekly // 2, 3: weekly, 5: weekly, 7: weekly, 9: weekly, 11: weekly - weekly // 2}
    return Contract(prices=prices, schedule=sched, payments=pays, clauses=["grim_trigger"])


def fake_llm_script(role: str):
    """A scripted 'LLM' that answers in the right JSON shape: accepts/proposes, pays what is due,
    orders what the schedule needs, produces everything it can. Lets the pipeline run without a key."""
    def script(messages, params):
        user = messages[-1]["content"]
        if "phase: negotiate" in user:
            if role == "customer":
                accept = "You MAY accept" in user
                return json.dumps({"action": {"response": "Fine." if accept else "Please propose.", "contract": "", "accept": accept, "terminate": False}})
            return json.dumps({"action": {"response": "Here is my proposal.", "contract": "see JSON", "terminate": False}})
        if "phase: payment" in user:
            import re
            m = re.search(r"Deduction allowed this week: \$(\d+)", user)
            pays = re.search(r"payments: (.*)", user)
            week = int(re.search(r"Week (\d+)", user).group(1))
            due = 0
            if pays:
                for w, amt in re.findall(r"week (\d+): \$(\d+)", pays.group(1)):
                    if int(w) == week:
                        due = int(amt)
            due = max(due - (int(m.group(1)) if m else 0), 0)
            return json.dumps({"action": {"amount": due}})
        if "phase: order" in user:
            return json.dumps({"action": {"order": [4, 4, 3]}})
        if "phase: produce" in user:
            import re
            inv = [int(x) for x in re.search(r"Inventory: (.*)", user).group(1).replace("mozzarella", "").replace("basil", "").replace("tomato", "").replace(",", " ").split()]
            pizza = min(inv[0], inv[1], inv[2], 2)
            pasta = min(inv[0] - pizza, inv[1] - pizza, 2)
            soup = min(inv[2] - pizza, 1)
            return json.dumps({"action": {"produce": [pizza, pasta, soup]}})
        return "{}"
    return script


def make_agent(kind: str, role: str, params, contract: Contract, args) -> object:
    neg = FixedContractNegotiator(contract)
    if kind in ("rc", "re", "rcc"):
        return PaperBaselineAgent(role, params, kind, neg)
    if kind == "memo":
        return MemoTrustAgent(role, params, contract, neg, prior_complier=args.prior, violation_cost=args.violation_cost)
    if kind == "llm":
        cfg = scaffold(role, args.model, args.provider)
        if args.model == "fake":
            transport = FakeTransport(fake_llm_script(role), model="fake")
        else:
            os.makedirs("runs/cassettes", exist_ok=True)
            inner = make_transport(args.provider, args.model)
            transport = CassetteTransport(f"runs/cassettes/contractsim_{args.provider}_{args.model}_{role}.jsonl", inner, mode="auto")
        return LLMAgent(role, cfg, transport, renderer=make_renderer(params))
    if kind == "decision":
        raise SystemExit("decision-model agents answer finite choice questions; ContractSim's payment/order/produce actions are integer records. "
                         "Use examples/norms_run.py --learner decision for a decision-model agent, or wrap ContractSim actions in a finite menu.")
    raise SystemExit(f"unknown agent kind {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=int, default=1)
    ap.add_argument("--customer", default="rcc")
    ap.add_argument("--supplier", default="memo")
    ap.add_argument("--model", default="fake")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--prior", type=float, default=0.9)
    ap.add_argument("--violation-cost", type=float, default=0.0)
    ap.add_argument("--out", default="runs/contractsim")
    args = ap.parse_args()

    params = env(args.env)
    contract = default_contract(params)
    spec = make_spec(params)
    agents = {"customer": make_agent(args.customer, "customer", params, contract, args),
              "supplier": make_agent(args.supplier, "supplier", params, contract, args)}
    print("spec check:", check_spec(spec, agents, rollout_steps=40)["dynamic"])
    cv = contract_value(params, contract, make_spec, n=5)
    print(f"contract value under RCC x RCC: U={cv['U']} gains={cv['gains']} P_sat~{cv['p_sat']:.2f}")
    trajs = []
    for seed in range(args.seeds):
        run_dir = os.path.join(args.out, f"env{args.env}_{args.customer}_vs_{args.supplier}_seed{seed}")
        res = run_episode(spec, agents, seed=seed, run_dir=run_dir)
        trajs.append(res.trajectory)
        s = res.final_state
        print(f"seed {seed}: payoffs {res.payoffs} violations cust={s['violations_customer']} supp={s['violations_supplier']} -> {run_dir}")
        replay_episode(make_spec, run_dir)          # card test on every run
    for role in ("customer", "supplier"):
        print(role, summarize(trajs, role))
    print("replay-by-readback from card + log: OK for all runs")


if __name__ == "__main__":
    main()
