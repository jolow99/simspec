"""Scripted baselines from the paper: Rational Complier (RC), Rational Exploiter (RE, non-engagement),
Rational Conditional Complier (RCC), plus a simple negotiator that proposes/accepts a given contract.

RC Supplier here is a P_sat-first safety-stock heuristic (order enough that the receipt law covers the
scheduled production with probability >= 1 - eps), with the paper's heuristic production rule. It is
not the paper's exact constrained-MDP DP; see docs/decisions.md D-011.
"""
from __future__ import annotations

import json
import math
from typing import Any

from ...agents.base import BaseAgent, ActionDistribution, ActContext, ScriptedAgent
from ...card import ScaffoldSection
from ...spec import Observation, ActionSpec
from .params import EnvParams, INGREDIENTS, PRODUCTS, RECIPES, SUBSTITUTION, PAY_WEEKS, PROD_WEEKS, receipt_support
from .contract import Contract, due_payment, contract_json_template


def contract_from_obs(obs: Observation) -> Contract | None:
    if not obs["agreed"]:
        return None
    return Contract(prices={p: obs["contract_prices"][i] for i, p in enumerate(PRODUCTS)},
                    schedule={w: {p: obs["contract_schedule"][wi * 3 + pi] for pi, p in enumerate(PRODUCTS)} for wi, w in enumerate(PROD_WEEKS)},
                    payments={w: obs["contract_payments"][i] for i, w in enumerate(PAY_WEEKS)},
                    clauses=[c for i, c in enumerate(["substitution", "grim_trigger", "payment_deduction", "rollover"]) if obs["contract_clauses"][i]])


def counterparty_breached(obs: Observation, me: str) -> bool:
    return (obs["violations_supplier"] > 0) if me == "customer" else (obs["violations_customer"] > 0)


# ----------------------------------------------------------------------------- negotiation scripts

class FixedContractNegotiator:
    """Supplier proposes `contract` in round 0; customer accepts it in round 1 (if it is the active supplier proposal)."""

    def __init__(self, contract: Contract):
        self.contract = contract
        self.text = contract.render() + "\n```json\n" + json.dumps({
            "prices": contract.prices, "schedule": {str(w): contract.schedule[w] for w in PROD_WEEKS},
            "payments": {str(w): contract.payments[w] for w in PAY_WEEKS}, "clauses": contract.clauses}) + "\n```"

    def customer(self, obs: Observation, spec: ActionSpec):
        if obs["proposal_by"] == "supplier" and obs["proposal_round"] == obs["neg_round"] - 1:
            return {"response": "Agreed.", "contract": "", "accept": True, "terminate": False}
        return {"response": "Please propose your contract.", "contract": "", "accept": False, "terminate": False}

    def supplier(self, obs: Observation, spec: ActionSpec):
        if obs["neg_round"] == 0 or obs["proposal_by"] != "supplier":
            return {"response": "Here is my proposal.", "contract": self.text, "terminate": False}
        return {"response": "Awaiting your acceptance.", "contract": "", "terminate": False}


# ----------------------------------------------------------------------------- performance policies

def rc_customer_payment(obs: Observation) -> int:
    c = contract_from_obs(obs)
    if c is None:
        return 0
    due = due_payment(c, obs["week"], obs["deduction_allowed"])
    return max(0, min(due, obs["budget_remaining"]))


def safety_order(need: int, law: str, eps: float, cap: int) -> int:
    """Smallest order x <= cap such that P(receipt >= need) >= 1 - eps; cap if impossible."""
    if need <= 0:
        return 0
    for x in range(need, cap + 1):
        sup = receipt_support(law, x)
        p = sum(1 for r in sup if r >= need) / len(sup)
        if p >= 1 - eps:
            return x
    return cap


def heuristic_production(inventory: list[int], target: list[int]) -> list[int]:
    """Paper's production rule: prefer full fulfilment of the scheduled quantities; otherwise maximise
    credited value with weights 4:2:1 (pizza, pasta, soup); then minimise ingredient use."""
    inv = list(inventory)
    out = [0, 0, 0]
    # full fulfilment first, in priority order pizza > pasta > soup (weights)
    order = sorted(range(3), key=lambda i: -SUBSTITUTION[PRODUCTS[i]])
    for i in order:
        p = PRODUCTS[i]
        k = target[i]
        while k > 0 and all(inv[INGREDIENTS.index(ing)] >= n for ing, n in RECIPES[p].items()):
            for ing, n in RECIPES[p].items():
                inv[INGREDIENTS.index(ing)] -= n
            out[i] += 1
            k -= 1
    # (no overproduction beyond the schedule: minimise ingredient use)
    return out


class RCSupplierPolicy:
    """Order to cover the scheduled production of this week (plus any carried deficit) with safety stock,
    subject to cash and caps; produce with the heuristic rule. Mode 'rc' | 're' | 'rcc'."""

    def __init__(self, params: EnvParams, eps: float = 0.05, mode: str = "rc"):
        self.params = params
        self.eps = eps
        self.mode = mode
        self.defected = False

    def _engaged(self, obs: Observation) -> bool:
        if self.mode == "re":
            return False
        if self.mode == "rcc" and counterparty_breached(obs, "supplier"):
            self.defected = True
            return False
        return True

    def order(self, obs: Observation, spec: ActionSpec):
        c = contract_from_obs(obs)
        if c is None or not self._engaged(obs):
            return {"order": [0, 0, 0]}
        week = obs["week"]
        target = [c.schedule[week][p] + obs["deficit"][i] for i, p in enumerate(PRODUCTS)]
        need = [0, 0, 0]
        for i, p in enumerate(PRODUCTS):
            for ing, n in RECIPES[p].items():
                need[INGREDIENTS.index(ing)] += n * target[i]
        inv = obs["inventory"]
        prices = obs["prices"]
        order = []
        for i in range(3):
            short = max(need[i] - inv[i], 0)
            x = safety_order(short, self.params.receipt, self.eps, self.params.order_cap)
            x = min(x, self.params.inventory_cap - inv[i] + short)   # no point ordering beyond what the cap lets us keep
            order.append(max(0, min(x, self.params.order_cap)))
        # cash constraint: scale down greedily (drop safety stock first, then quantities)
        budget = obs["balance"]
        cost = sum(q * p for q, p in zip(order, prices))
        while cost > budget and any(order):
            i = max(range(3), key=lambda j: order[j] * prices[j])
            order[i] -= 1
            cost = sum(q * p for q, p in zip(order, prices))
        return {"order": order}

    def produce(self, obs: Observation, spec: ActionSpec):
        c = contract_from_obs(obs)
        if c is None or not self._engaged(obs):
            return {"produce": [0, 0, 0]}
        week = obs["week"]
        target = [c.schedule[week][p] + obs["deficit"][i] for i, p in enumerate(PRODUCTS)]
        return {"produce": heuristic_production(obs["inventory"], target)}


class RCCustomerPolicy:
    def __init__(self, mode: str = "rc"):
        self.mode = mode

    def pay(self, obs: Observation, spec: ActionSpec):
        if self.mode == "re":
            return {"amount": 0}
        if self.mode == "rcc" and counterparty_breached(obs, "customer"):
            return {"amount": 0}
        return {"amount": rc_customer_payment(obs)}


# ----------------------------------------------------------------------------- agents

class PaperBaselineAgent(BaseAgent):
    """RC / RE / RCC for either role, with a scripted negotiator for the negotiation phase."""

    def __init__(self, role: str, params: EnvParams, mode: str = "rc", negotiator: FixedContractNegotiator | None = None, eps: float = 0.05):
        assert mode in ("rc", "re", "rcc")
        self.role = role
        self.mode = mode
        self.params = params
        self.negotiator = negotiator
        self.supplier = RCSupplierPolicy(params, eps, mode)
        self.customer = RCCustomerPolicy(mode)

    def act(self, obs: Observation, spec: ActionSpec, ctx: ActContext) -> ActionDistribution:
        if obs.phase == "negotiate":
            if self.negotiator is None:
                v = spec.default
            else:
                v = self.negotiator.customer(obs, spec) if self.role == "customer" else self.negotiator.supplier(obs, spec)
            return ActionDistribution.point(spec.key_of(v), source="scripted")
        if obs.phase == "payment":
            v = self.customer.pay(obs, spec)
        elif obs.phase == "order":
            v = self.supplier.order(obs, spec)
        elif obs.phase == "produce":
            v = self.supplier.produce(obs, spec)
        else:
            v = spec.default
        return ActionDistribution.point(spec.key_of(v), source="scripted")

    def describe(self, seat: str) -> ScaffoldSection:
        return ScaffoldSection(seat=seat, role=self.role, agent_class=f"contractsim:{self.mode.upper()}", distribution_source="scripted",
                               solver={"mode": self.mode, "eps": self.supplier.eps, "negotiator": "fixed-contract" if self.negotiator else "none",
                                       "note": "RC supplier = safety-stock heuristic + paper production rule (not the exact cMDP DP)"})
