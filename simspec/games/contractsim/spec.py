"""ContractSim as a simspec: negotiation protocol phase + alternating payment/order/produce phases.

Fidelity notes vs arXiv 2608.10475 are listed in docs/decisions.md (D-010+).
"""
from __future__ import annotations

from typing import Any

from ...domains import Enum, IntRange, Bool, Table, Text, Product
from ...spec import Field, StateSchema, State, ActionSpec, View, Derived, Phase, GameSpec, TransitionResult, StepContext
from ...chance import Chance
from .params import (EnvParams, env as make_env, INGREDIENTS, PRODUCTS, RECIPES, L, PAY_WEEKS, PROD_WEEKS, CLAUSES,
                     receipt_support, spoilage_support, STORIES)
from .contract import Contract, Translator, due_payment, scheduled, credited_delivery, contract_rules

Ingredient = Enum(name="ingredient", labels=INGREDIENTS)
ProductD = Enum(name="product", labels=PRODUCTS)
Week = IntRange(name="week", lo=0, hi=L)
Stage = Enum(name="stage", labels=["negotiation", "payment", "order", "produce", "done"])
Money = IntRange(name="money", lo=0, hi=1000)
Qty = IntRange(name="qty", lo=0, hi=12)
Role = Enum(name="role", labels=["none", "customer", "supplier"])
ClauseIdx = Enum(name="clause", labels=CLAUSES)
SchedIdx = IntRange(name="sched_idx", lo=0, hi=len(PROD_WEEKS) * len(PRODUCTS) - 1)
PayIdx = IntRange(name="pay_idx", lo=0, hi=len(PAY_WEEKS) - 1)


def schema() -> StateSchema:
    F = Field
    return StateSchema(fields=[
        F(name="week", type=Week, visibility="public"),
        F(name="stage", type=Stage, visibility="public"),
        F(name="stage_done", type=Stage, visibility="public", doc="stage whose action was just resolved"),
        F(name="neg_round", type=IntRange(name="neg_round", lo=0, hi=51), visibility="public"),
        F(name="active_proposal", type=Text(name="proposal", max_chars=6000), visibility="public"),
        F(name="proposal_by", type=Role, visibility="public"),
        F(name="proposal_round", type=IntRange(name="pr", lo=0, hi=51), visibility="public"),
        F(name="agreed", type=Bool(name="b"), visibility="public"),
        F(name="terminated", type=Bool(name="b"), visibility="public"),
        F(name="contract_text", type=Text(name="contract_text", max_chars=6000), visibility="public"),
        F(name="contract_prices", type=Table(name="cp", index=ProductD, cell=IntRange(name="price", lo=0, hi=200)), visibility="public"),
        F(name="contract_schedule", type=Table(name="cs", index=SchedIdx, cell=IntRange(name="q", lo=0, hi=30)), visibility="public"),
        F(name="contract_payments", type=Table(name="cpay", index=PayIdx, cell=Money), visibility="public"),
        F(name="contract_clauses", type=Table(name="cc", index=ClauseIdx, cell=Bool(name="b")), visibility="public"),
        F(name="budget_remaining", type=Money, visibility="private", owner="customer"),
        F(name="total_paid", type=Money, visibility="public"),
        F(name="paid_this_week", type=Money, visibility="public"),
        F(name="balance", type=Money, visibility="private", owner="supplier"),
        F(name="inventory", type=Table(name="inv", index=Ingredient, cell=IntRange(name="inv", lo=0, hi=10)), visibility="private", owner="supplier"),
        F(name="prices", type=Table(name="prices", index=Ingredient, cell=IntRange(name="p", lo=1, hi=3)), visibility="private", owner="supplier"),
        F(name="last_order", type=Table(name="lo", index=Ingredient, cell=Qty), visibility="private", owner="supplier"),
        F(name="last_received", type=Table(name="lr", index=Ingredient, cell=Qty), visibility="private", owner="supplier"),
        F(name="last_spoilage", type=Table(name="ls", index=Ingredient, cell=IntRange(name="s", lo=0, hi=10)), visibility="private", owner="supplier"),
        F(name="delivered_this_week", type=Table(name="d", index=ProductD, cell=IntRange(name="q", lo=0, hi=30)), visibility="public"),
        F(name="cum_delivered", type=Table(name="cd", index=ProductD, cell=IntRange(name="q", lo=0, hi=200)), visibility="public"),
        F(name="cum_ordered_cost", type=Money, visibility="private", owner="supplier"),
        F(name="shortfall_last", type=Bool(name="b"), visibility="public"),
        F(name="deduction_allowed", type=Money, visibility="public"),
        F(name="deficit", type=Table(name="def", index=ProductD, cell=IntRange(name="d", lo=0, hi=30)), visibility="public"),
        F(name="violations_customer", type=IntRange(name="v", lo=0, hi=20), visibility="public"),
        F(name="violations_supplier", type=IntRange(name="v", lo=0, hi=20), visibility="public"),
        F(name="values", type=Table(name="values", index=ProductD, cell=IntRange(name="v", lo=0, hi=50)), visibility="private", owner="customer"),
        F(name="env_id", type=IntRange(name="env", lo=1, hi=6), visibility="private", owner="supplier", doc="which stochastic laws apply; the supplier knows them"),
    ])


# ----------------------------------------------------------------------------- initial

def initial(params: EnvParams, chance: Chance) -> State:
    prices = [chance.choice(f"price:{ing}", [p for p, _ in law], p=[q for _, q in law]) for ing, law in zip(INGREDIENTS, params.price_laws)]
    return State(dict(
        week=0, stage="negotiation", stage_done="negotiation", neg_round=0, active_proposal="", proposal_by="none", proposal_round=0,
        agreed=False, terminated=False, contract_text="", contract_prices=[0, 0, 0], contract_schedule=[0] * 15,
        contract_payments=[0] * 6, contract_clauses=[False] * 4,
        budget_remaining=params.budget, total_paid=0, paid_this_week=0, balance=params.capital,
        inventory=[0, 0, 0], prices=prices, last_order=[0, 0, 0], last_received=[0, 0, 0], last_spoilage=[0, 0, 0],
        delivered_this_week=[0, 0, 0], cum_delivered=[0, 0, 0], cum_ordered_cost=0, shortfall_last=False, deduction_allowed=0,
        deficit=[0, 0, 0], violations_customer=0, violations_supplier=0,
        values=list(params.values), env_id=params.env_id,
    ))


# ----------------------------------------------------------------------------- phases / actions

NegCustomer = ActionSpec(name="turn", kind="record",
                         record={"response": Text(name="response", max_chars=2000), "contract": Text(name="contract", max_chars=6000),
                                 "accept": Bool(name="b"), "terminate": Bool(name="b")},
                         call_to_action="It is your negotiation turn. You may continue the dialogue, propose or revise a complete contract, accept the Supplier's proposal from the previous round, or terminate.",
                         default={"response": "", "contract": "", "accept": False, "terminate": False})
NegSupplier = ActionSpec(name="turn", kind="record",
                         record={"response": Text(name="response", max_chars=2000), "contract": Text(name="contract", max_chars=6000),
                                 "terminate": Bool(name="b")},
                         call_to_action="It is your negotiation turn. You may continue the dialogue, propose or revise a complete contract, or terminate. You cannot accept; if you agree with the Customer's terms, propose the complete agreed contract so the Customer can accept it next round.",
                         default={"response": "", "contract": "", "terminate": False})

Pay = ActionSpec(name="pay", kind="record", record={"amount": Money},
                 call_to_action="Choose how much to pay this week (whole dollars, at most your remaining budget).",
                 valid=lambda s, r, v: 0 <= v["amount"] <= s["budget_remaining"], default={"amount": 0})

Order = ActionSpec(name="order", kind="record", record={"order": Table(name="order", index=Ingredient, cell=Qty)},
                   call_to_action="Choose how many units of each ingredient to order (0-12 each; total cost at current prices must not exceed your balance).",
                   default={"order": [0, 0, 0]})

Produce = ActionSpec(name="produce", kind="record", record={"produce": Table(name="produce", index=ProductD, cell=IntRange(name="q", lo=0, hi=10))},
                     call_to_action="Choose how many of each product to make from your inventory (recipes: pizza = mozzarella+basil+tomato, pasta = mozzarella+basil, soup = tomato). Products are delivered immediately.",
                     default={"produce": [0, 0, 0]})


def _order_valid(s: State, role: str, v: Any) -> bool:
    cost = sum(q * p for q, p in zip(v["order"], s["prices"]))
    return cost <= s["balance"]


def _produce_valid(s: State, role: str, v: Any) -> bool:
    need = _ingredients_needed(v["produce"])
    return all(n <= have for n, have in zip(need, s["inventory"]))


Order = Order.model_copy(update={"valid": _order_valid})
Produce = Produce.model_copy(update={"valid": _produce_valid})


def _ingredients_needed(produce: list[int]) -> list[int]:
    need = [0, 0, 0]
    for pi, p in enumerate(PRODUCTS):
        for ing, k in RECIPES[p].items():
            need[INGREDIENTS.index(ing)] += k * produce[pi]
    return need


def _after_negotiation(s: State) -> str | None:
    return "payment" if s["agreed"] else None


def _after_produce(s: State) -> str | None:
    return "payment" if s["week"] <= L else None


PHASES = [
    Phase(name="negotiate", kind="protocol", actors=["customer", "supplier"], action={"customer": NegCustomer, "supplier": NegSupplier},
          max_rounds=50, until=lambda s: s["agreed"] or s["terminated"], next=_after_negotiation),
    Phase(name="payment", kind="sequential", actors=["customer"], action=Pay, max_rounds=1, next="order"),
    Phase(name="order", kind="sequential", actors=["supplier"], action=Order, max_rounds=1, next="produce"),
    Phase(name="produce", kind="sequential", actors=["supplier"], action=Produce, max_rounds=1, next=_after_produce),
]


def terminal(s: State) -> bool:
    return s["stage"] == "done"


# ----------------------------------------------------------------------------- transition

def make_transition(params: EnvParams, translator: Translator):
    def transition(s: State, joint: dict[str, Any], chance: Chance, ctx: StepContext) -> TransitionResult:
        if ctx.phase == "negotiate":
            return _negotiate(s, joint, ctx, params, translator)
        if ctx.phase == "payment":
            return _payment(s, joint["customer"], chance, params)
        if ctx.phase == "order":
            return _order(s, joint["supplier"], chance, params)
        if ctx.phase == "produce":
            return _produce(s, joint["supplier"], chance, params)
        raise ValueError(ctx.phase)
    return transition


def _negotiate(s: State, joint: dict[str, Any], ctx: StepContext, params: EnvParams, translator: Translator) -> TransitionResult:
    role = ctx.actor
    turn = joint[role]
    msgs = []
    events = []
    rnd = s["neg_round"]
    upd: dict[str, Any] = {"stage_done": "negotiation"}
    if turn.get("response"):
        msgs.append({"channel": "dialogue", "from": role, "to": "all", "text": turn["response"][:2000]})
    if turn.get("terminate"):
        upd["terminated"] = True
        upd["stage"] = "done"
        events.append({"kind": "terminate", "by": role})
    elif role == "customer" and turn.get("accept") and not turn.get("contract"):
        if s["proposal_by"] == "supplier" and s["proposal_round"] == rnd - 1 and s["active_proposal"]:
            c = translator(s["active_proposal"])
            if c is not None and c.is_complete():
                upd.update(agreed=True, contract_text=s["active_proposal"], week=1, stage="payment", **c.to_state_fields())
                events.append({"kind": "agreement", "round": rnd, "contract": c.model_dump(exclude={"text"})})
            else:
                events.append({"kind": "accept_failed", "reason": "proposal not parse-complete"})
                msgs.append({"channel": "dialogue", "from": "system", "to": "all", "text": "Acceptance failed: the proposal is not a complete contract (prices, 5 delivery rows, 6 payments)."})
        else:
            events.append({"kind": "accept_invalid", "reason": "no eligible supplier proposal from the previous round"})
    elif turn.get("contract"):
        upd.update(active_proposal=turn["contract"][:6000], proposal_by=role, proposal_round=rnd)
        c = translator(turn["contract"])
        events.append({"kind": "proposal", "by": role, "round": rnd, "parsed": c is not None, "complete": bool(c and c.is_complete())})
        msgs.append({"channel": "proposals", "from": role, "to": "all", "text": turn["contract"][:6000]})
    if role == "supplier" and not upd.get("terminated"):
        upd["neg_round"] = rnd + 1
        if rnd + 1 >= params.max_rounds and not upd.get("agreed"):
            upd["terminated"] = True
            upd["stage"] = "done"
            events.append({"kind": "round_limit"})
    return TransitionResult(s.update(**upd), events=events, messages=msgs)


def _payment(s: State, a: dict[str, Any], chance: Chance, params: EnvParams) -> TransitionResult:
    c = Contract.from_state(s)
    amount = int(a["amount"])
    paid = max(0, min(amount, s["budget_remaining"]))
    due = due_payment(c, s["week"], s["deduction_allowed"]) if c else 0
    violation = paid < due
    upd = dict(budget_remaining=s["budget_remaining"] - paid, total_paid=s["total_paid"] + paid, paid_this_week=paid,
               balance=s["balance"] + paid, stage_done="payment", deduction_allowed=0,
               violations_customer=s["violations_customer"] + int(violation))
    ev = {"kind": "payment", "week": s["week"], "paid": paid, "due": due, "violation": violation}
    rewards = {"customer": -float(paid), "supplier": float(paid)}
    if s["week"] >= L:
        upd.update(stage="done")
        return TransitionResult(s.update(**upd), events=[ev], rewards=rewards)
    # spoilage on carried inventory happens before the supplier observes the purchase request
    inv = list(s["inventory"])
    sp = []
    for i, ing in enumerate(INGREDIENTS):
        shock = chance.choice(f"spoilage:{ing}", spoilage_support(params.spoilage))
        loss = min(shock, inv[i])
        inv[i] -= loss
        sp.append(loss)
    upd.update(week=s["week"] + 1, stage="order", inventory=inv, last_spoilage=sp, delivered_this_week=[0, 0, 0], paid_this_week=paid)
    return TransitionResult(s.update(**upd), events=[ev, {"kind": "spoilage", "week": s["week"] + 1, "loss": sp}], rewards=rewards)


def _order(s: State, a: dict[str, Any], chance: Chance, params: EnvParams) -> TransitionResult:
    order = [int(x) for x in a["order"]]
    prices = s["prices"]
    cost = sum(q * p for q, p in zip(order, prices))
    if cost > s["balance"]:            # should have been rejected by validity; clamp defensively
        order = [0, 0, 0]
        cost = 0
    received, lost = [], []
    inv = list(s["inventory"])
    for i, ing in enumerate(INGREDIENTS):
        r = chance.choice(f"receipt:{ing}", receipt_support(params.receipt, order[i]))
        room = params.inventory_cap - inv[i]
        kept = min(r, room)
        inv[i] += kept
        received.append(r)
        lost.append(r - kept)
    upd = dict(balance=s["balance"] - cost, inventory=inv, last_order=order, last_received=received,
               cum_ordered_cost=s["cum_ordered_cost"] + cost, stage="produce", stage_done="order")
    ev = {"kind": "order", "week": s["week"], "order": order, "prices": list(prices), "cost": cost, "received": received, "lost_to_cap": lost}
    return TransitionResult(s.update(**upd), events=[ev], rewards={"supplier": -float(cost)})


def _produce(s: State, a: dict[str, Any], chance: Chance, params: EnvParams) -> TransitionResult:
    c = Contract.from_state(s)
    produce = [int(x) for x in a["produce"]]
    need = _ingredients_needed(produce)
    inv = list(s["inventory"])
    if any(n > h for n, h in zip(need, inv)):   # infeasible: clamp to nothing (validity should prevent this)
        produce = [0, 0, 0]
        need = [0, 0, 0]
    inv = [h - n for h, n in zip(inv, need)]
    delivered = {p: produce[i] for i, p in enumerate(PRODUCTS)}
    week = s["week"]
    # contract accounting
    deficit = list(s["deficit"])
    deduction = 0
    violation = False
    met = True
    if c is not None:
        clauses = c.clauses
        eff = {p: delivered[p] + 0 for p in PRODUCTS}
        met, short = credited_delivery(c, eff, week)
        # rollover: a carried deficit must be cured this round; new deficit above cap is a violation
        if "rollover" in clauses:
            carried = {p: deficit[i] for i, p in enumerate(PRODUCTS)}
            # surplus this week cures carried deficit first
            new_def = {}
            for i, p in enumerate(PRODUCTS):
                extra = max(delivered[p] - c.schedule[week][p], 0)
                cured = min(extra, carried[p])
                uncured = carried[p] - cured
                new_def[p] = short[p]
                if uncured > 0:
                    violation = True
                if new_def[p] > c.rollover_cap:
                    violation = True
            met = met or all(short[p] == 0 for p in PRODUCTS)
            if not met and not violation:
                pass                                   # deficit carried, not yet a violation
            deficit = [min(new_def[p], 30) for p in PRODUCTS]
        elif "payment_deduction" in clauses:
            if not met:
                deduction = sum(c.prices[p] * short[p] for p in PRODUCTS)
                if s["shortfall_last"]:
                    violation = True                   # two consecutive shortfall rounds
        else:
            violation = not met
    cum = [s["cum_delivered"][i] + produce[i] for i in range(3)]
    # redraw prices for the next production round
    prices = [chance.choice(f"price:{ing}", [p for p, _ in law], p=[q for _, q in law]) for ing, law in zip(INGREDIENTS, params.price_laws)]
    upd = dict(inventory=inv, delivered_this_week=produce, cum_delivered=cum, prices=prices, week=week + 1, stage="payment",
               stage_done="produce", shortfall_last=(c is not None and not met), deduction_allowed=min(deduction, 1000),
               deficit=deficit, violations_supplier=s["violations_supplier"] + int(violation))
    ev = {"kind": "delivery", "week": week, "delivered": produce, "scheduled": [c.schedule[week][p] for p in PRODUCTS] if c else None,
          "met": met, "violation": violation, "deduction": deduction}
    rewards = {}
    return TransitionResult(s.update(**upd), events=[ev], rewards=rewards)


# ----------------------------------------------------------------------------- views

def _laws_description(params: EnvParams):
    def f(s):
        _ = s.env_id
        return describe_laws(params)
    return f


def describe_laws(params: EnvParams) -> str:
    pl = "; ".join(f"{ing}: " + ", ".join(f"${p} w.p. {q:.2f}" for p, q in law) for ing, law in zip(INGREDIENTS, params.price_laws))
    rec = {"exact": "you receive exactly what you order", "low": "you receive between 3/4 and all of what you order (uniform)",
           "high": "you receive between half and all of what you order (uniform)"}[params.receipt]
    spo = {"none": "no spoilage", "low": "0 or 1 unit of each carried ingredient spoils (uniform)", "high": "0, 1 or 2 units of each carried ingredient spoil (uniform)"}[params.spoilage]
    return f"Prices per unit next round: {pl}. Delivery: {rec}. Spoilage before each production round: {spo}."


COMMON_PUBLIC = ["week", "stage", "stage_done", "neg_round", "active_proposal", "proposal_by", "proposal_round", "agreed", "terminated",
                 "contract_text", "contract_prices", "contract_schedule", "contract_payments", "contract_clauses",
                 "total_paid", "paid_this_week", "delivered_this_week", "cum_delivered", "shortfall_last", "deduction_allowed", "deficit",
                 "violations_customer", "violations_supplier"]


def views(params: EnvParams) -> dict[str, View]:
    return {
        "customer": View(role="customer", fields=COMMON_PUBLIC + ["budget_remaining", "values"], channels=["dialogue", "proposals"], history="own_actions"),
        "supplier": View(role="supplier", fields=COMMON_PUBLIC + ["balance", "inventory", "prices", "last_order", "last_received", "last_spoilage", "cum_ordered_cost"],
                         derived=[Derived(name="laws", reads=["env_id"], fn=_laws_description(params), type=Text(name="laws", max_chars=1000))],
                         channels=["dialogue", "proposals"], history="own_actions"),
    }


# ----------------------------------------------------------------------------- payoff

def payoff(traj) -> dict[str, float]:
    s = traj.final_state()
    s0 = traj.initial_state()
    B = s0["budget_remaining"]
    values = s0["values"]
    if not s["agreed"]:
        return {"customer": float(B), "supplier": 0.0}
    cust = B + sum(v * q for v, q in zip(values, s["cum_delivered"])) - s["total_paid"]
    supp = s["total_paid"] - s["cum_ordered_cost"]
    return {"customer": float(cust), "supplier": float(supp)}


# ----------------------------------------------------------------------------- factory

def make_spec(params: EnvParams | dict | int | None = None, translator: Translator | None = None, leak: str | None = None) -> GameSpec:
    if params is None:
        params = make_env(1)
    elif isinstance(params, int):
        params = make_env(params)
    elif isinstance(params, dict):
        params = EnvParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()}) if "env_id" in params and len(params) > 2 else make_env(params.get("env_id", 1), params.get("story", "catering"))
    if not isinstance(params, EnvParams):
        raise TypeError(f"params must be EnvParams, env id or dict, got {type(params)}")
    translator = translator or Translator()
    v = views(params)
    if leak == "view":
        v["customer"] = v["customer"].model_copy(update={"fields": v["customer"].fields + ["inventory"]})
    if leak == "derived":
        v["customer"] = v["customer"].model_copy(update={"derived": [Derived(name="peek", reads=["total_paid"], fn=lambda s: s.balance)]})

    def dynamic_rules(s: State):
        c = Contract.from_state(s)
        return contract_rules(c) if c is not None else []

    return GameSpec(id="contractsim", version="1.0", roles=["customer", "supplier"], state=schema(), phases=PHASES, views=v,
                    initial=initial, transition=make_transition(params, translator), terminal=terminal, payoff=payoff,
                    rules=[], dynamic_rules=dynamic_rules, params=params, channels=["dialogue", "proposals"],
                    doc="ContractSim (Sajja, Kleiman-Weiner, Zimmermann, Zhi-Xuan 2026): negotiation-performance game between a Customer and a Supplier.")
