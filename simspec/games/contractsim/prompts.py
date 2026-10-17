"""Observation renderers and scaffold configs for LLM agents playing ContractSim.

The renderers read only the Observation (never state). Text is compact but carries every rule the
paper's prompts state: production-week process, caps, pay-for-ordered, no refunds, BATNA, the
negotiation protocol asymmetry, and the structured JSON block required in every proposal.
"""
from __future__ import annotations

import json

from ...spec import Observation
from ...agents.llm import ScaffoldConfig
from .params import PRODUCTS, INGREDIENTS, PAY_WEEKS, PROD_WEEKS, CLAUSES, STORIES, EnvParams
from .contract import contract_json_template

RULES_COMMON = (
    "The game has a negotiation phase (at most 50 rounds; each round the Customer moves, then the Supplier) followed, if a contract is agreed, "
    "by an 11-week performance phase: odd weeks 1,3,5,7,9,11 are payment weeks (the Customer chooses an amount to pay), even weeks 2,4,6,8,10 are "
    "production weeks (the Supplier orders ingredients, then produces and delivers products). Week 11 is a final payment with no production after it. "
    "Recipes: pizza = 1 mozzarella + 1 basil + 1 tomato; pasta = 1 mozzarella + 1 basil; soup = 1 tomato. The Customer needs at least 1 of each product "
    "per production week. There is no renegotiation, no refund and no third-party enforcement after agreement."
)
NEGOTIATION_RULES = (
    "Negotiation protocol: dialogue and formal proposals are visible to both parties; private information is not. A proposal replaces the previous active proposal. "
    "Only the Customer can accept, and only the Supplier's formal proposal from the immediately preceding round; the Supplier never accepts (it re-proposes the agreed terms instead). "
    "Proposing and accepting in the same turn is not allowed. Either side may terminate; reaching the round limit means no deal. "
    "Every proposal must state the unit price of each product, the exact quantity of each product for each production week (2,4,6,8,10), and the exact payment for each payment week (1,3,5,7,9,11), "
    "all whole numbers, and must include this JSON block (clauses may be any of " + ", ".join(CLAUSES) + "):\n```json\n" + contract_json_template() + "\n```"
)

CUSTOMER_SYSTEM = ("You are the {role} (the buyer) in a supply-contracting game. " + RULES_COMMON + " " +
                   "You privately value each delivered product; your utility is budget + value of delivered products - payments made. "
                   "If no contract is agreed you keep your budget and an outside provider covers your minimum needs at exactly your private value. "
                   "Answer only in the requested JSON format.")
SUPPLIER_SYSTEM = ("You are the {role} (the supplier/caterer) in a supply-contracting game. " + RULES_COMMON + " " +
                   "Each production week: (1) carried inventory spoils, (2) you observe this round's ingredient prices and order (max 12 per ingredient, must be affordable from your balance), "
                   "(3) you receive a possibly smaller amount but pay for the full order, (4) inventory is capped at 10 per ingredient (excess is lost but paid for), "
                   "(5) you produce and deliver immediately, (6) prices are re-randomised. Your utility is payments received - ingredient costs. "
                   "If no contract is agreed you walk away with your initial capital. Answer only in the requested JSON format.")


def _contract_lines(obs: Observation, story: dict) -> list[str]:
    if not obs["agreed"]:
        return ["Contract: none agreed yet."]
    outs = dict(zip(PRODUCTS, story["outputs"]))
    sched = obs["contract_schedule"]
    lines = ["Agreed contract:",
             "  unit prices: " + ", ".join(f"{outs[p]} ${obs['contract_prices'][i]}" for i, p in enumerate(PRODUCTS)),
             "  deliveries: " + "; ".join(f"week {w}: " + ", ".join(f"{sched[wi*3+pi]} {outs[p]}" for pi, p in enumerate(PRODUCTS)) for wi, w in enumerate(PROD_WEEKS)),
             "  payments: " + ", ".join(f"week {w}: ${obs['contract_payments'][i]}" for i, w in enumerate(PAY_WEEKS)),
             "  clauses: " + (", ".join(c for i, c in enumerate(CLAUSES) if obs["contract_clauses"][i]) or "none")]
    return lines


def _negotiation_lines(obs: Observation) -> list[str]:
    lines = [f"Negotiation round {obs['neg_round']} (of at most 50)."]
    if obs["active_proposal"]:
        lines.append(f"Active proposal (by {obs['proposal_by']}, round {obs['proposal_round']}):\n{obs['active_proposal']}")
    else:
        lines.append("No active proposal.")
    if obs.role == "customer":
        eligible = obs["proposal_by"] == "supplier" and obs["proposal_round"] == obs["neg_round"] - 1
        lines.append("You MAY accept the active proposal this turn." if eligible else "You may NOT accept this turn (no eligible Supplier proposal from the previous round).")
    return lines


def make_renderer(params: EnvParams):
    story = STORIES[params.story]

    def render(obs: Observation) -> str:
        role_name = story[obs.role]
        lines = [f"You are the {role_name}. Week {obs['week']}, phase: {obs.phase}."]
        if obs.phase == "negotiate":
            lines += _negotiation_lines(obs)
            lines.append(NEGOTIATION_RULES)
        lines += _contract_lines(obs, story)
        if obs.role == "customer":
            lines.append("Your private values per unit: " + ", ".join(f"{o} ${v}" for o, v in zip(story["outputs"], obs["values"])))
            lines.append(f"Budget remaining: ${obs['budget_remaining']}. Total paid so far: ${obs['total_paid']}.")
        else:
            lines.append(f"Balance: ${obs['balance']}. Inventory: " + ", ".join(f"{i} {q}" for i, q in zip(story["inputs"], obs["inventory"])))
            lines.append("Current ingredient prices: " + ", ".join(f"{i} ${p}" for i, p in zip(story["inputs"], obs["prices"])))
            lines.append("Private information about supply uncertainty: " + obs["laws"])
            if obs["week"] > 0:
                lines.append(f"Last order {obs['last_order']}, received {obs['last_received']}, spoilage this round {obs['last_spoilage']}.")
        if obs["agreed"]:
            lines.append(f"Paid this week: ${obs['paid_this_week']}. Delivered this week: {obs['delivered_this_week']}. Cumulative deliveries: {obs['cum_delivered']}.")
            lines.append(f"Contract violations so far: customer {obs['violations_customer']}, supplier {obs['violations_supplier']}. "
                         f"Deduction allowed this week: ${obs['deduction_allowed']}. Carried deficit: {obs['deficit']}.")
        if obs.phase == "payment":
            lines.append("Decide your payment for this week. Reply with JSON: {\"action\": {\"amount\": <whole dollars>}}.")
        elif obs.phase == "order":
            lines.append("Decide your ingredient order [mozzarella, basil, tomato]. Reply with JSON: {\"action\": {\"order\": [m, b, t]}}.")
        elif obs.phase == "produce":
            lines.append("Decide what to produce [pizza, pasta, soup] from your inventory. Reply with JSON: {\"action\": {\"produce\": [p, q, s]}}.")
        elif obs.phase == "negotiate":
            if obs.role == "customer":
                lines.append('Reply with JSON: {"action": {"response": "<dialogue>", "contract": "<full contract text incl. JSON block, or empty>", "accept": <true|false>, "terminate": <true|false>}}.')
            else:
                lines.append('Reply with JSON: {"action": {"response": "<dialogue>", "contract": "<full contract text incl. JSON block, or empty>", "terminate": <true|false>}}.')
        return "\n".join(lines)
    return render


def scaffold(role: str, model: str, provider: str = "openai", **overrides) -> ScaffoldConfig:
    base = dict(model=model, provider=provider, elicitation="json", history_k=6, include_messages=True,
                decoding={"temperature": 0.0, "max_tokens": 1500}, max_retries=2, on_parse_failure="default",
                system_template=CUSTOMER_SYSTEM if role == "customer" else SUPPLIER_SYSTEM,
                turn_template="{observation}\n\n{history}{format}", renderer="contractsim.v1")
    base.update(overrides)
    return ScaffoldConfig(**base)
