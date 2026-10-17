"""The contract: a closed structured schema (the paper's translator output C^omega), the translator
from text to schema, the validator, and the clause semantics as Rules.

C^omega = (prices p_d, delivery schedule q_{w,d} for the 5 production weeks, payment schedule M_w for
the 6 payment weeks, clauses kappa subset of {substitution, grim_trigger, payment_deduction, rollover}).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, field_validator

from ...rules import Rule, PyPred, Sanction
from .params import PRODUCTS, PAY_WEEKS, PROD_WEEKS, CLAUSES, SUBSTITUTION, EnvParams


class Contract(BaseModel):
    """Parse-complete structured contract. ``None`` entries mean 'not stated' (parse-incomplete)."""
    model_config = ConfigDict(frozen=True)
    prices: dict[str, int]                               # product -> unit price
    schedule: dict[int, dict[str, int]]                  # production week -> product -> qty
    payments: dict[int, int]                             # payment week -> amount
    clauses: list[str] = []
    substitution_min: dict[str, int] = {"pizza": 1, "pasta": 1, "soup": 1}
    rollover_cap: int = 2
    text: str = ""

    @field_validator("clauses")
    @classmethod
    def _clauses(cls, v):
        bad = [c for c in v if c not in CLAUSES]
        if bad:
            raise ValueError(f"unknown clauses {bad}")
        return sorted(set(v))

    def is_complete(self) -> bool:
        return (all(p in self.prices for p in PRODUCTS) and all(w in self.schedule and all(p in self.schedule[w] for p in PRODUCTS) for w in PROD_WEEKS)
                and all(w in self.payments for w in PAY_WEEKS))

    def total_payments(self) -> int:
        return sum(self.payments.values())

    def total_value(self, values: dict[str, int]) -> int:
        return sum(values[p] * q for w in PROD_WEEKS for p, q in self.schedule[w].items())

    def to_state_fields(self) -> dict[str, Any]:
        return {
            "contract_prices": [self.prices[p] for p in PRODUCTS],
            "contract_schedule": [self.schedule[w][p] for w in PROD_WEEKS for p in PRODUCTS],
            "contract_payments": [self.payments[w] for w in PAY_WEEKS],
            "contract_clauses": [c in self.clauses for c in CLAUSES],
        }

    @classmethod
    def from_state(cls, s) -> "Contract | None":
        if not s["agreed"]:
            return None
        return cls(prices={p: s["contract_prices"][i] for i, p in enumerate(PRODUCTS)},
                   schedule={w: {p: s["contract_schedule"][wi * 3 + pi] for pi, p in enumerate(PRODUCTS)} for wi, w in enumerate(PROD_WEEKS)},
                   payments={w: s["contract_payments"][i] for i, w in enumerate(PAY_WEEKS)},
                   clauses=[c for i, c in enumerate(CLAUSES) if s["contract_clauses"][i]], text=s["contract_text"])

    def render(self, story: dict | None = None) -> str:
        outs = (story or {}).get("outputs", PRODUCTS)
        names = dict(zip(PRODUCTS, outs))
        lines = ["CONTRACT",
                 "Unit prices: " + ", ".join(f"{names[p]} ${self.prices[p]}" for p in PRODUCTS),
                 "Deliveries (each production week): " + "; ".join(f"week {w}: " + ", ".join(f"{self.schedule[w][p]} {names[p]}" for p in PRODUCTS) for w in PROD_WEEKS),
                 "Payments: " + ", ".join(f"week {w}: ${self.payments[w]}" for w in PAY_WEEKS),
                 "Clauses: " + (", ".join(self.clauses) if self.clauses else "none")]
        return "\n".join(lines)


@dataclass
class Validation:
    ok: bool
    problems: list[str] = field(default_factory=list)


def validate_contract(c: Contract, params: EnvParams) -> Validation:
    """Deterministic validator (paper: arithmetic consistency, budget, minimums; inconsistencies are
    retained as outcomes, so this only reports)."""
    probs = []
    if not c.is_complete():
        probs.append("parse-incomplete")
        return Validation(False, probs)
    if c.total_payments() > params.budget:
        probs.append(f"payments {c.total_payments()} exceed budget {params.budget}")
    for w in PROD_WEEKS:
        for i, p in enumerate(PRODUCTS):
            if c.schedule[w][p] < params.minimums[i]:
                probs.append(f"week {w}: {p} {c.schedule[w][p]} below minimum {params.minimums[i]}")
    implied = sum(c.prices[p] * c.schedule[w][p] for w in PROD_WEEKS for p in PRODUCTS)
    if implied != c.total_payments():
        probs.append(f"unit prices x quantities = {implied} but payments total {c.total_payments()}")
    return Validation(len(probs) == 0, probs)


# ----------------------------------------------------------------------------- translators

class Translator:
    """text -> Contract | None. The structured translator expects a JSON block in the contract text
    (the prompts ask agents to include one); an LLM translator (as in the paper) has the same interface."""

    name = "structured-json"

    def __call__(self, text: str) -> Contract | None:
        obj = _find_contract_json(text)
        if obj is None:
            return None
        try:
            prices = {p: int(obj["prices"][p]) for p in PRODUCTS}
            sched = {int(w): {p: int(obj["schedule"][str(w)][p]) for p in PRODUCTS} for w in PROD_WEEKS} if isinstance(obj.get("schedule"), dict) and all(str(w) in obj["schedule"] for w in PROD_WEEKS) else \
                    {int(w): {p: int(obj["schedule"][w][p]) for p in PRODUCTS} for w in PROD_WEEKS}
            pays = {int(w): int(obj["payments"][str(w)]) for w in PAY_WEEKS} if all(str(w) in obj.get("payments", {}) for w in PAY_WEEKS) else \
                   {int(w): int(obj["payments"][w]) for w in PAY_WEEKS}
            clauses = [c for c in obj.get("clauses", []) if c in CLAUSES]
            return Contract(prices=prices, schedule=sched, payments=pays, clauses=clauses, text=text)
        except (KeyError, TypeError, ValueError):
            return None

    def describe(self) -> dict[str, Any]:
        return {"translator": self.name}


def _find_contract_json(text: str) -> dict | None:
    cands = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        cands.append(text[i:j + 1])
    for c in cands:
        try:
            obj = json.loads(c)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and "prices" in obj and "schedule" in obj and "payments" in obj:
            return obj
    return None


def contract_json_template() -> str:
    return json.dumps({"prices": {p: 0 for p in PRODUCTS},
                       "schedule": {str(w): {p: 0 for p in PRODUCTS} for w in PROD_WEEKS},
                       "payments": {str(w): 0 for w in PAY_WEEKS},
                       "clauses": []})


# ----------------------------------------------------------------------------- clause semantics as rules

def due_payment(c: Contract, week: int, deduction_allowed: int) -> int:
    """Contractually due amount this payment week after any authorised deduction."""
    return max(c.payments[week] - deduction_allowed, 0)


def scheduled(c: Contract, week: int) -> dict[str, int]:
    return dict(c.schedule[week])


def credited_delivery(c: Contract, delivered: dict[str, int], week: int) -> tuple[bool, dict[str, int]]:
    """Returns (met, shortfall per product). With the substitution clause, above-minimum output is
    credited at 4:2:1; without it, each product must meet its scheduled quantity."""
    sched = scheduled(c, week)
    short = {p: max(sched[p] - delivered.get(p, 0), 0) for p in PRODUCTS}
    if "substitution" not in c.clauses:
        return all(v == 0 for v in short.values()), short
    mins = c.substitution_min
    if any(delivered.get(p, 0) < mins[p] for p in PRODUCTS):
        return False, short
    need = sum(SUBSTITUTION[p] * max(sched[p] - mins[p], 0) for p in PRODUCTS)
    have = sum(SUBSTITUTION[p] * max(delivered.get(p, 0) - mins[p], 0) for p in PRODUCTS)
    return have >= need, short


def contract_rules(c: Contract) -> list[Rule]:
    """The contract as role-specific absorbing constraints (C^omega_{i,t}) with grim trigger as release."""
    grim = "grim_trigger" in c.clauses
    cust_sanction = Sanction(kind="release", to="supplier") if grim else None
    supp_sanction = Sanction(kind="release", to="customer") if grim else None

    def customer_violation(h, _a=None):
        s = h["state"]
        if s["stage_done"] != "payment":
            return False
        return any(ev.get("kind") == "payment" and ev["paid"] < ev["due"] for ev in h["events"])

    def supplier_violation(h, _a=None):
        return any(ev.get("kind") == "delivery" and bool(ev.get("violation", False)) for ev in h["events"])

    return [
        Rule(id="C_customer", kind="constraint", scope=["customer"], classifier=PyPred(name="customer_pays_due", fn=customer_violation),
             sanction=cust_sanction, tolerance=0.05, text=c.text or None, source="contract"),
        Rule(id="C_supplier", kind="constraint", scope=["supplier"], classifier=PyPred(name="supplier_delivers", fn=supplier_violation),
             sanction=supp_sanction, tolerance=0.05, text=c.text or None, source="contract"),
    ]
