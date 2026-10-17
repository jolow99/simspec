"""ContractSim environment parameters (Sajja et al. 2026, Appendix: six Catering environments).

Hotel Cleaning and AI Hosting are relabelings (x100 / x1000 money) of the same abstract game; the
`story` field only changes names in prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

INGREDIENTS = ["mozzarella", "basil", "tomato"]          # I1, I2, I3
PRODUCTS = ["pizza", "pasta", "soup"]                     # O1 = I1+I2+I3, O2 = I1+I2, O3 = I3
RECIPES = {"pizza": {"mozzarella": 1, "basil": 1, "tomato": 1}, "pasta": {"mozzarella": 1, "basil": 1}, "soup": {"tomato": 1}}
SUBSTITUTION = {"pizza": 4, "pasta": 2, "soup": 1}        # 1 pizza = 2 pasta = 4 soup
L = 11
PAY_WEEKS = [1, 3, 5, 7, 9, 11]
PROD_WEEKS = [2, 4, 6, 8, 10]
CLAUSES = ["substitution", "grim_trigger", "payment_deduction", "rollover"]

STORIES = {
    "catering": {"customer": "Customer", "supplier": "Caterer", "inputs": ["Mozzarella", "Basil", "Tomato"],
                 "outputs": ["Margherita Pizza", "Pesto Pasta", "Tomato Soup"], "unit": "$", "mult": 1},
    "hotel": {"customer": "Hotel owner", "supplier": "Cleaning provider", "inputs": ["Detergent", "Disinfectant", "Wet-wipes"],
              "outputs": ["Conference Room Cleaning", "King-Sized Room Cleaning", "Single Room Cleaning"], "unit": "$", "mult": 100},
    "ai": {"customer": "AI startup founder", "supplier": "AI service provider", "inputs": ["CPU", "Memory", "GPU"],
           "outputs": ["Fine-tuning", "Model-hosting", "Inference"], "unit": "$", "mult": 1000},
}


@dataclass(frozen=True)
class EnvParams:
    env_id: int
    stochasticity: Literal["none", "low", "high"]
    values: tuple[int, int, int]                         # customer private value per product
    capital: int                                         # supplier initial cash
    budget: int = 200
    minimums: tuple[int, int, int] = (1, 1, 1)
    # price laws per ingredient: list of (price, prob)
    price_laws: tuple[tuple[tuple[int, float], ...], ...] = (((1, 1.0),), ((1, 1.0),), ((2, 1.0),))
    receipt: Literal["exact", "low", "high"] = "exact"   # low: U{floor(3x/4)..x}; high: U{floor(x/2)..x}
    spoilage: Literal["none", "low", "high"] = "none"    # low: U{0,1}; high: U{0,1,2}
    inventory_cap: int = 10
    order_cap: int = 12
    rollover_cap: int = 2
    max_rounds: int = 50
    story: str = "catering"
    seed_note: str = ""

    def to_dict(self):
        return asdict(self)


_LOW_PRICES = (((1, 2 / 3), (3, 1 / 3)), ((1, 2 / 3), (2, 1 / 3)), ((2, 2 / 3), (3, 1 / 3)))
_HIGH_PRICES = (((1, 0.5), (3, 0.5)), ((1, 0.5), (2, 0.5)), ((2, 0.5), (3, 0.5)))


def env(env_id: int, story: str = "catering") -> EnvParams:
    table = {
        1: dict(stochasticity="none", values=(12, 6, 3), capital=20),
        2: dict(stochasticity="none", values=(20, 10, 5), capital=40),
        3: dict(stochasticity="low", values=(12, 6, 3), capital=20, price_laws=_LOW_PRICES, receipt="low", spoilage="low"),
        4: dict(stochasticity="low", values=(20, 10, 5), capital=40, price_laws=_LOW_PRICES, receipt="low", spoilage="low"),
        5: dict(stochasticity="high", values=(12, 6, 3), capital=20, price_laws=_HIGH_PRICES, receipt="high", spoilage="high"),
        6: dict(stochasticity="high", values=(20, 10, 5), capital=40, price_laws=_HIGH_PRICES, receipt="high", spoilage="high"),
    }[env_id]
    return EnvParams(env_id=env_id, story=story, **table)


def receipt_support(law: str, ordered: int) -> list[int]:
    if law == "exact" or ordered == 0:
        return [ordered]
    lo = (3 * ordered) // 4 if law == "low" else ordered // 2
    return list(range(lo, ordered + 1))


def spoilage_support(law: str) -> list[int]:
    return {"none": [0], "low": [0, 1], "high": [0, 1, 2]}[law]
