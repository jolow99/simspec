"""Observation renderer and scaffold for an LLM agent in the norms gridworld."""
from __future__ import annotations

from ...spec import Observation
from ...agents.llm import ScaffoldConfig
from .world import Layout, ACTIONS, ROLES, PAPER_MAP
from .spec import features_of, world_of, FEATURE_NAMES

SYSTEM = ("You are {role}, an agent in a shared gridworld with apples, a polluted river and territories. Apples regrow faster when "
          "neighbouring apples are present and when the river is clean. You gain 1 point for each apple you eat (walk onto an apple to pick it, "
          "then EAT), lose 0.01 per action, and lose 15 if zapped. Actions: " + ", ".join(ACTIONS) + ". FIRE_CLEAN cleans dirt in the 3 cells "
          "ahead of you; PAY gives one apple to a cleaner in the 3 cells ahead; FIRE_CLAIM claims the cell ahead as your territory. "
          "Other agents may follow social norms; you may infer them from behaviour. Answer with JSON only.")


def make_renderer(layout: Layout | None = None):
    def render(obs: Observation) -> str:
        w = world_of(obs)
        i = _index_of(obs.role)
        f = {k: obs[k] for k in FEATURE_NAMES}
        n_apples = sum(1 for a in w["apple"] if a)
        lines = [f"Step {obs.step}. You are {obs.role} (look: {ROLES[f['AGENT_LOOK']]}). Position {f['POSITION']}, facing {['N','E','S','W'][f['ORIENTATION']]}.",
                 f"Inventory {f['INVENTORY']} apples. Apples on the map: {n_apples}. Apples around you: {f['NUM_APPLES_AROUND']}. "
                 f"River dirt fraction: {f['DIRT_FRACTION']:.2f}. Steps since you last cleaned: {f['SINCE_AGENT_LAST_CLEANED']}, since you last paid: {f['SINCE_AGENT_LAST_PAID']}.",
                 f"Your current cell has an apple: {f['CUR_CELL_HAS_APPLE']}; it is foreign property: {f['CUR_CELL_IS_FOREIGN_PROPERTY']}.",
                 "Other agents (index: look, position, inventory): " + "; ".join(f"{j}: {ROLES[lk]}, {p}, {inv}" for j, (lk, p, inv) in enumerate(zip(w['look'], w['pos'], w['inventory'])) if j != i)]
        return "\n".join(lines)
    return render


def _index_of(role: str) -> int:
    digits = "".join(ch for ch in role if ch.isdigit())
    return int(digits) if digits else 0


def scaffold(role: str, model: str, provider: str = "openai", **overrides) -> ScaffoldConfig:
    base = dict(model=model, provider=provider, elicitation="json", history_k=5, include_messages=False,
                decoding={"temperature": 0.0, "max_tokens": 64}, max_retries=1, on_parse_failure="default",
                system_template=SYSTEM, turn_template="{observation}\n\n{history}{call_to_action}\n{menu}\n{format}", renderer="norms.v1")
    base.update(overrides)
    return ScaffoldConfig(**base)
