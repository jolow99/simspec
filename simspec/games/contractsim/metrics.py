"""Offline metrics for ContractSim, computed from trajectory logs (Sajja et al. 2026, Appendix).

Negotiation/contract metrics need the contract's value under joint RCC play; the paper computes it by
exact DP, here it is estimated by simulating RCC x RCC episodes (`contract_value`). Performance
metrics follow the appendix definitions: utility, compliant utility, defection (any / unilateral /
reciprocal), compliance (overall / conditional / exploited / tit-for-tat), and ex-post regret vs a
re-simulated RCC counterfactual with the same seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...log import Trajectory
from ...runtime import run_episode
from .params import EnvParams, PAY_WEEKS, PROD_WEEKS, PRODUCTS
from .contract import Contract
from .baselines import PaperBaselineAgent, FixedContractNegotiator


def week_compliance(traj: Trajectory) -> dict[str, dict[int, bool]]:
    """Per-week compliance indicators c_{i,w} from transition events."""
    out = {"customer": {}, "supplier": {}}
    for e in traj.of("transition"):
        for ev in e.get("events", []):
            if ev.get("kind") == "payment" and ev.get("due") is not None:
                if ev["due"] > 0 or ev["paid"] > 0 or True:
                    out["customer"][ev["week"]] = not ev["violation"]
            if ev.get("kind") == "delivery" and ev.get("scheduled") is not None:
                out["supplier"][ev["week"]] = not ev["violation"]
    return out


def first_defector(comp: dict[str, dict[int, bool]]) -> tuple[str | None, int | None]:
    firsts = []
    for role, d in comp.items():
        for w, ok in sorted(d.items()):
            if not ok:
                firsts.append((w, role))
                break
    if not firsts:
        return None, None
    w, role = min(firsts)
    return role, w


def performance_metrics(traj: Trajectory, role: str) -> dict[str, Any]:
    comp = week_compliance(traj)
    mine = comp[role]
    other = comp["supplier" if role == "customer" else "customer"]
    fd_role, fd_week = first_defector(comp)
    weeks = sorted(mine)
    n = len(weeks)
    overall = float(np.mean([mine[w] for w in weeks])) if n else float("nan")
    defection = any(not mine[w] for w in weeks)
    # weeks before / after the counterparty's first violation
    other_first = min([w for w, ok in other.items() if not ok], default=None)
    before = [mine[w] for w in weeks if other_first is None or w < other_first]
    after = [mine[w] for w in weeks if other_first is not None and w > other_first]
    conditional = float(np.mean(before)) if before else float("nan")
    exploited = float(np.mean(after)) if after else float("nan")
    unilateral = any(not mine[w] for w in weeks if other_first is None or w < other_first)
    reciprocal = (other_first is not None) and any(not mine[w] for w in weeks if w > other_first)
    # tit-for-tat: my (non)compliance matches whether the counterparty had defected before this week
    tft = []
    for w in weeks:
        d = other_first is not None and other_first < w
        tft.append((not mine[w]) == d)
    payoffs = traj.payoffs() or {}
    return {"utility": payoffs.get(role), "compliance": overall, "conditional_compliance": conditional, "exploited_compliance": exploited,
            "tit_for_tat": float(np.mean(tft)) if tft else float("nan"), "defection": defection, "unilateral_defection": unilateral,
            "reciprocal_defection": reciprocal, "first_defector": fd_role, "first_defection_week": fd_week, "n_weeks": n}


def contract_value(params: EnvParams, contract: Contract, spec_factory, n: int = 20, seed0: int = 1000, eps: float = 0.05) -> dict[str, Any]:
    """U_i(omega) and P_sat under joint RCC play, estimated by simulation (the paper uses exact DP)."""
    neg = FixedContractNegotiator(contract)
    utils = {"customer": [], "supplier": []}
    sat = []
    for i in range(n):
        spec = spec_factory(params)
        ag = {"customer": PaperBaselineAgent("customer", params, "rcc", neg, eps), "supplier": PaperBaselineAgent("supplier", params, "rcc", neg, eps)}
        res = run_episode(spec, ag, seed=seed0 + i)
        s = res.final_state
        utils["customer"].append(res.payoffs["customer"])
        utils["supplier"].append(res.payoffs["supplier"])
        sat.append(s["violations_customer"] == 0 and s["violations_supplier"] == 0)
    U = {r: float(np.mean(v)) for r, v in utils.items()}
    gains = {"customer": U["customer"] - params.budget, "supplier": U["supplier"] - 0.0}
    return {"U": U, "gains": gains, "mutual_benefit": all(g >= -1e-9 for g in gains.values()), "p_sat": float(np.mean(sat)), "n": n}


def regret_vs_rcc(traj: Trajectory, role: str, params: EnvParams, contract: Contract, spec_factory, seed: int, eps: float = 0.05) -> float | None:
    """Ex-post regret: u_i(tau^RCC) - u_i(tau), re-simulating with the focal agent replaced by RCC and the
    counterparty replayed from the log where possible (same seed, same chance stream)."""
    from ...agents.base import ReplayAgent
    other = "supplier" if role == "customer" else "customer"
    neg = FixedContractNegotiator(contract)
    spec = spec_factory(params)
    ag = {role: PaperBaselineAgent(role, params, "rcc", neg, eps)}
    try:
        ag[other] = ReplayAgent(other, traj.actions())
        res = run_episode(spec, ag, seed=seed)
    except Exception:
        ag[other] = PaperBaselineAgent(other, params, "rcc", neg, eps)
        res = run_episode(spec, ag, seed=seed)
    mine = (traj.payoffs() or {}).get(role)
    if mine is None:
        return None
    return float(res.payoffs[role] - mine)


def summarize(trajs: list[Trajectory], role: str) -> dict[str, Any]:
    ms = [performance_metrics(t, role) for t in trajs]
    keys = ["utility", "compliance", "conditional_compliance", "exploited_compliance", "tit_for_tat"]
    out = {}
    for k in keys:
        vals = [m[k] for m in ms if m[k] is not None and not (isinstance(m[k], float) and np.isnan(m[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    for k in ["defection", "unilateral_defection", "reciprocal_defection"]:
        out[k] = float(np.mean([m[k] for m in ms]))
    out["n"] = len(ms)
    return out
