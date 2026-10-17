"""Offline metrics for the norms game: per-rule posterior trajectories, precision/recall of adopted
rules at checkpoints, collective cumulative reward, desiccated-field ratio, violation counts."""
from __future__ import annotations

from typing import Any

import numpy as np

from ...log import Trajectory
from ...rules import Rule


def belief_trajectory(traj: Trajectory, role: str) -> list[dict[str, Any]]:
    return [e["data"] for e in traj.of("info") if e["source"] == f"beliefs:{role}"]


def precision_recall(adopted: list[str], true_ids: list[str]) -> tuple[float, float]:
    a, t = set(adopted), set(true_ids)
    tp = len(a & t)
    prec = tp / len(a) if a else 0.0
    rec = tp / len(t) if t else 0.0
    return prec, rec


def learning_summary(traj: Trajectory, role: str, space: list[Rule], true_rules: list[Rule], fractions=(0.25, 0.5, 0.75, 1.0)) -> dict[str, Any]:
    """Precision/recall of adopted rules at fractions of the episode, matching true rules by description."""
    bt = belief_trajectory(traj, role)
    if not bt:
        return {}
    desc = {r.id: r.signature() for r in space}
    true_desc = {r.signature() for r in true_rules}
    true_ids = [rid for rid, d in desc.items() if d in true_desc]
    out = {}
    for f in fractions:
        rec = bt[min(int(len(bt) * f) - 1, len(bt) - 1)]
        p, r = precision_recall(rec["adopted"], true_ids)
        out[f] = {"precision": p, "recall": r, "adopted": rec["adopted"]}
    final = bt[-1]
    out["final_beliefs_true_rules"] = {rid: final["beliefs"][[s.id for s in space].index(rid)] for rid in true_ids}
    return out


def social_outcomes(traj: Trajectory) -> dict[str, Any]:
    tr = list(traj.of("transition"))
    cum = 0.0
    series = []
    des = []
    for e in tr:
        cum += sum(e.get("rewards", {}).values())
        series.append(cum)
        ev = e.get("events", [{}])[0]
        des.append(ev.get("desiccated_ratio", 0.0))
    viol = {}
    for e in traj.rule_events():
        if e["status"] == "violated":
            viol[e["rule_id"]] = viol.get(e["rule_id"], 0) + 1
    return {"cumulative_reward": series[-1] if series else 0.0, "reward_series": series, "desiccated_final": des[-1] if des else 0.0,
            "desiccated_series": des, "violations": viol, "steps": len(tr)}
