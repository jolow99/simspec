"""Agents for the norms gridworld.

NormPlanner: a model-based agent (the paper's RTDP planner, simplified to BFS-distance values) that
complies with the rules it has adopted: prohibitions mask actions whose predicted effect violates
them, obligations switch the goal to the obligation's discharge condition.

NormLearner: NormPlanner plus Bayesian rule induction over a finite rule space from the observed
actions of the other agents (mean-field, bounded-compliance likelihood built from the planner's
Q-values with and without each rule; thresholding at 0.95), with an exact memo implementation of
the per-rule posterior update (`MemoRulePosterior`).

Both read only their Observation (everything in this game is public).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from ...agents.base import BaseAgent, ActionDistribution, ActContext
from ...card import ScaffoldSection
from ...rules import Rule
from ...spec import Observation, ActionSpec
from .world import Layout, ACTIONS, ACTION_CLASS, LOOK, apply_action, bfs_distances, beam_cells, dirt_fraction
from .spec import features_of, world_of, agent_names

GAMMA = 0.9
VIOLATION_COST = 1.0
OBLIGATION_REWARD = 1.0
OBLIGATION_DEADLINE = 20
P_OBEY = 0.9
TAU = 1.0
INIT_PRIOR = 0.05
THRESHOLD = 0.95


def _goal_of(rule: Rule) -> str | None:
    if rule.kind != "obligation" or rule.post is None:
        return None
    f = getattr(rule.post, "feature", "")
    if "CLEANED" in f:
        return "clean"
    if "PAID" in f:
        return "pay"
    return None


class Planner:
    """Q-values for agent i in world w under a set of rules, from BFS distances and a one-step forward model."""

    def __init__(self, layout: Layout, names: list[str]):
        self.layout = layout
        self.names = names

    def goal_cells(self, w: dict[str, Any], i: int, goal: str) -> set[int]:
        L = self.layout
        if goal == "apple":
            return {c for c in range(L.n) if w["apple"][c]}
        if goal == "clean":
            # cells from which a dirt cell is in beam range (any orientation): approximate by cells within 3 of a dirt cell, walkable
            dirt = [c for c in range(L.n) if w["dirt"][c] == "dirt"]
            out = set()
            for d in dirt:
                x, y = L.xy(d)
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if (dx == 0) != (dy == 0) and abs(dx) + abs(dy) <= 3:
                            c = L.cell(x + dx, y + dy)
                            if c is not None and L.walkable[c]:
                                out.add(c)
            return out
        if goal == "pay":
            out = set()
            for j, lk in enumerate(w["look"]):
                if j != i and lk == LOOK["cleaner"]:
                    x, y = L.xy(w["pos"][j])
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if (dx == 0) != (dy == 0) and abs(dx) + abs(dy) <= 3:
                                c = L.cell(x + dx, y + dy)
                                if c is not None and L.walkable[c]:
                                    out.add(c)
            return out
        return set()

    def successors(self, w: dict[str, Any], i: int) -> list[tuple[dict[str, Any], float, dict[str, Any]]]:
        """(next world, immediate reward, next features) for each action; computed once per (world, agent)."""
        key = (id(w), i)
        if getattr(self, "_succ_key", None) == key:
            return self._succ
        out = []
        for a in ACTIONS:
            w2, r = apply_action(self.layout, w, i, a)
            out.append((w2, r, features_of(self.layout, w2, i)))
        self._succ_key, self._succ = key, out
        return out

    def prohibited_by(self, w: dict[str, Any], i: int, rules: list[Rule]) -> list[bool]:
        succ = self.successors(w, i)
        flags = [False] * len(ACTIONS)
        for ai, a in enumerate(ACTIONS):
            ac = ACTION_CLASS[a]
            for rule in rules:
                if rule.kind == "prohibition" and rule.action_class == ac and rule.post is not None:
                    try:
                        if rule.post.eval(succ[ai][2], ac):
                            flags[ai] = True
                            break
                    except KeyError:
                        pass
        return flags

    def q_base(self, w: dict[str, Any], i: int, goal: str) -> np.ndarray:
        """Q over actions for `goal` with no rules (cached per (world, agent, goal))."""
        key = (id(w), i, goal)
        cache = getattr(self, "_qcache", None)
        if cache is None or cache[0] != id(w):
            self._qcache = (id(w), {})
            cache = self._qcache
        if key in cache[1]:
            return cache[1][key]
        L = self.layout
        goals = self.goal_cells(w, i, goal)
        q = np.zeros(len(ACTIONS))
        if w["frozen"][i] > 0:
            cache[1][key] = q
            return q
        pos = w["pos"][i]
        dist = bfs_distances(L, w, pos)
        for ai, (w2, r, _) in enumerate(self.successors(w, i)):
            a = ACTIONS[ai]
            npos = w2["pos"][i]
            dn = bfs_distances(L, w2, npos) if npos != pos else dist
            d = min((dn[c] for c in goals), default=10 ** 6)
            if goal == "apple":
                v = APPLE_VAL * (1.0 if w2["harvested"][i] else 0.0)
                v += (GAMMA ** d) * APPLE_VAL if d < 10 ** 6 else 0.0
                if w2["inventory"][i] > 0:
                    v += APPLE_VAL * (1.0 if a == "EAT" else GAMMA)
            else:
                done = (goal == "clean" and w2["cleaned"][i]) or (goal == "pay" and w2["paid"][i])
                if done:
                    v = OBLIGATION_REWARD
                else:
                    v = (GAMMA ** (d + 1)) * OBLIGATION_REWARD if d < 10 ** 6 else 0.0
                    if d == 0:
                        bc = beam_cells(L, npos, w2["orient"][i])
                        hit = any(w2["dirt"][c] == "dirt" for c in bc) if goal == "clean" else any(w2["pos"][j] in bc and w2["look"][j] == LOOK["cleaner"] for j in range(len(w2["pos"])) if j != i)
                        if hit:
                            v = GAMMA * OBLIGATION_REWARD
            q[ai] = r + v
        cache[1][key] = q
        return q

    def q_values(self, w: dict[str, Any], i: int, goal: str, rules: list[Rule]) -> tuple[np.ndarray, list[bool]]:
        """Q under `rules` = base Q minus the violation cost on prohibited actions."""
        q = self.q_base(w, i, goal).copy()
        prohibited = self.prohibited_by(w, i, rules)
        for ai in range(len(ACTIONS)):
            if prohibited[ai]:
                q[ai] -= VIOLATION_COST
        return q, prohibited


APPLE_VAL = 1.0


def boltzmann(q: np.ndarray, tau: float = TAU) -> np.ndarray:
    z = (q - q.mean()) / (q.std() + 1e-9)
    e = np.exp(z / tau)
    return e / e.sum()


class NormPlanner(BaseAgent):
    def __init__(self, role: str, layout: Layout, names: list[str], rules: list[Rule], seed: int = 0):
        self.role = role
        self.i = names.index(role)
        self.layout = layout
        self.names = names
        self.rules = [r for r in rules if role in r.scope]
        self.active: list[tuple[str, int]] = []           # (rule id, since step)
        self.planner = Planner(layout, names)
        self.rng = np.random.default_rng(seed)
        self.history_feats: list[dict[str, Any]] = []

    # obligation bookkeeping (the agent's own view of its obligations)
    def _update_obligations(self, feats: dict[str, Any], step: int) -> None:
        for r in self.rules:
            if r.kind != "obligation":
                continue
            act = next((a for a in self.active if a[0] == r.id), None)
            try:
                pre = r.pre.eval(feats) if r.pre else False
                post = r.post.eval(feats) if r.post else False
            except KeyError:
                pre, post = False, False
            if act is None and pre:
                self.active.append((r.id, step))
            elif act is not None and (post or step - act[1] > (r.deadline or OBLIGATION_DEADLINE)):
                self.active.remove(act)

    def current_goal(self) -> str:
        for rid, _ in self.active:
            r = next(x for x in self.rules if x.id == rid)
            g = _goal_of(r)
            if g:
                return g
        return "apple"

    def act(self, obs: Observation, spec: ActionSpec, ctx: ActContext) -> ActionDistribution:
        w = world_of(obs)
        feats = features_of(self.layout, w, self.i)
        self._update_obligations(feats, ctx.step)
        goal = self.current_goal()
        q, prohibited = self.planner.q_values(w, self.i, goal, self.rules)
        allowed = [k for k in range(len(ACTIONS)) if not prohibited[k]] or list(range(len(ACTIONS)))
        best = max(q[k] for k in allowed)
        ties = [k for k in allowed if abs(q[k] - best) < 1e-9]
        k = int(self.rng.choice(ties))
        meta = {"goal": goal, "q": [round(float(x), 4) for x in q], "prohibited": [ACTIONS[j] for j in range(len(ACTIONS)) if prohibited[j]],
                "active_obligations": [a[0] for a in self.active]}
        return ActionDistribution.point(ACTIONS[k], source="exact", **meta)

    def describe(self, seat: str) -> ScaffoldSection:
        return ScaffoldSection(seat=seat, role=self.role, agent_class="norms:planner", distribution_source="exact",
                               solver={"planner": "one-step forward model + BFS distance values (RTDP in the paper)", "gamma": GAMMA,
                                       "violation_cost": VIOLATION_COST, "rules": [r.id for r in self.rules]})


# ----------------------------------------------------------------------------- rule induction

MEMO_POSTERIOR = '''
@memo(install_module=m.install)
def posterior[r: R, a: A](prior_tab: ..., lik_tab: ...):
    observer: knows(r, a)
    observer: thinks[
        agent: knows(r),
        agent: chooses(h in H, wpp=prior_fn(r, h, prior_tab)),
        agent: chooses(a_ in A, wpp=lik_fn(r, h, a_, lik_tab)),
    ]
    observer: observes [agent.a_] is a
    return observer[Pr[agent.h == 1]]
'''

MEMO_HEADER = '''
R = jnp.arange({nr})
A = jnp.arange({na})
H = jnp.arange(2)

@jax.jit
def prior_fn(r, h, prior_tab):
    return jnp.where(h == 1, prior_tab[r], 1.0 - prior_tab[r])

@jax.jit
def lik_fn(r, h, a, lik_tab):
    return lik_tab[r, h, a]
'''


class MemoRulePosterior:
    """Exact per-rule Bayes update written in memo: for every rule r, an observer models the acting
    agent as either following r (h=1) or not (h=0), with the given action likelihoods, and
    conditions on the observed action. Vectorised over rules and actions."""

    def __init__(self, n_rules: int, n_actions: int):
        from ...agents.memo_agent import build_memo_module
        self.n_rules, self.n_actions = n_rules, n_actions
        self.m = build_memo_module(f"norm_posterior_{n_rules}_{n_actions}", MEMO_HEADER.format(nr=n_rules, na=n_actions), MEMO_POSTERIOR)

    def update(self, prior: np.ndarray, lik: np.ndarray, observed_action: int) -> np.ndarray:
        import jax.numpy as jnp
        post = np.asarray(self.m.posterior(jnp.asarray(prior, dtype=jnp.float32), jnp.asarray(lik, dtype=jnp.float32)))   # (R, A)
        return post[:, observed_action]


class NormLearner(NormPlanner):
    def __init__(self, role: str, layout: Layout, names: list[str], space: list[Rule], seed: int = 0,
                 prior: float = INIT_PRIOR, p_obey: float = P_OBEY, threshold: float = THRESHOLD, use_memo: bool = True,
                 known_rules: list[Rule] | None = None):
        super().__init__(role, layout, names, [], seed)
        self.space = list(space)
        self.beliefs = np.full(len(space), prior)
        self.p_obey = p_obey
        self.threshold = threshold
        self.prev_world: dict[str, Any] | None = None
        self.prev_feats: dict[int, dict[str, Any]] = {}
        self.use_memo = use_memo
        self.memo_post = MemoRulePosterior(len(space), len(ACTIONS)) if use_memo and space else None
        self.belief_history: list[list[float]] = []
        self.other_active: dict[tuple[int, str], int] = {}
        if known_rules:
            for r in known_rules:
                for k, s in enumerate(self.space):
                    if s.signature() == r.signature():
                        self.beliefs[k] = threshold
        self._refresh_adopted()

    def _refresh_adopted(self) -> None:
        self.rules = [r for r, b in zip(self.space, self.beliefs) if b >= self.threshold and self.role in r.scope]

    def observe(self, obs: Observation, ctx: ActContext) -> None:
        w_now = world_of(obs)
        joint = ctx.extra.get("joint", {})
        if self.prev_world is not None and joint:
            self._learn(self.prev_world, w_now, joint, ctx)
        self.prev_world = w_now
        self.belief_history.append([round(float(b), 4) for b in self.beliefs])
        if ctx.log_info is not None:
            adopted = [self.space[k].id for k in range(len(self.space)) if self.beliefs[k] >= self.threshold]
            ctx.log_info(f"beliefs:{self.role}", {"step": ctx.step, "beliefs": self.belief_history[-1], "adopted": adopted})

    def _learn(self, w_prev: dict[str, Any], w_now: dict[str, Any], joint: dict[str, str], ctx: ActContext) -> None:
        for j, name in enumerate(self.names):
            if name == self.role or name not in joint:
                continue
            a_obs = ACTIONS.index(joint[name])
            feats_prev = features_of(self.layout, w_prev, j)
            feats_now = features_of(self.layout, w_now, j)
            q0, _ = self.planner.q_values(w_prev, j, "apple", [])
            p0 = boltzmann(q0)
            lik = np.zeros((len(self.space), 2, len(ACTIONS)))
            informative = np.zeros(len(self.space), bool)
            for k, rule in enumerate(self.space):
                if name not in rule.scope:
                    continue
                if rule.kind == "prohibition":
                    q1, proh = self.planner.q_values(w_prev, j, "apple", [rule])
                    if not any(proh):
                        continue
                    p1 = boltzmann(q1)
                    p1 = p1 * np.array([0.0 if proh[x] else 1.0 for x in range(len(ACTIONS))])
                    if p1.sum() > 0:
                        p1 = p1 / p1.sum()
                    informative[k] = True
                    lik[k, 1] = self.p_obey * p1 + (1 - self.p_obey) * p0
                    lik[k, 0] = p0
                else:
                    key = (j, rule.id)
                    try:
                        pre = rule.pre.eval(feats_prev) if rule.pre else False
                        post_now = rule.post.eval(feats_now) if rule.post else False
                    except KeyError:
                        continue
                    if pre and key not in self.other_active:
                        self.other_active[key] = ctx.step
                    active = key in self.other_active and ctx.step - self.other_active[key] <= (rule.deadline or OBLIGATION_DEADLINE)
                    if key in self.other_active and (post_now or not active):
                        self.other_active.pop(key, None)
                    if not active:
                        continue
                    goal = _goal_of(rule)
                    if goal is None:
                        continue
                    q1, _ = self.planner.q_values(w_prev, j, goal, [])
                    # evidence only when the goal is achievable in one step (paper: could_be_satisfied)
                    satisfiable = any(s2["cleaned" if goal == "clean" else "paid"][j] for s2, _, _ in self.planner.successors(w_prev, j))
                    if not satisfiable:
                        continue
                    p1 = boltzmann(q1)
                    informative[k] = True
                    lik[k, 1] = self.p_obey * p1 + (1 - self.p_obey) * p0
                    lik[k, 0] = p0
            if not informative.any():
                continue
            if self.memo_post is not None:
                lik_safe = np.where(informative[:, None, None], lik, np.ones_like(lik) / len(ACTIONS))
                post = self.memo_post.update(self.beliefs, lik_safe, a_obs)
                self.beliefs = np.where(informative, post, self.beliefs)
            else:
                for k in np.where(informative)[0]:
                    pr = self.beliefs[k]
                    num = lik[k, 1, a_obs] * pr
                    den = num + lik[k, 0, a_obs] * (1 - pr)
                    if den > 0:
                        self.beliefs[k] = num / den
        self._refresh_adopted()

    def get_state(self) -> dict[str, Any]:
        return {"beliefs": [float(b) for b in self.beliefs]}

    def describe(self, seat: str) -> ScaffoldSection:
        sec = super().describe(seat)
        return sec.model_copy(update={"agent_class": "norms:bayesian_learner" + (":memo" if self.use_memo else ":numpy"),
                                      "solver": {**sec.solver, "rule_space": len(self.space), "prior": float(self.beliefs.min()) if len(self.beliefs) else None,
                                                 "p_obey": self.p_obey, "threshold": self.threshold, "update": "memo exact two-hypothesis posterior" if self.use_memo else "numpy",
                                                 "memo_source_hash": hashlib.sha256(MEMO_POSTERIOR.encode()).hexdigest()[:16]}})
