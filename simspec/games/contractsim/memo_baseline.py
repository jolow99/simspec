"""memo-based rational baselines for ContractSim.

Both roles face the same abstract problem during performance: a finite-horizon trust game against a
counterparty of hidden type (complier / non-engager), with a binary choice each turn (comply with the
contract or not), an observation of whether the counterparty complied, and an optional intrinsic
violation cost. That problem is a small belief-space POMDP, which memo solves exactly by value
iteration over a discretised belief (the pattern of memo's own POMDP demo). The game-specific
numbers (what complying costs, what the counterparty's compliance is worth, the compliance
likelihoods under each type) are tables installed into the generated memo module.

The abstraction is declared in the Harness Card (`solver`) so the comparison with LLM agents is
explicit about what the rational agent is optimising.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from ...agents.base import BaseAgent, ActionDistribution, ActContext
from ...agents.memo_agent import build_memo_module
from ...card import ScaffoldSection
from ...spec import Observation, ActionSpec
from .params import EnvParams, PRODUCTS, INGREDIENTS, RECIPES, PAY_WEEKS, PROD_WEEKS, receipt_support, spoilage_support
from .contract import Contract, due_payment
from .baselines import FixedContractNegotiator, RCSupplierPolicy, contract_from_obs, heuristic_production

N_BELIEF = 21
MEMO_SRC = '''
@memo(install_module=m.install, cache=True)
def Q[b: B, d: D, a: A](k):
    me: knows(b, d, a)
    me: thinks[
        env: knows(b, d, a),
        env: chooses(ty in T, wpp=belief(b, ty)),
        env: chooses(d_ in D, wpp=d_next(d, a, d_)),
        env: chooses(o in O, wpp=p_comply(ty, d_, o, k)),
    ]
    me: snapshots_self_as(future)
    return me[ E[ r_now(a, k) + gain(env.o, k) ] + (0.0 if k >= {KMAX} else imagine[
        future: observes [env.d_] is env.d_,
        future: observes [env.o] is env.o,
        future: chooses(b_ in B, wpp=exp(-200.0 * abs(E[env.ty == 1] - b_))),
        future: chooses(a_ in A, to_maximize=Q[b_, env.d_, a_](k + 1)),
        E[ future[ Q[b_, env.d_, a_](k + 1) ] ]
    ]) ]
'''

HEADER = '''
B = jnp.linspace(0.0, 1.0, {nb})
D = jnp.arange(2)       # 0: I have complied so far, 1: I have defected (a complier counterparty then stops)
A = jnp.arange(2)       # 0: comply with the contract now, 1: defect now
T = jnp.arange(2)       # 0: counterparty never engages, 1: counterparty complies (conditionally)
O = jnp.arange(2)       # 0: counterparty did not comply this round, 1: complied
KMAX = {kmax}

@jax.jit
def belief(b, ty):
    return jnp.where(ty == 1, b, 1.0 - b)

@jax.jit
def d_next(d, a, d_):
    return 1.0 * (d_ == jnp.maximum(d, a))

@jax.jit
def p_comply(ty, d_, o, k):
    # P(o | type, my defection status, round k); P_TAB[ty, d_, k] = P(o == 1)
    p1 = P_TAB[ty, d_, k]
    return jnp.where(o == 1, p1, 1.0 - p1)

@jax.jit
def r_now(a, k):
    return R_TAB[a, k]

@jax.jit
def gain(o, k):
    return G_TAB[k] * (o == 1)
'''


def _nearest(b: float) -> int:
    return int(round(min(max(b, 0.0), 1.0) * (N_BELIEF - 1)))


class TrustGameTables:
    """Game-specific numbers for one role, built from the contract and environment."""

    def __init__(self, role: str, params: EnvParams, contract: Contract, violation_cost: float, p_comply_env: float):
        self.role = role
        self.params = params
        self.contract = contract
        self.violation_cost = violation_cost
        self.p_env = p_comply_env     # P(a complier actually meets its obligation this round) under env stochasticity
        if role == "customer":
            K = len(PAY_WEEKS)                                    # decisions at payment weeks 1..11
            self.K = K
            M = [contract.payments[w] for w in PAY_WEEKS]
            V = [sum(params.values[i] * contract.schedule[w][p] for i, p in enumerate(PRODUCTS)) for w in PROD_WEEKS] + [0.0]
            # r_now[a, k]: comply = pay M_k; defect = intrinsic violation cost (only if M_k > 0)
            self.R = np.array([[-float(M[k]) for k in range(K)], [-(violation_cost if M[k] > 0 else 0.0) for k in range(K)]])
            # gain[k]: value of the delivery that follows payment k (none after week 11)
            self.G = np.array([float(V[k]) for k in range(K)])
            # P(o=1 | ty, d', k): complier delivers iff customer has not defected; non-engager never
            P = np.zeros((2, 2, K))
            for k in range(K):
                if k < K - 1:
                    P[1, 0, k] = p_comply_env
            self.P = P
        else:
            K = len(PROD_WEEKS)                                   # decisions at production weeks 2..10
            self.K = K
            exp_price = [sum(p * q for p, q in law) for law in params.price_laws]
            cost = []
            for w in PROD_WEEKS:
                need = [0.0, 0.0, 0.0]
                for pi, p in enumerate(PRODUCTS):
                    for ing, n in RECIPES[p].items():
                        need[INGREDIENTS.index(ing)] += n * contract.schedule[w][p]
                cost.append(sum(n * c for n, c in zip(need, exp_price)))
            M_next = [contract.payments[w + 1] for w in PROD_WEEKS]   # payment that follows this production week
            self.R = np.array([[-float(cost[k]) for k in range(K)], [-float(violation_cost) for k in range(K)]])
            self.G = np.array([float(M_next[k]) for k in range(K)])
            P = np.zeros((2, 2, K))
            for k in range(K):
                P[1, 0, k] = 1.0 if M_next[k] > 0 else p_comply_env   # a payer pays what is due if the supplier has not breached
            self.P = P
        self.KMAX = self.K - 1

    def digest(self) -> str:
        return hashlib.sha256(self.R.tobytes() + self.G.tobytes() + self.P.tobytes()).hexdigest()[:12]


def solve(tables: TrustGameTables) -> np.ndarray:
    """Returns Q[k, b, d, a] via the generated memo program."""
    import jax.numpy as jnp
    name = f"contractsim_trust_{tables.role}_{tables.digest()}"
    header = HEADER.format(nb=N_BELIEF, kmax=tables.KMAX)
    m = build_memo_module(name, header, MEMO_SRC,
                          globals_={"R_TAB": jnp.asarray(tables.R), "G_TAB": jnp.asarray(tables.G), "P_TAB": jnp.asarray(tables.P)})
    Q = np.stack([np.asarray(m.Q(k)) for k in range(tables.K)])      # (K, B, D, A)
    return Q


class MemoTrustAgent(BaseAgent):
    """A memo-rational Customer or Supplier: Bayesian belief over the counterparty's type, exact
    finite-horizon planning over {comply, defect}, mapped to concrete contract actions."""

    def __init__(self, role: str, params: EnvParams, contract: Contract, negotiator: FixedContractNegotiator | None = None,
                 violation_cost: float = 0.0, beta: float = 5.0, prior_complier: float = 0.5, p_comply_env: float | None = None):
        self.role = role
        self.params = params
        self.contract = contract
        self.negotiator = negotiator
        self.violation_cost = violation_cost
        self.beta = beta
        self.prior = prior_complier
        if p_comply_env is None:
            p_comply_env = {"none": 1.0, "low": 0.97, "high": 0.9}[params.stochasticity] if role == "customer" else 1.0
        self.tables = TrustGameTables(role, params, contract, violation_cost, p_comply_env)
        self.Q = solve(self.tables)
        self.belief = prior_complier
        self.defected = 0
        self.k = 0
        self._supplier_policy = RCSupplierPolicy(params, 0.05, "rc")
        self._last_obs_week = None

    # -- belief update from public observations only ------------------------------------------
    def observe(self, obs: Observation, ctx: ActContext) -> None:
        if not obs["agreed"] or obs.phase == "negotiate":
            return
        self.defected = 1 if obs["violations_customer" if self.role == "customer" else "violations_supplier"] > 0 else self.defected
        week = obs["week"]
        if self.role == "customer" and obs["stage_done"] == "produce" and obs.phase == "produce":
            k = PROD_WEEKS.index(week - 1) if (week - 1) in PROD_WEEKS else None
            if k is None or self._last_obs_week == week:
                return
            self._last_obs_week = week
            met = all(obs["delivered_this_week"][i] >= self.contract.schedule[week - 1][p] for i, p in enumerate(PRODUCTS))
            self._update(met, k)
        if self.role == "supplier" and obs["stage_done"] == "payment" and obs.phase == "payment":
            pw = week if week in PAY_WEEKS else week - 1
            if pw not in PAY_WEEKS or self._last_obs_week == pw:
                return
            self._last_obs_week = pw
            due = max(self.contract.payments[pw] - obs["deduction_allowed"], 0)
            met = obs["paid_this_week"] >= due
            k = PROD_WEEKS.index(pw + 1) if (pw + 1) in PROD_WEEKS else self.tables.K - 1
            # a payment at week w is evidence about the "o" after production decision k-1 (or the initial payment)
            kk = max(k - 1, 0)
            self._update(met, kk, initial=(pw == 1))

    def _update(self, met: bool, k: int, initial: bool = False) -> None:
        P = self.tables.P
        if initial:
            # the first payment precedes any supplier decision: P(pay | payer) = 1 (if due), non-engager 0
            l1, l0 = (1.0 if met else 0.0), (0.0 if met else 1.0)
            due0 = self.contract.payments[1] > 0
            if not due0:
                return
        else:
            l1 = P[1, self.defected, k] if met else 1 - P[1, self.defected, k]
            l0 = P[0, self.defected, k] if met else 1 - P[0, self.defected, k]
        num = self.belief * l1
        den = num + (1 - self.belief) * l0
        self.belief = num / den if den > 0 else self.belief

    # -- act -----------------------------------------------------------------------------------
    def act(self, obs: Observation, spec: ActionSpec, ctx: ActContext) -> ActionDistribution:
        if obs.phase == "negotiate":
            v = (self.negotiator.customer(obs, spec) if self.role == "customer" else self.negotiator.supplier(obs, spec)) if self.negotiator else spec.default
            return ActionDistribution.point(spec.key_of(v), source="scripted")
        week = obs["week"]
        if self.role == "customer" and obs.phase == "payment":
            k = PAY_WEEKS.index(week)
            return self._decide(k, obs, spec, lambda a: {"amount": max(0, min(due_payment(self.contract, week, obs["deduction_allowed"]), obs["budget_remaining"])) if a == 0 else 0})
        if self.role == "supplier" and obs.phase == "order":
            k = PROD_WEEKS.index(week)
            dist = self._decide(k, obs, spec, lambda a: a)
            # commit to the sampled abstract action via the stored plan: order to fulfil, or nothing
            self._pending = dist
            return ActionDistribution(probs={spec.key_of(self._supplier_policy.order(obs, spec) if a == "0" else {"order": [0, 0, 0]}): p for a, p in dist.probs.items()},
                                      source="exact", meta=dist.meta)
        if self.role == "supplier" and obs.phase == "produce":
            # produce whatever the inventory supports toward the schedule (the order decision carried the choice)
            target = [self.contract.schedule[week][p] + obs["deficit"][i] for i, p in enumerate(PRODUCTS)]
            prod = heuristic_production(obs["inventory"], target)
            if sum(prod) == 0:
                self.defected = 1 if sum(target) > 0 else self.defected
            return ActionDistribution.point(spec.key_of({"produce": prod}), source="exact")
        return ActionDistribution.point(spec.key_of(spec.default), source="scripted")

    def _decide(self, k: int, obs: Observation, spec: ActionSpec, to_value) -> ActionDistribution:
        bi = _nearest(self.belief)
        q = self.Q[k, bi, self.defected]                        # (A,)
        z = self.beta * (q - q.max())
        p = np.exp(z)
        p = p / p.sum()
        probs = {}
        for a in range(2):
            key = spec.key_of(to_value(a)) if not callable(to_value) or True else None
            val = to_value(a)
            key = spec.key_of(val) if isinstance(val, dict) else str(val)
            probs[key] = probs.get(key, 0.0) + float(p[a])
        meta = {"k": k, "belief_complier": round(self.belief, 4), "defected": self.defected, "q": [float(x) for x in q]}
        dist = ActionDistribution(probs=probs, source="exact", meta=meta)
        # bookkeeping for the next step: if the sampled action is 'defect' the runtime's choice is not known here;
        # we track defection from public violation counts instead (see observe), so nothing to do.
        return dist

    def end_episode(self, payoffs):
        pass

    def describe(self, seat: str) -> ScaffoldSection:
        return ScaffoldSection(seat=seat, role=self.role, agent_class="memo:trust_pomdp", distribution_source="exact",
                               decoding={}, solver={"abstraction": "binary comply/defect per turn vs hidden counterparty type (complier | non-engager); "
                                                                    "belief-space value iteration via memo (demo-pomdp pattern)",
                                                     "belief_grid": N_BELIEF, "beta": self.beta, "prior_complier": self.prior,
                                                     "violation_cost": self.violation_cost, "p_comply_env": self.tables.p_env,
                                                     "tables_hash": self.tables.digest(), "memo_source_hash": hashlib.sha256(MEMO_SRC.encode()).hexdigest()[:16],
                                                     "negotiator": "fixed-contract" if self.negotiator else "none"})
