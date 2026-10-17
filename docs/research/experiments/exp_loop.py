"""Experiment: use memo as a one-step policy inside a Python game loop.
Game: 2-player repeated 'sequential trust' style game over a finite typed state,
with hidden type for player 2. Player 1 (memo rational) picks action given belief
over p2 type; the Python loop samples actions, updates state, and feeds the
*observation history* back into memo as an axis value (history encoded as int).
"""
import time, sys
t0 = time.time()
import jax, jax.numpy as np
from memo import memo, domain
from enum import IntEnum

class A(IntEnum): COOP = 0; DEFECT = 1
class Ty(IntEnum): NICE = 0; MEAN = 1
# History of up to 3 rounds of (a1, a2) pairs; encode "unplayed" as extra value 2.
H = domain(a1_1=3, a2_1=3, a1_2=3, a2_2=3, a1_3=3, a2_3=3)
N_ROUNDS = 3

@jax.jit
def payoff1(a1, a2):
    return np.array([[3., 0.], [5., 1.]])[a1, a2]
@jax.jit
def payoff2(a1, a2):
    return np.array([[3., 5.], [0., 1.]])[a1, a2]
@jax.jit
def round_of(h):
    # number of completed rounds = count of filled slots / 2
    t = H._tuple(h)
    return sum([(t[2*i] != 2) * 1 for i in range(N_ROUNDS)])
@jax.jit
def step(h, a1, a2, h_):
    t = np.array(H._tuple(h))
    r = round_of(h)
    t = t.at[2*r].set(a1).at[2*r+1].set(a2)
    return h_ == H(*t)
@jax.jit
def p2_wpp(ty, a2, a1):
    # NICE copies p1's current action; MEAN always defects
    return np.where(ty == Ty.NICE, 1.0 * (a2 == a1) + 0.05, 1.0 * (a2 == A.DEFECT) + 0.05)
@jax.jit
def hist_lik(ty, h):
    t = np.array(H._tuple(h))
    lik = 1.0
    for i in range(N_ROUNDS):
        a1p, a2p = t[2*i], t[2*i+1]
        filled = (a2p != 2)
        lik = lik * np.where(filled, p2_wpp(ty, np.minimum(a2p, 1), np.minimum(a1p, 1)) / (1.0 + 0.1), 1.0)
    return lik

@jax.jit
def valid(h):
    # history must be a prefix: no gaps
    t = np.array(H._tuple(h))
    filled = (t != 2)
    ok = True
    for i in range(2*N_ROUNDS - 1):
        ok = ok & ~(filled[i+1] & ~filled[i])
    return ok * 1.0

@memo(cache=True)
def Q1[h: H, a1: A](t):
    p1: knows(h, a1)
    p1: thinks[
        p2: chooses(ty in Ty, wpp=1),
        p2: knows(h, a1),
    ]
    # p1 conditions type on history via likelihood of past p2 actions
    p1: observes_event(wpp=hist_lik(p2.ty, h))
    p1: thinks[ p2: chooses(a2 in A, wpp=p2_wpp(ty, a2, a1)) ]
    p1: thinks[ p2: chooses(h_ in H, wpp=step(h, a1, a2, h_)) ]
    p1: thinks[ p2: chooses(a1_ in A, wpp=exp(2.0 * Q1[h_, a1_](t - 1)) if t > 0 else 1) ]
    return p1[E[payoff1(a1, p2.a2) + (Q1[p2.h_, p2.a1_](t - 1) if t > 0 else 0.0)]]

t1 = time.time()
q = Q1(N_ROUNDS - 1).block_until_ready()
t2 = time.time()
print(f"import {t1-t0:.1f}s, compile+run Q1 over |H|={len(H)} x |A|={len(A)}: {t2-t1:.1f}s, shape {q.shape}", file=sys.stderr)

# Python game loop: sample from softmax policy, scripted p2.
import numpy as onp
rng = onp.random.default_rng(0)
h = H(2,2,2,2,2,2)
for r in range(N_ROUNDS):
    logits = 2.0 * q[h]
    p = onp.exp(logits - logits.max()); p /= p.sum()
    a1 = rng.choice(2, p=p)
    a2 = int(A.COOP) if a1 == A.COOP else int(A.DEFECT)
    print(f"round {r}: p1 policy {p.round(3)} -> a1={A(a1).name}, a2={A(a2).name}")
    t = list(H._tuple(h)); t[2*r] = a1; t[2*r+1] = a2; h = H(*t)
print("done in", f"{time.time()-t0:.1f}s")
