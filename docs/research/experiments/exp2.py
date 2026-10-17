"""Experiment 2: (a) generate memo source as a string and install it into a module
(needed if our spec layer compiles to memo); (b) exotic (array) parameters;
(c) recursion depth 50 cost; (d) joint-array scaling."""
import time, sys
import jax, jax.numpy as np
from memo import memo, domain, make_module

# (a)+(b): programmatic module with array-valued parameters
m = make_module('gen')
m.install('''
import jax, jax.numpy as np
S = np.arange(6)
A = np.arange(3)
@jax.jit
def R(s, a, Rtab):
    return Rtab[s, a]
@jax.jit
def T(s, a, s_, Ttab):
    return Ttab[s, a, s_]
''')
src = '''
@memo(install_module=m.install)
def Q[s: S, a: A](t, Rtab: ..., Ttab: ...):
    ag: knows(s, a)
    ag: given(s_ in S, wpp=T(s, a, s_, Ttab))
    ag: chooses(a_ in A, to_maximize=0.0 if t <= 0 else Q[s_, a_](t - 1, Rtab, Ttab))
    return E[R(s, a, Rtab) + (0.0 if t <= 0 else 0.9 * Q[ag.s_, ag.a_](t - 1, Rtab, Ttab))]
'''
# install_module path: memo() compiles the decorated function from source via inspect, so the
# function must live in a real file; write a temp module file and import it.
import pathlib, importlib.util
p = pathlib.Path('gen_q.py'); p.write_text("from memo import memo\nfrom __main__ import m\n" + src)
spec = importlib.util.spec_from_file_location('gen_q', p); mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
key = jax.random.PRNGKey(0)
Rtab = jax.random.normal(key, (6, 3))
Ttab = jax.nn.softmax(jax.random.normal(key, (6, 3, 6)), axis=-1)
print("NOTE: @memo(cache=True) + array ('exotic') params fails: functools.cache needs hashable args -> no memoized recursion with array params")
for t in [1, 3, 6, 9]:
    t0 = time.time(); q = m.Q(t, Rtab, Ttab).block_until_ready(); t1 = time.time()
    print(f"(a,b) generated memo w/ array params, uncached recursion t={t}: {t1-t0:.2f}s shape={q.shape}")
t0 = time.time(); q2 = m.Q(3, Rtab * 2, Ttab).block_until_ready(); t1 = time.time()
print(f"     call w/ new array values t=3 (no recompile expected): {t1-t0:.2f}s")
# globals + cache=True version, depth 50
m.install('''
Rg = Rtab_g; Tg = Ttab_g
@jax.jit
def Rg_(s, a): return Rg[s, a]
@jax.jit
def Tg_(s, a, s_): return Tg[s, a, s_]
''') if False else None
m.Rg = Rtab; m.Tg = Ttab
m.install('''
@jax.jit
def Rg_(s, a): return Rg[s, a]
@jax.jit
def Tg_(s, a, s_): return Tg[s, a, s_]
''')
p = pathlib.Path('gen_q2.py'); p.write_text('''from memo import memo
from __main__ import m
@memo(install_module=m.install, cache=True)
def Qg[s: S, a: A](t):
    ag: knows(s, a)
    ag: given(s_ in S, wpp=Tg_(s, a, s_))
    ag: chooses(a_ in A, to_maximize=0.0 if t <= 0 else Qg[s_, a_](t - 1))
    return E[Rg_(s, a) + (0.0 if t <= 0 else 0.9 * Qg[ag.s_, ag.a_](t - 1))]
''')
spec = importlib.util.spec_from_file_location('gen_q2', p); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
t0 = time.time(); qg = m.Qg(50).block_until_ready(); t1 = time.time()
print(f"(c) globals + cache=True, depth 50: {t1-t0:.2f}s; equals array-param t=9 prefix? n/a")
m.Qg.cache_clear(); t0 = time.time(); qg = m.Qg(50).block_until_ready(); t1 = time.time()
print(f"     rerun after cache_clear (JIT warm): {t1-t0:.2f}s")


# (d) scaling: 2 agents with nested frames; joint array size grows as product of axes
m2 = make_module('scale')
for n in [20, 50, 100, 200]:
    m2.install(f'''
import jax, jax.numpy as np
S = np.arange({n})
A = np.arange(5)
Ty = np.arange(4)
@jax.jit
def U(s, a, b, ty): return np.sin(s * 0.1 + a + b * ty)
''')
    p2 = pathlib.Path(f'gen_scale_{n}.py'); p2.write_text(f"""from memo import memo
from __main__ import m2
@memo(install_module=m2.install)
def pol[s: S, a: A]():
    p1: knows(s, a)
    p1: thinks[
        p2: chooses(ty in Ty, wpp=1),
        p2: knows(s),
        p2: thinks[ p1: chooses(a in A, wpp=1) ],
        p2: chooses(b in A, wpp=exp(E[U(s, p1.a, b, ty)])),
    ]
    return p1[E[U(s, a, p2.b, p2.ty)]]
""")
    spec = importlib.util.spec_from_file_location(f'gen_scale_{n}', p2); mod = importlib.util.module_from_spec(spec)
    t0 = time.time(); spec.loader.exec_module(mod); t1 = time.time()
    r = m2.pol().block_until_ready(); t2 = time.time()
    print(f"(d) |S|={n}: compile {t1-t0:.2f}s first-run {t2-t1:.2f}s; joint cells ~ {n*5*4*5}")
