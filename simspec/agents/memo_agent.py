"""memo-based rational agents.

memo computes tables over finite axes; the runtime indexes them. The shared machinery here is:
- ``build_memo_module``: install Python globals (domains, tables, jitted functions) and memo source
  into a fresh module (memo needs real source on disk, so the source is written to a cache dir);
- ``TablePolicyAgent``: an agent whose policy is a (information-state index -> distribution) table,
  with an encoder that reads only the Observation (so it cannot see hidden state by construction);
- ``InfoStateEncoder``: builds a Product domain from named observation features and encodes an
  Observation into the memo axis index.
Game-specific memo models (the actual `@memo` programs) live with each game.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..domains import Product, Domain
from ..spec import Observation, ActionSpec
from ..card import ScaffoldSection, sha
from .base import BaseAgent, ActionDistribution, ActContext


_CACHE_DIR = os.environ.get("SIMSPEC_MEMO_CACHE", os.path.join(tempfile.gettempdir(), "simspec_memo"))


def build_memo_module(name: str, header: str, memo_source: str, globals_: dict[str, Any] | None = None):
    """Create a module containing `header` (plain Python: domains, tables, @jax.jit functions) and the
    memo models in `memo_source` (which must use `@memo(install_module=m.install, ...)` on functions
    whose bodies are memo programs). Returns the memo module `m` with the compiled memos as attributes.

    `globals_` are injected as module globals before the header runs (e.g. JAX tables built in Python)."""
    from memo import make_module
    m = make_module(name)
    for k, v in (globals_ or {}).items():
        setattr(m, k, v)
    m.install("import jax\nimport jax.numpy as jnp\nfrom enum import IntEnum\nfrom memo import domain\n" + header)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    digest = hashlib.sha256((header + memo_source).encode()).hexdigest()[:12]
    path = os.path.join(_CACHE_DIR, f"{name}_{digest}.py")
    src = "from memo import memo\nimport jax\nimport jax.numpy as jnp\n" + memo_source
    with open(path, "w") as fh:
        fh.write(src)
    modname = f"simspec_memo_{name}_{digest}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    mod.m = m                       # the memo source refers to `m.install`
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return m


@dataclass
class InfoStateEncoder:
    """Encodes an Observation into an index of a Product domain built from named features."""
    product: Product
    extract: Callable[[Observation], dict[str, Any]]

    def encode(self, obs: Observation) -> int:
        return self.product.encode(self.extract(obs))

    def describe(self) -> dict[str, Any]:
        from ..spec import _fn_hash
        return {"product": self.product.model_dump(), "extract": _fn_hash(self.extract)}


class TablePolicyAgent(BaseAgent):
    """Policy = table[info_state_index] -> probabilities over action keys. The encoder sees only the
    Observation, so hidden state cannot leak in. `source` is 'exact' for memo/DP tables."""

    def __init__(self, role: str, encoder: InfoStateEncoder, table: np.ndarray, keys: list[str],
                 name: str = "memo", source: str = "exact", solver_info: dict[str, Any] | None = None,
                 text_policy: Callable[[Observation, ActionSpec], str] | None = None):
        self.role = role
        self.encoder = encoder
        self.table = np.asarray(table, dtype=float)
        self.keys = list(keys)
        self.name = name
        self.source = source
        self.solver_info = solver_info or {}
        self.text_policy = text_policy
        assert self.table.shape == (encoder.product.size(), len(keys)), (self.table.shape, encoder.product.size(), len(keys))

    def act(self, obs: Observation, spec: ActionSpec, ctx: ActContext) -> ActionDistribution:
        if spec.kind == "text":
            txt = self.text_policy(obs, spec) if self.text_policy else ""
            return ActionDistribution.of_text(txt, source="scripted")
        i = self.encoder.encode(obs)
        row = self.table[i]
        probs = {k: float(p) for k, p in zip(self.keys, row) if p > 0}
        if obs.legal is not None:
            probs = {k: p for k, p in probs.items() if k in obs.legal} or {k: 1.0 for k in obs.legal}
        return ActionDistribution(probs=probs, source=self.source, meta={"info_state": i}).normalized()

    def describe(self, seat: str) -> ScaffoldSection:
        return ScaffoldSection(seat=seat, role=self.role, agent_class=f"table:{self.name}", distribution_source=self.source,
                               solver={**self.solver_info, "encoder": self.encoder.describe(),
                                       "table_hash": hashlib.sha256(self.table.tobytes()).hexdigest()[:16], "keys": self.keys})


def softmax_rows(q: np.ndarray, beta: float) -> np.ndarray:
    z = beta * (q - q.max(axis=1, keepdims=True))
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
