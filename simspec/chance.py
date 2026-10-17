"""Seeded, logged randomness. Every draw is named and recorded, so a run can be replayed
by reading draws back instead of re-sampling (replay-by-readback)."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


class ChanceExhausted(Exception):
    pass


class Chance:
    def __init__(self, seed: int, replay: list[dict] | None = None, sink: list[dict] | None = None):
        self.seed = seed
        self.gen = np.random.default_rng(seed)
        self._replay = list(replay) if replay is not None else None
        self._pos = 0
        self.sink = sink if sink is not None else []

    def draw(self, name: str, sampler: Callable[[np.random.Generator], Any]) -> Any:
        """Draw a JSON-serialisable value. In replay mode the logged value is returned instead."""
        if self._replay is not None:
            if self._pos >= len(self._replay):
                raise ChanceExhausted(f"replay log exhausted at draw {name!r}")
            rec = self._replay[self._pos]
            self._pos += 1
            if rec["name"] != name:
                raise ValueError(f"replay mismatch: expected draw {rec['name']!r}, got {name!r}")
            value = rec["value"]
        else:
            value = _jsonable(sampler(self.gen))
        self.sink.append({"name": name, "value": value})
        return value

    # convenience samplers
    def integers(self, name: str, lo: int, hi: int) -> int:
        """Uniform integer in lo..hi inclusive."""
        return int(self.draw(name, lambda g: int(g.integers(lo, hi + 1))))

    def uniform(self, name: str) -> float:
        return float(self.draw(name, lambda g: float(g.random())))

    def choice(self, name: str, items: list[Any], p: list[float] | None = None) -> Any:
        idx = self.draw(name, lambda g: int(g.choice(len(items), p=p)))
        return items[idx]

    def categorical(self, name: str, probs: dict[str, float]) -> str:
        keys = list(probs)
        p = np.array([probs[k] for k in keys], dtype=float)
        s = p.sum()
        if s <= 0:
            raise ValueError("categorical with zero mass")
        p = p / s
        # the drawn value is the key itself, so a replayed draw is valid even if the distribution differs
        return str(self.draw(name, lambda g: keys[int(g.choice(len(keys), p=p))]))

    def drain(self) -> list[dict]:
        out = list(self.sink)
        self.sink.clear()
        return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v
