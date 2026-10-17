"""Finite, typed, enumerable domains.

Every non-text value in a spec lives in a Domain. Domains know how to encode values to integer
indices and back, and how to emit memo source for themselves, so a memo agent can be generated
from a spec without hand-translation.
"""
from __future__ import annotations

from typing import Any, Literal, Union
import math

from pydantic import BaseModel, Field as PField, model_validator


class _DomainBase(BaseModel):
    model_config = {"frozen": True}
    name: str

    def size(self) -> int:
        raise NotImplementedError

    def values(self) -> list[Any]:
        return [self.decode(i) for i in range(self.size())]

    def encode(self, v: Any) -> int:
        raise NotImplementedError

    def decode(self, i: int) -> Any:
        raise NotImplementedError

    def contains(self, v: Any) -> bool:
        try:
            i = self.encode(v)
        except (KeyError, ValueError, TypeError):
            return False
        return 0 <= i < self.size()

    def to_memo(self, var: str | None = None) -> str:
        """Python source defining this domain for a generated memo module."""
        raise NotImplementedError

    def __len__(self) -> int:  # so `len(D)` works like memo domains
        return self.size()


class Enum(_DomainBase):
    """A finite set of labelled values; index = position."""
    kind: Literal["enum"] = "enum"
    labels: list[str]

    @model_validator(mode="after")
    def _check(self):
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"duplicate labels in enum {self.name}")
        if not self.labels:
            raise ValueError(f"enum {self.name} has no labels")
        return self

    def size(self) -> int:
        return len(self.labels)

    def encode(self, v: Any) -> int:
        if isinstance(v, bool):
            raise TypeError("bool is not an enum label")
        if isinstance(v, int) and 0 <= v < len(self.labels):
            return v
        return self.labels.index(v)

    def decode(self, i: int) -> str:
        return self.labels[i]

    def to_memo(self, var: str | None = None) -> str:
        var = var or self.name
        body = "; ".join(f"{lab} = {i}" for i, lab in enumerate(self.labels))
        return f"class {var}(IntEnum): {body}"


class IntRange(_DomainBase):
    """Integers lo..hi inclusive."""
    kind: Literal["int"] = "int"
    lo: int
    hi: int

    @model_validator(mode="after")
    def _check(self):
        if self.hi < self.lo:
            raise ValueError(f"empty range {self.name}: {self.lo}..{self.hi}")
        return self

    def size(self) -> int:
        return self.hi - self.lo + 1

    def encode(self, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"{self.name} expects int, got {v!r}")
        if not (self.lo <= v <= self.hi):
            raise ValueError(f"{v} outside {self.name} {self.lo}..{self.hi}")
        return v - self.lo

    def decode(self, i: int) -> int:
        return self.lo + i

    def to_memo(self, var: str | None = None) -> str:
        var = var or self.name
        return f"{var} = jnp.arange({self.lo}, {self.hi + 1})"


class Grid(_DomainBase):
    """A discretised real interval: n evenly spaced points from lo to hi inclusive."""
    kind: Literal["grid"] = "grid"
    lo: float
    hi: float
    n: int

    def size(self) -> int:
        return self.n

    def step(self) -> float:
        return (self.hi - self.lo) / (self.n - 1) if self.n > 1 else 0.0

    def encode(self, v: Any) -> int:
        if self.n == 1:
            return 0
        i = round((float(v) - self.lo) / self.step())
        if not (0 <= i < self.n) or not math.isclose(self.decode(i), float(v), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{v} is not a grid point of {self.name}")
        return i

    def nearest(self, v: float) -> int:
        if self.n == 1:
            return 0
        return int(min(max(round((float(v) - self.lo) / self.step()), 0), self.n - 1))

    def decode(self, i: int) -> float:
        return self.lo + i * self.step()

    def to_memo(self, var: str | None = None) -> str:
        var = var or self.name
        return f"{var} = jnp.linspace({self.lo}, {self.hi}, {self.n})"


class Bool(_DomainBase):
    kind: Literal["bool"] = "bool"

    def size(self) -> int:
        return 2

    def encode(self, v: Any) -> int:
        return 1 if bool(v) else 0

    def decode(self, i: int) -> bool:
        return bool(i)

    def to_memo(self, var: str | None = None) -> str:
        var = var or self.name
        return f"{var} = jnp.arange(2)"


Scalar = Union[Enum, IntRange, Grid, Bool]


class Product(_DomainBase):
    """A product of named scalar domains, packed mixed-radix into one integer (memo.domain)."""
    kind: Literal["product"] = "product"
    fields: dict[str, Scalar]

    def size(self) -> int:
        n = 1
        for d in self.fields.values():
            n *= d.size()
        return n

    def _radices(self) -> list[tuple[str, int]]:
        # same convention as memo.domain: last field is least significant
        return [(k, d.size()) for k, d in self.fields.items()]

    def encode(self, v: Any) -> int:
        if not isinstance(v, dict):
            raise TypeError(f"{self.name} expects a dict with keys {list(self.fields)}")
        z = 0
        for k, d in self.fields.items():
            z = z * d.size() + d.encode(v[k])
        return z

    def decode(self, i: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, d in reversed(list(self.fields.items())):
            out[k] = d.decode(i % d.size())
            i //= d.size()
        return {k: out[k] for k in self.fields}

    def to_memo(self, var: str | None = None) -> str:
        var = var or self.name
        args = ", ".join(f"{k}={d.size()}" for k, d in self.fields.items())
        return f"{var} = domain({args})"


class Table(BaseModel):
    """A fixed-length array of values of one scalar domain, indexed by another finite domain.

    Not itself a Domain (its joint size is usually too big to enumerate), but each cell is typed.
    """
    model_config = {"frozen": True}
    kind: Literal["table"] = "table"
    name: str
    index: Scalar
    cell: Scalar

    def size(self) -> int:
        return self.cell.size() ** self.index.size()

    def contains(self, v: Any) -> bool:
        if not isinstance(v, (list, tuple)) or len(v) != self.index.size():
            return False
        return all(self.cell.contains(x) for x in v)


class Text(BaseModel):
    """A declared free-text slot. The only place unstructured text enters a game."""
    model_config = {"frozen": True}
    kind: Literal["text"] = "text"
    name: str
    max_chars: int = 4000

    def contains(self, v: Any) -> bool:
        return isinstance(v, str) and len(v) <= self.max_chars


Domain = Union[Enum, IntRange, Grid, Bool, Product]
AnyType = Union[Enum, IntRange, Grid, Bool, Product, Table, Text]
