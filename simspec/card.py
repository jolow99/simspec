"""The Harness Card: GAME SPEC + AGENT SCAFFOLD (per seat) + RUNTIME, saved with every run.

Design follows arXiv 2605.23950 (ETCSOVG disclosure layers), Harbor/ATIF (machine-readable,
hashed), HAL (dated model identity), AgentCompass (semantic vs execution parameters) and
General Agent Evaluation App. G (record provider defaults verbatim). See docs/research.md §6b.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


DistributionSource = Literal["exact", "scripted", "native", "logprob", "verbalized", "degenerate", "replay"]


class DecodingParam(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: Any
    provider_default: bool = False     # True if left at the provider's default (value recorded verbatim)


class ScaffoldSection(BaseModel):
    """Everything about how one seat's agent turns observations into action distributions."""
    model_config = ConfigDict(frozen=True)
    seat: str
    role: str
    agent_class: str
    # model identity (HAL hurdles: dated id, endpoint, quantisation/routing)
    provider: str | None = None
    model_id: str | None = None
    model_resolved: str | None = None
    endpoint: str | None = None
    api_format: str | None = None
    served_weights_note: str | None = None
    # prompts (hashed templates)
    templates: dict[str, str] = {}                  # name -> sha256 of template text
    template_text: dict[str, str] = {}              # name -> full text (kept; cards are small)
    renderer: str | None = None                     # observation renderer id/hash
    # context policy
    context: dict[str, Any] = {}                    # history_k, memory, summarisation, cache, cross_episode_state
    # decoding, verbatim
    decoding: dict[str, DecodingParam] = {}
    # elicitation + parsing + failure policies
    elicitation: dict[str, Any] = {}                # mode, schema_hash, option_label_scheme, shuffle_options
    parsing: dict[str, Any] = {}                    # parser id, on_parse_failure, max_retries, on_illegal
    # distribution source
    distribution_source: DistributionSource = "degenerate"
    distribution: dict[str, Any] = {}               # n_samples, normalisation, evidence_stored
    # seeds, budgets
    seeds: dict[str, Any] = {}
    budgets: dict[str, Any] = {}
    # rational-agent specifics (memo/DP): abstraction, horizon, level, temperature, code hash
    solver: dict[str, Any] = {}
    # execution-only (excluded from the semantic hash)
    execution: dict[str, Any] = {}
    notes: str = ""


class GameSection(BaseModel):
    model_config = ConfigDict(frozen=True)
    game_id: str
    game_version: str
    spec_hash: str
    spec: dict[str, Any]                            # GameSpec.describe()
    seats: dict[str, str]                           # seat -> role
    instance: dict[str, Any] = {}                   # params, instance id, environment id
    illegal_action_policy: dict[str, Any] = {}
    baselines: list[dict[str, Any]] = []            # named reference strategies with code hashes
    budgets: dict[str, Any] = {}


class RuntimeSection(BaseModel):
    model_config = ConfigDict(frozen=True)
    simspec_version: str
    python: str
    platform: str
    libraries: dict[str, str] = {}
    seed: int = 0
    replay_mode: Literal["live", "readback", "recompute"] = "live"
    created_at: str = ""
    code_version: dict[str, Any] = {}
    execution: dict[str, Any] = {}                  # concurrency, log verbosity, infra retries


class HarnessCard(BaseModel):
    model_config = ConfigDict(frozen=True)
    card_version: str = "0.1"
    run_id: str
    game: GameSection
    scaffolds: dict[str, ScaffoldSection]
    runtime: RuntimeSection
    card_hash: str = ""

    def semantic_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        d.pop("card_hash", None)
        d.pop("run_id", None)
        for s in d["scaffolds"].values():
            s.pop("execution", None)
        d["runtime"].pop("execution", None)
        d["runtime"].pop("created_at", None)
        return d

    def compute_hash(self) -> str:
        blob = json.dumps(self.semantic_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def sealed(self) -> "HarnessCard":
        return self.model_copy(update={"card_hash": self.compute_hash()})

    def to_json(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "HarnessCard":
        return cls.model_validate(d)


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    import importlib.metadata as md
    for name in ("simspec", "memo-lang", "jax", "numpy", "pydantic", "httpx"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return out


def runtime_section(seed: int, replay_mode: str = "live", execution: dict | None = None) -> RuntimeSection:
    from . import __version__
    code: dict[str, Any] = {}
    try:
        import subprocess
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if rev.returncode == 0:
            code["git"] = rev.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
            code["dirty"] = bool(dirty.stdout.strip())
    except Exception:
        pass
    return RuntimeSection(simspec_version=__version__, python=sys.version.split()[0], platform=platform.platform(),
                          libraries=library_versions(), seed=seed, replay_mode=replay_mode,
                          created_at=datetime.now(timezone.utc).isoformat(), code_version=code,
                          execution=execution or {})


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
