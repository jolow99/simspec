"""Decision-model ("System One") agent: observation + typed choice question -> native distribution.

Two wire formats cover the ecosystem (docs/research/decision_models.md): TypeSafe-style
``/v1/systemone`` (Jev, Perplexity, Cloudflare, Microsoft, Ollama, llama.cpp, vLLM, OpenRouter) and
OpenAI-style ``/v1/decisions`` (OpenAI, LiteLLM, Vercel, Portkey). Both always return a full
distribution for choice questions; LLM adapters (Anthropic/Google via Vercel, TypeSafe's own adapter)
may not, hence the missing-distribution guard.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from ..spec import Observation, ActionSpec
from ..card import ScaffoldSection, sha
from .base import BaseAgent, ActionDistribution, ActContext, DryRunTransport
from .llm import default_renderer, _fn_hash


@dataclass
class DecisionAnswer:
    choice: str | None
    probs: dict[str, float] | None          # None when the provider returned no distribution
    confidence: float | None = None
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    latency_ms: float = 0.0
    refusal: bool = False


class DecisionTransport(Protocol):
    def decide(self, state: Any, instructions: str, options: dict[str, str | None]) -> DecisionAnswer: ...


class FakeDecisionTransport:
    """For tests: a callable (state, instructions, options) -> DecisionAnswer, or a fixed probs dict / None."""

    def __init__(self, script: Callable[..., DecisionAnswer] | dict[str, float] | None, model: str = "fake-decider"):
        self.script = script
        self.model = model
        self.calls: list[dict] = []

    def decide(self, state, instructions, options):
        self.calls.append({"state": state, "instructions": instructions, "options": options})
        if callable(self.script):
            return self.script(state, instructions, options)
        if self.script is None:  # provider without distributions
            return DecisionAnswer(choice=next(iter(options)), probs=None, model=self.model)
        probs = {k: self.script.get(k, 0.0) for k in options}
        return DecisionAnswer(choice=max(probs, key=probs.get), probs=probs, confidence=None, model=self.model)


class TypeSafeWire:
    """POST {base_url}/v1/systemone  (TypeSafe Jev and compatible servers)."""

    def __init__(self, model: str = "jev-latest", base_url: str = "https://api.typesafe.ai", api_key: str | None = None,
                 api_key_env: str = "TYPESAFE_API_KEY", timeout: float = 60.0, extra_headers: dict | None = None):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.extra_headers = extra_headers or {}

    def request_body(self, state, instructions, options) -> dict:
        return {"model": self.model, "state": state,
                "questions": {"q": {"type": "choice", "instructions": instructions, "criteria": {k: (v or None) for k, v in options.items()}}}}

    @staticmethod
    def parse(data: dict, options: dict) -> DecisionAnswer:
        ans = data["answers"]["q"]
        probs = ans.get("probabilities")
        return DecisionAnswer(choice=ans.get("choice"), probs={k: float(probs.get(k, 0.0)) for k in options} if isinstance(probs, dict) else None,
                              confidence=ans.get("confidence"), model=data.get("model", ""), usage=data.get("usage", {}), raw=data)

    def decide(self, state, instructions, options):
        import httpx
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/v1/systemone", json=self.request_body(state, instructions, options),
                            headers={"Authorization": f"Bearer {self.api_key}", **self.extra_headers})
        r.raise_for_status()
        a = self.parse(r.json(), options)
        a.latency_ms = (time.perf_counter() - t0) * 1000
        return a


class OpenAIDecisionsWire:
    """POST {base_url}/decisions  (OpenAI Decisions API and gateways exposing it)."""

    def __init__(self, model: str = "gpt-6-luna", base_url: str = "https://api.openai.com/v1", api_key: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 60.0, extra_headers: dict | None = None):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.extra_headers = extra_headers or {}

    def request_body(self, state, instructions, options) -> dict:
        return {"model": self.model, "input": state if isinstance(state, str) else json.dumps(state),
                "questions": [{"type": "choice", "name": "q", "instructions": instructions,
                               "choices": [({"value": k, "description": v} if v else {"value": k}) for k, v in options.items()]}]}

    @staticmethod
    def parse(data: dict, options: dict) -> DecisionAnswer:
        ans = data["answers"][0]
        if ans.get("type") == "refusal":
            return DecisionAnswer(choice=None, probs=None, model=data.get("model", ""), raw=data, refusal=True)
        probs = ans.get("probabilities")
        pd = {str(p["value"]): float(p["probability"]) for p in probs} if isinstance(probs, list) else None
        return DecisionAnswer(choice=str(ans.get("choice")) if ans.get("choice") is not None else None,
                              probs={k: pd.get(k, 0.0) for k in options} if pd is not None else None,
                              confidence=ans.get("confidence"), model=data.get("model", ""), usage=data.get("usage", {}), raw=data)

    def decide(self, state, instructions, options):
        import httpx
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/decisions", json=self.request_body(state, instructions, options),
                            headers={"Authorization": f"Bearer {self.api_key}", **self.extra_headers})
        r.raise_for_status()
        a = self.parse(r.json(), options)
        a.latency_ms = (time.perf_counter() - t0) * 1000
        return a


def make_decision_transport(wire: str, model: str, **kw) -> DecisionTransport:
    if wire == "typesafe":
        return TypeSafeWire(model, **kw)
    if wire == "openai":
        return OpenAIDecisionsWire(model, **kw)
    raise ValueError(f"unknown wire {wire}")


class DecisionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    wire: Literal["typesafe", "openai", "fake"] = "typesafe"
    endpoint: str | None = None
    instructions_template: str = "You are {role}. {call_to_action} Choose the best option given the state."
    renderer: str = "default"
    history_k: int = 4
    shuffle_options: bool = True
    n_permutations: int = 1                       # average over this many option orderings (order sensitivity)
    option_descriptions: dict[str, str] = {}
    require_distribution: bool = False            # raise instead of degrading when no distribution is returned
    notes: str = ""


class DecisionModelAgent(BaseAgent):
    def __init__(self, role: str, config: DecisionConfig, transport: DecisionTransport,
                 renderer: Callable[[Observation], str] | None = None):
        self.role = role
        self.config = config
        self.transport = transport
        self._live = transport
        self.renderer = renderer or default_renderer
        self.history: list[tuple[int, str]] = []
        self.stats = {"calls": 0, "no_distribution": 0, "refusals": 0}

    def enter_dry_run(self):
        self.transport = DryRunTransport()

    def exit_dry_run(self):
        self.transport = self._live

    def act(self, obs: Observation, spec: ActionSpec, ctx: ActContext) -> ActionDistribution:
        cfg = self.config
        if spec.kind != "choice":
            raise ValueError("DecisionModelAgent handles finite choice actions only; wrap text/record phases with another agent")
        keys = list(obs.legal) if obs.legal is not None else list(spec.keys() or [])
        state_text = self.renderer(obs)
        if obs.messages:
            state_text += "\nMessages:\n" + "\n".join(f"  [{m.get('channel')}] {m.get('from')}: {m.get('text')}" for m in obs.messages)
        if cfg.history_k and self.history:
            state_text += "\nYour recent decisions: " + ", ".join(f"step {s}: {a}" for s, a in self.history[-cfg.history_k:])
        instr = cfg.instructions_template.format(role=self.role, call_to_action=spec.call_to_action or f"Choose your {spec.name}.")
        if ctx.dry_run:
            return ActionDistribution(probs={k: 1.0 / len(keys) for k in keys}, source="degenerate", meta={"dry_run": True})
        import random
        rnd = random.Random(hash((ctx.seed, ctx.step, self.role)) & 0xFFFFFFFF)
        acc: dict[str, float] = {k: 0.0 for k in keys}
        n_native = 0
        answers = []
        for p in range(max(1, cfg.n_permutations)):
            order = list(keys)
            if cfg.shuffle_options and (p > 0 or cfg.n_permutations > 1 or True):
                rnd.shuffle(order)
            options = {k: cfg.option_descriptions.get(k) for k in order}
            t0 = time.perf_counter()
            ans = self.transport.decide(state_text, instr, options)
            self.stats["calls"] += 1
            ref = ctx.log_call({"state": state_text, "instructions": instr, "options": options},
                               {"choice": ans.choice, "probs": ans.probs, "confidence": ans.confidence, "model": ans.model, "usage": ans.usage},
                               {"role": self.role, "step": ctx.step, "permutation": p}) if ctx.log_call else None
            answers.append({"order": order, "choice": ans.choice, "probs": ans.probs, "confidence": ans.confidence, "call_ref": ref,
                            "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "model": ans.model})
            if ans.refusal:
                self.stats["refusals"] += 1
                continue
            if ans.probs is not None and sum(ans.probs.values()) > 0:
                n_native += 1
                for k, v in ans.probs.items():
                    if k in acc:
                        acc[k] += v
            elif ans.choice in acc:
                self.stats["no_distribution"] += 1
                if cfg.require_distribution:
                    raise ValueError(f"{self.role}: provider returned no distribution")
                acc[ans.choice] += 1.0
        meta = {"model": cfg.model, "wire": cfg.wire, "answers": answers, "n_native": n_native}
        if sum(acc.values()) <= 0:
            meta["fallback"] = "refusal or empty"
            return ActionDistribution(probs={k: 1.0 / len(keys) for k in keys}, source="degenerate", meta=meta)
        source = "native" if n_native == len(answers) and n_native > 0 else "degenerate"
        if 0 < n_native < len(answers):
            source = "native"
            meta["mixed"] = True
        return ActionDistribution(probs=acc, source=source, meta=meta).normalized()

    def record_action(self, step: int, key: str) -> None:
        self.history.append((step, key))

    def describe(self, seat: str) -> ScaffoldSection:
        cfg = self.config
        return ScaffoldSection(
            seat=seat, role=self.role, agent_class="decision_model", provider=cfg.wire, model_id=cfg.model, endpoint=cfg.endpoint,
            api_format={"typesafe": "/v1/systemone", "openai": "/v1/decisions", "fake": "fake"}[cfg.wire],
            templates={"instructions": sha(cfg.instructions_template)}, template_text={"instructions": cfg.instructions_template},
            renderer=cfg.renderer if self.renderer is default_renderer else f"custom:{_fn_hash(self.renderer)}",
            context={"history_k": cfg.history_k, "memory": "none"},
            elicitation={"mode": "choice_question", "shuffle_options": cfg.shuffle_options, "n_permutations": cfg.n_permutations,
                         "option_descriptions": sha(json.dumps(cfg.option_descriptions, sort_keys=True))},
            parsing={"parser": "wire-native", "on_no_distribution": "error" if cfg.require_distribution else "degenerate-on-choice"},
            distribution_source="native",
            distribution={"n_samples": cfg.n_permutations, "normalisation": "average over permutations then renormalise", "evidence_stored": "probs+confidence per call"},
            notes=cfg.notes)
