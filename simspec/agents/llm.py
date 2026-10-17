"""Generic LLM agent: a declared scaffold (prompts, context policy, decoding, elicitation, parsing,
retry policy, distribution source) over a provider-agnostic transport, constrained to the finite
action set of the current ActionSpec. Everything about the scaffold is reported in the Harness Card.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import string
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from ..spec import Observation, ActionSpec
from ..card import ScaffoldSection, DecodingParam, sha
from .base import BaseAgent, ActionDistribution, ActContext, DryRunTransport


# ----------------------------------------------------------------------------- transports

@dataclass
class Completion:
    text: str
    model: str = ""
    logprobs: list[dict] | None = None        # per generated token: {"token": str, "top": {tok: logprob}}
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    latency_ms: float = 0.0


class Transport(Protocol):
    def complete(self, messages: list[dict], params: dict) -> Completion: ...


class FakeTransport:
    """Scripted responses for tests: a callable (messages, params) -> str, or a list of strings."""

    def __init__(self, script: Callable[[list[dict], dict], str] | list[str], model: str = "fake"):
        self.script = script
        self.model = model
        self.calls: list[tuple[list[dict], dict]] = []

    def complete(self, messages, params):
        self.calls.append((messages, params))
        if callable(self.script):
            text = self.script(messages, params)
        else:
            text = self.script[(len(self.calls) - 1) % len(self.script)]
        return Completion(text=text, model=self.model, usage={"input_tokens": 0, "output_tokens": 0})


class CassetteTransport:
    """Record/replay wrapper. Keyed by the hash of (messages, params). ``replay`` mode never calls
    the inner transport and fails on a cache miss, so recorded runs are exactly reproducible."""

    def __init__(self, path: str, inner: Transport | None = None, mode: Literal["record", "replay", "auto"] = "auto"):
        self.path = path
        self.inner = inner
        self.mode = mode
        self.store: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self.store[rec["key"]] = rec

    @staticmethod
    def key(messages, params) -> str:
        blob = json.dumps({"messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def complete(self, messages, params):
        k = self.key(messages, params)
        if k in self.store and self.mode in ("replay", "auto"):
            self.hits += 1
            r = self.store[k]
            return Completion(text=r["text"], model=r.get("model", ""), logprobs=r.get("logprobs"), usage=r.get("usage", {}), raw=r.get("raw"))
        if self.mode == "replay" or self.inner is None:
            raise KeyError(f"cassette miss in replay mode (key {k}); record this run first")
        self.misses += 1
        c = self.inner.complete(messages, params)
        rec = {"key": k, "messages": messages, "params": params, "text": c.text, "model": c.model,
               "logprobs": c.logprobs, "usage": c.usage, "raw": _jsonable(c.raw)}
        self.store[k] = rec
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return c


class OpenAICompatTransport:
    """Chat-completions over HTTP (OpenAI, OpenRouter, vLLM, Ollama, LiteLLM, ...)."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", extra_headers: dict | None = None, timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.extra_headers = extra_headers or {}
        self.timeout = timeout

    def complete(self, messages, params):
        import httpx
        body = {"model": self.model, "messages": messages}
        for k in ("temperature", "top_p", "max_tokens", "seed", "stop", "logprobs", "top_logprobs", "response_format", "reasoning_effort"):
            if k in params and params[k] is not None:
                body[k] = params[k]
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {self.api_key}", **self.extra_headers})
        r.raise_for_status()
        data = r.json()
        ch = data["choices"][0]
        text = ch["message"].get("content") or ""
        lp = None
        if ch.get("logprobs") and ch["logprobs"].get("content"):
            lp = [{"token": t["token"], "logprob": t["logprob"],
                   "top": {x["token"]: x["logprob"] for x in t.get("top_logprobs", [])}} for t in ch["logprobs"]["content"]]
        return Completion(text=text, model=data.get("model", self.model), logprobs=lp, usage=data.get("usage", {}),
                          raw=data, latency_ms=(time.perf_counter() - t0) * 1000)


class AnthropicTransport:
    def __init__(self, model: str, api_key: str | None = None, api_key_env: str = "ANTHROPIC_API_KEY",
                 base_url: str = "https://api.anthropic.com", timeout: float = 120.0):
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, messages, params):
        import httpx
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        body = {"model": self.model, "messages": msgs, "max_tokens": params.get("max_tokens", 1024)}
        if system:
            body["system"] = system
        for k in ("temperature", "top_p", "stop_sequences"):
            if k in params and params[k] is not None:
                body[k] = params[k]
        if params.get("thinking"):
            body["thinking"] = params["thinking"]
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/v1/messages", json=body,
                            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        r.raise_for_status()
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return Completion(text=text, model=data.get("model", self.model), usage=data.get("usage", {}), raw=data,
                          latency_ms=(time.perf_counter() - t0) * 1000)


def make_transport(provider: str, model: str, **kw) -> Transport:
    if provider in ("openai", "openai_compat", "openrouter", "vllm", "ollama", "litellm"):
        defaults = {"openrouter": "https://openrouter.ai/api/v1", "ollama": "http://localhost:11434/v1",
                    "vllm": "http://localhost:8000/v1", "litellm": "http://localhost:4000/v1"}
        if provider in defaults:
            kw.setdefault("base_url", defaults[provider])
        if provider == "openrouter":
            kw.setdefault("api_key_env", "OPENROUTER_API_KEY")
        return OpenAICompatTransport(model, **kw)
    if provider == "anthropic":
        return AnthropicTransport(model, **kw)
    raise ValueError(f"unknown provider {provider}")


# ----------------------------------------------------------------------------- scaffold config

DEFAULT_SYSTEM = ("You are playing the role of {role} in a game. Follow the instructions exactly and answer in the "
                  "requested format only.")
DEFAULT_TURN = ("{observation}\n\n{history}{call_to_action}\n{menu}\n{format}")


class ScaffoldConfig(BaseModel):
    """Everything about how the LLM is prompted, decoded and parsed. Serialised into the Harness Card."""
    model_config = ConfigDict(frozen=True)
    model: str
    provider: str = "openai"
    endpoint: str | None = None
    system_template: str = DEFAULT_SYSTEM
    turn_template: str = DEFAULT_TURN
    renderer: str = "default"
    history_k: int = 8                          # how many of this agent's past (obs, action) pairs to include
    include_messages: bool = True               # include channel messages from the observation
    decoding: dict[str, Any] = {"temperature": 0.0, "max_tokens": 512}
    provider_default_keys: list[str] = []       # decoding keys left at provider default (recorded verbatim above)
    elicitation: Literal["json", "letter", "name"] = "json"
    option_labels: Literal["name", "letter"] = "name"
    shuffle_options: bool = False
    distribution_source: Literal["degenerate", "verbalized", "logprob", "samples"] = "degenerate"
    n_samples: int = 1
    max_retries: int = 2
    on_parse_failure: Literal["default", "uniform", "error"] = "default"
    retry_feedback: str = "Your previous answer could not be parsed. {reason}. Answer again using exactly the requested format."
    max_calls_per_decision: int = 4
    notes: str = ""


# ----------------------------------------------------------------------------- the agent

def default_renderer(obs: Observation) -> str:
    lines = [f"Step {obs.step}, phase {obs.phase}. You are {obs.role}.", "Observation:"]
    for k, v in obs.items():
        lines.append(f"  {k}: {json.dumps(v) if not isinstance(v, str) else v}")
    return "\n".join(lines)


class LLMAgent(BaseAgent):
    def __init__(self, role: str, config: ScaffoldConfig, transport: Transport,
                 renderer: Callable[[Observation], str] | None = None, seat_name: str | None = None):
        self.role = role
        self.config = config
        self.transport = transport
        self._live_transport = transport
        self.renderer = renderer or default_renderer
        self.history: list[tuple[str, str]] = []     # (rendered obs summary, action key)
        self.stats = {"calls": 0, "parse_failures": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0}
        self._rng_state = 0

    # dry-run support used by check_spec
    def enter_dry_run(self):
        self.transport = DryRunTransport()

    def exit_dry_run(self):
        self.transport = self._live_transport

    # -- prompt construction -------------------------------------------------------------------
    def _menu(self, spec: ActionSpec, legal: list[str] | None, ctx: ActContext) -> tuple[str, list[str], dict[str, str]]:
        """Returns (menu text, option keys in presented order, label->key map)."""
        if spec.kind == "choice":
            keys = legal if legal is not None else (spec.keys() or [])
            keys = list(keys)
            if self.config.shuffle_options:
                import random
                rnd = random.Random(hash((ctx.seed, ctx.step, self.role)) & 0xFFFFFFFF)
                rnd.shuffle(keys)
            if self.config.option_labels == "letter":
                labels = {string.ascii_uppercase[i]: k for i, k in enumerate(keys)}
                menu = "Options:\n" + "\n".join(f"  {lab}: {k}" for lab, k in labels.items())
            else:
                labels = {k: k for k in keys}
                menu = "Options: " + ", ".join(keys)
            return menu, keys, labels
        if spec.kind == "record":
            fields = {k: _type_hint(t) for k, t in (spec.record or {}).items()}
            return "Answer with a JSON object with fields: " + json.dumps(fields), [], {}
        return f"Answer with free text (at most {spec.text.max_chars if spec.text else 4000} characters).", [], {}

    def _format(self, spec: ActionSpec) -> str:
        if spec.kind == "choice":
            if self.config.elicitation == "json":
                base = 'Reply with JSON only: {"action": "<option>"}'
                if self.config.distribution_source == "verbalized":
                    base = 'Reply with JSON only: {"probabilities": {"<option>": <p>, ...}} giving your probability of choosing each option (they must sum to 1).'
                return base
            if self.config.elicitation == "letter":
                return "Reply with the single letter of your chosen option and nothing else."
            return "Reply with the exact name of your chosen option and nothing else."
        if spec.kind == "record":
            return 'Reply with JSON only: {"action": {<fields>}}'
        return 'Reply with JSON only: {"text": "<your text>"}'

    def _messages(self, obs: Observation, spec: ActionSpec, ctx: ActContext, menu: str) -> list[dict]:
        rendered = self.renderer(obs)
        if self.config.include_messages and obs.messages:
            rendered += "\nMessages:\n" + "\n".join(f"  [{m.get('channel')}] {m.get('from')} -> {m.get('to', 'all')}: {m.get('text')}" for m in obs.messages)
        hist = ""
        if self.config.history_k > 0 and self.history:
            h = self.history[-self.config.history_k:]
            hist = "Your recent decisions:\n" + "\n".join(f"  step {s}: {a}" for s, a in h) + "\n\n"
        turn = self.config.turn_template.format(observation=rendered, history=hist, call_to_action=spec.call_to_action or f"Choose your {spec.name}.",
                                                menu=menu, format=self._format(spec), role=self.role)
        return [{"role": "system", "content": self.config.system_template.format(role=self.role)},
                {"role": "user", "content": turn}]

    # -- parsing ---------------------------------------------------------------------------------
    def _parse(self, text: str, spec: ActionSpec, keys: list[str], labels: dict[str, str]) -> tuple[Any, str | None]:
        """Returns (parsed, error). parsed is a key (choice), dict (record), str (text), or probs dict."""
        t = text.strip()
        if spec.kind == "choice":
            if self.config.elicitation == "json":
                obj = _extract_json(t)
                if obj is None:
                    return None, "no JSON object found"
                if self.config.distribution_source == "verbalized":
                    probs = obj.get("probabilities")
                    if not isinstance(probs, dict):
                        return None, "missing 'probabilities' object"
                    out = {}
                    for lab, p in probs.items():
                        k = labels.get(str(lab), str(lab))
                        if k in keys:
                            try:
                                out[k] = max(float(p), 0.0)
                            except (TypeError, ValueError):
                                return None, f"non-numeric probability for {lab}"
                    if not out or sum(out.values()) <= 0:
                        return None, "no probability mass on legal options"
                    return out, None
                a = obj.get("action")
                if a is None:
                    return None, "missing 'action'"
                k = labels.get(str(a), str(a))
                if k not in keys:
                    near = [x for x in keys if x.lower() == str(a).lower()]
                    if near:
                        return near[0], None
                    return None, f"'{a}' is not one of the options"
                return k, None
            # letter / name
            tok = t.split()[0].strip(".,:;'\"()[]") if t else ""
            if self.config.elicitation == "letter":
                k = labels.get(tok.upper())
                return (k, None) if k else (None, f"'{tok}' is not an option letter")
            for k in keys:
                if t.lower().startswith(k.lower()):
                    return k, None
            return None, f"'{t[:40]}' is not an option"
        if spec.kind == "record":
            obj = _extract_json(t)
            if obj is None:
                return None, "no JSON object found"
            val = obj.get("action", obj)
            if not isinstance(val, dict):
                return None, "'action' is not an object"
            if not spec.validate(val):
                return None, f"fields invalid; expected {list((spec.record or {}).keys())}"
            return val, None
        obj = _extract_json(t)
        if obj is not None and isinstance(obj.get("text"), str):
            return obj["text"], None
        return t, None

    # -- act -------------------------------------------------------------------------------------
    def act(self, obs: Observation, spec: ActionSpec, ctx: ActContext) -> ActionDistribution:
        cfg = self.config
        legal = obs.legal
        menu, keys, labels = self._menu(spec, legal, ctx)
        messages = self._messages(obs, spec, ctx, menu)
        params = dict(cfg.decoding)
        if spec.kind == "choice" and cfg.distribution_source == "logprob" and cfg.elicitation == "letter":
            params.update({"logprobs": True, "top_logprobs": min(20, max(5, len(keys))), "max_tokens": 4})
        if ctx.dry_run:
            # the prompt has been built from the Observation (any leak would have raised); do not call the model
            return self._fallback(spec, keys, {"dry_run": True, "prompt_chars": sum(len(m["content"]) for m in messages)}, "dry run")
        n = cfg.n_samples if (spec.kind == "choice" and cfg.distribution_source == "samples") else 1
        counts: dict[str, float] = {}
        parsed_final: Any = None
        probs_final: dict[str, float] | None = None
        calls = 0
        attempts_log = []
        for _ in range(n):
            msgs = list(messages)
            reason = None
            for attempt in range(cfg.max_retries + 1):
                if calls >= cfg.max_calls_per_decision:
                    break
                if reason:
                    msgs = msgs + [{"role": "user", "content": cfg.retry_feedback.format(reason=reason)}]
                t0 = time.perf_counter()
                comp = self.transport.complete(msgs, params)
                calls += 1
                self.stats["calls"] += 1
                self.stats["input_tokens"] += int(comp.usage.get("input_tokens", comp.usage.get("prompt_tokens", 0)) or 0)
                self.stats["output_tokens"] += int(comp.usage.get("output_tokens", comp.usage.get("completion_tokens", 0)) or 0)
                ref = ctx.log_call({"messages": msgs, "params": params}, {"text": comp.text, "model": comp.model, "usage": comp.usage, "logprobs": comp.logprobs},
                                   {"role": self.role, "step": ctx.step, "attempt": attempt}) if ctx.log_call else None
                parsed, err = self._parse(comp.text, spec, keys, labels)
                attempts_log.append({"attempt": attempt, "ok": err is None, "error": err, "call_ref": ref, "model": comp.model,
                                     "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
                if err is None:
                    if spec.kind == "choice" and cfg.distribution_source == "logprob" and comp.logprobs:
                        probs_final = _logprob_distribution(comp.logprobs, labels, keys)
                    parsed_final = parsed
                    break
                reason = err
                self.stats["parse_failures"] += 1
                self.stats["retries"] += 1
            if parsed_final is not None and spec.kind == "choice" and cfg.distribution_source == "samples":
                counts[parsed_final] = counts.get(parsed_final, 0) + 1
                parsed_final = None
        meta = {"model": cfg.model, "attempts": attempts_log, "calls": calls, "menu_order": keys if spec.kind == "choice" else None}
        # assemble distribution
        if spec.kind == "choice":
            if cfg.distribution_source == "samples" and counts:
                return ActionDistribution(probs=counts, source="verbalized", meta={**meta, "n_samples": n}).normalized()
            if parsed_final is None:
                return self._fallback(spec, keys, meta, "parse failure")
            if cfg.distribution_source == "verbalized" and isinstance(parsed_final, dict):
                return ActionDistribution(probs=parsed_final, source="verbalized", meta=meta).normalized()
            if probs_final is not None:
                return ActionDistribution(probs=probs_final, source="logprob", meta={**meta, "argmax_parsed": parsed_final}).normalized()
            return ActionDistribution.point(parsed_final, source="degenerate", **meta)
        if parsed_final is None:
            return self._fallback(spec, keys, meta, "parse failure")
        if spec.kind == "text":
            return ActionDistribution.of_text(str(parsed_final), source="degenerate", **meta)
        return ActionDistribution.point(spec.key_of(parsed_final), source="degenerate", **meta)

    def _fallback(self, spec: ActionSpec, keys: list[str], meta: dict, why: str) -> ActionDistribution:
        meta = {**meta, "fallback": why}
        if self.config.on_parse_failure == "error":
            raise ValueError(f"{self.role}: {why}")
        if spec.kind == "choice" and keys:
            if self.config.on_parse_failure == "uniform":
                return ActionDistribution(probs={k: 1.0 / len(keys) for k in keys}, source="degenerate", meta=meta)
            return ActionDistribution.point(spec.key_of(spec.default) if spec.default is not None else keys[0], source="degenerate", **meta)
        if spec.kind == "text":
            return ActionDistribution.of_text("", source="degenerate", **meta)
        return ActionDistribution.point(spec.key_of(spec.default), source="degenerate", **meta)

    def observe(self, obs: Observation, ctx: ActContext) -> None:
        pass

    def record_action(self, step: int, key: str) -> None:
        self.history.append((step, key))

    def get_state(self):
        return {"history": list(self.history), "stats": dict(self.stats)}

    def set_state(self, s):
        self.history = [tuple(x) for x in s.get("history", [])]

    def describe(self, seat: str) -> ScaffoldSection:
        cfg = self.config
        return ScaffoldSection(
            seat=seat, role=self.role, agent_class="llm", provider=cfg.provider, model_id=cfg.model, endpoint=cfg.endpoint,
            api_format="chat", templates={"system": sha(cfg.system_template), "turn": sha(cfg.turn_template)},
            template_text={"system": cfg.system_template, "turn": cfg.turn_template},
            renderer=cfg.renderer if self.renderer is default_renderer else f"custom:{_fn_hash(self.renderer)}",
            context={"history_k": cfg.history_k, "include_messages": cfg.include_messages, "memory": "none", "cross_episode_state": False},
            decoding={k: DecodingParam(value=v, provider_default=k in cfg.provider_default_keys) for k, v in cfg.decoding.items()},
            elicitation={"mode": cfg.elicitation, "option_labels": cfg.option_labels, "shuffle_options": cfg.shuffle_options},
            parsing={"parser": "simspec.agents.llm.LLMAgent._parse", "max_retries": cfg.max_retries,
                     "on_parse_failure": cfg.on_parse_failure, "retry_feedback": sha(cfg.retry_feedback)},
            distribution_source={"degenerate": "degenerate", "verbalized": "verbalized", "logprob": "logprob", "samples": "verbalized"}[cfg.distribution_source],
            distribution={"n_samples": cfg.n_samples, "normalisation": "renormalise over legal keys"},
            budgets={"max_calls_per_decision": cfg.max_calls_per_decision},
            notes=cfg.notes)


# ----------------------------------------------------------------------------- helpers

def _extract_json(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cands = [m.group(1)] if m else []
    cands.append(text)
    # also the first {...} block
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        cands.append(text[i:j + 1])
    for c in cands:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            continue
    return None


def _logprob_distribution(logprobs: list[dict], labels: dict[str, str], keys: list[str]) -> dict[str, float] | None:
    """Distribution over options from the top logprobs of the first generated token (letter elicitation)."""
    if not logprobs:
        return None
    top = logprobs[0].get("top", {})
    out: dict[str, float] = {}
    for tok, lp in top.items():
        lab = tok.strip().upper()
        if lab in labels:
            k = labels[lab]
            out[k] = out.get(k, 0.0) + math.exp(lp)
    if not out:
        return None
    return out


def _type_hint(t: Any) -> Any:
    kind = getattr(t, "kind", None)
    if kind == "enum":
        return {"one_of": list(t.labels)}
    if kind == "int":
        return {"int": [t.lo, t.hi]}
    if kind == "bool":
        return {"bool": True}
    if kind == "text":
        return {"text": t.max_chars}
    if kind == "table":
        return {"list_of": _type_hint(t.cell), "length": t.index.size()}
    if kind == "product":
        return {k: _type_hint(v) for k, v in t.fields.items()}
    return str(t)


def _fn_hash(fn) -> str:
    import inspect
    try:
        return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return hashlib.sha256(repr(fn).encode()).hexdigest()[:16]


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return repr(x)
