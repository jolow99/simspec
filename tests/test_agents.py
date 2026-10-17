import json
import os

import numpy as np
import pytest

from simspec import run_episode, ActionSpec, Enum, IntRange, Product
from simspec.agents.base import ActContext
from simspec.agents.llm import LLMAgent, ScaffoldConfig, FakeTransport, CassetteTransport, Completion, _extract_json
from simspec.agents.decision import (DecisionModelAgent, DecisionConfig, FakeDecisionTransport, TypeSafeWire, OpenAIDecisionsWire, DecisionAnswer)
from simspec.agents.memo_agent import build_memo_module, InfoStateEncoder, TablePolicyAgent
from simspec.spec import Observation
from tests.toy import make_spec, Act


def obs(legal=("COOP", "DEFECT")):
    return Observation("p1", 0, "play", {"round": 0, "last_p2": "NONE"}, [], list(legal))


SPEC = ActionSpec(name="move", kind="choice", domain=Act, default="COOP")
CTX = ActContext(step=0, phase="play", round=0, seed=0)


def test_llm_json_parse_and_retry():
    t = FakeTransport(["garbage", '{"action": "DEFECT"}'])
    ag = LLMAgent("p1", ScaffoldConfig(model="m", provider="fake", max_retries=2), t)
    d = ag.act(obs(), SPEC, CTX)
    assert d.probs == {"DEFECT": 1.0} and d.source == "degenerate"
    assert len(t.calls) == 2 and "could not be parsed" in t.calls[1][0][-1]["content"]
    assert ag.stats["parse_failures"] == 1


def test_llm_parse_failure_fallback_default():
    t = FakeTransport(["nope"] * 5)
    ag = LLMAgent("p1", ScaffoldConfig(model="m", provider="fake", max_retries=1, on_parse_failure="default"), t)
    d = ag.act(obs(), SPEC, CTX)
    assert d.probs == {"COOP": 1.0} and d.meta["fallback"]


def test_llm_verbalized_distribution():
    t = FakeTransport(['{"probabilities": {"COOP": 0.7, "DEFECT": 0.3}}'])
    ag = LLMAgent("p1", ScaffoldConfig(model="m", provider="fake", distribution_source="verbalized"), t)
    d = ag.act(obs(), SPEC, CTX)
    assert d.source == "verbalized" and abs(d.probs["COOP"] - 0.7) < 1e-9


def test_llm_logprob_distribution_from_letters():
    class LP:
        def complete(self, messages, params):
            assert params.get("logprobs") is True
            return Completion(text="A", logprobs=[{"token": "A", "logprob": -0.1, "top": {"A": np.log(0.8), "B": np.log(0.2)}}])
    ag = LLMAgent("p1", ScaffoldConfig(model="m", provider="fake", elicitation="letter", option_labels="letter", distribution_source="logprob"), LP())
    d = ag.act(obs(), SPEC, CTX)
    assert d.source == "logprob" and abs(d.probs["COOP"] - 0.8) < 1e-6 and abs(d.probs["DEFECT"] - 0.2) < 1e-6


def test_llm_record_and_text_actions():
    rec = ActionSpec(name="pay", kind="record", record={"amount": IntRange(name="m", lo=0, hi=100)}, default={"amount": 0})
    t = FakeTransport(['{"action": {"amount": 42}}'])
    ag = LLMAgent("p1", ScaffoldConfig(model="m", provider="fake"), t)
    d = ag.act(Observation("p1", 0, "pay", {}, [], None), rec, CTX)
    assert d.probs == {'{"amount":42}': 1.0}
    from simspec import Text
    txt = ActionSpec(name="say", kind="text", text=Text(name="t", max_chars=100))
    t2 = FakeTransport(['{"text": "hello there"}'])
    d2 = LLMAgent("p1", ScaffoldConfig(model="m", provider="fake"), t2).act(Observation("p1", 0, "chat", {}, [], None), txt, CTX)
    assert d2.text == "hello there"


def test_cassette_record_then_replay(tmp_path):
    path = str(tmp_path / "cassette.jsonl")
    inner = FakeTransport(['{"action": "DEFECT"}'])
    rec = CassetteTransport(path, inner, mode="record")
    ag = LLMAgent("p1", ScaffoldConfig(model="m", provider="fake"), rec)
    d1 = ag.act(obs(), SPEC, CTX)
    assert rec.misses == 1 and os.path.exists(path)
    rep = CassetteTransport(path, None, mode="replay")
    ag2 = LLMAgent("p1", ScaffoldConfig(model="m", provider="fake"), rep)
    d2 = ag2.act(obs(), SPEC, CTX)
    assert d2.probs == d1.probs and rep.hits == 1
    with pytest.raises(KeyError):
        ag2.act(Observation("p1", 1, "play", {"round": 1, "last_p2": "COOP"}, [], ["COOP", "DEFECT"]), SPEC, ActContext(step=1, phase="play", round=1, seed=0))


def test_llm_agent_plays_toy_episode_and_card_records_scaffold():
    def script(messages, params):
        user = messages[-1]["content"]
        if "free text" in user:
            return '{"text": "hello"}'
        return '{"action": "COOP"}'
    t = FakeTransport(script)
    cfg = ScaffoldConfig(model="fake-1", provider="fake")
    agents = {"p1": LLMAgent("p1", cfg, t), "p2": LLMAgent("p2", cfg, t)}
    res = run_episode(make_spec(), agents, seed=0)
    assert res.trajectory.steps() == 5
    sc = res.card.scaffolds["p1"]
    assert sc.agent_class == "llm" and sc.model_id == "fake-1" and sc.distribution_source == "degenerate"
    assert sc.templates["system"] and sc.decoding["temperature"].value == 0.0
    assert len(res.trajectory.calls) == 8   # 2 chat turns + 3 rounds x 2 players


def test_extract_json_variants():
    assert _extract_json('Sure! ```json\n{"action": "A"}\n```')["action"] == "A"
    assert _extract_json('text before {"action": "B"} after')["action"] == "B"
    assert _extract_json("nothing") is None


# ----------------------------------------------------------------------------- decision models

def test_decision_wire_schemas():
    ts = TypeSafeWire("jev-latest")
    body = ts.request_body("state text", "pick", {"COOP": None, "DEFECT": "defect"})
    assert body["questions"]["q"]["type"] == "choice" and body["questions"]["q"]["criteria"] == {"COOP": None, "DEFECT": "defect"}
    ans = TypeSafeWire.parse({"model": "jev-1.13.0", "answers": {"q": {"type": "choice", "choice": "COOP", "probabilities": {"COOP": 0.88, "DEFECT": 0.12}, "confidence": 0.76}}, "usage": {}}, body["questions"]["q"]["criteria"])
    assert ans.probs == {"COOP": 0.88, "DEFECT": 0.12} and ans.model == "jev-1.13.0"
    oa = OpenAIDecisionsWire("gpt-6-luna")
    body = oa.request_body("state text", "pick", {"COOP": None, "DEFECT": "defect"})
    assert body["questions"][0]["choices"] == [{"value": "COOP"}, {"value": "DEFECT", "description": "defect"}]
    ans = OpenAIDecisionsWire.parse({"model": "gpt-6-luna", "answers": [{"type": "choice", "name": "q", "choice": "DEFECT",
                                     "probabilities": [{"value": "COOP", "probability": 0.3}, {"value": "DEFECT", "probability": 0.7}], "confidence": 0.4}]}, {"COOP": None, "DEFECT": None})
    assert ans.probs == {"COOP": 0.3, "DEFECT": 0.7}
    ref = OpenAIDecisionsWire.parse({"model": "x", "answers": [{"type": "refusal", "name": "q"}]}, {"COOP": None})
    assert ref.refusal and ref.probs is None


def test_decision_agent_native_and_guard():
    ag = DecisionModelAgent("p1", DecisionConfig(model="fake", wire="fake", n_permutations=2), FakeDecisionTransport({"COOP": 0.6, "DEFECT": 0.4}))
    d = ag.act(obs(), SPEC, CTX)
    assert d.source == "native" and abs(d.probs["COOP"] - 0.6) < 1e-9 and d.meta["n_native"] == 2
    # provider without distributions -> degenerate on its choice, flagged
    ag2 = DecisionModelAgent("p1", DecisionConfig(model="fake", wire="fake"), FakeDecisionTransport(None))
    d2 = ag2.act(obs(), SPEC, CTX)
    assert d2.source == "degenerate" and ag2.stats["no_distribution"] == 1
    ag3 = DecisionModelAgent("p1", DecisionConfig(model="fake", wire="fake", require_distribution=True), FakeDecisionTransport(None))
    with pytest.raises(ValueError):
        ag3.act(obs(), SPEC, CTX)
    assert ag.describe("p1").distribution_source == "native"


# ----------------------------------------------------------------------------- memo

def test_build_memo_module_and_table_agent():
    header = "S = jnp.arange(4)\nA = jnp.arange(2)\n@jax.jit\ndef R(s, a):\n    return RT[s, a]\n"
    src = '''
@memo(install_module=m.install, cache=True)
def Q[s: S, a: A](t):
    ag: knows(s, a)
    ag: chooses(s_ in S, wpp=1.0 * (s_ == (s + a) % 4))
    ag: chooses(a_ in A, to_maximize=0.0 if t <= 0 else Q[s_, a_](t - 1))
    return E[R(s, a) + (0.0 if t <= 0 else 0.9 * Q[ag.s_, ag.a_](t - 1))]
'''
    import jax.numpy as jnp
    RT = jnp.array([[0., 1.], [1., 0.], [0., 1.], [1., 0.]])
    m = build_memo_module("test_q", header, src, globals_={"RT": RT})
    q = np.asarray(m.Q(5))
    assert q.shape == (4, 2)
    from simspec.agents.memo_agent import softmax_rows
    table = softmax_rows(q, 10.0)
    enc = InfoStateEncoder(Product(name="is", fields={"s": IntRange(name="s", lo=0, hi=3)}), lambda o: {"s": o["s"]})
    ag = TablePolicyAgent("p", enc, table, ["0", "1"])
    d = ag.act(Observation("p", 0, "x", {"s": 1}, [], None), ActionSpec(name="a", kind="choice", domain=IntRange(name="a", lo=0, hi=1)), CTX)
    assert d.source == "exact" and d.argmax() == "0"
    assert ag.describe("p").solver["table_hash"]
