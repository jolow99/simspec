"""Leak tests: a deliberately planted information leak is caught by the spec checker before any
model call is made (success criterion a)."""
import pytest

from simspec import check_spec, ObservationLeak, Engine, ScriptedAgent
from simspec.agents.base import DryRunAttempted, ActionDistribution
from simspec.agents.llm import LLMAgent, ScaffoldConfig, FakeTransport
from tests.toy import make_spec


class CallCounter:
    """A transport that records whether it was ever called."""
    def __init__(self):
        self.calls = 0

    def complete(self, messages, params):
        self.calls += 1
        from simspec.agents.llm import Completion
        return Completion(text='{"action": "COOP"}', model="counter")


def llm_agents(transport):
    cfg = ScaffoldConfig(model="fake", provider="fake", elicitation="json")
    return {"p1": LLMAgent("p1", cfg, transport), "p2": LLMAgent("p2", cfg, transport)}


def test_leak_in_view_is_caught_before_any_model_call():
    t = CallCounter()
    with pytest.raises(ObservationLeak, match="type2"):
        check_spec(make_spec(leak="view"), llm_agents(t))
    assert t.calls == 0


def test_leak_in_derived_feature_is_caught_before_any_model_call():
    t = CallCounter()
    with pytest.raises(ObservationLeak, match="sneaky"):
        check_spec(make_spec(leak="derived"), llm_agents(t))
    assert t.calls == 0


def test_dry_run_never_calls_the_model_on_a_clean_spec():
    t = CallCounter()
    rep = check_spec(make_spec(), llm_agents(t))
    assert rep["dynamic"]["ok"] and t.calls == 0


def test_smuggled_state_reference_is_caught_by_the_vault_lock():
    """An agent that somehow holds a reference to the engine and reads the live state during act()
    hits the lock before it can build a prompt."""
    spec = make_spec()

    class Sneaky(ScriptedAgent):
        engine = None

        def act(self, obs, aspec, ctx):
            if self.engine is not None:
                _ = self.engine._vault_for_test.state      # planted leak
            return super().act(obs, aspec, ctx)

    agents = {"p1": Sneaky("p1", lambda o, s: "hi" if s.kind == "text" else "COOP"),
              "p2": ScriptedAgent("p2", lambda o, s: "yo" if s.kind == "text" else "COOP")}
    eng = Engine(spec, agents, seed=0)
    # expose the vault to the sneaky agent the way a buggy harness might (e.g. a global)
    from simspec import runtime as rt
    orig = rt.Vault.__init__

    def patched(self, state):
        orig(self, state)
        eng._vault_for_test = self
    rt.Vault.__init__ = patched
    try:
        agents["p1"].engine = eng
        with pytest.raises(ObservationLeak, match="locked"):
            eng.run()
    finally:
        rt.Vault.__init__ = orig


def test_observation_object_cannot_be_mutated():
    spec = make_spec()
    from simspec.chance import Chance
    s0 = spec.initial({}, Chance(0))
    obs = spec.observe(s0, "p1", 0, "play", [])
    with pytest.raises(AttributeError):
        obs.role = "p2"
    with pytest.raises(TypeError):
        obs["round"] = 5
