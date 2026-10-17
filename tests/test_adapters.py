import pytest

from simspec.adapters.pettingzoo_env import ParallelGameSpecEnv
from simspec.adapters.textarena_env import TextArenaGameSpecEnv
from simspec.games.norms import make_spec, agent_names, SMALL_MAP, ACTIONS
from tests.toy import make_spec as toy_spec


def test_pettingzoo_parallel_adapter_runs_norms():
    roles = ["cleaner", "farmer"]
    spec = make_spec(roles, ascii_map=SMALL_MAP, horizon=5)
    env = ParallelGameSpecEnv(spec, seed=0)
    obs, infos = env.reset()
    assert set(obs) == set(agent_names(roles)) and all(len(o["action_mask"]) == len(ACTIONS) for o in obs.values())
    total = 0
    for t in range(5):
        acts = {a: ACTIONS.index("FORWARD") for a in env.agents}
        obs, rew, term, trunc, infos = env.step(acts)
        total += sum(rew.values())
        if all(term.values()):
            break
    assert env.agents == [] and t == 4
    assert env._engine.writer.events[-1]["kind"] == "episode_end"


def test_textarena_adapter_runs_toy_chat_then_play():
    spec = toy_spec()
    env = TextArenaGameSpecEnv(spec, renderers={"p1": lambda o: f"p1 sees {dict(o)}", "p2": lambda o: f"p2 sees {dict(o)}"})
    env.reset()
    done = False
    n = 0
    while not done and n < 50:
        pid, text = env.get_observation()
        assert "sees" in text
        done, info = env.step('{"text": "hi"}' if n < 2 else '{"action": "COOP"}')
        n += 1
    assert done and n == 2 + 3 * 2
    rewards = env.close()
    assert set(rewards) == {0, 1}


def test_inspect_export_if_available(tmp_path):
    pytest.importorskip("inspect_ai")
    from simspec import run_episode
    from simspec.export.inspect_log import export_run
    from tests.test_core import agents
    res = run_episode(toy_spec(), agents(), seed=0, run_dir=str(tmp_path / "r"))
    out = export_run(str(tmp_path / "r"))
    assert out.endswith(".eval")
