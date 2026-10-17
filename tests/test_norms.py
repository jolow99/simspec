import numpy as np
import pytest

from simspec import run_episode, check_spec, replay_episode, ObservationLeak
from simspec.chance import Chance
from simspec.games.norms import make_spec, default_rules, rule_space, agent_names, NormPlanner, NormLearner, MemoRulePosterior, SMALL_MAP, ACTIONS
from simspec.games.norms.world import Layout, initial_world, apply_action, env_step, dirt_fraction, apples_around, beam_cells
from simspec.games.norms.spec import features_of, world_of
from simspec.games.norms.metrics import learning_summary, social_outcomes

ROLES = ["cleaner", "farmer", "free", "free"]


def test_layout_and_world_mechanics():
    L = Layout(SMALL_MAP)
    assert L.wall.sum() > 0 and L.river.sum() > 0 and L.apple_capable.sum() > 0 and L.spawns
    w = initial_world(L, ROLES, Chance(0))
    assert len(w["pos"]) == 4 and len(set(w["pos"])) == 4
    df0 = dirt_fraction(L, w["dirt"])
    assert 0 <= df0 <= 1
    # harvesting: move an agent onto an apple cell
    i = 0
    apple_cells = [c for c in range(L.n) if w["apple"][c]]
    c = apple_cells[0]
    x, y = L.xy(c)
    w["pos"][i] = L.cell(x, y + 1) if L.cell(x, y + 1) is not None and L.walkable[L.cell(x, y + 1)] else L.cell(x - 1, y)
    w["orient"][i] = 0 if w["pos"][i] == L.cell(x, y + 1) else 1
    w2, r = apply_action(L, w, i, "FORWARD")
    assert w2["pos"][i] == c and w2["harvested"][i] and w2["inventory"][i] == 1 and not w2["apple"][c]
    w3, r = apply_action(L, w2, i, "EAT")
    assert r > 0.9 and w3["inventory"][i] == 0 and w3["ate"][i]
    # env step increments counters and keeps types (env_step mutates in place: snapshot first)
    before = w3["since_paid"][i]
    w4 = env_step(L, w3, Chance(1))
    assert w4["t"] == 1 and w4["since_paid"][i] == before + 1


def test_features_and_default_rules():
    spec = make_spec(ROLES, ascii_map=SMALL_MAP, horizon=10)
    s0 = spec.initial(spec.params, Chance(0))
    f = features_of(spec.layout, world_of(s0), 0)
    assert set(f) >= {"DIRT_FRACTION", "NUM_APPLES_AROUND", "HARVESTED", "AGENT_LOOK"}
    rules = default_rules(ROLES)
    assert {r.id for r in rules} == {"P1_no_empty_orchard", "P2_no_stealing", "O1_farmer_pays", "O2_cleaner_cleans", "O3_free_cleans"}
    assert set(next(r for r in rules if r.id == "O2_cleaner_cleans").scope) == {"cleaner0"}
    space = rule_space(ROLES)
    assert len(space) > 40 and len({r.id for r in space}) == len(space)
    sigs = {r.signature() for r in space}
    assert all(r.signature() in sigs for r in rules), "every default norm must be in the hypothesis space"


def test_episode_with_planners_runs_logs_and_replays(tmp_path):
    spec = make_spec(ROLES, ascii_map=SMALL_MAP, horizon=15)
    names = agent_names(ROLES)
    agents = {n: NormPlanner(n, spec.layout, names, default_rules(ROLES), seed=i) for i, n in enumerate(names)}
    rep = check_spec(spec, agents, rollout_steps=3)
    assert rep["dynamic"]["ok"]
    run_dir = str(tmp_path / "norms")
    res = run_episode(spec, agents, seed=1, run_dir=run_dir)
    assert res.trajectory.steps() == 15 and set(res.payoffs) == set(names)
    assert all(e["source"] == "exact" for e in res.trajectory.agent_calls())
    so = social_outcomes(res.trajectory)
    assert so["steps"] == 15 and 0 <= so["desiccated_final"] <= 1
    res2 = replay_episode(lambda params: make_spec(params["roles"], ascii_map=SMALL_MAP, horizon=params["horizon"]), run_dir)
    assert res2.payoffs == res.payoffs


def test_planted_leak_is_caught():
    with pytest.raises(ObservationLeak, match="peek"):
        check_spec(make_spec(ROLES, ascii_map=SMALL_MAP, horizon=5, leak="derived"))


def test_memo_posterior_matches_numpy():
    rng = np.random.default_rng(0)
    R, A = 7, len(ACTIONS)
    prior = rng.uniform(0.05, 0.9, R)
    lik = rng.dirichlet(np.ones(A), size=(R, 2))
    mp = MemoRulePosterior(R, A)
    for a in range(A):
        post = mp.update(prior, lik, a)
        expected = lik[:, 1, a] * prior / (lik[:, 1, a] * prior + lik[:, 0, a] * (1 - prior))
        assert np.allclose(post, expected, atol=1e-5)


def test_learner_updates_beliefs_and_logs_them():
    spec = make_spec(ROLES, ascii_map=SMALL_MAP, horizon=30)
    names = agent_names(ROLES)
    space = rule_space(ROLES)
    rules = default_rules(ROLES)
    agents = {n: NormPlanner(n, spec.layout, names, rules, seed=i) for i, n in enumerate(names[:3])}
    learner = NormLearner(names[3], spec.layout, names, space, seed=3, use_memo=True)
    agents[names[3]] = learner
    res = run_episode(spec, agents, seed=2)
    infos = [e for e in res.trajectory.of("info") if e["source"] == f"beliefs:{names[3]}"]
    assert len(infos) == 30
    assert any(abs(b - 0.05) > 1e-6 for b in learner.beliefs), "some belief should have moved"
    ls = learning_summary(res.trajectory, names[3], space, rules)
    assert set(ls[1.0]) == {"precision", "recall", "adopted"}
    assert res.card.scaffolds[names[3]].agent_class.startswith("norms:bayesian_learner")


def test_numpy_and_memo_learners_agree():
    spec = make_spec(ROLES, ascii_map=SMALL_MAP, horizon=12)
    names = agent_names(ROLES)
    space = rule_space(ROLES)
    rules = default_rules(ROLES)
    out = []
    for use_memo in (False, True):
        agents = {n: NormPlanner(n, spec.layout, names, rules, seed=i) for i, n in enumerate(names[:3])}
        learner = NormLearner(names[3], spec.layout, names, space, seed=3, use_memo=use_memo)
        agents[names[3]] = learner
        run_episode(spec, agents, seed=5)
        out.append(np.array(learner.beliefs))
    assert np.allclose(out[0], out[1], atol=1e-4)
