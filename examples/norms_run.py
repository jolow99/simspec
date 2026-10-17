"""Run the norms gridworld: experienced planners that follow the five default norms plus one
Bayesian learner (memo-backed posterior) and report learning and social-outcome metrics.

  python examples/norms_run.py --steps 300 --seeds 1
  python examples/norms_run.py --map small --steps 100 --no-norms       # social outcomes without norms
  python examples/norms_run.py --learner llm --model fake               # an LLM agent in the learner's seat (fake = scripted)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simspec import run_episode, check_spec, replay_episode
from simspec.games.norms import make_spec, default_rules, rule_space, agent_names, NormPlanner, NormLearner, PAPER_MAP, SMALL_MAP, ACTIONS
from simspec.games.norms.metrics import learning_summary, social_outcomes


def llm_learner(role, args):
    from simspec.agents.llm import LLMAgent, ScaffoldConfig, FakeTransport, CassetteTransport, make_transport
    from simspec.games.norms.prompts import make_renderer, scaffold
    cfg = scaffold(role, args.model, args.provider)
    if args.model == "fake":
        transport = FakeTransport(lambda m, p: '{"action": "FORWARD"}', model="fake")
    else:
        os.makedirs("runs/cassettes", exist_ok=True)
        transport = CassetteTransport(f"runs/cassettes/norms_{args.provider}_{args.model}_{role}.jsonl", make_transport(args.provider, args.model), mode="auto")
    return LLMAgent(role, cfg, transport, renderer=make_renderer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="paper", choices=["paper", "small"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--roles", default="cleaner,farmer,free,free")
    ap.add_argument("--no-norms", action="store_true")
    ap.add_argument("--learner", default="bayes", choices=["bayes", "bayes-numpy", "planner", "llm", "decision"])
    ap.add_argument("--wire", default="typesafe", choices=["typesafe", "openai"], help="decision-model wire format")
    ap.add_argument("--model", default="fake")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--out", default="runs/norms")
    args = ap.parse_args()
    roles = args.roles.split(",")
    ascii_map = PAPER_MAP if args.map == "paper" else SMALL_MAP
    rules = [] if args.no_norms else default_rules(roles)
    spec = make_spec(roles, ascii_map=ascii_map, horizon=args.steps, rules=rules)
    names = agent_names(roles)
    space = rule_space(roles)
    for seed in range(args.seeds):
        agents = {n: NormPlanner(n, spec.layout, names, rules, seed=i) for i, n in enumerate(names[:-1])}
        last = names[-1]
        if args.learner == "planner":
            agents[last] = NormPlanner(last, spec.layout, names, rules, seed=len(names))
        elif args.learner == "llm":
            agents[last] = llm_learner(last, args)
        elif args.learner == "decision":
            from simspec.agents.decision import DecisionModelAgent, DecisionConfig, FakeDecisionTransport, make_decision_transport
            from simspec.games.norms.prompts import make_renderer
            cfg = DecisionConfig(model=args.model, wire=("fake" if args.model == "fake" else args.wire), n_permutations=2,
                                 instructions_template="You are {role} in a gridworld with apples, a river to clean and territories. {call_to_action}")
            transport = FakeDecisionTransport({a: (0.5 if a == "FORWARD" else 0.5 / (len(ACTIONS) - 1)) for a in ACTIONS}) if args.model == "fake" \
                else make_decision_transport(args.wire, args.model)
            agents[last] = DecisionModelAgent(last, cfg, transport, renderer=make_renderer())
        else:
            agents[last] = NormLearner(last, spec.layout, names, space, seed=len(names), use_memo=(args.learner == "bayes"))
        if seed == 0:
            print("spec check:", check_spec(spec, agents, rollout_steps=3)["dynamic"])
        run_dir = os.path.join(args.out, f"{args.map}_{args.learner}_seed{seed}")
        res = run_episode(spec, agents, seed=seed, run_dir=run_dir)
        so = social_outcomes(res.trajectory)
        print(f"seed {seed}: cumulative reward {so['cumulative_reward']:.2f}, desiccated {so['desiccated_final']:.3f}, violations {so['violations']} -> {run_dir}")
        if args.learner.startswith("bayes"):
            ls = learning_summary(res.trajectory, last, space, rules)
            for f in (0.25, 0.5, 0.75, 1.0):
                print(f"  at {int(f*100)}%: precision {ls[f]['precision']:.2f} recall {ls[f]['recall']:.2f} adopted {len(ls[f]['adopted'])}")
            print("  final beliefs in the true norms:", ls["final_beliefs_true_rules"])
        replay_episode(lambda params: make_spec(params["roles"], ascii_map=ascii_map, horizon=params["horizon"], rules=rules), run_dir)
        print("  replay-by-readback from card + log: OK")


if __name__ == "__main__":
    main()
