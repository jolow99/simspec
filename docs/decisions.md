# Design decisions log

Running record of design choices and why. Newest at the bottom. Each entry: context, decision, alternatives rejected, status.

## D-001 (2026-10-16, Phase 0): memo is the per-decision oracle, not the simulator
- Context: memo compiles nested enumerative inference to JAX arrays over statically known finite axes. It has no sampling, mutable state, loops, strings, or runtime-variable agent sets (verified in `memo/core.py`, `memo/codegen.py`; see `docs/research.md` §1).
- Decision: the runtime owns the game loop, RNG, logs, free-text slots and phase structure in plain Python. memo is called at decision points with the agent's finite information state as an axis index, returning policy tables and beliefs. Spec types must export integer-coded domains and jittable transition/payoff/observation functions so memo source can be generated.
- Rejected: writing the game loop inside memo via recursion over `t` with enumerated histories. Works for toy games (`demo-i-pomdp`) but blows up combinatorially and cannot host negotiation or free text.
- Status: decided.

## D-002 (2026-10-16, Phase 0): depend on `memo-lang>=1.3.2` from PyPI, not the vendored 1.2.3
- Context: this repo is a fork of kach/memo pinned at 1.2.3, which fails on JAX 0.11 (`jax.lib.xla_extension` removed). Upstream 1.3.2 (2026-10-14) fixes this and adds array parameters, `forgets_about`, multiple returns, keyword args, `has_theory` variants.
- Decision: the new library declares `memo-lang>=1.3.2` as a dependency and does not modify memo. Where the library lives (this repo vs separate) is pending the user's answer (`docs/research.md` §9).
- Rejected: patching the vendored copy (that is forking memo).
- Status: decided; location pending.

## D-003 (2026-10-16, Phase 0): generated memo modules keep tables in module globals, scalars as parameters
- Context: memo 1.3 array ("exotic") parameters cannot be combined with `@memo(cache=True)` (`functools.cache` needs hashable arguments), and cached recursion is required for value iteration (`docs/research/experiments/exp2.py`).
- Decision: when the library generates memo source for a game instance, transition/payoff/observation tables are installed as globals of a fresh `make_module` module; only scalars (horizon, temperature, level) are memo parameters. One module per game instance.
- Status: decided; revisit if upstream makes exotic parameters hashable.

## D-004 (2026-10-16, Phase 0): one agent interface, distributions in, probabilities logged
- Context: decision-model APIs (Jev, OpenAI Decisions) return full distributions for choice questions; Anthropic/Google adapters do not; memo returns exact softmax policies; scripted agents are deterministic.
- Decision: `Agent.act(observation, action_spec) -> distribution over the finite action set` plus metadata (`native_distribution: bool`, model version, latency, usage, raw). Sampling from the distribution is done by the runtime with the episode RNG so that replay-by-readback is exact. Agents without a native distribution return a degenerate or verbalised distribution flagged as such.
- Status: decided.

## D-005 (2026-10-16, Phase 0): log format follows Inspect/clembench, not Concordia
- Context: see `docs/research.md` §4, §6, §7.
- Decision: one JSONL event stream per episode (typed events: observation, agent call with prompt/raw output/probabilities/usage, action with parse status, state patch, payoff, rule event, chance outcome) plus periodic full snapshots; raw model calls in a separate file; scores in a separate file written by offline metrics. Chance outcomes are logged so replay-by-readback is exact; replay-by-recompute is a separately labelled mode.
- Rejected: Concordia-style per-component `{Key, Summary, Value, Prompt}` dumps; LangGraph-style replay that re-executes agents.
- Status: decided; concrete schema in `docs/design.md` (Phase 1).

## D-006 (2026-10-16, amendment): custom thin runtime, Inspect AI as an export only, no LangGraph
- Context: the amendment asks to build runtime pieces only where existing tools fail. Both were probed hands-on (`docs/research.md` §6c, `docs/research/experiments/eval_inspect.py`, `eval_langgraph.py`).
- Decision: the loop, state vault, chance logging, JSONL event log, snapshots, replay-by-readback and forks are implemented in `simspec.runtime` (small, dependency-free). `simspec.export.inspect_log` writes an Inspect `EvalLog` (one sample per episode, custom events, Harness Card in metadata) when `inspect-ai` is installed. LangGraph is not used.
- Why not Inspect as primary: 100-package, 339 MB stack; sample model is single-model; our unit is an episode with N seats and joint actions; logs must stay plain JSONL. Why not LangGraph: replay re-executes agents; `update_state` forks overwrote the original thread in the probe; the serializer changed types inside state; the loop it replaces is about 200 lines.
- Status: decided.

## D-007 (2026-10-16, amendment): Harness Card = GAME SPEC + AGENT SCAFFOLD, semantic-hash only
- Context: arXiv 2605.23950 (ETCSOVG card), Harbor/ATIF, HAL hurdles, AgentCompass semantic-vs-execution split, General Agent Evaluation App. G (record provider defaults verbatim).
- Decision: `HarnessCard` pydantic model with `game`, `scaffolds[seat]`, `runtime` sections; every field tagged semantic or execution; `card_hash` over semantic fields; saved as `card.json` in every run directory and referenced by hash in `EpisodeStart`. Decoding parameters are recorded verbatim with a `provider_default` flag per parameter. Distribution source (`exact|scripted|native|logprob|verbalized|degenerate`), option ordering and shuffling, parse-failure and illegal-action policies are mandatory scaffold fields.
- Status: decided.

## D-008 (2026-10-16): pydantic v2 for spec objects, cards and events
- Context: cards, rules and events need schema'd JSON with hashing; stdlib dataclasses would need a hand-written (de)serialiser for nested unions.
- Decision: pydantic v2 (already a transitive dependency of most LLM SDKs). Runtime hot-path records (State, Observation) are plain frozen dataclasses/dicts, not pydantic, for speed.
- Status: decided.

## D-009 (2026-10-16): the repository
- Context: the user allowed deleting the memo fork and working from scratch.
- Decision: the vendored memo 1.2.3, demos and assets are removed; this repo is now the `simspec` project depending on `memo-lang>=1.3.2` from PyPI. History is kept.
- Status: done.

## D-010 (2026-10-16): ContractSim fidelity choices
- Contracts: the negotiated artefact is free text, as in the paper; the structured schema is produced by a pluggable `Translator`. The default translator is deterministic and expects a JSON block in the contract text (the prompts require it). An LLM translator with the same interface (the paper's Gemini step) can be plugged in; it is not needed for scripted or memo agents.
- Clause semantics implemented in the transition's accounting (substitution credit at 4:2:1 above minimums, payment deduction = unit-price value of the shortfall with two consecutive shortfalls a violation, rollover with cap 2 and cure-next-round, grim trigger as a `release` sanction that frees the victim from its obligations). Explicit grim trigger only; the paper's implicit solver fallback is replaced by the RCC agents' own conditional behaviour.
- Horizon, alternating weeks, production-week order (spoilage, observe prices, order with cash cap, receipt, inventory cap, produce and deliver, redraw prices), pay-for-ordered, caps 10/12, six environments: as in the paper.
- Payment/order/produce are `record` actions (integer amounts and vectors), not enumerated choices, because the domains are large; legality is a `valid` predicate (budget, affordability, feasibility).

## D-011 (2026-10-16): ContractSim rational baselines
- RC Customer: exact (pay the due amount after authorised deductions). RE: non-engagement, as in the paper. RCC: RC until an observed counterparty violation, then RE.
- RC Supplier: a P_sat-first safety-stock heuristic (order the smallest quantity whose receipt law covers the scheduled need with probability ≥ 1−ε, subject to cash and caps) with the paper's heuristic production rule. The paper solves an exact constrained MDP by backward DP; the state space (inventory 11^3 × cash × memory × prices) is too large for memo's dense joint arrays and the exact DP in JAX is deferred. The difference is declared in the Harness Card (`solver.note`).
- memo baseline (`MemoTrustAgent`): both roles are modelled as a finite-horizon belief-space POMDP against a counterparty of hidden type (complier | non-engager) with a binary comply/defect decision per turn and an optional intrinsic violation cost. memo's POMDP pattern (belief grid, `snapshots_self_as`, `imagine`) solves it exactly; the agent updates its belief from public observations only. This is an abstraction of the paper's RC/RE/RCC, not a reproduction: its value is that it predicts *when* a rational agent defects (unenforceable final deliveries, low prior trust, low surplus), which is exactly the behaviour the paper measures in LLMs.
- Contract value and P_sat are estimated by simulating RCC × RCC episodes rather than by exact DP.

## D-012 (2026-10-16): norms gridworld fidelity choices
- Environment: the paper's ASCII map (23×18 playable), 12 actions, roles by appearance, apples with the paper's density regrowth probabilities and pollution throttle, desiccation, river dirt spawning on potential cells, straight-line beams for clean/zap/pay/claim, 5×5 territory at spawn, steal records, zap freeze and penalty. Simplifications: 4-neighbour regrowth neighbourhood, one dirt spawn per step with probability 0.2, movement conflicts resolved in agent order, no mortality/rebirth (the intergenerational experiment is out of scope).
- Rules: prohibitions are evaluated on the *effect* of the action (features of the next state such as `HARVESTED`, `HARVEST_APPLES_AROUND`, `HARVESTED_FOREIGN`) rather than on a predicted destination cell, which is the same semantics stated in the paper (`<Prohib, Post>` with Post on s'). Obligations carry their deadline (20) on the rule instead of a global constant. Obligations are role-conditioned by scope rather than by an `AGENT_LOOK ==` conjunct.
- Hypothesis space: generated from the paper's feature grid by `RuleSpace`; 57 rules here (21 prohibitions, 36 role-conditioned obligations) vs 68 in the paper, because `RIOTS` (sanctions were disabled in the paper's experiments) and 3-feature prohibitions are omitted.
- Planner: one-step forward model plus BFS-distance values instead of RTDP; obligations switch the goal; prohibitions mask actions and cost 1. Learning: mean-field two-hypothesis Bayes with the paper's bounded-compliance mixture (`p_obey` 0.9, temperature 1, prior 0.05, threshold 0.95), the likelihood built from the planner's Q with and without the rule. The per-rule posterior update is computed by a generated memo program (`MemoRulePosterior`), vectorised over rules and actions, with the likelihood tables passed as memo 1.3 array parameters; a numpy fallback is kept for comparison.
