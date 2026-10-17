# simspec

A typed spec language and thin runtime for LLM multi-agent game studies, with memo-based rational baselines.
Write a spec, not a harness.

The contributions are (1) a typed, Python-embedded **game spec**; (2) **enforced observation structure** (agents cannot read state they are not declared to observe, and a planted leak is caught before any model call); (3) **`Rule`** as a first-class object (contracts and norms with compliance checks, sanctions and a hypothesis-space enumerator); (4) **memo-based rational baselines** that play through the same agent interface as LLM agents; (5) a **standard trajectory log** plus a **Harness Card** that makes every run reproducible and comparable. The runtime around them is deliberately small.

Research and design records: `docs/research.md` (Phase 0 findings, verified against sources), `docs/design.md`, `docs/decisions.md` (every design choice and fidelity deviation), `docs/adapters.md`.

## Install and test

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,llm]"          # memo-lang>=1.3.2, jax, numpy, pydantic, httpx, pytest
python -m pytest -q                  # 52 tests, ~5 s, no API keys needed
```

simspec builds on [memo](https://github.com/kach/memo) (MIT), installed from PyPI as `memo-lang`.

## The two games as specs

Both games are expressed in the same language: a `StateSchema` of typed fields with visibility, per-role `View`s (the only thing agents ever see), `Phase`s, a pure `transition`, `Rule`s, a trajectory payoff.

**ContractSim** (`simspec/games/contractsim/spec.py`, Sajja et al. 2026). Two roles, a 50-round negotiation protocol over free-text proposals with the paper's asymmetric accept rule, then eleven alternating payment / order / produce weeks with spoilage, stochastic prices and receipts, inventory caps and pay-for-ordered. The negotiated contract is parsed into a closed schema and becomes two absorbing `constraint` rules with grim trigger as a `release` sanction.

```python
StateSchema(fields=[
    Field(name="week", type=Week, visibility="public"),
    Field(name="budget_remaining", type=Money, visibility="private", owner="customer"),
    Field(name="inventory", type=Table(index=Ingredient, cell=IntRange(0, 10)), visibility="private", owner="supplier"),
    Field(name="prices", type=Table(index=Ingredient, cell=IntRange(1, 3)), visibility="private", owner="supplier"),
    Field(name="contract_schedule", type=Table(index=SchedIdx, cell=IntRange(0, 30)), visibility="public"),
    Field(name="values", type=Table(index=ProductD, cell=IntRange(0, 50)), visibility="private", owner="customer"),
    Field(name="env_id", type=IntRange(1, 6), visibility="private", owner="supplier"),   # the supplier knows the stochastic laws
    ...])
PHASES = [
    Phase(name="negotiate", kind="protocol", actors=["customer", "supplier"], action={"customer": NegCustomer, "supplier": NegSupplier},
          max_rounds=50, until=lambda s: s["agreed"] or s["terminated"], next=lambda s: "payment" if s["agreed"] else None),
    Phase(name="payment", kind="sequential", actors=["customer"], action=Pay, max_rounds=1, next="order"),
    Phase(name="order", kind="sequential", actors=["supplier"], action=Order, max_rounds=1, next="produce"),
    Phase(name="produce", kind="sequential", actors=["supplier"], action=Produce, max_rounds=1, next=lambda s: "payment" if s["week"] <= 11 else None),
]
views = {"customer": View(role="customer", fields=COMMON_PUBLIC + ["budget_remaining", "values"], channels=["dialogue", "proposals"]),
         "supplier": View(role="supplier", fields=COMMON_PUBLIC + ["balance", "inventory", "prices", ...],
                          derived=[Derived(name="laws", reads=["env_id"], fn=...)], channels=["dialogue", "proposals"])}
# the contract, once agreed, becomes rules:
Rule(id="C_customer", kind="constraint", scope=["customer"], classifier=PyPred("customer_pays_due", ...), sanction=Sanction(kind="release", to="supplier"))
```

**Norms gridworld** (`simspec/games/norms/spec.py`, Oldenburg & Zhi-Xuan 2024). The paper's 23×18 Melting Pot map, 12 actions, apples with density regrowth throttled by river pollution, cleaning, territory, pay and zap. One simultaneous phase, fully public state, and the rule vocabulary as per-agent derived features.

```python
Rule(id="P1_no_empty_orchard", kind="prohibition", scope=names, action_class="MOVE",
     post=And(args=[IsTrue(feature="HARVESTED"), Cmp(feature="HARVEST_APPLES_AROUND", cmp="<", value=3)]))
Rule(id="P2_no_stealing", kind="prohibition", scope=names, action_class="MOVE", post=IsTrue(feature="HARVESTED_FOREIGN"))
Rule(id="O2_cleaner_cleans", kind="obligation", scope=cleaners, pre=Cmp(feature="DIRT_FRACTION", cmp=">", value=0.3),
     post=Cmp(feature="SINCE_AGENT_LAST_CLEANED", cmp="==", value=0), deadline=20)
rule_space(roles)   # 57 candidate norms enumerated from the paper's feature grid by RuleSpace
```

Run them:

```
python examples/contractsim_run.py --env 2 --customer llm --supplier memo --model fake    # scripted "LLM", no key
python examples/norms_run.py --map paper --steps 300                                       # planners + memo-backed Bayesian learner
python scripts/shared_code.py                                                              # code-sharing report
```

Every run writes `runs/<name>/{card.json, events.jsonl, calls.jsonl, payoffs.json}` and is replayed from its card and log before the script exits.

## One agent interface

`Agent.act(observation, action_spec, ctx) -> ActionDistribution` (probabilities over the finite action set, with `source ∈ {exact, scripted, native, logprob, verbalized, degenerate, replay}`). The runtime samples with the episode RNG and logs both the distribution and the sampled action. Adapters: `ScriptedAgent`, `LLMAgent` (declared scaffold: templates, context policy, decoding, elicitation, parser and retry policy, distribution source; transports for OpenAI-compatible and Anthropic APIs with a record/replay cassette), `DecisionModelAgent` (TypeSafe `/v1/systemone` and OpenAI `/v1/decisions` wire formats, averaging over option permutations, with a guard that flags providers that return no distribution), memo agents (`build_memo_module`, `TablePolicyAgent`, and the game-specific `MemoTrustAgent` and `MemoRulePosterior`), `ReplayAgent`.

## Success criteria

**(a) Leak test: pass.** `tests/test_leaks.py` and the two game test files plant leaks (a view listing a private field of another role; a derived feature reading an undeclared field; an agent smuggling a reference to the live state) and assert that `check_spec` raises `ObservationLeak` with zero model calls made. Enforcement is by construction: agents receive only an `Observation` built from the declared view; derived features run on a tracing proxy; the state vault is locked during every `act`; `check_spec` dry-runs the whole pipeline with model transports stubbed.

**(b) Baseline test: informative, and negative in the direction the paper reports.** With no intrinsic violation cost, the memo rational customer defects on the final week-11 payment in every environment (it is pure trust: no delivery follows), gaining exactly that payment over RCC. The memo supplier complies fully when the contract's final payment makes the last delivery enforceable, and skips the last delivery when it does not. Against a non-engaging counterparty both stop after one unanswered turn. In the high-stochasticity environment the default contract is not satisfiable with probability 0.95 even for RCC, so everyone "defects". Table from `runs/baseline_table.json` (3 seeds each; U = utility, def = any violation, uni = unilateral):

| env | customer | supplier | U_c | U_s | def_c | def_s | uni_c | uni_s |
|---|---|---|---|---|---|---|---|---|
| 1 | rcc | rcc | 225 | 100 | 0 | 0 | 0 | 0 |
| 1 | memo | rcc | 242 | 83 | 1.0 | 0 | 1.0 | 0 |
| 1 | rcc | memo | 225 | 100 | 0 | 0 | 0 | 0 |
| 1 | memo | re | 183 | 17 | 1.0 | 1.0 | 0 | 1.0 |
| 2 | memo | rcc | 231 | 104 | 1.0 | 0 | 1.0 | 0 |
| 5 | rcc | rcc | 221 | -12 | 0.67 | 0.67 | 0 | 0.67 |
| 5 | memo | memo | 193 | 15 | 1.0 | 1.0 | 0 | 1.0 |

This is the comparison the paper makes about LLMs ("they beat RCC utility only by defecting"), now with an exact-inference agent whose abstraction and prior are declared in the Harness Card; with `--violation-cost` above the final payment the memo customer complies. Running the same table against real LLM agents needs a key (below).

**(c) Card test: pass.** `replay_episode(spec_factory, run_dir)` rebuilds the spec from `card.json`, checks the spec hash, replays logged actions and chance draws, and asserts every state snapshot and the payoffs match. It runs in the test suite and at the end of both example scripts.

**Code sharing.** Static: the shared library is 1122 code lines; ContractSim adds 848 (shared fraction 0.57), the norms game 503 (0.69). Dynamic (lines executed in an episode): 64% shared for ContractSim, 63% for the norms game. The irreducible game-specific parts are the simulators themselves and the memo baselines.

**Norms learning.** On the paper map with three experienced planners and one Bayesian learner (memo posterior), 300 steps: recall 0.4 of the five true norms with the paper's over-generalisation artefact (the learner also adopts "don't move when the river is dirty", exactly the silly norm the paper reports), precision 0.10. Numbers vary by seed; the planner is a BFS-value approximation of the paper's RTDP, see `docs/decisions.md` D-012.

## Running with real models

Set a key and pick a provider; cassettes under `runs/cassettes/` record every call so re-runs replay for free.

```
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, OPENROUTER_API_KEY
python examples/contractsim_run.py --env 5 --customer llm --supplier rcc --model gpt-4.1-mini --provider openai --seeds 3
python examples/contractsim_run.py --env 5 --customer llm --supplier llm --model claude-sonnet-4-5 --provider anthropic
python examples/norms_run.py --map small --steps 60 --learner llm --model gpt-4.1-mini --provider openai
export TYPESAFE_API_KEY=...      # decision models (choice questions over the 12 gridworld actions)
python examples/norms_run.py --map small --steps 60 --learner decision --model jev-latest --wire typesafe
```

Two LLM agents negotiate through the protocol phase themselves (the prompts require the structured JSON block; the deterministic translator parses it). With a scripted or memo counterparty the contract is fixed by a `FixedContractNegotiator`.

## Layout

```
simspec/            domains, spec, rules, chance, log, card, runtime, agents/, games/, adapters/, export/
simspec/games/contractsim   params, contract (schema, translator, clause rules), spec, baselines, memo_baseline, prompts, metrics
simspec/games/norms         world, spec (features, default norms, rule space), agents (planner, learner, memo posterior), prompts, metrics
examples/            contractsim_run.py, norms_run.py
scripts/             shared_code.py
docs/                research.md (+ research/), design.md, decisions.md, adapters.md
tests/               52 tests incl. leak and card tests
```
