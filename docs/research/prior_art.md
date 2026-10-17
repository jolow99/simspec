# Prior art for a typed multi-agent game spec + thin runtime + replayable trajectory log

Research date: 2026-10-16. Scope: frameworks evaluated as *references* (concepts to borrow), not dependencies.
Verification level is marked per item: **[docs]** = read official docs/source, **[search]** = only search snippets / abstracts, **[local]** = inspected the local `memo` checkout.

Project framing recap: (1) a typed, Python-embedded spec language for multi-agent games (state, actions, per-agent observation structure, transition, payoffs, first-class Rules/norms/contracts); (2) a thin runtime that executes a spec with pluggable agents (LLM, decision-model, memo-based rational, scripted) and writes a standard replayable trajectory log; metrics/baselines are computed offline from logs.

---

## 0. Executive summary

- **Cleanest formalization of "per-agent observation + finite action set + sequential/simultaneous turns": OpenSpiel's `State`/`GameType` model** (explicit `kSimultaneousPlayerId`, `kChancePlayerId`, `apply_actions` for joint moves, `legal_actions(player)`, `information_state_string` vs `observation_string`, `rewards()` vs `returns()`, `history()`/`serialize()`). PettingZoo's AEC/Parallel split is the cleanest *runtime loop* formalization (`agent_iter` / `last()` vs `step(actions: dict)`), and its `possible_agents` vs `agents` distinction plus `action_mask` convention are worth copying verbatim. Gymnasium `Space` objects are overkill for finite typed action sets but their `sample(mask=)`, `contains`, `to_jsonable/from_jsonable` and `seed` methods are the right method set for our typed domains.
- **Best log-format reference: Inspect AI's EvalLog / event transcript.** Specifically: a per-sample *event list* with typed events (`ModelEvent` carrying `input`, `tools`, `config`, `output`, `call` (raw request/response), `cache`, `retries`; `StateEvent`/`StoreEvent` as **JSON-patch diffs**; `ScoreEvent`; `SpanBegin/End`), `ModelUsage`, `GenerateConfig.logprobs/top_logprobs`, and a header-only / sample-streaming reader API. Also clembench's `interactions.json` + `requests.json` + `scores.json` split (game events vs raw LLM calls vs scores, joined by timestamp) is a very good small-scale model.
- **LangGraph** is worth borrowing only in the small: the *shape* of a checkpoint (`channel_values`, `channel_versions`, `versions_seen`, parent pointer, `metadata.step/source`), the `thread_id` + `checkpoint_id` addressing scheme, the "fork = `update_state` on a historical checkpoint" semantics, and the warning that *replay re-executes nodes (LLM calls fire again) unless cached*. Its Pregel/reducer graph model is a poor fit for a game loop and its interrupt mechanism re-runs the node from the top, which is wrong for a turn-based game.
- **Concordia v2 (DeepMind)** is the closest *conceptual* sibling for LLM social simulation: an `Engine` with `Sequential` / `Simultaneous` / `Asynchronous` variants, a Game Master that implements `make_observation`, `next_acting`, `resolve`, `terminate`, and a typed `ActionSpec(call_to_action, output_type ∈ {FREE, CHOICE, FLOAT}, options, tag)`. Its weakness for us: state is free text adjudicated by an LLM GM; no typed state/transition/payoff.
- **Missing from the 2024–2026 landscape:** nobody has a *typed, declarative game spec* with first-class norms/contracts that is decoupled from a particular agent class and produces a standard log. Every benchmark (GovSim, NegotiationArena, GTBench, MAgIC, SOTOPIA, TextArena, clembench, MARBLE, Welfare Diplomacy, LLM-Deliberation, CoopEval, GLEE…) re-implements the harness listed in §6. The only cross-cutting "trajectory format" efforts (Harbor ATIF, NVIDIA ATOF) are single-agent tool-use-centric and have no notion of simultaneous moves, per-agent observation, or payoffs.
- **memo:** the local `demo/` folder shows how sequential decision problems are modelled (recursion over a horizon parameter `t` inside the memo program: `demo-pomdp`, `demo-i-pomdp`, `demo-takeaway`), and the parent session's `scratchpad/exp_loop.py` already demonstrates the pattern we need: memo as a *one-step policy / Q-function* called from a Python environment loop, with the observation history encoded as an axis value. No one has published a reusable environment loop around memo (nearest: a 2026 paper pairing memo with JAX2d, abstract only).

---

## 1. LangGraph (langchain-ai/langgraph) **[docs]**

Sources: docs.langchain.com `persistence`, `checkpointers`, `graph-api`, `use-time-travel` pages (the old langchain-ai.github.io URLs now redirect).

### What it is
A Pregel-style graph runtime: nodes are Python functions `(state, config, runtime) -> partial update`; the state schema is a `TypedDict` / dataclass / Pydantic model; each key may have a **reducer** (`Annotated[list, operator.add]`, `add_messages`), otherwise last-write-wins. Execution proceeds in **super-steps**; nodes activated in the same super-step run in parallel and their updates are merged by reducers. `Send(node, state)` fans out; `Command(update=, goto=, graph=, resume=)` combines state update + routing. `Overwrite` bypasses a reducer; `UntrackedValue` marks state never checkpointed.

### Checkpointing
- `BaseCheckpointSaver`: `put`, `put_writes`, `get_tuple`, `list`, `delete_thread` (+ `a*` async variants).
- A **Checkpoint** is stored once per super-step: `channel_values`, `channel_versions`, `versions_seen`, an `id` (ULID), plus a parent link. `CheckpointMetadata`: `source` ∈ {`input`,`loop`,`update`}, `step`, `writes`.
- `CheckpointTuple(config, checkpoint, metadata, parent_config, pending_writes)`.
- Addressing: `config["configurable"] = {thread_id, checkpoint_ns, checkpoint_id}`; `checkpoint_ns` is `""` for root, `"node:uuid"` for subgraphs, nested with `|`.
- `StateSnapshot(values, next, config, metadata, created_at, parent_config, tasks[PregelTask(id, name, error, interrupts)])`.
- Durability modes `"exit" | "async" | "sync"`; **pending writes**: each node's output is saved as a task-level write as soon as it finishes, so a failed sibling doesn't force re-running successful nodes on resume. Time travel can only resume from full super-step checkpoints.
- Serializer: `JsonPlusSerializer` (ormsgpack + JSON, `pickle_fallback`), `EncryptedSerializer`.

### Time travel / replay
- `graph.get_state_history(config)` → snapshots newest-first; select by `next == ("node",)`.
- **Replay**: `graph.invoke(None, snapshot.config)` skips nodes before that checkpoint (their results are stored) and **re-executes** nodes after it. The docs are explicit that LLM calls fire again and may differ; there is no built-in replay-from-cache.
- **Fork**: `graph.update_state(snapshot.config, values=..., as_node=...)` writes a new branch; original history stays intact.
- Subgraphs count as one step unless `checkpointer=True` on the subgraph.

### Interrupt / HITL
`interrupt(value)` inside a node pauses at the super-step boundary; `Command(resume=value)` makes it the return value. **The interrupted node re-runs from the top on resume**, so side effects before the interrupt must be idempotent.

### Verdict for a game loop with replay
Borrow:
1. The *checkpoint-as-versioned-channels* idea only insofar as: **a game log should contain an immutable, addressable state snapshot per step (`step`, `parent`, `source`), so that any step can be used as a fork point.** `thread_id` ≈ our `episode_id`; `checkpoint_id` ≈ `(episode_id, step)`.
2. "Fork = write a modified state at historical step N and continue" as the formal semantics of counterfactual replay (useful for norm/contract ablations: *replay from step 7 with a different contract in force*).
3. The explicit distinction between **replay-by-recompute** (agents are called again; nondeterministic) and **replay-by-readback** (log is sufficient). LangGraph only has the former; we need both, so the log must record agent outputs, not just state.
4. Pending-writes semantics as a reminder that in a simultaneous-move round, each agent's action should be persisted as soon as it is produced, before the transition is applied, so a crash mid-round doesn't re-query completed agents.

Do not borrow:
- The Pregel/reducer model. A game loop is `s_{t+1} = T(s_t, a_t, ω_t)`; reducers-per-key encourage implicit, order-dependent merging, which is exactly what a game spec must forbid.
- `interrupt()` re-execution semantics. For a human-in-the-loop *agent*, the right model is "an agent whose `act()` blocks", not re-running the step.
- TypedDict + `Annotated` reducers as the state typing scheme; a frozen dataclass / Pydantic model with an explicit `transition` is clearer and serialises deterministically.

---

## 2. Inspect AI (UK AISI, `inspect_ai`) **[docs]**

Sources: inspect.aisi.org.uk `eval-logs`, `reference/inspect_ai.event`, `reference/inspect_ai.model`, `agents`, `agent-custom`, `agent-bridge`, `sandboxing`, `caching`. (The `agent-multi.html` URL I tried 404'd; the multi-agent material below is from `agents`/`agent-custom`.)

### Model
`Task(dataset, solver, scorer)`; `Sample(id, input, target, metadata, files, setup, sandbox)`; solvers transform `TaskState`; scorers produce `Score(value, explanation, metadata)`; metrics aggregate. Agents are `async (state: AgentState) -> AgentState` with `AgentState.messages`/`.output`; `react()` is the builtin tool loop; `as_solver()`, `as_tool()` (string-in/string-out, isolated context), `handoff()` (transfers the full conversation), `run()`. Limits: token/message/time/working/turn/cost. Shared state across agents: `StoreModel` + `store_as(instance=...)` (named instances e.g. `"red_team"`). `transcript().info(...)`, `span()` for grouping, `collect()` for concurrency.

### EvalLog format
- Top level: `version` (2), `status` ∈ {started, success, error, cancelled}, `eval` (EvalSpec: task, model, created…), `plan` (solvers + generate config), `results`, `stats` (model usage), `error`, `tags`, `metadata`, `log_updates` (append-only provenance of post-hoc edits), `samples`, `reductions`.
- `EvalSample`: `id`, `epoch`, `input`, `target`, `messages`, `output` (`ModelOutput.response_id`), `scores{name: Score(value, explanation, metadata, history)}`, `metadata`, `store`, `events`, `model_usage`, `attachments` (de-duplicated content), `error`, `limit`.
- Two encodings: `.eval` (binary, zstd, ~1/8 the size of JSON, incremental sample reads; default since 0.3.46) and `.json`. `inspect log convert`. Since 0.3.206 repeated messages are de-duplicated (~10:1 on agentic logs).
- Reader API: `read_eval_log(path, header_only=True)`, `read_eval_log_sample(path, id, epoch)`, `read_eval_log_samples(path)` (generator), `read_eval_log_sample_summaries`, `write_eval_log(log, if_match_etag=)`, `resolve_sample_attachments`, `edit_score`, `recompute_metrics`. Location is an fsspec URI (`file://`, `s3://`).

### Event transcript (the part worth copying)
Common fields on every event: `uuid`, `span_id`, `timestamp`, `working_start`, `metadata`, `pending`. Event types:
- `SampleInitEvent(sample, state)`; `SampleLimitEvent(type ∈ message/time/working/token/turn/cost/operator/custom, message, limit)`.
- **`StateEvent(changes: list[JsonChange])` and `StoreEvent(changes)`** — state is logged as JSON-patch diffs, not full copies.
- **`ModelEvent(model, role, requested_model, input: list[ChatMessage], input_refs, tools, tool_choice, config: GenerateConfig, output: ModelOutput, retries, request_ids, error, cache ∈ read/write, call: ModelCall(request, response, time, call_key), completed, working_time)`**.
- `ToolEvent(id, function, arguments, view, result, truncated, error, completed, working_time, agent, agent_span_id, failed, message_id, cancelled)`.
- `SandboxEvent(action ∈ exec/read_file/write_file, cmd, options, file, input, result, output)`; `ApprovalEvent(decision ∈ approve/modify/reject/escalate/terminate)`; `InputEvent`; `ScoreEvent(score, target, intermediate, scorer, scorer_args, model_usage)`; `ScoreEditEvent`; `ErrorEvent`; `LoggerEvent`; `InfoEvent(source, data)`; `SpanBeginEvent(id, parent_id, type, name)` / `SpanEndEvent(id)`; `BranchEvent(from_anchor)`; `CompactionEvent`; `InterruptEvent(source, interrupted ∈ generate/tool_call/between_turns)`.
- `GenerateConfig` has `logprobs: bool`, `top_logprobs: int`, `seed`, `temperature`, `cache`; `ChatCompletionChoice(message, stop_reason, logprobs, prompt_logprobs)`; `ModelUsage(input_tokens, output_tokens, total_tokens, input_tokens_cache_write, input_tokens_cache_read, reasoning_tokens, total_cost)`.
- Raw provider request/response payloads are logged for the first few calls per model and all errors; `--log-model-api` logs all.

### Caching / replay
Cache key = model name + base URL + message history + epoch + generate config + tools/tool_choice; `CachePolicy(expiry, scopes={"role": ...}, per_epoch)`. The docs do not call this "replay"; it is deterministic re-runs for development. There is no "replay an EvalLog without calling the model" mode documented.

### Agent bridge / sandboxing
`agent_bridge()` (in-process; the agent uses model name `"inspect"` via its own OpenAI/Anthropic/Google client) and `sandbox_agent_bridge()` (proxy on port 13131 inside the sandbox) route foreign-framework model calls through Inspect so they land in the transcript, with `requested_model`, `subagent_id/type`, `is_root_agent()`. `SandboxEnvironment`: `exec`, `read_file`, `write_file`, `connection`; providers `docker`, `local`, external `k8s`, `modal`, `daytona`, `ec2`, `proxmox`, `vagrant`, `openshell`; per-sample `sandbox`/`files`/`setup`.

### Verdict for our trajectory log
Borrow:
1. **Typed event stream per episode** with a small set of event kinds. Ours: `EpisodeInit`, `Observation(agent, step, obs)`, `AgentCall(agent, step, kind ∈ llm/memo/scripted, input, output, logprobs, usage, call, cache, retries)`, `Action(agent, step, raw, parsed, valid, mask)`, `StateChange(step, changes: JSON patch)` or full snapshot, `Payoff(step, per-agent)`, `RuleEvent(step, rule_id, triggered/violated/sanction)`, `Score`, `Error`, `Span`.
2. `ModelEvent` field set verbatim for LLM agents: `model`, `requested_model`, `input`, `tools`, `config` (incl. `seed`, `temperature`, `logprobs`, `top_logprobs`), `output` incl. `logprobs`, `call` (raw request/response), `cache`, `retries`, `usage`.
3. `StateEvent` as JSON-patch diffs + periodic full snapshots (cheap, diffable, and supports time-travel to any step).
4. Header-only and per-sample streaming reads; `log_updates`/`ScoreEditEvent` provenance for post-hoc score edits (we compute metrics offline, so "scores are mutable annotations on an immutable transcript" is exactly our model).
5. `store_as(instance="agent_name")` as the pattern for per-agent private memory that still lands in the log.
6. Cache-key definition (messages + config + tools + epoch + model@version) for an LLM-call cache that doubles as a deterministic replay source.
7. Agent bridge idea: route *any* external agent framework's model calls through our runtime so they are logged, instead of mandating our agent class.

Do not borrow:
- `Task/Dataset/Solver/Scorer` as the top-level organisation: a *sample* in Inspect is one model's run on one prompt; our unit is an *episode of a game with N agents*, and scoring is per-agent + social. Mapping "game = Task, episode = Sample, agents = multiple models" is possible but awkward (Inspect's `eval.model` is a single model; multi-model is via roles).
- The `.eval` binary container; JSONL/zstd-JSON is enough for us and keeps `jq`-ability. (Do copy the de-duplicated attachments idea if we store long prompts repeatedly.)
- Sandboxing: irrelevant unless agents execute code.

---

## 3. Multi-agent environment suites

### 3.1 PettingZoo **[docs]**
- **AEC API** (`AECEnv`): `reset(seed, options)`, `agent_iter()`, `agent_selection`, `last() -> (observation, reward, termination, truncation, info)`, `step(action)` (pass `None` for a terminated/truncated agent), `observe(agent)`, `observation_space(agent)`, `action_space(agent)`, dicts `rewards`, `terminations`, `truncations`, `infos`; `agents` (live) vs `possible_agents` (fixed, keys of the space dicts), `max_num_agents`. Action masks live in `observation["action_mask"]` (Classic) or `info["action_mask"]` (Shimmy/OpenSpiel).
- **Parallel API** (`ParallelEnv`): `reset -> (observations, infos)`; `step(actions: dict) -> (observations, rewards, terminations, truncations, infos)` all keyed by agent; `state()` for a global view.
- Conversions `aec_to_parallel` / `parallel_to_aec` with the constraint that an AEC env must only update at the end of a cycle. Wrappers: `OrderEnforcingWrapper`, `TerminateIllegalWrapper(illegal_reward)` (illegal move ⇒ game ends, mover loses), `AssertOutOfBoundsWrapper`, `ClipOutOfBoundsWrapper`.
- Borrow: the vocabulary (`possible_agents`, `agents`, `terminations` vs `truncations`, `action_mask`), the explicit statement that observation/action spaces are fixed per agent id for the episode, and the two *views* (sequential cursor vs joint dict) over one underlying game. Our spec should be declared once and the runtime should expose both an AEC-style iterator (for LLM agents acting one at a time with a shared transcript) and a Parallel-style joint step (simultaneous games). Also borrow the *illegal action policy* as an explicit, logged choice (`terminate_loser` / `resample` / `noop` / `reprompt`).
- Don't borrow: Gym-style per-step reward scalars only; we need payoffs as typed, possibly vector/structured values, and terminal payoff vectors; also PettingZoo has no observation *structure* (what is public / private / per-agent) — it is just spaces.

### 3.2 Melting Pot **[docs for README/concepts/using_substrates/scenario.py; TimeStep/Policy details from memory, unverified]**
- `substrate.build(name, roles=[...])` → a `dm_env`-style env; `observation_spec()`, `action_spec()`, `reward_spec()`; roles from `valid_roles` / `default_player_roles`. Scenarios: `scenario.build(name)` assembles a `ScenarioFactory(substrate, roles, bots, bots_by_role, is_focal, permitted_observations)`; background bots fill non-focal slots; `PERMITTED_OBSERVATIONS` whitelists keys (`RGB`, `INVENTORY`, `READY_TO_SHOOT`, `VOTING`, `COLLECTIVE_REWARD`, …); `WORLD.RGB` is the global view. Substrates are DMLab2D grid worlds with Lua `GameObject`/`Component` logic configured from Python dicts.
- I could not verify from source the exact `Policy` bot interface (`initial_state()`, `step(timestep, prev_state) -> (action, state)`), nor the exact per-player list structure of `TimeStep`; those are from memory.
- Borrow: **the focal-vs-background population split** and "scenario = substrate + fixed background policies + held-out evaluation" as our definition of *baseline/scenario*: a game spec plus an assignment of agent classes to seats, with some seats held by scripted/rational background agents. Also the `permitted_observations` whitelist as a tiny model of observation structure. Also "roles" as a first-class seat attribute.
- Don't borrow: anything about pixels, DMLab2D, or evaluation of SavedModels; no log format of interest.

### 3.3 OpenSpiel **[docs: spiel.h, api_reference, concepts.md]**
- `GameType`: `dynamics ∈ {kSequential, kSimultaneous, kMeanField}`, `chance_mode ∈ {kDeterministic, kExplicitStochastic, kSampledStochastic}`, `information ∈ {kOneShot, kPerfectInformation, kImperfectInformation}`, `utility ∈ {kZeroSum, kConstantSum, kGeneralSum, kIdentical}`, `reward_model ∈ {kRewards, kTerminal}`, `min/max_num_players`, `provides_information_state_string/tensor`, `provides_observation_string/tensor`, `parameter_specification`.
- `State`: `current_player()` (or `kSimultaneousPlayerId`, `kChancePlayerId`, `kTerminalPlayerId`), `is_chance_node`, `is_simultaneous_node`, `is_player_node`, `is_terminal`, `legal_actions(player)`, `legal_actions_mask`, `apply_action`, **`apply_actions(joint)`**, `chance_outcomes()`, `rewards()` (last transition) vs `returns()` (cumulative; must agree), `player_return`, `information_state_string(player)` (perfect-recall, unique across players) vs `observation_string(player)` (at most the same info; observation history + own actions must reconstruct the info state), `history()`, `full_history()`, `serialize()`, `clone()`, `undo_action()`, `action_to_string`/`string_to_action`, `resample_from_infostate`. `Game`: `new_initial_state`, `num_players`, `max_game_length`, `get_type`, `make_observer`, `deserialize_state`, `utility_sum`, `min/max_utility`.
- Transforms: `LoadGameAsTurnBased` converts simultaneous → sequential (the same trick PettingZoo's `parallel_to_aec` does).
- Borrow (strongly): **this is the vocabulary our spec should expose as declared metadata** (`dynamics`, `chance_mode`, `information`, `utility`, `reward_model`), the chance player as an explicit seat (so randomness in transitions is logged as chance outcomes and replay is exact), `legal_actions(player)` as the canonical action-mask generator, `rewards` vs `returns`, and the **information-state vs observation distinction** (our per-agent observation structure should make it possible to say which fields are *observation* (what arrives this step) vs *information state* (what the agent could know given perfect recall) — LLM agents get a rendered observation; memo agents often need the information state as an index).
- Don't borrow: flat integer action ids as the *primary* representation (keep typed actions, derive ints), tensors, and C++-first implementation.

### 3.4 Gymnasium spaces **[docs]**
`Box`, `Discrete(n, start)`, `MultiBinary`, `MultiDiscrete`, `Text`, `Dict`, `Tuple`, `Sequence`, `Graph`, `OneOf`; `Space.sample(mask=None, probability=None)`, `contains`, `seed`, `to_jsonable/from_jsonable`, `flatten/unflatten/flatdim`. Borrow the *method set* (`contains`, `sample(mask)`, `seed`, `to_jsonable`) for our typed domains (enum / product / sum types), not the classes.

### 3.5 Which is the cleanest formalization to align with?
OpenSpiel for the **game semantics** (per-player legal actions + explicit simultaneous/chance nodes + information state/observation split + rewards/returns), PettingZoo for the **loop/API shape** (AEC iterator vs Parallel joint step, `possible_agents`, `action_mask`, `terminations/truncations`), Concordia for the **LLM-side action elicitation** (`ActionSpec(call_to_action, output_type, options)`). Melting Pot for the **evaluation protocol** (focal vs background populations). None of the four has (a) typed state/transition declared declaratively, (b) norms/contracts, (c) a replayable LLM-aware log.

---

## 4. LLM multi-agent game / social-simulation / negotiation frameworks (2023–2026)

Format per item: what it is · state/actions/observations · log/replay · relevance.

**Concordia (google-deepmind/concordia, v2, 2025)** [docs: README, environment/README, CHEATSHEET, typing/entity.py]
- Generative agent-based modelling library with a Game Master entity. `Engine.run_loop` with `next_acting`, `make_observation`, `resolve`, `terminate`; engines `Sequential` (turn-based, everyone observes every event), `Simultaneous` (batched; agents don't see others' actions until round end), `Asynchronous`. `Entity.act(action_spec) -> str`, `Entity.observe(str)`; `ActionSpec(call_to_action, output_type ∈ FREE/CHOICE/FLOAT (+ GM-internal MAKE_OBSERVATION/NEXT_ACTING/RESOLVE/TERMINATE…), options, tag)`. Prefabs + `Config(default_premise, default_max_steps, prefabs, instances=[InstanceConfig(prefab, role ∈ ENTITY/GAME_MASTER/INITIALIZER, params)])`.
- State is natural language held by GM components; actions are free text (or choice); observations are strings.
- Log: `simulation.play(return_structured_log=True)` → `SimulationLog` with `to_json()/from_json()`, `AIAgentLogInterface(get_overview, get_entity_actions)`, `concordia-log` CLI, `utils/log_viewer.html`. (I could not fetch `simulation_log.py` to confirm field names.)
- Relevance: high as the conceptual sibling; borrow the engine/GM phase decomposition and `ActionSpec`. Its untyped state is the gap we fill.

**GovSim (giorgiopiatti/GovSim, NeurIPS 2024)** [docs: README + paper]
- Common-pool resource dilemmas (fishery/pasture/pollution), 12 monthly rounds: strategy → concurrent private harvest (random allocation on over-demand) → group discussion (LLM moderator picks speaker) ; regen = ×2 capped at 100, collapse < 5. Generative-agent architecture (memory/reflection/plan). Action = integer parsed after `Answer:`. Hydra configs (`experiment=fish_baseline_concurrent`, `multiple_llm.yaml`, `mix_llm`); `pathfinder` LLM wrapper. Metrics: survival time/rate, total gain, efficiency, equality (Gini), over-usage; 4 "subskill" tests. Web UI links each resource-graph point to the prompt that produced it. No documented log schema.
- Relevance: high as a *target game* (CPR + norms/discussion); harness fully bespoke.

**NegotiationArena (vinid/NegotiationArena, ICML 2024)** [docs: README; tag names not verified]
- Buy/Sell, Ultimatum, Resource Exchange; agents are `ChatGPTAgent(...)` objects holding their own history; `BuySellGame(players, iterations, player_goals, player_initial_resources, player_roles, player_social_behaviour, log_dir)`; games "share a very weak link" (copy, don't inherit). Logs to `log_dir`/`NEGOTIATION_LOG_FOLDER` (format undocumented; `example_logs/` exists). Structured XML-ish tags in messages (from the paper; not verified in code).
- Relevance: medium; a cautionary example of no shared game abstraction.

**GTBench (jinhaoduan/GTBench, NeurIPS 2024)** [docs: README]
- 10 OpenSpiel games (TicTacToe, Connect-4, Kuhn poker, Liar's dice, IPD…) with LLM agents (prompt/CoT/ToT/SC-CoT) vs MCTS/LLM; YAML configs for games/agents/models; `gamingbench.main --num-matches --num-workers --exchange-first-player`; LangChain for inference. Metric: Normalized Relative Advantage. Log schema not documented.
- Relevance: medium-high as proof that OpenSpiel `observation_string` + `legal_actions` is a workable LLM prompt basis; the glue code is what every study rebuilds.

**MAgIC (cathyxl/MAgIC, EMNLP 2024)** [search]
- Chameleon, Undercover, Cost Sharing, PD, Public Good; `arena_runner.py`; OpenAI-only; "PGM" (probabilistic graphical model) prompting. Metrics: judgment, reasoning, deception, self-awareness, cooperation, coordination, rationality. No log format info.
- Relevance: low-medium (metrics menu).

**SOTOPIA (sotopia-lab/sotopia, ICLR 2024)** [docs: README, concepts/agents, concepts/environments, database/logs.py]
- Social scenarios with `AgentProfile`, `EnvironmentProfile`; `AgentAction(action_type ∈ none/speak/non-verbal communication/action/leave, argument, to=[recipients] for private messages)`; `ParallelSotopiaEnv` with `action_order` random / round-robin / simultaneous; response evaluators (`RuleBasedTerminatedEvaluator`) and terminal evaluators (`ReachGoalLLMEvaluator[SotopiaDimensions]`, 7 dims). **`EpisodeLog` (redis-om): `environment`, `agents`, `tag`, `models`, `agent_classes`, `messages: list[list[tuple[sender, receiver, text]]]` (by turn), `reasoning`, `rewards: list[tuple[float, dict[str,float]] | float]`, `rewards_prompt`**; `render_for_humans()`. Storage: Redis or local JSON (`~/.sotopia/data/`).
- Relevance: high for the *shape* of a minimal multi-agent episode log and for the public/private message addressing; weak on state (no typed state).

**OASIS (camel-ai/oasis, 2024–25)** [search]
- Social-media simulator up to 1M agents (X/Reddit), 23 actions (follow/comment/repost…), recommender systems, LLM + rule-based agents, dataset on HF. Logging: not covered by my sources.
- Relevance: low (platform simulation, not games).

**Generative Agents / Smallville (joonspk-research)** [docs: README]
- Django environment server + `reverie.py` backend; steps of 10 s; saved sims in `environment/frontend_server/storage`; `/replay/<sim>/<step>` and `/demo/...` web replay; memory stream/reflection/plan.
- Relevance: low-medium; the *replayable simulation storage + web replay* pattern is the one to emulate, in spirit.

**AgentVerse (OpenBMB/AgentVerse)** [docs: README + rules dir listing; paper]
- Simulation framework where an `Environment` is composed of five rule components: `order/`, `visibility/`, `selector/`, `updater/`, `describer/` (folder names verified; concrete rule names e.g. sequential/random/concurrent not verified). YAML task configs (`simulation/nlp_classroom_9players`, prisoner dilemma, pokemon).
- Relevance: medium — the **decomposition of a game loop into order × visibility × selector × updater × describer** is a good checklist for our runtime's pluggable policies (turn order, who sees what, which messages are kept, state update, observation rendering).

**CAMEL (camel-ai/camel)** [docs: README]
- `ChatAgent.step()`, `RolePlaying` society, `Workforce`; JSON request/response model logs via `CAMEL_MODEL_LOG_ENABLED`/`CAMEL_LOG_DIR`. No typed env/game abstraction documented on the README (there is an RL-ish environments module in the codebase that I did not verify).
- Relevance: low.

**AutoGen (microsoft/autogen)** [docs: README]
- **Maintenance mode; Microsoft Agent Framework is the successor.** Core (event-driven actors) / AgentChat (teams: RoundRobin, Selector, Swarm) / Extensions. Team `save_state/load_state` exists (from memory, not verified on this pass).
- Relevance: low; group-chat is an untyped sequential game with LLM-chosen speaker.

**TextArena (LeonGuertler/TextArena, 2025)** [docs: README + core.py]
- 57–100+ text games, Gym-like: `ta.make(env_id)`, `env.reset(num_players, seed)`, `env.get_observation() -> (player_id, observation)`, `env.step(action: str) -> (done, info)`, `env.close() -> rewards`; `State(num_players, max_turns, current_player_id, game_state, logs, game_info, observations, done, rewards)`, `add_observation(message, observation_type ∈ PROMPT/PLAYER_ACTION/GAME_ACTION_DESCRIPTION/GAME_MESSAGE/GAME_BOARD/GAME_ADMIN, from_id, to_id)`; `Agent.__call__(observation: str) -> str`; `ObservationWrapper/ActionWrapper/RenderWrapper`; TrueSkill leaderboard; MindGames trajectories on HF. No formal replay format beyond `State.logs`.
- Relevance: medium-high — the **observation as a list of `(sender_id, message, type)` entries addressed to a player** is a clean minimal representation of per-agent text observations; the `ObservationType` enum is worth copying. Weak: `game_state` is an untyped dict.

**clembench / clemcore (clp-research)** [docs: clembench README, docs/logging_and_scoring.md, clemcore README]
- Game Master + Players dialogue games; `clemgame.json` registry; `clem run / score / eval / transcribe`. **Per-episode: `interactions.json` (`players{GM, Player 1, ...}`, `turns: [[event]]`, event `{timestamp, from, to, action{type ∈ "send message"/"get message"/"metadata"/"parse"/"error"/"invalid format"/custom, content}}`), `requests.json` (`timestamp`, `manipulated_prompt_obj`, `raw_response_obj`), `scores.json` (`"turn scores"{turn: {...}}`, `"episode scores"`, main score `BENCH_SCORE`, aborted ⇒ NaN)**; `records/<model-pair>/<game>/<experiment>/episode_N/`. Backends incl. `_player_programmed` (scripted) and `slurk` (human).
- Relevance: **high for log design**: the three-file split (game-level events / raw LLM calls / scores) joined by timestamp, the explicit `parse`/`invalid format`/`error` event types, and "aborted vs played vs quality" separation are exactly what offline metrics need.

**MultiAgentBench / MARBLE (ulab-uiuc/MARBLE, ACL 2025)** [docs: README; paper]
- Coordination engine, agent graph (relations: collaborates/supervises/negotiates), protocols star/tree/graph/chain, planners (vanilla/CoT/group discussion/cognitive self-evolving), shared + individual memory, scenarios incl. werewolf and bargaining, LLM-judged milestones → KPI, LLM-rated communication/planning scores. Per-agent event logs mentioned; schema not published.
- Relevance: low-medium (task-solving orientation, LLM-judge metrics).

**Welfare Diplomacy (mukobi/welfare-diplomacy, 2023)** [docs: README]
- Fork of the `diplomacy` engine with Welfare Points; `LLMAgent` + message rounds (default 3); W&B sweeps; CSV results; engine `to_saved_game_format` JSON; "super exploiter" baselines that switch to RL agents. Utility = WP.
- Relevance: medium (general-sum variant + exploiter baselines as a baseline-agent pattern).

**LLM-Deliberation (S-Abdelnabi, 2024)** [docs: README]
- 6 parties × 5 issues negotiation with per-agent score files + thresholds, roles (p1/p2/target/player), incentives (greedy/targeted_adv/untargeted_adv/cooperative); rounds = 4×agents; CoT kept private vs public answer; history window. Logs: per round `{agent, prompt, full_answer, public_answer}`. Metrics: any/final success, wrong scores, score leakage (GPT-4 judge).
- Relevance: medium — the private-reasoning vs public-message split per step is a must-have log field.

**EconAgent (tsinghua-fib-lab/ACL24-EconAgent)** [search]; **AI Economist Foundation (salesforce/ai-economist, archived Nov 2023)** [docs: base_env.py]
- Foundation: `BaseEnvironment(components, n_agents, world_size, episode_length, multi_action_mode_agents/planner, flatten_observations, dense_log_frequency, world_dense_log_frequency, seed)`; obs keys namespaced `world-*`, `<component>-*`, `p<idx>` for planner, `action_mask` per agent; **`dense_log = {world[], states[], actions[], rewards[]}` appended per timestep, plus component logs at episode end**; replay log recorded in `step`. Two-level agents (mobile + planner).
- Relevance: medium — `dense_log` is a compact precedent for "states/actions/rewards per step + periodic world snapshot"; component-namespaced observation keys and `generate_masks` are nice.

**Deal-or-No-Deal lineage**: Lewis et al. 2017 dataset; **AgoraBench / BargainArena (2025–26, ericoh929/Agorabench)**, MERIT feedback [search]. Relevance: low (dataset/metric work).

**GLEE (eilamshapira/GLEE, 2024)** [search]: bargaining/negotiation/persuasion two-player language games, 80K simulated games dataset, oTree human-vs-LLM. Relevance: medium as a games+dataset release pattern.

**CoopEval (ICML 2026, Tewolde/Zhang/…/Conitzer/Jin)** [search]: PD, Traveler's dilemma, Trust, 3-player Public Good with cooperation-sustaining *mechanisms* (repetition, reputation, mediation, **contracts with outcome-conditional payments**). Code "being assembled" per one mirror. **Relevance: high conceptually** — it is the closest published thing to "first-class contracts/mechanisms as spec objects", but it is a benchmark, not a spec language.

**SidConArena (2026)** [search]: open-ended positive-sum bargaining with three phases (negotiation with binding trades → deterministic production → sealed-bid auctions). No code link found. Relevance: medium (phase-structured game).

**SPIN-Bench (2025)** [search/abstract]: PDDL + board + cooperative card + negotiation (reported Hanabi/Diplomacy, unverified); code link not found. **DSGBench (DeciBrain-Group/DSGBench)** [search]: StarCraft II, Civ, Street Fighter, Werewolf, Diplomacy; Gym-based; "decision tracking". **lmgame-Bench** [search]: video games via Gym. **Werewolf Arena (Google, 2024)** [search]. **GAMEBoT** (uses PettingZoo envs) [search]. **GAMA(γ)-Bench (ICLR 2025)** [search]: 8 classical game-theory scenarios, multi-player. **Board Game Arena (2025)** [search]. Relevance: all are benchmark harnesses with bespoke loops; none exposes a spec language.

**Democratizing Diplomacy harness / goodstartlabs/ai_diplomacy (AAAI-26)** [search]: text state representation that lets 24B local models complete full-press Diplomacy; private diaries; "Critical State Analysis". Relevance: low-medium (state rendering matters more than model size — supports our "observation rendering is a spec concern" stance).

**LMAct (google-deepmind/lm_act, ICML 2025)** [search]: single-agent in-context imitation with up to 512 demo episodes; multiple state representations (text/image). Relevance: low, but note "multiple renderings of the same state" as a spec feature.

**MEMO (openverse-ai/MEMO, 2026; unrelated to memo-lang)** [abstract]: self-play prompt evolution with TrueSkill + prioritized replay of decisive states over 5 text games. Relevance: low; note the name collision.

**ChatArena (Farama/chatarena)** [search]: deprecated Aug 2025. **AgentScope / AgentSociety / TinyTroupe / Mesa-LLM** [search]: large-scale social simulation platforms; not game-theoretic; not examined further.

**Norms-specific**: "Social Learning and Collective Norm Formation … CPR" (AAMAS 2026, arXiv 2510.14401), "Evolution of Social Norms in LLM Agents using Natural Language" (2409.00993, metanorms), "Can LLMs Understand Social Norms in Autonomous Driving Games" (2408.12680, explicitly Markov games), Barrie & Törnberg 2505.23796 (emergence ≈ data leakage caveat). All [search]; none releases a reusable norm-aware harness as far as I could see.

**Cross-cutting trajectory formats**: **Harbor ATIF v1.7** [docs]: `schema_version`, `session_id`, `trajectory_id`, `agent{name, version, model_name}`, `steps[{step_id (1-based), source ∈ system/user/agent, message, tool_calls[{tool_call_id, function_name, arguments}], observation{results[{source_call_id, content}]}, metrics}]`, `final_metrics`, `subagent_trajectories`, `extra`; Pydantic models reject undeclared fields; loading an ATIF can seed a new session. **NVIDIA ATOF** [search]: event-level JSONL wire format with scopes, converts to ATIF. Both are single-agent, tool-use-centric; no joint actions/payoffs. Relevance: medium — align field *names* where they overlap (`step_id`, `source`, `metrics.prompt_tokens/completion_tokens/cost_usd`, `extra`) so our per-agent LLM sub-trajectories can be exported to ATIF.

---

## 5. agentmodels.org / WebPPL tradition and memo **[docs + local]**

- **agentmodels.org** (Evans, Stuhlmüller, Salvatier, Filan): agents are WebPPL functions; (PO)MDP agents plan by simulating future selves; multi-agent chapter (ch. 7) covers Schelling coordination (mutual recursion with `depth`), tic-tac-toe (`act(state, player)` samples a move, `simulate` applies `transition` and recurses into the opponent's `act` until `isTerminal`; `factor(eu)` = softmax with scale 1), and an RSA language game with explicit `alpha`. **There is no outer environment loop in the book**: multi-step behaviour is *inside* planning (`simulate`); the "world" is the agent's model of it. The `webppl-agents` library provides gridworld/bandit MDP/POMDP constructors.
- **memo (kach/memo, OOPSLA 2025)** [local + Handbook]: `@memo def f[axes](params)`; statements `chooses(x in D, wpp=…)`, `thinks[...]`, `observes [x] is y`, `observes_event(wpp=…)`, `knows`, `wants`, `snapshots_self_as`, `forgets_about`; expressions `E[]`, `Pr[]`, `Var[]`, `H[]`, `KL[]`, `imagine[...]`, `cost @`. Enumerative inference compiled to JAX arrays; exact posteriors; differentiable; `@memo(cache=True)` memoises recursive calls. The Handbook has **no construct for time or environment interaction**; sequential problems are written as **recursion over a horizon argument `t`** with the state/belief as an *axis*:
  - `demo-pomdp.ipynb`: `Q[b: B, a: A](t)` with an `env` frame choosing `s`, `s_`, `o`, and `future_alice` updating belief `b_` and choosing `a_ = argmax Q[b_, a_](t-1)`.
  - `demo-i-pomdp.ipynb`: trust game with `hist[ig, tg, h](r, level, beta)` building a prior over **encoded histories `H`** and `investor[h, ig, fi](r, level, beta)` as a Q-function — history is a product domain with an "unplayed" sentinel.
  - `demo-takeaway.ipynb`: `EU[n](t)` / `π[n, m](t)` mutual recursion, zero-sum alternating game.
  - `demo-ultimatum.ipynb`: one-shot two-stage game (`offerer` imagines `receiver`).
- **Has anyone built multi-step environment loops around memo?** Not that I could find. The only candidate is "Theories of Mind as Domain-Specific Languages of Thought" (arXiv 2609.19598, 2026) which (per a search snippet, unverified) pairs memo with the JAX2d physics simulator; the abstract says nothing about an environment loop or code release. The ACM paper mentions a ~15-line POMDP solver. No GitHub project combining memo with PettingZoo/OpenSpiel/an LLM harness surfaced.
- **Local precedent**: `scratchpad/exp_loop.py` (this project) already does the right thing: a jitted `step(h, a1, a2, h_)` transition over an encoded history domain, `Q1[h, a1](t)` as a memo Q-function with type inference over the opponent (`observes_event(wpp=hist_lik(...))`), and a plain Python loop that samples actions from the softmax policy, advances the state, and feeds the new history back in as an axis value. This is the "memo-based rational agent" plug-in shape: **memo computes a policy table over (information-state-index, action) for a horizon; the runtime indexes it.**
- Implication for the spec: to support memo agents, the spec must be able to *export* (a) finite typed domains for state/observation/action (so they become memo axes / `domain(...)` products), (b) the transition and payoff as JAX-jittable pure functions of integer-coded values, and (c) the information-state encoding (OpenSpiel's `information_state_string` analogue, but as an integer index). That is a stronger requirement than any LLM-only framework imposes and is the main thing none of the above provides.

---

## 6. Comparison table: "harness code every study rebuilds" vs. who provides it

Legend: ✔ provided and reusable · ◐ partial / bespoke per game · ✘ absent · n/a not applicable. Columns are the frameworks most worth citing; the many benchmarks (GovSim, NegotiationArena, GTBench, MAgIC, LLM-Deliberation, MARBLE, Welfare Diplomacy, CoopEval, GLEE…) are collapsed into "typical LLM-game benchmark".

| Harness concern | LangGraph | Inspect AI | PettingZoo | OpenSpiel | Melting Pot | Concordia v2 | TextArena | clembench | SOTOPIA | AI-Economist Foundation | Typical LLM-game benchmark | Our library (target) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Engine loop / turn order (sequential, simultaneous, chance) | ◐ (super-steps; no turn concept) | ✘ (solver chain; one model) | ✔ AEC + Parallel | ✔ (`current_player`, `kSimultaneousPlayerId`, chance nodes) | ✔ (simultaneous only) | ✔ Sequential / Simultaneous / Async engines | ◐ (per-env `current_player_id`) | ◐ (GM-coded per game) | ✔ random / round-robin / simultaneous | ✔ (simultaneous, planner period) | ◐ hard-coded per paper | ✔ declared in spec (`dynamics`, phases) |
| Typed state + transition | ◐ TypedDict + reducers | ✘ (`Store`, untyped) | ✘ (hidden in env) | ✔ (C++/Python classes; imperative) | ✔ (Lua components) | ✘ (free text, GM) | ✘ (`game_state` dict) | ✘ | ✘ | ◐ (components) | ✘ | ✔ declarative |
| Per-agent observation structure (public/private, info-state vs obs) | ✘ | ✘ | ◐ (spaces only) | ✔ (`information_state_*` vs `observation_*`, per player) | ◐ (`permitted_observations`) | ◐ (GM `make_observation`, text) | ◐ (`(sender,msg,type)` list to a player) | ◐ (GM-coded) | ◐ (`to=` private messages) | ✔ (namespaced obs dict + planner view) | ✘ | ✔ |
| Finite action set / legal-action mask | ✘ | ✘ | ✔ (`action_mask`) | ✔ (`legal_actions`, mask) | ◐ (fixed Discrete) | ◐ (`CHOICE` options) | ✘ (strings) | ✘ | ◐ (`available_actions`) | ✔ (`generate_masks`) | ✘ | ✔ |
| Action parsing / validation + illegal-action policy | n/a | ◐ (tool-call parsing) | ✔ (`TerminateIllegalWrapper`) | ✔ (`apply_action_with_legality_check`, `string_to_action`) | ✘ | ◐ (`action_spec_parser`) | ◐ (per env, `set_invalid_move`) | ✔ (`parse` / `invalid format` events, Aborted) | ◐ | ◐ | ✘ (regex per paper) | ✔ pluggable, logged |
| LLM prompt + call + logprob + usage logging | ◐ (LangSmith tracing, external) | ✔ (`ModelEvent`, `ModelCall`, `logprobs`, `ModelUsage`, cache) | n/a | n/a | n/a | ◐ (language_model wrappers, log channels) | ✘ | ✔ (`requests.json` raw prompt/response) | ◐ (`reasoning`, `rewards_prompt`) | n/a | ◐ (prompts saved, rarely logprobs) | ✔ |
| State update / checkpoint per step | ✔ (checkpoint tuples, pending writes) | ✔ (`StateEvent` JSON patch) | ✘ | ◐ (`serialize`, `clone`, `undo`) | ✘ | ✘ | ◐ (`State.logs`) | ✘ | ✘ | ✔ (`dense_log.states`) | ✘ | ✔ |
| Payoff / reward computation (per-step vs terminal, general-sum) | ✘ | ◐ (scorer per sample) | ◐ (scalar rewards) | ✔ (`rewards`, `returns`, `utility` type) | ✔ (scalar + `COLLECTIVE_REWARD`) | ✘ (LLM GM) | ◐ (`close() -> rewards`) | ◐ (scores) | ◐ (LLM-judged 7 dims) | ✔ | ◐ | ✔ typed payoffs in spec |
| Norms / contracts / rules as first-class objects | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ (GM prose) | ✘ | ✘ | ✘ | ◐ (planner tax component) | ✘ (CoopEval has mechanisms as prompt variants) | ✔ |
| Logging / replay format | ✔ (checkpointer; replay re-executes) | ✔ (`.eval`/`.json`, events, header/streaming reads) | ✘ | ◐ (`history()` + `serialize()` = exact replay) | ✘ | ◐ (`SimulationLog.to_json`, viewer) | ◐ (`logs` list) | ✔ (`interactions.json` + `requests.json`) | ✔ (`EpisodeLog`) | ✔ (`dense_log`) | ◐ (ad hoc JSON/W&B) | ✔ JSONL events + snapshots, replay-by-readback |
| Offline metrics from logs | ✘ | ✔ (`recompute_metrics`, `edit_score`) | ✘ | ✘ | ◐ (evaluation lib) | ◐ (`concordia-log`) | ◐ (TrueSkill) | ✔ (`clem score/eval`) | ◐ | ◐ | ◐ (notebooks) | ✔ |
| Baseline agents (scripted / rational / random) | n/a | ✘ | ◐ (random sampling) | ✔ (random, MCTS, CFR, tabular) | ✔ (background bots) | ◐ (prefabs) | ◐ (basic agents) | ✔ (`_player_programmed`, human via slurk) | ✔ (`ScriptWritingAgent`, `HumanAgent`) | ✔ (RL agents) | ◐ (exploiters, MCTS) | ✔ scripted + memo-rational + LLM |
| Config / seeding / reproducibility | ◐ (`thread_id`) | ✔ (`EvalSpec`, `seed`, cache keys, epochs) | ✔ (`reset(seed)`) | ✔ (`GameParameters`, chance outcomes) | ✔ | ◐ | ◐ (`reset(seed)`) | ✔ (instances.json, experiments) | ◐ | ✔ (`seed`) | ◐ (Hydra/YAML) | ✔ spec hash + seeds + chance log |

Reading: the row that is ✘ almost everywhere is **norms/contracts**; the second-weakest is **per-agent observation structure**; the third is **typed state/transition**. Those three are the library's distinct contribution; everything else has a clear best-of-breed reference (loop: PettingZoo/OpenSpiel/Concordia; log: Inspect/clembench; payoffs: OpenSpiel; baselines: OpenSpiel/Melting Pot).

---

## 7. Concrete recommendations distilled

1. **Spec metadata**: expose OpenSpiel-style declared properties (`dynamics`, `chance_mode`, `information`, `utility`, `reward_model`, `num_players`, `max_length`) and compute `legal_actions(state, agent)` from the typed action domain.
2. **Seats, roles, populations**: Melting Pot's focal/background split and PettingZoo's `possible_agents` → spec declares seats with roles; a *scenario* binds agent classes to seats.
3. **Observation structure**: per seat, a typed observation record with a `visibility` annotation per field (public / private-to / hidden) and an explicit information-state encoding for memo agents; TextArena's `(sender, message, ObservationType)` list for the chat channel.
4. **Action elicitation for LLMs**: Concordia `ActionSpec(call_to_action, output_type, options)` generated from the action domain + legal mask; parsing/validation produces clembench-style `parse` / `invalid format` events and a declared illegal-action policy.
5. **Log**: one JSONL event stream per episode (Inspect-style typed events, incl. `ModelEvent` fields and `StateEvent` JSON-patch) + periodic full snapshots + a separate raw-LLM-call file (clembench `requests.json`) + a scores file written by offline metrics. Chance outcomes are logged as events so replay-by-readback is exact; replay-by-recompute is a separate mode and must be labelled as such (LangGraph lesson).
6. **Fork semantics**: `fork(episode, step, patch)` à la LangGraph `update_state` — the mechanism for counterfactual norm/contract interventions.
7. **Rules/norms/contracts**: no prior art to copy; design from CoopEval's mechanism taxonomy (repetition, reputation, mediation, contracts) and GovSim-style discussion phases, implemented as spec objects that (a) restrict/relabel legal actions, (b) add payoff terms (sanctions, transfers), (c) emit `RuleEvent`s.
8. **memo agents**: spec must export integer-coded domains and jittable `transition`/`payoff`/`observe` functions; the runtime indexes a memo-computed policy table (pattern from `exp_loop.py`, `demo-i-pomdp`).
9. **Export adapters** (cheap wins): PettingZoo AEC/Parallel wrapper, OpenSpiel-like `observation_string`, ATIF export of per-agent LLM sub-trajectories, Inspect agent-bridge-style interception for third-party agents.

---

## 8. What I could not verify

- LangGraph: `Checkpoint.v`/`ts` fields and `CheckpointMetadata.parents` were not on the fetched pages.
- Inspect: `EvalSample` full field list came from the eval-logs page + memory (`store`, `model_usage`, `total_time` not confirmed); `Logprobs/Logprob/TopLogprob` field names not confirmed; multi-agent page 404'd.
- Melting Pot: `Policy` interface, per-player `TimeStep` lists, `WORLD.RGB` details — from memory, not from fetched source.
- Concordia: `SimulationLog` fields (file path 404'd); only `to_json/from_json` + viewer confirmed via CHEATSHEET.
- NegotiationArena message tags, MAgIC/OASIS/EconAgent internals, GTBench logging, MARBLE log schema, AgentVerse concrete rule names, AutoGen `save_state`, CAMEL environments module, CoopEval/SidConArena/SPIN-Bench code availability, DSGBench "decision tracking" format: search-level only.
- memo + JAX2d pairing (2609.19598): from a search snippet; the arXiv abstract page does not mention JAX2d.
- No official Inspect example of two models playing a game against each other was found.

---

## Sources

LangGraph
- https://docs.langchain.com/oss/python/langgraph/persistence
- https://docs.langchain.com/oss/python/langgraph/checkpointers
- https://docs.langchain.com/oss/python/langgraph/graph-api
- https://docs.langchain.com/oss/python/langgraph/use-time-travel

Inspect AI
- https://inspect.aisi.org.uk/eval-logs.html
- https://inspect.aisi.org.uk/reference/inspect_ai.event.html
- https://inspect.aisi.org.uk/reference/inspect_ai.model.html
- https://inspect.aisi.org.uk/reference/inspect_ai.log.html
- https://inspect.aisi.org.uk/agents.html
- https://inspect.aisi.org.uk/agent-custom.html
- https://inspect.aisi.org.uk/agent-bridge.html
- https://inspect.aisi.org.uk/sandboxing.html
- https://inspect.aisi.org.uk/caching.html

Multi-agent RL suites
- https://pettingzoo.farama.org/api/aec/
- https://pettingzoo.farama.org/api/parallel/
- https://pettingzoo.farama.org/api/wrappers/pz_wrappers/
- https://gymnasium.farama.org/api/spaces/
- https://github.com/google-deepmind/open_spiel/blob/master/docs/concepts.md
- https://raw.githubusercontent.com/google-deepmind/open_spiel/master/open_spiel/spiel.h
- https://openspiel.readthedocs.io/en/latest/api_reference.html
- https://github.com/google-deepmind/meltingpot
- https://github.com/google-deepmind/meltingpot/blob/main/docs/concepts.md
- https://github.com/google-deepmind/meltingpot/blob/main/docs/using_substrates.md
- https://raw.githubusercontent.com/google-deepmind/meltingpot/main/meltingpot/scenario.py

LLM multi-agent / social simulation / negotiation
- Concordia: https://github.com/google-deepmind/concordia , https://github.com/google-deepmind/concordia/blob/main/concordia/environment/README.md , https://github.com/google-deepmind/concordia/blob/main/CHEATSHEET.md , https://raw.githubusercontent.com/google-deepmind/concordia/main/concordia/typing/entity.py , https://arxiv.org/abs/2507.08892 , https://www.cooperativeai.com/post/google-deepmind-releases-concordia-library-v2-0
- GovSim: https://github.com/giorgiopiatti/GovSim , https://arxiv.org/html/2404.16698v4
- NegotiationArena: https://github.com/vinid/NegotiationArena , https://proceedings.mlr.press/v235/bianchi24a.html
- GTBench: https://github.com/jinhaoduan/GTBench , https://arxiv.org/abs/2402.12348
- MAgIC: https://github.com/cathyxl/MAgIC
- SOTOPIA: https://github.com/sotopia-lab/sotopia , https://docs.sotopia.world/concepts/agents , https://docs.sotopia.world/concepts/environments , https://raw.githubusercontent.com/sotopia-lab/sotopia/main/sotopia/database/logs.py
- OASIS: https://arxiv.org/abs/2411.11581 , https://docs.oasis.camel-ai.org/introduction.md
- Generative Agents: https://github.com/joonspk-research/generative_agents
- AgentVerse: https://github.com/OpenBMB/AgentVerse , https://github.com/OpenBMB/AgentVerse/tree/main/agentverse/environments/simulation_env/rules , https://arxiv.org/abs/2308.10848
- CAMEL: https://github.com/camel-ai/camel
- AutoGen: https://github.com/microsoft/autogen
- TextArena: https://github.com/LeonGuertler/TextArena , https://raw.githubusercontent.com/LeonGuertler/TextArena/main/textarena/core.py , https://arxiv.org/abs/2504.11442
- clembench / clemcore: https://github.com/clembench/clembench , https://github.com/clembench/clembench/blob/main/docs/logging_and_scoring.md , https://github.com/clp-research/clemcore , https://arxiv.org/abs/2405.20859
- MultiAgentBench / MARBLE: https://github.com/ulab-uiuc/MARBLE , https://arxiv.org/html/2503.01935
- Welfare Diplomacy: https://github.com/mukobi/welfare-diplomacy , https://arxiv.org/abs/2310.08901
- LLM-Deliberation: https://github.com/S-Abdelnabi/LLM-Deliberation
- EconAgent: https://aclanthology.org/2024.acl-long.829 , https://github.com/tsinghua-fib-lab/ACL24-EconAgent
- AI Economist Foundation: https://github.com/salesforce/ai-economist , https://raw.githubusercontent.com/salesforce/ai-economist/master/ai_economist/foundation/base/base_env.py
- GLEE: https://arxiv.org/abs/2410.05254
- CoopEval: https://arxiv.org/abs/2604.15267 , https://icml.cc/virtual/2026/poster/66503
- SidConArena: https://arxiv.org/abs/2606.27397
- SPIN-Bench: https://arxiv.org/abs/2503.12349
- DSGBench: https://arxiv.org/abs/2503.06047
- lmgame-Bench: https://arxiv.org/abs/2505.15146 , https://github.com/lmgame-org/GamingAgent/tree/main/lmgame-bench
- Werewolf Arena: https://arxiv.org/abs/2407.13943
- GAMA-Bench: https://proceedings.iclr.cc/paper_files/paper/2025/hash/a46adbe2f0ca0e16ef8857e188991ad7-Abstract-Conference.html
- GAMEBoT: https://arxiv.org/abs/2412.13602
- Democratizing Diplomacy: https://arxiv.org/abs/2508.07485 , https://ojs.aaai.org/index.php/AAAI/article/view/41067
- LMAct: https://arxiv.org/abs/2412.01441
- AgoraBench / BargainArena / MERIT: https://arxiv.org/abs/2602.10467 , https://arxiv.org/abs/2505.22998
- MEMO (self-play, name collision): https://arxiv.org/abs/2603.09022
- ChatArena (deprecated): https://pypi.org/project/chatarena/
- AgentScope: https://arxiv.org/abs/2407.17789 ; AgentSociety: https://arxiv.org/abs/2502.08691 (not fetched); TinyTroupe: https://github.com/microsoft/TinyTroupe (not fetched)
- Norms: https://arxiv.org/abs/2510.14401 , https://arxiv.org/abs/2409.00993 , https://arxiv.org/abs/2408.12680 , https://arxiv.org/abs/2505.23796
- Surveys: https://arxiv.org/abs/2412.03920 , https://arxiv.org/abs/2412.03563

Trajectory formats
- Harbor ATIF: https://docs.harborframework.com/agents/atif , https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md , https://pypi.org/project/atif/
- NVIDIA ATOF: https://docs.nvidia.com/nemo/relay/reference/atof-event-format

agentmodels / memo
- https://agentmodels.org/ , https://agentmodels.org/chapters/7-multi-agent.html
- https://github.com/kach/memo , https://github.com/kach/memo/blob/main/Handbook.md , https://dl.acm.org/doi/10.1145/3763078 , https://pypi.org/project/memo-lang/
- https://arxiv.org/abs/2609.19598 (Theories of Mind as DSLs; memo + JAX2d per search snippet)
- Local: /home/user/memo/demo/demo-pomdp.ipynb , demo-i-pomdp.ipynb , demo-takeaway.ipynb , demo-ultimatum.ipynb ; (local scratch, not committed)
