# Concordia (google-deepmind/concordia, v2.x) — research notes for a typed game-spec + thin-runtime library

Date: 2026-10-16. Repo state examined: `main` (latest release v2.4.0, 6 Mar 2026; v2.0.0 was 4 Jul 2025). Older v1 code was checked at tag `v1.8.10` for comparison.

Method and caveats (honesty section):
- All reading was done with WebFetch, which summarizes pages with a small model. Verbatim quotes below were returned by that tool from the raw source; when the tool only paraphrased, I say so. Short files (`typing/entity.py`, `typing/entity_component.py`, `environment/engine.py`, `typing/logging.py`, `typing/scene.py`, `typing/simulation.py`, `typing/prefab.py`, `setup.py`) were returned in full and are reliable.
- The arXiv HTML of 2312.03664 was read in two chunks (0–100k and 100k–end chars); the HTML of 2507.08892 was read once, in summary form. I did not read the PDFs.
- I did not fetch the readthedocs/Cloudflare tutorial site (`concordia-tutorial.pages.dev`), `CHEATSHEET.md` was summarized rather than reproduced, and `pyproject.toml` has no `[project]` table (dependencies live in `setup.py`).
- I could not open `examples/games/haggling/configs/*` (the concrete `action_to_scores` payoff functions); I only saw the payoff *component* that calls them.
- "Concordia v2 paper": arXiv 2507.08892 is **"Multi-Actor Generative Artificial Intelligence as a Game Engine"** (Vezhnevets, Matyas, Cross, Paglieri, Chang, Cunningham, Osindero, Isaac, Leibo; 10 Jul 2025). It is a design-philosophy paper, not a contest paper. The Concordia Contest (NeurIPS 2024) results paper is a separate one: arXiv 2512.03318, "Evaluating Generalization Capabilities of LLM-Based Agents in Mixed-Motive Scenarios Using Concordia" (Smith, Abdulhai, Diaz, Tesic, Trivedi, Vezhnevets, … Leibo; NeurIPS 2025 D&B).

---

## 1. One-paragraph summary

Concordia is a Python library for "generative agent-based models" (GABMs). Everything is an **Entity** with `name`, `act(action_spec) -> str`, `observe(observation: str)`. An entity is "a box of components with a name": many **ContextComponents** (each with `pre_act / post_act / pre_observe / post_observe / update` hooks) and exactly one **ActingComponent**. The **Game Master (GM)** is itself an entity built from components; the **Engine** drives the loop by *asking the GM* (via special `OutputType`s such as `MAKE_OBSERVATION`, `NEXT_ACTING`, `NEXT_ACTION_SPEC`, `RESOLVE`, `TERMINATE`) and asking players to `act`. All inter-entity traffic is strings; all state is either free text in an associative memory or small Python dicts held inside components. Typing of actions exists only at the leaf (`ActionSpec`: FREE / CHOICE / FLOAT). Logging is per-component dicts aggregated per step into a deduplicated JSON "structured log"; replay is via JSON checkpoints of component state, but the docs explicitly say models/components are not made deterministic.

---

## 2. Repository layout (as of main, Oct 2026)

```
concordia/
  typing/            entity.py, entity_component.py, prefab.py, scene.py, simulation.py, logging.py, human_input.py
  agents/            entity_agent.py, entity_agent_with_logging.py
  components/
    agent/           memory.py, observation.py, concat_act_component.py, question_of_recent_memories.py, plan.py,
                     constant.py, instructions.py, all_similar_memories.py, action_spec_ignored.py, scripted_act.py,
                     puppet_act.py, human_act_component.py, select_act_component.py, image_text_act_component.py,
                     report_function.py, pubsub.py, concat.py
    game_master/     switch_act.py, make_observation.py, next_acting.py, event_resolution.py, terminate.py,
                     next_game_master.py, scene_tracker.py, world_state.py, inventory.py, payoff_matrix.py,
                     instructions.py, questionnaire.py, open_ended_questionnaire.py, asynchronous_questionnaire.py,
                     formative_memories_initializer.py, script.py, interrupt_*.py (6 files)
  environment/       engine.py, step_controller.py, engines/{sequential,simultaneous,asynchronous}.py, scenes/
  prefabs/
    entity/          basic.py, basic_with_plan.py, basic_scripted.py, conversational.py, minimal.py, puppet.py,
                     rational.py, fake_assistant_with_configurable_system_prompt.py
    game_master/     generic.py, dialogic.py, dialogic_and_dramaturgic.py, game_theoretic_and_dramaturgic.py,
                     situated.py, situated_in_time_and_place.py, physically_situated_and_dramaturgic.py,
                     marketplace.py, psychology_experiment.py, interviewer.py, open_ended_interviewer.py,
                     scripted.py, formative_memories_initializer.py, interrupt_driven.py, async_social_media.py
    simulation/      generic.py, questionnaire_simulation.py
  associative_memory/basic_associative_memory.py
  document/          interactive_document.py (LLM chain-of-thought scratchpad)
  language_model/    language_model.py (ABC), no_language_model.py, retry_wrapper.py, call_limit_wrapper.py
  utils/             measurements.py, structured_logging.py, structured_logging_html.py, log_viewer.html,
                     sampling.py, concurrency.py, simulation_server.py, visual_interface.py, project_*.py, ...
  contrib/           language_models/ (all provider wrappers), components/, prefabs/, persona_generators/, tools/mcp
  command_line_interface/   (`concordia-log` CLI, GUI)
  docs/skills/       analyze-logs.md, autorater-evals.md
examples/            tutorial.ipynb, games/{haggling, haggling_multi_item, pub_coordination, reality_show,
                     labor_collective_action, state_formation, social_deception}, resource_dilemma/, social_media/, ...
```

Dependencies (`setup.py`, v2.4.0): `python_requires>=3.12`; `install_requires = absl-py, ipython, matplotlib, numpy>=1.26, pandas, python-dateutil, reactivex, tenacity, termcolor`. Extras per provider: google (`google-genai`, `google-cloud-aiplatform`), openai, mistralai, ollama, together, huggingface (`torch`, `transformers`, `accelerate`), vllm, groq, amazon (boto3), langchain, mcp. The examples' runner also hard-requires `sentence_transformers` (`all-mpnet-base-v2`) for the embedder, even in mock-LLM mode.

---

## 3. Core type contract (`concordia/typing/entity.py`) — verbatim essentials

```python
@enum.unique
class OutputType(str, enum.Enum):
  # Common output types
  FREE = 'free'
  CHOICE = 'choice'
  FLOAT = 'float'
  # Game master output types
  MAKE_OBSERVATION = 'make_observation'
  NEXT_ACTING = 'next_acting'
  NEXT_ACTION_SPEC = 'next_action_spec'
  RESOLVE = 'resolve'
  TERMINATE = 'terminate'
  NEXT_GAME_MASTER = 'next_game_master'
  SKIP_THIS_STEP = 'skip_this_step'

PLAYER_ACTION_TYPES = (FREE, CHOICE, FLOAT)
GAME_MASTER_ACTION_TYPES = (MAKE_OBSERVATION, NEXT_ACTING, NEXT_ACTION_SPEC, RESOLVE, TERMINATE, NEXT_GAME_MASTER, SKIP_THIS_STEP)
FREE_ACTION_TYPES   = (FREE, MAKE_OBSERVATION, NEXT_ACTION_SPEC, RESOLVE)
CHOICE_ACTION_TYPES = (CHOICE, NEXT_ACTING, TERMINATE, NEXT_GAME_MASTER)
BINARY_OPTIONS = {'affirmative': 'Yes', 'negative': 'No'}

@dataclasses.dataclass(frozen=True, kw_only=True)
class ActionSpec:
  call_to_action: str            # '{name}' and '{timedelta}' substituted by the entity
  output_type: OutputType
  options: Sequence[str] = ()    # required iff output_type in CHOICE_ACTION_TYPES; no duplicates
  tag: str | None = None         # e.g. 'action', 'speech' — tag for the activity memory
  def validate(self, action: str) -> None: ...   # FREE: anything; CHOICE: must be in options; FLOAT: float(action) parses
  def to_dict(self) -> dict ...                  # JSON-serialisable
def action_spec_from_dict(d) -> ActionSpec
def free_action_spec(**kw) / float_action_spec / choice_action_spec / skip_this_step_action_spec

DEFAULT_CALL_TO_ACTION = ('What would {name} do next? Give a specific activity. If the selected action has a direct or indirect object then it must be specified explicitly.')
DEFAULT_ACTION_SPEC = free_action_spec(call_to_action=DEFAULT_CALL_TO_ACTION, tag='action')
DEFAULT_CALL_TO_SPEECH = 'Given the above, what is {name} likely to say next? Respond in the format `{name} -- "..."` ...'

class Entity(metaclass=abc.ABCMeta):
  @functools.cached_property @abc.abstractmethod
  def name(self) -> str
  @abc.abstractmethod
  def act(self, action_spec: ActionSpec = DEFAULT_ACTION_SPEC) -> str   # "The action will always be a string"
  @abc.abstractmethod
  def observe(self, observation: str) -> None                            # "Always a string."

class EntityWithLogging(Entity):
  @abc.abstractmethod
  def get_last_log(self) -> dict[str, Any]
```

Key design facts:
- **Actions are always strings**, even FLOAT (returned as `str(float(x))`, `'nan'` on parse failure) and CHOICE (one of the option strings). There is no structured/record-valued action.
- **Observations are always strings.** There is no observation schema.
- The GM is driven through the *same* `act()` method using GM-only `OutputType`s. That is the whole trick that makes "GM is just an entity" work.
- `SKIP_THIS_STEP` lets the GM do a setup step with no player action (used by initializer GMs, e.g. formative-memory generation).

### Component contract (`concordia/typing/entity_component.py`) — verbatim essentials

```python
ComponentName = str; ComponentContext = str
ComponentContextMapping = Mapping[ComponentName, ComponentContext]
_LeafT = str | int | float | None
_ValueT = _LeafT | Collection['_ValueT'] | Mapping[str, '_ValueT']
ComponentState = Mapping[str, _ValueT]                    # JSON-ish, used for checkpoints
EntityState = Mapping[str, ComponentState | Mapping[str, ComponentState]]

class Phase(enum.Enum):   # READY -> (PRE_ACT -> POST_ACT | PRE_OBSERVE -> POST_OBSERVE) -> UPDATE -> READY
  READY, PRE_ACT, POST_ACT, PRE_OBSERVE, POST_OBSERVE, UPDATE
  def successors(self) -> Collection[Phase]; def check_successor(self, successor)

class BaseComponent(metaclass=abc.ABCMeta):
  def set_entity(self, entity) / get_entity(self)
  @abc.abstractmethod def get_state(self) -> ComponentState
  @abc.abstractmethod def set_state(self, state: ComponentState) -> None
  def get_dynamic_state(self) -> ComponentState   # subset editable at runtime in the visual editor

class ComponentWithLogging(BaseComponent):
  def set_logging_channel(self, logging_channel: LoggingChannel)   # LoggingChannel = Callable[[Mapping[str, Any]], None]

class EntityWithComponents(Entity):
  get_phase(); get_component(name, *, type_=BaseComponent); get_state(); set_state(state)

class ContextComponent(BaseComponent):
  def pre_act(self, action_spec) -> str      # "Returns the relevant information for the entity to act."
  def post_act(self, action_attempt: str) -> str
  def pre_observe(self, observation: str) -> str
  def post_observe(self) -> str
  def update(self) -> None                   # after all post_act / post_observe; replace cached info

class ActingComponent(BaseComponent, metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def get_action_attempt(self, context: ComponentContextMapping, action_spec) -> str
```

Note that every hook returns **`str`** — component "context" is text to be concatenated into a prompt. `ComponentState` is the only structured thing, and it is for checkpoints, not for reasoning.

### Entity lifecycle (`concordia/agents/entity_agent.py`, `EntityAgent`)

`EntityAgent(agent_name, act_component: ActingComponent, context_components: Mapping[str, ContextComponent])`.
- `act(action_spec)`: under `_control_lock`: phase PRE_ACT → `contexts = self._parallel_call_('pre_act', action_spec)` (every context component, in a thread pool, deduplicated by object identity) → `action_attempt = self._act_component.get_action_attempt(contexts, action_spec)` → phase POST_ACT → `self._parallel_call_('post_act', action_attempt)` → phase UPDATE → `self._parallel_call_('update')` → READY → return `action_attempt`.
- `observe(observation)`: PRE_OBSERVE → `_parallel_call_('pre_observe', observation)` → POST_OBSERVE → `_parallel_call_('post_observe')` → UPDATE → `_parallel_call_('update')` → READY.
- `get_state()` → `{'act_component': ..., 'context_components': {name: state}}`; `set_state` restores by name, swallowing per-component errors (logs instead of raising).
- Components within a phase run **in parallel threads** (because LLM calls are the bottleneck); order of side effects within a phase is not guaranteed. (The 2023 paper says the same: "component updates run concurrently, so update order isn't guaranteed. A sequential wrapper is available.")
- `EntityAgentWithLogging` (`entity_agent_with_logging.py`) adds a `Measurements` registry; each `ComponentWithLogging` publishes to a channel named after its component key, the act component to channel `'__act__'`; `get_last_log()` returns `{channel_name: last_datum}` sorted by channel. Per-thread "capture keys" exist so a *shared* GM called from several async entity threads attributes logs to the calling entity.

---

## 4. Engines (`concordia/environment/engine.py`, `engines/*.py`)

### Abstract `Engine`
```python
class Engine(metaclass=abc.ABCMeta):
  def make_observation(self, game_master, entity) -> str
  def next_acting(self, game_master, entities) -> tuple[Entity, ActionSpec]
  def resolve(self, game_master, event: str) -> None
  def terminate(self, game_master) -> bool
  def next_game_master(self, game_master, game_masters) -> Entity
  def run_loop(self, game_masters, entities, premise: str, max_steps: int, verbose: bool,
               log: list[Mapping[str, Any]] | None, checkpoint_callback: Callable[[int], None] | None = None,
               step_controller: StepController | None = None, step_callback: Callable[[StepData], None] | None = None)
```
Plus `action_spec_parser(str) -> ActionSpec` (JSON preferred; legacy `"prompt: ...;;type: choice options: a, b"` format still parsed with a warning) and `action_spec_to_string(spec) -> json`.

Every one of those engine methods is implemented as **`game_master.act(ActionSpec(output_type=<GM type>, ...))`**, i.e. the engine *asks the GM entity a question in a specific output type* and parses the string reply:
- `make_observation`: `ActionSpec(call_to_action=call_to_make_observation.format(name=entity.name), output_type=MAKE_OBSERVATION)` → string observation; delivered via `entity.observe(obs)` only if non-blank.
- `next_acting`: `CHOICE`-style `NEXT_ACTING` with `options=[e.name for e in entities]` → name; then `NEXT_ACTION_SPEC` (free) → parsed by `action_spec_parser`. Raises `ValueError` if the name is not an entity.
- `resolve(putative_event)`: `game_master.observe(f'{PUTATIVE_EVENT_TAG} {putative_event}')` → `game_master.act(RESOLVE, call_to_action='Because of all that came before, what happens next?')` → `game_master.observe(f'{EVENT_TAG} {result}')`. (`PUTATIVE_EVENT_TAG='[putative_event]'`, `EVENT_TAG='[event]'`, defined in `components/game_master/event_resolution.py`.)
- `terminate`: `TERMINATE` with `options=BINARY_OPTIONS.values()`; true iff reply == `'Yes'`.
- `next_game_master`: returns the sole GM, else `NEXT_GAME_MASTER` choice over GM names.

### `Sequential` (`engines/sequential.py`) — run_loop, step by step
1. `if premise: game_master.observe(f'{EVENT_TAG} {premise}')`.
2. `while not self.terminate(gm) and steps < max_steps:`
3. `step_controller.wait_for_step_permission()` gate (play/pause/step from a UI thread).
4. `gm = next_game_master(...)`.
5. **For every entity, in parallel** (`concurrency.run_tasks`): `obs = make_observation(gm, entity)`; `entity.observe(obs)` if non-blank.
6. `(entity, spec) = next_acting(gm, entities)`.
7. If `spec.output_type == SKIP_THIS_STEP`: steps += 1, checkpoint, `StepData(acting_entity='(setup)')`, continue.
8. `action = entity.act(spec)`; prefix with `f'{entity.name} '` if the reply lacks it.
9. `resolve(gm, action)`.
10. steps += 1; build log entry (see §7), `checkpoint_callback(steps)`, `step_callback(StepData(...))`.

So one step = one acting entity; every entity gets a fresh observation each step (the GM decides what each sees).

### `Simultaneous` (`engines/simultaneous.py`)
Same skeleton, but `next_acting` returns a **list** (GM answers `NEXT_ACTING` with a comma-separated list, then one `NEXT_ACTION_SPEC` per entity). Each selected entity runs `_entity_act` concurrently (`concurrency.run_tasks_in_background`, fault tolerant — failures are logged and dropped): observe (GM calls serialized by `_gm_log_lock`) then act. Actions are joined: `resolve_input = '\n'.join(actions.values())`, and **one** `resolve` call handles the joint action. Entities don't see each other's actions until the round resolves.

### `Asynchronous` (`engines/asynchronous.py`, added v2.4.0)
One thread per player entity, each running its own observe→act→resolve loop against a shared, thread-safe GM; a shared `threading.Event` for termination, a pause event, `sleep_time=0.1` back-off when the GM returns nobody to act. No rounds, no global clock, no fairness policy. Used with `NextActingActiveEntity`, interrupt-driven GM components (`interrupt_scheduling.py`, `interrupt_time_model.py`, …) and the social-media example.

### Scenes (`typing/scene.py`, `components/game_master/scene_tracker.py`)
```python
@dataclasses.dataclass(frozen=True)
class SceneTypeSpec:
  name: str
  game_master_name: str | None = None
  default_premise: Mapping[str, Sequence[str | Callable[[str], str]]] | None = None   # per-player premise
  action_spec: Mapping[str, ActionSpec] | ActionSpec | None = None                     # per-scene (per-player) action spec
  possible_participants: Sequence[str] | None = None

@dataclasses.dataclass(frozen=True)
class SceneSpec:
  scene_type: SceneTypeSpec
  participants: Sequence[str]
  num_rounds: int
  start_time: datetime.datetime | None = None
  premise: Mapping[str, Sequence[str | Callable[[str], str]]] | None = None
```
`SceneTracker` flattens scenes into a global round index, queues per-participant premises into `MakeObservation` at scene start, advances a counter on each `RESOLVE` by writing a memory line `'[scene counter](N)'` (progress lives *in memory text*, not in a field), answers `NEXT_GAME_MASTER` with the scene's GM name, and signals `TERMINATE` via `SceneBasedTerminator` when `is_done()`. `NextActingFromSceneSpec` round-robins the scene's participants; `NextActionSpecFromSceneSpec` returns the scene's declared `ActionSpec` as JSON (no LLM). **This scene machinery is the closest thing Concordia has to a typed, declarative game spec.**

---

## 5. The Game Master in v2

### GM = entity + `SwitchAct` + GM components
`prefabs/game_master/generic.py` builds `EntityAgentWithLogging(agent_name=name, act_component=SwitchAct(model, entity_names, component_order), context_components={...})` with components, in order:

| key | class | role |
|---|---|---|
| `instructions` | `gm_components.instructions.Instructions` | constant text: "you are the game master in a social-science experiment modeled on a tabletop RPG; track world state, control turn order, end the game, track who knows what, keep the story moving; always third-person limited" |
| `examples` | `ExamplesSynchronous` | few-shot examples of answering the engine's seven calls (make_observation, next_acting, next_action_spec×2, resolve, check_termination×2) |
| `player_characters` | `PlayerCharacters` | list of player names |
| `relevant_memories` | `AllSimilarMemories` | top-5 associative retrieval |
| `__observation__` | `LastNObservations(1000)` | GM's recent observation text |
| `observation_to_memory` | `ObservationToMemory` | writes `[observation] ...` lines into memory |
| `display_events` | `DisplayEvents` | numbered list of last 100 `[event]` lines |
| `__memory__` | `AssociativeMemory` | shared GM memory bank (`allow_duplicates=True`) |
| `__make_observation__` | `MakeObservation` | answers MAKE_OBSERVATION |
| `__next_acting__` | `NextActing` / `NextActingInFixedOrder` / `NextActingInRandomOrder` (param `acting_order`) | answers NEXT_ACTING |
| `__next_action_spec__` | `NextActionSpec` | LLM writes a JSON ActionSpec |
| `__resolution__` | `EventResolution(steps=[maybe_inject_narrative_push, AccountForAgencyOfOthers, result_to_who_what_where, *extra])` | answers RESOLVE |

**`SwitchAct.get_action_attempt(contexts, action_spec)`** (`components/game_master/switch_act.py`) dispatches on `action_spec.output_type`. For each GM type it first checks whether the designated component key is present in `contexts` and, if so, returns that component's `pre_act` string **without an LLM call**; only if absent does it fall back to an LLM question ("YOLO" mode) over the concatenated context. Mapping: TERMINATE→`__terminate__`, NEXT_GAME_MASTER→`__next_game_master__`, MAKE_OBSERVATION→`__make_observation__`, NEXT_ACTING→`__next_acting__`, NEXT_ACTION_SPEC→`__next_action_spec__` (default free spec if empty), RESOLVE→`__resolution__`. Logs `{'Summary','Action Spec','Value','Prompt'}`.

This means **the GM's "decisions" are really made inside components during `pre_act`**; `SwitchAct` is a router. That is the main thing to borrow: *the engine's questions to the GM are a fixed enum, each answered by a pluggable component, with the LLM only as a fallback.*

### How observations are delivered (`components/game_master/make_observation.py`)
- `ObservationQueue`: thread-safe `{entity_name: [event, ...]}`; `add(entity_name, event, player_names)` with `'all'` broadcast; `get_and_clear(name)`. Can be shared across GMs (`external_queue`) so handoffs don't lose pending observations.
- `MakeObservation.pre_act` on MAKE_OBSERVATION: parse the entity name out of the call-to-action (default: `'What is the current situation faced by {name}? What do they now observe? Only include information of which they are aware.'`); pop that entity's queue; **if the queue is non-empty, the observation is the queued events joined by `'\n\n\n'` and the LLM is not called**; else, if `allow_llm_fallback` (default True), ask the LLM `"What does {name} observe now? Never repeat information that was already provided to {name} unless absolutely necessary. Keep the story moving forward."` (1200 tokens); else `''`. Optional `reformat_observations_in_specified_style`.
- Who puts things in the queue: `EventResolution(notify_observers=True)` asks the LLM "Which entities are aware of the event? Answer with a comma-separated list of entity names."; `SendEventToRelevantPlayers.post_act` (optionally asks "Is {name} aware of the latest event above?"); `PayoffMatrix`, `SequentialBargainPayoff`, `SceneTracker` push typed-to-text results (scores, premises) directly. The game-theoretic GM prefab sets `allow_llm_fallback=False`, making observation delivery fully deterministic given the queue.

### How actions are resolved (`components/game_master/event_resolution.py`)
- The engine has already stored `'[putative_event] <name>: <action>'` in GM memory. `EventResolution.pre_act` on RESOLVE scans memory for the latest `[putative_event]` by the active player (name from `NextActing`), strips the name prefix, writes `Putative event to resolve: ...` into an `InteractiveDocument` seeded with the configured components' pre_act text, then runs `run_chain_of_thought(steps)` where each step is `f(document, premise, active_player_name) -> str`. Built-in steps: `identity` (pass through: used by the game-theoretic GMs, so **resolution = the action verbatim**), `extract_direct_quote`, `determine_success_and_why` ("Does the attempted action succeed?" / "Why did the attempt fail?"), `result_to_causal_statement`, `attempt_to_result`, `attempt_to_most_likely_outcome` (lists ≥3 consequences then picks), `result_to_who_what_where`, `result_to_effect_caused_by_active_player`, `restore_direct_quote`, `get_action_category_and_player_capability`, `maybe_inject_narrative_push` (if "repetitive", draw 5 complications and pick one at random), `maybe_cut_to_next_scene`, and callable classes `AccountForAgencyOfOthers` (asks *other players' agents* yes/no whether they'd go along with implied voluntary actions — i.e. the GM calls `player.act()` mid-resolution), `Conversation` (spins an inner dialogue by calling participants' `act`/`observe`), `RemoveSpecificText`.
- Output: `"Event: {event_statement}\n"`; the engine then stores `'[event] ...'` in GM memory. Logs `{'Key','Summary','Value','Prompt','Details'}`.
- The 2023 paper's A.3 ordering for v1 was analogous: `update` → `partial_state` (→ player observes) → `player.act` → `update_before_event` → `state` → chain of thought → `update_after_event` → `terminate_episode`. In v1 (`v1.8.10/concordia/environment/game_master.py`) the GM was a hard-coded class with a `step()` loop, `randomise_initiative`, `players_act_simultaneously`, `action_spec.validate(action)` after each act, and a seeded RNG; v2 replaced that class with engine + components.

### How state is represented
There is **no single world-state object**. State is distributed across:
- **GM associative memory** (free text lines tagged `[observation]`, `[putative_event]`, `[event]`, `[scene counter](n)`, `[scene premise queued](n)`): the primary "ground truth" is the event log in text.
- **`WorldState` component** (`world_state.py`): a `dict[str, str]` of name→value; after each RESOLVE the LLM is asked "what state variables are important to write down now…" and answers `name|value, name|value`; values are overwritten, never deleted. Free-text keys and values.
- **`Locations`** component: LLM-generated location catalogue (`location|properties`) plus `entity_locations: dict[str,str]`, updated per event by asking the LLM, with optional `valid_locations` constraint and fuzzy normalization (`SAME`, substring match).
- **`GenerativeClock`**: `_num_steps: int` plus a *free-text* `_time` string the LLM updates per event.
- **`Inventory`** (`inventory.py`): the only numerically typed grounded state in core: `InventoryType = Mapping[str, dict[str, float]]`, `ItemTypeConfig(name, minimum, maximum, force_integer)`, bounds clamped in Python. But *deltas are extracted by a chain of LLM yes/no and open questions* ("Did any listed individual gain or lose {item_type}?", amounts as `"number|explanation"`, fallback ±1 or full loss on parse failure), and an `apply(fn)` hook for programmatic edits with validation.
- **`PayoffMatrix`** (`payoff_matrix.py`): `_partial_joint_action: dict[str, str|None]`, `_player_scores: dict[str, float]`; on each RESOLVE `post_act`, it reads `(active_entity_name, putative_action)` from `EventResolution` and stores the action *string* as the player's choice; when all current scene participants have acted it calls user-supplied `action_to_scores(joint_action: Mapping[str,str]) -> Mapping[str,float]` and `scores_to_observation(cumulative_scores) -> Mapping[str,str]`, queues the observations and adds `'{name} learned that ...'` to memory. No LLM. (Bugs noted: `_history` never appended; memory line uses wrong loop variable; observation reports cumulative not per-step scores.)
- Player state: each player's own `AssociativeMemory` (text) plus per-component caches (e.g. `LastNObservations`, the "three questions" answers).

Everything in the paper's "grounded variables" story is therefore: Python holds the number, the LLM is asked (in natural language) how the number changed, Python clamps it, then renders it back to text for the next prompt.

### Termination
`Terminate` (flag set by `terminate()`), `NeverTerminate`, `SceneBasedTerminator` (delegates to `SceneTracker.is_done()`). Engine also stops at `max_steps`. No LLM needed if a terminate component is present; otherwise `SwitchAct` falls back to asking the LLM yes/no.

---

## 6. Player agents and memory

- **Prefab** (`typing/prefab.py`): `@dataclass class Prefab(abc.ABC)` with `description: ClassVar[str]`, `params: Mapping[str, Any]`, `entities: Sequence[EntityWithComponents] | None` (GMs get the entity list; players never do), `build(model, memory_bank) -> EntityWithComponents`, plus `extra_components` / `extra_components_index` / `extra_components_dependencies` validation. `Role = {ENTITY, GAME_MASTER, INITIALIZER}`; `InstanceConfig(prefab: str, role: Role, params)`; `Config(prefabs, instances, default_premise='', default_max_steps=100)`.
- **`basic__Entity`** (`prefabs/entity/basic.py`, the "three key questions" agent): components `Instructions`, optional `Goal` (Constant), `ObservationToMemory`, `SituationPerception` ("What situation is {name} in right now?"), `SelfPerception` ("What kind of person is {name}?"), `PersonBySituation` ("What would a person like {name} do in a situation like this?"), `LastNObservations(1_000_000)`, `AssociativeMemory`; act = `ConcatActComponent`. Each perception component is an LLM call over recent memories, chained via `components=[...]` (so `PersonBySituation` sees the other two answers).
- **`rational__Entity`** (`prefabs/entity/rational.py`): same shape with Situation → "Which options are available to {name} right now?" → "Of the options available to {name}, which choice of action or strategy is best…?". **It is LLM-based; there is no utility computation or planner.** The 2025 paper's "simple RationalActor component" is not a non-LLM baseline.
- Non-LLM actors that *do* exist: `ScriptedActComponent` (plays back `[{name, line}]` lines via `forced_response` — still goes through the prompt machinery, logs an `SFT datum`), `PuppetActComponent` (`fixed_responses: Mapping[call_to_action -> response]`, LLM fallback), `HumanActComponent`, and `NoLanguageModel` / `RandomChoiceLanguageModel` / `BiasedMedianChoiceLanguageModel` (`language_model/no_language_model.py`: `sample_text` returns `''`, `sample_choice` returns index 0 / random / 80% median; the random ones call `random.seed(seed)` — global state side effect). The example games each ship a `puppet` scenario "for deterministic scoring verification".
- **`ConcatActComponent`** (`components/agent/concat_act_component.py`): concatenates component contexts (ordered by `component_order`, missing ones appended sorted) into an `InteractiveDocument`; FREE → `open_question(max_tokens=2200)`; CHOICE → `multiple_choice_question(options, randomize_choices=True)` → returns `options[idx]`; FLOAT → `open_question` then `float()`. Logs `{'Summary': 'Action: ...', 'Value', 'Prompt'}`.
- **`InteractiveDocument`** (`document/interactive_document.py`): a running transcript used as chain-of-thought scratchpad. `open_question(...)` → `model.sample_text(view().text())`; `multiple_choice_question(question, answers, randomize_choices=True)` shuffles with `self._rng.permutation`, labels options with letters, and calls **`model.sample_choice(prompt, responses=letters)`** — constrained choice, not free text parsing; `yes_no_question` = `multiple_choice_question(['No','Yes'])`. `utils/sampling.py` has `extract_choice_response` (regex `\(?\b(\w)\)`) and `dynamically_adjust_temperature(attempts, max_attempts)` (0.0 → 0.5 → 0.75 on retries) for wrappers that implement `sample_choice` by sampling text.
- **`LanguageModel`** ABC (`language_model/language_model.py`): `sample_text(prompt, *, max_tokens=5000, terminators=(), temperature=1.0, top_p=0.95, top_k=64, timeout=60, seed=None) -> str`; `sample_choice(prompt, responses, *, seed=None) -> tuple[int, str, Mapping[str, Any]]`. Docstring: "the sampling method is up to the implementation and may not reflect the log probabilities."
- **Memory** (`associative_memory/basic_associative_memory.py`): `AssociativeMemoryBank(sentence_embedder, allow_duplicates=False)` — a pandas DataFrame `['text','embedding']`; `add` (newlines stripped, dedup by Python `hash()` — which is salted per process, so dedup state does not survive restarts unless `PYTHONHASHSEED` is set), `retrieve_associative(query, k)` by raw dot product (cosine only if embeddings are normalized), `retrieve_recent(k)`, `scan(selector_fn)`; `get_state()` = `{'stored_hashes', 'memory_bank': df.to_json()}`. The component wrapper `AssociativeMemory` (`components/agent/memory.py`) buffers `add()`s and commits them in `update()`; access is forbidden during the UPDATE phase. `ListMemory` is the no-embedding variant. `ObservationToMemory.pre_observe` writes `'[observation] ...'`; `LastNObservations` / `ObservationsSinceLastPreAct` read it back.

---

## 7. Logging, measurement, replay

### Raw per-step log (engines)
`run_loop(..., log=list)` appends one dict per step. Sequential: `{'Step': n, 'Entity [name]': entity.get_last_log(), '<gm name>': {terminate, next_game_master, make_observation, next_acting, next_action_spec, resolve → each the GM's get_last_log() at that moment}, 'Summary': ...}`; empty components are dropped. Simultaneous: same with one `Entity [name]` key per acting entity. Skipped setup steps are not logged. Async: `RawLogEvent`s collected by `AsyncLogCollector` and materialized on pause/finish.

Each component decides what it logs; conventional keys are `'Key'`, `'Summary'`, `'Value'`, `'Prompt'` (full LLM prompt text as lines), sometimes `'Details'`, `'Chain of thought'`, `'Joint Action'`, `'Player Scores'`. `Measurements` (`utils/measurements.py`) is just a thread-safe `dict[str, list[Any]]` of named channels, in memory only (and `close()` has a self-deadlock/iteration bug).

### Structured log (`utils/structured_logging.py`)
`SimulationLog.from_raw_log(raw)` converts the list above. Schema:
```python
@dataclasses.dataclass
class StructuredLogEntry:
  step: int; timestamp: str; entity_name: str; component_name: str; entry_type: str
  summary: str = ''
  deduplicated_data: Mapping[str, Any]   # original dict; strings >= 50 chars replaced by {'_ref': sha256[:16]}
```
`SimulationLog.to_dict()` → `{'content_store': {id: text}, 'entries': [...], 'entity_memories': {name: [...]}, 'game_master_memories': [...]}`; `to_json()`, `to_html()`, `from_json()`; indexes by entity/step/component. `AIAgentLogInterface` offers `get_overview`, `get_entity_actions(name)` (per-step `__act__.Value`), `get_entity_action_context(name, step)` (action + observations + prompt + all component values), `get_component_values(component_key, value_key)`, `filter_entries`, `search_entries`, memories. CLI `concordia-log {overview, entities, actions, context, step, timeline, search, memories, components, dump, bundle}` (`--json`, `--include-images`), and `utils/log_viewer.html` renders the JSON in a browser. `examples/games/run.py` writes `/tmp/simulation_results_{game}_{scenario}.html` and `/tmp/simulation_log_{game}_{scenario}.json` and prints `focal_scores` / `background_scores` to 2 dp.

### Metrics
- v1 had `concordia/metrics/` (`goal_achievement.py`, `common_sense_morality.py`, `opinion_of_others.py`, `dass_questionnaire.py`): components that, after each event, ask the LLM a multiple-choice question (e.g. scale `['activity unrelated to the goal', 'somewhat working towards the goal', 'working towards the goal', 'goal achieved']`) and publish `{time_str, clock_step, timestep, value_float = idx/(len-1), value_str, player, goal, time}` to a `Measurements` channel. These were removed from core in v2 (there is a `typing/logging.py::Metric(question, output_type, options, context)` dataclass left over).
- v2's measurement story is: (a) **scores from payoff components** (`PayoffMatrix.get_scores()`, `SequentialBargainPayoff.get_scores()`, `inventory.Score.get_scores()`), computed in Python from action strings; (b) **questionnaire GMs** (`interviewer`, `open_ended_interviewer`, DASS questionnaire, `sequence_of_events` stimuli, `randomize_choices` to cut position bias); (c) **LLM autoraters over the structured log** (`docs/skills/autorater-evals.md`): a markdown rubric with weighted 1–5 dimensions (e.g. behavioral realism 20%, social emergence 25%, …), quantitative "data extraction recipes" via `concordia-log actions/search`, and an `ask_evaluation_question` helper that regex-extracts `FINAL_SCORE:`; plus an "auto-improvement loop" that feeds the report to a coding agent. The doc itself says "The autorater is only as good as the rubric."
- The 2025 contest paper (2512.03318) evaluates "general cooperative intelligence" by focal-agent scores in mixed-motive scenarios (haggling, pub coordination, reality-show matrix games, labor collective action, state formation) with background `basic__`/`rational__`/`puppet__` entities and zero-shot held-out scenarios; scenario registry in `examples/games/scenarios.py`: `ScenarioConfig(description, config_module, background_agent_prefab, num_background_players, num_supporting_players=0, focal_is_resident=True, tags=())` in an `immutabledict` keyed by id (e.g. `pub_coordination_london_0`, `_puppet_test`).

### Replay / checkpoints
- v2.0.0 made `get_state()/set_state()` mandatory on components and added checkpointing. `prefabs/simulation/generic.py::Simulation.make_checkpoint_data()` → `{'engine': {class, module}, 'entities': [...], 'game_masters': [...], 'raw_log', 'checkpoint_counter'}`, each entity entry = `{prefab_type, entity_params, components, component_info: {class, module, ctor params, state[, dynamic_state]}}`; `save_checkpoint(step, path)` writes `step_{n}_checkpoint.json` (pretty JSON, non-serializable values silently dropped); `load_from_checkpoint`. The docstring states that **engine cursors are not restored, and components, external services and models are not made deterministic**. So a checkpoint lets you *resume* but not *replay* an identical run.
- Determinism inventory: `LanguageModel.sample_*` accept `seed` but nothing threads a seed through engines/components; `InteractiveDocument` has its own `rng` for option shuffling; `NextActingInRandomOrder` uses global `random.choice`; `maybe_inject_narrative_push` picks a complication at random; component hooks run in thread pools; `AssociativeMemoryBank` dedup depends on process hash salt. There is no run-level seed and no event-sourced replay from the log.

---

## 8. The papers' framing

### 2312.03664 — "Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia" (Vezhnevets, Agapiou, Aharon, Ziv, Matyas, Duéñez-Guzmán, Cunningham, Osindero, Karmon, Leibo; Dec 2023, v2)
- Thesis: "Generative Agent-Based Models (GABM) are not just classic Agent-Based Models (ABM)s where the agents talk to one another." A GABM = generative agents + a *generative model of the environment* (the GM), after tabletop RPGs; agents "act by describing intended actions in natural language, and the GM turns them into outcomes".
- GM responsibilities: maintain a consistent world state, communicate observable state, decide action effects, resolve conflicts among simultaneous actions; "checks the physical plausibility of agent actions and describes their effects"; "Whenever an agent tries to perform an action that violates the grounding" the GM tells the agent the action was invalid. The GM "can use non-LLM modeling tools (ODEs, graphical models, finite state machines)" for parts of the simulation; "For grounded variables, which are tracked in Python, a specialised component is created to maintain the variable's state," updated after events and rendered to language.
- Formalism (no POMDP; four sampling equations): agent long-term memory **m** (set of strings) and working memory **z** = {zⁱ} (component states). (1) `aₜ ~ p(·|fᵃ(zₜ))`; (2) `zⁱₜ₊₁ ~ p(·|fⁱ(zₜ, mₜ))`; (3) GM event `eₜ ~ p(·|fᵉ(zₜ), aₜ)`; (4) GM observation `oⁱₜ₊₁ ~ p(·|fᵒ(zₜ₊₁))`. "Unlike RL, we do not assume that the agent responds with an action to every observation." Components are a Minsky-style "society of mind"; they may be LLM summarizers, memory queries, or "classic code (e.g. a hunger component)".
- Grounding in digital space: `PhoneGameMaster` / `PhoneUniverse` — prompt for available apps/functions, prompt for arguments, invoke, add notifications; apps subclass `PhoneApp` with decorated methods whose docstrings + type annotations generate descriptions. Four fidelity levels for digital functions: natural-language only, simulated simple behaviour, LLM-prompt implementation, real API.
- Logging: "Each episode creates a complete trace of component states zₜ and the resulting actions aₜ" for auditing and credit assignment to LLM vs components.
- Validation hierarchy proposed: generalization, algorithmic fidelity, model comparison, robustness, parsimony; worries: train-test contamination (LLMs have read the Prisoner's Dilemma literature so "setups may need to hide their interpretation"), stereotypes, individual fidelity, digital fidelity.
- Examples shipped in 2023: Calendar (phone app), Riverbend election, Day in Riverbend (Park et al. re-implementation), snowed-in pub (March & Olsen's three questions), Magic Beans (inventory), Cyberball.

### 2507.08892 — "Multi-Actor Generative Artificial Intelligence as a Game Engine" (Jul 2025)
- Argues for the **Entity-Component pattern** (after Unity-style game engines): an entity is "a simple, unique identifier—essentially a container", "a box of components with a name"; behaviour emerges from attached components; "composition over inheritance" avoids "complex and brittle inheritance chains"; entities can be collectives (companies, courts).
- Engine = the ECS "system": it "processes entities by calling observe and act on every entity whose components implement those functions"; `observe` triggers `pre_observe`/`post_observe` on all components; the single Acting component gathers `pre_act` outputs and returns one action, then `post_act` runs. "This reflects a basic assumption that an actor takes exactly one action per step" (time between steps is arbitrary, so not restrictive). Three engines: sequential, simultaneous, asynchronous.
- GM "is not a hardcoded computer game but is itself a configurable entity"; it "generat[es] all parts of the story not directly determined by the voluntary actions of player characters"; can be given "free rein" or "hard-coded guardrails"; can act as "narrative director" managing information asymmetry.
- Role split: engineers build "stable, reusable, and testable components"; designers compose scenarios without component-level code — their stated key to "rapid iteration".
- User-goal typology (after Ron Edwards' GNS theory): **Evaluationist** (fair playing field, "standardized scenarios, clear success metrics, controlled variability to test generalization, cross-play"; "consistent evaluation protocols, and reproducible outcomes"), **Dramatist** ("narrative satisfaction over procedural consistency"), **Simulationist** ("a little pocket universe without fudging"; psychologically plausible actors needing no scalar reward; predictive validity, causal consistency, empirical grounding; component fidelity "from a simple RationalActor to a more complex BeliefsAndReasoning component"). Claim: one scenario cannot serve all three equally; synthetic-data generation is cross-cutting. Section 4.3 is "Implications for Multi-Actor System Definition Language Design" (I only saw the heading).
- The paper does **not** present a formal model, does not discuss determinism or logging, does not use the word "harness", and mentions "actions grounded in physical, social, or digital space" only as the 2023 title. The closest thing to "minimal set of things a harness must do" is the engine's five calls (observe, next-acting, action-spec, resolve, terminate) plus the one-action-per-step assumption and the component lifecycle.

---

## 9. Weaknesses / baggage (from the viewpoint of typed state, finite actions, declared observations, rational baselines)

1. **Everything is a string at the interface.** `act() -> str`, `observe(str)`, every component hook returns `str`, `ActionSpec.options` are strings, joint actions are `Mapping[str, str]`. The only validation is `ActionSpec.validate` (membership / `float()`), and in v2 the engines don't even call it (v1's `GameMaster._step_player` did). Scores are computed from raw action strings (`PayoffMatrix`, `SequentialBargainPayoff` store `putative_action` verbatim).
2. **No world-state type.** State is scattered: text memories with tags, `WorldState` as LLM-maintained `dict[str,str]` that never deletes keys, `Locations`/`GenerativeClock` as LLM-rewritten strings, `Inventory` floats whose *deltas are extracted by LLM questioning*. Scene progress is counted by scanning memory for `[scene counter]` lines. You cannot write `state: GameState` and reason about it.
3. **LLM-based GM is the default** for observation, next-acting, action-spec writing, resolution, termination; deterministic alternatives exist only as opt-in components (`FixedActionSpec`, `NextActingInFixedOrder`, `NextActionSpecFromSceneSpec`, `SceneBasedTerminator`, `identity` resolution, `allow_llm_fallback=False`). The game-theoretic prefab shows that you *can* configure a GM that never calls the LLM for control decisions, but it is assembled by hand out of ~14 components and still carries `Instructions`/`ExamplesSynchronous`/`AssociativeMemory`.
4. **Observations are not declared.** What a player sees is whatever text ends up in its `ObservationQueue` (or the LLM's guess). Information sets are therefore not inspectable or provable; "who is aware of the event" is itself an LLM question. Hidden-information games (`social_deception`) rely on prompt discipline.
5. **No non-LLM rational baseline.** `rational__Entity` is a prompting pattern; `puppet__Entity`/`ScriptedActComponent` are fixed scripts; `NoLanguageModel` returns `''`/index 0. Nothing computes a best response or equilibrium against the typed game, because there is no typed game to compute against.
6. **Non-deterministic, non-replayable runs.** No run-level seed; thread pools inside every `act`/`observe`; global `random` in several places; `AssociativeMemoryBank` dedup tied to the process hash seed; checkpoints restore component state but the docs say models/engines are not made deterministic. The log records prompts and outputs but there is no "replay from log with a recorded-response model" facility.
7. **Heavy and framework-shaped.** Runtime deps: pandas, numpy, matplotlib, ipython, reactivex, absl, tenacity; a sentence-transformer embedder is effectively required (the `generic` GM prefab includes `AllSimilarMemories`; examples hard-load `all-mpnet-base-v2`); a visual simulation server, project browser, CLI, GUI, MCP tool-use. Threading locks everywhere (`_control_lock`, `_phase_lock`, `_gm_log_lock`, per-component locks).
8. **Prompt-fragility in the control path.** Action specs produced by the LLM are JSON strings parsed with a legacy fallback; next-acting names are validated only after a second LLM call; `Simultaneous` splits the GM's comma-separated reply; `WorldState` parsing breaks on commas in values; `Locations` uses substring matching; `Inventory` falls back to ±1 guesses on parse failure.
9. **Log is a dump, not a model.** The structured log is `{Key, Summary, Value, Prompt}` dicts per component with 16-hex content refs; metrics are extracted by regex/grep or by an LLM autorater. There is no typed event stream (step, actor, action, state-before, state-after, payoff) that an analysis script can trust.
10. **Code-quality smells** noticed while reading (useful as cautionary examples): `Measurements.close()` deadlocks/iterates while deleting; `PayoffMatrix` memory line uses the wrong loop variable; `_history` never appended in two payoff components; stale docstrings (`ValueError` for duplicate components that is never raised; `script` key `action` vs `line`); `retrieve_recent` returns oldest-first despite docstring.

---

## 10. "What a harness has to do" — Concordia's answer, and what to borrow

| Harness duty | What Concordia does | Worth borrowing (conceptually) | Do differently |
|---|---|---|---|
| **Engine loop** | `Engine.run_loop` = `while not terminate and steps < max_steps: next_gm → observe all → next_acting → act → resolve → log → checkpoint → step_callback`. Sequential / simultaneous / asynchronous variants share the same five GM queries. `SKIP_THIS_STEP` for setup steps. `StepController` for play/pause/single-step from another thread. | The **fixed enum of engine→GM queries** (`MAKE_OBSERVATION, NEXT_ACTING, NEXT_ACTION_SPEC, RESOLVE, TERMINATE`) is a clean minimal protocol; sequential vs simultaneous as the two engine kinds; `max_steps` + explicit terminate; `step_callback`/`checkpoint_callback` hooks; one-action-per-step. | Make the engine deterministic and single-threaded by default; pass a run seed; have `resolve` return a typed transition `(state', events, payoffs)` rather than a string; validate every action against the spec in the engine (as v1 did). |
| **Entities / agents** | `Entity{name, act(spec)->str, observe(str)}`; components with `pre_act/post_act/pre_observe/post_observe/update`; one `ActingComponent`; `get_state/set_state` for checkpoints; `EntityWithLogging.get_last_log()`. Prefabs = typed factories with `params`. | The tiny `Entity` surface; separation of *context providers* from the single *policy*; `get_state/set_state` and `get_last_log` as mandatory parts of the contract; prefab + `InstanceConfig(prefab, role, params)` for configuration. | Type `act` as `Observation -> Action` with an `Action` ADT (choice enum / float / text) and let the policy be any Python callable — so a rational/best-response policy is a first-class agent, not a mock LLM. Keep the component hook pattern optional (LLM agents) rather than the base contract. |
| **Game master** | GM is an entity with `SwitchAct` routing each `OutputType` to a component (`MakeObservation`, `NextActing*`, `NextActionSpec*`, `EventResolution`, `Terminate*`, `SceneTracker`, `PayoffMatrix`, `Inventory`, `WorldState`, …), LLM fallback when a component is absent. | The idea that the GM is a *bundle of pluggable answers to the engine's questions*, with explicit separation: observation (information set), scheduling (who acts), action-spec (what they may do), resolution (transition), termination. `ObservationQueue` as the single channel through which anything reaches a player. `identity` resolution + `allow_llm_fallback=False` show the "GM as pure rules" configuration. | Make the GM a typed `Game` object: `initial_state`, `legal_actions(state, player) -> ActionSpec`, `observe(state, player) -> Observation`, `transition(state, joint_action) -> (state', events)`, `payoff`, `is_terminal`. Allow an *optional* LLM narrator component layered on top (Dramatist mode), never in the control path by default. |
| **Components / state** | Component state is free text for prompting plus a JSON-ish `ComponentState` for checkpoints; grounded variables live in Python but are updated via LLM Q&A (`Inventory`, `WorldState`, `Locations`). | `ItemTypeConfig(min, max, force_integer)` + `apply(fn)` with validation is the right shape for grounded variables; rendering typed state to text for prompts is necessary and should be a single `render(state, player)` function. | Store state in dataclasses/pydantic; derive prompt text from state, never the reverse; if LLM-mediated state updates are ever needed, make them an explicit, logged, validated "parser" step. |
| **Actions** | `ActionSpec(call_to_action, output_type ∈ {FREE, CHOICE, FLOAT}, options, tag)`; per-scene / per-player specs via `SceneTypeSpec.action_spec`; `sample_choice` constrained choice; option shuffling to fight position bias (`randomize_choices=True`). | `ActionSpec` as a small sum type; per-scene/per-player specs; constrained choice + shuffled labels + `validate`; `tag` for distinguishing speech vs action. | Add structured actions (records with typed fields) and keep FREE text as an explicitly separate "utterance" channel. |
| **Observations** | GM-generated text; queued events delivered on the next `make_observation`; per-player premises in scenes; `notify_observers`/`SendEventToRelevantPlayers` decide awareness (LLM). | The *queue* model (events accumulate, delivered at the player's next turn) and scene premises. | Declare observation structure per player in the game spec; compute awareness from state, not from an LLM. |
| **Scenario config** | `Config(prefabs, instances: [InstanceConfig], default_premise, default_max_steps)`; `SceneSpec`/`SceneTypeSpec` (participants, rounds, per-player premise, action spec, GM per scene); `examples/games/scenarios.py` registry of `ScenarioConfig(config_module, background_agent_prefab, num_background_players, tags, …)`; focal vs background vs supporting players. | Scenes as the unit of structure (phase of a game with its own action spec and participants); scenario registry with tags; focal/background distinction for evaluation; `puppet` scenarios for deterministic scoring tests. | Make the scenario file fully declarative (no callables in config — Concordia's README admits callables make configs "harder to serialize as JSON"). |
| **Logging** | Per-component dict logs → per-step raw log → `SimulationLog` (dedup content store, entries with `step/entity_name/component_name/entry_type/summary/deduplicated_data`) → JSON/HTML, `concordia-log` CLI, `AIAgentLogInterface`. Prompts stored verbatim. | Always store the full prompt and raw model output per decision; content-addressed dedup of long strings; a query API for logs; an HTML viewer; `bundle` into a self-contained file. | Add a typed event log (`StepRecord{step, actor, action_spec, action, state_before, state_after, payoffs, llm_calls[]}`) that is the *source of truth*, from which metrics are computed without regexes; record LLM responses so a run can be replayed with a "replay model". |
| **Metrics** | Payoff components (`action_to_scores` callables), questionnaires, LLM autorater over the log with a weighted rubric. | Autorater-as-separate-pass over a frozen log (not inside the run); questionnaires as a GM mode; payoffs as pure functions of the joint action. | Compute payoffs/metrics from typed state; exact rational baselines become possible once `legal_actions`/`transition`/`payoff` are pure Python. |
| **Checkpoint / replay** | `step_{n}_checkpoint.json` of component states + raw log; resume but not deterministic replay. | `get_state/set_state` discipline on every stateful object; checkpoint after every step. | Seeded RNG object threaded through everything; recorded LLM I/O for replay; no threads in the control loop. |

### Bottom-line borrowings for the new library
1. The **engine ↔ GM protocol** as a closed enum of five questions, with sequential and simultaneous engines sharing it.
2. **GM = composition of answerers** (schedule, observe, action-spec, resolve, terminate) — but implement them as typed methods of a `Game` object, not as prompt-producing components.
3. **`ActionSpec`** (`CHOICE | FLOAT | FREE`, options, tag) with per-scene/per-player specs; constrained choice with shuffled labels.
4. **Scenes** as declarative phases with participants, rounds, premises and action specs; **scenario registry** with focal/background roles and `puppet` deterministic test scenarios.
5. **Observation queue** semantics (events accumulate, delivered at next act), with awareness computed from state.
6. **Logging discipline**: every decision logs prompt + raw output + parsed value; dedup content store; query interface; HTML bundle.
7. **Mandatory `get_state/set_state`** on anything stateful, checkpoint every step.
8. Avoid: free-text state, LLM-maintained grounded variables, LLM in the control path by default, threading in the loop, global RNG, callables in config, "metrics by grep".

---

## Sources

Repository (google-deepmind/concordia, `main` unless noted)
- https://github.com/google-deepmind/concordia and README: https://raw.githubusercontent.com/google-deepmind/concordia/main/README.md
- Releases: https://github.com/google-deepmind/concordia/releases
- CHEATSHEET: https://raw.githubusercontent.com/google-deepmind/concordia/main/CHEATSHEET.md
- setup.py: https://raw.githubusercontent.com/google-deepmind/concordia/main/setup.py ; pyproject: https://raw.githubusercontent.com/google-deepmind/concordia/main/pyproject.toml
- typing: https://raw.githubusercontent.com/google-deepmind/concordia/main/concordia/typing/entity.py , .../typing/entity_component.py , .../typing/prefab.py , .../typing/scene.py , .../typing/simulation.py , .../typing/logging.py , .../typing/README.md
- agents: .../concordia/agents/entity_agent.py , .../concordia/agents/entity_agent_with_logging.py
- environment: .../concordia/environment/engine.py , .../environment/step_controller.py , .../environment/README.md , .../environment/engines/sequential.py , .../engines/simultaneous.py , .../engines/asynchronous.py
- GM components: .../concordia/components/game_master/{switch_act,make_observation,next_acting,event_resolution,terminate,scene_tracker,world_state,inventory,payoff_matrix,instructions}.py ; README: .../concordia/components/README.md
- agent components: .../concordia/components/agent/{memory,observation,concat_act_component,scripted_act,puppet_act}.py
- prefabs: .../concordia/prefabs/README.md , .../prefabs/entity/{basic,rational}.py , .../prefabs/game_master/{generic,game_theoretic_and_dramaturgic}.py , .../prefabs/simulation/generic.py
- memory / document / model: .../concordia/associative_memory/basic_associative_memory.py , .../concordia/document/interactive_document.py , .../concordia/language_model/{language_model,no_language_model}.py , .../concordia/utils/{measurements,structured_logging,sampling}.py
- docs/skills: .../concordia/docs/skills/analyze-logs.md , .../concordia/docs/skills/autorater-evals.md
- examples: .../examples/README.md , .../examples/games/run.py , .../examples/games/scenarios.py , .../examples/games/haggling/{sequential_bargain_payoff,sequential_bargain_game_master}.py
- v1 comparison: https://raw.githubusercontent.com/google-deepmind/concordia/v1.8.10/concordia/environment/game_master.py , https://raw.githubusercontent.com/google-deepmind/concordia/v1.8.10/concordia/metrics/goal_achievement.py , https://github.com/google-deepmind/concordia/tree/v1.8.10/concordia/metrics

Papers
- Vezhnevets et al. 2023, "Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia", arXiv:2312.03664 — https://arxiv.org/html/2312.03664
- Vezhnevets et al. 2025, "Multi-Actor Generative Artificial Intelligence as a Game Engine", arXiv:2507.08892 — https://arxiv.org/html/2507.08892
- Smith et al. 2025, "Evaluating Generalization Capabilities of LLM-Based Agents in Mixed-Motive Scenarios Using Concordia", arXiv:2512.03318 — https://arxiv.org/abs/2512.03318
- NeurIPS 2024 Concordia Contest: https://neurips.cc/virtual/2024/competition/84791 , https://www.cooperativeai.com/contests/concordia-2024
