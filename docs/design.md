# Design: `simspec`

Status: design of record, updated after implementation (Phase 2). Where the implementation differs from the pseudo-code below, the implementation is authoritative and the difference is noted in `docs/decisions.md`. The module map is: `simspec/domains.py`, `spec.py`, `rules.py`, `chance.py`, `log.py`, `card.py`, `runtime.py`, `agents/{base,llm,decision,memo_agent}.py`, games under `simspec/games/{contractsim,norms}`.

## 1. Scope and the four contributions

The runtime is a commodity. The contributions are:

1. a typed, Python-embedded **game spec** (state, actions, per-role observation structure, phases, transition, payoffs, rules);
2. **enforced observation structure**: an agent cannot read state it is not declared to observe, and a planted leak is caught by `spec.check()` before any model call;
3. **`Rule`** as a first-class object (prohibitions, obligations, contract clauses) with compliance checking, sanctions and payments, serialisable and enumerable;
4. **memo-based rational baselines** generated from the spec, so an LLM agent and an exact Bayesian/decision-theoretic agent play the same game through the same interface;
5. a **standard trajectory log** plus a **Harness Card** that makes a run reproducible and comparable.

Everything else (loop, sampling, retries) is deliberately thin and borrows from PettingZoo/OpenSpiel (semantics), Concordia (engine↔game-master protocol, `ActionSpec`), Inspect/clembench (log events).

## 2. Core types

All spec objects are plain Python with `pydantic` models where serialisation matters (domains, rules, phases, harness card, log events) and frozen dataclasses for runtime records. Everything typed, finite and enumerable unless declared as a text slot.

### 2.1 Domains

```python
class Domain(BaseModel):                  # finite, ordered, enumerable
    name: str
    def size(self) -> int
    def values(self) -> list[Any]
    def encode(self, v) -> int             # value -> index
    def decode(self, i: int) -> Any
    def to_memo(self) -> str               # Python source for the memo domain (IntEnum / arange / memo.domain)

class Enum(Domain):   labels: list[str]
class IntRange(Domain): lo: int; hi: int                 # inclusive
class Grid(Domain):   lo: float; hi: float; n: int       # discretised real
class Product(Domain): fields: dict[str, Domain]         # mixed-radix, maps onto memo.domain(...)
class Text(BaseModel): name: str; max_chars: int         # declared free-text slot; NOT a Domain
```

`Product` is the bridge to memo: `Product.encode(dict)` gives the integer memo sees, and `to_memo()` emits `memo.domain(a=n_a, b=n_b, ...)`.

### 2.2 State, actions, roles

```python
class Field(BaseModel):
    name: str
    domain: Domain | Text | "Table"        # Table = array of a Domain (e.g. inventory per ingredient)
    visibility: Literal["public", "private", "hidden"]   # default hidden
    owner: str | None                       # role name for private fields

class StateSchema(BaseModel):  fields: list[Field]
class State:                   # runtime record; immutable; keyed by field name
```

Roles are declared once (`roles: list[str]`); a *seat* binds a role to an agent in a scenario. Actions are per-phase `ActionSpec`s:

```python
class ActionSpec(BaseModel):
    name: str
    kind: Literal["choice", "record", "text"]
    domain: Domain | None                   # choice: finite; record: Product of fields
    text: Text | None                       # text slot (contract text, messages)
    legal: Callable[[State, str], Mask] | None   # mask over domain for (state, role)
    call_to_action: str                     # Concordia-style prompt fragment
```

`kind="text"` actions are the only place free text enters the game.

### 2.3 Observation structure (the enforced part)

```python
class View(BaseModel):                      # what one role sees
    role: str
    fields: list[str]                       # state fields; must be public or private-to-role
    derived: dict[str, Derived]             # named features computed from the state
    channels: list[str]                     # message channels this role reads (public, or addressed to it)
    history: Literal["none", "own", "all_visible"]

class Derived(BaseModel):
    name: str
    reads: list[str]                        # declared state fields it may touch
    domain: Domain
    fn: Callable[[State], Any]              # evaluated on a tracing proxy at check time
```

Enforcement has three layers, all exercised by `spec.check()`:

1. **Static**: every `View.fields` entry must be `public` or `private` with `owner == role`; every `Derived.reads` entry likewise. Violations raise `ObservationLeak` with the field and role.
2. **Dynamic, pre-run**: each `Derived.fn` is executed on a `TracingState` proxy over sample states (initial state plus a short scripted rollout); any attribute read outside `reads` raises `ObservationLeak`. This catches a planted leak in a feature function.
3. **Runtime**: the engine keeps the state in a `Vault`. Before calling an agent it materialises the role's `Observation` (an immutable mapping containing only the view's fields, derived values and channel messages), then **locks** the vault; any access to `vault.state` while locked raises `ObservationLeak`. Agents are constructed without a vault reference, so the only way to leak is to smuggle a reference, and the first attempt fails inside the agent's `act` before any model call. `spec.check(scaffolds)` runs one dry-run episode with all model transports replaced by a `DryRunTransport` that raises if invoked, so a scaffold that builds its prompt from leaked state is caught before any real call.

For memo agents the observation view is compiled into `knows`/`observes` statements, so memo's own static "no mind reading" check applies.

### 2.4 Phases and protocols

```python
class Phase(BaseModel):
    name: str
    kind: Literal["simultaneous", "sequential", "protocol"]
    actors: list[str] | Callable[[State], list[str]]     # who acts; sequential = in order
    action: ActionSpec | dict[str, ActionSpec]           # per role
    max_rounds: int | None
    next: str | Callable[[State], str | None]            # next phase, or None = terminal
```

A **protocol** phase is a sequential phase whose transition is the protocol function `alpha(history) -> Continue | Agreed(payload) | Disagree`; it is how negotiation is expressed. The negotiated payload (contract text) is parsed into `Rule`s by a declared `Translator` before the next phase.

### 2.5 Transition, chance, payoffs

```python
transition(state: State, joint: dict[str, Action], rng: Chance) -> tuple[State, list[Event]]
payoff(trajectory: Trajectory) -> dict[str, float]      # trajectory-level, as in ContractSim
reward(state, joint, next_state) -> dict[str, float]    # optional per-step, as in the norms game
terminal(state) -> bool
```

`Chance` is a seeded wrapper around `numpy.random.Generator` whose every draw is logged as a `ChanceEvent(name, value)`; replay-by-readback feeds the logged draws back, so transitions are deterministic functions of (state, joint action, logged chance).

### 2.6 Rules

```python
class Rule(BaseModel):
    id: str
    kind: Literal["prohibition", "obligation", "constraint"]
    scope: list[str]                        # roles it binds
    text: str | None                        # declared free-text source (contract clause), optional
    pre: Predicate | None                   # prohibition: Post on (s, a, s'); obligation: trigger on s
    post: Predicate | None                  # obligation: discharge condition
    action: str | None                      # prohibition: action class restricted
    deadline: int | None                    # obligation: τ steps
    sanction: Sanction | None               # payment transfer, action restriction, release
    tolerance: float = 0.0                  # ε for probabilistic satisfaction (ContractSim)

class Predicate(BaseModel):                 # small serialisable expression language
    # Cmp(feature, op, value) | And | Or | Not | ActionIs(cls) | Field(name)==... | PyPred(source_hash) escape hatch
```

Semantics (matching the norms paper, generalised): a prohibition is violated at step t if `action ∈ action_class` and `post(s_{t+1})`; an obligation is triggered when `pre(s_t)`, discharged when `post(s_j)` for some `t ≤ j ≤ t+deadline`, violated otherwise. A `constraint` is an arbitrary history classifier `ν(h) ∈ {0,1}` (ContractSim's `C^ω`). The `RuleEngine` evaluates all rules per step per bound role, emits `RuleEvent(rule_id, role, status ∈ triggered|discharged|violated)`, applies sanctions (payoff transfers via the transition, action masks via `legal`, obligation release). Rules are serialisable so they can appear in the Harness Card and be enumerated from a feature grammar (`RuleSpace`).

### 2.7 Agents

```python
class ActionDistribution(BaseModel):
    probs: dict[str, float]                 # over encoded action keys; sums to 1
    source: Literal["exact", "scripted", "native", "logprob", "verbalized", "degenerate"]
    text: str | None                        # for text actions: the sampled text (probs == {text: 1})
    meta: dict                              # model, resolved version, latency_ms, usage, raw, prompt_hash

class Agent(Protocol):
    role: str
    def act(self, obs: Observation, spec: ActionSpec, ctx: ActContext) -> ActionDistribution
    def observe(self, obs: Observation, ctx: ActContext) -> None     # optional: non-acting updates
    def get_state(self) -> dict; def set_state(self, s: dict) -> None
```

The **runtime samples** from the distribution with the episode RNG and logs both the distribution and the sampled action. This is what makes LLM-vs-rational comparisons direct and replay exact. Agents never see `State`.

Adapters: `ScriptedAgent` (policy table or callable), `LLMAgent` (scaffold config + transport), `MemoAgent` (generated memo module + information-state encoder), `DecisionModelAgent` (TypeSafe-style or OpenAI-style wire format), `ReplayAgent` (reads actions from a log).

### 2.8 Trajectory log

One directory per run: `card.json` (Harness Card), `events.jsonl` (typed events in order), `calls.jsonl` (raw model requests/responses, content-addressed), `scores.json` (written by offline metrics, never by the run). Event kinds:

`EpisodeStart{seed, card_hash}` · `PhaseStart{phase, round}` · `Observation{step, role, obs}` · `AgentCall{step, role, kind, prompt_hash, distribution, source, usage, latency_ms, call_ref}` · `Action{step, role, raw, parsed, valid, retries}` · `Chance{step, name, value}` · `Transition{step, state_after}` (full snapshot; states are small) · `Rule{step, rule_id, role, status, detail}` · `Payoff{step|final, per_role}` · `Message{step, channel, from, to, text}` · `Error{...}` · `EpisodeEnd{returns}`.

`Trajectory` is the in-memory view of one episode's events with helpers (`states()`, `actions(role)`, `rule_events(role)`), from which `Metric`s are computed offline.

### 2.9 Harness Card (`simspec/card.py`)

Two sections, hashed separately and together:

- **game**: spec id and version, source hash of the spec module, parameter values (environment id, horizon, caps), rule set (serialised), roles and seats, phases.
- **scaffold** (per seat): agent class; model id and resolved version; prompt templates with hashes; context policy (history window, memory, recalled observations); decoding parameters; action elicitation (choice list order, shuffling, call-to-action); parsing and retry policy (max retries, fallback action); distribution source; tool use; budgets.
- **runtime**: library versions (`simspec`, `memo-lang`, `jax`, provider SDKs), seed, replay mode, timestamp.

`run_from_card(card, log)` re-executes an episode: with `ReplayAgent`s it must reproduce `events.jsonl` exactly (card test); with live agents it reproduces the configuration.

## 3. Runtime: thin layer or custom? (decided: D-006)

Both candidates were probed hands-on (`docs/research.md` §6c). Inspect AI's `EvalLog` can carry our events and card, so it is an **export** (`simspec/export/inspect_log.py`); LangGraph's checkpointer re-executes agents on replay and its fork overwrote the original thread in the probe, so it is not used. The engine (`simspec/runtime.py`, about 300 lines) owns phases, the vault lock, chance logging, rule evaluation, JSONL events with full state snapshots, replay-by-readback and the checker.

Implemented signatures that differ from §2: `ActionSpec` gained `valid` (a predicate for non-enumerable record actions) and `classes` (key → action class for prohibitions); `GameSpec` gained `dynamic_rules` (rules created at run time, e.g. from a negotiated contract) and `forward` (a single-agent forward model for planners); `ActContext` carries `log_call` and `log_info` so agents can write raw model calls and belief traces into the log without touching state.

## 4. Game A as a pseudo-spec: ContractSim (Catering)

```python
Ingredient = Enum("ingredient", ["mozzarella", "basil", "tomato"])
Product    = Enum("product", ["pizza", "pasta", "soup"])
Qty        = IntRange("qty", 0, 12)
Money      = IntRange("money", 0, 400)

state = StateSchema([
  Field("week", IntRange(1, 11), "public"),
  Field("phase_kind", Enum(["negotiation", "payment", "production", "done"]), "public"),
  Field("budget_remaining", Money, "private", owner="customer"),
  Field("total_paid", Money, "public"),
  Field("balance", Money, "private", owner="supplier"),
  Field("inventory", Table(Ingredient, IntRange(0, 10)), "private", owner="supplier"),
  Field("prices", Table(Ingredient, Grid(1, 3, 3)), "private", owner="supplier"),
  Field("delivered", Table(Product, Qty), "public"),          # this week's deliveries
  Field("cum_delivered", Table(Product, IntRange(0, 60)), "public"),
  Field("contract", ContractRecord, "public"),                  # parsed C^ω: prices, schedule, payments, clauses
  Field("contract_text", Text(4000), "public"),
  Field("values", Table(Product, IntRange(0, 30)), "private", owner="customer"),   # θ_cust
  Field("env_params", EnvParams, "hidden"),                     # stochastic laws (supplier knows them: derived view)
  Field("shortfall_memory", ..., "public"),
])

phases = [
  Phase("negotiate", "protocol", actors=["customer", "supplier"],
        action={"customer": ActionSpec("turn", "record", Product(response=Text, contract=Text, accept=Bool, terminate=Bool)),
                "supplier": ActionSpec("turn", "record", Product(response=Text, contract=Text, terminate=Bool))},
        max_rounds=50, protocol=contractsim_alpha, next="payment"),
  Phase("payment", "sequential", actors=["customer"], action=ActionSpec("pay", "choice", Money, legal=le_budget), next="production"),
  Phase("production", "sequential", actors=["supplier"],
        action=ActionSpec("order_and_produce", "record", Product(order=Table(Ingredient, Qty), produce=Table(Product, Qty)), legal=affordable_and_feasible),
        next=lambda s: "payment" if s.week < 11 else None),
]

views = {
  "customer": View(fields=["week", "total_paid", "budget_remaining", "delivered", "cum_delivered", "contract", "contract_text", "values"], channels=["dialogue", "proposals"]),
  "supplier": View(fields=["week", "total_paid", "balance", "inventory", "prices", "delivered", "cum_delivered", "contract", "contract_text"], derived={"laws": Derived(reads=["env_params"], ...)}, channels=["dialogue", "proposals"]),
}

rules = contract_rules(contract)   # built by the Translator after agreement:
#   Obligation(customer, pre=is_payment_week(w), post=paid >= M_w - allowed_deduction, deadline=0)
#   Obligation(supplier, pre=is_production_week(w), post=delivered >= q_w (after substitution/rollover), deadline=0)
#   Constraint(both, C^ω absorbing prefix indicator, tolerance=0.05)
#   clauses: payment_deduction -> modifies allowed_deduction; rollover -> extends supplier deadline by one production week with cap 2; substitution -> changes the post predicate; grim_trigger -> Sanction(release_all_obligations_for(victim))

payoff = lambda τ: {"customer": B + Σ v_d q'_d − Σ M_paid, "supplier": Σ M_paid − Σ o·p}
```

Strain points found while writing this:
- **Negotiation is a protocol over text actions**, so `ActionSpec.kind="record"` with text sub-fields and a `protocol` function is needed; a plain transition function would have to re-implement turn legality (only the customer accepts, only the previous-round proposal). Resolution: the `protocol` phase kind, with the protocol function owning legality.
- **The contract becomes rules at run time.** Rules are created after the negotiation phase by the Translator, so the rule set is part of the state (serialised into the snapshot and the log), not only of the spec. The Harness Card records the Translator (model, schema version, prompt hash), not the rules.
- **Continuous money**: budgets and payments are integers in the paper, so `IntRange` works; prices are a 2-point distribution per ingredient, so `Grid` with 3 points is exact.
- **Supplier knows its stochastic laws, the customer does not**: expressed as a hidden field plus a derived feature on the supplier view with declared `reads`. This is exactly the case the leak checker must get right.
- **Memo baseline size**: the full supplier problem has ~10^6–10^7 states per week with 13^3 orders. The memo Supplier agent uses a declared `abstraction`: fulfilment level ∈ {none, partial, full} mapped to an order by the paper's own heuristic production rule, with belief over the customer's type (payer / defector) updated from payment history; the memo Customer agent plans payments against a belief over the supplier's type (reliable / unreliable) updated from deliveries. Both are exact given the abstraction and both are generated from the view (so they cannot see hidden fields). The paper's RC/RE/RCC are scripted agents; RC's exact DP is implemented in plain JAX at reduced caps if time permits.

## 5. Game B as a pseudo-spec: norm-augmented harvest gridworld

```python
Cell  = Product("cell", x=IntRange(0, W-1), y=IntRange(0, H-1))
Act   = Enum("act", ["NOOP","FORWARD","BACKWARD","STEP_LEFT","STEP_RIGHT","TURN_LEFT","TURN_RIGHT","FIRE_ZAP","FIRE_CLEAN","FIRE_CLAIM","EAT","PAY"])
Role  = Enum("role", ["cleaner", "farmer", "free", "learner"])

state = StateSchema([
  Field("t", IntRange(0, 300), "public"),
  Field("apples", Table(Cell, Bool), "public"), Field("desiccated", Table(Cell, Bool), "public"),
  Field("dirt", Table(RiverCell, Enum(["clean","potential","dirt"])), "public"),
  Field("property", Table(Cell, IntRange(0, N)), "public"),
  Field("agents", Table(AgentId, Product(pos=Cell, orient=IntRange(0,3), role=Role, inventory=IntRange(0, 50), since_cleaned=..., since_paid=..., frozen=...)), "public"),
  Field("stolen_records", Table(AgentId×AgentId, Int), "public"),
  Field("norm_beliefs", Table(AgentId×RuleId, Grid(0,1,101)), "private", owner=each agent),   # per-agent belief state lives in the agent, mirrored for logging
])

phases = [Phase("step", "simultaneous", actors=all_agents, action=ActionSpec("act", "choice", Act, legal=in_bounds), max_rounds=300)]

views = {role: View(fields=[public fields], derived={
   "DIRT_FRACTION": Derived(reads=["dirt"]), "NUM_APPLES_AROUND": Derived(reads=["apples","agents"]),
   "CUR_CELL_HAS_APPLE", "CUR_CELL_IS_FOREIGN_PROPERTY", "SINCE_AGENT_LAST_PAID", "AGENT_LOOK", "ORIENTATION", "RIOTS"}) for role in roles}

rule_space = RuleSpace(features={...paper's feature grid...}, actions=["MOVE"], max_len=3)   # enumerates the 68 norms
default_rules = [Prohibition("P1", action="MOVE", post=Cmp("CUR_CELL_HAS_APPLE")&Cmp("NUM_APPLES_AROUND","<",3)), Prohibition("P2", ...),
                 Obligation("O1", scope=["farmer"], pre=Cmp("SINCE_AGENT_LAST_PAID",">",30), post=Cmp("SINCE_AGENT_LAST_PAID","==",0), deadline=20), ...]
reward = per-step: +1 apple eaten, −0.01 per action, −1 per violation of an adopted rule, +1 obligation discharged, −15 zapped
```

Strain points:
- **Rules are evaluated on the agent's observation features, not raw state** (the paper's rules read `obs[...]`). Resolution: `Predicate`s reference `View.derived` names; the `RuleEngine` evaluates a rule for role r on r's observation record. This makes rules first-class *and* observation-respecting, and is what lets the same `Rule` type serve both games.
- **Prohibitions are checked on the predicted next observation** (the planner masks actions whose successor satisfies `post`). The engine needs `transition` to be callable as a pure forward model for one agent with others frozen, so `GameSpec.forward(state, role, action)` is exposed for planners. That is a mild extension of the API.
- **Norm beliefs are agent state, not game state.** They are logged through `Agent.get_state()` snapshots, not as a state field. The pseudo-spec above shows them as a private field only to make the logging requirement explicit; the implementation keeps them in the agent.
- **Scale**: the full map is 23×22 with 12 actions and up to 6 agents; the simultaneous phase and the forward model are plain NumPy. memo is used for the exact posterior over single-rule hypotheses given Q tables (`observer: thinks[agent: chooses(rule), agent: chooses(a, wpp=mixture)]; observes a`) with a `Product` over (rule, local features) as the axis, not for planning over the map.

## 6. What is shared between the two games (measured)

Shared: domains, state schema and vault, views and the leak checker, phases and engine, chance logging, `Rule` and `RuleEngine`, `RuleSpace` enumeration, agents (scripted, LLM scaffold, decision model, memo module builder, replay), log, Harness Card, adapters. Game-specific: state fields, transition/forward model, payoff, protocol function and translator (A), rule-space feature grid and planner (B), prompts, metrics.

`scripts/shared_code.py` reports (2026-10-16): shared library 1122 code lines; ContractSim 848 game-specific lines (shared fraction 0.57); norms 503 (0.69). Dynamically, of the lines executed by an episode, 64% (ContractSim) and 63% (norms) are shared library lines. The prediction of "more than 70%" was not met: the game-specific code is dominated by the two simulators themselves (ContractSim's production-week accounting and clause semantics; the gridworld mechanics and planner), which is irreducible, and by the memo baselines, which are game-specific by design.
