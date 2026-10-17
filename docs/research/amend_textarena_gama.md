# Amendment: TextArena, MindGames, GAMA vs. the (i)+(ii)+(iii) claim

Research date: 2026-10-16. Claim under test: *no existing framework offers (i) enforced observation
structure + (ii) first-class Rules with compliance checks + (iii) rational (non-LLM, exact-inference)
baseline agents together.* Each framework below is checked on the full paper text (arXiv HTML) and
the GitHub code at the commit noted. Quotes are verbatim.

Code snapshots read:
- TextArena `LeonGuertler/TextArena` @ `1e3160dea659971f120bf42708fafa8ea3001b81` (2026-10-03), 114 env dirs.
- MindGames starter kit `mind-games-challenge/mindgames-starter-kit` @ `b7ff02ab4a6b01398d186290e0981c37e31ba3ab` (2026-05-04).
- GAMA (now "MAGiF") `dicelab-rhul/GAMA` @ `c26eafd5d11ef163981a6e0372e2323f7ca5ca17` (2025-12-17).

---

## 1. TextArena (arXiv 2504.11442; Guertler et al.)

### What it standardizes
- **A Gym-style text API**, not a game format. Paper §2: "we kept the code interfaces used as similar to
  OpenAI Gym (aka Gymnasium) ... as possible and adopted their philosophy of stack-able wrappers."
  Loop (`README.md`, paper §2): `player_id, observation = env.get_observation(); action = agents[player_id](observation); done, info = env.step(action=action); rewards, game_info = env.close()`.
- **Agent contract is a string->string callable**: `textarena/core.py`, `class Agent(ABC)`:
  `def __call__(self, observation: str) -> str`. README: "Agents only need to implement a `__call__`
  function that accepts a string observation and returns a string action."
- **Leaderboard**: online TrueSkill ladder (paper §4: "The online leaderboard tracks performance through
  TrueSkill™ ... Each model is initialized with a TrueSkill™ rating"), plus per-env "soft skill" tags.
- **No game-description format**: each game is a hand-written Python `Env` subclass under
  `textarena/envs/<Name>/env.py` with its own `reset`/`step` and regex action parsing
  (e.g. `KuhnPoker/env.py`: `re.compile(r"\[(Check|Bet|Fold|Call)\]", re.IGNORECASE)`).
- Type aliases in `core.py`: `Message = Tuple[int, str, ObservationType]`,
  `Observations = dict[int, List[Message]]`, `Rewards = Dict[int, int]`.

### What it assumes
- Competitive *and* cooperative/mixed: README: "evaluating and training language-model agents through
  competitive and cooperative text-based games"; the paper title/abstract says "competitive text-based
  games". Envs include IPD, PublicGoodsGame, Negotiation, Codenames (cooperative teams), SecretMafia.
- **Text state and text actions only.** Paper §3: "all games used are either naively text-based or
  adapted to be text-based." State is an untyped `Dict[str, Any]` (`State.standard_resets(game_state=...)`).
- **Turn-based by construction; simultaneous moves are emulated.** `State` has a single
  `current_player_id`; `get_observation()` returns `(self.state.current_player_id, ...)`. Simultaneous
  games (IPD, ColonelBlotto, Blotto) take sequential private submissions and resolve when all are in:
  `IteratedPrisonersDilemma/env.py` `_handle_decision_phase`: `# resolve only after both players decided`
  `if all(d is not None for d in self.state.game_state["decisions"].values()): self._resolve_round()`.
- Players: 1, 2, N (`SinglePlayerState`, `TwoPlayerState`, `FFAMultiPlayerState`, `TeamMultiPlayerState`
  in `textarena/state.py`). Rewards are terminal scalars per pid (`set_winner` -> `{player_id: 1, 1-player_id: -1}`).

### Observation structure: declared? enforced?
- There is a **per-player observation inbox** (`State.observations = {pid: []}`), and envs route
  messages with `add_observation(to_id=...)` (`to_id=-1` = broadcast). Only the acting player's inbox is
  returned and then cleared: `get_current_player_observation()` does
  `current_player_observation = self.observations[self.current_player_id]; self.observations[self.current_player_id] = []`.
- **This is convention, not enforcement.** (a) The full hidden state lives in plain Python on the env
  object (`self.state.game_state["player_cards"]`, SecretMafia `self.player_roles`), and the agent is
  called *by the user's driver loop*, which holds `env`; nothing stops `agents[pid]` from being given
  `env.state.game_state`. (b) There is no declared information set per player; what a player sees is
  whatever the env author happened to `add_observation(to_id=p)`; a missing `to_id` leaks to everyone.
  (c) `Wrapper.__getattr__` forwards everything, so wrapped envs expose `state` too.
  (d) `secret_roles` exists only to label logs: `core.py` comment: `"role": ... # important for RL
  training on games like secret mafia`.
- Verdict: **(i) not satisfied** — per-player message routing exists, but there is no declared,
  type-checked, enforced information set.

### First-class rules / contracts / norms with compliance checks?
- **No.** Rules are prose in the player prompt (`_prompt()` returns localized text such as
  `self.m("player_prompt", "intro", ...)`) and legality is ad-hoc regex/if-logic in `step()`. The only
  generic "compliance" object is `set_invalid_move(reason)` with an `error_allowance` counter
  (`state.py`), i.e. format/legality, not norms or contracts. Negotiation envs (Negotiation,
  SimpleNegotiation, TwoDollar) implement deals in env code, not as declarable rule objects.
  Nothing matching `norm|contract|obligation` as an abstraction exists in `textarena/core.py` or `state.py`.
- Verdict: **(ii) not satisfied.**

### Rational / exact baseline agents?
- **None.** `textarena/agents/basic_agents.py` classes: `HumanAgent, OpenRouterAgent, GeminiAgent,
  OpenAIAgent, HFLocalAgent, CerebrasAgent, AWSBedrockAgent, AnthropicAgent, GroqAgent, OllamaAgent,
  LlamaCppAgent` — all LLM/human. `grep -rliE "mcts|minimax|nash|equilibri|alpha.?beta|cfr"` over the
  package hits only `envs/utils/jury.py`, which is an **LLM jury** (`OpenRouterJury`, `default_models =
  ["amazon/nova-pro-v1", "openai/gpt-4o-mini", ...]`) used for judged games, not a solver.
- Verdict: **(iii) not satisfied.**

### Trajectory / log format
- In-process only: `State.logs: List[Tuple[int, str]]` — one canonical `(from_id, text)` per
  `add_observation` (`core.py` comment: "Logs: one canonical entry ... Choose what's most useful for
  replay/debug"). No file/JSON writer in the library (`grep json.dump` hits only websocket code in
  `api.py`). `env.close()` returns `(rewards, game_info)` with `game_info[pid] = {"role", "invalid_move",
  "turn_count", "reason"}`. The paper promises a data release ("We will release datasets including
  game-play trajectories"), delivered later as the MindGames dataset (see §2 below).
- Observations are untyped text lines tagged with `ObservationType` enum
  `{PROMPT, PLAYER_ACTION, GAME_ACTION_DESCRIPTION, GAME_MESSAGE, GAME_BOARD, GAME_ADMIN}`.

### Exporter from a typed spec -> TextArena
- **Cheap for the mechanics, lossy for the guarantees.** A generic `ta.Env` subclass can be generated:
  typed state -> `game_state` dict; per-player observation projection -> `add_observation(to_id=p, ...)`
  strings; typed actions -> a regex grammar + parser in `step()`; simultaneous rounds -> the IPD
  "collect then resolve" pattern; payoffs -> `set_game_outcome`. Cost drivers: (1) every action/obs must be
  rendered to and parsed from free text; (2) information-set enforcement is lost (the host loop can read
  `env.state`); (3) rule/contract compliance must be re-implemented as `set_invalid_move` calls or
  reward shaping; (4) rewards must be flattened to terminal scalars (MindGames: "Players do not receive
  intermediate rewards"). Benefit: access to the TrueSkill ladder via `ta.make_online` / `register_model` (`api.py`).

---

## 2. MindGames (arXiv 2605.29512; "MindGames: A Live Arena for Evaluating Social and Strategic Reasoning in Multi-Agent LLMs")

Code: no separate engine — `mind-games-challenge/mindgames-starter-kit` (`pip install textarena>=0.7.2`),
4 env copies under `envs/` (Codenames, ColonelBlotto, SecretMafia, ThreePlayerIPD), top submissions as
git submodules, and the dataset `huggingface.co/datasets/mindgameschallenge/MGC2025`.

### What it standardizes
- A **benchmark protocol on top of TextArena**: 4 envs, 2 tracks (Generalization: Blotto+IPD+Codenames;
  Social Deduction: SecretMafia), 2 divisions (Efficient <9B open weights / Unlimited), TrueSkill
  rating, minimum-play floors ("50 games in Secret Mafia, 30 games for each Generalization environment").
- **MG-Ref offline protocol** (§5.4): "a deterministic offline tournament protocol that pairs a new agent
  against a frozen reference pool of top-ranked Stage II Efficient submissions" (198 games/agent).
- **A trajectory dataset schema** (Appendix B.5, Table 7; verified against HF `first-rows`):
  `player_game_id:int64, game_id:int64, env_name:string, model_name:string, player_id:int64,
  opponent_names:json, rewards:json, observations:json, num_turns:int64, status:string, reason:string`,
  where `observations` is a "Turn-indexed logs of this player's observation text and emitted action"
  (actual rows key turns by timestamp strings, e.g. `"2025-08-04 07:40:53.432037+00": {"observation": "[-1] You are Player 0 ...", ...}`).
- **A formal game model** (§3.4): tuple of Players, States ("internal, unobservable states maintained by
  the environment's simulator"), initial state, Actions ("all possible natural language token sequences
  an LLM can generate" with an admissible subspace per player/state), Observations ("The observation
  function ... computes the player-specific observation ... from the current internal state"), Turn
  Logic ("active-player function ... indicating which subset of players must submit an action"),
  Transitions, terminal Rewards. This is a **paper formalism**, not a machine-checked spec.
- Agent interface: HTTP/text. §7: "Any system that accepts text observations and returns text actions
  can participate through a unified HTTP API, with no game-specific integration required."

### What it assumes
- Explicitly mixed-motive by design: envs chosen along "(1) the information structure (full vs. partial
  observability, private vs. public communication) and (2) the incentive alignment (cooperative,
  competitive, or mixed-motive)"; 2–6 players.
- Text in/out only; terminal rewards only ("There are no intermediate rewards.").
- Simultaneous moves are modelled in the formalism (Blotto/IPD: "All players are simultaneously
  active") but implemented via TextArena's sequential private-submission pattern.

### Observation structure
- Described, per-role, in prose (§3.4: Codenames "Spymasters observe the full alignment mappings,
  whereas Operatives only observe the public board"; Mafia "share a private observation channel").
  Implementation is TextArena `add_observation(to_id=...)` routing in the env code (`SecretMafia/env.py`
  lines 213–286: `to_id=p` for mafia, `to_id=doc`, `to_id=det`, `to_id=-1` for day phase). In the live
  arena agents are remote processes behind HTTP, so hidden state is **operationally** unreachable there;
  offline (`src/offline_play.py`) the driver holds `env` as in TextArena. No declared/typed information
  set; same verdict as TextArena: **(i) not satisfied by construction** (satisfied only operationally
  in the hosted arena).

### First-class rules / norms
- **No.** Rules are prompt text; compliance is "invalid action handling" per env (Appendix C, Table 8:
  fatal vs non-fatal errors; "Some environments auto-correct invalid actions into default moves or allow
  skips, while others treat them as fatal"). The paper itself identifies this as a confound ("error-survival
  confound", §6). IPD promises/communication are free text with no commitment or contract object.
  **(ii) not satisfied.**

### Rational / exact baselines
- **None provided as artifacts.** "Organizer-provided baseline agents of varying strength were injected
  into the matchmaking pool" (App. I.1) — unnamed, never described as solvers; the dataset lists
  `model_name` values like `Humanity`, `qwen/qwen-2.5-7b-instruct`. The only equilibrium mention is a
  *participant's training-data generator*: Team RLGaming (App. A.1.3) "Rule-based agents: Nash
  equilibrium strategies for Colonel Blotto, defection strategies for IPD, providing coverage of
  game-theoretic baselines" — internal to one submission, not part of the framework. MG-Ref reference
  opponents are fine-tuned/prompted LLMs (In2AI Qwen3-8B, STARS qwen3:8b, RLGaming Llama-3.1-8B LoRA, etc.).
  Starter kit `src/agent.py` has only `Agent`, `LLMAgent` (HF transformers), `HumanAgent`.
  **(iii) not satisfied.**

### Trajectory / log format
- As above (Table 7 / HF parquet). Per-player rows, one `observations` JSON blob of text per turn, no
  structured state, no action typing, no information-set annotation. Platform-level logging (§3.1:
  "it records complete game trajectories, including acting player identity, role, public and private
  observations, emitted actions, game phase, and resulting state and reward updates") is server-side;
  only the Table 7 columns are released.

### Exporter from a typed spec -> MindGames
- Same work as the TextArena exporter plus: (a) an HTTP agent shim; (b) a converter from our log to the
  Table 7 row schema is **cheap** (one row per player, turn-keyed `{observation, action}` text, terminal
  `rewards` dict, `status/reason`); (c) adopting MG-Ref's TrueSkill + error-attribution reporting
  (Caused / Witnessed / Self-Forf. / Opp-Forf.) is a reporting adapter. Our typed observations would need
  a text rendering to populate `observations`. Nothing in MindGames consumes a game spec; a new game
  cannot be "submitted", only new agents.

---

## 3. GAMA / MAGiF (arXiv 2412.08805v3; Mensfelt, Stathis, Trencsenyi; ECAI 2025)

### What it standardizes
- **A Prolog game-description target** ("light GDL specifications that act as target representation for
  the autoformalization", §2.3) in Situation Calculus style, split into:
  - game-independent solver `magif/solver/solver.pl`:
    `game(F,F):- final(F).  game(S,F):- \+ final(S), legal(M,S), game(do(M,S),F).`
    `holds(F, S):- initially(F, S).  holds(F, do(M, S)):- effect(F, M, S).  holds(F, do(A, S)):- holds(F, S), \+ abnormal(F, A, S).`
    plus `initialise/2` (retract/assert of `initially/2`, used to inject `last_move`).
  - game-dependent predicates (§3.5.2): `initial/1, initially/2, possible/2, legal/2, effect/3,
    abnormal/3, final/1, finally/2, payoff/4` (example `DATA/MISC/general_agent.pl`).
  - strategy predicate `select(P, O, S, M)` (§3.5.3), e.g. `DATA/STRATEGIES/tit-for-tat.pl`.
- **Not a DSL of its own**: it is plain SWI-Prolog consulted through `swiplserver` (`magif/solver/engine.py`
  `PrologEngine(thread_creator=PrologMQI().create_thread)`). Paper: "The framework is implemented in
  Python and Prolog ... though its modular architecture supports substituting alternative solvers."
- A three-level validation pipeline: "(1) syntactic validation using formal solvers, (2) runtime
  validation through simulation-based testing, and (3) semantic validation based on ground truth outcomes
  (when available) or compliance with game-specific constraints."
- A JSON agent log and tournament log (see below).

### What it assumes
- **2x2 simultaneous-move normal-form games only.** README: "Currently, it supports 2x2 simultaneous-move
  games"; paper §6: "focusing on those modelled as 2x2 simultaneous-move games." Mixed-motive (PD,
  Hawk-Dove, Stag Hunt, Battle of the Sexes, Matching Pennies) — all five canonical games.
- Typed/symbolic state and actions (Prolog terms `move(P,M)`, situations `do(M,S)`), numeric payoffs
  `payoff(M1,M2,U1,U2)`.
- Exactly 2 players per match; `Environment._play_match` calls `agent1.mind.act(), agent2.mind.act()`
  then `agent1.mind.observe(move_agent_2); agent2.mind.observe(move_agent_1)` — simultaneous by
  construction. Rounds are independent one-shot games replayed `num_rounds` times; only `last_move` carries over.

### How LLM agents interact with the formal spec
- **The LLM never plays.** It is a translator: "autoformalization shifts the role of LLMs from
  decision-making to format translation, converting natural language interaction descriptions into
  formal representations." `Autoformalizer` (one-shot prompt with a PD example, up to `max_attempts`
  feedback loops carrying the erroneous Prolog line) produces the game-dependent predicates and/or a
  `select/4` strategy; thereafter `Mind.act()` calls `solver.select_move(agent_name)` which runs the
  Prolog query `select({agent_name}, _, s0, M).` (`magif/solver/game_logic.py:75`). Payoff is computed by
  the solver too: `finally(goal({player}, U), do(move({player}, '{move_p}'), do(move({opponent}, '{move_o}'), s0))).` (`game_logic.py:93-97`).
- **Yes, the formal spec is executed by a solver** (SWI-Prolog), both for move selection and payoff.

### Observation / information-set structure
- Minimal and implicit. Each agent has a `Memory` of `payoffs, moves, opponent_moves`
  (`magif/agent/memory.py`) and the solver state gets only `initialise(last_move(Opp,'M'), s0)`
  (`game_logic.py:112`). The environment hands each agent **only the opponent's move string**
  (`Mind.observe(opponent_move)`). Agents hold separate Prolog engines, so an agent cannot query the
  opponent's solver. But there is **no declared information set** and nothing to hide: 2x2 games are
  perfect-information after resolution. The paper's "Information" component (§2.1) is never
  operationalized; `grep -rniE "information set|hidden|imperfect"` over the repo returns nothing.
  Verdict on (i): **not addressed** (trivially satisfied by the game class, not enforced by a mechanism).

### First-class rules / norms with compliance checks?
- **Game rules are first-class executable objects** (the Prolog `legal/2`, `effect/3`, `final/1`,
  `finally/2` predicates) and the solver only generates legal evolutions (`game/2`). There are also
  **constraint validators** per game class, e.g. `DATA/EVAL/pd.pl`:
  `pd(T,R,P,S,C,D):- T>R, R>P, P>S, payoff(C,C,R,R), payoff(C,D,S,T), payoff(D,C,T,S), payoff(D,D,P,P).`
  used by `magif/utils/validator.py` for "constraint-based validation" of an autoformalized payoff matrix.
- But these are **rules of the game and validity constraints on the formalization**, not norms/contracts
  that agents can violate at runtime and be checked against. There is no notion of an agent-level
  obligation, promise, sanction, or compliance score; an agent whose `select/4` returns an illegal move
  simply gets `AgentStatus.RUNTIME_ERROR`. Verdict on (ii): **partial** — closest of the three (rules are
  declarative, solver-checked objects) but no runtime norm/contract compliance.

### Rational / exact baselines
- **Yes, within 2x2.** Strategies are Prolog programs run by the solver: `DATA/STRATEGIES/` contains
  `default_move, anti-default-move, tit-for-tat, anti-tit-for-tat, random, best_response`. `best_response.pl`
  does exact enumeration via the solver:
  `findall(Ui-Mi, (game(S, F), finally(outcome(P, Mi, Ui, O, LMo, Uo), F), Ui >= Uo), Options), sort(0, @>, Options, Ranked), highest(Ranked, M).`
  (best response to the opponent's last move). **No Nash-equilibrium solver, no MCTS, no DP, no CFR**:
  `grep -rniE "nash|equilibri|mcts|minimax"` over code returns nothing; the paper's only game-theoretic
  computation is the payoff-dominance constraint check. Verdict on (iii): **partial** — exact-inference
  (Prolog) non-LLM strategies exist, but no equilibrium computation and only for 2x2 one-shot repeated games.

### Trajectory / log format
- `Environment.log_tournament()` writes `tournament_info.json`:
  `{"num_agents", "num_rounds", "target_payoffs", "winners_payoffs": [(name, strategy_name, total)]}` and one
  `agent_<Name>.json` per agent via `Agent.save()` with keys
  `name, strategy_name, strategy_rules (Prolog text), status, game_rules (Prolog text), game_moves,
  game_players, default_move, moves, payoffs, total_payoff, trace_messages, attempts`.
  Example (`LOGS/experiment_4_gpt4/strategies_20250123_114700/agent_Zuvega.json`): `"moves": ["C","D","C","D"],
  "payoffs": [5.0, 3.0, 5.0, 3.0], "total_payoff": 16.0`. Per-agent move/payoff lists, **no joint
  per-round record, no observations, no timestamps per step**, and the opponent is not recorded in the
  agent file (only recoverable from `game_players`).

### Exporter from a typed spec -> GAMA
- **Cheap for 2x2 normal-form games; hard/impossible beyond.** A typed spec for a one-shot or repeated
  2-player, 2-action game maps mechanically to the eight game-dependent predicates plus `payoff/4`
  (template = `DATA/MISC/general_agent.pl`), and rational baselines then come for free (`best_response.pl`).
  Beyond that, GAMA's `Environment`, `Mind`, and `game_logic.py` hard-code two players, one opponent
  (`game_players[0]/[1]`), `last_move` as the only carried state, and `select(P,_,s0,M)` from `s0` every
  round. Sequential, N-player, hidden-information or multi-round-state games would require rewriting
  the Python runtime, not just the Prolog. The paper agrees: "Future work will extend the framework
  beyond 2x2 simultaneous-move games."

---

## Comparison table

| | TextArena | MindGames | GAMA/MAGiF |
|---|---|---|---|
| Game format | Python `Env` subclass, text | same (TextArena) + paper tuple formalism | Prolog predicates (GDL-like, Situation Calculus) |
| Observations | per-pid text inbox, routed by env author (`to_id`) | same; HTTP isolation in live arena | opponent's last move only |
| (i) enforced info sets | No (convention) | No (operational only) | N/A (perfect info 2x2) |
| (ii) first-class rules w/ compliance | No (prompt + regex + `set_invalid_move`) | No (Appendix C error rules) | Partial: executable game rules + payoff constraint validators; no agent-level norms |
| (iii) rational baselines | None (all LLM) | None in framework; one team used Nash-Blotto for training data | Partial: Prolog strategies incl. exact best-response; no equilibrium solver |
| Log | in-memory `State.logs`; `close()` -> rewards, game_info | HF parquet, Table 7 columns, text per turn | `agent_<Name>.json` + `tournament_info.json` |
| Players / timing | 1–N, sequential (sim. emulated) | 2–6, sequential (sim. emulated) | 2, simultaneous |

## Verdict (summary, ~300 words)

**TextArena** standardizes a Gym-like string-in/string-out loop and a TrueSkill ladder, not a game
description. Observation privacy is an env-author convention (`State.add_observation(to_id=...)`);
hidden state is plain Python reachable by the driver loop, and no information set is declared. Rules are
prompt prose plus regex legality checks (`set_invalid_move`); there are no norm/contract objects. Every
shipped agent is an LLM or human; the only "solver-like" module is an LLM jury. Logging is an in-memory
`(from_id, text)` list. **Verdict: fails (i), (ii) and (iii).**

**MindGames** is a benchmark protocol (4 envs, tracks, TrueSkill, MG-Ref offline tournament, HF dataset
schema) built entirely on TextArena. Its §3.4 tuple formalism names an observation function and active-
player function, but only on paper; the implementation inherits TextArena's conventions. No norm
objects; invalid-action handling is per-env and the authors flag it as an evaluation confound. No
organizer rational baselines are released; the single Nash-equilibrium mention is one team's private
training-data generator. **Verdict: fails (i), (ii) and (iii); its genuine contribution to our project is
the trajectory row schema (Table 7) and the MG-Ref reporting protocol, both cheap to export to.**

**GAMA/MAGiF** is the closest prior work and the one that most threatens the claim, but only in part. It
has executable, solver-checked game rules in Prolog (GDL-flavoured Situation Calculus), constraint
validators for payoff structure, and non-LLM strategies executed by SWI-Prolog, including an exact
best-response computed by enumerating `game/2` evolutions. However: the LLM is only a translator, never
a player; the system is hard-wired to two-player 2x2 simultaneous-move games with `last_move` as the
only carried state; there is no information-set concept at all; "rules" are game mechanics and
formalization validators, not agent-level norms/contracts with compliance scoring; and there is no
equilibrium, MCTS or DP solver. **Verdict: partial (ii) and partial (iii), no (i); does not offer the
conjunction.**

**Overall: the lead's claim survives for all three.** The accurate framing is: TextArena/MindGames give
the ecosystem (API, ladder, log schema) with none of (i)–(iii); GAMA gives declarative executable rules
and exact Prolog strategies for 2x2 games but no observation structure, no norms, no equilibria, and
no LLM players at runtime. The related-work text should cite GAMA as "formal rules + solver-driven
baselines, LLM as autoformalizer only, 2x2 scope" rather than as a competitor on (i)+(ii)+(iii).

## What I could not access / caveats
- GAMA `DATA/PROMPTS/`, `DATA/GAMES/` referenced in the README config were not enumerated (file listing
  was truncated at 80 of 1,684 files; most are `LOGS/`). This does not affect any finding above.
- MindGames organizer "baseline agents of varying strength" are never identified in the paper or
  starter kit; I could not confirm whether any was rule-based.
- TextArena's hosted server code (matchmaking, server-side trajectory logger described in MindGames
  §3.1) is not in the public repo; only the client in `textarena/api.py` is.
- arXiv HTML renders math as `[math]`; numeric table values for GAMA Tables 5/7/8/9 and MindGames
  formula symbols were therefore read from prose, not tables.

## Sources
- TextArena paper: https://arxiv.org/html/2504.11442 (read in full; local copy `.../dl/amend_a/2504.11442.txt`)
- TextArena code: https://github.com/LeonGuertler/TextArena @ 1e3160d (files: `textarena/core.py`, `textarena/state.py`,
  `textarena/agents/basic_agents.py`, `textarena/wrappers/ObservationWrappers/llm_observation_wrapper.py`,
  `textarena/envs/KuhnPoker/env.py`, `textarena/envs/IteratedPrisonersDilemma/env.py`, `textarena/envs/SecretMafia/env.py`,
  `textarena/envs/ColonelBlotto/env.py`, `textarena/envs/utils/jury.py`, `textarena/envs/registration.py`, `textarena/api.py`, `README.md`)
- MindGames paper: https://arxiv.org/html/2605.29512 (read §1–7, App. B.5, C, I; A.1.3 excerpt)
- MindGames starter kit: https://github.com/mind-games-challenge/mindgames-starter-kit @ b7ff02a (`README.md`, `src/agent.py`,
  `src/offline_play.py`, `src/offline_evaluation.py`, `.gitmodules`, `envs/`)
- MindGames dataset: https://huggingface.co/datasets/mindgameschallenge/MGC2025 (README card; `first-rows` API for `threeplayeripd`)
- GAMA paper: https://arxiv.org/html/2412.08805 (v3, 29 May 2025; read in full)
- GAMA code: https://github.com/dicelab-rhul/GAMA @ c26eafd (`README.md`, `magif/solver/solver.pl`, `magif/solver/solver.py`,
  `magif/solver/game_logic.py`, `magif/agent/{agent,mind,memory}.py`, `magif/environment/environment.py`,
  `magif/utils/validator.py`, `DATA/STRATEGIES/*.pl`, `DATA/EVAL/pd.pl`, `DATA/MISC/general_agent.pl`,
  `LOGS/experiment_4_gpt4/strategies_20250123_114700/{agent_Zuvega.json,tournament_info.json}`)
