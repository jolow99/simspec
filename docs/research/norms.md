# Oldenburg & Zhi-Xuan (AAMAS 2024): Learning and Sustaining Shared Normative Systems via Bayesian Rule Induction in Markov Games

Research notes for the "typed spec language + thin runtime for LLM multi-agent game studies with first-class Rules" project.
Compiled 2026-10-16 from the full arXiv PDF (v2, 22 Feb 2024, 18 pages incl. 7-page appendix) and a full clone of the released code.

Local copies:
- PDF: `(local scratch, not committed)`
- pdftotext output: `.../scratchpad/dl/norms/paper_nolayout.txt` (and `paper.txt` with `-layout`)
- Code clone (commit `2c40f279`, 2024-03-12 "Update README.md", `main`): `.../scratchpad/dl/norms/repo/`

What was and was not obtained:
- Obtained: full paper text (body + appendix incl. Table A1 listing all 68 potential norms, Table A2 emergent norms), abstract-page metadata, the complete code repo (README, rule classes, rule generator, planners, learners, substrate configs, Lua backend, eval scripts, result CSVs for the intergen experiment).
- Not obtained: numeric values behind Figures 2-5 and A2-A4 (they are plots; only the axis labels/legends survive text extraction). The arXiv HTML rendering exists (`https://arxiv.org/html/2402.13399v2`) but WebFetch returned only a summary; the PDF was used instead. `examples/results_learning/` and `examples/results_emerge/` directories in the repo are empty (only `.DS_Store`); only `results_intergen/max_life_span{10,30,50,70,90,100}/bot{1..6}/trial1.csv` contain data. No fork by `ztangent` was found; GitHub shows 14 stars, 1 unnamed fork, 394 commits, MIT license.

---

## 1. Metadata

- Title: Learning and Sustaining Shared Normative Systems via Bayesian Rule Induction in Markov Games
- Authors: Ninell Oldenburg (University of Copenhagen, niol@hum.ku.dk), Tan Zhi-Xuan (MIT, xuan@mit.edu)
- arXiv 2402.13399; v1 20 Feb 2024, v2 22 Feb 2024 (current). cs.AI. ACM classes I.2.0; I.6.5; G.3. CC BY 4.0.
- Venue: Proc. of the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2024), Auckland, May 6-10 2024, IFAAMAS, 18 pages.
- Keywords: Norm Learning, Cooperative Intelligence, Norm Emergence, Bayesian Learning, Normative Systems, Markov Games.
- Code: https://github.com/ninell-oldenburg/social-contracts (MIT License, (c) 2023 Ninell Oldenburg). The paper's "Supplementary Material" section says: "Source code is available at https://github.com/ninell-oldenburg/socialcontracts" (hyphen dropped in the PDF; the real URL has the hyphen).
- Funding: PIBBSS Summer Research Fellowship; Tan Zhi-Xuan supported by the Open Philanthropy AI Fellowship.

### Abstract (verbatim)

> A universal feature of human societies is the adoption of systems of rules and norms in the service of cooperative ends. How can we build learning agents that do the same, so that they may flexibly cooperate with the human institutions they are embedded in? We hypothesize that agents can achieve this by assuming there exists a shared set of norms that most others comply with while pursuing their individual desires, even if they do not know the exact content of those norms. By assuming shared norms, a newly introduced agent can infer the norms of an existing population from observations of compliance and violation. Furthermore, groups of agents can converge to a shared set of norms, even if they initially diverge in their beliefs about what the norms are. This in turn enables the stability of the normative system: since agents can bootstrap common knowledge of the norms, this leads the norms to be widely adhered to, enabling new entrants to rapidly learn those norms. We formalize this framework in the context of Markov games and demonstrate its operation in a multi-agent environment via approximately Bayesian rule induction of obligative and prohibitive norms. Using our approach, agents are able to rapidly learn and sustain a variety of cooperative institutions, including resource management norms and compensation for pro-social labor, promoting collective welfare while still allowing agents to act in their own interests.

### Core thesis (Introduction, verbatim excerpts)

> "each agent assumes that all other agents may be complying with a shared set of norms even while pursuing their own interests, allowing them to infer the existence of a norm from apparent violations of self-interest, and to aggregate such observations across a population of agents, updating their beliefs about which norms best explain collective behavior."

> "Under sufficiently high levels of compliance, shared normativity is thus self-sustaining: Once enough agents infer a shared set of norms, they conform to them, thereby creating public knowledge that the norms are practiced, and enabling new entrants to rapidly learn the norms."

> "our approach foregrounds the epistemic and cognitive aspects of social norms, framing norm learning as an (approximately) rational Bayesian process of learning structured rules [...], rather than a reactive process of adapting one's habitual actions"

Figure 1 caption (verbatim): "Overview of our framework, summarized by (a) the (simplified) graphical model for a norm-augmented Markov game. Agents learn to comply with rule-based social norms (b), which are factored into prohibitions (e.g. P1, P2) and obligations (e.g. O1, O2, O3). Agents comply with norms while pursuing their interests via model-based planning (c), which switches between a reward-oriented mode that obeys prohibitions while collecting rewards (left), and an obligation-oriented mode that plans to satisfy a postcondition (right). Agents learn these norms via (d) approximate Bayesian inference, updating the probabilities of each potential norm based on observations of others' actions."

---

## 2. Formal framework: Norm-Augmented Markov Games (NMGs) (Sec. 2.1)

### Markov game base (verbatim)

> "In Markov games [...], a set of M agents take actions a_i ∈ A_i in a series of environment states s ∈ S over time, where i ∈ I := {1, ..., M} is each agent's index. Actions change the environment state via a transition function T : S × A_1 × ... × A_M → Δ(S) that maps the current state s and joint actions (a_1, ..., a_M) to a distribution over successor states s'. Each agent also has a reward function R_i : S × A_i × S → R that maps state transitions (s, a, s') to scalar rewards r ∈ R. We also assume a shared discount factor γ. In our context, an agent's reward function can be understood as a representation of their individual desires. Markov games begin with an initial state s_1, and end after a horizon of H steps."

### Social norms as history classifiers (verbatim)

> "We extend Markov games with the addition of social norms, formalized as functions ν : H → {0, 1} where H_i := ⋃_{t=1}^{H} ((S × A_i)^t × S) is the set of state-action histories for agent i, and H := ⋃_{i∈I} H_i is the set of histories across all agents. In words, a social norm ν classifies whether the behavioral history of an agent i complies with the norm [36], with 0 indicating compliance and 1 indicating violation. We call a set of norms N a normative system."

### Prior over normative systems, Eq. (1) (verbatim, reconstructed from PDF)

> "Let 𝒩 be a set of functions defining the space of possible social norms, and ε ∈ [0, 1] be the frequency of a norm violation. We now define norm-augmented Markov games as a (subjective) Bayesian extension of a Markov game, endowing each agent i with a subjective prior P_i^0(N) over which sets of norms N ⊆ 𝒩 will be complied with by all agents with frequency at least 1 − ε:"

    P_i^0(ν) = P_i( (1/(H M)) Σ_{t,j=1}^{H,M} (1 − ν(h_t^j)) ≥ 1 − ε ),     P_i^0(N) = Π_{ν∈N} P_i^0(ν)        (1)

> "where h_t^j is the state-action history for agent j at step t, treated as a random variable. Each agent also has conditional prior P_i(h_t | N) over agent histories h_t given norms N, and this induces a sequence of posteriors over norms P_i^t(N) := P_i(N | h_t) ∝ P_i^0(N) P_i(h_t | N), defining agent i's belief in norms N at each t after observing h_t."

Note: the prior over a *set* N factorizes as a product of per-norm priors (independence assumption, mirrored later by the mean-field posterior).

### Norm violation cost and norm-augmented reward, Eq. (2) (verbatim)

> "We also endow each agent i with a norm violation cost function C_i : 𝒩 → R, which defines how intrinsically motivated an agent is to comply with norms that they believe to be true. This induces a norm-augmented reward function R'_i : H → R. Given a history h = (h', s, a_i, s'), R'_i decomposes into a base reward function R_i and the expected cost of norm violation under the posterior P_i(N | h_t):"

    R'_i(h', s, a_i, s') := R_i(s, a_i, s') − Σ_{N⊆𝒩} P_i(N | h) Σ_{ν∈N} C_i(ν) ν(h)        (2)

> "By adding priors over norms to a Markov game, each agent i effectively factors their beliefs about other agent's policies π_{−i} into two parts: Beliefs about whether the policies will comply with a shared set N of relatively simple yes/no rules, and beliefs about non-normative features of those policies. [...] An agent's prior over norms thus serves as a correlating device over their beliefs about how other agents behave [5, 30]."

### Stability / normative equilibria (verbatim)

> "When is a normative system N stable? In general, a set of beliefs about norms {P_i^t(N)}_{i=1}^M at time t is consistent with a wide range of correlated strategies π_{1:M}, some of which may form (subjective) correlated equilibria: joint policies which no agent has an incentive to deviate from [5, 59]. Out of these equilibria, consider those where the expected frequency of each agent i complying with norms N ∼ P_i^t(N) over all future steps is at least 1 − ε, where the expectation is taken over their own belief distribution P_i^t(N) over the norms. Since these are policies where agents frequently comply with the norms they (subjectively) believe might be true, we call them subjective normative equilibria. If all agents have converged to a single distribution P^t(N) over normative systems (or δ-close to it), this forms an (objective) normative equilibrium (respectively, a normative δ-equilibrium). If P^t(N) concentrates on a specific N, then N is stable with respect to that normative equilibrium. Note that if P^t(N) is concentrated enough on some N, and agents are (norm-augmented) reward-maximizers, we can guarantee that N is stable by setting violation costs for ν ∈ N to be sufficiently high."

---

## 3. The environment

### Paper description (Sec. 2.1, verbatim)

> "As an example setting for norm-augmented Markov games, we adapt and combine several existing games from the Melting Pot suite of environments [1, 50], including elements of Commons Harvest [43, 72], Clean Up [42], and Territory [50]. [...] In the environment, agents can collect and eat apples to gain reward but otherwise incur a cost for actions. Apples regrow at a density-dependent rate (more surrounding apples lead to faster regrowth), but the rate decreases as the nearby river gets polluted over time. To prevent pollution, agents have to use clean-up action on the river. In addition, several regions are visually marked as each agent's territory (but do not otherwise have functional effects), and agents have one of three visually distinct appearances that demarcate roles. By combining these elements, we allowed for a range of social norms, including resource management norms, property norms, role-based division of labor, and compensation for pro-social labor. In our experiments, we gave experienced agents high prior beliefs in some norm set N, and learning agents a low prior belief in all (non-empty) norm sets."

Appendix Figure A1 caption (verbatim): "Example frame of our environment. The three experienced agents are depicted in grey each representing a specific role based on their appearance. The learning agent is completely colored, mimicking the "egalitarian" role based on its appearance. Each agent's territory is visible as an agent-colored overlay. The river is located on the top of the frame with brighter parts representing the dirt. Apples are located in orchards that might, in turn, be in part on an agent's territory. If a field is desiccated and hence unlikely but not impossible to regrow apples, it is signified by a sand-like color underneath."

Appendix A1 "Parameters for the Norm-augmented Markov Game" (verbatim): "The individual reward for eating an apple for every agent is 1 while the action cost is 0.01. Each prohibition norm violation results in a violation cost of 1 that is applied to the value function as described in Equation 4 while, as mentioned above, the reward for the fulfillment of an obligation is 1, as specified in Equation 5."

### As implemented (substrate `rule_obeying_harvest`)

Substrate configs: `meltingpot/python/configs/substrates/rule_obeying_harvest__{complete,empty,apples,cleaning,territory,apples_cleaning,apples_territory,cleaning_territory}.py` (one variant per combination of active norm "components"; `view_learning_model.get_name_from_rules` picks the variant from which default rules are active). Lua backend: `meltingpot/lua/levels/rule_obeying_harvest/{init.lua,components.lua}`.

Gridworld map (`ASCII_MAP` in `rule_obeying_harvest__complete.py`, 23 x 22 chars incl. walls; the bottom four rows are an inventory/avatar-copy HUD, not walkable):

```
WWWWWWWWWWWWWWWWWWWWWWW
WFHFHFHFHFHFHFHHFHFFHFW
WFHFHFHFHFHFHFHHFHFFHFW
W========+~FHFHHFHf===W
W   P     ===+~Sf     W
W      P     <~Sf  P  W
W          P <~S>     W
WT^A^T^T^T^T^;~S^T^A^TW
W_AAA_____________AAA_W
WAAAAA_____A_____AAAAAW
W_AAA_____AAA_____AAA_W
W__A_____AAAAA_____A__W
W_____A___AAA___A_____W
W_Q__AAA___A___AAA__G_W
W___AAAAA_____AAAAA___W
W____AAA____Q__AAA____W
W__G__A___Q_____A__G__W
WWWWWWWWWWWWWWWWWWWWWWW
WD-----WD-----WD------W
WWWWWWWWWWWWWWWWWWWWWWW
WD-----WD-----WD------W
WWWWWWWWWWWWWWWWWWWWWWW
```

Legend (`CHAR_PREFAB_MAP`): `W` wall; `S` river; `H` river + potential_dirt; `F` river + actual_dirt; `~` river (+shadow); `A` grass + resource + apple; `_` grass + resource; `P`/`G` spawn points (sand / grass); `Q` inside_spawn_point; `T ^ ; , = + f < >` sand/grass-edge/resource decorations; `D` avatar_copy and `-` inventory_display (HUD rows). Every walkable cell carries a `resource` prefab, which is what territory claiming paints.

Agents / roles: `config.valid_roles = frozenset({"free", "cleaner", "farmer", "learner"})`; default `("cleaner", "farmer", "free", "learner")`. Paper names: C = cleaner, F = farmer, E = "egalitarian" (= `free` in code); the `learner` role uses the `free` sprite (`ROLE_SPRITE_DICT = {'free': CUTE_AVATAR, 'cleaner': CUTE_AVATAR_W_SHORTS, 'farmer': CUTE_AVATAR_W_FARMER_HAT, 'learner': CUTE_AVATAR}`), and `INT_TO_ROLE = {0:'free', 1:'cleaner', 2:'farmer', 3:'free'}` in `rule_adjusting_policy.py`. Experienced agents are drawn in grey palettes; learners in colour.

Action set (12 discrete actions, `ACTION_SET` order = integer index used everywhere in the policies):

| idx | name | Lua action vector |
|---|---|---|
| 0 | NOOP | all zeros |
| 1 | FORWARD | move=1 |
| 2 | BACKWARD | move=3 |
| 3 | STEP_LEFT | move=4 |
| 4 | STEP_RIGHT | move=2 |
| 5 | TURN_LEFT | turn=-1 |
| 6 | TURN_RIGHT | turn=+1 |
| 7 | FIRE_ZAP | fireZap=1 (sanction beam, length 3, radius 1, cooldown 2; `penaltyForBeingZapped = 15`, `rewardForZapping = 0`, `removeHitPlayer = False`, `framesTillRespawn = 4`) |
| 8 | FIRE_CLEAN | fireClean=1 (cleaning beam, length 3, radius 1, cooldown 2) |
| 9 | FIRE_CLAIM | fireClaim=1 (territory claim beam, length 1, radius 0) |
| 10 | EAT | eat=1 (consume 1 apple from inventory, reward 1.0 via `Eating.rewardForEating`) |
| 11 | PAY | pay=1 (pay beam length 3 radius 2, transfers `amount=1.0` apple from payer inventory to a payee hit by the beam) |

The policy code maps these into action classes with `get_action_name`: 0 -> `NOOP_ACTION`, 1-4 -> `MOVE_ACTION`, 5-6 -> `TURN_ACTION`, 7..11 -> `ZAP_ACTION, CLEAN_ACTION, CLAIM_ACTION, EAT_ACTION, PAY_ACTION`. Rules are stated over these class names.

Key dynamics (Lua `components.lua` + policy constants in `rule_adjusting_policy.py`):
- Apples: walking onto an apple cell harvests it into `INVENTORY` (`Harvestable`); eating gives reward 1. Regrowth (`DensityRegrow`): per-apple regrowth probability chosen by number of live neighbouring apples within `APPLE_RESPAWN_RADIUS = 1.0` from `REGROWTH_PROBABILITIES = [0.02, 0.05, 0.07, 0.09]`, then scaled by river pollution: `interpolation = (dirtFraction − thresholdDepletion) / (thresholdRestoration − thresholdDepletion)` clipped to <= 1, `probability = maxAppleGrowthRate * interpolation`, with `THRESHOLD_APPLE_DEPLETION = 0.7`, `THRESHOLD_APPLE_RESTAURATION = 0.1`, `MAX_APPLE_GROWTH_RATE = 1`. A field whose neighbourhood is emptied becomes `dessicated` (sand-coloured; "unlikely but not impossible to regrow"). `DEAD_APPLE_RATIO` = dessicated fields / total apple fields (`Surroundings:setRatioDessicatedFields`).
- River/pollution: `H` cells are `dirtWait` (potential dirt), `F` cells start as `dirt`; a `DirtSpawner` turns potential dirt into dirt with `DIRT_SPAWN_PROB = 0.2`; a `RiverMonitor` counts dirt vs clean cells; `FIRE_CLEAN` hitting a `dirt` cell sets it to `dirtWait` (`DirtCleaning:onHit`) and resets the agent's `sinceLastCleaned`. On the Python side `DIRT_FRACTION = #(-3 cells) / (#(-3) + #(-4))` over the `SURROUNDINGS` tensor.
- Territory: each avatar has a `Property` component with `radius = 2`; at start (and on rebirth) it claims the 5x5 rectangle of `resource` cells around its spawn (`Property:markRectangle`). `PROPERTY` is a map-sized int tensor of owner indices (0 = unowned). `CUR_CELL_IS_FOREIGN_PROPERTY` is true iff `PROPERTY[x][y]` is non-zero and not the agent's own Lua index. A global `stolenRecords[stolen_from][thief]` tensor (never reset) records harvesting on foreign property; exposed as `STOLEN_RECORDS`. Territory "does not otherwise have functional effects" (paper).
- Paying: at `postStart`, each `farmer`'s `Paying` component assigns itself to pay up to `maxPayeesPerPayer` `cleaner`s (`payingTo` one-hot over players; `ALWAYS_PAYING_TO` / `ALWAYS_PAID_BY` observations). `sinceLastPayed` increments every step for agents with a payee and resets to 0 when a `payHit` succeeds (`Paying:getPayed`, requires `hasEnough()` inventory). `TIME_TO_GET_PAID` is set on payees while a payer is attempting to pay, which makes the cleaner "wait" (`env_step`: NOOP by a cleaner sets `TIME_TO_GET_PAID = 0`).
- Age / mortality (intergen experiment): `Age` component with `max_life_span` (`DEFAULT_MAX_LIFE_SPAN = 300` in code); at max age the avatar goes to `playerWait`, inventory resets, a 5-frame freeze runs, then it respawns at age 0 and re-marks its territory. The Python harness mirrors this (`handle_age`, `freeze_counter`) and resets `rule_beliefs` to the init prior when `freeze_counter == 1` (rebirth as a "child" with no norm beliefs).
- Observation: `OBSERVATION_RADIUS = 15` (full-map observability for role `learner`, see `Surroundings:update`: "full observability for learning agents"). `SURROUNDINGS` is a map-sized int tensor: `>0` = player Lua index at that cell, `-1` apple, `-2` water, `-3` dirt, `-4` potential (clean) dirt, `0` otherwise. Individual observation names: `READY_TO_SHOOT, AGENT_CLEANED, AGENT_LOOK, PROPERTY, DEAD_APPLE_RATIO, INVENTORY, SURROUNDINGS, TOTAL_NUM_CLEANERS, ALWAYS_PAYING_TO, ALWAYS_PAID_BY, TIME_TO_GET_PAID, STOLEN_RECORDS, POSITION, ORIENTATION`, plus `SINCE_AGENT_LAST_CLEANED`, `SINCE_AGENT_LAST_PAID` metrics; global `WORLD.RGB` (168x240). The Python policy then derives the *rule-relevant* features in `add_non_physical_info` / `update_observation`: `CUR_CELL_HAS_APPLE` (SURROUNDINGS at own cell == -1), `NUM_APPLES_AROUND` (count of -1 in the 3x3 neighbourhood excluding centre), `CUR_CELL_IS_FOREIGN_PROPERTY`, `AGENT_HAS_STOLEN`, `DIRT_FRACTION`, `AGENT_LOOK` (sprite string -> role int), `POSITION_OTHERS`, `RIOTS` (list of agents the observer has seen violating an *active* rule, cleared when zapped), `SINCE_AGENT_LAST_{CLEANED,PAID,ZAPPED}`, `GOAL_COUNT`.
- Episode: `maxEpisodeLengthFrames = 500` in the substrate, but experiments run `episodes = 300` steps per run (H = 300) from the harness; intergen runs use `DEFAULT_MAX_LIFE_SPAN * 9` steps in `eval_intergen.py` (paper: 1.5·M·L steps "three generational cycles" with M = 6).
- Rewards/costs (policy side): `DEFAULT_APPLE_REWARD = 1`, `DEFAULT_ACTION_COST = 0.01`, `DEFAULT_BLOCKING_COST = 0.1` (moving into another agent; x10 for water), `DEFAULT_VIOLATION_COST = 1` (intrinsic), `DEFAULT_OBLIGATION_REWARD = 1`, `PENALTY_FOR_BEING_ZAPPED = 15`.

---

## 4. Rule grammar: representing social norms (Sec. 2.2)

### Paper (verbatim)

> "Drawing upon deontic and temporal logic representations of norms [2, 21, 46], we consider learning both (maintenance-based) prohibitions 𝒩_P, which forbid the effects of certain actions at all times, and (achievement-based) obligations 𝒩_O, which require pursuing temporally extended goals, giving our full set of norms 𝒩 = 𝒩_P ∪ 𝒩_O. [...] Note that the distinction between prohibitions and obligations is not fundamental: Prohibitions can be represented as obligations via negating the effects and vice versa. Nonetheless, we found it more natural to represent maintenance norms in their prohibitive form (e.g. "Don't empty the apple orchard.") and achievement norms in their obligative form (e.g. "Pay the cleaner in the next 30 steps.")."

> "Following prior work on norm and rule learning [33, 101], we express norms as rules in a first-order language ℒ, which includes predicates and functions defined over objects in a state s. Given a first-order expression φ, we denote its valuation in s as s[φ]."

Prohibition (verbatim):
> "we represent a prohibition norm ν_P ∈ 𝒩_P as a tuple ⟨Prohib, Post⟩ where Prohib is a function of objects in s that returns a set of potentially prohibited actions, and the postcondition Post holds in a state s whenever the prohibited effect occurs. In other words, if an action a_i ∈ s[Prohib] leads to state s' such that s'[Post] is true, this means that the prohibition has been violated, and hence that ν_P(h, s, a, s') = 1. As an example, an agent m in our environment is prohibited from moving to a cell c (Prohib = Move(m, c)) if that cell contains apples and is in foreign territory (Post = Apl(c) ∧ Foreign(c)) as depicted in Figure 1b."

Obligation (verbatim):
> "we define obligations ν_O ∈ 𝒩_O as tuples ⟨Pre, Post, τ⟩, where Pre is a precondition formula that holds true when the obligation is triggered, and Post is a postcondition formula that holds true once the obligation has been discharged, with τ as the duration during which the obligation must be satisfied. Given a history h = (s_{t−k}, a_{−i,t−k}, ..., s_t, a_{n,t}) up to time step t, this means that ν_O(h) = 1 (the obligation is violated) if there exists some k ≤ τ such that s_{t−k}[Pre] is true but s_{t−k+j}[Post] is false for all k < j ≤ τ. As another example, when agents are in the cleaner role (Look(m) = C) and find the river more than 30% polluted (Dirt(r) > 0.3) they are obliged to have cleaned the river (Cleaned(m, r)) within the next 20 steps (τ = 20). Therefore, (Pre = (Dirt(r) > 0.3 ∧ Look(m) = C), Post = Cleaned(m, r), τ = 20)."

The five norms practiced by experienced agents, `N_active = {P1, P2, O1, O2, O3}` (Figure 1b / Figure 2 legend): P1 "Don't empty orchard" (Fig. 4a: "Don't pick up apples, if there are too few apples around"), P2 "Don't steal", O1 "Farmer should pay", O2 "Cleaner should clean" ("If you're a cleaner, clean the river when there is >30% dirt"), O3 "Egalitarian should clean".

### As implemented: `ast_rules.py` (verbatim class skeletons)

Rules are *Python boolean expressions over the observation dict `obs`*, compiled into lambdas via `ast` (no separate DSL):

```python
class EnvironmentRule():
    def __init__(self, precondition):
        self.precondition = "lambda obs : " + precondition
        self.precondition_formula = self.walk_lambda(ast.parse(self.precondition))
    def holds_precondition(self, obs):
        return self.precondition_formula(obs)
    def is_subset_of(self, other_rule):  # same feature names, same operator, stricter threshold

class ProhibitionRule(EnvironmentRule):
    """Contains rules that prohibit an action."""
    def __init__(self, precondition, prohibited_action): ...
    def holds(self, obs, action):
        """Returns True if a rule holds given a certain observation."""
        if not action == self.prohibited_action:
            return False
        return super().holds_precondition(obs)
    def make_str_repr(self):
        return self.pure_precon + ' -> !' + self.prohibited_action

class ObligationRule(EnvironmentRule):
    """Contains rules that emit a subgoal."""
    def __init__(self, precondition, goal): ...
    def holds_in_history(self, observations):
        """Returns True if a precondition holds given a certain vector of observation."""
        for i, obs in enumerate(observations):
            if super().holds_precondition(obs):
                for j in range(i, len(observations)):
                    if self.satisfied(observations[j]):
                        return False
                return True
        return False
    def satisfied(self, observation: dict) -> bool:
        return self.goal_formula(observation)
    def make_str_repr(self):
        return self.pure_precon + ' -> ' + self.pure_goal
```

Semantics as implemented:
- A prohibition `⟨precondition, prohibited_action⟩` is checked on the *predicted next observation*: the planner simulates each action, computes the resulting `obs` (e.g. new `POSITION`, `CUR_CELL_HAS_APPLE`, `NUM_APPLES_AROUND`, `CUR_CELL_IS_FOREIGN_PROPERTY` at the destination cell) and calls `rule.holds(new_obs, action_name)` (`available_actions` / `check_all`). So "precondition" == the paper's `Post` evaluated on s', and `prohibited_action` == the paper's `Prohib` (an action class, in all experiments `MOVE_ACTION`).
- An obligation `⟨precondition, goal⟩` is *triggered* when `precondition(obs)` held at some step in the agent's rolling history (`deque(maxlen=10)`) and `goal` has not held since. τ is not stored on the rule; it is a global `max_obligation_depth = DEFAULT_OBLIGATION_DEPTH = 20` (steps since trigger, tracked via `GOAL_COUNT` / `nonself_active_obligations_count`).
- Both are grounded purely in observation-dict keys (`obs["DIRT_FRACTION"]`, `obs["AGENT_LOOK"]`, ...) - i.e. in state features the environment exports or the policy derives (Sec. 3 above). There is no object-level quantification; "agent m", "cell c", "river r" are implicit (ego agent, ego cell, the single river).

### The default (practised) norms, `lambda_rules.py` (verbatim)

```python
cleaning_precon_free_30 = 'obs["DIRT_FRACTION"] > 0.45 and obs["AGENT_LOOK"] == 0'
cleaning_goal_free_30 = 'obs["SINCE_AGENT_LAST_CLEANED"] == 0'
payment_precon_farmer_15 = 'obs["SINCE_AGENT_LAST_PAID"] > 30 and obs["AGENT_LOOK"] == 2'
payment_goal_farmer_15 = 'obs["SINCE_AGENT_LAST_PAID"] == 0'
cleaning_precon_cleaner_5 = 'obs["DIRT_FRACTION"] > 0.3 and obs["AGENT_LOOK"] == 1'
cleaning_goal_cleaner_5 = 'obs["SINCE_AGENT_LAST_CLEANED"] == 0'
zap_precondition = 'len(obs["RIOTS"]) > 0'
zap_goal = 'len(obs["RIOTS"]) == 0'

DEFAULT_OBLIGATIONS = [
  ObligationRule(cleaning_precon_free_30, cleaning_goal_free_30),      # O3 egalitarian cleans
  ObligationRule(payment_precon_farmer_15, payment_goal_farmer_15),    # O1 farmer pays
  ObligationRule(cleaning_precon_cleaner_5, cleaning_goal_cleaner_5),  # O2 cleaner cleans
  ObligationRule(zap_precondition, zap_goal)                           # sanction meta-norm (removed from DEFAULT_RULES in eval_learning.py)
]
harvest_apple_precon_standard = 'obs["CUR_CELL_HAS_APPLE"] and obs["NUM_APPLES_AROUND"] < 3'
steal_from_forgein_cell_precon = 'obs["CUR_CELL_HAS_APPLE"] and obs["CUR_CELL_IS_FOREIGN_PROPERTY"]'
DEFAULT_PROHIBITIONS = [
  ProhibitionRule(harvest_apple_precon_standard, 'MOVE_ACTION'),   # P1
  ProhibitionRule(steal_from_forgein_cell_precon, 'MOVE_ACTION'),  # P2
]
```

`eval_learning.py`'s `RULE_DICT` names them: "Too few apples prohibition", "Don't steal prohibition", "Egalitarian clean obligation", "Farmer pay obligation", "Cleaner clean obligation". Note the thresholds actually used: O3 fires at dirt > 0.45 (egalitarian), O2 at dirt > 0.3 (cleaner), O1 at 30 steps unpaid (farmer); P1 is `NUM_APPLES_AROUND < 3`.

### The hypothesis space 𝒩: `rule_generation.py`

Paper (Sec. 3.1): "The full space 𝒩 of 68 possible norms was generated by enumerating over a variety of predicates and numeric thresholds (see Appendix)." Appendix Table A1 caption (verbatim): "Generated list of feasible and non-trivial potential norms 𝒩 up to a length of three parameters. (e.g. norms 24-31). The parameters are variables of the agents' observation space. The norms are feasible in the sense that the state the parameters specify is reachable and non-trivial in the sense that the state is non-tautological."

Generator feature schema (verbatim from `rule_generation.py`):

```python
DEFAULT_FEATURES = {
    "bool": [ "CUR_CELL_HAS_APPLE", "CUR_CELL_IS_FOREIGN_PROPERTY" ],
    "discrete": {
        # 'KEY': (compare_condition, values, goal_condition)
        "DIRT_FRACTION": ('>', [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], '== 0'),
        "SINCE_AGENT_LAST_PAID": ('>', list(np.arange(10,31,5)), '== 0'),      # 10,15,20,25,30
        "RIOTS": ('len(obs["RIOTS"]) > 0', None, 'len(obs["RIOTS"]) == 0'),
        "NUM_APPLES_AROUND": ('<', [1, 2, 3, 4, 5, 6, 7, 8], '>'),
        "ORIENTATION": ('==', [0, 1, 2, 3], '!='),
    },
    "categorical": { "AGENT_LOOK": ["0", "1", "2"] },
}
CELL_BASED_FEATURES  = ["NUM_APPLES_AROUND", "CUR_CELL_HAS_APPLE", "CUR_CELL_IS_FOREIGN_PROPERTY"]
ENV_BASED_FEATURES   = ["DIRT_FRACTION"]
AGENT_BASED_FEATURES = ["SINCE_AGENT_LAST_PAID", "ORIENTATION", "AGENT_LOOK"]
DEFAULT_ACTIONS = ["MOVE_ACTION"]   # TURN/ZAP/CLEAN/CLAIM/EAT/PAY commented out
```

Generation procedure (`generate_rules_of_length(3)`): for every combination of 1..3 features,
- Prohibitions: only if all features are from the same group (cell/env/agent); `NUM_APPLES_AROUND` only together with `CUR_CELL_HAS_APPLE`; never `SINCE_AGENT_LAST_PAID` (`no_prohibitions`); conjunction of per-feature conditions (cartesian product over thresholds) × each action in `DEFAULT_ACTIONS` -> `ProhibitionRule(cond, 'MOVE_ACTION')`.
- Obligations: never with `CUR_CELL_HAS_APPLE, CUR_CELL_IS_FOREIGN_PROPERTY, NUM_APPLES_AROUND, ORIENTATION` (`no_obligations`); never both `DIRT_FRACTION` and `SINCE_AGENT_LAST_PAID`; goal is fixed by the feature (`DIRT_FRACTION` -> `obs["SINCE_AGENT_LAST_CLEANED"] == 0`; `SINCE_AGENT_LAST_PAID` -> `obs["SINCE_AGENT_LAST_PAID"] == 0`; `RIOTS` -> `len(obs["RIOTS"]) == 0`); every non-RIOTS obligation is role-conditioned by appending ` and obs["AGENT_LOOK"] == {look}` for look in {0,1,2}; duplicates removed by string repr.

Resulting space (Table A1, paraphrased in the paper's English glosses):
- Prohibitions 1-31: "If a cell holds an apple -> Don't move there"; "If a cell is not your property -> Don't move there"; "If the river dirt fraction > {0.3,...,0.6} -> Don't move at all" (7); "If you're looking towards {N,E,S,W} -> Don't move at all" (4); "If a cell holds an apple and it's not your property -> Don't move there"; "If a cell holds an apple and the number of apples around < {1..8} -> Don't move there" (8; Table A1 prints the "< 8" row twice as rows 22 and 23); "If a cell holds an apple and it's not your property and the number of apples around < {1..8} -> Don't move there" (8).
- Obligations 32-68: "If the river dirt fraction > {0.3,...,0.6} and you look like a {cleaner, farmer, egalitarian} -> Clean the river" (21); "If the number of steps you haven't paid anyone > {10,15,20,25,30} and you look like a {cleaner, farmer, egalitarian} -> Pay someone" (15); "If another agent violated a norms -> Sanction them" (1).
- Hand count of the generator: 30 prohibitions + 37 obligations = 67 distinct rules; Table A1 reaches 68 by the duplicated row 22/23. (I did not execute the generator; this is a manual trace of `rule_generation.py`.)
- Note: `view_learning_model.py` actually calls `generator.generate_rules_of_length(2)` (length 2) for the passive learning runs, while `eval_learning.py`, `eval_intergen.py`, `eval_emergence.py` call `generate_rules_of_length(3)` but only `view_*` constructs the bots - the `POTENTIAL_*` lists generated in `eval_*` are unused there. So the learner in the shipped learning harness sees the length <= 2 subset (which still contains all five practised norms). The paper reports 68.

`lambda_rules.py` also contains an older hand-written `POTENTIAL_OBLIGATIONS` / `POTENTIAL_PROHIBITIONS` list (incl. `EAT_ACTION`/`TURN_ACTION` prohibitions and a `POSITION[0] == POSITION[1]` rule) used only by the legacy `RuleLearningPolicy` default args; the generator supersedes it.

---

## 5. Norm-compliant planning (Sec. 2.3)

Paper (verbatim):
> "Even when an agent is certain about N, complying with all of them can be challenging due to temporally extended obligations. As such, we design our agents to use two different planning modes, namely reward-oriented planning and obligation-oriented planning. Each of these corresponds to a different sub-problem that is a proxy of the true game. Agents switch from the former mode to the latter mode whenever an obligation precondition is triggered."

> "In reward-oriented planning, agents maximize Equation 2 except with the cost of obligation norms ignored. Since prohibition norms are not temporally extended, this gives us a proxy reward R_i^{N_P}(s, a_i, s') that only depends on the current state transition. We maximize this reward through model-based planning, assuming that each agent i has a reasonably accurate model T_i(s'|s, a_i) of the environment and other agents. In particular, we use real-time dynamic programming (RTDP) [9] to compute estimates of the state and action value functions:"

    Q_i^{N_P}(s, a_i) = Σ_{s'∈S} T_i(s'|s, a_i) [ R_i^{N_P}(s, a_i, s') + γ V_i^{N_P}(s') ]      (3)
    V_i^{N_P}(s) = max_{a_i∈A_i} Q_i^{N_P}(s, a_i)                                            (4)

> "our agents reactively switch to obligation-oriented planning once the precondition Pre of some obligation ν_O = ⟨Pre, Post, τ⟩ is met and then plan to achieve Post as a goal. This creates a (stochastic) shortest-path problem, where any state s with s[Post] = 1 is treated as a terminal state, with a modified reward function that accounts only for action costs C_{a_i}, prohibition costs C_i(ν_P) and delivers a reward of 1 when Post is satisfied:"

    R_i^{ν_O ∪ N_P}(s, a_i, s') = C_{a_i} + Σ_{ν_P∈N_P} C_i(ν_P) ν_P(s, a, s') + s'[Post]      (5)

> "Agents thus act by selecting actions that maximize either Q_i^{N_P} or Q_i^{ν_O ∪ N_P} depending on whether an obligation ν_O is active, performing several iterations of RTDP to update the value functions before acting. Obligations are satisfied in the order that their preconditions are triggered, and each agent maintains a queue of such obligations to satisfy."

Acting under uncertainty (verbatim):
> "we consider two simple approaches, thresholding and sampling, which avoid the difficulty of taking expectations over all possible norm sets N ⊆ 𝒩. When thresholding, agents comply with a norm ν whenever its probability is higher than a threshold P_i^t(ν ∈ N) ≥ θ. When sampling, agents sample of set of norms N' ∼ P_i^t(N) to comply with for a certain number of steps K, before resampling. This is also known as probability matching [29] or Thompson sampling [76]. [...] Thresholding is low variance and may be appropriate when norm violation is sufficiently costly. In contrast, sampling can promote exploration, especially under high uncertainty about which norms are true [27]."

Appendix parameters (verbatim): "We use real-time dynamic programming (RTDP) [9] for the norm-compliant planning algorithm with a rollout depth of 20 every 2 steps and a discount factor of γ = 0.9. Agents thereby comply with the norms at threshold θ = 0.95 or a sampling frequency of K = 10 for the intergenerational norm learning setting (Section 3.3) and K = 15 for the norm emergence setting (Section 3.4) to secure more robust emergence possibilities."

As implemented (`rule_obeying_policy.py` / `rule_adjusting_policy.py`):
- Each agent has its own *single-agent forward model* `env_step(timestep, action, idx)` that simulates one action on a deep-copied observation dict (movement, apple pickup, turning, zap/clean/claim/eat/pay bookkeeping). Other agents are treated as static obstacles (`element_blocking_cost`), not modelled.
- Value tables: `self.V = {goal: {state_hash: Q[12]}}` for goals `'apple','clean','pay','zap'` (reward-oriented mode = goal `'apple'`; obligation mode = goal named after the obligation's goal key), plus `self.V_wo_rules = {state_hash: Q[12]}` (values *as if no rules were active*, used for the learner's likelihood). State hash = SHA-256 of a goal-specific subset of observation keys (`relevant_keys`) + reward. Unseen states are initialised with a heuristic (`init_heuristic`, distance-based).
- `rtdp(ts_start)`: `n_rollouts` greedy rollouts of depth `max_depth = 20`, each step `update(ts)` doing a one-step Bellman backup over all 12 actions, followed by backward sweep over visited states. Defaults in `RuleAdjustingPolicy`: `n_steps = 2` (re-plan every 2 steps), `n_rollouts = 2`, `max_depth = 20`, `gamma = 0.9`, `tau = 1.0`. Action selection `get_act(..., temp=0.0)` is greedy argmax.
- `get_bellmann_update` (the de facto Eq. 2-5): `v_rules = γ·max V[goal][s'] + r_cur − norm_cost − obligation_violation_cost`, where `r_cur = reward − action_cost` (apple mode) or `r_cur = obligation_reward if goal satisfied else 0` (obligation mode); `norm_cost = intrinsic_violation_cost (1)` if the action is not in `available_actions` (i.e. violates a currently-adopted prohibition), plus `punish_cost · γ^{avg_steps_to_punishment}` if a sanction ("RIOTS") obligation is adopted in the population (expected-punishment term; `avg_steps_to_punishment = 50 / num_players`); `obligation_violation_cost = intrinsic_violation_cost (+ expected punishment)` once `GOAL_COUNT > max_obligation_depth (20)` without satisfying the goal. Blocking costs for walking into agents (0.1) or water (1.0).
- Obligation queue: `self.current_obligations = deque()`; `step()` scans `self.obligations` with `holds_in_history` over the last <= 10 timesteps and appends newly triggered ones; satisfied ones are removed; `set_goal()` picks the front of the queue (`get_cur_obl` maps the goal string to `'clean' | 'pay' | 'zap'`), else `'apple'`.
- Thresholding/sampling: `threshold_rules()` adopts every potential rule with belief >= `threshold (0.95)`; `sample_rules()` Thompson-samples each rule independently (`belief > U(0,1)`) with subset-pruning (`add_rule_with_pruning` keeps the most specific of nested threshold rules). Harness: learning mode re-thresholds every step; intergen and emergence modes resample every `SAMPLE_RHYTHM` steps (15 in `view_emergence_model.py`).

---

## 6. Bayesian norm learning (Sec. 2.4)

Paper (verbatim):
> "To specify this distribution, we make two assumptions: (i) Agents model each other as norm-compliant planners, and (ii) they assume that other agents act as if they know the norms N with certainty. Since agents have full observability and can observe the actions a_{−i} that all other agents take at state s after history h. This induces the following posterior:"

    P_i^{t+1}(N | h, s, a_{−i}) ∝ π_{−i}(a_{−i} | h, s, N) P_i^t(N) / P_i^t(a_{−i} | h, s)      (6)

> "The likelihood term π_{−i}(a_{−i}|h, s, N) is the policy over actions that results from all other agents following the procedure in Section 2.3, conditional on all of them knowing that N is true. This means that actions that comply with a norm ν ∈ N are more likely to occur, and violations are less likely. Note, however, that this may not be an accurate model in general. After all, other agents might also be uncertain about the norms, and act based on their beliefs about which norms are true. Nonetheless, the assumption of shared normativity greatly reduces the complexity of norm learning: [...] our agents avoid having to infer everyone else's beliefs about the norms, and directly infer the norms themselves."

Mean-field approximation (verbatim):
> "inferring the exact posterior over N — the rule induction problem [33, 63] — is still highly intractable, since it requires enumerating over all subsets N' ⊆ 𝒩 and updating their probabilities. Instead, we compute a mean-field approximation [85] of the posterior, assuming that it factorizes into approximate local posteriors P̃^t(ν|h, s, a_{−i}) for each potential norm ν ∈ 𝒩:"

    P_i^{t+1}(N | h, s, a_{−i}) ∝ Π_{ν∈𝒩} P̃^{t+1}(ν | h, s, a_{−i})      (7)

> "Making this approximation allows us to update beliefs about each norm ν independently via Bayes rule:"

    P̃^{t+1}(ν | h, s, a_{−i}) ∝ π_{−i}(a_{−i} | h, s, ν) P̃_i^t(ν)      (8)

> "where π_{−i}(a_{−i}|h, s, ν) approximates the probability that the other agents take actions a_{−i,t} under the assumption that ν is the only norm that is true. We compute this probability by using the Q-values described in Section 2.3, assuming that each agent j selects its action a_j according to a Boltzmann distribution:"

    π_{−i}(a_j | h, s, ν) ∝ exp(Q^ν(s, a_j))      (9)

> "By using the Q-values derived by planning, we automatically take into account the fact that other agents are unlikely to violate norms since such actions will have lower Q-values."

Appendix (verbatim): "Every potential norm out of the 68 generated ones (see Table A1) was given an initial prior of P(ν) = 0.05. We employed a Boltzmann temperature of 1."

### As implemented (`RuleAdjustingPolicy.update_beliefs / compute_posterior / comp_*_llh`)

Constants: `DEFAULT_INIT_PRIOR = 0.05`, `DEFAULT_P_OBEY = 0.9` (bounded-compliance parameter; the paper's ε corresponds to 1 − p_obey), `DEFAULT_TAU = 1.0` (Boltzmann temperature, applied to z-scored Q-values in `compute_boltzmann`), `DEFAULT_THRESHOLD = 0.95`. Experienced agents start with belief `threshold` (0.95) in their active rules and `init_prior` in all others; learners start at `init_prior` everywhere (`rule_beliefs` array over `potential_rules`).

Per step, for every *other* agent j (own actions are not used: "Assumption: players are not updating on their own actions") and every potential rule r, the update in `compute_posterior` (verbatim comments):

```python
# P(r = 1)
prior = self.rule_beliefs[rule_idx]
# P(a | r = 1)
log_llh = comp_prohib_llh(...) or comp_oblig_llh(...)
# P(a) = P(a | r = 1) P(r = 1) + P(a | r = 0) P(r = 0)
marginal = np.exp(log_llh) * prior + p_a_obs_no_rules * (1 - prior)
# P(r=1 | a) = P(a | r = 1) * P(r = 1) / P(a)
posterior = exp(log_prior + log_llh - log_marginal)
```

where `p_a_obs_no_rules = Boltzmann(agent j's V_wo_rules[s_past])[a_j]` is the no-rule likelihood P(a | r=0) (the learner reads agent j's *actual* planner tables via `self.all_bots[j]` - a simplification exploiting shared code rather than re-planning from j's perspective).

Prohibition likelihood `comp_prohib_llh`: compute `prohib_actions` = actions from s_past that would make `rule.holds(new_obs, action_name)` true (plus off-map/water moves); recompute Q-values for j with those actions unavailable (`get_bellmann_update(..., available, ...)` adds the violation cost), Boltzmann them and zero out prohibited actions -> `p_a_obs_rule_is_active`. Then

```python
if could_be_violated:           # len(prohib_actions) != 0
    if was_violated: self.maybe_mark_riot(player_idx, rule)
    return log( p_a_obs_rule_is_active * p_obey + p_a_obs_no_rules * (1 - p_obey) )
else:
    return log( p_a_obs_no_rules )      # rule not applicable in this state: no evidence
```

So a *violation* (where `p_a_obs_rule_is_active = 0`) yields likelihood `(1−p_obey)·P(a|no rule)` = 0.1 × baseline; compliance yields a mixture >= baseline when the compliant action is also attractive under the rule.

Obligation likelihood `comp_oblig_llh`: track, per observed agent and rule, how many steps the obligation has been active (`nonself_active_obligations_count`, reset when `holds_in_history` stops holding); `rule_is_active` iff count <= `max_obligation_depth` (20 = τ). If active and `could_be_satisfied(rule, past_ts, j)` (some single action from s_past would satisfy the goal):

```python
if rule.satisfied(this_obs):   # obeyed
    return log( p_obey * p_a_rule_is_active + p_a_obs_no_rules * (1 - p_obey) )
else:                          # disobeyed
    maybe_mark_riot(player_idx, rule)
    return log( p_a_obs_no_rules * (1 - p_obey) )
else:  # can't be satisfied in one step, or not active/expired
    return log( p_a_obs_no_rules )   # uninformative
```

Hence obligations only generate evidence at the step where the goal is one action away; this is why the paper notes O1 (farmer must chase the cleaner to pay) was learned poorly ("the learning agent often inferred that the norm was non-existent rather than merely hard to satisfy").

Sanctions: `maybe_mark_riot` appends a violator to `self.riots` only if the violated rule is among the observer's *active* rules; the `RIOTS` obligation (`len(obs["RIOTS"]) > 0 -> len(obs["RIOTS"]) == 0`, "If another agent violated a norms -> Sanction them") is in `DEFAULT_OBLIGATIONS` and the generated space, and the planner adds expected-punishment cost when it is adopted, but `eval_learning.py` explicitly strips it from `DEFAULT_RULES` (`if "RIOTS" in rule.make_str_repr(): DEFAULT_RULES.remove(rule)`), consistent with the paper's statement that experiments were run *without* sanctions.

---

## 7. Experiments (Sec. 3)

General: "In all of our experiments, we used a sample size of 13 runs for each condition, determined based on an effect size of 0.8 with α = 0.05 (one-sample t-test assuming a normal distribution)." (`stats_relevance = 13` in all `eval_*.py`.) Four phenomena studied: (1) Passive Norm Learning, (2) Norm-Enabled Social Outcomes, (3) Intergenerational Norm Transmission, (4) Norm Emergence and Convergence.

### 3.1 Passive norm learning

Setup (verbatim): "we investigated whether a single learning agent l could acquire the same norms believed in by a set of M_Exp = 3 experienced agents, each playing one of three roles z ∈ Z := {C, F, E} (read as "cleaner", "farmer" and "egalitarian"). The learning agent also had an egalitarian role (z_Lea = E), and each episode lasted H = 300 steps. All experienced agents assigned a probability of 1 to the norms N_active = {P1, P2, O1, O2, O3} shown in Figure 1b. The full space 𝒩 of 68 possible norms was generated by enumerating over a variety of predicates and numeric thresholds (see Appendix). The learning agent handled uncertainty over norms via thresholding, and had a high norm violation cost, motivating it to comply with learned norms."

Metrics: posterior belief per norm over time (Figure 2, "Compliance Threshold" line at 0.95, "False Positives" band); precision/recall of acquiring (P(ν) >= 0.95) the active norms at 25/50/75/100% of the episode as |N_active| varies 0..5 (Table 1).

Table 1 (verbatim values, Pre/Rec):

| \|N_active\| | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| 0 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| 1 | 0.15/0.09 | 0.08/0.40 | 0.06/0.54 | 0.06/0.57 |
| 2 | 0.19/0.27 | 0.14/0.59 | 0.13/0.76 | 0.12/0.82 |
| 3 | 0.24/0.37 | 0.21/0.63 | 0.18/0.77 | 0.18/0.83 |
| 4 | 0.35/0.36 | 0.28/0.67 | 0.24/0.77 | 0.22/0.83 |
| 5 | 0.47/0.25 | 0.43/0.57 | 0.34/0.79 | 0.32/0.83 |

Results (verbatim): "The majority of norms practiced by the experienced agents were rapidly and effectively internalized within 300 steps, in contrast to the millions of steps required by model-free norm learning approaches [49]. This was despite some norms being complied with only occasionally (e.g. the cleaner's obligation to clean the dirty river), providing a limited learning signal for observers. Interestingly, Obligation 1 (concerning the farmer's payment) was not learned very well, with an average confidence of 0.4 even after 300 steps."

> "While we observed a higher recall with more time steps, precision was low in most cases, indicating that the learner was "over-learning" norms." Two causes: (a) "accidentally fulfilled prohibitions" - "agents learned to not be allowed to move when there was > 55% of dirt in the river which was a state that was implicitly avoided due to the river being cleaned on a regular basis"; (b) "agents often learned logically overlapping norms, reflecting the fact that some rules in 𝒩 logically entailed other rules (e.g. if Dirt(r) > 0.4 is true, then Dirt(r) > 0.3 is also true). This behavior was likely due to the mean-field approximation of the posterior in Equation 7, which prevented explaining away among overlapping norms".

Harness: `eval_learning.py` runs `EXPERIMENT_ROLES = ('cleaner','farmer','free','learner')` for every subset of the 5 default rules (`RULE_COMBINATIONS`, 32 incl. empty) × 13 seeds, writing per-step CSVs (`reward per role, action per role, DESSICATED, belief per potential rule`) to `examples/results_learning/rule_trials/scenario{i}/trial{k}.csv`; plus baselines with no learner.

### 3.2 Norm-enabled social outcomes

Metrics (verbatim): "social welfare in terms of the collective cumulative reward obtained by all agents [92] (hereafter cumulative reward) (Figure 3a) and sustainability of the environment [72] (Figure 3b), as measured by the ratio of desiccated apple fields in our environment relative to the total number of fields." Independent variable: N ∈ {∅, {P1, P2, O1, O2, O3}}; same 1 learner + 3 experienced setup.

Results (verbatim): "the presence of norms P1 through O3 led to a clear increase in social welfare (Figure 3). Specifically, agents tended to achieve greater collective rewards with these norms in our environment (Figure 3a). Unlike the scenario with no active norms, cumulative reward exhibited a consistent uptrend with time. [...] The percentage of desiccated fields (where apples do not regrow) increased rapidly in the absence of resource management norms, but remained at a low value when norms were present (Figure 3b)." Figure 3a y-axis spans 0-100 cumulative reward over 300 steps; Figure 3b desiccated ratio 0-0.6.

> "not all norms are equally beneficial [...] In particular, the prohibitions against eating apples in less dense areas appeared to have limited impact. This might have been because the potential sustainability gain from obeying the prohibition directly traded off against the welfare gain of eating more apples. In contrast, once the cleaning obligations were fulfilled, the environment became significantly better maintained, enabling more apples to grow, and causing cumulative reward to keep increasing." (Figures A2/A3: cumulative reward and desiccated ratio for every 1-, 2-, 3-, 4-norm subset.)

### 3.3 Intergenerational norm transmission

Setup (verbatim): "we ran an experiment with M = 6 learning agents, while assigning each of the three role-associated appearances z ∈ Z to two out of the six agents (effectively forming a parent-child pair for each role). Agents used sampling to select which norms to comply with. Each agent i had a maximum life span of L, and an initial age of (i/M)·L [typeset M_i L in the PDF]. Each run was simulated for 1.5ML steps (i.e. three generational cycles). Upon an agent i's demise, it was replaced by an equivalent "child" agent with the same role-associated appearance z_i, but with a low uniform prior over norms P_i(ν) = 1 − θ for all ν ∈ 𝒩. Initially, 3 experienced agents were present to seed the norms. As in earlier experiments, all agents were intrinsically motivated to comply with the learned norms." Life spans tested (Figure 4 legend): L ∈ {10, 30, 50, 100, 300, 600}; K = 10 resampling.

Results (verbatim): "Starting with the prohibition against eating too many apples (Figure 4a), we observed striking rapidity in both its acquisition and transmission. Indeed, a life span as brief as 50 steps was sufficient for the effective transmission of these prohibition norms". "Obligation norms presented a more complex scenario [...] When learning the norms that applied to a particular role, an agent could only observe the two agents that were in that role, not all 6 agents. If they were the "child" agent in that role, they had only one agent to learn from [...] our results indicated that even an extended life span of 600 steps did not ensure that obligations were reliably transmitted." "What we found instead was collective forgetting of all initially active obligations. [...] Since agents resampled the norms they would comply with every K = 10 steps, this added stochasticity that interfered with the long-horizon goals involved in obligation-oriented planning: If an obligation was not sampled at any step while fulfilling its goal, an agent would give up on the goal and go back to maximizing reward." Conclusion: "Under some conditions, social norms are successfully maintained and transmitted (Figure 4c), and this is more likely to occur with more opportunities to learn those norms from agents who practice them."

Harness: `eval_intergen.py` with `ROLES = ('cleaner','farmer','free','cleaner','farmer','free')`, `episodes = DEFAULT_MAX_LIFE_SPAN*9`, staggered initial ages `range(0, 6*L/5, L/5)`; on `freeze_counter == 1` the bot's `rule_beliefs` reset to `DEFAULT_INIT_PRIOR`. Result CSVs per bot: `reward, action, dead_apple_ratio, is_frozen, <belief per potential rule>` in `examples/results_intergen/max_life_span{L}/bot{j}/trial{k}.csv` (only trial1 for L=30 is checked in).

### 3.4 Norm emergence and convergence

Setup (verbatim): "we simulated a population of M = 3 learning agents with z ∈ {C, E, F} (every role represented once). As in our norm transmission experiments, agents select norms to comply with via sampling, effectively performing a form of norm exploration when uncertain about which norms hold." K = 15. No agent starts with any norm; no sanctions.

Metric: "Geometric and arithmetic mean posterior beliefs of all agents for the top 5 norms with final belief P^{t=300}(n) ≥ θ = 0.95" (Figure 5); Table A2 lists the top-10 spontaneously emerging norms and their occurrence frequency averaged over 13 runs: "If the river dirt fraction > 0.55 -> Don't move" 0.61%; "> 0.6 -> Don't move" 0.61%; "dirt > 0.3 and you look like an egalitarian -> Clean the river" 0.61%; "looking towards East -> Don't move" 0.46%; "cell holds an apple and apples around < 1 -> Don't move there" 0.38%; "not your property -> Don't move there" 0.31%; "apple and not your property -> Don't move there" 0.31%; "looking West -> Don't move" 0.31%; etc.

Results (verbatim): "The gradual increase in the arithmetic mean over multiple runs signifies an emergent belief in the existence of certain norms, even though such beliefs may not initially be shared by all agents. This eventually results in a sharp rise in the geometric mean, demonstrating convergence and unanimous agreement among agents on specific norms." "we found that many of the learned norms were non-functional, or "silly" norms [38]. This was likely due to intrinsic norm violation costs, which help to stabilize inefficient or even harmful norms [13]. As such, we expect that emergent norms are more likely to be beneficial when agents are more strategic about which norms they comply with."

### Summary of experimental conditions

| Experiment | Agents | Learners | Norm selection | Sanctions | Steps | Metrics |
|---|---|---|---|---|---|---|
| Passive learning | 3 experienced (C,F,E) + 1 learner (E) | 1 | threshold θ=0.95 | none | 300 | posterior per norm; precision/recall at P>=0.95 |
| Social outcomes | same | 1 | threshold | none | 300 | cumulative collective reward; desiccated-field ratio |
| Intergenerational | 6 (2 per role), 3 seeded | all | sampling K=10 | none | 1.5·M·L, L∈{10..600} | mean belief per norm across generations |
| Emergence | 3 (C,E,F), none seeded | all | sampling K=15 | none | 300 | arithmetic/geometric mean belief of top-5 norms; occurrence table |

---

## 8. Discussion points relevant to a Rules-first spec language (Sec. 4, verbatim excerpts)

Sanctions: "Perhaps the most important difference between our experiments and previous work on decentralized norm learning [...] is the absence of sanctions — acts of punishment or approval — in motivating and sustaining norm compliance. Rather, our experiments focused on the epistemic aspect of norm learning, i.e., learning the content of normative rules, and ensuring they remain commonly known, leaving open the motivational aspect of norm learning." "Sanctioning could be introduced as a primitive action (e.g. the "zapping beams" supported by Melting Pot [1, 50]), a learned policy (e.g. taking others' apples in retaliation for theft), or a (meta)-norm that obligates punishment of agents who do not comply with norms." "Sanctions provide both evidence that a norm exists, and motivation to comply with it."

Symbolic rules and LLMs: "If norms lacked a symbolic, language-like structure, we would find it hard to compose, generalize, communicate, and argue about them [60]." "a norm or principle ν could be represented in natural language itself, using an LLM to classify whether behavior complies with the norm [64]. Alternatively, LLMs might translate natural language, such as legal documents, into formal specifications [19, 58] [...] LLMs could thus serve as priors over norms P_i^0(N) for a norm learner, or as approximate posteriors P_i(N|u) over norms N given a piece of normative language u."

Model-free vs model-based: "In general, learning how to generate norm-compliant behavior is harder than learning to discriminate it [99]. Oftentimes, it may be more efficient to learn habitual practices that comply with a norm, rather than planning them."

Norm adaptation: "defaulting to learned rules in tried-and-true situations, adjusting them in new scenarios, and forging new norms via agreement when it is feasible and worthwhile."

---

## 9. The code release

Repo: https://github.com/ninell-oldenburg/social-contracts - MIT, 394 commits, last commit 2024-03-12, 14 stars, 1 fork (unnamed), no issues/topics. It is a *vendored fork of Melting Pot 2.0* (DeepMind, Apache-2.0 headers) with one new substrate and new Python policies; README instructs to install upstream Melting Pot and overwrite its `meltingpot/` directory with this repo's `meltingpot/`. Python >= 3.10. Rendering via `dmlab2d.ui_renderer.pygame`; videos via ffmpeg.

README (verbatim, key parts): "With this piece of code, we formalize the problem of constrained norm learning from observing other agents' actions in the context of Markov games in a multi-agent environment via approximately Bayesian rule induction of obligative and prohibitive norms. The environment is based on the Melting Pot project by DeepMind taking together the different games Commons Harvest, Clean Up, and Territory to allow agents to learn and sustain a range of norms to handle various dilemmas such as a tragedy of the commons, social-role conditioned labor, and territorial norms." Three modes: "1. With `learning` you can see how well and how fast norms are being picked up by one learning agent [...] 2. With `intergen` [...] on a regular basis (`episode length` / `# agents`) agents die and then resurrect without any norm beliefs. [...] 3. Lastly, with `emergence` you can see what norms evolve when no agent believes in any norm." "The potental norms are created at agent creation time from a set of observable environment features (see `meltingpot/python/utils/rule_generation.py`)" - actual path is `meltingpot/python/utils/policies/rule_generation.py`.

Structure (paths relative to the clone):
- `meltingpot/python/utils/policies/ast_rules.py` (146 lines) - `EnvironmentRule`, `ProhibitionRule`, `ObligationRule`; role/sprite maps.
- `meltingpot/python/utils/policies/lambda_rules.py` (163) - `DEFAULT_PROHIBITIONS`, `DEFAULT_OBLIGATIONS`, named rule groups `CLEANING_RULES`, `PICK_APPLE_RULES`, `TERRITORY_RULES` (used to select the substrate variant), legacy `POTENTIAL_*` lists.
- `meltingpot/python/utils/policies/rule_generation.py` (220) - `RuleGenerator` hypothesis-space enumerator.
- `meltingpot/python/utils/policies/rule_obeying_policy.py` (1185) - base planner: forward model `env_step`, feature derivation, RTDP, Bellman update with norm costs, Boltzmann, `available_actions`/`check_all` (prohibition filtering), state hashing.
- `meltingpot/python/utils/policies/rule_learning_policy.py` (338) - earlier learner (threshold/sample, likelihoods); mostly superseded.
- `meltingpot/python/utils/policies/rule_adjusting_policy.py` (548) - the agent used by all experiments (`RuleAdjustingPolicy(RuleLearningPolicy)`): all hyperparameters as module constants, `step()`, obligation queue, age handling, `update_beliefs`, `comp_oblig_llh`, `comp_prohib_llh`, `maybe_mark_riot`.
- `meltingpot/python/utils/policies/agent_timestep.py` - `AgentTimestep` (step_type, reward, observation dict, goal, age).
- `meltingpot/python/configs/substrates/rule_obeying_harvest__*.py` (8 variants, ~1300-1450 lines each) - map, prefabs, avatar components, observation metrics, roles.
- `meltingpot/lua/levels/rule_obeying_harvest/components.lua` (1735) - `DensityRegrow, DirtTracker, DirtCleaning, Harvestable, Neighborhoods, Paintbrush, Resource, AllNonselfCumulants, Age, Cleaner, Eating, Inventory, ResourceClaimer, Paying, Property, Surroundings, GlobalData, RiverMonitor, DirtSpawner, ...`; `init.lua`.
- `meltingpot/python/utils/puppeteers/rule_obeying_agent_v{1,2}.py`, `rule_obeying_harvest.py` - older puppeteer-based attempts.
- `meltingpot/python/human_players/play_rule_obeying_harvest.py` - keyboard play.
- `examples/eval_scripts/view_{learning,intergen,emergence,selfish-learning}_model.py` - the environment loop (`main(roles, episodes, num_iteration, rules, env_seed, create_video, log_output, log_weights, save_csv, ...)`): builds substrate, constructs one `RuleAdjustingPolicy` per role, each step: `add_non_physical_info` -> `append_to_history` -> `bot.step()` -> (learners) `update_beliefs(last_actions)` + `threshold_rules()`/`sample_rules()` -> `env.step(actions)` -> `append_to_dict` logging; optional pygame render / PNG frames / ffmpeg video; optional CSV dump of value tables.
- `examples/eval_scripts/eval_{learning,intergen,emergence,selfish-learning}.py` - experiment drivers (13 seeds, scenario loops, CSV output); `plots_*.ipynb` and `notebooks/evaluation_results.ipynb` for figures.
- `examples/results_intergen/max_life_span{10,30,50,70,90,100}/bot{1..6}/trial1.csv` - the only checked-in raw results (CSV header: index, reward, action, dead_apple_ratio, is_frozen, then one column per potential rule string repr holding the posterior at each step).

Harness pieces that exist: synchronous lock-step environment loop; per-agent planners with a hand-written single-agent forward model (no learned world model); per-step CSV logging of reward/action/beliefs/sustainability; role assignment by substrate `roles` tuple; mortality/rebirth; rendering/video. Pieces that do not exist: a declarative rule file format (rules are Python strings in code), config files/CLI (everything is module constants and hard-coded `main()` kwargs), tests for the new policies, any LLM involvement.

---

## 10. Takeaways for the spec-language project

- Rule type system in the paper = two constructors over first-order state predicates: `Prohibition(Prohib: actions-of-objects, Post: formula over s')` and `Obligation(Pre: formula, Post: formula, τ: horizon)`. Both reduce to history classifiers ν: H -> {0,1}. Implementation collapses them to `(python-bool-expr over obs, action-class)` and `(python-bool-expr, python-bool-expr)` with a single global τ = 20.
- Grounding: rules reference only *named scalar/boolean observation features* (`DIRT_FRACTION`, `NUM_APPLES_AROUND`, `CUR_CELL_IS_FOREIGN_PROPERTY`, `AGENT_LOOK`, `SINCE_AGENT_LAST_PAID`, `RIOTS`), each with an explicit comparator and threshold grid; the hypothesis space is the product of (feature-combos <= 3) × thresholds × roles × action classes, filtered by hand-coded feasibility constraints (feature groups, exclusions). Table A1 enumerates the 68.
- Compliance semantics: prohibitions are enforced by *masking actions* whose simulated successor satisfies the postcondition (plus a cost of 1 if nevertheless taken); obligations are enforced by *switching the planning goal* to the postcondition with reward 1 and a violation cost of 1 after τ steps. Obligations are queued FIFO.
- Learning semantics: independent Bernoulli belief per rule (prior 0.05), updated by a two-hypothesis Bayes step per observed (other-agent, step): likelihood = mixture `p_obey·π_rule(a) + (1−p_obey)·π_norule(a)` with `p_obey = 0.9`, where π are Boltzmann(τ=1) over the planner's Q-values with/without the rule; uninformative when the rule is not applicable in that state. Adoption by threshold (0.95) or Thompson sampling every K steps.
- Sanction hook: a meta-obligation "if someone violated a norm you hold, zap them" and an expected-punishment term in the value function exist in code but were switched off in the reported experiments.
- Metrics used: per-rule posterior trajectories; precision/recall of adopted vs. true rules; collective cumulative reward; desiccated-field ratio; belief persistence across generations; arithmetic vs geometric mean belief as a convergence signal; occurrence frequency of emergent rules.

---

## Sources

- arXiv abstract page: https://arxiv.org/abs/2402.13399 (metadata, versions, code link)
- arXiv PDF v2: https://arxiv.org/pdf/2402.13399 (full text incl. appendix; local copy and pdftotext output as listed at top)
- arXiv HTML: https://arxiv.org/html/2402.13399v2 (exists; only summarised by the fetch tool, not used for quotes)
- PhilArchive record: https://philarchive.org/rec/OLDLAS
- Code: https://github.com/ninell-oldenburg/social-contracts (cloned at commit 2c40f279b3104d1e800dfa9cc772fba6018f581a; files cited by path above)
- Upstream environment: DeepMind Melting Pot 2.0, https://github.com/google-deepmind/meltingpot (substrates `commons_harvest__closed.py`, `clean_up.py`, `territory.py` referenced by the README)
