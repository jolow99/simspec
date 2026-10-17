# ContractSim: "Evaluating Rational Contracting in Natural Language" (arXiv 2608.10475)

Research notes compiled 2026-10-16 from the **full paper** (TeX source + PDF). Everything below is sourced from the paper's own text; verbatim passages are quoted. Notation is reproduced in LaTeX-ish plain text.

## 0. Provenance and what was obtained

- **arXiv ID resolves.** `https://arxiv.org/abs/2608.10475`, v1 submitted Tue 11 Aug 2026 04:44:50 UTC by Tan Zhi-Xuan. Subjects: cs.AI (primary), cs.CL, cs.GT. Comments: "9 pages, 5 figures, 2 tables (Appendix: 34 pages, 6 figures, 9 tables)". Template is AAAI 2027 (`aaai2027.sty`).
- **Authors (verbatim):** Bhavyesh Sajja (NUS CS), Max Kleiman-Weiner (U. Washington CS), Roger Zimmermann (NUS CS), Tan Zhi-Xuan (NUS CS + A*STAR Institute of Advanced Intelligence and Computing). Emails: bsajja@u.nus.edu, maxkw@uw.edu, dcsrz@nus.edu.sg, xuan.cs@nus.edu.sg.
- **Full text obtained three ways:** (1) PDF downloaded with curl (2,881,945 bytes, PDF 1.7) and extracted with `pdftotext -layout` to 228 KB of text at `(local scratch, not committed)`; (2) the arXiv HTML (`https://arxiv.org/html/2608.10475v1`, 1.5 MB); (3) **the arXiv TeX source** (`https://arxiv.org/src/2608.10475`, gzip tar), unpacked to `(local scratch, not committed)` with `main.tex` and `sections/{abstract,introduction,formal-framework,contractsim,evals,discussion,front_tables,appendix-body,appendix_prompts_env9,appendix_performance_tables_full,appendix_performance_tables_micro,appendix_story_performance_tables}.tex`. The TeX source is the authoritative basis for all quotes below.
- **Code/data release: none.** The arXiv page lists no code link; the TeX source, bib, and HTML contain no GitHub/HF/OSF/project-page URL for this work (the three openreview.net links in the HTML are bibliography entries for cited papers). An extended web search found no repository. A third-party review at pith.science states "no code or data appears to be released." The paper's harness is described in prose only (Section 8 below).
- **Nothing was fabricated.** Where the paper does not specify something (e.g. exact Concordia component wiring, dynamic turn-request wording) I say so.

---

## 1. The formal model: negotiation-performance game

### 1.1 Top-level game (Section "Contracting as a Negotiation-Performance Game")

> A *negotiation-performance game* G is described by the tuple: G = (I, Θ, Ω, N, P), where I={1,2} is the player set, Θ is the type-profile space, Ω is the contract space, N is the negotiation game, and P is the performance game. Each run realizes types θ=(θ_i)_{i∈I}∈Θ that describe each player's private information about their preferences or environment. Players negotiate over Ω in N; agreement produces ω∈Ω, which they perform in P with their types retained. Disagreement is terminal.

### 1.2 Negotiation game

> A *negotiation game* N is defined by the tuple (M, Ω, α), where M is the natural-language message space, and Ω is the natural-language contract space, and a protocol α that maps a message history h ∈ M* to an agreed contract ω ∈ Ω, disagreement ⊥, or continuation ⊤. Each turn, a player sends a message m ∈ M. Based on the history h, α determines if negotiation continues (⊤), ends in disagreement (⊥), or ends with a contract ω extracted from h.

### 1.3 Performance game (POSG)

> A *performance game* P is a partially-observable stochastic game [Hansen et al. 2004] defined by the tuple (S, (A_i)_{i∈I}, (O_i)_{i∈I}, T, (O_i)_{i∈I}, L, (u_i)_{i∈I}), where S is the environment state space, A_i is player i's action space, O_i is player i's observation space, T(·|s,a) is the transition kernel for joint action a=(a_i)_{i∈I}, O_i(·|s,a) is player i's observation model, L ∈ ℕ is the length of the game, and u_i(τ;θ_i) is player i's utility over complete trajectories τ ∈ T, where T := S×(A×S)^L. Execution starts from an initial state s_0 ∈ S, after which each player i follows a policy π_i which may condition on their observations, the contract, and the negotiation message history: π_i(a_{i,t} | o_{i,1:t}, θ_i, ω, h).

Note: utilities are **trajectory-level** (not per-step rewards), and policies may condition on the full negotiation transcript h (though in the actual performance experiments the agents are *not* shown a transcript; see 8.3).

### 1.4 Contracts as constraints on joint policies (key modelling move)

> we assume that a contract ω can be interpreted as a set of contractually-compliant policies Π^ω ⊆ Π, where Π is the full space of joint policies π = (π_i, π_{-i}). We further refine this by assuming Π^ω is defined by a constraint C^ω: T → {0,1} on the performance trajectory τ and a violation probability ε, such that Π^ω := {π ∈ Π | P_sat(π, C^ω) ≥ 1−ε}. P_sat(π, C^ω) := E_τ[C^ω(τ) | π] is the probability of π satisfying the constraint C^ω, so a contractually-compliant policy is one that satisfies C^ω with high probability.

The paper says this captures: *contingent contracts* (history-dependent C^ω) and *incomplete contracts* via "*under-specification* --- Π^ω does not determine a unique policy --- or *unanticipated unsatisfiability* --- constraints C^ω may be impossible to satisfy in certain states".

### 1.5 Rational performance baselines (definitions)

**Rational Complier (RC)** (Eq. rc):

> π_i^RC ∈ argmax_{π_i∈Π_i} E[u_i(τ;θ_i) | ω, π_i, π_{-i}^ref]  s.t. (π_i, π_{-i}^ref) ∈ Π^ω.

with the reference counterparty π^ref_{-i} fixed by the contract: "in the supplier-customer games our benchmark studies, most contracts fully constrain the customer's policy (i.e. their payment schedule), leaving a unique choice for π^ref_{-i} where −i is the customer. This in turn determines the supplier's RC policy." In less constrained settings they suggest a Nash/Kalai-Smorodinsky bargaining solution among compliant policies.

**Rational Exploiter (RE)** (Eq. re):

> π_i^RE ∈ argmax_{π_i∈Π_i} E[u_i(τ;θ_i) | ω, π_i, π_{-i}^RC].

**Rational Conditional Complier (RCC)** (Eq. rcc), with D_{i,t}∈{0,1} = "player i has observed a counterparty violation by time t":

> π_i^RCC = π_i^RC if D_{i,t}=0; BR_i(π_{-i}^RE) if D_{i,t}=1.

### 1.6 Rational negotiation (upper bound, Pareto frontier)

> we can find *upper bounds* on what rational negotiators can achieve by assuming all private information θ_{i,-i} is made public, then solving for the Pareto frontier of joint contract utilities.
> ... we assume that both players perform ω cooperatively by following a joint RCC policy π^RCC := (π_i^RCC, π_{-i}^RCC), giving us U_i(ω) = E[u_i(τ;θ_i) | ω, π^RCC].

Eq. (rational-negotiation):

> ω*(η) ∈ argmax_{ω∈Ω} U_i(ω)  s.t.  U_{-i}(ω) ≥ η,  P_sat(π^RCC, C^ω) > 1 − ε.

Sweeping η gives "a Pareto frontier of ε-satisfiable contracts". The paper explicitly does **not** model negotiation dynamics ("leaving the analysis of negotiation dynamics to future work").

### 1.7 Repetition theorem (Appendix "Rationality of Contract Compliance under Repetition")

Theorem: in an infinitely repeated episodic performance game with public deterministic monitoring of violations, common discount δ, episode payoffs R_i (mutual compliance), T_i (defect vs complier), P_i (mutual defection) with T_i > R_i > P_i and credible mutual defection, grim trigger sustains compliance as an SPE for δ ≥ δ* = max_i (T_i − R_i)/(T_i − P_i). Standard folk-theorem argument via one-shot deviation principle.

---

## 2. ContractSim instantiation: players, types, timing, dynamics, observation, payoffs

### 2.1 Players and types

> ContractSim instantiates our contracting formalism as G' = (I', Θ', Ω', N', P'), where I'={Cust, Supp} indexes the Customer and Supplier ... Here θ_Cust specifies the Customer's budget, minimum delivery requirements, consumption utilities, and substitution rule, while θ_Supp specifies the Supplier's production function, initial resources, and private information about supply uncertainty, including prices, delivery losses, and spoilage.

**Exactly two players**: one Customer, one Supplier. There is no multi-supplier or market setting ("contracting in an open-market" is listed as future work).

Appendix "Domain-agnostic environment specification": products D={O_1,O_2,O_3}, inputs X={I_1,I_2,I_3}; shared recipes

> O_1 = I_1 + I_2 + I_3,  O_2 = I_1 + I_2,  O_3 = I_3.

Catering mapping: I_1=Mozzarella, I_2=Basil, I_3=Tomato; O_1=Margherita Pizza, O_2=Pesto Pasta, O_3=Tomato Soup. Customer minimums: at least 1 of each product per production week ("mandatory and cannot be substituted"). Above the minimum, Customer-side substitution rate "1 Pizza = 2 Pastas = 4 Soups". Supplier starts with empty inventory, 10-unit inventory cap per ingredient, 12-unit per-round order cap ("chosen to keep the rational baselines computationally tractable").

### 2.2 Horizon and timing

> an accepted contract is executed over the course of L=11 weeks with alternating payment and production periods.

Appendix: "All environments last for L=11 weeks in the performance phase, with (L+1)/2=6 payment weeks W_pay and (L−1)/2=5 production weeks W_prod: W_pay := {2k−1}_{k=1}^{6} (weeks 1,3,5,7,9,11), W_prod = {2k}_{k=1}^{5} (weeks 2,4,6,8,10)."

Week 11 is a terminal payment week with no production after it.

### 2.3 What a "production week" is (dynamics)

Main text:

> In each payment week w the Customer chooses how much to pay M_w^paid. In each production week, spoilage is applied, prices are realized, the Supplier orders inputs, delivery losses determine receipts, and the Supplier chooses a feasible production plan. ... Completed products are delivered immediately, unused inventory carries forward, and no renegotiation or post-contract communication is allowed after agreement.

Agent prompt (Caterer, Env 5), "PRODUCTION WEEK PROCESS (in order)":

> (1) inventory spoilage occurs, (2) you order ingredients (delivery loss applies), (3) you produce dishes and deliver them, (4) prices are re-randomized for next production round.

Appendix stochastic laws, per production week w and input x:

> p_{w,x} ~ P_x,  r_{w,x} | o_{w,x} ~ R_x(·|o_{w,x}),  s_{w,x} ~ S_x,
> where p_{w,x} and o_{w,x} are respectively the realized unit price and ordered amount, r_{w,x} is the amount received, and s_{w,x} is the spoilage shock. The realized spoilage loss is the smaller of s_{w,x} and the inventory available immediately before spoilage. Draws are independent across inputs and production weeks; environments without stochasticity use point-mass laws. The main performance experiments use environment seed 42 for these stochastic draws.

Further rules from the prompts and appendix:
- "You pay for the full amount ordered, not the amount received."
- Inventory cap 10/ingredient: "Any received units that would exceed this cap are lost --- you still pay for them. This cap applies before production, so you cannot order beyond capacity to use the surplus in the same round. Maximum order: 12 units of each ingredient per production round."
- Orders must be affordable from the current balance (cash constraint): in the cMDP the Supplier "chooses an order o_t ∈ {0,…,12}^3 whose cost does not exceed the balance after the current scheduled payment."
- Spoilage "happens before you order new ingredients" and only applies to inventory carried over from the previous round.
- Production plan: any output vector feasible from post-receipt inventory (nonnegative integers); produced goods are delivered immediately in the same week.
- Customer payments: nonnegative whole dollars (×100 / ×1000 granularity in other settings), including $0, cannot exceed remaining budget. "There is no refund mechanism in this system."

### 2.4 Observation asymmetry

Main text:

> Both parties observe the contract, realized payments, and delivered products, but only the Supplier observes its inventory, current prices, spoilage outcomes, and delivered inputs.

Appendix performance scaffolding:

> In each payment week, the Customer receives the week number, total paid, remaining budget, and a payment request. In each production week, the Supplier first observes spoilage and receives its current balance, inventory, and realized input prices in a purchase request. After stochastic delivery, it receives its updated inventory in a production request. Outcome observations record payments, deliveries, inventory changes, and cumulative production for use in later decisions.

So the Supplier sees **this round's realized prices before ordering** (prices are observed, then the order is placed). The Customer's private values v_d and budget B are private; the Supplier's capital, price/loss/spoilage distributions are private (the negotiation prompts say the Customer gets "PRIVATE INFO" and both private prompts are hidden from the counterparty). During negotiation: "Dialogue and formal proposals are visible to both parties, whereas each agent's private prompt, private information, and reasoning traces are hidden from the counterparty."

### 2.5 Payoff functions (verbatim)

> **Customer Utility.** Let B denote the Customer's initial budget, v_d its private value for product d, and q'_{w,d} the quantity of d delivered in week w. The Customer's accumulated private value is
> U_Cust(τ) = B + Σ_{w∈W_prod} Σ_{d∈D} v_d q'_{w,d} − Σ_{w∈W_pay} M_w^paid.

> **Supplier Utility.** Let o_{w,x} be the amount of ingredient x ordered in production period w, and let p_{w,x} be its realized price. The Supplier's profit is therefore
> U_Supp(τ) = Σ_{w∈W_pay} M_w^paid − Σ_{w∈W_prod} Σ_{x∈X} o_{w,x} p_{w,x}.

Disagreement payoffs (appendix): u_Cust(⊥) = B and u_Supp(⊥) = 0. The Customer's prompted BATNA is: "If no contract is agreed upon, you are guaranteed to receive your minimum required dishes in every production round from an outside provider. You pay a total amount exactly equal to the cumulative private value you assign to all of those dishes" (net utility B). Supplier's BATNA: walks away with initial capital (profit 0). Note the Supplier's initial capital ($20/$40) is a liquidity constraint, not part of profit. Note also that the Customer's value counts **all** delivered units at v_d (the "substitution rule" matters for contract compliance accounting, not for the Customer's utility).

Per-week decomposition used for compliant utility (appendix): Δu_Cust,j,w = Σ_d v_d q'_{j,w+1,d} − M^paid_{j,w} for w∈W_pay (payment week paired with the following production week); Δu_Supp,j,w = M^paid_{j,w−1} + 1[w=L−1]·M^paid_{j,L} − Σ_x o_{j,w,x} p_{j,w,x} for w∈W_prod (final week-11 payment grouped with week 10).

---

## 3. Contract schema

### 3.1 Form: free natural language, then translated to a closed structured schema

Contracts are **natural-language text** (a string in the `contract` JSON field of a negotiation turn). They are *not* structured at negotiation time. For evaluation and for the rational baselines, each contract is translated offline by Gemini 3.6 Flash into

> C^ω = (p^d, q, M, κ), where p^d=(p_d)_{d∈D} are product prices, q=(q_{w,d}) is the delivery schedule of integer product counts q_{w,d}∈ℤ_{≥0}, M is the payment schedule, and κ specifies the presence of four supported contingencies: substitution, payment deduction, rollover, and grim trigger.

Translation JSON schema (appendix Table "Structured schema"; exact prompt reproduced in appendix):
- `dish_prices`: {A,B,C} numeric or null (A=Pizza, B=Pasta, C=Soup).
- `production_schedule`: exactly one row per week 2,4,6,8,10 with nonneg integer A,B,C or null.
- `payment_schedule`: exactly one row per week 1,3,5,7,9,11 with amount or null.
- `contingency_set`: subset of {"substitution","grim_trigger","payment_deduction","rollover"}; only clauses **explicitly** in the text.
- `contingency_params`: `substitution.min_qty{A,B,C}` (default 1,1,1), `payment_deduction.min_qty{A,B,C}`, `rollover.max_deficit` (default 2; clamped to 0..2).

"Parse-complete" = all 3 prices + all 5 quantity vectors + all 6 payments present. Parse-incomplete contracts get "no RC, RE, RCC, utility, or satisfaction estimate." A separate deterministic validator checks arithmetic consistency (unit prices × quantities vs payment total vs any stated total), payment total ≤ budget, minimum quantities, inventory/resource feasibility; inconsistencies are "retained as contract outcomes rather than silently repaired."

**Closed clause vocabulary.** "Text expressing another remedy remains in the retained raw agreement and audit response, but it is not added to κ, enforced by C^ω, credited in clause counts, or passed to the solver." Payment-deduction amounts are "derived deterministically from the agreed dish prices rather than being treated as independently translated terms."

### 3.2 Semantics of the four contingencies

Main text: "grim trigger terminates engagement after a violation, payment deduction reduces the next payment after a shortfall, rollover carries a shortfall forward, and substitution permits delivery of value-equivalent products."

Appendix cMDP memory μ_t:
- Substitution: "enforcing each named minimum exactly and crediting above-minimum output at the 4:2:1 substitution rate." No memory.
- Payment deduction: μ_t∈{0,1} = prior round had a shortfall; "two consecutive shortfall rounds constitute a contract violation." Customer may deduct the unit-price value of the shortfall from the next scheduled payment, floored at $0; excess forfeited.
- Rollover: μ_t = per-product carried deficit vector; "A deficit above the cap, or failure to cure a carried deficit in the following production round, constitutes a contract violation."
- Grim trigger: "a violation by a party entitles the counterparty to cease all further performance immediately and permanently." The RCC solver always needs a post-breach response, so when the text has no such clause, "preprocessing supplies grim trigger solely as an implicit solver fallback"; explicit vs implicit contingencies are stored separately and only explicit ones count in statistics.

Example fully-rendered synthetic contract (Env 5, three elective clauses) is reproduced verbatim in the appendix: Parties; Term (weeks 1–11); Unit prices (Pizza $11, Pasta $6, Soup $3); Delivery (each of weeks 2,4,6,8,10: 2 Pizza, 2 Pasta, 1 Soup); Payments (W1 $34; W3 $99; W5 $18; W7 $10; W9 $24; W11 $0); Weekly minimums; Substitution; Rollover; Payment deduction; Termination.

### 3.3 Constraint semantics: role-specific absorbing prefix constraints

> it is possible to decompose the contract constraint C^ω into role-specific prefix indicators C^ω_{i,t}(τ_{≤t})∈{0,1} ... The indicator is active while role i has met every obligation due through week t. ... C^ω_{i,t}(τ_{≤t})=0 ⟹ C^ω_{i,t'}(τ̃_{≤t'})=0 ∀t'>t ... The joint terminal constraint is C^ω(τ)=∏_{i∈I'} C^ω_{i,L}(τ_{≤L}).

Customer compliance = paying the contractually due amount each payment week (after any capped deduction). Supplier compliance = delivering scheduled quantities (after substitution / deduction / rollover rules). ε = 0.05 (P_sat threshold 0.95) throughout.

### 3.4 Enforcement

There is **no third-party enforcement, no penalty payments beyond payment deduction, no escrow, no refunds, no renegotiation**. Enforcement is purely by counterparty conditional compliance (RCC / grim trigger) and by the evaluation metrics. Agents can and do violate contracts; the environment does not prevent it.

---

## 4. Negotiation protocol

Main text:

> In N', the Customer and Supplier exchange structured natural-language proposals for at most 50 rounds. A typical contract ω∈Ω' might specify per-product prices, a delivery schedule over production weeks, a payment schedule over payment weeks, and several contingency clauses. Agreement is reached only when the Customer explicitly accepts the Caterer's most recent formal contract; otherwise, both agents walk away with their initial capital.

Appendix "Negotiation Scaffolding" (the actual protocol α):

> Negotiation lasts for at most 50 rounds, with a Customer turn followed by a Supplier turn. The Customer's opening request asks it to begin the negotiation. Later requests identify the current round, the counterparty's latest message, and, when present, the most recent formal contract together with its proposer and proposal round. The request then lists the actions available in that state: continue the dialogue, propose or revise a contract, terminate, or, for the Customer only, accept an eligible Supplier proposal. A proposal replaces the previously active proposal. Acceptance succeeds only when the Customer accepts the Supplier's formal proposal from the immediately preceding round; proposing and accepting in the same response is not permitted. Either party may terminate immediately, and reaching the round limit without a valid acceptance yields disagreement.

Turn output format:

> {"response": "<dialogue>", "contract": "<contract>"}
> {"response": "<dialogue>", "accept": true}
> {"response": "<dialogue>", "terminate": true}

"Malformed negotiation outputs are conservatively recovered when possible; otherwise, they are treated as ordinary dialogue and cannot trigger a proposal, acceptance, or termination. Empty responses are retried once, after which the run is marked invalid."

Key asymmetries: **only the Customer can accept; only Supplier proposals are acceptable**. The Supplier's prompt: "You cannot formally accept a contract. If you agree with terms proposed by the customer, propose the complete agreed contract for the customer to formally accept on their next turn." There is no explicit "reject" or "counter" action: a new proposal simply replaces the active proposal; dialogue-only turns are allowed. The prompts require every proposed/accepted contract to "explicitly state the price per unit for each dish, the exact quantity of each dish per production round, and the exact payment amount for each payment week", with whole-number quantities and whole-dollar payments, and say "Only propose a contract once you believe you have gathered sufficient information".

---

## 5. Environments and settings studied

### 5.1 Three settings (stories), structurally identical

> We study three supplier settings that are structurally equivalent but have different linguistic descriptions: Catering, Hotel Cleaning, and AI Hosting.

| Entity | Catering | Hotel Cleaning | AI Hosting |
|---|---|---|---|
| Customer role | Customer | Hotel owner | AI startup founder |
| Supplier role | Caterer | Cleaning provider | AI service provider |
| I_1, I_2, I_3 | Mozzarella, Basil, Tomato | Detergent, Disinfectant, Wet-wipes | CPU, Memory, GPU |
| O_1 (=I1+I2+I3) | Margherita Pizza | Conference Room Cleaning | Fine-tuning |
| O_2 (=I1+I2) | Pesto Pasta | King-Sized Room Cleaning | Model-hosting |
| O_3 (=I3) | Tomato Soup | Single Room Cleaning | Inference |
| Money multiplier / granularity | 1× / $1 | 100× / $100 | 1,000× / $1,000 |

"A setting (i.e. story) instantiation only relabels the abstract players, inputs, and outputs and rescales the monetary unit; it does not change the underlying production graph or transition dynamics." Hotel Cleaning and AI Hosting were run only for the matched Environment 5 robustness experiment.

### 5.2 Six Catering environments (appendix Table "catering-environments")

Varied: Customer values (v_Pizza/v_Pasta/v_Soup), Supplier capital, and the price/receipt/spoilage laws. Fixed: budget B=$200, minimums 1/1/1 per production week, substitution 1:2:4, recipes, caps 10/12.

| Env | Stochasticity | Values (Pi/Pa/S) | Capital | Ingredient prices (M/B/T) | Receipt for order x | Spoilage per ingredient |
|---|---|---|---|---|---|---|
| 1 | none | 12/6/3 | $20 | M $1; B $1; T $2 (fixed) | exactly x | exactly 0 |
| 2 | none | 20/10/5 | $40 | same | exactly x | exactly 0 |
| 3 | low | 12/6/3 | $20 | M $1 (2/3), $3 (1/3); B $1 (2/3), $2 (1/3); T $2 (2/3), $3 (1/3) | Uniform{⌊3x/4⌋,…,x} | Uniform{0,1} |
| 4 | low | 20/10/5 | $40 | same | same | same |
| 5 | high | 12/6/3 | $20 | M $1/$3, B $1/$2, T $2/$3 each 50/50 | Uniform{⌊x/2⌋,…,x} | Uniform{0,1,2} |
| 6 | high | 20/10/5 | $40 | same | same | same |

(The appendix notes the prompts call Env 5 "canonical Environment 9" in the authors' internal numbering; figure filenames env-01/04/05/08/12 likewise reflect internal IDs.)

### 5.3 Experimental grids
- **Negotiation:** six Catering envs × ordered 3×3 model pairs × 3 repeats = 162 negotiations; 159 agreements (98.1%). 1,403 proposals total in the corpus.
- **Performance:** 180 synthetic contracts (30 per env: 5 P_sat bands {0.9,0.8,0.7,0.6,0.5}±0.05 × 6 contingency variants) × {Customer,Supplier} roles × 3 LLMs × 3 baseline counterparties (RC, RE, RCC) ... "Altogether this gives 4,860 performance games." Synthetic contracts maximize U_Cust+U_Supp subject to individual rationality and the P_sat band, via multi-start coordinate ascent over 3 quantities, 3 prices, 6 payments; two of eight contingency combinations (those involving payment deduction without rollover, per the text "payment deduction and payment deduction combined") were dropped because they "rendered 46 contracts mutually unbeneficial". Envs 1–2 reuse terms generated for Envs 3–4.
- **Ablations:** all LLM experiments rerun with one added instruction (negotiation: Contingencies / Net utility / No worse prior; performance: Prohibition / Institutional incentives / Reciprocity / Planning). Exact clause texts are in the appendix.
- **Cross-setting robustness:** Env 5 ported to Hotel Cleaning and AI Hosting.

---

## 6. Metrics and baselines (precise definitions)

### 6.1 Baseline implementations (appendix "Rational Baselines")

- **RC Customer:** fully determined by the parsed payment schedule: M^RC_w = min{B^rem_w, M_w − min{δ_w, M_w}}, where δ_w is any contract-authorized deduction. "RC applies contract-authorized deductions but never stops paying merely because the Supplier violated."
- **RC Supplier:** a finite-horizon **constrained MDP** solved by exact backward dynamic programming over state s_t=(n_t, β_t, μ_t) (inventory vector, integer cash balance, contingency memory), price scenario p_t observed before ordering, action = order o_t∈{0..12}^3 affordable. Production is **not** optimized jointly: a "deterministic, solver-aligned production rule ... first prefers full fulfillment (or, under rollover, any delivery that avoids rollover failure), then maximizes credited service using deduction prices when applicable and fixed whole-number weights 4, 2, and 1 for O_1, O_2, O_3 otherwise, and finally minimizes total ingredient use." Bellman recursions compute both continuation satisfaction probability P̂_t(s,p,o) and value V̂_t(s,p,o) (Eqs rc-bellman-psat/value). Selection is lexicographic: among orders with P̂_t ≥ 0.95 pick max value; if none, pick the max-P̂_t order (not the unconstrained profit maximizer). Deterministic policies only ("Our implementation does not optimize over such mixtures"). Output is a lookup table indexed by (t, n_t, β_t, μ_t, p_t); P_sat of the contract = P_0(s_0).
- **RE (approximation):** universal non-engagement: Customer pays M_w=0; Supplier orders and produces 0. The paper is explicit that this is "a fixed adversarial counterparty, not the exact unconstrained best response for every contingent contract."
- **RCC:** D_{i,t} = 1[∃k<t: C^ω_{-i,k}(τ_{≤k})=0]; follow RC while D=0, switch permanently to the non-engagement RE policy when D=1.
- **Translation:** Gemini 3.6 Flash, batch JSON, up to 3 corrective retries, batch rejected on any id mismatch; cached by SHA-256 of contract text + schema + model.

### 6.2 Negotiation / contract-quality metrics (appendix "Negotiation Metric Definitions")

- Expected utility U_i(ω) = E_τ[u_i(τ;θ_i) | ω, π^RCC]; **gain** G_i(ω) = U_i(ω) − u_i(⊥), with u_Cust(⊥)=B, u_Supp(⊥)=0.
- **Satisfaction** P_sat(π^RCC, C^ω) = E_τ[C^ω(τ) | ω, π^RCC]; threshold 0.95.
- **Mutual benefit** MB(ω) = 1[G_Cust ≥ −1e−9 ∧ G_Supp ≥ −1e−9].
- **Proposal feasibility** Feasibility(ω^(k)) = 1[WF(ω^(k)) ∧ Σ_{w∈W_pay} M_w^(k) ≤ B], WF = parse-complete and arithmetically consistent; PF_{ℓ,i} = proposal-weighted average over all proposals by model ℓ in role i.
- **Acceptance regret** AR_i(h) = 1[∃k<n_h: role_h(k)=−i, Feas(ω^(k))=1, U_i(ω^(k)) > U_i(ω_h)+1e−9] (passed over an earlier feasible counterparty offer that was strictly better for you); averaged over accepted negotiations.
- **Contingency count** CC(ω) = number of explicit elective clauses among {payment deduction, rollover, substitution} (grim trigger excluded).
- **Contract completeness** = 1 − (1/|W_prod|) Σ_t (1 − S_t), where S_t is the probability mass still un-breached after production round t under the RCC policy, computed by an exact forward occupancy push with an absorbing DEAD state (negative balance, uncured shortfall, unaffordable deduction all route to DEAD). "This is one minus the expected fraction of the episode spent in breach of the contract."
- **Pareto frontier** F_1 (non-contingent) and F_2 (contingent) at P_sat ≥ 0.95, by sweeping η in Eq. rational-negotiation; exact enumeration in deterministic envs, sampled nondominated contracts in stochastic envs.

There is **no explicit fairness metric** (no Nash product, no Gini); "efficiency" = proximity to Pareto frontier; "welfare" (U_Cust+U_Supp) is used only to generate synthetic contracts.

### 6.3 Performance metrics (appendix "Performance Metric Definitions")

Per-week compliance indicator c_{i,j,w}∈{0,1} (local, not absorbing; weeks where a contingency reduces the obligation to zero are excluded from denominators).
- **Utility** u_i(τ_j;θ_i); **Compliant utility** u_i^c = u_i(⊥) + Σ_w [c_{i,j,w} Δu_{i,j,w} + (1−c_{i,j,w}) min(Δu_{i,j,w}, 0)] (strips positive gains in violating weeks).
- **Ex-post regret** = u_i(τ_j^RCC) − u_i(τ_j) where τ_j^RCC re-simulates with the focal agent replaced by RCC (same contract, counterparty, seed); **compliant regret** uses u^c on both.
- **Defection** (run-level): Defection_i = fraction of runs with any violation; **Unilateral** = violation in a week before any counterparty violation, over runs with such an opportunity; **Reciprocal** = violation after the counterparty was the strict first defector, over eligible runs.
- **Compliance** (week-level, macro-averaged per run): overall rate; **Conditional** (weeks before counterparty defection); **Exploited** (weeks after counterparty defection; lower is better); **Tit-for-Tat** = fraction of weeks where 1−c_{i,j,w} = D_{i,j,w} (matched response). Micro-averaged (pooled-week) variants reported as sensitivity analysis.

---

## 7. LLMs tested and headline results

**Models:** Claude Opus 5, Gemini 3.6 Flash, GPT-5.6-Sol, "implemented within Concordia ... with high reasoning enabled." Inference config: Opus 5 adaptive thinking, effort high, 16,384 output tokens in negotiation / 8,192 in performance; Gemini 3.6 Flash thinking level high; GPT-5.6-Sol reasoning effort high; no temperature/top-p override. Translator: Gemini 3.6 Flash.

**Negotiation results:**
- 159/162 agreements (98.1%).
- Mean P_sat 90.0%; only 77.4% of contracts meet the 95% threshold; 84.9% mutually beneficial ("LLM agents still accept deals that make them worse off 15.1% of the time"); contract completeness 93.8%; **zero contingency clauses in all negotiated contracts** (unless prompted).
- In high-stochasticity Env 5 only 3/9 pairings are mutually beneficial, 7/9 in Env 6, all pairings in Envs 1–4. In deterministic/low-stochasticity envs many negotiated contracts lie on or near the Pareto frontier.
- Role asymmetry: Customer proposals feasible 96.5–99.3%, Supplier (caterer) proposals 69.0–88.7%; Customer acceptance regret 1.9–11.1%, Supplier 15.7–35.2%.
- Parsing: all 1,403 proposals and all 159 finals parse-complete; 44 arithmetically inconsistent, 160 over budget (5 both) → 1,204 feasible; among finals 4 arithmetic inconsistencies, 0 over budget.

**Performance results (Tables 1–2, mean ± SD over 180 cases):**
- LLM Customers vs RCC Supplier: utility 323.6–326.0 vs RCC 323.4, but compliance 78.9–90.0% vs 98.7%, defection 43.9–83.9% vs 1.7%. GPT-5.6-Sol highest utility, lowest compliance (78.9%), highest defection (83.9% of games, 82.2% unilateral).
- LLM Suppliers vs RCC Customer: Opus 5 profit 62.5, GPT-5.6-Sol 63.5 vs RCC 52.6, with unilateral defection 41.7% / 40.6% (Flash 11.7%, RCC 3.9%); compliant profit (45.0 / 45.6) is *below* RCC's 52.4.
- Vs RE: reciprocal defection 100% for all LLMs in nearly all cells (Gemini 3.6 Flash Customer vs RCC shows 66.7% reciprocal, 60.0% exploited compliance); exploited compliance otherwise 0–10%.
- Non-compliance persists as P_sat rises from 50% to 95% and as contingencies are added ("a failure of disposition not capability").
- Prompt ablations: "Prohibition" and "Institutional incentives" clauses substantially reduce defection; "Contingencies" prompt makes agents add clauses but barely improves completeness/P_sat; other prompts insignificant.
- Cross-setting (Env 5 in Catering / Hotel Cleaning / AI Hosting): behaviour "mostly consistent"; e.g. Opus 5 unilateral defection 38.0 / 48.7 / 59.3%, GPT-5.6-Sol 36.7 / 46.7 / 55.3%, Flash 24.0 / 22.7 / 26.7%.

---

## 8. Harness / code

**No code or data release.** What the paper says about the harness:

1. **Framework:** agents are instantiated "in Concordia [Vezhnevets et al. 2023] using the same scaffold." Static role prompt = concatenation of a *Goal* field and a *Private Info* field (full prompts for all 3 settings × 2 roles × 2 phases reproduced in appendix). Environment parameters substituted into prompt text before a run.
2. **Memory:** "the scaffold maintains an independent, append-only provider conversation for each agent. Each new observation is appended once and remains in the conversation history, leaving a stable prompt prefix that enables provider prompt caching; the five observations most relevant to the current request are additionally recalled."
3. **Performance phase I/O:** fresh LLM agent per synthetic contract game, no negotiation transcript shown; contract text inserted verbatim at `{{NEGOTIATED_CONTRACT}}`. Customer returns one integer `payment`; Supplier returns `buy_{resource}` per input and `produce_{product}` per output. Parser tries full response then first flat JSON object; missing/malformed → corrective feedback, up to 3 attempts, then action defaults to zero; invalid-but-well-formed (over budget, unaffordable, infeasible) → state-specific feedback, up to 3 attempts, then zero. Provider exceptions retried 5× with exponential backoff without changing env state.
4. **Translation pipeline:** separate offline Gemini 3.6 Flash call, batch JSON with contract ids, deterministic normalization, validator, cache keyed on (schema version, record id, SHA-256 of raw text, model, terms), audit trail of raw responses.
5. **Solvers:** exact tabular backward DP for RC Supplier (no Monte Carlo), forward occupancy push for completeness, multi-start coordinate ascent + RCC re-solve for synthetic contract generation and Pareto frontiers.

---

## 9. Where the user's abstract-based summary is wrong or incomplete

The user's summary: "contract schema, negotiation protocol (<=50 rounds), production-week dynamics (spoilage, prices, delivery loss, production plan), observation asymmetry, payoffs, environments, and metrics/baselines", plus the framing "agents (supplier(s), buyer/firm)".

1. **Players.** It is strictly **two** players: one *Customer* and one *Supplier* (Caterer / Cleaning provider / AI service provider). There is no multi-supplier, market, or "firm" variant; "contracting in an open-market" is explicitly future work. Calling the buyer a "firm" is inaccurate for the Catering/Hotel settings (it's a Customer with private per-unit values and a hard budget).
2. **Negotiation protocol is more specific than "≤50 rounds".** Each round = Customer turn then Supplier turn. **Only the Customer can accept, and only the Supplier's formal proposal from the immediately preceding round.** The Supplier can never accept; it must re-propose the Customer's terms for the Customer to accept. There is no explicit reject/counter action: a new proposal replaces the active one, dialogue-only turns are allowed, proposing and accepting in the same turn is forbidden, either side may terminate at any time, and hitting 50 rounds = disagreement. Turn outputs are JSON `{response, contract | accept | terminate}`.
3. **Contract schema.** The contract is free natural-language text in a `contract` field; the *structured* schema exists only in the offline translator: C^ω = (prices p^d for 3 products, delivery schedule q over 5 production weeks, payment schedule M over 6 payment weeks, contingency set κ ⊆ {substitution, grim_trigger, payment_deduction, rollover} with params). Anything else written into the contract is ignored by the evaluation. Prompts require every price, quantity-per-week, and payment-per-week to be explicit integers. The user's summary omits the four-clause closed vocabulary, the "parse-complete" requirement, the default implicit grim-trigger fallback, and that ε is fixed at 0.05.
4. **Timing/horizon.** The user's summary omits the **alternating** structure: L=11 weeks, odd weeks 1,3,5,7,9,11 are payment weeks (6, Customer acts), even weeks 2,4,6,8,10 are production weeks (5, Supplier acts). Week 11 is a terminal payment with no subsequent delivery (so the Customer's last payment is pure trust).
5. **Production-week dynamics.** Order inside the week is (1) spoilage on carried-over inventory, (2) Supplier observes realized prices and places an order (0–12/ingredient, must be affordable from cash), (3) stochastic receipt ⌊x/2⌋..x (high) or ⌊3x/4⌋..x (low) — **you pay for what you ordered, not what you received**, (4) 10-unit inventory cap applied, excess lost but paid for, (5) production from inventory via fixed recipes O1=I1+I2+I3, O2=I1+I2, O3=I3 and immediate delivery, (6) next-round prices redrawn. The user's list has the right ingredients but misses the cash constraint, caps, pay-for-ordered rule, and that prices are observed *before* ordering.
6. **Observation asymmetry** is correct in spirit; the details: both see contract, payments, deliveries; only the Supplier sees inventory, realized prices, spoilage, receipts, its balance; Customer sees its total paid and remaining budget. Private valuations, budget, capital and the stochastic laws are each side's private type (the Supplier's prompt contains the exact distributions; the Customer's does not). During negotiation, dialogue + formal proposals are public; private prompts and reasoning hidden.
7. **Payoffs.** The exact forms are U_Cust = B + Σ v_d q' − Σ M_paid and U_Supp = Σ M_paid − Σ o·p (profit, excludes initial capital). Disagreement payoffs are B and 0; Customer's BATNA is an outside provider at exactly its private value. There is no refund, penalty payment, or deposit mechanism; the only money-side contingency is payment deduction.
8. **Environments.** Not "one supplier vs multiple". Six Catering environments = {none, low, high stochasticity} × {low (12/6/3, $20) vs high (20/10/5, $40) values/capital}; plus Hotel Cleaning and AI Hosting as pure relabelings (×100, ×1000 money) used only for an Env 5 robustness check. Seed 42 for main performance experiments.
9. **Metrics/baselines.** The user's summary is missing: (a) RE is implemented as *non-engagement*, not a true best response; (b) RC Supplier is a cMDP solved by exact DP but with a *fixed heuristic production rule* (4:2:1 weights) and deterministic policies only; (c) contract value U_i(ω) is defined under joint RCC play; (d) negotiation metrics include proposal feasibility, acceptance regret, contingency count, and *completeness* (occupancy-based); (e) performance metrics include compliant utility, ex-post and compliant regret vs a re-simulated RCC counterfactual, and the compliance/defection family (conditional, exploited, TFT, unilateral, reciprocal) with macro and micro averaging; (f) the Pareto frontier is a full-information upper bound computed by sweeping η; (g) there is no fairness metric; (h) performance is evaluated on 180 *synthetic* optimizer-generated contracts against RC/RE/RCC, not on the negotiated contracts.
10. **Things the abstract-level summary could not know but that matter for a spec language:** contracts are translated by an LLM (Gemini 3.6 Flash) and the authors treat the translation as untrusted and closed-vocabulary; the RCC solver's grim-trigger fallback; the compliance indicator is role-specific and *absorbing* for the solver but *local per week* for the metrics; constrained-MDP semantics (P_sat ≥ 0.95 lexicographic before value); no renegotiation or communication during performance.

---

## 10. Design-relevant takeaways for a typed spec language + runtime (memo)

- The paper's formal object is a two-phase game: (cheap-talk negotiation game with protocol α) → (POSG with trajectory utilities). A spec language should let the *contract* be a first-class value that (i) is produced by the negotiation phase and (ii) compiles to a constraint C^ω on trajectories plus a tolerance ε, with role-specific absorbing prefix indicators.
- Rational baselines are *constrained* planners (cMDP via DP), which maps naturally onto memo-style probabilistic programs with explicit conditioning on satisfaction probability; the paper's state (inventory, balance, contingency memory, observed price) is small enough for exact enumeration (caps 10/12 chosen for that reason).
- The "contract translation" layer (NL → structured schema, closed vocabulary, nulls not inferred, deterministic defaults, validator) is a separable component worth typing.
- Metrics are defined over (realized trajectory, counterfactual RCC trajectory, per-week compliance indicators), so the runtime must log per-week obligations/fulfilments and support re-simulation with a swapped policy and fixed seed.

---

## Sources

- arXiv abstract page: https://arxiv.org/abs/2608.10475
- arXiv PDF: https://arxiv.org/pdf/2608.10475 (downloaded to `(local scratch, not committed)`; text at `.../paper.txt`)
- arXiv HTML: https://arxiv.org/html/2608.10475v1
- arXiv TeX source (authoritative for quotes): https://arxiv.org/src/2608.10475 (unpacked to `(local scratch, not committed)`)
- DOI: https://doi.org/10.48550/arXiv.2608.10475
- Third-party review noting no code release: https://pith.science/paper/2608.10475
- Web search (no repository found): query `"ContractSim" "Evaluating Rational Contracting in Natural Language" github code`
