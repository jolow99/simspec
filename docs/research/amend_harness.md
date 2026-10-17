# Harness Card literature review (amendment C)

Date: 2026-10-16. Purpose: ground the "Harness Card" requirement (serialized GAME SPEC + AGENT SCAFFOLD CONFIG saved with every run) in the current agent-evaluation literature and tooling.

Access notes: all six primary sources were read in full text (arXiv HTML for papers 1, 2, 3, 5, 6; Harbor ATIF RFC v1.8 markdown from GitHub plus docs.harborframework.com pages; GitHub READMEs for HAL, Harbor, AgentBeats SDK, AgentCompass). Items I could not access or verify are flagged inline.

---

## 1. "Stop Comparing LLM Agents Without Disclosing the Harness" (arXiv 2605.23950v1, 7 May 2026)

Authors: Yunbei Zhang, Janet Wang, Yingqiang Ge, Weijie Xu, Jihun Hamm, Chandan K. Reddy (Tulane, Rutgers, Virginia Tech). Position paper.

### 1.1 Core thesis

- **Binding Constraint Thesis**: for long-horizon tasks across comparable frontier models, harness-induced variance (HV) is "often comparable to or larger than" model-induced variance (MV). Total variance decomposes exactly into Var_model + Var_harness + Var_interaction (Eq. 1); the paper asserts the last two dominate.
- Control-theoretic framing: the LLM is an open-loop stochastic policy pi_theta(a | c_t), where c_t = phi(s_t) is the **context construction function** implemented by the harness; the harness is the controller that updates state from (s_t, a_t, o_t). Three controller-side reliability quantities: **Stability** (Lyapunov-like non-increasing distance to goal set), **Context Drift** (divergence of context-window embeddings from initial task-relevant state), **Control Lag** (steps between anomaly detection and corrective signal reaching the policy).
- Two valid comparison regimes: **locked-harness protocol** (one specified harness for all models) or **factorial protocol** (vary harness as a controlled factor, report variance components). Either way, "the configuration must be disclosed and treated as part of the experimental condition."

### 1.2 Harness factors identified (what enters every benchmark score)

Section 2.1: "Every step in this pipeline (context construction, tool invocation, parsing, retry, summarization, and stopping) is a harness decision." The paper organises these into the seven-layer **ETCSOVG** taxonomy (Sec. 5):

| Layer | Scope (Sec. 5 wording) |
|---|---|
| Execution | runtime substrate, sandboxing, step and task budgets |
| Tool | tool list, schemas, error contract |
| Context | window cap, compression and retrieval policy, persistent memory |
| Scheduling | agent loop, retry and escalation rules |
| Observability | logged artifacts and traces |
| Verification | validation, self-checking, anomaly detection |
| Governance | permission model, side-effect boundaries, human approval points |

Appendix B, Table 8 enumerates the concrete harness mechanisms they actually varied: context strategy (append-all vs compressed retrieval), context cap (200k vs 32k tokens), history compression (none vs summarize outside recent window of 8), retrieval (none vs BM25 top-5 over older steps), context ordering (chronological vs relevance-then-recency), drift monitoring (KL check every 5 steps, threshold 2.0), tool schema (verbose vs minimal), tool exposure (all vs task-relevant subset), tool error format (raw vs structured), runtime retry (none vs 3-attempt exponential backoff vs + malformed-output retry), retryable error classes (timeout, rate limit, transient network, malformed output), failure escalation, parser behavior (basic parser with one tolerated malformed-output retry vs normalization vs normalization+validation), output validation (disabled / schema-only / full), empty-output handling, self-verification, anomaly detection (repeated action loops, context contradictions), checkpointing (last 10 states), rollback policy (up to 3 checkpoints), observability (minimal / full trajectory / full trace with verification metadata), governance (permissive vs allowlist).

Shared controls they held fixed (Table 8, Table 6): source tasks + seed + fixed task order; runtime (Docker SWE-bench pipeline); step budget (50 steps, 120 s/step timeout); model API settings (provider defaults, max output 4096, 180 s timeout, "no model receives harness-specific API tuning").

Note on decoding: the experiment is "intentionally a harness study rather than a decoding study" and uses provider-default sampling; the ETCSOVG card itself does **not** list sampling/decoding parameters as a layer, though the conclusion says reviewers should ask "what harness was used?" "as routinely as they ask about hyperparameters and decoding settings." For our card, decoding must be added explicitly (see Sec. 7).

### 1.3 Evidence that harness swamps model (effect sizes)

Observational (Sec. 4.1, Table 1):
- SWE-bench Pro, Claude Opus 4.5 fixed: SEAL scaffold 45.9% -> Claude Code 55.4% (+9.5 pp), vs. a 4.9 pp spread across the six leading models under the fixed SEAL scaffold. "The within-model harness range thus exceeds the within-harness model range by roughly twofold."
- Adding one search subagent (WarpGrep) adds +2.1 to 2.2 pp and flips the MiniMax 2.5 vs Claude Opus 4.6 ordering.
- HAL SWE-bench Verified Mini, SWE-Agent vs HAL Generalist: 34 pp for Claude Sonnet 4.5 (68 -> 34), 34 pp for GPT-5 Medium (46 -> 12), ~48 pp for o4-mini.
- Epoch AI monitoring: 11 pp (GPT-5) to 15 pp (Kimi K2 Thinking) scaffold-only variation on SWE-bench Verified.
- Grok 4, SWE-agent -> xAI scaffold: +14-16 pp. Terminal-Bench 2: prompt+middleware+verification +13.7 pp; AHE harness evolution +7.3 pp (69.7 -> 77.0). Vercel: 15 tools -> 2 tools, 80% -> 100%.

Controlled 3x3 factorial (Sec. 4.2, Table 2; GPT-5.4, Kimi K2.6, GLM-5.1 x H_min/H_improved/H_full on a 100-task stratified SWE-bench Verified subset, 2 runs each):
- Mean HV = 18.48 pp^2 vs mean MV = 2.37 pp^2; **HV/MV ratio 7.80x** (8.72x and 6.76x in the two individual runs).
- Harness change moves GLM-5.1 by 13.0 pp and the other two by 8.5 pp; model change within a fixed harness moves scores by only 3.0 / 2.5 / 5.0 pp.
- **6 of 9** model-pair / harness-pair comparisons show ranking reversals (interaction term is large).
- Appendix E mechanism analysis: H_min -> H_improved gain is "reduced control noise" (stricter JSON-only action formatting, smaller tool schemas, compressed context, structured error feedback reduce malformed outputs and repeated exploration); H_improved -> H_full gain comes from closing the verification/recovery loop.

Caveat they state: the ratio is not claimed universal; the regime is long-horizon tasks with comparable frontier models.

### 1.4 The proposed disclosure artifact: the Harness Card (Appendix A, Table 3) -- verbatim

> "Table 3: ETCSOVG disclosure fields. A submission can provide more detail, but omitting any of these fields makes it difficult to separate model effects from harness effects."
>
> | Layer | Minimum disclosure |
> |---|---|
> | Execution | Runtime substrate, sandboxing, filesystem access, network access, task timeout, step timeout, maximum steps, and evaluation entry point. |
> | Tool | Tool list, schema style, tool-selection policy, error format, retryable error classes, and whether tools are model-visible or hidden middleware. |
> | Context | Context cap, ordering policy, summarization/compression policy, retrieval method, persistent memory, and cache policy. |
> | Scheduling | Agent loop, stopping rule, retry policy, escalation behavior, delegation policy, and rollback behavior. |
> | Observability | Logged artifacts, trajectory format, checkpoints, traces, validation logs, and whether failures can be audited after the run. |
> | Verification | Output parser, schema validation, self-checking, test execution, anomaly detection, and final patch validation. |
> | Governance | Permission model, allowlist/denylist rules, side-effect boundaries, secret handling, and human approval points. |

Compact example (Table 4, H_full): Execution "Docker SWE-bench runner; 50 steps; 120s step timeout"; Tool "Task-relevant tools; minimal schema; structured errors"; Context "32k token cap; summarize old steps; BM25 top-5 retrieval"; Scheduling "3-attempt backoff; checkpoint rollback depth 3"; Observability "Full trace logs; 10 retained checkpoints"; Verification "Self-checking; full output validation; anomaly detection"; Governance "Allowlist-oriented tool governance".

The card is described as "an attention checklist rather than a design constraint"; "the goal is to make configuration visible, not to standardize it." No machine-readable schema (JSON/YAML) is provided; it is a prose table.

### 1.5 Protocol and metrics they ask for alongside the card

- Minimal valid factorial design: 2x2 model-by-harness grid "with task order, execution environment, evaluation script, API parameters, and stopping rules held constant."
- Report: HV per model, MV per harness, aggregate HV/MV, count of model-pair ranking reversals, partial eta-squared from two-way ANOVA (with caveats: fixed-effects, small-grid bias, pair with reversal counts).
- Trajectory-level metrics (reporting requirements, not estimated in the paper): **Recovery Rate R(k)** (probability of returning from a detected anomaly to a task-advancing state within k steps; report as a curve), **Context Retention** (fraction of task-relevant files/tests/constraints present in constructed context at each step), **Control Lag tau_c**. Future diagnostics: **Schema Compliance Rate** (fraction of outputs that parse without harness intervention) and **Action Efficiency**.
- Appendix C: a (not yet run) perturbation stress test: shuffle retrieved context chunks; multiplicatively perturb numeric tool outputs; measure output instability.

### 1.6 Answers to the standard questions

- **Standardizes**: a disclosure standard (ETCSOVG Harness Card), a variance-decomposition protocol, three trajectory metrics.
- **Assumes**: single-agent, long-horizon tool-use (coding) with a text/JSON action channel; the "model" is a black-box API endpoint; harness = controller around one policy. No multi-agent games, no typed observation spaces.
- **Reproducibility artifact**: yes, the Harness Card (fields above) plus Table 8-style full spec, plus task subset spec (dataset, split, size, sampling rule, seed, task order) and model API settings (provider, generation settings, timeout).
- **Observation structure / rules / rational baselines**: observation structure appears only as "context construction policy phi" and tool "error format"; nothing on game rules or rational/equilibrium baselines. The Schema Compliance Rate is the closest thing to a parse-validity baseline.

---

## 2. AgentBeats / Agentified Agent Assessment (AAA)

Found: "AgentBeats: Agentifying Agent Assessment for Openness, Standardization, and Reproducibility", arXiv 2606.13608 (v2, 14 Jun 2026; 15 institutions incl. UC Berkeley, EPFL, IBM Research, Mila, ServiceNow, CMU). The ICML 2026 poster (icml.cc/virtual/2026/poster/67210) is titled "Position: Agent Evaluation Should Be Agentified for Openness, Standardization, and Reproducibility" (Xiaoyuan Liu, Tianneng Shi, Wenbo Guo, Dawn Song; OpenReview C4vsSnUJuH). The platform is agentbeats.dev (agentbeats.org redirects there); the agentbeats/agentbeats GitHub SDK is marked "DEPRECATED, New version will be released soon."

### 2.1 What it standardizes

- The **interface**, not the harness: a benchmark is itself an agent (**judge agent**, a.k.a. green agent / assessor) that evaluates **subject agents** (purple agents / assessees), with a **delegator** that kicks off the assessment. All interaction via **A2A** (task management: task messages, progress, multimodal artifacts) and **MCP** (tool access). Integration cost drops from N*M to N+M.
- Terminology on agentbeats.dev: "Green Agent (Judge Agent)" = "Sets tasks, scores results"; "Purple Agent (Subject Agent)" = "Attempts tasks, submits answers". Competition page (rdi.berkeley.edu/agentx-agentbeats): "A green agent (evaluator) defines the tasks, environment, and scoring"; "a purple agent (competitor) is the AI agent under test." The older SDK README used red/blue/green (attacker/defender/orchestrator) for a TensorTrust example. "White agent" is not used on the current site.
- Five-step interaction pattern: (1) evaluation request delegation, (2) task instruction distribution + access authorization (judge launches/configures MCP server), (3) task fulfillment, (4) performance scoring (judge revokes access, inspects responses + final environment state), (5) result reporting "in the requested format".
- Five **operation modes** (Local, Remote, Hosted, Proxy, CI) differing in where agents run, who kicks off, where results are shown. CI mode (GitHub Actions) is their auditable/reproducible mode: all files for local-mode execution committed to a repo; secrets via CI; "results are available in CI logs or committed to a separate branch as artifacts."
- Two ways to internalize evaluation logic in a judge: **programmatic** (hard-coded, reproduces original benchmark precisely) vs **semantic** (natural-language instructions, LLM-as-judge). 78% of competition judge agents contained prompts (avg 241 LOC / 11.6k chars).
- Extensions: agent control plane (restart + stream logs), MCP gateway (fixed endpoint, dynamic tool registration), sandbox manager, agent registry (A2A agent cards), leaderboard with "recommended metric reporting formats", Gym-to-MCP translation for RL envs, **agent governance** (audit requirements, resource budgets, restricted actions, permission policies, human-approval gates as experimental conditions).

### 2.2 Artifacts a run produces

The paper does **not** define a schema. What it does say: the kick-off request "typically includes three components: a prompt describing the evaluation goal, the URLs of subject agents with role assignments (e.g., black and white in a chess match), and a customizable configuration provided by the requester." Judge agents should include "metadata such as sampled data point identifiers and dataset size" to enable aggregation. Agent blueprints = Git URL or Docker image (Dockerfile with LLM key + A2A host as env vars recommended for reproducibility); agent instances = A2A endpoint URL. Results are returned to the delegator or published; in CI mode they are committed files. Scores sync to per-benchmark leaderboards. I could not find a documented result/report JSON format on agentbeats.dev or docs.agentbeats.dev (the docs overview page does not define one; the tutorial was not fetched).

### 2.3 Evidence relevant to harness disclosure

- Sec. 8.2 (Table 5): public records mix harnesses; same model shifts up to ~8-10 pp across harnesses (Gemini 3.1 Pro SWE-Bench Pro 54.2% proprietary vs 46.1% MSA; GPT-5.4 Terminal-Bench 75.1% vendor vs 65.4% third-party Terminus-2).
- Sec. 8.3: each agent run with its default prompt, reasoning setting (xHigh / Medium / High), inference config; cost ~$6k.
- Sec. 8.5 **harness-swapping**: native model-harness pairings win 5 of 6 comparisons by 5.3 pp average ("co-adaptation"); the exception reverses when time budget is lifted (Opus+Claude Code 79.8% vs Opus+Codex 73.6%) -- i.e. the **time budget** is itself a harness factor that can flip rankings. Token usage differs ~2.5x between harnesses for the same model.

### 2.4 Answers

- **Standardizes**: protocol-level interface (A2A + MCP), roles, lifecycle, operation modes; not the agent internals, not a trace format.
- **Assumes**: agents are black-box A2A services; **explicitly supports multi-agent** assessments (adversarial: chess, werewolf, CTF; collaborative) with role assignment by the judge. Tasks must be "self-contained" natural-language A2A messages; tools are MCP (typed JSON schemas), so observations are text/multimodal plus typed tool results.
- **Reproducibility artifact**: no formal card; reproducibility is achieved by Docker/Git blueprints + CI mode + judge determinism. Kick-off request = (goal prompt, subject URLs + roles, config).
- **Observation structure / rules / baselines**: rules live inside the judge agent (programmatic or semantic). Role assignment ("black and white in a chess match") is the only multi-agent-game structure; nothing on rational baselines.

---

## 3. HAL: Holistic Agent Leaderboard (Princeton; arXiv 2510.11977v1, 13 Oct 2025; hal.cs.princeton.edu; github.com/princeton-pli/hal-harness)

Status note: the hal-harness README now says "HAL leaderboard results are no longer being updated through this harness ... the repository is archived for historical reference"; current work is the Reliability Dashboard (hal.cs.princeton.edu/reliability/).

### 3.1 What it standardizes

- A **minimal agent interface**: a module exposing `run(input: dict[str, dict], **kwargs) -> dict[str, str]` mapping task IDs to submissions; agent args via `-A key=value`, benchmark args via `-B`. "No constraints on agent implementation or agent framework."
- **Benchmark contract**: task data, evaluation logic, scoring procedure; HAL setup env separated from agent env for frozen dependencies.
- **Logging via W&B Weave**: client initialised per run with a unique run identifier; a `weave_task_id` attribute on downstream calls groups LLM and tool calls by task; captures "model API calls, tool calls, API parameters, and usage information." Unified cost tracking "regardless of provider"; per-token prices as of a stated date (Sept 24, 2025) and editable. Agents must not spawn processes Weave cannot trace.
- **Cross-model via LiteLLM**; three execution tiers (local / Docker / Azure VMs); `--run_id` + `--continue_run`.
- **Agent naming convention**: `Name (model1, model2)` with exact model versions.
- Traces uploaded to HuggingFace via `hal-upload`, **encrypted** to avoid contamination; `hal-decrypt` for consumers. "We share all agent logs, comprising 2.5B tokens."
- Automated log inspection with Docent (found HF-search shortcuts, credit-card misuse, data leakage in TAU-bench few-shot agent).
- Reliability dashboard (2026): 12 metrics in 4 dimensions -- consistency (outcome: each task run K times, (2p-1)^2; trajectory; resource), predictability (calibration, AUROC, Brier), robustness (fault, structural, prompt paraphrase), safety.

### 3.2 Reproducibility requirements and hurdles (Appendix A3 -- directly relevant to our card)

12 documented hurdles, several of which are card fields in disguise: (2) providers swap weights behind stable endpoint names (DeepSeek R1 -> R1-0528); (3) API changes break scaffolds (`stop_keyword` removed); (4) aggregators (OpenRouter) serve different **quantizations** per call; (5) rate-limit errors cause silent false negatives if agents lack retries; (8) **reasoning-effort settings are not comparable across providers** (LiteLLM "high" = 4,096 tokens; OpenAI undisclosed); (9) parameter formats differ per provider; (10) task specs and scaffolds entangled ("don't guess the answer" in benchmark text); (11) frozen dependencies vs evolving providers; (12) logging-library bugs. Known gap: cache hits not accounted in cost (SWE-Agent).

### 3.3 Answers

- **Standardizes**: runner + agent interface + Weave trace capture + cost accounting + leaderboard upload; scaffold is a declared dimension (model x scaffold x benchmark, 21,730 rollouts).
- **Assumes**: single-agent task-to-submission mapping; text/JSON in, string out; no multi-agent.
- **Reproducibility artifact**: Weave traces (per-call params/usage/cost) + run_id + agent name with model versions + `-A` args; no declarative card. Public traces are the artifact.
- **Observation/rules/baselines**: none on observation structure or rational baselines; reliability dashboard adds outcome/trajectory consistency over K repeats, which is a useful "seed/replicate" requirement.

---

## 4. Harbor (harborframework.com; github.com/harbor-framework/harbor, formerly laude-institute/harbor) and ATIF

Harbor is "the official harness for Terminal-Bench-2.0" from the Terminal-Bench creators; Zenodo DOI 10.5281/zenodo.20953922. The ATIF RFC lives at harbor-framework/harbor/rfcs/0001-trajectory-format.md (maintainer Boxuan Li; RFC changelog v1.8, April 2026; Harbor docs say "Harbor's current format is ATIF-v1.7" and accepts v1.0-v1.7).

### 4.1 How tasks (environment) are specified: `task.toml` (schema_version "1.3")

Sections and key fields: `[task]` name/authors/description/keywords; `[metadata]` arbitrary; `[verifier]` timeout_sec, network_mode, allowed_hosts, env, user, environment_mode shared|separate, nested `[verifier.environment]`; `[agent]` timeout_sec, network_mode, allowed_hosts, user; `[solution]` env; `[environment]` build_timeout_sec, network_mode no-network|public|allowlist, allowed_hosts, docker_image, os, cpus, memory_mb, storage_mb, gpus, gpu_types, tpu{type,topology}, env, `[[environment.mcp_servers]]` {name, transport stdio|sse|streamable-http, url, headers, command, args}, skills_dir, healthcheck{...}, workdir; `artifacts` list {source,destination,exclude}; `source` provenance; `[[steps]]` for multi-step with per-step agent/verifier/min_reward/artifacts and `multi_step_reward_strategy` mean|final. Task directory also holds `instruction.md`, `environment/` (Dockerfile), `solution/`, `tests/`.

### 4.2 How agents/jobs are specified: job config (`config.json` written per job; `harbor job schema` prints JSON Schema)

Job: job_name, jobs_dir, n_attempts, timeout_multiplier (+ agent/verifier/setup/build multipliers), n_concurrent_trials, retry{max_retries, include_exceptions, exclude_exceptions, wait_multiplier, min/max_wait_sec}, environment{type (docker, modal, daytona, e2b, gke, ...), import_path, force_build, delete, cpu/memory_enforcement_policy, override_*, mounts, env, kwargs, extra_allowed_hosts}, verifier{override_timeout_sec, max_timeout_sec, include/exclude_logs, env, import_path, kwargs, disable}, metrics[{type sum|min|max|mean|uv-script, kwargs}], agents[], user_agent (simulated user: user_persona_path, user_prompt_template_path (Jinja2), bridge{kind: acp, prompt_path, kwargs}), datasets[] {path|name, version|ref, registry_url/path, repo@ref, task_names globs, n_tasks}, tasks[] {path|name, git_url, git_commit_id, ref}, artifacts[], extra_instruction_paths, extra_instructions, source_jobs (regrade).

`agents[]` (AgentConfig Pydantic model): name | import_path (`module.path:ClassName`), **model_name** (LiteLLM slug, parsed to provider + model id), model_base_url, model_api_key_name, model_api_format, skip_model_parsing, n_concurrent, concurrency_group, skills[], override_timeout_sec, override_setup_timeout_sec, max_timeout_sec, resume_trajectory, load_trajectory (ATIF or native), extra_allowed_hosts, include/exclude_logs, **kwargs** (integration-specific, e.g. reasoning effort via `--effort`), env, mcp_servers[]. Agent classes declare `MODEL_CONNECTION = ModelConnectionSpec(default_provider, agent_provider_access{api_key_candidate_names, base_url_candidate_names}, supported_api_formats, default_api_format_by_provider)`. Prompt templating for installed agents via a `@with_prompt_template` decorator.

Trial output (`TrialResult`, result.json): id, task_name, trial_name, trial_uri, task_id, source, **task_checksum**, **config: TrialConfig**, agent_info{name, version, model_info{name, provider}}, agent_result (AgentContext incl. n_input/n_cache/n_output tokens, cost), verifier_result, verifier_environment_mode, exception_info{type, message, traceback, occurred_at}, timing (environment_setup, agent_setup, agent_execution, verifier), step_results. Trial directory also holds `agent/trajectory.json` (ATIF), verifier files, `artifacts/manifest.json`, and a "replayable trial lock" shown in the viewer (I could not fetch its model file; 404 on guessed paths). **Error classification** (`error_policy.py`) maps every exception to a cause with three decisions -- retry?, grade?, reward valid? -- e.g. `agent_exhausted_budget` (AgentTimeoutError, ContextWindowExceededError, OutputTokenExceededError): reward valid, no retry, grade; `model_provider_failed` rate-limit/overload: retry, reward invalid. **Regrade** re-runs an updated verifier on recorded artifacts without rerunning the agent.

### 4.3 ATIF trajectory schema (RFC, Section II)

Root: `schema_version` (req), `session_id` (run-scoped, optional since v1.7), `trajectory_id` (document-scoped; required for embedded subagents), `agent` (req), `steps` (req), `notes`, `final_metrics`, `continued_trajectory_ref`, `extra`, `subagent_trajectories[]` (v1.7, each an independently valid ATIF trajectory).

`agent`: `name` (req), `version` (req), `model_name`, `tool_definitions[]` (OpenAI function-calling schema), `extra` ("e.g., prompting strategy, custom parameters").

`final_metrics`: total_prompt_tokens, total_completion_tokens, total_cached_tokens, total_cost_usd, total_steps, extra.

Step: `step_id` (1-based sequential, req), `timestamp` (ISO 8601), `source` system|user|agent (req), `model_name` (per-step override), `reasoning_effort` (string|float), `message` (string or ContentPart[] for multimodal; req, may be empty), `reasoning_content`, `tool_calls[]` {tool_call_id, function_name, arguments (JSON object), extra (timeout, retry count, tool version)}, `observation{results[]}` with ObservationResult {source_call_id, content, subagent_trajectory_ref[] {trajectory_id | trajectory_path, session_id, extra}, extra (retrieval score, doc id)}, `metrics` {prompt_tokens, completion_tokens, cached_tokens, cost_usd, **prompt_token_ids**, **completion_token_ids**, **logprobs** (per completion token), extra (reasoning_tokens, cache_creation_input_tokens)}, `extra`, `llm_call_count` (0 = deterministic non-LLM dispatch; >1 = aggregated metrics), `is_copied_context` (steps copied across a compaction boundary; excluded from SFT).

Context Management Convention (Sec. VII): a system step may carry `extra.context_management = {type: compaction|pruning|injection, boundary: replace|append|truncate}`; normative rule: after a `replace` boundary, the effective context for later steps = boundary observation content + subsequent turns. Validator checks schema, sequential IDs, tool-call references, timestamps, image paths.

### 4.4 Answers

- **Standardizes**: task packaging (task.toml + Dockerfile + tests), job config schema, trial result schema, ATIF trajectory interchange format (with token ids and logprobs), error classification (what counts as a valid reward), regrade.
- **Assumes**: single agent in a terminal/container (the AgentBeats and GAE papers both note Harbor is "CLI only"); tasks are instruction + environment + verifier; `user_agent` adds a simulated user (ACP bridge); subagents supported in ATIF; no symmetric multi-agent games.
- **Reproducibility artifact**: yes -- config.json (full resolved JobConfig incl. model slug, kwargs, env with sensitive values redacted), result.json (task_checksum, agent name/version, timings, exception cause), agent/trajectory.json (ATIF), artifacts manifest, trial lock. `model_name` is a LiteLLM slug; reasoning effort via `--effort`/kwargs. Decoding params (temperature etc.) are not first-class fields -- they live in agent `kwargs`.
- **Observation/rules/baselines**: observation = tool results/stdout; rules = verifier scripts; `oracle` agent (runs the reference solution) is the default agent -- a "gold" baseline, not a rational baseline. ATIF `agent.extra` is where a prompting strategy would go; nothing on game rules.

---

## 5. AgentCompass -- two different systems share the name

### 5a. AgentCompass: A Unified Evaluation Infrastructure for Agent Capabilities (Shanghai AI Lab / OpenCompass; arXiv 2607.13705, 13 Jul 2026; github.com/open-compass/AgentCompass; EMNLP 2026 Demo; PyPI v1.0.0 2026-09-30)

- **Standardizes**: decoupling of **Model, Benchmark, Harness, Environment** via "stable interfaces"; a declarative **RunRequest** composed of `BenchmarkSpec` (task + metrics), `HarnessSpec` ("the agent procedure used to interact with each task": prompt formatting, interaction state, multi-turn tool invocation, provider-specific API handling), `EnvironmentSpec` (execution context), `ModelSpec` ("model endpoint, credentials, inference parameters, and supported API protocols"), and a separate `ExecutionSpec` (concurrency, retries: `max_retries`, `retry_pattern_list`, `attempts.k`). Material protocol: Benchmark compiles `TaskSpec -> PreparedTask` (prompts, tools, media, expected outputs); harness returns `RunResult` (prediction, score, complete trajectory). Benchmark declares a scorer execution mode none|reuse|fresh.
- **Trajectory**: "a comprehensive, versioned trajectory for each task ... reasoning process, issued tool calls, environment feedback, and granular metrics (e.g., token consumption, inference latency, and stop reasons)"; pluggable analyzers classify failures as model-side / environment-side / framework-side and detect truncation, latency spikes, repetitive loops, reward hacking.
- **Reproducibility (Sec. 3.5)**: "Each run is persisted by benchmark and model, storing exact configurations, task-level logs, and aggregated summaries. AgentCompass distinguishes **semantic parameters that alter agent behavior from execution parameters** such as concurrency, so execution-only changes do not invalidate results." Appendix A documents per-benchmark harness versions (Mini-SWE-agent v2.3.0, OpenHands SDK v1.23.0, OpenClaw v2026.5.9), context window (250k), per-turn token limit (80k), max message length, timeout multipliers, judge model ids, dataset commits -- a de facto card.
- **Assumes**: single-agent benchmarks (tool use, web research, science, coding, productivity); 10+ harnesses (Claude Code, Codex, OpenHands, OpenClaw, Mini-SWE-agent, Terminus2, chat wrapper). No multi-agent games, no typed observations beyond tool schemas.
- **Observation/rules/baselines**: none on game rules or rational baselines; "reward hacking" detection is the closest to a rules check.
- Not accessed: field-level definitions of the five specs (the architecture docs page does not list them; the "Runtime Contracts" docs page was not fetched).

### 5b. AgentCompass: Towards Reliable Evaluation of Agentic Workflows in Production (FutureAGI; arXiv 2509.14647v1, Sept 2025)

- A **post-hoc trace analyzer**, not a harness: consumes OpenTelemetry traces, reconstructs the span tree into a numbered hierarchical text, runs a plan-and-execute LLM pipeline over four stages (error identification against a 5-category taxonomy -- Thinking & Response, Safety & Security, Tool & System, Workflow & Task Gaps, Reflection Gaps; thematic clustering; quantitative scoring on dimensions like factual grounding/safety/plan execution; synthesis) plus HDBSCAN cross-trace clustering and episodic/semantic memory. Evaluated on TRAIL (148 traces, 1,987 spans).
- Standardizes nothing about run configuration; assumes OTel-instrumented single workflows; no reproducibility card; no rules/baselines. Relevant only as a consumer of our traces (error taxonomy could seed our failure labels).

---

## 6. "General Agent Evaluation" (IBM Research; arXiv 2602.22953v2, 11 May 2026; ICML 2026 "Agents in the Wild" workshop poster, icml.cc/virtual/2026/67864, OpenReview aRRM5iHXpl; also listed at ICLR 2026)

Bandel, Yehudai, Eden, Sagron, Perlitz, Venezian, Razinkov, Ergas, Shachor-Ifergan, Shlomov, Jacovi, Choshen, Ein-Dor, Katz, Shmueli-Scheuer. Not a position paper: an empirical factorial study plus infrastructure (Exgentic harness, Unified Protocol, Open General Agent Leaderboard at exgentic.ai). The "principles" the search snippet mentioned are not an enumerated list in v2; the stated principles are (a) present the benchmark to the agent "in the most natural way" via its native protocol, and (b) benchmarks must be self-contained (also cited by AgentBeats as "Bandel et al. 2026b").

### 6.1 What it proposes

- **Unified Protocol**: every benchmark instance has three fields -- **task** (what to do), **context** (what to know; the agent may append, store, or retrieve it), **actions** (typed parameters, returning observations of arbitrary types); optionally one action designated the **message action** and one the **final-answer action**. Adaptors preserve agents' native protocols (CLI, tool-calling, MCP, code-gen).
- **Agent architecture** defined as "the full bundle shipped with the agent: scaffold, available tools, memory, schema guards, and any auxiliary components -- everything the agent author commits to except the backbone model." Component taxonomy (App. B.3): execution runtime, tool shortlisting, tool schema guard, communication protocol, memory, planning.
- **Factorial evidence** (5 architectures x 5 LLMs x 6 benchmarks, 100 tasks each, 100-turn cap, provider-default sampling, ~$20K): architecture swings results up to 12 pp within one model; variance explained -- model 27.8%, architecture 0.5%, interaction 5.4% among closed-source models; architectural spread 7-12 pp for closed vs 14-18 pp for open-weight ("generality sinks"). Schema guards and tool shortlisting give cross-model gains (+5.5 pp for ReAct+GPT-5.2). This is a useful **counterweight** to paper 1: with provider-default decoding and short-ish tasks, model dominates aggregate variance, but interaction is 10x the architecture main effect -- same conclusion that the pairing must be disclosed.
- **Reproducibility (App. G)**: Table 26 shared run configuration (LLMs, "LLM sampling: Provider API defaults (temperature, top-p)", "Reasoning mode: provider default", max turns 100, instances, cost); **Table 27 provider-documented default inference settings reported verbatim per model and endpoint** (temperature, top-p, top-k, thinking; "undisclosed"/"unsupported" where applicable); released per-run configuration ("agent version strings, system prompts, tool description templates, benchmark adaptor code"), the exact task_id pool per cell, sessions.csv / runs.csv, and **audit scripts** that recompute every numeric claim with an OK/DRIFT/CONTRADICTED verdict. App. B prints each agent's first-message template verbatim (e.g., "Context: {context} ... Use the designated finish tool(s): {finish_tool_names} ... {task}").
- Limitation they state: each cell run once; "per-cell variability across reruns is not measured."

### 6.2 Answers

- **Standardizes**: a benchmark-side canonical representation (task/context/actions with typed action parameters) and a reproducibility bundle (config table, verbatim provider defaults, prompts, task-id pools, audit scripts).
- **Assumes**: single agent; typed discrete actions (they note continuous/pixel actions would need protocol extensions); no multi-agent.
- **Reproducibility artifact**: not a named card, but App. G is the most complete worked example of the fields a card needs (model + endpoint + decoding defaults verbatim + reasoning mode + turn cap + agent version + prompt templates + tool description templates + task ids).
- **Observation/rules/baselines**: typed actions returning "observations of arbitrary types" is the only structural element; nothing on rules or rational baselines.

---

## 7. Proposed Harness Card for our typed spec language + thin runtime

Design stance derived from the sources: paper 1's ETCSOVG table is the disclosure checklist; Harbor/ATIF show how to make it machine-readable (schema_version, hashes/checksums, resolved config, per-step metrics incl. logprobs, context-management boundaries); GAE App. G shows that decoding defaults must be recorded **verbatim per endpoint** rather than as "default"; HAL App. A3 shows which identity fields drift (endpoint weight swaps, quantization, reasoning-effort semantics); AgentBeats shows multi-agent needs explicit role assignment and that time budgets can flip rankings; AgentCompass 5a shows the semantic-vs-execution parameter split so that concurrency changes do not invalidate a card.

Top-level envelope

- `card_version` (schema version of the card itself), `run_id`, `created_at`, `card_hash` (hash of the canonicalised card minus this field), `code_version` {runtime git commit, dirty flag, spec-language version}, `library_versions` {python, provider SDKs (anthropic/openai/litellm), pydantic, our runtime package, any sampling/parsing libs} -- pinned exact versions (HAL hurdle 11; AgentCompass records harness versions).
- `semantic_vs_execution`: every field below is tagged `semantic` (changes behaviour; changes `card_hash`) or `execution` (concurrency, log verbosity, retry on infra failure; excluded from `card_hash`) -- AgentCompass Sec. 3.5.

### 7.1 GAME SPEC section (what paper 1 calls the "task" and what AgentBeats puts inside the judge agent)

- **Identity**: `game_id`, `game_version`, `spec_hash` (hash of the serialized typed spec), `spec_language_version`.
- **Players & roles**: `n_players`, `roles[]` with role ids and the AgentBeats-style assignment of which scaffold instance plays which role; symmetric/asymmetric flag; who moves when (simultaneous / sequential / turn order rule).
- **Rules**: typed action space per role (name, typed parameters, legality predicate), state transition function reference (code path + hash), terminal condition, horizon / max rounds, payoff function (typed; per-role), tie-break and illegal-action rule (what happens on invalid/unparseable action: default action, resample, forfeit) -- the latter is the game-side half of paper 1's "Verification / output parser" layer.
- **Observation structure**: per-role observation schema (typed fields), information structure (perfect / imperfect; what history each role sees), history window exposed by the game (distinct from the scaffold's context policy), observation rendering id (which serializer turns the typed observation into text; hash of its template) -- paper 1's phi is split here into game-side projection and scaffold-side context construction.
- **Rational baselines**: named reference strategies (e.g., equilibrium / best-response / random / scripted) with code hash and parameters; which baselines were run as comparators in this run; any solver settings (e.g., equilibrium computation tolerance).
- **Task instance sampling**: scenario/parameter distribution, instance ids, `instance_seed`, sampling rule, fixed instance order (paper 1 Table 5: dataset, split, size, sampling rule, seed, order).
- **Environment runtime**: step/time budgets per move and per episode (AgentBeats shows time budget flips rankings), evaluation entry point, any sandbox/network settings if tools touch the outside world.

### 7.2 AGENT SCAFFOLD section (one block per role / per scaffold instance)

Model identity and version
- `provider`, `model_id` (exact dated version string, HAL naming guideline), `endpoint` (API base URL / region / aggregator), `api_format` (chat completions / responses / messages), `served_weights_note` (quantization / routing policy if via aggregator; HAL hurdles 2 and 4), `model_snapshot_date`, `system_fingerprint` if the API returns one, `pricing_table_version` (date) for cost accounting (ATIF stores cost_usd but not prices).

Prompt templates (with hashes)
- `system_prompt` {template_id, sha256, path}, `user_turn_template` {id, sha256}, `role_instruction_template` (per role), `observation_renderer` {id, sha256} (how typed observations become text; reference back to GAME SPEC), `few_shot_examples` {ids, sha256}, `tool/function descriptions` {sha256} (GAE releases "tool description templates"), `prompt_language`, `template_variables` actually filled (resolved values or their hashes).

Context / memory policy (paper 1 Context layer; ATIF context_management)
- `context_cap_tokens`, `history_policy` (full / last-k rounds / summarized / retrieval), `k`, `ordering_policy` (chronological / relevance-then-recency), `summarization` {enabled, model, prompt sha256, trigger}, `retrieval` {method, top_n}, `persistent_memory` {type, scope across episodes y/n}, `cache_policy` (prompt caching on/off, cache_control placement), `cross_episode_state` (does the scaffold carry anything between games -- must be declared).

Decoding parameters (record verbatim per endpoint, GAE Table 27 style)
- `temperature`, `top_p`, `top_k`, `max_output_tokens`, `stop_sequences`, `seed` (API-level seed if supported), `reasoning` {mode on/off, effort level, budget_tokens, provider-specific semantics note -- HAL hurdle 8}, `logprobs_requested` {top_logprobs n}, `response_format` / structured-output schema id, `provider_default_flags` (which of the above were left at provider default and the documented default value at run time).

Action elicitation and parsing / retry policy (paper 1 Tool + Verification + Scheduling layers)
- `elicitation_mode` (free text / JSON schema / function-calling tool / constrained enum / single-token), `action_schema` {sha256 of the JSON schema or tool definition}, `parser` {id, version, normalization rules}, `schema_validation` (none / schema-only / full legality check against game rules), `on_parse_failure` {policy: retry with error feedback / resample / default action / forfeit; max_retries; feedback template sha256}, `on_illegal_action` (same shape), `retryable_api_errors` {classes, backoff schedule, max attempts} tagged `execution`, `malformed_output_retry_counts_as_step` (y/n), `stopping_rule`, metrics to log per call: parse_success, retries_used (-> paper 1 Schema Compliance Rate).

Distribution source (how the per-action distribution is obtained -- novel for our setting, but ATIF's `logprobs`/`completion_token_ids` give the storage format)
- `distribution_source` ∈ {`native_sample` (one sampled action per call; distribution estimated by n_samples), `logprob` (token/option logprobs over an enumerated action set; record option tokenization and normalization/renormalization rule), `verbalized` (model states probabilities; record the elicitation prompt sha256 and the parsing/normalization rule), `hybrid`}.
- `n_samples_per_decision`, `enumeration_order_of_options` (and whether it is shuffled per call -- paper 1 App. C perturbation), `option_label_scheme` (A/B/C vs names), `temperature_used_for_distribution`, `calibration_adjustment` (none / any post-hoc transform), `evidence_stored` (raw logprobs array per ATIF `metrics.logprobs`, verbalized text, all samples).

Seeds and replication
- `rng_seed_scaffold` (any local randomness: sampling among ties, option shuffling), `api_seed` (if supported), `episode_seed`, `n_replicates` (HAL reliability: run K times), `determinism_note` (which components are non-deterministic even with seeds).

Budgets and limits
- `max_llm_calls_per_decision`, `max_tokens_per_episode`, `max_cost_usd_per_episode`, `wall_clock_timeout_per_decision`, `budget_exhaustion_policy` (what action is taken when a budget is hit; whether the episode's reward still counts -- Harbor `agent_exhausted_budget` = reward valid).

Tools (if any beyond the game action)
- `tool_list` with schemas (sha256), model-visible vs hidden middleware, tool error format, tool-selection policy.

Observability (paper 1 Observability layer)
- `trajectory_format` (ATIF version or our own, with schema id), `per_call_logging` (prompt text, completion text, token ids, logprobs, usage, latency, stop_reason), `checkpoints`, `redaction_policy` (secrets/env), `audit_possible_after_run` (y/n).

Governance (lightweight for games, but keep the slot)
- `permission_model`, side-effect boundary (none for pure games), human-in-the-loop points (none / where).

Run-level accounting (filled at end of run, mirrors ATIF final_metrics / Harbor TrialResult)
- totals of prompt/completion/cached tokens, cost_usd, llm_call_count, parse-failure count, retries, wall-clock, exception cause classification (Harbor-style: reward valid? retried?).

### 7.3 Minimal "must-not-omit" subset (per paper 1 Table 3 + GAE App. G + HAL A3)

model_id (dated) + endpoint; reasoning mode/effort; decoding values verbatim (or documented defaults); system/user/observation templates with hashes; context policy; action elicitation mode + schema hash + parse-failure policy; distribution_source + n_samples; seeds; budgets; game spec hash + instance seeds + role assignment; library versions. Omitting any of these makes it impossible to attribute a difference to the game, the model, or the scaffold -- which is exactly paper 1's criterion for its Table 3.

---

## Sources

1. Zhang, Wang, Ge, Xu, Hamm, Reddy. "Stop Comparing LLM Agents Without Disclosing the Harness." arXiv:2605.23950v1 (7 May 2026). https://arxiv.org/html/2605.23950 (full text read; Appendix A Table 3/4, Appendix B Tables 5-9 quoted).
2. "AgentBeats: Agentifying Agent Assessment for Openness, Standardization, and Reproducibility." arXiv:2606.13608v2 (14 Jun 2026). https://arxiv.org/html/2606.13608 (full text read). ICML 2026 poster "Position: Agent Evaluation Should Be Agentified..." https://icml.cc/virtual/2026/poster/67210 (OpenReview C4vsSnUJuH). Platform https://agentbeats.dev (agentbeats.org redirects); competition page https://rdi.berkeley.edu/agentx-agentbeats.html; SDK README https://github.com/agentbeats/agentbeats (marked deprecated); docs overview https://docs.agentbeats.dev/ (tutorial not fetched).
3. Kapoor, Stroebl, Kirgis, Nadgir, Siegel, Wei, Xue, Chen, Chen, Utpala, et al. "Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation." arXiv:2510.11977v1 (13 Oct 2025). https://arxiv.org/html/2510.11977 (full text read incl. App. A2-A4). README https://github.com/princeton-pli/hal-harness (archived notice); agents/README.md; https://hal.cs.princeton.edu/ and /reliability/.
4. Harbor Framework Team. "Harbor" (2026), DOI 10.5281/zenodo.20953922. README https://github.com/harbor-framework/harbor (laude-institute/harbor README fetched from raw.githubusercontent.com/laude-institute/harbor). ATIF RFC https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md (v1.8 changelog). Docs: https://docs.harborframework.com/agents/atif.md, /tasks/configuration.md, /jobs/configs.md, /jobs/run-a-job.md, /jobs/error-classification.md, /jobs/regrade.md, /agents/custom-agents.md, /agents/agent-auth-resolution.md, /results/view-job-results.md; Pydantic models src/harbor/models/trial/result.py and config.py.
5a. Chen, Ding, Ge, et al. "AgentCompass: A Unified Evaluation Infrastructure for Agent Capabilities." arXiv:2607.13705 (13 Jul 2026). https://arxiv.org/html/2607.13705 (full text read); https://github.com/open-compass/AgentCompass README; architecture docs https://agent-compass.mintlify.app/en/developer_guide/architecture.
5b. "AgentCompass: Towards Reliable Evaluation of Agentic Workflows in Production." arXiv:2509.14647v1 (FutureAGI). https://arxiv.org/html/2509.14647 (full text read).
6. Bandel et al. "General Agent Evaluation." arXiv:2602.22953v2 (11 May 2026). https://arxiv.org/html/2602.22953 (full text read incl. App. A, B, G, I). ICML 2026 workshop listing https://icml.cc/virtual/2026/67864 (OpenReview aRRM5iHXpl). Leaderboard www.exgentic.ai (not fetched).

Not found / not accessible: a machine-readable Harness Card schema from paper 1 (prose table only); an AgentBeats result/report JSON schema; the Harbor "trial lock" model file (404 on guessed paths); field-level definitions of AgentCompass' five Spec classes; any "purple/white agent" definition in the AgentBeats paper itself (the colours appear only on the platform/competition pages).

---

## Summary (300 words)

The strongest grounding for a mandatory Harness Card is Zhang et al. (arXiv 2605.23950), which formalises the harness as the controller of a closed loop around an open-loop LLM policy and shows, in a controlled 3x3 model-by-harness grid on SWE-bench Verified, that harness-induced variance exceeds model-induced variance 7.8x (18.5 vs 2.4 pp^2), with harness changes moving a fixed model by 8.5-13 pp versus 2.5-5 pp for model swaps, and six of nine ranking reversals. Public leaderboards show the same pattern (9.5-48 pp scaffold-only swings). Their remedy is a seven-layer ETCSOVG disclosure card (Execution, Tool, Context, Scheduling, Observability, Verification, Governance), quoted verbatim above, plus a variance-decomposition protocol and trajectory metrics (recovery rate, context retention, control lag, schema compliance). The card is prose, single-agent, and silent on decoding, distributions, game rules and baselines.

The other systems supply the missing machinery. Harbor's ATIF gives a versioned JSON trajectory schema with agent name/version/model, tool definitions, per-step token ids, logprobs, cost, reasoning effort, subagent trajectories and explicit context-compaction boundaries; Harbor's job config and TrialResult show resolved-config persistence, task checksums and an error taxonomy deciding when a reward is valid. HAL documents the identity hazards a card must pin (endpoint weight swaps, quantization, non-comparable reasoning effort, frozen dependencies) and standardises cost accounting and K-repeat consistency. General Agent Evaluation (IBM) supplies the best worked reproducibility bundle: provider-default decoding settings recorded verbatim per endpoint, prompt and tool-description templates, agent versions, task-id pools, audit scripts; its factorial also shows the model-architecture interaction (5.4%) dwarfs the architecture main effect (0.5%), reinforcing disclosure of the pairing. AgentBeats contributes the multi-agent framing (judge/green vs subject/purple agents, explicit role assignment, A2A/MCP) and evidence that time budgets flip rankings; AgentCompass contributes the semantic-vs-execution parameter split and Benchmark/Harness/Environment/Model specs.

The proposed card therefore splits into a GAME SPEC (identity hash, roles, typed rules and observation schema, rational baselines, instance seeds, budgets) and an AGENT SCAFFOLD (dated model identity and endpoint, hashed prompt templates, context policy, verbatim decoding, elicitation/parsing/retry policy, distribution source, seeds, budgets, library versions).
