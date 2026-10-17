# Decision-model ("System One") APIs: schemas, ecosystem, and the Large Language Bayes pattern

Research date: 2026-10-16. Purpose: ground the design of a provider-agnostic `DecisionModelAgent` adapter (observation + typed questions / finite action set -> distribution over actions) for LLM multi-agent game simulations.

Verification status legend: **[official]** = read from the vendor's own docs/SDK source; **[secondary]** = press, aggregator, or community; **[unverified]** = could not reach a primary source.

---

## 0. Corrections to the brief (what the user remembered vs. what exists)

| Claim from memory | Finding |
| --- | --- |
| "TypeSafe AI's Jev, launched Sept 2026" | Correct. Early access announced 15 Sep 2026; Jev 1.14 shipped 14 Oct 2026. Still early-access/waitlist via TypeSafe directly, but GA-style access exists through OpenRouter, Vercel AI Gateway, LiteLLM. [official + secondary] |
| "Choice/Score/Boolean question types" | Partly wrong naming. TypeSafe's native primitives are **Choice, Score, Noul** (Noul = yes/no, returns P(yes)) plus a new **Box** (bounding box, Jev 1.14+). "Boolean" is the name the **Vercel AI SDK** gives to the noul/predicate primitive in its provider-agnostic `experimental_decide` API. [official] |
| "Returns probabilities" | Correct for Choice (full distribution over every option) and Score (full distribution over levels); Noul returns a single probability and **no** `confidence`. [official] |
| "OpenAI Decisions API, `/v1/decisions`, public beta Oct 2026" | Correct. `POST https://api.openai.com/v1/decisions`, public beta from 6 Oct 2026 (limited preview announced 29 Sep at DevDay), only model `gpt-6-luna`. Question types are **predicate / choice / score** (OpenAI uses "predicate" where TypeSafe uses "noul"). [official] |
| "Gateways exposing `/v1/decisions`" | Yes: LiteLLM, Vercel AI Gateway, Portkey, agentgateway, llmgateway.io, OpenRouter (`/api/alpha/decisions`, but with the **TypeSafe** schema). Cloudflare **AI Gateway**: no evidence; Cloudflare's decision models (Clef) are on **Workers AI** instead. [official/secondary] |

Important ecosystem fact: there are now **two competing wire formats**. TypeSafe's `/v1/systemone` (state + questions-map + noul/choice/score, answers keyed by question id) has been adopted nearly industry-wide (Perplexity, Microsoft Foundry, Cloudflare Workers AI, Ollama, llama.cpp, vLLM, OpenRouter, Vercel, Upstage per reports). OpenAI's `/v1/decisions` (input + questions-array + predicate/choice/score, answers as an ordered array) is the outlier; gateways translate between the two.

---

## 1. The common conceptual contract (what every decision model shares)

Across vendors the contract is the same and maps cleanly onto our agent interface:

- **Input**: one `state` (string | JSON object | JSON array; some accept images) shared by all questions. Questions are evaluated **independently and in parallel** against the same state (exception: Cloudflare Clef via llama.cpp decides jointly).
- **Questions**: a set of typed questions. Three universal types:
  - **choice**: finite unordered option set (name -> optional description). Returns argmax `choice`, a **full probability distribution over every option** (sums to 1), and a scalar `confidence`.
  - **score**: ordered levels (array, lowest first). Returns per-level distribution, `score` = probability-weighted mean of 0-based level indices (can be fractional), `legend`, `confidence`.
  - **noul / predicate / boolean**: a proposition. Returns one number P(true). No confidence field.
- **Output**: no generated text; `output_tokens` is 0 or a small constant; billed on input tokens only.
- **Confidence** (TypeSafe formula, adopted by most clones): Choice `(p_max - 1/n) / (1 - 1/n)`; Score `max(0, 1 - sum_i p_i |i - m| / MAD_unif)` where m = modal level and `MAD_unif = (1/n) sum_i |i - (n-1)/2|`; Noul has none (use `|2p-1|` if needed). OpenAI does not document its confidence formula.

For our purposes: **choice is exactly "distribution over a finite action set"**. The action set is the `criteria` map; the returned `probabilities` map is the policy. Score gives an ordinal action space; noul gives a Bernoulli action.

---

## 2. TypeSafe AI — Jev (`POST https://api.typesafe.ai/v1/systemone`) [official]

Sources: https://docs.typesafe.ai/api (API reference), /introduction/quickstart, /primitives/choice, /primitives/score, /primitives/noul, /confidence, /models, /concepts/state, https://typesafe.ai/blog/introducing-system-one-models-and-jev.

### 2.1 Endpoint, auth, transport
- `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <API_KEY>`, `Content-Type: application/json`. Keys from https://console.typesafe.ai/keys; playground at https://console.typesafe.ai/playground.
- Body <= 32 MiB, UTF-8 JSON (no BOM), <= 64 nesting levels in `state`/`instructions`/`criteria`, finite numbers only. Optional `Content-Encoding: gzip|zstd` (SDKs zstd-compress bodies >= 1 KiB).
- No streaming.
- Optional danger-zone headers: `TypeSafe-Danger-Zone-Long-Context: true` (raises 64k/32k limits to 262k; slower, less accurate), `TypeSafe-Danger-Zone-Images: true` (image input; PNG/JPEG via multipart; Jev 1.14 for Box).

### 2.2 Request schema

```jsonc
{
  "model": "jev-latest",                 // string, required. jev-1.13.0 | jev-1.14.0 | jev-latest | jev-preview
  "state": <string | object | array>,    // required. Shared content; non-strings are given to the model as JSON
  "questions": {                         // required. map<question_id, Question>; ids are yours, not seen by model
    "<id>": {
      "type": "noul",
      "instructions": <string | object | array>,   // required
      "criteria": { "true": <desc>, "false": <desc> }  // optional
    },
    "<id>": {
      "type": "choice",
      "instructions": <string | object | array>,   // required
      "criteria": { "<option>": <string | object | array | null>, ... }  // required; 1..255 options
    },
    "<id>": {
      "type": "score",
      "instructions": <string | object | array>,   // required
      "criteria": [ <level0 desc>, <level1 desc>, ... ]   // required; ordered low->high; 2..10 levels
    },
    "<id>": {                            // Jev 1.14+, images only, experimental
      "type": "box",
      "instructions": <string | object | array>
    }
  }
}
```

Notes: `instructions` and criteria descriptions may be structured JSON; backtick-quoted field names inside instructions reference fields of `state` (or of the instructions object). Field names like `what`, `not_for`, `examples` are conventions, not reserved. Descriptions `null` are allowed when the option name is self-explanatory. Option and level names are sent to the model (so names matter).

### 2.3 Response schema

```jsonc
{
  "model": "jev-1.13.0",                 // resolved versioned model id
  "answers": {
    "<id>": { "type": "noul", "noul": 0.95 },
    "<id>": {
      "type": "choice",
      "choice": "billing",               // highest-probability option
      "probabilities": { "billing": 0.88, "technical": 0.12, "sales": 0.0 },  // map<option, number>, sums to 1
      "confidence": 0.81                 // 0..1, derived from probabilities
    },
    "<id>": {
      "type": "score",
      "score": 1.05,                     // sum_i i * p_i, may be fractional
      "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
      "probabilities": { "0": 0.0, "1": 0.95, "2": 0.05 },  // string level keys
      "confidence": 0.92
    },
    "<id>": {                            // box answer
      "type": "box",
      "box": { "top": 0.49, "bottom": 0.89, "left": 0.58, "right": 0.96 },  // fractions of image, from top-left
      "confidence": 0.93
    }
  },
  "usage": { "input_tokens": 296, "output_tokens": 20 }
}
```

- **A full distribution is always returned** for choice and score. Values are rounded to **two decimals**, so a distribution may not sum to exactly 1 (AI SDK models this as `rounding.probabilityDecimals = 2`; it validates rather than renormalizes).
- Noul answers carry no `confidence`.
- Errors: 400 (rules/`max_tokens_exceeded`), 401, 413, 415, 422 (validation; body names the field), 429 (rate limit; honor `retry-after`), 529 (overloaded).

### 2.4 Models, limits, pricing, latency, rate limits
- Models: `jev-1.13.0` (= `jev-latest` today), `jev-1.14.0` (available 14 Oct 2026; = `jev-preview`; adds Box + images). Text-only otherwise. English primary; CJK accepted with lower accuracy.
- Context: 64k tokens per request (state + all questions), 32k per question (state + one question); no limit on number of questions as long as both fit. 262k with long-context header.
- Pricing: **$0.042 / 1M input tokens** ("$42 per Btok"), output free. ~$0.0004 per decision in TypeSafe's evals [secondary].
- Latency: TypeSafe quotes 70–500 ms end-to-end; third-party tweet analysis median 76 ms, upper quartile 270 ms [secondary]; added questions barely change latency (parallel evaluation).
- Rate limits: per-organization, in tokens/sec and requests/sec, values shown in console (not in docs). 429 on excess. Higher limits on custom/enterprise plans. (A secondary source claims 250k TPS / 1,200 RPM; treat as unverified.)
- Calibration: trained with "Reinforcement Learning for Calibrated Decisions (RLCD)"; marketing claims "calibrated probabilities" and "higher confidence -> higher accuracy" but **no published calibration metrics**. The confidence page explicitly makes no calibration claim. Known "jaggedness" page for jev-1.13 (counting, arithmetic, date comparison, large noisy state).
- Batching: multiple questions per request only; no async batch API documented.

### 2.5 SDKs
- Python `typesafe-sdk` (3.10+): `TypeSafeClient(model="jev-latest").system_one(state=..., questions={"id": Choice(instructions=..., criteria={...}), ...})`; results via `response.answers["id"].choice` or `response.choices / .scores / .nouls`. Env: `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`. Retries 429/529 with backoff by default.
- JS/TS `@typesafe-ai/sdk` (Node 20+): `new TypeSafeClient({ defaultModel: "jev-latest" }).systemOne({ state, questions })` with typed `ChoiceResponse<T>`, `ScoreResponse<T>`, `NoulResponse`, `SystemOneResult<Q>`; `usage.input_tokens | null`.
- Vercel AI SDK provider `@ai-sdk/typesafe-ai`: `typeSafeAi.decisionModel('jev-latest')` with `experimental_decide` (question types `choice | score | boolean | box`; confidence surfaced at `providerMetadata.typesafe.confidence[questionId]`).
- Agent skill: `claude plugin install typesafe@typesafe-ai`.
- `TYPESAFE_BASE_URL` / `baseURL` lets the same SDK target any `/v1/systemone`-compatible server (OpenRouter, Vercel, Ollama, llama.cpp, Foundry, etc.).

---

## 3. OpenAI — Decisions API (`POST https://api.openai.com/v1/decisions`) [official]

Sources: https://developers.openai.com/api/docs/guides/decisions (+ `.md`), https://developers.openai.com/api/docs/guides/decisions-voice, https://developers.openai.com/api/docs/pricing, OpenAI Python SDK source `src/openai/types/decision.py`, `decision_create_params.py`, `resources/decisions.py` (generated from the OpenAPI spec), community announcement https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877.

Status: public beta since 6 Oct 2026; "GA in the coming weeks". Only `gpt-6-luna`. Playground: https://platform.openai.com/decisions. SDK minimums: Python 3.26.0, JS 7.30.0, Go 3.73.0, Ruby 0.101.0, Java 4.78.0. Not available on Azure/AWS at time of writing (community thread; no official answer).

### 3.1 Request schema (from SDK `DecisionCreateParams`)

```jsonc
{
  "model": "gpt-6-luna",                              // required; only supported value
  "input": <string | [ { "role": "user",             // required; shared evidence for all questions
              "content": <string | [ {"type":"input_text","text":"..."},
                                     {"type":"input_image","image_url":"data:image/png;base64,..."} ] > } ]>,
  "questions": [                                      // required; 1..200 questions; answers come back in same order
    { "type": "predicate", "name": "visible_damage", "instructions": "..." },            // name optional but recommended (unique)
    { "type": "choice", "name": "department", "instructions": "...",
      "choices": [ { "value": "billing", "description": "..." }, { "value": true } ] },  // 2..255 unique; value: string | boolean (typed: "true" != true); description optional
    { "type": "score", "name": "severity", "instructions": "...",
      "levels": [ { "label": "Cosmetic", "description": "..." }, { "label": "Fully blocked" } ] }  // ordered low->high; label required
  ],
  "prompt_cache_key": "<string>",                     // optional
  "prompt_cache_options": { "mode": "implicit|explicit", "ttl": "30m" },  // optional (SDK exposes; community reports caching not yet live for Decisions)
  "safety_identifier": "<opaque end-user id, <=128 chars>" // optional
}
```

Discrepancy to note: the guide says images **must** be inline base64 data URLs (no hosted URLs, no `file_id`), while the SDK docstring says "base64 data URLs or publicly accessible HTTP(S) URLs; at most 128 images". Treat base64 as the safe path. Files, audio, tools, item references unsupported. AI SDK exposes `imageDetail: low|high|auto|original` per image part.

### 3.2 Response schema (from SDK `Decision`)

```jsonc
{
  "model": "gpt-6-luna",
  "answers": [                                           // array, question order; each answer may instead be a refusal
    { "type": "predicate", "name": "visible_damage", "probability": 0.92 },
    { "type": "choice", "name": "department",
      "choice": "billing",                               // string | boolean (one of the supplied values)
      "probabilities": [ { "value": "billing", "probability": 0.95 },
                         { "value": "technical", "probability": 0.02 }, ... ],  // array, one entry per option
      "confidence": 0.93 },
    { "type": "score", "name": "severity",
      "score": 1.1,                                      // probability-weighted mean of 0-based indices
      "probabilities": [ { "value": 0, "label": "Cosmetic", "probability": 0.1 },
                         { "value": 1, "label": "Workaround available", "probability": 0.7 },
                         { "value": 2, "label": "Fully blocked", "probability": 0.2 } ],
      "confidence": 0.55 },
    { "type": "refusal", "name": "<question name or null>" }   // other questions still answered
  ],
  "usage": {
    "input_tokens": 0, "input_tokens_details": { "cached_tokens": 0, "cache_write_tokens": 0 },
    "output_tokens": 0, "output_tokens_details": { "reasoning_tokens": 0 },
    "total_tokens": 0
  }
}
```

- **Distributions are always returned** for choice and score (per-option `probabilities` array + `confidence`); predicate returns only `probability`. No logprobs field; no single-choice-only mode.
- `confidence` formula is **undocumented**.
- No streaming; no batch endpoint documented. Multiple independent questions share one request; dependent questions need separate requests.

### 3.3 Pricing, latency, compliance
- `/v1/decisions` with `gpt-6-luna`: **$0.10 / 1M input tokens**, input-only billing (no cache-read/write or output charges); regional premiums and long-context multipliers apply. (Pricing table: gpt-6-luna standard $0.10 in / $0.01 cached / $0.50 out; flex $0.05; priority $0.20.)
- Latency claim: "about 10x faster than the Responses API"; no absolute figure. Community: ~0.8 s with an image; DevDay demo ~150 ms vs ~1.6 s for a Luna chat call [secondary]; decision-bench reports 326 ms median [secondary].
- ZDR and HIPAA eligible; data residency US and EEA+CH.
- Calibration: no official data. Community coin-flip test: predicate matched a 70% bias; **choice gave 98% to heads and was order-sensitive** — recommend predicates over choice until fixed [secondary, single tester]. decision-bench PR: 90.5% accuracy (969/1071) on its corpus [secondary].

### 3.4 Realtime/voice integration (GPT-Live) [official]
`session.delegation = { "type": "decisions", "decisions": { "model": "gpt-6-luna", "questions": [ choice questions ] } }` and `session.triggers[]` of `type: "decisions"` fire on transcript deltas; results arrive as `decision.event` -> `event.decision.answers[]` (choice answers carry only `choice` in the trigger example). Context is added via `decision.item.create`. Relevant if a simulation drives decisions off streaming observations.

### 3.5 Vercel AI SDK mapping [official]
`openai.decisionModel('gpt-6-luna')`: boolean -> predicate (criteria appended to instructions as "Criteria for true/false"); choice criteria keys -> `choices[].value`, values -> `description`; score criteria -> `levels[]` with zero-based index labels. Native confidence at `providerMetadata.openai.confidence[questionId]`; refusal -> `InvalidResponseDataError` for the whole call. Reasoning options rejected.

---

## 4. Gateways, proxies, routers

| Gateway | Decision support | Wire format(s) | Notes |
| --- | --- | --- | --- |
| **LiteLLM** (proxy + SDK) [official] https://docs.litellm.ai/docs/decisions | `POST /v1/decisions` and `POST /v1/systemone`; SDK `litellm.decisions/adecisions`, `litellm.systemone/asystemone`, `router.*` | Both; translates either way, replies in the format of the route called | Providers: `openai/gpt-6-luna`, `typesafe/jev-latest`, `perplexity/pplx-decider-v1-27b`, `cloudflare/clef`, `cloudflare/clef-flash`, `fireworks_ai/.../rune`, `databricks/databricks-openjev-qwen35-4b`, `databricks/ai_decide`, `azure_ai/<Decision-1 deployment>`, `strands_decider/...`, `hosted_vllm/<model>` (choice only), `laya/*`, `bespoke/nimble-*`. 1–128 questions/request; no streaming; OpenAI requires >=2 choices/levels. v1.104.2+. Field map: input<->state, questions list<->map (name = key), predicate<->noul, choices[{value,description}]<->criteria{label:desc}, levels[{label,description}]<->criteria[], answers list<->map, probabilities list<->map; refusals returned as `{"type":"refusal"}` on `/v1/decisions`. |
| **OpenRouter** [official] https://openrouter.ai/docs/guides/community/jev | `POST https://openrouter.ai/api/alpha/decisions` and `POST https://openrouter.ai/api/v1/systemone` (TypeSafe SDK with `base_url=https://openrouter.ai/api`) | **TypeSafe** schema on both (state, questions map, noul/choice/score/box); extras: `provider` (routing prefs), `session_id`, `trace`, `user`; response adds `id`, `provider`, `usage.cost` | Models `typesafe/jev-1.14` (alias `~typesafe/jev-latest`), `typesafe/jev-1.13`; also `perplexity/pplx-decider-v1.1-27b`; Kev-4B listed by systemonemodels.org. $0.042/M in, output $0; 262,144 ctx; >~32 kB text or any image runs in TypeSafe long-context mode. OpenRouter SDKs: `alpha.decisions.create` (TS/Python/Go). Does **not** expose OpenAI's `/v1/decisions` shape. |
| **Vercel AI Gateway** [official] https://vercel.com/docs/ai-gateway/modalities/decision | Three HTTP surfaces: native `POST https://ai-gateway.vercel.sh/v1/evaluate` (model/state/questions with `boolean|choice|score`, camelCase usage, `providerMetadata.gateway` cost/routing); **OpenAI-compatible** `https://ai-gateway.vercel.sh/v1` + `decisions.create` (predicate/choice/score, OpenAI response shape, any decision model answering); **TypeSafe-compatible** `https://ai-gateway.vercel.sh/typesafe/v1/systemone` (+ `/typesafe/v1/models`) | All three | Models: `typesafe-ai/jev` ($0.042/M, 64k/32k ctx), Laya, Liquid `d1`/`d1-ultra`, Microsoft Decision-1, OpenAI Decisions. "Decision fallbacks": `providerOptions.gateway.models: [{ model, when: { question, confidenceBelow } }]` reruns with another model; when an LLM answers, `confidence: 0` and `probabilities: {}` are sentinels for "unavailable", not zero confidence. BYOK, ZDR per request. Not available on Chat Completions/Responses endpoints. |
| **Portkey** [secondary: changelog] https://portkey.ai/docs/changelog/enterprise | `/v1/decisions` since v2.25.0 (Jev schema); v2.29.0 auto-detects OpenAI-native vs Jev schema and replies in the same shape; routes via OpenRouter (v2.27.0) and Cloudflare Workers AI clef | Both (auto-detected) | Could not open a dedicated Portkey decisions doc page; details from changelog summaries. |
| **agentgateway** (open source) [secondary: vendor blog] | Clients call `/v1/decisions` (OpenAI shape); translates to Jev `/v1/systemone` when model is TypeSafe | OpenAI in, both out | On `main`/nightly only at time of post (12 Oct 2026). |
| **llmgateway.io** [secondary] | OpenAI-compatible `/v1/decisions` with `gpt-6-luna-decisions` | OpenAI | PR theopenco/llmgateway#4607. |
| **Kong AI Gateway** [unverified] | Mentioned alongside Portkey as supporting Jev decisions in a third-party write-up | ? | No Kong doc found. |
| **Cloudflare AI Gateway** | **No evidence** of `/v1/decisions` or `/v1/systemone` support. Cloudflare's decision models (Clef) are served by **Workers AI** (see 5.2). | — | — |
| **Ollama / llama.cpp / vLLM** | Local `/v1/systemone` servers (see 5.5–5.7) | TypeSafe | Point `TYPESAFE_BASE_URL` at them. |

Design implication: a `DecisionModelAgent` adapter needs two serializers (TypeSafe-style and OpenAI-style) and one normalized internal answer type; everything else is base-URL + auth. LiteLLM's field-mapping table above is a good normalization reference.

---

## 5. Other decision-model vendors and open-source equivalents

### 5.1 Perplexity — `pplx-decider` [official] https://docs.perplexity.ai/docs/decisions/quickstart
- `POST https://api.perplexity.ai/v1/decisions` (note: **TypeSafe-style body despite the path**): `model` (`pplx-decider-v1.1-27b` | `pplx-decider-v1-27b`), `state` (string|object|array; may include OpenAI-style `image_url` parts with base64 data URLs; PNG/JPEG/WebP), `questions` map (1–128). Types `noul` (instructions and/or criteria{true,false}), `choice` (criteria map, 1–255), `score` (criteria array, 1–10 levels). Response `{model, answers{...}, usage{input_tokens, output_tokens}}` with the same answer shapes as TypeSafe. Unknown top-level fields -> 400.
- Limits: <262,144 input tokens; 32 MiB body; 10 RPS/org with `x-ratelimit-*` headers and `Retry-After`. Pricing **$0.02 / 1M input**, output free. Latency: <2 s small inputs; ~5 s at 90k tokens; ~23 s near limit. Weights Apache 2.0 (self-host ~49 GiB, SGLang cookbook exists). Reported Decision Index 61.56 vs Jev ~58 [secondary].

### 5.2 Cloudflare — Clef / Clef-flash / Clef-omni on Workers AI [official] https://developers.cloudflare.com/workers-ai/models/clef/
- `POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef` (or `env.AI.run("@cf/cloudflare/clef", {...})`). Body: `model` (`clef`|`clef-flash`), `state` (string|object|array; truncated to fit), `questions` (1–64; ids `[A-Za-z0-9_.-]{1,100}`), `images` (<=4; PNG/JPEG/WebP; 4 MiB & 16 MP each, 8 MiB total, 13 MiB body). Question types noul/choice/score with TypeSafe shapes. Output `{model, answers, usage}` (answer field names as TypeSafe). Context 65,536 tokens. Pricing $0.24/M input (Clef 27B), ~$0.09/M (Clef-flash 9B) [secondary]; Clef-omni listed at $0.15. Open weights at https://huggingface.co/Cloudflare/clef (Apache 2.0 per llama.cpp table). Launched 1 Oct 2026.

### 5.3 Microsoft — Microsoft-Decision-1 on Microsoft Foundry [official] https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision
- `POST <foundry-resource-endpoint>/providers/microsoft/v1/systemone`, Entra bearer or `api-key` header. `model` = your **deployment name**; response `model` = underlying model. TypeSafe-shaped `state`/`questions` (noul with optional criteria, choice criteria map, score criteria array) and `answers` keyed by question name + usage. Deployment types GlobalStandard / DataZoneStandard. Errors 400/401/403/404/422/429. Docs explicitly warn: scores "can change based on how you phrase or order questions and options", "calibration is strongest on familiar task types", and recommend randomizing option order and adding an abstention option. Pricing reported $0.042/M [secondary]. Also on Vercel AI Gateway and LiteLLM (`azure_ai/<deployment>`).

### 5.4 Ollama 0.35 [official blog] https://ollama.com/blog/ollama-now-supports-jev-style-decision-models
- Local `POST http://localhost:11434/v1/systemone` "based on TypeSafe's Jev API". 1–64 named questions; choice 2–26 options; score 2–26 levels (per Ollama's announcement). Same request/response shapes. Models: `nimble` (Bespoke Labs, 9B), `tev1` (Together AI, 4B), `tev1:0.8b`. API only (not CLI / client libs yet). Use `typesafe-sdk` with `TYPESAFE_BASE_URL=http://localhost:11434`, `TYPESAFE_API_KEY=ollama`.

### 5.5 llama.cpp server [official README] https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md (PR #29818, merged 2 Oct 2026)
- `POST /v1/systemone` "follows the TypeSafe API", no streaming. Fields: `state` (string|object|array|null for some models), `files`/`images` (data URLs; audio for lfm2-d1-omni), `questions` (choice/score/noul; score 2–10 levels; noul criteria optional). Response identical to TypeSafe; `usage.output_tokens` always 0; `confidence` "0 means all options equally likely". **"The probabilities are scaled with the temperatures stored in the model file. They are not guaranteed to be calibrated for your data."** Options cap per model (52 for openjev; 255 for laya/clef/pplx-decider/lfm2-d1 ...). Day-0 GGUFs: Julia-1 (144M, mmBERT), Laya (421M, ModernBERT), Kev-4B (Qwen3.5-4B-Base, Apache 2.0, by Jared Palmer), lev (4B), OpenJev (27B, vision, CC BY-NC 4.0), Clef (27B). Latencies 3–43 ms/question on an RTX PRO 6000. Decision heads run on embedding graphs loaded from GGUF metadata; `/v1/models` marks native decision models with `output_modalities: ["decisions"]`.

### 5.6 vLLM [official PR] https://github.com/vllm-project/vllm/pull/59299 (merged 8 Oct 2026)
- `POST /v1/systemone` for **any autoregressive chat model** (no decision head needed). Mechanism: build chat prompt with assistant reply prefilled up to `id:`, generate one token with `logprob_token_ids` restricted to single-token labels (A–Z, AA–ZZ), **softmax over the label logprobs = answer distribution**; labels shuffled per question (optional `seed`); `confidence` = chosen label's probability over the full vocabulary; `diagnostics.{label_mass, argmax_is_label}` per question. Explicitly **uncalibrated**. Initially `choice` only (<=64 questions, <=128 options); follow-ups added `noul`/`score` (#60696), LiquidAI d1-3B native heads (#60676). Response adds `id`, `object: "structured_decision"`, `created`. This is the canonical open-source "logprob-based choice scoring" implementation behind a System One facade.

### 5.7 Other models in the category (hosted/open) [secondary: systemonemodels.org, Vercel/LiteLLM changelogs]
Decider 1 (meraGPT, $0.03/M), Solar Decide (Upstage, beta, reportedly `https://api.upstage.ai/v1/systemone` — unverified), Tev1 (Together), Drex (Nace.AI, `/v1/systemone`), Span-01 (Respan, noul-only), Kai (Hanzo), d1 / d1-ultra / d1-3B / d1-omni-600M (Liquid AI), GLiDE & GLiNER2.5-Decide (Fastino), Mercury Decide (Inception), Celeris-1 Decision, Strands-Decider 2B (AWS Strands; runnable on Bedrock AgentCore), H2O-Lightning-4B, Quyet-1.0-Large, Torchcast Decision 27B, Surogate Rune 26B-A4B (Fireworks "rune" router), Bespoke Nimble, CLM. AWS: **no Bedrock-native decision endpoint found**; the "AWS adopted the schema" headline (The New Stack) could not be verified from the article body.

### 5.8 Anthropic and Google: no native decision API
- Searched; **no Anthropic or Google endpoint** returns typed option distributions. Vercel AI SDK provides *adapters*: `anthropic.decisionModel('claude-haiku-4-5-20251001')` and `google.decisionModel('gemini-4-argon' / 'gemini-3.5-flash-lite')` use native structured output; **Choice/Score answers carry no probability distribution**, Boolean is a *prompted* (verbalized) P(true) "not guaranteed to be calibrated". Defaults to no reasoning/low thinking. Refusal/truncation fails the whole call.
- TypeSafe's own `system-one-adapter-python` (MIT) does the same for OpenAI/Anthropic/Gemini: `SystemOneAdapterClient(structured_outputs=True, llm_answer_mode="probabilities"|"discrete", normalize_probabilities=True)`; the LLM is prompted to emit a JSON distribution (no logprobs); debug trace of attempts in `response.debug["llm_attempts"]`.
- Classic open-source alternatives for a *true* distribution from an LLM without a decision head: next-token logprob scoring over option labels (what vLLM's `/v1/systemone` does; same idea as lm-evaluation-harness multiple-choice loglikelihood scoring and Outlines/guided-decoding `choice` constraints, which constrain generation but do not by themselves return a distribution). Caveats shared by all: position bias, label-token bias, and lack of calibration; shuffle labels and average over permutations.

---

## 6. Side-by-side schema comparison (for the adapter's normalization layer)

| Concept | TypeSafe `/v1/systemone` (and clones) | OpenAI `/v1/decisions` | AI SDK `experimental_decide` (neutral) |
| --- | --- | --- | --- |
| Context | `state`: string/object/array (+ `images`/`files` on some) | `input`: string or `[{role:"user", content:[input_text/input_image]}]` | `state`: string/object/array of parts `{type:text|json|file}` |
| Question container | map `questions.<id>` | array `questions[]` with `name` | map keyed by id |
| Yes/no | `type:"noul"`, `criteria:{true,false}` optional | `type:"predicate"` (criteria folded into instructions by gateways) | `type:"boolean"`, `criteria:{true,false}` optional |
| Finite choice | `criteria: {option: desc|null}` (1–255) | `choices: [{value: string|bool, description?}]` (2–255 unique) | `criteria: {key: desc|null}` |
| Ordinal | `criteria: [level0, level1, ...]` (2–10) | `levels: [{label, description?}]` (>=2) | `criteria: [..]` (>=2) |
| Yes/no answer | `{type:"noul", noul: p}` | `{type:"predicate", name, probability: p}` | `{type:"boolean", probability: p}` |
| Choice answer | `{choice, probabilities:{opt:p}, confidence}` | `{choice, probabilities:[{value,probability}], confidence}` | `{choice, probabilities?:{key:p}}`; confidence in `providerMetadata.<provider>.confidence` |
| Score answer | `{score, legend:{"0":..}, probabilities:{"0":p}, confidence}` | `{score, probabilities:[{value:int,label,probability}], confidence}` | `{score, probabilities?:{"0":p}}` |
| Refusal | not modeled (422/400 errors) | `{type:"refusal", name}` per question | throws `Experimental_DecisionRefusalError` |
| Usage | `{input_tokens, output_tokens}` | `{input_tokens, input_tokens_details, output_tokens, output_tokens_details, total_tokens}` | `{inputTokens, outputTokens, totalTokens?}` |
| Rounding | 2 decimals (Jev) | unspecified | `rounding.probabilityDecimals` |

Suggested normalized internal type for our agent: `Answer = Choice{choice, probs: dict[str,float], confidence: float|None} | Score{score, probs: dict[int,float], legend, confidence|None} | Bool{p_true} | Refusal{reason}`, plus `meta{model, latency_ms, usage, raw}`. Renormalize `probs` after rounding only in our own layer, and record whether the distribution was *native* (decision head / logprobs) or *verbalized* (LLM adapter), because calibration differs materially.

---

## 7. Evidence on calibration and robustness (relevant to using these as agent policies)

- **JevOut (arXiv 2609.30243, v2 8 Oct 2026)**: benign-looking context additions flipped 61–73% of initially-correct decisions across four decision systems incl. Jev on seven datasets, often into high-confidence wrong answers; attacks built by "probability-guided optimization" of surrounding text. Confidence does not flag these failures. Directly relevant to multi-agent games where other agents write the observation.
- **Evaluating Decision Models for Text Annotation in CSS (arXiv 2609.24574)**: first commercial decision model trails the best LLM in 14/15 tasks (median −11.6 macro-F1) at ~44x lower cost; confidence better calibrated than verbalized confidence of 16/19 LLMs but worse than three frontier models; items with confidence >0.9 have median accuracy 0.815; confidence-gated routing to an LLM matches LLM-only at 25–50% cost.
- **OpenAI community coin test**: predicate calibrated (0.70), choice badly over-confident (0.98) and order-sensitive [single tester].
- **Microsoft's own docs**: order/wording sensitivity acknowledged; recommend randomizing option order.
- **llama.cpp / vLLM**: explicitly uncalibrated (temperature-scaled or raw logprob softmax).
- Jev benchmarks: TypeSafe publishes only internal workflow evals; third-party "Decision Index" places pplx-decider-v1.1 above Jev [secondary]. decision-bench PR: OpenAI Decisions 90.5% on its corpus [secondary].

Practical consequences for the simulation adapter: (a) always request the full distribution and keep it; (b) permute option order across repeated calls and average (cheap: input-only billing, 70–300 ms); (c) treat `confidence` as a dispersion statistic, not accuracy; (d) prefer noul/predicate for binary gates; (e) log the resolved `model` version since aliases move.

---

## 8. Large Language Bayes (Domke, arXiv 2504.14025, NeurIPS 2025; v2 23 Oct 2025)

Source: https://arxiv.org/abs/2504.14025 and https://arxiv.org/html/2504.14025 (full HTML read).

### 8.1 Abstract (verbatim)
"Many domain experts do not have the time or expertise to write formal Bayesian models. This paper takes an informal problem description as input, and combines a large language model and a probabilistic programming language to define a joint distribution over formal models, latent variables, and data. A posterior over latent variables follows by conditioning on observed data and integrating over formal models. This presents a challenging inference problem. We suggest an inference recipe that amounts to generating many formal models from the large language model, performing approximate inference on each, and then doing a weighted average. This is justified and analyzed as a combination of self-normalized importance sampling, MCMC, and importance-weighted variational inference. Experimentally, this produces sensible predictions from only data and an informal problem description, without the need to specify a formal model."

### 8.2 The formal setup
- Inputs: informal text `t` (problem, data layout, assumptions, target variables) and observed data `x`.
- The LLM (with system prompt and sampling hyperparameters) **is** a prior over formal programs: `p(m | t)` (Eq. 1). The PPL (Stan) defines `p(z, x | m)` (Eq. 2). Gluing: `p(z, x, m | t) = p(m | t) p(z, x | m)` (Eq. 3).
- Target posterior marginalizes the program: `p(z | x, t) = Σ_m p(m | x, t) p(z | x, m)` (Eq. 4) with **posterior model weight** `p(m | x, t) ∝ p(m | t) · p(x | m)` (Eq. 5) — i.e. Bayesian model averaging with an LLM-defined prior over models.
- Models may have different latent spaces `u^(m)`; only the named target variables `z` must exist in every program (Sec. 2.1).
- Contrast with a **flat ensemble** `p_flat(z|x,t) = Σ_m p(m|t) p(z|x,m)` (Eq. 7): the true posterior additionally multiplies by marginal likelihood, favoring programs consistent with the data.

### 8.3 Why exact inference is impossible and what is done instead
- `p(m|t)` is unavailable (commercial APIs do not expose sequence probabilities; chain-of-thought makes it intractable since many reasoning traces yield the same program), and the model space is huge. So reversible-jump-style BMA does not apply.
- **Algorithm 2 (generic recipe)**: for n = 1..N: sample `m^(n) ~ p(m|t)` with the LLM; approximate `p̃(z|x,m^(n))` and `p̃(x|m^(n))` with the PPL; set `w^(n) ∝ p̃(x|m^(n))`; return `Σ_n w^(n) p̃(z|x,m^(n))`.
- **Algorithm 3 (as run)**: (a) sample a program; (b) run MCMC (Stan NUTS, 2 chains × 10,000 iters) to get samples `z^(n,k)`; (c) compute an **importance-weighted lower bound** on `log p(x|m^(n))`: `L(q) = E_{z^1..z^K ~ q} log (1/K) Σ_k p(z^k, x|m)/q(z^k) ≤ log p(x|m)` (Eq. 9) with `q` a Gaussian **moment-matched to the MCMC samples** (not optimized; justified because the IW bound is asymptotically tight at rate 1/K with a χ²(p‖q) constant, and moment matching minimizes the inclusive KL(p‖q), a pragmatic compromise); (d) weights `w^(n) ∝ exp L^(n)`; (e) output the pooled weighted samples.
- Validity filtering: reject outputs lacking THOUGHTS/MODEL blocks, that fail to compile, or that contain `target +=` (unnormalized densities). Stan constraints silently leave truncated densities unnormalized, biasing log marginal likelihoods **downward by a constant per model** — accepted as a conservative bias. Valid rates ranged from ~97% (coin) to 0.4% (hard "gold" problem).

### 8.4 Model generation details
- System prompt ("You are StanWriter…"): first write THOUGHTS (strategy in words), then MODEL (Stan code); "Be creative"; always give proper priors; never `target +=`; modern array syntax; `_rng` in generated quantities.
- Six in-context examples, each user input with PROBLEM / DATA / GOAL blocks and an exemplary THOUGHTS/MODEL output.
- Llama-3.3-70B (4-bit AWQ), 1024 programs per problem (16,384 for one), 10–15 min on one A100 via continuous batching. Stan chosen over NumPyro/PyMC because LLMs wrote it more reliably.

### 8.5 Theory (Sec. 5)
- Self-normalized importance sampling over model space: with proposals from `p(m|t)`, weights reduce to normalized marginal likelihoods (Eq. 10); asymptotic variance bounded by `δ²/N · (1 + χ²(p(m|x,t) ‖ p(m|t)))` (Eq. 12) — so **the LLM prior should be broad**; an overconfident prior (missing high-likelihood programs) hurts far more than an underconfident one.
- Variational view (Thm 1 / Cor 2): with per-model approximations `q(z|x,m)`, the KL-optimal model weights are `q(m|x) ∝ p(m|t) · exp(ELBO(m))` (Eq. 19); the joint divergence is `log p(x|t) − log E_{p(m|t)} exp(ELBO(m))`. Inaccuracy in a model's posterior approximation matters only in proportion to that model's posterior weight.
- Mixed variant (Sec. 5.3 / C.4): posterior from MCMC, weights from a bound; divergence `= −log E_{p(m|x,t)} exp(−δ(m) + δ̄)` where `δ(m)` is the bound gap — **constant gaps cancel; only gap variation among high-weight models matters**.

### 8.6 Experiments (new datasets to avoid memorization; asks future evaluators to use LLMs with cutoff before 21 Apr 2025)
Rain (22 binary days; posterior ≈0.79 vs flat 0.57; one multi-day dependency model dominates), Coin (user's stated assumptions shift the posterior appropriately), Polling (one model takes almost all weight; little gain over flat), City temperature (72% compiled; ≈ flat), Gold (weights concentrate; clearly better than flat at 150 obs). Limitations: expensive per-model inference (embarrassingly parallel), many near-duplicate programs, wasted invalid generations (suggests constrained decoding), no optimality claim, sensitivity to prior breadth and inference quality.

### 8.7 How to reuse the pattern: "natural-language contract -> candidate formal Rule(s) with posterior weighting"
Map the paper's objects onto our setting:

| LLB | Our design |
| --- | --- |
| informal text `t` | the natural-language contract / rule description (plus schema of the observation and the action set) |
| formal model `m` (Stan program) | a candidate **formal Rule** (e.g. a typed predicate/program over observations, or a decision-model question set + thresholds) |
| data `x` | observed episodes: (observation, action, outcome) triples or human-labelled rulings |
| target `z` | the quantity we act on: the action distribution / rule verdict for a new observation |
| `p(m|t)` | sample many Rule candidates from an LLM with a THOUGHTS-then-RULE prompt, few-shot examples, and syntactic validation (compile/typecheck/execute on a probe); keep the prior **broad** (temperature, diverse exemplars) |
| `p(x|m)` | the Rule's likelihood of the observed rulings — for deterministic rules use a noisy-channel likelihood (e.g. each ruling matches with prob 1−ε); for probabilistic rules use their predictive probability; tempered variants are fine because only *relative* weights matter |
| `w^(n) ∝ p(x|m^(n))` | posterior weights over Rule candidates; the final policy is the mixture `Σ_n w^(n) p(z|x,m^(n))` |

Practical recipe for a simulation:
1. Generate N candidate rules (structured output; reject non-compiling ones; dedupe by canonical form / behavioral signature on a probe set — the paper notes near-duplicates waste compute).
2. Score each on the labelled history to get log-likelihoods; if a candidate needs internal latent parameters (thresholds, weights), fit them and use an ELBO/IW bound or BIC-style approximation as the paper does rather than raw max-likelihood.
3. Self-normalize weights (`softmax` of log-evidences). Monitor effective sample size; if one rule takes ~all mass, suspect either real evidence or an inference bug (the paper flags this ambiguity).
4. At decision time, mix the candidates' action distributions by weight — this yields a **distribution over the finite action set** that integrates rule uncertainty, which is exactly the `probabilities` map our agent interface expects. Decision-model APIs (Sec. 1–5) can serve as cheap, fast *likelihood evaluators* (noul/choice questions asking "does rule R permit action a in observation o?") inside this loop.
5. Update incrementally: new rulings re-weight existing candidates (cheap) and periodically trigger regeneration (new proposals), mirroring SNIS with a fixed proposal pool.

Caveats carried over: constant evidence biases cancel but rule-dependent ones do not; weights are sensitive to likelihood temperature; a narrow proposal prior silently drops the right rule.

---

## 9. Sources

Official / primary
- TypeSafe API reference: https://docs.typesafe.ai/api (markdown: https://docs.typesafe.ai/api.md)
- TypeSafe quickstart: https://docs.typesafe.ai/introduction/quickstart
- TypeSafe primitives: https://docs.typesafe.ai/primitives/choice, https://docs.typesafe.ai/primitives/score, https://docs.typesafe.ai/primitives/noul, https://docs.typesafe.ai/primitives/box
- TypeSafe confidence: https://docs.typesafe.ai/confidence
- TypeSafe models/limits: https://docs.typesafe.ai/models
- TypeSafe state: https://docs.typesafe.ai/concepts/state
- TypeSafe docs index: https://docs.typesafe.ai/llms.txt
- TypeSafe launch post: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe JS SDK interfaces: https://docs.typesafe.ai/sdk/javascript/api/interfaces/SystemOneRequest, .../SystemOneResult, .../ChoiceResponse, .../ScoreResponse, .../Usage
- TypeSafe LLM adapter: https://github.com/typesafe-ai/system-one-adapter-python
- Vercel AI SDK TypeSafe provider: https://ai-sdk.dev/providers/ai-sdk-providers/typesafe-ai
- Vercel AI SDK decisions core: https://ai-sdk.dev/docs/ai-sdk-core/decisions
- Vercel AI SDK OpenAI provider (decision models): https://ai-sdk.dev/providers/ai-sdk-providers/openai
- Vercel AI SDK Anthropic provider (decision adapter): https://ai-sdk.dev/providers/ai-sdk-providers/anthropic
- Vercel AI SDK Google provider (decision adapter): https://ai-sdk.dev/providers/ai-sdk-providers/google-generative-ai
- OpenAI Decisions guide: https://developers.openai.com/api/docs/guides/decisions (markdown: .../decisions.md)
- OpenAI Decisions + voice: https://developers.openai.com/api/docs/guides/decisions-voice
- OpenAI pricing: https://developers.openai.com/api/docs/pricing
- OpenAI Python SDK types: https://github.com/openai/openai-python/blob/main/src/openai/types/decision.py, .../decision_create_params.py, .../resources/decisions.py
- OpenAI community announcement thread: https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877
- LiteLLM decisions docs: https://docs.litellm.ai/docs/decisions ; blog: https://docs.litellm.ai/blog/decision-models ; release: https://docs.litellm.ai/release_notes/v1.104.2/v1-104-2
- OpenRouter Jev guide: https://openrouter.ai/docs/guides/community/jev ; API reference: https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request ; blog: https://openrouter.ai/blog/insights/what-is-jev/
- Vercel AI Gateway Decision modality: https://vercel.com/docs/ai-gateway/modalities/decision ; TypeSafe-compatible API: https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe ; KB guide: https://vercel.com/kb/guide/typesafe-jev-and-ai-sdk
- Perplexity Decisions quickstart: https://docs.perplexity.ai/docs/decisions/quickstart
- Cloudflare Workers AI Clef: https://developers.cloudflare.com/workers-ai/models/clef/ ; changelog: https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/ ; blog: https://blog.cloudflare.com/clef-decision-models
- Microsoft Foundry Decision-1: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision
- Ollama blog: https://ollama.com/blog/ollama-now-supports-jev-style-decision-models
- llama.cpp server README (`/v1/systemone`): https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md ; PR: https://github.com/ggml-org/llama.cpp/pull/29818 ; HF blog: https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp
- vLLM `/v1/systemone` PR: https://github.com/vllm-project/vllm/pull/59299
- Large Language Bayes: https://arxiv.org/abs/2504.14025 ; https://arxiv.org/html/2504.14025
- JevOut: https://arxiv.org/abs/2609.30243
- Evaluating Decision Models for Text Annotation in CSS: https://arxiv.org/abs/2609.24574

Secondary (press, aggregators, community)
- InfoQ: https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/
- DataCamp: https://www.datacamp.com/blog/system-one-models-jev
- systemonemodels.org hub (model list, pricing): https://systemonemodels.org/
- Portkey enterprise changelog: https://portkey.ai/docs/changelog/enterprise
- agentgateway blog: https://agentgateway.dev/blog/2026-10-12-openai-decisions-api-jev-agentgateway/
- llmgateway.io changelog: https://llmgateway.io/changelog/openai-compatible-decisions-api
- The New Stack (headline only; body not retrievable): https://thenewstack.io/decision-models-system-one/
- aicoder on llama.cpp: https://aicoder.com/news/news-20261003-llamacpp-systemone-decision-models
- decision-bench PR (OpenAI Luna benchmark): https://github.com/atlanai/decision-bench/pull/15
- AWS Builder "Decision Models Explained": https://builder.aws.com/content/3KKNNtB4pr0Ul87PfCccFqr9UyX/decision-models-explained-where-they-came-from-how-to-read-the-leaderboards-and-how-to-choose-one-on-aws
- Unite.ai / MarkTechPost coverage of OpenAI Decisions beta: https://www.unite.ai/openai-releases-decisions-api-in-public-beta-powered-by-gpt-6-luna/ , https://www.marktechpost.com/2026/10/09/openai-decisions-api-hits-public-beta-with-10x-faster-typed-answers/
