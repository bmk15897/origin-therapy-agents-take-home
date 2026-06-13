# Origin AI Engineering Take-Home: Referral Inbox Triage Agent

Origin builds software for pediatric therapy practices. In this assignment, you are helping a fictional practice, Cedar Kids Therapy, triage its Monday inbox.

## Scenario

It is Monday at 8am at a multi-disciplinary pediatric therapy practice supporting speech-language pathology, occupational therapy, and physical therapy. The shared inbox accumulated items over the weekend from pediatrician fax referrals, parent voicemails, parent portal messages, and emails. Build an AI agent prototype that turns the messy batch into a sorted, human-reviewable action plan.

## What We Expect

Strong submissions are usually incomplete but honest. We are evaluating triage judgment, tool orchestration, and scoping, not whether you finished every nice-to-have. Produce some output for every item, even thin; document what you cut in the README.

You may use any AI coding agent (Claude Code, Cursor, Codex, etc.) while building. State your stack and assumptions in your README.

Runtime LLM usage is allowed and recommended, but not required. Origin will provide a temporary capped API key for either OpenAI or Anthropic; the email distributing the key will name the provider and the environment variable to set (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`). You may also use your own provider. You may install dependencies for the provider you choose (e.g., `npm install openai` or `npm install @anthropic-ai/sdk`). Use any key only with the provided synthetic data, store it in an environment variable, and do not commit it. Model choice is not part of the rubric.

## How To Run

```bash
npm install
npm run triage   -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
```

The commands also work with no flags and default to the paths above. Reviewers may run the same commands against similar hidden synthetic input. Do not hardcode input, output, or trace paths.

## Share And Submit

Create your own GitHub repo from this starter pack and implement your solution there. The repo can be public or private. When you are done, submit the repo link. If it is private, grant access to the Origin reviewer GitHub account `@nixu`.

Commit your code, your updated `README.md`, and your final generated `output.json`. Do not commit API keys, `.env` files, real PHI, `node_modules/`, or `.trace/`.

We expect you to spend about 2 hours. If you stop before finishing, commit what you have and describe the cuts in your README.

---

## 1. How to Run

Set your Anthropic API key before running:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Then install dependencies and run the triage agent:

```bash
npm install
npm run triage   # defaults: --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run validate # validates output.json against schema/output.schema.json
```

Custom paths:

```bash
npm run triage -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
```

Run tests (requires `vitest` to be installed per issue 10):

```bash
npm test
```

Expected runtime for `npm run triage` on the 8-item inbox: under 2 minutes (sequential, one LLM call chain per item).

---

## 2. Stack and Runtime

- **Language / runtime:** TypeScript, Node LTS, `tsx` (no compile step)
- **LLM provider:** Anthropic — `@anthropic-ai/sdk`
- **Model:** `claude-sonnet-4-6` (changeable via env var without structural changes)
- **Test framework:** Vitest
- **Key dependencies:** `ulid` (ULID generation for IDs), `ajv` + `ajv-formats` (JSON schema validation)
- **No database or external services** — all tool calls are deterministic in-memory stubs in `src/tools.ts`

---

## 3. Problem Framing and Design Thinking

### How we decomposed the problem

The inbox is a batch of **heterogeneous items** — referrals, voicemails, portal messages, emails — that each require a different action chain. The core challenge is not just reading them but routing them correctly and safely, because a wrong classification has asymmetric costs: missing a safeguarding signal is catastrophic; over-routing a spam item is merely wasteful.

We framed this as **ER triage**, not inbox automation. The goal is not to resolve items — it's to sort, prioritize, and prepare them for human action. This framing drove every key decision:

### Key tensions we resolved

**1. Safety before efficiency**

The most important structural choice is the **safeguarding pre-pass** (Step 2) running unconditionally before classification (Step 3). A naive design would classify first and then check for urgency signals — but that risks a P0 disclosure being buried inside an otherwise routine-looking referral and not being escalated. We hard-coded the order: scan for harm signals first, always, before any other reasoning.

**2. Automation ceiling**

Every tool in the system deliberately stops short of irreversible action. `draft_message` creates a draft, not a send. `hold_slot` creates a pending hold, not a booking. `create_task` queues work for a human, not executes it. This was not an implementation shortcut — it reflects the domain constraint that a pediatric therapy practice should never auto-send clinical communications or auto-book appointments without staff review.

**3. Structured output via tool call, not text parsing**

We chose the **sentinel tool pattern** (`submit_triage_result` defined as a tool) rather than parsing the LLM's final text for structured output. This gives us schema-enforced structured extraction without post-processing fragility: Claude calls the tool with typed args, we parse the args directly into `ItemOutput`. The tradeoff is that it requires the sentinel to be excluded from the audit trace (it's an implementation detail, not an action).

**4. Sequential processing over parallelism**

Items are processed one at a time. The reason is state consistency: a slot hold placed during item 1's processing should be visible context when item 3 arrives for the same provider. Parallelising would require a shared ledger for holds and patient records — a reasonable next step, but adds complexity that isn't justified at prototype scale.

**5. Classification as a hypothesis, not a gate**

Classification (Step 3) is an initial hypothesis that the tool loop can revise. For example, an item that looks like a new referral at classification time may hit a guardian mismatch during `search_patient` and pivot to a different path. The flowchart is intentionally designed so that tool results can redirect the agent mid-loop, not just confirm a pre-committed decision.

**6. Prompt encodes policy, not business logic**

We encode the decision flowchart and urgency definitions verbatim in the system prompt rather than implementing them as code branches. This allows the model to handle the long-tail of phrasing variations (implied disclosures, partial Spanish, ambiguous urgency) that would require extensive case handling in code. The code enforces structural invariants (always `requires_human_review: true`, spam never gets a draft reply) but delegates judgment to the LLM.

---

## 4. Architecture

The agent implements a **LLM-driven ReAct loop** per inbox item, running items **sequentially** so that earlier tool call side-effects (e.g., a slot hold placed for item 1) are visible when processing later items that reference the same patient or provider.

**Per-item loop:**
1. `src/index.ts` calls `configureTrace` then `runAgent(inbox)`
2. For each `InboxItem`, the agent sends the item to Claude with a system prompt and all tool definitions (8 real tools + an internal-only sentinel `submit_triage_result`)
3. Claude reasons, calls tools mid-generation; tool results are fed back as `tool_result` content blocks
4. The loop ends when Claude calls `submit_triage_result` (sentinel parsed as the structured `ItemOutput`) or after 10 iterations — whichever comes first
5. Per-item `try/catch` ensures a fallback output is emitted on any failure; no item is silently dropped

**Sentinel tool pattern:** `submit_triage_result` is defined as a tool alongside the 8 real tools so Claude can call it to "submit" its final structured output. It never appears in `tools_called[]` (excluded by `getToolCallsForItem`) and is not imported from `src/tools.ts`.

**System prompt encodes the decision flowchart** from the PRD:
- Step 1: Extract intake fields (no tool call)
- Step 2: Safeguarding pre-pass (P0 → escalate → stop)
- Step 3: Classify item type
- Step 4: ReAct tool loop per classification (new referral, scheduling, clinical question, missing paperwork)
- Step 5: `submit_triage_result`

All tool calls are wrapped in `withItemContext(item.id, ...)`. `tools_called[]` is populated exclusively via `getToolCallsForItem(item.id)` — never hand-constructed.

**Output:** `buildBatchOutput(items)` (from `src/tools.ts`) wraps the `ItemOutput[]` with a batch summary; `src/index.ts` writes it to `output.json`.

---

## 5. Failure Modes and Production Eval

**Known failure modes:**

| Mode | Mitigation in this prototype |
|------|------------------------------|
| LLM skips safeguarding pre-pass | System prompt enforces it as an unconditional first step; P0 items in `data/inbox.json` verify the path |
| Max iterations exceeded (10-iter cap) | Fallback `ItemOutput` emitted with `requires_human_review: true`; never a silent drop |
| Performative tool calls (calling tools without purpose) | Per-tool guidance in system prompt; rubric penalizes speculative calls |
| Guardian name mismatch false positive | Comparison is LLM-side; fuzzy matching (nicknames, typos) risks both false positives and misses |
| Model hallucinates tool args | Anthropic SDK validates tool input schemas; invalid calls raise an error caught by per-item handler |
| API rate limit / timeout | Per-item `try/catch` produces fallback output; full batch is not aborted |
| Out-of-network payer treated as in-network | `verify_insurance` is deterministic stub; production would need live billing system call |

**What a production eval harness would add:**
- Regression suite of synthetic inbox variants with known ground-truth outputs (classification, urgency, tools called)
- Precision/recall on safeguarding detection across adversarial phrasings
- Over-escalation rate (P0/P1 false positives are a cost — they erode staff trust)
- Hallucination detection: does `draft_reply` ever contain clinical advice? Runs a classifier over every draft
- Latency p50/p95 per item; budget alarm if p95 exceeds 30s
- Trace replay: re-run any item from its JSONL trace entry for reproducibility

---

## 6. What I Chose Not to Build, and Why

Scope cuts follow the PRD "Out of Scope" section:

- **Auto-sending messages** — `draft_message` only; no send action exists. Keeping a human in the loop before any outbound communication is a hard safety requirement.
- **Appointment booking** — `hold_slot` (pending review) is the ceiling. Scheduling is a staff decision after reviewing options with the family.
- **Parallel item processing** — the PRD specifies sequential processing so slot holds placed for item N are visible when processing item N+1 (same patient/provider). Parallelism would require a shared hold ledger.
- **Retry / backoff on API errors** — per-item fallback is the floor for a 2-hour prototype; production would add exponential backoff with jitter.
- **Crisis-in-progress handling** ("my child is hurting themselves now") — redirect to 911/crisis line; different escalation path not modeled.
- **Multi-child households** — one message with two children is treated as one item; the second child's intake would be missed. Noted in `missing_info[]` if detected.
- **Languages other than English and Spanish** — flag for human translation; no draft attempted in unsupported languages.
- **Age out of range** (referral for child over 18) — decline and redirect not implemented.
- **School-based / IEP-funded referrals** — different billing path, not modeled.
- **Cross-item deduplication within a batch** — each item is processed independently; staff correlates related items during human review.

---

## 7. What I Would Do With Another 4 Hours

1. **Complete test suite (issue 10):** Vitest tests for all 8 decision branches, asserting on `ItemOutput` shape and `tools_called[]` presence — not on prompt text or model internals. This is the highest-confidence way to lock in correct behaviour across model updates.

2. **Structured output / JSON mode:** Replace the sentinel tool pattern with Anthropic's native structured output (or a strict JSON schema response format) for the final `ItemOutput`. The sentinel works but is more fragile than a constrained decoding approach.

3. **Prompt eval harness:** A second synthetic inbox with 8–10 items covering adversarial phrasings (implied safeguarding, ambiguous urgency, borderline clinical questions) and known ground-truth labels. Run it as a CI job to catch regressions before shipping prompt changes.

4. **Guardian name normalization:** The current implementation relies on the LLM to compare guardian names. A production system would normalize (strip titles, lowercase, phonetic matching) before comparison to reduce false positives on "Sofia" vs "Sofía" or "Mike" vs "Michael".

5. **Latency observability:** Emit per-item timing into the JSONL trace so the audit log shows not just what was called but how long each tool invocation and LLM turn took. Useful for capacity planning and detecting slow items in production.

## Your Task

Implement the agent in `src/agent.ts`. It should read the `InboxItem[]` it receives, use the provided tools where appropriate, and return one output item per inbox item. `src/index.ts` wraps your items with `buildBatchOutput()` and writes the final `output.json`.

Available tools: `search_patient`, `verify_insurance`, `lookup_policy`, `find_slots`, `hold_slot`, `create_task`, `draft_message`, `escalate`.

Use `schema/output.schema.json` as the source of truth for the output shape. `data/example_output.json` shows one non-trivial worked item. It is illustrative and is not expected to pass validation by itself. **Do not copy the example call IDs** into your output — real outputs must use the `call_id` values returned by `getToolCallsForItem()`.

## Time Box

Spend about 2 hours. Suggested allocation: 20 minutes reading and designing, 70 minutes building, 20 minutes self-evaluating against the validator and the inbox, 10 minutes updating the README. Expected end-to-end runtime for `npm run triage` should be a few minutes or less; if your agent is much slower, that is worth noting in the README rather than optimizing under time pressure.

Minimum viable submission: processes every item in `data/inbox.json`, makes relevant tool calls including at least 3 distinct tools across the batch, writes a valid `output.json`, and passes `npm run validate`. Beyond that floor, your architecture, error handling, audit discipline, and scoping choices are part of what we evaluate.

## Constraints

- Use TypeScript, Node LTS, and npm. If this creates a real accessibility or environment issue, reach out.
- Use the provided tools in `src/tools.ts`; do not modify, reimplement, or bypass them. The tools create the audit trace used by the validator, so bypassing them fails validation.
- Use at least 3 distinct tools across the batch. Strong solutions use tools as part of the decision process across multiple items, not just once to satisfy the threshold. Irrelevant or performative tool calls will be penalized.
- Use `withItemContext(item.id, async () => ...)` around item-level tool calls.
- Use `getToolCallsForItem(item.id)` for `tools_called[]`; pass the returned entries through unchanged.
- Use `buildBatchOutput(items)` through the starter `src/index.ts`; do not hand-compute summary counts.
- Do not auto-send messages. Use `draft_message` only.
- Do not schedule appointments. `find_slots` and `hold_slot` are reviewable; scheduling is not.
- Use only synthetic data. Do not add real PHI.

## Urgency Calibration

- `P0`: safeguarding, imminent harm, mandated-reporter escalation. Same-hour human review.
- `P1`: same-day operational issue requiring prompt staff action.
- `P2`: normal intake, scheduling, billing, or clinical-review workflow.
- `P3`: low-priority admin, FYI, spam.

Default to `P2` unless there is a clear safety or same-day operational reason. Over-escalation is itself a production failure mode.

## Review Variants

Similar synthetic variants may be run during review. We will not tell you what they cover, but the visible 8 items show the kinds of cases we care about.

## Rubric

- Safety and domain judgment: 25%
- Tool orchestration and action model: 25%
- Output correctness and auditability: 20%
- Engineering quality: 15%
- README and production thinking: 15%

Draft replies should be clear, empathetic, concise, and operationally useful. They must not provide clinical advice or imply messages were sent.
