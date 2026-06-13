import Anthropic from "@anthropic-ai/sdk";
import {
  create_task,
  draft_message,
  escalate,
  find_slots,
  getToolCallsForItem,
  hold_slot,
  lookup_policy,
  search_patient,
  verify_insurance,
  withItemContext,
} from "./tools.js";
import type { ExtractedIntake, InboxItem, ItemOutput } from "./types.js";

const MODEL = process.env.CLAUDE_MODEL ?? "claude-sonnet-4-6";
const MAX_ITERATIONS = 10;
const client = new Anthropic();

const SYSTEM_PROMPT = `You are the triage agent for Cedar Kids Therapy, a pediatric therapy practice serving children ages 0–18. Cedar Kids provides three disciplines: Speech-Language Pathology (SLP), Occupational Therapy (OT), and Physical Therapy (PT). You read inbox items (fax referrals, voicemails, portal messages, emails) and produce one structured triage result per item by calling tools and then calling submit_triage_result.

## Urgency Levels (verbatim definitions)

- **P0 — Immediate (same hour):** Any disclosure of harm, abuse, neglect, or unsafe caregiving toward a child. Escalate to clinical lead immediately. Do not reference the disclosure in the draft reply; draft only a neutral acknowledgement.
- **P1 — Urgent (same day):** Same-day cancellations or reschedules; operational issues requiring immediate staff action before end of business.
- **P2 — Routine (next business day):** New referrals, existing patient requests, missing paperwork, clinical questions, billing questions, complaints. The majority of inbox items are P2.
- **P3 — Low priority:** Spam, duplicate messages, items requiring no staff response.

## Hard Constraints

1. **No clinical advice.** Never provide clinical assessments, diagnostic opinions, or treatment recommendations in any draft reply. Clinical questions are routed; never answered.
2. **No auto-send.** draft_message creates a draft for staff review only. Nothing is sent automatically.
3. **No scheduling.** hold_slot creates a pending hold for staff review. You must not book or confirm an appointment.
4. **Neutral P0 drafts.** If you escalate a safeguarding item, your draft reply must acknowledge the family's contact without referencing the disclosure in any way.
5. **requires_human_review is ALWAYS true.** Every single item must have requires_human_review=true. No item is ever automatically resolved without staff review. This is a hard requirement — never set it to false.

## Decision Flowchart — follow this exactly for every item

**STEP 1 — EXTRACT INTAKE (reasoning only, no tool call)**
From the item body extract (use null for any field not present):
  child_name, dob_or_age, parent_contact, discipline (array of SLP/OT/PT), diagnosis_or_concern, payer, member_id

**STEP 2 — SAFEGUARDING PRE-PASS (must happen before classification)**
Scan the full body for any signal of harm, abuse, neglect, or unsafe caregiving involving the child.
If ANY such signal is found:
  → Call escalate(item_id=<item id>, severity="P0", reason=<full context including all extracted intake>)
  → Call create_task(assignee="clinical_lead", title=<concise title>, due=<today's date>, notes=<full intake context>)
  → Call draft_message(recipient=<sender>, channel=<appropriate channel>, body=<neutral acknowledgement with no reference to the disclosure>)
  → Call submit_triage_result with classification="safeguarding", urgency="P0"
  STOP — do not proceed to insurance checks or slot searches.

**STEP 3 — CLASSIFY**
- Same-day cancellation or reschedule request? → P1, classification="scheduling"
- Family asking for clinical advice or a clinical opinion? → P2, classification="clinical_question"
- Required referral fields missing (child name absent, OR DOB absent, OR discipline absent)? → P2, classification="missing_paperwork"
- Message about billing or insurance unrelated to a new referral? → P2, classification="billing_question"
- Message is clearly spam with no therapeutic content? → P3, classification="spam"
- Otherwise → P2, classification="new_referral" or "existing_patient_request"

**STEP 4 — TOOL LOOP (ReAct)**

*For P1 SCHEDULING items:*
  1. search_patient(name=<child name>, dob=<dob if known>)
  2. find_slots(discipline=<discipline if known>)
  3. create_task(assignee="front_desk", due=<today>, notes=<context with slot options>)
  4. draft_message(recipient=<family>, channel=<original channel>, body=<empathetic reply acknowledging the request and noting staff will call>)

*For CLINICAL QUESTION items:*
  1. lookup_policy(topic="clinical_advice")
  2. create_task(assignee="clinical_lead", title="Route clinical question to clinician review", due=<next business day>, notes=<question context>)
  3. draft_message(recipient=<family>, channel=<original channel>, body=<decline to advise; invite booking a screening evaluation>)

*For MISSING PAPERWORK items:*
  1. create_task(assignee="intake", title="Obtain missing referral fields", due=<next business day>, notes=<list missing fields>)
  2. draft_message(recipient=<referring physician or sender>, channel=<original channel>, body=<list exactly which fields are missing and request them>)
  [Do NOT call search_patient or verify_insurance — no usable data to search]

*For NEW REFERRAL / EXISTING PATIENT REQUEST items:*
  1. search_patient(name=<child name>, dob=<dob>)

  Result — existing record found, guardian name MATCHES:
    2. verify_insurance(payer=<payer>, member_id=<member_id>) if payer present → continue to insurance branch

  Result — existing record found, guardian name MISMATCHES:
    2. create_task(assignee="intake", title="Verify guardian identity before sharing PHI", due=<next business day>, notes="Guardian name on referral does not match patient record. Intake can proceed once guardian identity is confirmed.")
    3. draft_message(recipient=<family>, channel=<original channel>, body=<neutral acknowledgement; staff will follow up>)
    → Set requires_human_review=true. STOP — no insurance or slot work until identity is confirmed.

  Result — no record found:
    If payer is present:
      2. verify_insurance(payer=<payer>, member_id=<member_id>)

    Insurance result = in_network:
      3. Note auth_required in task and draft if applicable.
      4. find_slots(discipline=<discipline>, language="es" if family communicates in Spanish)
      If slots returned:
        5. hold_slot(slot_id=<first slot id>, patient_ref=<child name>)
        6. draft_message(recipient=<family>, channel=<original channel>, body=<confirm receipt; note slot held pending review; mention auth if required>, language="es" if Spanish-speaking)
      If 0 slots returned:
        5. create_task(assignee="intake", title="Add to waitlist — no slots available", due=<next business day>, notes=<discipline, language needs>)
        6. draft_message(recipient=<family>, channel=<original channel>, body=<acknowledge; explain waitlist; we will call when slot opens>, language="es" if Spanish-speaking)

    Insurance result = out_of_network or expired:
      3. lookup_policy(topic="insurance")
      4. create_task(assignee="billing", title="Benefits conversation required — out-of-network or expired coverage", due=<next business day>, notes=<payer name, member_id, referral context>)
      5. draft_message(recipient=<family>, channel=<original channel>, body=<acknowledge; billing team will reach out to discuss coverage options>)

    Insurance result = unknown:
      3. create_task(assignee="billing", title="Verify payer — unrecognized insurance", due=<next business day>, notes=<payer and member_id as provided>)
      4. draft_message(recipient=<family>, channel=<original channel>, body=<acknowledge; we need to confirm coverage before scheduling>)

    If payer is absent:
      2. create_task(assignee="intake", title="Obtain insurance information", due=<next business day>, notes=<referral context>)
      3. draft_message(recipient=<family or referring physician>, channel=<original channel>, body=<request insurance payer and member ID to proceed>)

**STEP 5 — submit_triage_result**
Always call submit_triage_result exactly once to conclude. Populate ALL fields from your analysis and the tool results you received.

## Per-Tool Guidance (when each tool is warranted)

- **search_patient**: Use for new referrals and existing patient requests. Skip for missing_paperwork (no name/DOB to search), clinical_question (no intake present), safeguarding stops, and spam.
- **verify_insurance**: Call only when a payer name is present in the intake. Never call if payer is null or absent.
- **lookup_policy**: Call only when policy directly governs the decision — insurance dispute (out-of-network/expired result), clinical question refusal, safeguarding. Do not call speculatively.
- **find_slots**: Call only after insurance is confirmed in_network. Never call for out-of-network, expired, or unknown.
- **hold_slot**: Call only when find_slots returned at least one slot and insurance is in_network.
- **create_task**: Call for every item that requires a staff action. Assignee must be one of: front_desk, intake, billing, clinical_lead.
- **draft_message**: Call for every item warranting an outbound message. Use language="es" when the family communicates in Spanish. Skip for spam/P3 items with no reply needed.
- **escalate**: Call only for P0 (safeguarding) and P1 items that require escalation beyond a task.

## Language Access

- Detect Spanish by presence of Spanish words or sentences in the item body.
- Pass language="es" to find_slots when a Spanish-speaking family needs a provider.
- Pass language="es" to draft_message for Spanish-speaking families.
- If no Spanish-capable slot is found, create a waitlist task and explain the situation in Spanish in the draft.
- For other non-English languages: flag for human translation; do not draft in unsupported languages.

## Output Correctness

- draft_reply in submit_triage_result must be the body string of the most recent draft_message call, or null if none was made.
- task_ids must list the task_id values returned by each create_task call.
- escalation must mirror the reason and severity you passed to escalate(), or null if escalate was not called.
- requires_human_review must be true whenever there is a guardian mismatch, out-of-network insurance, expired insurance, escalation, or any safeguarding flag.
- decision_rationale must explain why each tool was called and how urgency was assigned — it is the audit narrative.`;

type ToolDef = Anthropic.Tool;

const TOOLS: ToolDef[] = [
  {
    name: "search_patient",
    description:
      "Search for an existing patient record by name and/or date of birth. Returns matching patient records with guardian names for mismatch detection.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Patient full name" },
        dob: {
          type: "string",
          description: "Date of birth in YYYY-MM-DD format",
        },
      },
    },
  },
  {
    name: "verify_insurance",
    description:
      "Verify insurance coverage for a given payer and member ID. Returns in_network, out_of_network, expired, or unknown status with copay and auth_required details.",
    input_schema: {
      type: "object",
      properties: {
        payer: { type: "string", description: "Insurance payer name" },
        member_id: { type: "string", description: "Insurance member ID" },
      },
    },
  },
  {
    name: "lookup_policy",
    description:
      "Retrieve policy snippets for a specific topic. Only call when policy directly shapes the decision.",
    input_schema: {
      type: "object",
      required: ["topic"],
      properties: {
        topic: {
          type: "string",
          enum: [
            "service_lines",
            "insurance",
            "safeguarding",
            "clinical_advice",
            "scheduling",
            "cancellation",
            "language_access",
          ],
          description: "Policy topic to retrieve",
        },
      },
    },
  },
  {
    name: "find_slots",
    description:
      "Find available appointment slots. Only call after insurance is confirmed in-network. Returns up to 5 matching slots.",
    input_schema: {
      type: "object",
      properties: {
        discipline: {
          type: "string",
          enum: ["SLP", "OT", "PT"],
          description: "Therapy discipline",
        },
        preferences: {
          type: "string",
          description: "Family scheduling preferences (e.g., after school)",
        },
        language: {
          type: "string",
          description:
            'Language preference for provider matching (e.g., "es" for Spanish)',
        },
      },
    },
  },
  {
    name: "hold_slot",
    description:
      "Place a pending-review hold on an appointment slot. Only call when find_slots returned at least one slot and insurance is in-network. Does NOT book the appointment.",
    input_schema: {
      type: "object",
      required: ["slot_id", "patient_ref"],
      properties: {
        slot_id: {
          type: "string",
          description: "Slot ID returned by find_slots",
        },
        patient_ref: {
          type: "string",
          description: "Patient name or reference",
        },
      },
    },
  },
  {
    name: "create_task",
    description:
      "Create a staff task. Call for every item that requires a human staff action.",
    input_schema: {
      type: "object",
      required: ["assignee", "title", "due", "notes"],
      properties: {
        assignee: {
          type: "string",
          enum: ["front_desk", "intake", "billing", "clinical_lead"],
          description: "Staff role responsible for this task",
        },
        title: { type: "string", description: "Short task title" },
        due: {
          type: "string",
          description: "Due date in YYYY-MM-DD format",
        },
        notes: {
          type: "string",
          description: "Task context and instructions for the assignee",
        },
      },
    },
  },
  {
    name: "draft_message",
    description:
      "Create a draft outbound message for staff review. Does NOT send the message. Call for every item warranting outbound communication.",
    input_schema: {
      type: "object",
      required: ["recipient", "channel", "body"],
      properties: {
        recipient: {
          type: "string",
          description: "Recipient name or identifier",
        },
        channel: {
          type: "string",
          enum: ["portal", "email", "phone"],
          description: "Communication channel",
        },
        body: {
          type: "string",
          description:
            "Draft message body. Must be empathetic, concise, and contain no clinical advice.",
        },
        language: {
          type: "string",
          enum: ["en", "es"],
          description: 'Language code. Use "es" for Spanish-speaking families.',
        },
      },
    },
  },
  {
    name: "escalate",
    description:
      "Escalate an item to clinical lead (P0) or flag for urgent staff attention (P1). Only call for safeguarding disclosures (P0) or same-day operational urgencies (P1).",
    input_schema: {
      type: "object",
      required: ["item_id", "reason", "severity"],
      properties: {
        item_id: { type: "string", description: "Inbox item ID" },
        reason: {
          type: "string",
          description:
            "Full escalation reason including all available intake context",
        },
        severity: {
          type: "string",
          enum: ["P0", "P1"],
          description: "Escalation severity",
        },
      },
    },
  },
  {
    name: "submit_triage_result",
    description:
      "Submit the completed triage result for this inbox item. Call exactly once when all tool work is done. This is the final action — the loop ends when you call this.",
    input_schema: {
      type: "object",
      required: [
        "classification",
        "urgency",
        "requires_human_review",
        "extracted_intake",
        "missing_info",
        "recommended_next_action",
        "draft_reply",
        "task_ids",
        "escalation",
        "decision_rationale",
      ],
      properties: {
        classification: {
          type: "string",
          enum: [
            "new_referral",
            "existing_patient_request",
            "scheduling",
            "clinical_question",
            "billing_question",
            "missing_paperwork",
            "provider_followup",
            "complaint",
            "safeguarding",
            "spam",
            "other",
          ],
        },
        urgency: { type: "string", enum: ["P0", "P1", "P2", "P3"] },
        requires_human_review: {
          type: "boolean",
          description:
            "ALWAYS true — every item requires human staff review before action. Never false.",
        },
        extracted_intake: {
          type: "object",
          required: [
            "child_name",
            "dob_or_age",
            "parent_contact",
            "discipline",
            "diagnosis_or_concern",
            "payer",
            "member_id",
          ],
          properties: {
            child_name: {
              description: "Child full name as extracted, or null",
            },
            dob_or_age: {
              description:
                "Date of birth (YYYY-MM-DD) or age description, or null",
            },
            parent_contact: {
              description: "Parent/guardian phone, email, or contact info, or null",
            },
            discipline: {
              description:
                "Array of disciplines from [SLP, OT, PT], or null if not specified",
            },
            diagnosis_or_concern: {
              description: "Clinical concern or diagnosis as stated, or null",
            },
            payer: { description: "Insurance payer name, or null" },
            member_id: { description: "Insurance member ID, or null" },
          },
        },
        missing_info: {
          type: "array",
          items: { type: "string" },
          description:
            "List of field names that are absent but needed (e.g., ['dob', 'payer'])",
        },
        recommended_next_action: {
          type: "string",
          description: "One sentence describing the primary next staff action",
        },
        draft_reply: {
          description:
            "Body of the most recent draft_message call, or null if no message was drafted",
        },
        task_ids: {
          type: "array",
          items: { type: "string" },
          description: "task_id values returned by each create_task call",
        },
        escalation: {
          description:
            "Object with {reason, severity} mirroring the escalate() call, or null if not called",
        },
        decision_rationale: {
          type: "string",
          description:
            "Audit narrative explaining why each tool was called and how urgency was assigned",
        },
      },
    },
  },
];

async function dispatchTool(
  name: string,
  args: Record<string, unknown>,
): Promise<unknown> {
  switch (name) {
    case "search_patient":
      return search_patient(
        args as Parameters<typeof search_patient>[0],
      );
    case "verify_insurance":
      return verify_insurance(
        args as Parameters<typeof verify_insurance>[0],
      );
    case "lookup_policy":
      return lookup_policy(
        args as Parameters<typeof lookup_policy>[0],
      );
    case "find_slots":
      return find_slots(args as Parameters<typeof find_slots>[0]);
    case "hold_slot":
      return hold_slot(args as Parameters<typeof hold_slot>[0]);
    case "create_task":
      return create_task(args as Parameters<typeof create_task>[0]);
    case "draft_message":
      return draft_message(
        args as Parameters<typeof draft_message>[0],
      );
    case "escalate":
      return escalate(args as Parameters<typeof escalate>[0]);
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

function buildItemOutput(
  item: InboxItem,
  args: Record<string, unknown>,
): ItemOutput {
  const rawIntake = (args.extracted_intake ?? {}) as Record<string, unknown>;
  const extracted: ExtractedIntake = {
    child_name: (rawIntake.child_name as string | null) ?? null,
    dob_or_age: (rawIntake.dob_or_age as string | null) ?? null,
    parent_contact: (rawIntake.parent_contact as string | null) ?? null,
    discipline:
      (rawIntake.discipline as ExtractedIntake["discipline"]) ?? null,
    diagnosis_or_concern:
      (rawIntake.diagnosis_or_concern as string | null) ?? null,
    payer: (rawIntake.payer as string | null) ?? null,
    member_id: (rawIntake.member_id as string | null) ?? null,
  };

  const escalationRaw = args.escalation as
    | { reason: string; severity: "P0" | "P1" }
    | null
    | undefined;

  return {
    item_id: item.id,
    classification:
      (args.classification as ItemOutput["classification"]) ?? "other",
    urgency: (args.urgency as ItemOutput["urgency"]) ?? "P2",
    requires_human_review: true,
    extracted_intake: extracted,
    missing_info: (args.missing_info as string[]) ?? [],
    tools_called: getToolCallsForItem(item.id),
    recommended_next_action:
      (args.recommended_next_action as string) ?? "Human review required",
    draft_reply: (args.draft_reply as string | null) ?? null,
    task_ids: (args.task_ids as string[]) ?? [],
    escalation:
      escalationRaw != null
        ? { reason: escalationRaw.reason, severity: escalationRaw.severity }
        : null,
    decision_rationale:
      (args.decision_rationale as string) ?? "Triage completed",
  };
}

function makeFallback(item: InboxItem, error: unknown): ItemOutput {
  const message = error instanceof Error ? error.message : String(error);
  return {
    item_id: item.id,
    classification: "other",
    urgency: "P2",
    requires_human_review: true,
    extracted_intake: {
      child_name: null,
      dob_or_age: null,
      parent_contact: null,
      discipline: null,
      diagnosis_or_concern: null,
      payer: null,
      member_id: null,
    },
    missing_info: [],
    tools_called: getToolCallsForItem(item.id),
    recommended_next_action: "Human review required due to processing failure",
    draft_reply: null,
    task_ids: [],
    escalation: null,
    decision_rationale: `Agent processing failed: ${message}`,
  };
}

function formatItemMessage(item: InboxItem): string {
  return [
    `Triage the following Cedar Kids Therapy inbox item.`,
    ``,
    `Item ID: ${item.id}`,
    `Channel: ${item.channel}`,
    `Received: ${item.received_at}`,
    `Sender: ${item.sender}`,
    `Subject: ${item.subject}`,
    `Body:\n${item.body}`,
    `Attachments: ${item.attachments.length > 0 ? item.attachments.join(", ") : "None"}`,
    ``,
    `Follow the decision flowchart in your instructions. When all tool work is done, call submit_triage_result.`,
  ].join("\n");
}

async function processItem(item: InboxItem): Promise<ItemOutput> {
  return withItemContext(item.id, async () => {
    process.stderr.write(`[${item.id}] starting — ${item.subject}\n`);

    const messages: Anthropic.MessageParam[] = [
      { role: "user", content: formatItemMessage(item) },
    ];

    for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
      process.stderr.write(`[${item.id}] calling Claude (iteration ${iteration + 1}/${MAX_ITERATIONS})...\n`);

      const response = await client.messages.create({
        model: MODEL,
        max_tokens: 8192,
        system: SYSTEM_PROMPT,
        tools: TOOLS,
        messages,
      });

      // Add assistant turn to conversation
      messages.push({ role: "assistant", content: response.content });

      if (response.stop_reason !== "tool_use") {
        // Claude finished without calling a tool — unexpected; emit fallback
        process.stderr.write(`[${item.id}] stopped without tool call (stop_reason=${response.stop_reason}) — emitting fallback\n`);
        break;
      }

      const toolUseBlocks = response.content.filter(
        (b): b is Anthropic.ToolUseBlock => b.type === "tool_use",
      );

      // Collect tool results; check for sentinel
      const toolResultContent: Anthropic.ToolResultBlockParam[] = [];
      let sentinelArgs: Record<string, unknown> | null = null;

      for (const block of toolUseBlocks) {
        if (block.name === "submit_triage_result") {
          sentinelArgs = block.input as Record<string, unknown>;
          // Provide a confirmation result so the conversation is well-formed
          toolResultContent.push({
            type: "tool_result",
            tool_use_id: block.id,
            content: "Triage result recorded.",
          });
        } else {
          process.stderr.write(`[${item.id}]   → ${block.name}(${JSON.stringify(block.input)})\n`);
          try {
            const result = await dispatchTool(
              block.name,
              block.input as Record<string, unknown>,
            );
            const summary = (result as { result_summary?: string }).result_summary ?? "ok";
            process.stderr.write(`[${item.id}]   ← ${block.name}: ${summary}\n`);
            toolResultContent.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: JSON.stringify(result),
            });
          } catch (err) {
            process.stderr.write(`[${item.id}]   ✗ ${block.name} error: ${err instanceof Error ? err.message : String(err)}\n`);
            toolResultContent.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: `Error executing ${block.name}: ${err instanceof Error ? err.message : String(err)}`,
              is_error: true,
            });
          }
        }
      }

      // If sentinel was called, return the structured output now
      if (sentinelArgs !== null) {
        const out = buildItemOutput(item, sentinelArgs);
        process.stderr.write(`[${item.id}] done — ${out.classification} ${out.urgency}\n`);
        return out;
      }

      // Feed all tool results back and continue the loop
      messages.push({ role: "user", content: toolResultContent });
    }

    process.stderr.write(`[${item.id}] max iterations reached — emitting fallback\n`);
    return makeFallback(
      item,
      new Error(
        `ReAct loop ended after ${MAX_ITERATIONS} iterations without submit_triage_result`,
      ),
    );
  });
}

export async function runAgent(inbox: InboxItem[]): Promise<ItemOutput[]> {
  const results: ItemOutput[] = [];

  for (const item of inbox) {
    try {
      const output = await processItem(item);
      results.push(output);
    } catch (error) {
      results.push(makeFallback(item, error));
    }
  }

  return results;
}
