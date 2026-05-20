/**
 * {@link SessionsTool} — bundles `sessions_list`, `sessions_history`,
 * `sessions_send`, and `sessions_spawn` over a {@link SessionBroker}.
 *
 * Equivalent to Rust's `brainwires_tools::sessions::sessions_tool` module.
 */

import {
  objectSchema,
  type Tool,
  type ToolContext,
  ToolResult,
} from "@brainwires/core";

import {
  type SessionBroker,
  SessionId,
  type SpawnRequest,
  defaultSpawnRequest,
} from "./broker.ts";

export const TOOL_SESSIONS_LIST = "sessions_list";
export const TOOL_SESSIONS_HISTORY = "sessions_history";
export const TOOL_SESSIONS_SEND = "sessions_send";
export const TOOL_SESSIONS_SPAWN = "sessions_spawn";

/**
 * Metadata key the host may set on ToolContext.metadata to carry the
 * caller's own session id.
 */
export const CTX_METADATA_SESSION_ID = "session_id";

/** Max messages `sessions_history` returns in a single call. */
export const MAX_HISTORY_LIMIT = 500;
const DEFAULT_HISTORY_LIMIT = 50;

/** Bundle of four session-control tools, all backed by a single SessionBroker. */
export class SessionsTool {
  private readonly broker: SessionBroker;
  private readonly current_session_id: SessionId | null;

  constructor(broker: SessionBroker, current_session_id: SessionId | null) {
    this.broker = broker;
    this.current_session_id = current_session_id;
  }

  /** Return the four tool definitions this bundle exposes to the LLM. */
  static getTools(): Tool[] {
    return [
      SessionsTool.listTool(),
      SessionsTool.historyTool(),
      SessionsTool.sendTool(),
      SessionsTool.spawnTool(),
    ];
  }

  // ── Tool schemas ─────────────────────────────────────────────────────────

  private static listTool(): Tool {
    return {
      name: TOOL_SESSIONS_LIST,
      description:
        "List every live chat session currently managed by the host — including the " +
        "caller's own session and any sessions the caller (or its peers) have spawned. " +
        "Use this to discover session ids before calling sessions_history or sessions_send. " +
        "Returns a JSON array of session summaries (id, channel, peer, timestamps, " +
        "message_count, optional parent).",
      input_schema: objectSchema({}, []),
      requires_approval: false,
    };
  }

  private static historyTool(): Tool {
    return {
      name: TOOL_SESSIONS_HISTORY,
      description:
        "Return a target session's recent transcript as a JSON array of " +
        "{role, content, timestamp} objects (newest last). Use this to catch up " +
        "on what a spawned sub-session has produced, or to read another user's " +
        "ongoing conversation before intervening.",
      input_schema: objectSchema({
        session_id: {
          type: "string",
          description: "The target session id (from sessions_list).",
        },
        limit: {
          type: "number",
          description:
            `Max messages to return (default ${DEFAULT_HISTORY_LIMIT}, hard-capped at ${MAX_HISTORY_LIMIT}).`,
        },
      }, ["session_id"]),
      requires_approval: false,
    };
  }

  private static sendTool(): Tool {
    return {
      name: TOOL_SESSIONS_SEND,
      description:
        "Inject a user-role message into another session's inbound queue. Fire-and-forget: " +
        `returns {"ok": true} as soon as the message is queued; the target session ` +
        "processes it asynchronously. Use this to nudge a spawned sub-session, relay " +
        "information between two user sessions, or ask a peer session a follow-up " +
        "question.",
      input_schema: objectSchema({
        session_id: {
          type: "string",
          description:
            "Target session id. Must not equal the caller's own session (self-send is rejected to prevent recursion).",
        },
        text: {
          type: "string",
          description:
            "The user-role message to inject into the target session's inbound queue.",
        },
      }, ["session_id", "text"]),
      requires_approval: true,
    };
  }

  private static spawnTool(): Tool {
    return {
      name: TOOL_SESSIONS_SPAWN,
      description:
        "Spawn a new chat sub-session as a child of the current session, seeded with " +
        "`prompt`. Returns {session_id, first_reply?}. Use this to delegate a focused " +
        "task (e.g. 'spawn a research sub-session and return in 5m') — the parent can " +
        "later inspect progress via sessions_history or push updates via sessions_send.",
      input_schema: objectSchema({
        prompt: {
          type: "string",
          description: "Initial user message to seed the new session with.",
        },
        model: {
          type: "string",
          description:
            "Optional model override (e.g. 'claude-opus-4-7'). Omit to inherit from parent.",
        },
        system: {
          type: "string",
          description:
            "Optional system prompt for the sub-session. Omit to inherit.",
        },
        tools: {
          type: "array",
          items: { type: "string" },
          description:
            "Optional allow-list of tool names the sub-session may invoke. Omit to inherit the parent's toolset.",
        },
        wait_for_first_reply: {
          type: "boolean",
          description:
            "If true, block this tool call until the sub-session produces its first assistant message (or wait_timeout_secs elapses). Default false — return immediately with just the session id.",
          default: false,
        },
        wait_timeout_secs: {
          type: "number",
          description:
            "Seconds to wait when wait_for_first_reply is true (default 60).",
          default: 60,
        },
      }, ["prompt"]),
      requires_approval: true,
    };
  }

  // ── Execution ────────────────────────────────────────────────────────────

  /** Dispatch a tool call by name. Never throws — broker errors become ToolResult.error. */
  async execute(
    tool_use_id: string,
    tool_name: string,
    input: Record<string, unknown>,
    context: ToolContext,
  ): Promise<ToolResult> {
    switch (tool_name) {
      case TOOL_SESSIONS_LIST:
        return this.execList(tool_use_id);
      case TOOL_SESSIONS_HISTORY:
        return this.execHistory(tool_use_id, input);
      case TOOL_SESSIONS_SEND:
        return this.execSend(tool_use_id, input, context);
      case TOOL_SESSIONS_SPAWN:
        return this.execSpawn(tool_use_id, input, context);
      default:
        return ToolResult.error(
          tool_use_id,
          `Unknown sessions tool: ${tool_name}`,
        );
    }
  }

  private async execList(tool_use_id: string): Promise<ToolResult> {
    try {
      const summaries = await this.broker.list();
      // Convert SessionId class instances to their underlying strings for JSON output.
      const plain = summaries.map((s) => ({
        id: s.id.value,
        channel: s.channel,
        peer: s.peer,
        created_at: s.created_at,
        last_active: s.last_active,
        message_count: s.message_count,
        parent: s.parent ? s.parent.value : null,
      }));
      return ToolResult.success(tool_use_id, JSON.stringify(plain));
    } catch (e) {
      return ToolResult.error(
        tool_use_id,
        `sessions_list failed: ${(e as Error).message}`,
      );
    }
  }

  private async execHistory(
    tool_use_id: string,
    input: Record<string, unknown>,
  ): Promise<ToolResult> {
    const sidRaw = input.session_id;
    if (typeof sidRaw !== "string" || sidRaw.length === 0) {
      return ToolResult.error(
        tool_use_id,
        "sessions_history requires a non-empty `session_id`",
      );
    }
    const sid = new SessionId(sidRaw);

    const limitIn = input.limit;
    const limit = typeof limitIn === "number"
      ? Math.min(limitIn, MAX_HISTORY_LIMIT)
      : Math.min(DEFAULT_HISTORY_LIMIT, MAX_HISTORY_LIMIT);

    try {
      const msgs = await this.broker.history(sid, limit);
      return ToolResult.success(tool_use_id, JSON.stringify(msgs));
    } catch (e) {
      return ToolResult.error(
        tool_use_id,
        `sessions_history failed: ${(e as Error).message}`,
      );
    }
  }

  private async execSend(
    tool_use_id: string,
    input: Record<string, unknown>,
    context: ToolContext,
  ): Promise<ToolResult> {
    const sidRaw = input.session_id;
    if (typeof sidRaw !== "string" || sidRaw.length === 0) {
      return ToolResult.error(
        tool_use_id,
        "sessions_send requires a non-empty `session_id`",
      );
    }
    const textRaw = input.text;
    if (typeof textRaw !== "string" || textRaw.length === 0) {
      return ToolResult.error(
        tool_use_id,
        "sessions_send requires a non-empty `text`",
      );
    }
    const sid = new SessionId(sidRaw);
    const selfId = this.resolveCurrentSessionId(context);
    if (selfId && selfId.equals(sid)) {
      return ToolResult.error(
        tool_use_id,
        "sessions_send cannot target the caller's own session — that would recurse. " +
          "Use a spawned sub-session id, or address a peer session from sessions_list.",
      );
    }

    try {
      await this.broker.send(sid, textRaw);
      return ToolResult.success(tool_use_id, JSON.stringify({ ok: true }));
    } catch (e) {
      return ToolResult.error(
        tool_use_id,
        `sessions_send failed: ${(e as Error).message}`,
      );
    }
  }

  private async execSpawn(
    tool_use_id: string,
    input: Record<string, unknown>,
    context: ToolContext,
  ): Promise<ToolResult> {
    const prompt = input.prompt;
    if (typeof prompt !== "string" || prompt.length === 0) {
      return ToolResult.error(
        tool_use_id,
        "sessions_spawn requires a non-empty `prompt`",
      );
    }
    const parent = this.resolveCurrentSessionId(context);
    if (!parent) {
      return ToolResult.error(
        tool_use_id,
        `sessions_spawn could not determine the caller's session id — ` +
          `host must set ToolContext.metadata["session_id"] or pass ` +
          `current_session_id into SessionsTool constructor.`,
      );
    }

    const req: SpawnRequest = {
      ...defaultSpawnRequest(),
      prompt,
      model: typeof input.model === "string" ? input.model : null,
      system: typeof input.system === "string" ? input.system : null,
      tools: Array.isArray(input.tools)
        ? input.tools.filter((x): x is string => typeof x === "string")
        : null,
      wait_for_first_reply: input.wait_for_first_reply === true,
      wait_timeout_secs: typeof input.wait_timeout_secs === "number"
        ? input.wait_timeout_secs
        : 60,
    };

    try {
      const spawned = await this.broker.spawn(parent, req);
      const plain = {
        id: spawned.id.value,
        first_reply: spawned.first_reply,
      };
      return ToolResult.success(tool_use_id, JSON.stringify(plain));
    } catch (e) {
      return ToolResult.error(
        tool_use_id,
        `sessions_spawn failed: ${(e as Error).message}`,
      );
    }
  }

  private resolveCurrentSessionId(context: ToolContext): SessionId | null {
    const raw = context.metadata[CTX_METADATA_SESSION_ID];
    if (raw && raw.length > 0) return new SessionId(raw);
    return this.current_session_id;
  }
}
