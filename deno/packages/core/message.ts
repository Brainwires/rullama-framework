// deno-lint-ignore-file no-explicit-any

/** Role of the message sender.
 * Equivalent to Rust's `Role` in rullama-core. */
export type Role = "user" | "assistant" | "system" | "tool";

/** Message content — simple text string or array of content blocks.
 * Equivalent to Rust's `MessageContent` (serde untagged). */
export type MessageContent = string | ContentBlock[];

/** Content block for structured messages.
 * Equivalent to Rust's `ContentBlock` (serde tag="type", rename_all="snake_case"). */
export type ContentBlock =
  | TextBlock
  | ImageBlock
  | ToolUseBlock
  | ToolResultBlock;

/** Text content block. */
export interface TextBlock {
  type: "text";
  text: string;
}

/** Image content block (base64 encoded). */
export interface ImageBlock {
  type: "image";
  source: ImageSource;
}

/** Tool use request block. */
export interface ToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input: any;
}

/** Tool result block. */
export interface ToolResultBlock {
  type: "tool_result";
  tool_use_id: string;
  content: string;
  is_error?: boolean;
}

/** Image source for image content blocks.
 * Equivalent to Rust's `ImageSource` (serde tag="type", rename_all="snake_case"). */
export interface ImageSource {
  type: "base64";
  media_type: string;
  data: string;
}

/** A message in the conversation.
 * Equivalent to Rust's `Message` in rullama-core. */
export interface MessageData {
  role: Role;
  content: MessageContent;
  name?: string;
  metadata?: any;
}

/** A message in the conversation with helper methods.
 * Equivalent to Rust's `Message` in rullama-core. */
export class Message implements MessageData {
  role: Role;
  content: MessageContent;
  name?: string;
  metadata?: any;

  constructor(data: MessageData) {
    this.role = data.role;
    this.content = data.content;
    this.name = data.name;
    this.metadata = data.metadata;
  }

  /** Create a new user message. */
  static user(content: string): Message {
    return new Message({ role: "user", content });
  }

  /** Create a new assistant message. */
  static assistant(content: string): Message {
    return new Message({ role: "assistant", content });
  }

  /** Create a new system message. */
  static system(content: string): Message {
    return new Message({ role: "system", content });
  }

  /** Create a tool result message. */
  static toolResult(toolUseId: string, content: string): Message {
    return new Message({
      role: "tool",
      content: [{ type: "tool_result", tool_use_id: toolUseId, content }],
    });
  }

  /** Get the text content if this is a simple text message. */
  text(): string | undefined {
    return typeof this.content === "string" ? this.content : undefined;
  }

  /** Get a text representation of the message content, including Blocks. */
  textOrSummary(): string {
    if (typeof this.content === "string") return this.content;
    const parts: string[] = [];
    for (const block of this.content) {
      switch (block.type) {
        case "text":
          parts.push(block.text);
          break;
        case "tool_use":
          parts.push(
            `[Called tool: ${block.name} with args: ${
              JSON.stringify(block.input)
            }]`,
          );
          break;
        case "tool_result":
          if (block.is_error) {
            parts.push(`[Tool error: ${block.content}]`);
          } else {
            parts.push(`[Tool result: ${block.content}]`);
          }
          break;
        case "image":
          parts.push("[Image]");
          break;
      }
    }
    return parts.join("\n");
  }

  /** Serialize to a plain JSON-compatible object (omits undefined fields). */
  toJSON(): Record<string, any> {
    const obj: Record<string, any> = { role: this.role, content: this.content };
    if (this.name !== undefined) obj.name = this.name;
    if (this.metadata !== undefined) obj.metadata = this.metadata;
    return obj;
  }
}

/** Usage statistics for a chat completion.
 * Equivalent to Rust's `Usage` in rullama-core. */
export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** Create a new Usage. */
export function createUsage(
  promptTokens: number,
  completionTokens: number,
): Usage {
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  };
}

/** Response from a chat completion.
 * Equivalent to Rust's `ChatResponse` in rullama-core. */
export interface ChatResponse {
  message: Message;
  usage: Usage;
  finish_reason?: string;
}

/** Streaming chunk from a chat completion.
 * Equivalent to Rust's `StreamChunk` in rullama-core. */
export type StreamChunk =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string }
  | { type: "tool_input_delta"; id: string; partial_json: string }
  | {
    type: "tool_call";
    call_id: string;
    response_id: string;
    chat_id?: string;
    tool_name: string;
    server: string;
    parameters: any;
  }
  | { type: "usage"; usage: Usage }
  | { type: "done" };

/** Serialize messages into the STATELESS protocol format for conversation history.
 * Equivalent to Rust's `serialize_messages_to_stateless_history` in rullama-core. */
export function serializeMessagesToStatelessHistory(
  messages: Message[],
): Record<string, any>[] {
  const history: Record<string, any>[] = [];

  for (const msg of messages) {
    if (msg.role === "system") continue;

    const roleStr = msg.role;

    if (typeof msg.content === "string") {
      if (msg.role === "assistant" && msg.content.trim() === "") continue;
      history.push({ role: roleStr, content: msg.content });
    } else {
      const textParts: string[] = [];
      for (const block of msg.content) {
        switch (block.type) {
          case "text":
            textParts.push(block.text);
            break;
          case "tool_use": {
            if (textParts.length > 0) {
              const combined = textParts.join("\n");
              if (!(msg.role === "assistant" && combined.trim() === "")) {
                history.push({ role: roleStr, content: combined });
              }
              textParts.length = 0;
            }
            history.push({
              role: "function_call",
              call_id: block.id,
              name: block.name,
              arguments: JSON.stringify(block.input),
            });
            break;
          }
          case "tool_result": {
            if (textParts.length > 0) {
              const combined = textParts.join("\n");
              if (!(msg.role === "assistant" && combined.trim() === "")) {
                history.push({ role: roleStr, content: combined });
              }
              textParts.length = 0;
            }
            history.push({
              role: "tool",
              call_id: block.tool_use_id,
              content: block.content,
            });
            break;
          }
          case "image":
            break;
        }
      }
      if (textParts.length > 0) {
        const combined = textParts.join("\n");
        if (!(msg.role === "assistant" && combined.trim() === "")) {
          history.push({ role: roleStr, content: combined });
        }
      }
    }
  }

  return history;
}
