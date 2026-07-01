// deno-lint-ignore-file no-explicit-any
/**
 * AWS Bedrock chat provider implementation.
 * Uses SigV4 signing via Web Crypto API (zero external dependencies).
 * Wraps the Anthropic Messages API format for Bedrock endpoints.
 * Equivalent to Rust's `anthropic/bedrock.rs` + Bedrock-specific auth.
 */

import {
  type ChatOptions,
  type ChatResponse,
  type ContentBlock,
  Message,
  type MessageContent,
  type Provider,
  type StreamChunk,
  type Tool,
  type Usage,
} from "@rullama/core";
import { parseSSEStream } from "./sse.ts";

const ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31";

// ---------------------------------------------------------------------------
// AWS SigV4 signer — minimal implementation using Web Crypto API
// ---------------------------------------------------------------------------

interface AwsCredentials {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken?: string;
}

/** HMAC-SHA256 using Web Crypto. */
async function hmacSha256(key: Uint8Array, data: string): Promise<Uint8Array> {
  const cryptoKey = await crypto.subtle.importKey(
    "raw",
    key.buffer as ArrayBuffer,
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign(
    "HMAC",
    cryptoKey,
    new TextEncoder().encode(data),
  );
  return new Uint8Array(sig);
}

/** SHA-256 hash of a string. */
async function sha256(data: string): Promise<string> {
  const hash = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(data),
  );
  return hexEncode(new Uint8Array(hash));
}

/** Hex-encode a Uint8Array. */
function hexEncode(bytes: Uint8Array): string {
  return Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Format a Date as YYYYMMDD'T'HHMMSS'Z'. */
function toAmzDate(date: Date): string {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}/, "");
}

/** Format a Date as YYYYMMDD. */
function toDateStamp(date: Date): string {
  return toAmzDate(date).slice(0, 8);
}

/** Derive the SigV4 signing key. */
async function deriveSigningKey(
  secretKey: string,
  dateStamp: string,
  region: string,
  service: string,
): Promise<Uint8Array> {
  const encoder = new TextEncoder();
  let key = await hmacSha256(encoder.encode("AWS4" + secretKey), dateStamp);
  key = await hmacSha256(key, region);
  key = await hmacSha256(key, service);
  key = await hmacSha256(key, "aws4_request");
  return key;
}

/** Sign headers for a request using AWS SigV4. Returns headers to add. */
async function signRequest(
  method: string,
  url: string,
  headers: Record<string, string>,
  body: string,
  credentials: AwsCredentials,
  region: string,
  service: string,
  date?: Date,
): Promise<Record<string, string>> {
  const now = date ?? new Date();
  const amzDate = toAmzDate(now);
  const dateStamp = toDateStamp(now);

  const parsedUrl = new URL(url);
  const canonicalUri = parsedUrl.pathname;
  const canonicalQuerystring = parsedUrl.searchParams.toString();
  const host = parsedUrl.host;

  // Build the headers to sign
  const headersToSign: Record<string, string> = {
    ...headers,
    host,
    "x-amz-date": amzDate,
  };

  if (credentials.sessionToken) {
    headersToSign["x-amz-security-token"] = credentials.sessionToken;
  }

  // Sort header names
  const signedHeaderNames = Object.keys(headersToSign)
    .map((k) => k.toLowerCase())
    .sort();
  const signedHeaders = signedHeaderNames.join(";");

  // Canonical headers
  const canonicalHeaders = signedHeaderNames
    .map((name) => {
      const value = headersToSign[
        Object.keys(headersToSign).find((k) => k.toLowerCase() === name)!
      ];
      return `${name}:${value.trim()}\n`;
    })
    .join("");

  const payloadHash = await sha256(body);

  const canonicalRequest = [
    method,
    canonicalUri,
    canonicalQuerystring,
    canonicalHeaders,
    signedHeaders,
    payloadHash,
  ].join("\n");

  const credentialScope = `${dateStamp}/${region}/${service}/aws4_request`;
  const stringToSign = [
    "AWS4-HMAC-SHA256",
    amzDate,
    credentialScope,
    await sha256(canonicalRequest),
  ].join("\n");

  const signingKey = await deriveSigningKey(
    credentials.secretAccessKey,
    dateStamp,
    region,
    service,
  );
  const signature = hexEncode(await hmacSha256(signingKey, stringToSign));

  const authorizationHeader =
    `AWS4-HMAC-SHA256 Credential=${credentials.accessKeyId}/${credentialScope}, ` +
    `SignedHeaders=${signedHeaders}, Signature=${signature}`;

  const result: Record<string, string> = {
    "x-amz-date": amzDate,
    "Authorization": authorizationHeader,
  };

  if (credentials.sessionToken) {
    result["x-amz-security-token"] = credentials.sessionToken;
  }

  return result;
}

// ---------------------------------------------------------------------------
// Bedrock wire types (Anthropic Messages format)
// ---------------------------------------------------------------------------

interface BedrockContentBlock {
  type: string;
  text?: string;
  id?: string;
  name?: string;
  input?: any;
  tool_use_id?: string;
  content?: string;
}

interface BedrockResponse {
  content: BedrockContentBlock[];
  stop_reason: string;
  usage: { input_tokens: number; output_tokens: number };
}

interface BedrockStreamEvent {
  type: string;
  delta?: { text?: string; stop_reason?: string };
  usage?: { input_tokens?: number; output_tokens?: number };
  content_block?: BedrockContentBlock;
  index?: number;
}

// ---------------------------------------------------------------------------
// Conversion helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Convert core Messages to Bedrock/Anthropic wire format. */
export function convertMessages(
  messages: Message[],
): { role: string; content: BedrockContentBlock[] }[] {
  return messages
    .filter((m) => m.role !== "system")
    .map((m) => {
      const role = m.role === "assistant" ? "assistant" : "user";
      let content: BedrockContentBlock[];

      if (typeof m.content === "string") {
        content = [{ type: "text", text: m.content }];
      } else {
        content = m.content
          .map((block): BedrockContentBlock | null => {
            switch (block.type) {
              case "text":
                return { type: "text", text: block.text };
              case "tool_use":
                return {
                  type: "tool_use",
                  id: block.id,
                  name: block.name,
                  input: block.input,
                };
              case "tool_result":
                return {
                  type: "tool_result",
                  tool_use_id: block.tool_use_id,
                  content: block.content,
                };
              default:
                return null;
            }
          })
          .filter((b): b is BedrockContentBlock => b !== null);
      }

      return { role, content };
    });
}

/** Convert core Tools to Bedrock/Anthropic wire format. */
export function convertTools(
  tools: Tool[],
): { name: string; description: string; input_schema: any }[] {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.input_schema.properties ?? {},
  }));
}

/** Extract the system message text. */
export function getSystemMessage(messages: Message[]): string | undefined {
  const sys = messages.find((m) => m.role === "system");
  if (!sys) return undefined;
  return typeof sys.content === "string" ? sys.content : undefined;
}

/** Parse a Bedrock response into a core ChatResponse. */
export function parseBedrockResponse(
  response: BedrockResponse,
): ChatResponse {
  let content: MessageContent;

  if (response.content.length === 1 && response.content[0].type === "text") {
    content = response.content[0].text ?? "";
  } else {
    content = response.content
      .map((block): ContentBlock | null => {
        switch (block.type) {
          case "text":
            return { type: "text", text: block.text ?? "" };
          case "tool_use":
            return {
              type: "tool_use",
              id: block.id ?? "",
              name: block.name ?? "",
              input: block.input ?? {},
            };
          default:
            return null;
        }
      })
      .filter((b): b is ContentBlock => b !== null);
  }

  const usage: Usage = {
    prompt_tokens: response.usage.input_tokens,
    completion_tokens: response.usage.output_tokens,
    total_tokens: response.usage.input_tokens + response.usage.output_tokens,
  };

  return {
    message: new Message({ role: "assistant", content }),
    usage,
    finish_reason: response.stop_reason,
  };
}

/** Build the Bedrock invoke URL. */
export function bedrockInvokeUrl(region: string, modelId: string): string {
  return `https://bedrock-runtime.${region}.amazonaws.com/model/${modelId}/invoke`;
}

/** Build the Bedrock streaming invoke URL. */
export function bedrockStreamUrl(region: string, modelId: string): string {
  return `https://bedrock-runtime.${region}.amazonaws.com/model/${modelId}/invoke-with-response-stream`;
}

// ---------------------------------------------------------------------------
// BedrockProvider
// ---------------------------------------------------------------------------

/** Chat provider for AWS Bedrock (Anthropic Claude models).
 * Uses SigV4 signing — no AWS SDK dependency.
 * Equivalent to Rust's `BedrockAuth` + `AnthropicChatProvider` with Bedrock auth. */
export class BedrockProvider implements Provider {
  readonly name: string;
  private readonly region: string;
  private readonly model: string;
  private readonly credentials: AwsCredentials;

  constructor(
    region: string,
    model: string,
    credentials: AwsCredentials,
    providerName?: string,
  ) {
    this.region = region;
    this.model = model;
    this.credentials = credentials;
    this.name = providerName ?? "bedrock";
  }

  /** Create from environment variables. */
  static fromEnvironment(
    model: string,
    regionOverride?: string,
  ): BedrockProvider {
    const accessKeyId = Deno.env.get("AWS_ACCESS_KEY_ID");
    if (!accessKeyId) {
      throw new Error(
        "AWS_ACCESS_KEY_ID not set. Configure AWS credentials for Bedrock access.",
      );
    }
    const secretAccessKey = Deno.env.get("AWS_SECRET_ACCESS_KEY");
    if (!secretAccessKey) {
      throw new Error(
        "AWS_SECRET_ACCESS_KEY not set. Configure AWS credentials for Bedrock access.",
      );
    }
    const sessionToken = Deno.env.get("AWS_SESSION_TOKEN");
    const region = regionOverride ??
      Deno.env.get("AWS_DEFAULT_REGION") ??
      "us-east-1";

    return new BedrockProvider(region, model, {
      accessKeyId,
      secretAccessKey,
      sessionToken,
    });
  }

  // -----------------------------------------------------------------------
  // Provider interface
  // -----------------------------------------------------------------------

  async chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse> {
    const body = this.buildRequestBody(messages, tools, options);
    const url = bedrockInvokeUrl(this.region, this.model);

    const baseHeaders: Record<string, string> = {
      "Content-Type": "application/json",
      "anthropic_version": ANTHROPIC_BEDROCK_VERSION,
    };

    const bodyStr = JSON.stringify(body);
    const sigHeaders = await signRequest(
      "POST",
      url,
      baseHeaders,
      bodyStr,
      this.credentials,
      this.region,
      "bedrock",
    );

    const response = await fetch(url, {
      method: "POST",
      headers: { ...baseHeaders, ...sigHeaders },
      body: bodyStr,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Bedrock API error (${response.status}): ${errorText}`,
      );
    }

    const bedrockResponse: BedrockResponse = await response.json();
    return parseBedrockResponse(bedrockResponse);
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const body = this.buildRequestBody(messages, tools, options);
    const url = bedrockStreamUrl(this.region, this.model);

    const baseHeaders: Record<string, string> = {
      "Content-Type": "application/json",
      "anthropic_version": ANTHROPIC_BEDROCK_VERSION,
    };

    const bodyStr = JSON.stringify(body);
    const sigHeaders = await signRequest(
      "POST",
      url,
      baseHeaders,
      bodyStr,
      this.credentials,
      this.region,
      "bedrock",
    );

    const response = await fetch(url, {
      method: "POST",
      headers: { ...baseHeaders, ...sigHeaders },
      body: bodyStr,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Bedrock API error (${response.status}): ${errorText}`,
      );
    }

    if (!response.body) {
      throw new Error("Bedrock streaming response has no body");
    }

    for await (const data of parseSSEStream(response.body)) {
      let event: BedrockStreamEvent;
      try {
        event = JSON.parse(data);
      } catch {
        continue;
      }

      switch (event.type) {
        case "content_block_delta":
          if (event.delta?.text) {
            yield { type: "text", text: event.delta.text };
          }
          break;
        case "message_delta":
          if (event.usage) {
            yield {
              type: "usage",
              usage: {
                prompt_tokens: 0,
                completion_tokens: event.usage.output_tokens ?? 0,
                total_tokens: event.usage.output_tokens ?? 0,
              },
            };
          }
          break;
        case "message_stop":
          yield { type: "done" };
          break;
      }
    }
  }

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

  private buildRequestBody(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Record<string, any> {
    const bedrockMessages = convertMessages(messages);
    const system = options.system ?? getSystemMessage(messages);

    const body: Record<string, any> = {
      anthropic_version: ANTHROPIC_BEDROCK_VERSION,
      messages: bedrockMessages,
      max_tokens: options.max_tokens ?? 4096,
    };

    if (system) body.system = system;
    if (options.temperature !== undefined) {
      body.temperature = options.temperature;
    }
    if (options.top_p !== undefined) body.top_p = options.top_p;
    if (tools && tools.length > 0) {
      body.tools = convertTools(tools);
    }

    return body;
  }
}

// Export SigV4 helpers for testing
export {
  type AwsCredentials,
  deriveSigningKey,
  hexEncode,
  hmacSha256,
  sha256,
  signRequest,
  toAmzDate,
  toDateStamp,
};
