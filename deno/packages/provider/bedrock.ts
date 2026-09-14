// deno-lint-ignore-file no-explicit-any
/**
 * AWS Bedrock chat provider implementation.
 * Uses SigV4 signing via Web Crypto API (zero external dependencies).
 * Wraps the Anthropic Messages API format for Bedrock endpoints.
 * Equivalent to Rust's `anthropic/bedrock.rs` + Bedrock-specific auth.
 */

import type {
  ChatOptions,
  ChatResponse,
  Message,
  Provider,
  StreamChunk,
  Tool,
} from "@rullama/core";
import { parseBedrockEvents } from "./eventstream.ts";
import {
  type AnthropicStreamEvent,
  type AnthropicStreamState,
  buildAnthropicBody,
  parseAnthropicResponse as parseBedrockResponse,
  streamEventToChunks,
} from "./anthropic_format.ts";

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

// ---------------------------------------------------------------------------
// Conversion helpers (exported for testing)
// ---------------------------------------------------------------------------

// Message/tool conversion and response parsing are the Anthropic Messages
// format, shared with anthropic.ts via anthropic_format.ts.
export {
  convertMessages,
  convertTools,
  getSystemMessage,
  parseAnthropicResponse as parseBedrockResponse,
} from "./anthropic_format.ts";

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

  /**
   * Create a provider that signs each request with the given credentials.
   *
   * @param region AWS region of the Bedrock runtime endpoint (e.g. `us-east-1`).
   * @param model Bedrock model id (e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`).
   * @param credentials Access key, secret and optional session token used for SigV4.
   * @param providerName Value reported as `name` (default: `"bedrock"`).
   */
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

    // Bedrock streams AWS event-stream frames, not SSE; each chunk wraps an
    // Anthropic-format event as base64 JSON.
    const state: AnthropicStreamState = { promptTokens: 0 };
    for await (const raw of parseBedrockEvents(response.body)) {
      yield* streamEventToChunks(raw as unknown as AnthropicStreamEvent, state);
    }
  }

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

  /** Build the Anthropic Messages body for Bedrock: same mapping as the
   * direct Anthropic provider, but carrying `anthropic_version` instead of a
   * `model` field (the model is part of the invoke URL). */
  private buildRequestBody(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Record<string, unknown> {
    return buildAnthropicBody(messages, tools, options, {
      anthropic_version: ANTHROPIC_BEDROCK_VERSION,
    });
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
