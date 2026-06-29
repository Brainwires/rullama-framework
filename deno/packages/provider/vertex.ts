// deno-lint-ignore-file no-explicit-any
/**
 * Google Vertex AI chat provider implementation.
 * Uses JWT-based service account auth via Web Crypto API (zero external dependencies).
 * Wraps the Gemini generateContent API format for Vertex AI endpoints.
 * Equivalent to Rust's `anthropic/vertex.rs` + Vertex-specific auth.
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
import { parseNDJSONStream } from "./sse.ts";

// ---------------------------------------------------------------------------
// JWT / Google OAuth2 service account auth
// ---------------------------------------------------------------------------

/** Google Cloud service account credentials (from JSON key file). */
export interface ServiceAccountCredentials {
  client_email: string;
  private_key: string;
  token_uri: string;
  project_id?: string;
}

/** Base64url-encode a Uint8Array. */
function base64urlEncode(data: Uint8Array): string {
  const base64 = btoa(String.fromCharCode(...data));
  return base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** Base64url-encode a string. */
function base64urlEncodeStr(str: string): string {
  return base64urlEncode(new TextEncoder().encode(str));
}

/** Import a PEM RSA private key for signing. */
async function importPrivateKey(pem: string): Promise<CryptoKey> {
  // Strip PEM headers and decode
  const pemBody = pem
    .replace(/-----BEGIN (?:RSA )?PRIVATE KEY-----/g, "")
    .replace(/-----END (?:RSA )?PRIVATE KEY-----/g, "")
    .replace(/\s/g, "");
  const binaryDer = Uint8Array.from(atob(pemBody), (c) => c.charCodeAt(0));

  return await crypto.subtle.importKey(
    "pkcs8",
    binaryDer,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"],
  );
}

/** Create a signed JWT for Google OAuth2 token exchange. */
export async function createSignedJwt(
  credentials: ServiceAccountCredentials,
  scopes: string[],
  nowSeconds?: number,
): Promise<string> {
  const now = nowSeconds ?? Math.floor(Date.now() / 1000);
  const expiry = now + 3600; // 1 hour

  const header = {
    alg: "RS256",
    typ: "JWT",
  };

  const claimSet = {
    iss: credentials.client_email,
    scope: scopes.join(" "),
    aud: credentials.token_uri,
    iat: now,
    exp: expiry,
  };

  const headerB64 = base64urlEncodeStr(JSON.stringify(header));
  const claimsB64 = base64urlEncodeStr(JSON.stringify(claimSet));
  const unsignedToken = `${headerB64}.${claimsB64}`;

  const key = await importPrivateKey(credentials.private_key);
  const signature = await crypto.subtle.sign(
    "RSASSA-PKCS1-v1_5",
    key,
    new TextEncoder().encode(unsignedToken),
  );

  const signatureB64 = base64urlEncode(new Uint8Array(signature));
  return `${unsignedToken}.${signatureB64}`;
}

/** Exchange a signed JWT for an OAuth2 access token. */
export async function exchangeJwtForAccessToken(
  jwt: string,
  tokenUri: string,
): Promise<{ access_token: string; expires_in: number }> {
  const response = await fetch(tokenUri, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
      assertion: jwt,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `Google OAuth2 token exchange failed (${response.status}): ${errorText}`,
    );
  }

  return await response.json();
}

/** Get an access token from service account credentials. */
export async function getAccessToken(
  credentials: ServiceAccountCredentials,
): Promise<string> {
  const jwt = await createSignedJwt(credentials, [
    "https://www.googleapis.com/auth/cloud-platform",
  ]);
  const tokenResponse = await exchangeJwtForAccessToken(
    jwt,
    credentials.token_uri,
  );
  return tokenResponse.access_token;
}

// ---------------------------------------------------------------------------
// Vertex AI wire types (Gemini generateContent format)
// ---------------------------------------------------------------------------

interface VertexPart {
  text?: string;
  functionCall?: { name: string; args: Record<string, any> };
  functionResponse?: { name: string; response: any };
}

interface VertexContent {
  role: string;
  parts: VertexPart[];
}

interface VertexTool {
  functionDeclarations: {
    name: string;
    description: string;
    parameters: Record<string, any>;
  }[];
}

interface VertexCandidate {
  content: { role: string; parts: VertexPart[] };
  finishReason?: string;
}

interface VertexResponse {
  candidates: VertexCandidate[];
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
  };
}

// ---------------------------------------------------------------------------
// Conversion helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Convert core Messages to Vertex AI (Gemini) wire format.
 * Returns [contents, systemInstruction]. */
export function convertMessages(
  messages: Message[],
): [VertexContent[], string | undefined] {
  const contents: VertexContent[] = [];
  let systemInstruction: string | undefined;

  for (const msg of messages) {
    if (msg.role === "system") {
      if (typeof msg.content === "string") {
        systemInstruction = msg.content;
      }
      continue;
    }

    const role = msg.role === "assistant" ? "model" : "user";
    const parts: VertexPart[] = [];

    if (typeof msg.content === "string") {
      parts.push({ text: msg.content });
    } else {
      for (const block of msg.content) {
        switch (block.type) {
          case "text":
            parts.push({ text: block.text });
            break;
          case "tool_use":
            parts.push({
              functionCall: { name: block.name, args: block.input },
            });
            break;
          case "tool_result":
            parts.push({
              functionResponse: {
                name: block.tool_use_id,
                response: { result: block.content },
              },
            });
            break;
        }
      }
    }

    if (parts.length > 0) {
      contents.push({ role, parts });
    }
  }

  return [contents, systemInstruction];
}

/** Convert core Tools to Vertex AI wire format. */
export function convertTools(tools: Tool[]): VertexTool {
  return {
    functionDeclarations: tools.map((t) => ({
      name: t.name,
      description: t.description,
      parameters: t.input_schema.properties
        ? { type: "object", properties: t.input_schema.properties }
        : { type: "object" },
    })),
  };
}

/** Parse a Vertex AI response into a core ChatResponse. */
export function parseVertexResponse(
  resp: VertexResponse,
): ChatResponse {
  const candidate = resp.candidates?.[0];
  if (!candidate) {
    throw new Error("No candidates in Vertex AI response");
  }

  const contentBlocks: ContentBlock[] = [];

  for (const part of candidate.content.parts) {
    if (part.text !== undefined) {
      contentBlocks.push({ type: "text", text: part.text });
    } else if (part.functionCall) {
      contentBlocks.push({
        type: "tool_use",
        id: `call_${crypto.randomUUID().slice(0, 8)}`,
        name: part.functionCall.name,
        input: part.functionCall.args ?? {},
      });
    }
  }

  let content: MessageContent;
  if (contentBlocks.length === 1 && contentBlocks[0].type === "text") {
    content = contentBlocks[0].text;
  } else if (contentBlocks.length === 0) {
    content = "";
  } else {
    content = contentBlocks;
  }

  const meta = resp.usageMetadata;
  const usage: Usage = {
    prompt_tokens: meta?.promptTokenCount ?? 0,
    completion_tokens: meta?.candidatesTokenCount ?? 0,
    total_tokens: meta?.totalTokenCount ?? 0,
  };

  return {
    message: new Message({ role: "assistant", content }),
    usage,
    finish_reason: candidate.finishReason ?? "STOP",
  };
}

/** Build the Vertex AI generateContent endpoint URL. */
export function vertexGenerateContentUrl(
  region: string,
  projectId: string,
  model: string,
): string {
  return `https://${region}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:generateContent`;
}

/** Build the Vertex AI streaming endpoint URL. */
export function vertexStreamUrl(
  region: string,
  projectId: string,
  model: string,
): string {
  return `https://${region}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:streamGenerateContent?alt=sse`;
}

// ---------------------------------------------------------------------------
// VertexAiProvider
// ---------------------------------------------------------------------------

/** Chat provider for Google Vertex AI (Gemini models).
 * Uses JWT-based service account auth — no google-auth-library dependency.
 * Equivalent to Rust's `VertexAuth` + Gemini chat logic. */
export class VertexAiProvider implements Provider {
  readonly name: string;
  private readonly region: string;
  private readonly projectId: string;
  private readonly model: string;
  private readonly credentials: ServiceAccountCredentials;
  private cachedToken?: { token: string; expiresAt: number };

  constructor(
    region: string,
    projectId: string,
    model: string,
    credentials: ServiceAccountCredentials,
    providerName?: string,
  ) {
    this.region = region;
    this.projectId = projectId;
    this.model = model;
    this.credentials = credentials;
    this.name = providerName ?? "vertex-ai";
  }

  /** Create from a service account JSON key file. */
  static async fromServiceAccountFile(
    filePath: string,
    region: string,
    model: string,
    projectIdOverride?: string,
  ): Promise<VertexAiProvider> {
    const raw = await Deno.readTextFile(filePath);
    const creds: ServiceAccountCredentials = JSON.parse(raw);
    const projectId = projectIdOverride ?? creds.project_id;
    if (!projectId) {
      throw new Error(
        "project_id not found in service account JSON and no override provided",
      );
    }
    return new VertexAiProvider(region, projectId, model, creds);
  }

  /** Get a valid access token, refreshing if needed. */
  private async getToken(): Promise<string> {
    const now = Date.now();
    if (this.cachedToken && this.cachedToken.expiresAt > now + 60_000) {
      return this.cachedToken.token;
    }

    const jwt = await createSignedJwt(this.credentials, [
      "https://www.googleapis.com/auth/cloud-platform",
    ]);
    const tokenResponse = await exchangeJwtForAccessToken(
      jwt,
      this.credentials.token_uri,
    );
    this.cachedToken = {
      token: tokenResponse.access_token,
      expiresAt: now + tokenResponse.expires_in * 1000,
    };
    return this.cachedToken.token;
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
    const url = vertexGenerateContentUrl(
      this.region,
      this.projectId,
      this.model,
    );
    const token = await this.getToken();

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Vertex AI API error (${response.status}): ${errorText}`,
      );
    }

    const vertexResponse: VertexResponse = await response.json();
    return parseVertexResponse(vertexResponse);
  }

  async *streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    const body = this.buildRequestBody(messages, tools, options);
    const url = vertexStreamUrl(
      this.region,
      this.projectId,
      this.model,
    );
    const token = await this.getToken();

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `Vertex AI API error (${response.status}): ${errorText}`,
      );
    }

    if (!response.body) {
      throw new Error("Vertex AI streaming response has no body");
    }

    for await (const data of parseNDJSONStream(response.body)) {
      let chunk: VertexResponse;
      try {
        chunk = JSON.parse(data);
      } catch {
        continue;
      }

      const candidate = chunk.candidates?.[0];
      if (candidate) {
        for (const part of candidate.content.parts) {
          if (part.text !== undefined) {
            yield { type: "text", text: part.text };
          } else if (part.functionCall) {
            yield {
              type: "tool_use",
              id: `call_${crypto.randomUUID().slice(0, 8)}`,
              name: part.functionCall.name,
            };
          }
        }
      }

      if (chunk.usageMetadata) {
        yield {
          type: "usage",
          usage: {
            prompt_tokens: chunk.usageMetadata.promptTokenCount ?? 0,
            completion_tokens: chunk.usageMetadata.candidatesTokenCount ?? 0,
            total_tokens: chunk.usageMetadata.totalTokenCount ?? 0,
          },
        };
      }
    }

    yield { type: "done" };
  }

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

  private buildRequestBody(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Record<string, any> {
    const [contents, systemFromMessages] = convertMessages(messages);
    const systemInstruction = options.system ?? systemFromMessages;

    const body: Record<string, any> = { contents };

    if (systemInstruction) {
      body.systemInstruction = {
        parts: [{ text: systemInstruction }],
      };
    }

    const generationConfig: Record<string, any> = {};
    if (options.max_tokens !== undefined) {
      generationConfig.maxOutputTokens = options.max_tokens;
    }
    if (options.temperature !== undefined) {
      generationConfig.temperature = options.temperature;
    }
    if (options.top_p !== undefined) {
      generationConfig.topP = options.top_p;
    }
    if (options.stop) {
      generationConfig.stopSequences = options.stop;
    }
    if (Object.keys(generationConfig).length > 0) {
      body.generationConfig = generationConfig;
    }

    if (tools && tools.length > 0) {
      body.tools = [convertTools(tools)];
    }

    return body;
  }
}
