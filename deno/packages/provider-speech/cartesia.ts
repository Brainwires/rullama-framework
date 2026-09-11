/**
 * Cartesia text-to-speech client. `CartesiaClient.ttsBytes` posts a
 * `CartesiaTtsRequest` (voice, model, output format) and returns the raw audio
 * as `Uint8Array`; `withRateLimit` caps requests per minute.
 * Equivalent to Rust's `rullama_provider_speech::cartesia`.
 *
 * @module
 */

import { vendorBytes } from "./http.ts";

import { RateLimiter } from "@rullama/core";

/** Default Cartesia API base URL. */
export const CARTESIA_API_BASE = "https://api.cartesia.ai";
/** Cartesia API version sent in the `Cartesia-Version` header. */
export const CARTESIA_VERSION = "2024-06-10";

/** Voice configuration. */
export interface CartesiaVoice {
  /** "id" for pre-built voices. */
  mode: string;
  /** Voice ID (required when `mode` is `"id"`). */
  id?: string;
}

/** Output format configuration. */
export interface CartesiaOutputFormat {
  /** "raw" | "wav". */
  container: string;
  /** "pcm_f32le" | "pcm_s16le" | "pcm_mulaw". */
  encoding: string;
  /** Sample rate in Hz. */
  sample_rate: number;
}

/** TTS request. */
export interface CartesiaTtsRequest {
  /** Model ID (e.g. `sonic-english`). */
  model_id: string;
  /** Text to synthesize. */
  transcript: string;
  /** Voice selection. */
  voice: CartesiaVoice;
  /** Audio container / encoding / sample rate. */
  output_format: CartesiaOutputFormat;
  /** e.g., "en". */
  language?: string;
}

function serializeTts(req: CartesiaTtsRequest): Record<string, unknown> {
  const voice: Record<string, unknown> = { mode: req.voice.mode };
  if (req.voice.id !== undefined) voice.id = req.voice.id;
  const out: Record<string, unknown> = {
    model_id: req.model_id,
    transcript: req.transcript,
    voice,
    output_format: {
      container: req.output_format.container,
      encoding: req.output_format.encoding,
      sample_rate: req.output_format.sample_rate,
    },
  };
  if (req.language !== undefined) out.language = req.language;
  return out;
}

/** Exposed for tests. */
export const _serializeTts = serializeTts;

/** Cartesia API client. */
export class CartesiaClient {
  /** API base URL. */
  readonly base_url: string;
  private readonly api_key: string;
  private rate_limiter: RateLimiter | null = null;

  /** Create a client with the given API key and base URL. */
  constructor(api_key: string, base_url: string = CARTESIA_API_BASE) {
    this.api_key = api_key;
    this.base_url = base_url;
  }

  /** Cap requests per minute with a token bucket. Returns `this`. */
  withRateLimit(requests_per_minute: number): this {
    this.rate_limiter = new RateLimiter(requests_per_minute);
    return this;
  }

  /** Wait for a rate-limit token, if a limiter is configured. */
  private async acquire(): Promise<void> {
    if (this.rate_limiter) await this.rate_limiter.acquire();
  }

  /** Text-to-speech synthesis. Returns raw audio bytes. */
  async ttsBytes(req: CartesiaTtsRequest): Promise<Uint8Array> {
    await this.acquire();
    return vendorBytes("Cartesia TTS", `${this.base_url}/tts/bytes`, {
      method: "POST",
      headers: {
        "X-API-Key": this.api_key,
        "Cartesia-Version": CARTESIA_VERSION,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(serializeTts(req)),
    });
  }
}
