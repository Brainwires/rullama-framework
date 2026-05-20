/**
 * Cartesia API client for text-to-speech.
 *
 * Equivalent to Rust's `brainwires_providers::cartesia` module.
 */

import { RateLimiter } from "../rate_limiter.ts";

export const CARTESIA_API_BASE = "https://api.cartesia.ai";
export const CARTESIA_VERSION = "2024-06-10";

/** Voice configuration. */
export interface CartesiaVoice {
  /** "id" for pre-built voices. */
  mode: string;
  id?: string;
}

/** Output format configuration. */
export interface CartesiaOutputFormat {
  /** "raw" | "wav". */
  container: string;
  /** "pcm_f32le" | "pcm_s16le" | "pcm_mulaw". */
  encoding: string;
  sample_rate: number;
}

/** TTS request. */
export interface CartesiaTtsRequest {
  model_id: string;
  transcript: string;
  voice: CartesiaVoice;
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
  readonly base_url: string;
  private readonly api_key: string;
  private rate_limiter: RateLimiter | null = null;

  constructor(api_key: string, base_url: string = CARTESIA_API_BASE) {
    this.api_key = api_key;
    this.base_url = base_url;
  }

  withRateLimit(requests_per_minute: number): this {
    this.rate_limiter = new RateLimiter(requests_per_minute);
    return this;
  }

  private async acquire(): Promise<void> {
    if (this.rate_limiter) await this.rate_limiter.acquire();
  }

  /** Text-to-speech synthesis. Returns raw audio bytes. */
  async ttsBytes(req: CartesiaTtsRequest): Promise<Uint8Array> {
    await this.acquire();
    const res = await fetch(`${this.base_url}/tts/bytes`, {
      method: "POST",
      headers: {
        "X-API-Key": this.api_key,
        "Cartesia-Version": CARTESIA_VERSION,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(serializeTts(req)),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Cartesia TTS API error (${res.status}): ${body}`);
    }
    return new Uint8Array(await res.arrayBuffer());
  }
}
