/**
 * Fish Audio client for text-to-speech (`tts`, returning `Uint8Array` audio)
 * and speech recognition (`asr`, returning a `FishAsrResponse` transcript);
 * `withRateLimit` caps requests per minute.
 * Equivalent to Rust's `rullama_provider_speech::fish`.
 *
 * @module
 */

import { vendorBytes, vendorJson } from "./http.ts";

import { RateLimiter } from "@rullama/core";

/** Default Fish Audio API base URL. */
export const FISH_API_BASE = "https://api.fish.audio/v1";

/** TTS request. */
export interface FishTtsRequest {
  /** Text to synthesize. */
  text: string;
  /** Reference audio / voice ID. */
  reference_id?: string;
  /** "wav" | "mp3" | … */
  format?: string;
  /** 0.5 – 2.0. */
  speed?: number;
}

/** ASR request parameters. */
export interface FishAsrRequest {
  /** Language hint (e.g. `en`). */
  language?: string;
}

/** ASR response. */
export interface FishAsrResponse {
  /** Transcript text. */
  text: string;
  /** Audio duration in seconds. */
  duration?: number;
}

function serializeTts(req: FishTtsRequest): Record<string, unknown> {
  const out: Record<string, unknown> = { text: req.text };
  if (req.reference_id !== undefined) out.reference_id = req.reference_id;
  if (req.format !== undefined) out.format = req.format;
  if (req.speed !== undefined) out.speed = req.speed;
  return out;
}

/** Exposed for tests. */
export const _serializeTts = serializeTts;

/** Fish Audio API client. */
export class FishClient {
  /** API base URL. */
  readonly base_url: string;
  private readonly api_key: string;
  private rate_limiter: RateLimiter | null = null;

  /** Create a client with the given API key and base URL. */
  constructor(api_key: string, base_url: string = FISH_API_BASE) {
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

  /** Text-to-speech. Returns raw audio bytes. */
  async tts(req: FishTtsRequest): Promise<Uint8Array> {
    await this.acquire();
    return vendorBytes("Fish TTS", `${this.base_url}/tts`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.api_key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(serializeTts(req)),
    });
  }

  /** Automatic speech recognition (multipart upload). */
  async asr(
    audio_data: Uint8Array,
    req: FishAsrRequest,
  ): Promise<FishAsrResponse> {
    await this.acquire();
    const form = new FormData();
    form.append(
      "audio",
      new Blob([audio_data as BlobPart], { type: "audio/wav" }),
      "audio.wav",
    );
    if (req.language) form.append("language", req.language);
    return vendorJson<FishAsrResponse>("Fish ASR", `${this.base_url}/asr`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${this.api_key}` },
      body: form,
    });
  }
}
