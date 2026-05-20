/**
 * Fish Audio API client for text-to-speech and speech recognition.
 *
 * Equivalent to Rust's `brainwires_providers::fish` module.
 */

import { RateLimiter } from "../rate_limiter.ts";

export const FISH_API_BASE = "https://api.fish.audio/v1";

/** TTS request. */
export interface FishTtsRequest {
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
  language?: string;
}

/** ASR response. */
export interface FishAsrResponse {
  text: string;
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
  readonly base_url: string;
  private readonly api_key: string;
  private rate_limiter: RateLimiter | null = null;

  constructor(api_key: string, base_url: string = FISH_API_BASE) {
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

  /** Text-to-speech. Returns raw audio bytes. */
  async tts(req: FishTtsRequest): Promise<Uint8Array> {
    await this.acquire();
    const res = await fetch(`${this.base_url}/tts`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.api_key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(serializeTts(req)),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Fish TTS API error (${res.status}): ${body}`);
    }
    return new Uint8Array(await res.arrayBuffer());
  }

  /** Automatic speech recognition (multipart upload). */
  async asr(audio_data: Uint8Array, req: FishAsrRequest): Promise<FishAsrResponse> {
    await this.acquire();
    const form = new FormData();
    form.append(
      "audio",
      new Blob([audio_data as BlobPart], { type: "audio/wav" }),
      "audio.wav",
    );
    if (req.language) form.append("language", req.language);
    const res = await fetch(`${this.base_url}/asr`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${this.api_key}` },
      body: form,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Fish ASR API error (${res.status}): ${body}`);
    }
    return await res.json() as FishAsrResponse;
  }
}
