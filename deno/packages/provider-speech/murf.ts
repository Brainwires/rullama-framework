/**
 * Murf AI API client for text-to-speech.
 *
 * Equivalent to Rust's `rullama_providers::murf` module.
 *
 * Murf returns a URL to the generated audio rather than bytes directly;
 * use {@link MurfClient.downloadAudio} to fetch the payload.
 */

import { vendorBytes, vendorJson } from "./http.ts";

import { RateLimiter } from "@rullama/core";

export const MURF_API_BASE = "https://api.murf.ai/v1";

/** Generate-speech request (wire format uses camelCase). */
export interface MurfGenerateRequest {
  voiceId: string;
  text: string;
  /** "WAV" | "MP3" | "FLAC". */
  format?: string;
  /** 0.5 – 2.0. */
  rate?: number;
  /** -50 – 50. */
  pitch?: number;
  /** 8000 | 16000 | 22050 | 24000 | 44100 | 48000. */
  sampleRate?: number;
}

/** Generate-speech response. */
export interface MurfGenerateResponse {
  audioFile?: string;
  audioDuration?: number;
}

/** A single Murf voice. */
export interface MurfVoice {
  voiceId: string;
  name: string;
  gender?: string;
  languageCode?: string;
}

/** Voices list response. */
export interface MurfVoicesResponse {
  voices: MurfVoice[];
}

function serializeGenerate(req: MurfGenerateRequest): Record<string, unknown> {
  const out: Record<string, unknown> = { voiceId: req.voiceId, text: req.text };
  if (req.format !== undefined) out.format = req.format;
  if (req.rate !== undefined) out.rate = req.rate;
  if (req.pitch !== undefined) out.pitch = req.pitch;
  if (req.sampleRate !== undefined) out.sampleRate = req.sampleRate;
  return out;
}

/** Exposed for tests. */
export const _serializeGenerate = serializeGenerate;

/** Murf AI API client. */
export class MurfClient {
  readonly base_url: string;
  private readonly api_key: string;
  private rate_limiter: RateLimiter | null = null;

  constructor(api_key: string, base_url: string = MURF_API_BASE) {
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

  /** Generate speech from text. Returns a URL to the generated audio. */
  async generateSpeech(
    req: MurfGenerateRequest,
  ): Promise<MurfGenerateResponse> {
    await this.acquire();
    return vendorJson<MurfGenerateResponse>(
      "Murf",
      `${this.base_url}/speech/generate`,
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.api_key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(serializeGenerate(req)),
      },
    );
  }

  /** Download audio from a URL returned by {@link generateSpeech}. */
  downloadAudio(audio_url: string): Promise<Uint8Array> {
    // The URL comes from the vendor's response; refuse anything that is not a
    // public https URL so a spoofed response cannot point us at localhost or
    // a metadata endpoint.
    const parsed = new URL(audio_url);
    const host = parsed.hostname;
    if (
      parsed.protocol !== "https:" || host === "localhost" ||
      /^[\d.]+$|^\[/.test(host)
    ) {
      return Promise.reject(
        new Error(
          `Murf download refused: not a public https URL (${audio_url})`,
        ),
      );
    }
    return vendorBytes("Murf download", audio_url, { method: "GET" });
  }

  /** List available voices. */
  async listVoices(): Promise<MurfVoicesResponse> {
    await this.acquire();
    return vendorJson<MurfVoicesResponse>(
      "Murf voices",
      `${this.base_url}/speech/voices`,
      {
        method: "GET",
        headers: { "Authorization": `Bearer ${this.api_key}` },
      },
    );
  }
}
