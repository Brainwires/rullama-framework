/**
 * Google Cloud Text-to-Speech API client.
 *
 * Equivalent to Rust's `brainwires_providers::google_tts` module.
 *
 * Returns base64-encoded audio content (per Google's API contract);
 * consumers can decode with `atob` + TextEncoder or Uint8Array conversion.
 */

import { RateLimiter } from "../rate_limiter.ts";

export const GOOGLE_TTS_API_BASE = "https://texttospeech.googleapis.com/v1";

/** Text input for synthesis — set exactly one of `text` or `ssml`. */
export interface GoogleTtsInput {
  text?: string;
  ssml?: string;
}

/** Voice selection parameters. */
export interface GoogleTtsVoiceSelection {
  languageCode: string;
  name?: string;
  /** "MALE" | "FEMALE" | "NEUTRAL". */
  ssmlGender?: string;
}

/** Audio configuration. */
export interface GoogleTtsAudioConfig {
  /** "LINEAR16" | "MP3" | "OGG_OPUS" | "MULAW" | "ALAW". */
  audioEncoding: string;
  /** 0.25 – 4.0. */
  speakingRate?: number;
  /** -20.0 – 20.0. */
  pitch?: number;
  sampleRateHertz?: number;
}

/** Synthesize request. */
export interface GoogleTtsSynthesizeRequest {
  input: GoogleTtsInput;
  voice: GoogleTtsVoiceSelection;
  audioConfig: GoogleTtsAudioConfig;
}

/** Synthesize response. */
export interface GoogleTtsSynthesizeResponse {
  /** Base64-encoded audio. */
  audioContent: string;
}

/** A single voice entry. */
export interface GoogleTtsVoiceEntry {
  languageCodes: string[];
  name: string;
  ssmlGender?: string;
  naturalSampleRateHertz?: number;
}

/** Voices list response. */
export interface GoogleTtsVoicesResponse {
  voices: GoogleTtsVoiceEntry[];
}

function serializeRequest(req: GoogleTtsSynthesizeRequest): Record<string, unknown> {
  const input: Record<string, unknown> = {};
  if (req.input.text !== undefined) input.text = req.input.text;
  if (req.input.ssml !== undefined) input.ssml = req.input.ssml;

  const voice: Record<string, unknown> = { languageCode: req.voice.languageCode };
  if (req.voice.name !== undefined) voice.name = req.voice.name;
  if (req.voice.ssmlGender !== undefined) voice.ssmlGender = req.voice.ssmlGender;

  const ac: Record<string, unknown> = { audioEncoding: req.audioConfig.audioEncoding };
  if (req.audioConfig.speakingRate !== undefined) ac.speakingRate = req.audioConfig.speakingRate;
  if (req.audioConfig.pitch !== undefined) ac.pitch = req.audioConfig.pitch;
  if (req.audioConfig.sampleRateHertz !== undefined) ac.sampleRateHertz = req.audioConfig.sampleRateHertz;

  return { input, voice, audioConfig: ac };
}

/** Exposed for tests. */
export const _serializeRequest = serializeRequest;

/** Google Cloud TTS API client. */
export class GoogleTtsClient {
  readonly base_url: string;
  private readonly api_key: string;
  private rate_limiter: RateLimiter | null = null;

  constructor(api_key: string, base_url: string = GOOGLE_TTS_API_BASE) {
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

  /** Synthesize speech from text. Returns base64-encoded audio content. */
  async synthesize(req: GoogleTtsSynthesizeRequest): Promise<GoogleTtsSynthesizeResponse> {
    await this.acquire();
    const res = await fetch(`${this.base_url}/text:synthesize`, {
      method: "POST",
      headers: {
        "X-Goog-Api-Key": this.api_key,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(serializeRequest(req)),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Google TTS API error (${res.status}): ${body}`);
    }
    return await res.json() as GoogleTtsSynthesizeResponse;
  }

  /** List available voices. */
  async listVoices(language_code?: string): Promise<GoogleTtsVoicesResponse> {
    await this.acquire();
    const url = language_code
      ? `${this.base_url}/voices?languageCode=${encodeURIComponent(language_code)}`
      : `${this.base_url}/voices`;
    const res = await fetch(url, {
      method: "GET",
      headers: { "X-Goog-Api-Key": this.api_key },
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Google TTS voices API error (${res.status}): ${body}`);
    }
    return await res.json() as GoogleTtsVoicesResponse;
  }
}
