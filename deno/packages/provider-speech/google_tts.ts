/**
 * Google Cloud Text-to-Speech client. `GoogleTtsClient.synthesize` returns
 * the API's base64-encoded audio content (decode it with `atob` on the
 * consumer side) and `listVoices` enumerates the
 * available voices; `withRateLimit` caps requests per minute.
 * Equivalent to Rust's `rullama_provider_speech::google_tts`.
 *
 * @module
 */

import { vendorJson } from "./http.ts";

import { RateLimiter } from "@rullama/core";

/** Default Google Cloud Text-to-Speech API base URL. */
export const GOOGLE_TTS_API_BASE = "https://texttospeech.googleapis.com/v1";

/** Text input for synthesis — set exactly one of `text` or `ssml`. */
export interface GoogleTtsInput {
  /** Plain text input (mutually exclusive with `ssml`). */
  text?: string;
  /** SSML input (mutually exclusive with `text`). */
  ssml?: string;
}

/** Voice selection parameters. */
export interface GoogleTtsVoiceSelection {
  /** BCP-47 language code (e.g. `en-US`). */
  languageCode: string;
  /** Voice name (e.g. `en-US-Neural2-C`). */
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
  /** Output sample rate in Hz. */
  sampleRateHertz?: number;
}

/** Synthesize request. */
export interface GoogleTtsSynthesizeRequest {
  /** What to synthesize. */
  input: GoogleTtsInput;
  /** Which voice to use. */
  voice: GoogleTtsVoiceSelection;
  /** Output encoding and tuning. */
  audioConfig: GoogleTtsAudioConfig;
}

/** Synthesize response. */
export interface GoogleTtsSynthesizeResponse {
  /** Base64-encoded audio. */
  audioContent: string;
}

/** A single voice entry. */
export interface GoogleTtsVoiceEntry {
  /** Language codes the voice supports. */
  languageCodes: string[];
  /** Voice name. */
  name: string;
  /** `"MALE"`, `"FEMALE"` or `"NEUTRAL"`. */
  ssmlGender?: string;
  /** Native sample rate of the voice in Hz. */
  naturalSampleRateHertz?: number;
}

/** Voices list response. */
export interface GoogleTtsVoicesResponse {
  /** Available voices. */
  voices: GoogleTtsVoiceEntry[];
}

function serializeRequest(
  req: GoogleTtsSynthesizeRequest,
): Record<string, unknown> {
  const input: Record<string, unknown> = {};
  if (req.input.text !== undefined) input.text = req.input.text;
  if (req.input.ssml !== undefined) input.ssml = req.input.ssml;

  const voice: Record<string, unknown> = {
    languageCode: req.voice.languageCode,
  };
  if (req.voice.name !== undefined) voice.name = req.voice.name;
  if (req.voice.ssmlGender !== undefined) {
    voice.ssmlGender = req.voice.ssmlGender;
  }

  const ac: Record<string, unknown> = {
    audioEncoding: req.audioConfig.audioEncoding,
  };
  if (req.audioConfig.speakingRate !== undefined) {
    ac.speakingRate = req.audioConfig.speakingRate;
  }
  if (req.audioConfig.pitch !== undefined) ac.pitch = req.audioConfig.pitch;
  if (req.audioConfig.sampleRateHertz !== undefined) {
    ac.sampleRateHertz = req.audioConfig.sampleRateHertz;
  }

  return { input, voice, audioConfig: ac };
}

/** Exposed for tests. */
export const _serializeRequest = serializeRequest;

/** Google Cloud TTS API client. */
export class GoogleTtsClient {
  /** API base URL. */
  readonly base_url: string;
  private readonly api_key: string;
  private rate_limiter: RateLimiter | null = null;

  /** Create a client with the given API key and base URL. */
  constructor(api_key: string, base_url: string = GOOGLE_TTS_API_BASE) {
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

  /** Synthesize speech from text. Returns base64-encoded audio content. */
  async synthesize(
    req: GoogleTtsSynthesizeRequest,
  ): Promise<GoogleTtsSynthesizeResponse> {
    await this.acquire();
    return vendorJson<GoogleTtsSynthesizeResponse>(
      "Google TTS",
      `${this.base_url}/text:synthesize`,
      {
        method: "POST",
        headers: {
          "X-Goog-Api-Key": this.api_key,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(serializeRequest(req)),
      },
    );
  }

  /** List available voices. */
  async listVoices(language_code?: string): Promise<GoogleTtsVoicesResponse> {
    await this.acquire();
    const url = language_code
      ? `${this.base_url}/voices?languageCode=${
        encodeURIComponent(language_code)
      }`
      : `${this.base_url}/voices`;
    return vendorJson<GoogleTtsVoicesResponse>("Google TTS voices", url, {
      method: "GET",
      headers: { "X-Goog-Api-Key": this.api_key },
    });
  }
}
