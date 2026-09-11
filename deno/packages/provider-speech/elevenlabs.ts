/**
 * ElevenLabs client for text-to-speech (`textToSpeech`, returning
 * `Uint8Array` audio), speech-to-text (`speechToText`, multipart upload) and
 * voice listing (`listVoices`); `withRateLimit` caps requests per minute.
 * Equivalent to Rust's `rullama_provider_speech::elevenlabs`.
 *
 * @module
 */

import { vendorBytes, vendorJson } from "./http.ts";

import { RateLimiter } from "@rullama/core";

/** Default ElevenLabs API base URL. */
export const ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1";

/** Voice settings for fine-tuning synthesis. */
export interface ElevenLabsVoiceSettings {
  /** Voice stability (0-1). */
  stability: number;
  /** Similarity boost (0-1). */
  similarity_boost: number;
  /** Style exaggeration (0-1). */
  style?: number;
  /** Enable speaker boost. */
  use_speaker_boost?: boolean;
}

/** TTS request body. */
export interface ElevenLabsTtsRequest {
  /** Text to synthesize. */
  text: string;
  /** Model ID (e.g. `eleven_multilingual_v2`). */
  model_id?: string;
  /** Voice tuning parameters. */
  voice_settings?: ElevenLabsVoiceSettings;
  /** Output format (e.g., "mp3_44100_128", "pcm_16000"). */
  output_format?: string;
}

/** STT request parameters. */
export interface ElevenLabsSttRequest {
  /** Transcription model (e.g. `scribe_v1`). */
  model?: string;
  /** ISO language code hint. */
  language_code?: string;
}

/** STT response. */
export interface ElevenLabsSttResponse {
  /** Transcript text. */
  text: string;
  /** Detected language code. */
  language_code?: string;
}

/** A single voice entry. */
export interface ElevenLabsVoice {
  /** Voice ID to pass to `textToSpeech`. */
  voice_id: string;
  /** Voice name. */
  name: string;
  /** Descriptive labels (accent, gender, …). */
  labels: Record<string, string>;
}

/** Voices list response. */
export interface ElevenLabsVoicesResponse {
  /** Available voices. */
  voices: ElevenLabsVoice[];
}

/** Serialize a TTS request, skipping undefined fields to match Rust's skip_serializing_if. */
export function serializeTtsRequest(
  req: ElevenLabsTtsRequest,
): Record<string, unknown> {
  const out: Record<string, unknown> = { text: req.text };
  if (req.model_id !== undefined) out.model_id = req.model_id;
  if (req.voice_settings !== undefined) {
    const vs: Record<string, unknown> = {
      stability: req.voice_settings.stability,
      similarity_boost: req.voice_settings.similarity_boost,
    };
    if (req.voice_settings.style !== undefined) {
      vs.style = req.voice_settings.style;
    }
    if (req.voice_settings.use_speaker_boost !== undefined) {
      vs.use_speaker_boost = req.voice_settings.use_speaker_boost;
    }
    out.voice_settings = vs;
  }
  if (req.output_format !== undefined) out.output_format = req.output_format;
  return out;
}

/** ElevenLabs API client. */
export class ElevenLabsClient {
  /** API base URL. */
  readonly base_url: string;
  private readonly api_key: string;
  private rate_limiter: RateLimiter | null = null;

  /** Create a client with the given API key and base URL. */
  constructor(api_key: string, base_url: string = ELEVENLABS_API_BASE) {
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

  /** Text-to-speech synthesis. Returns raw audio bytes (mp3 by default). */
  async textToSpeech(
    voice_id: string,
    req: ElevenLabsTtsRequest,
  ): Promise<Uint8Array> {
    await this.acquire();
    const url = `${this.base_url}/text-to-speech/${voice_id}`;
    return vendorBytes("ElevenLabs TTS", url, {
      method: "POST",
      headers: {
        "xi-api-key": this.api_key,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(serializeTtsRequest(req)),
    });
  }

  /** Speech-to-text transcription (multipart upload). */
  async speechToText(
    audio_data: Uint8Array,
    req: ElevenLabsSttRequest,
  ): Promise<ElevenLabsSttResponse> {
    await this.acquire();
    const form = new FormData();
    form.append(
      "audio",
      new Blob([audio_data as BlobPart], { type: "audio/wav" }),
      "audio.wav",
    );
    if (req.model) form.append("model_id", req.model);
    if (req.language_code) form.append("language_code", req.language_code);
    return vendorJson<ElevenLabsSttResponse>(
      "ElevenLabs STT",
      `${this.base_url}/speech-to-text`,
      {
        method: "POST",
        headers: { "xi-api-key": this.api_key },
        body: form,
      },
    );
  }

  /** List available voices. */
  async listVoices(): Promise<ElevenLabsVoicesResponse> {
    await this.acquire();
    return vendorJson<ElevenLabsVoicesResponse>(
      "ElevenLabs voices",
      `${this.base_url}/voices`,
      {
        method: "GET",
        headers: { "xi-api-key": this.api_key },
      },
    );
  }
}
