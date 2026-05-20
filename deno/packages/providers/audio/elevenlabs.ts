/**
 * ElevenLabs API client for text-to-speech and speech-to-text.
 *
 * Equivalent to Rust's `brainwires_providers::elevenlabs` module.
 */

import { RateLimiter } from "../rate_limiter.ts";

export const ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1";

/** Voice settings for fine-tuning synthesis. */
export interface ElevenLabsVoiceSettings {
  stability: number;
  similarity_boost: number;
  style?: number;
  use_speaker_boost?: boolean;
}

/** TTS request body. */
export interface ElevenLabsTtsRequest {
  text: string;
  model_id?: string;
  voice_settings?: ElevenLabsVoiceSettings;
  /** Output format (e.g., "mp3_44100_128", "pcm_16000"). */
  output_format?: string;
}

/** STT request parameters. */
export interface ElevenLabsSttRequest {
  model?: string;
  language_code?: string;
}

/** STT response. */
export interface ElevenLabsSttResponse {
  text: string;
  language_code?: string;
}

/** A single voice entry. */
export interface ElevenLabsVoice {
  voice_id: string;
  name: string;
  labels: Record<string, string>;
}

/** Voices list response. */
export interface ElevenLabsVoicesResponse {
  voices: ElevenLabsVoice[];
}

/** Serialize a TTS request, skipping undefined fields to match Rust's skip_serializing_if. */
export function serializeTtsRequest(req: ElevenLabsTtsRequest): Record<string, unknown> {
  const out: Record<string, unknown> = { text: req.text };
  if (req.model_id !== undefined) out.model_id = req.model_id;
  if (req.voice_settings !== undefined) {
    const vs: Record<string, unknown> = {
      stability: req.voice_settings.stability,
      similarity_boost: req.voice_settings.similarity_boost,
    };
    if (req.voice_settings.style !== undefined) vs.style = req.voice_settings.style;
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
  readonly base_url: string;
  private readonly api_key: string;
  private rate_limiter: RateLimiter | null = null;

  constructor(api_key: string, base_url: string = ELEVENLABS_API_BASE) {
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

  /** Text-to-speech synthesis. Returns raw audio bytes (mp3 by default). */
  async textToSpeech(voice_id: string, req: ElevenLabsTtsRequest): Promise<Uint8Array> {
    await this.acquire();
    const url = `${this.base_url}/text-to-speech/${voice_id}`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "xi-api-key": this.api_key,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(serializeTtsRequest(req)),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`ElevenLabs TTS API error (${res.status}): ${body}`);
    }
    return new Uint8Array(await res.arrayBuffer());
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
    const res = await fetch(`${this.base_url}/speech-to-text`, {
      method: "POST",
      headers: { "xi-api-key": this.api_key },
      body: form,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`ElevenLabs STT API error (${res.status}): ${body}`);
    }
    return await res.json() as ElevenLabsSttResponse;
  }

  /** List available voices. */
  async listVoices(): Promise<ElevenLabsVoicesResponse> {
    await this.acquire();
    const res = await fetch(`${this.base_url}/voices`, {
      method: "GET",
      headers: { "xi-api-key": this.api_key },
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`ElevenLabs voices API error (${res.status}): ${body}`);
    }
    return await res.json() as ElevenLabsVoicesResponse;
  }
}
