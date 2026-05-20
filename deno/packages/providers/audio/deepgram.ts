/**
 * Deepgram API client for text-to-speech (Aura) and speech-to-text (Listen).
 *
 * Equivalent to Rust's `brainwires_providers::deepgram` module.
 *
 * The Deno client returns/accepts `Uint8Array` for audio payloads; hardware
 * capture/playback (microphone, speaker) must be handled by the consumer.
 */

import { RateLimiter } from "../rate_limiter.ts";

export const DEEPGRAM_API_BASE = "https://api.deepgram.com/v1";

/** Speak (TTS) request. */
export interface DeepgramSpeakRequest {
  /** Text to synthesize. */
  text: string;
  /** Model name (e.g., "aura-asteria-en"). */
  model?: string;
  /** Output encoding (e.g., "linear16", "mp3"). */
  encoding?: string;
  /** Sample rate for output audio. */
  sample_rate?: number;
}

/** Listen (STT) request parameters. */
export interface DeepgramListenRequest {
  model?: string;
  language?: string;
  punctuate?: boolean;
  diarize?: boolean;
  /** Content type of the audio (e.g., "audio/wav"). Default: "audio/wav". */
  content_type?: string;
}

/** A single word with timing. */
export interface DeepgramWord {
  word: string;
  start: number;
  end: number;
  confidence: number;
}

/** A transcription alternative. */
export interface DeepgramAlternative {
  transcript: string;
  confidence: number;
  words: DeepgramWord[];
}

/** A single channel's transcription. */
export interface DeepgramChannel {
  alternatives: DeepgramAlternative[];
}

/** Transcription results container. */
export interface DeepgramResults {
  channels: DeepgramChannel[];
}

/** Listen (STT) response. */
export interface DeepgramListenResponse {
  results: DeepgramResults;
}

/** Deepgram API client. */
export class DeepgramClient {
  readonly base_url: string;
  private readonly api_key: string;
  private rate_limiter: RateLimiter | null = null;

  constructor(api_key: string, base_url: string = DEEPGRAM_API_BASE) {
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

  /** Text-to-speech (Aura). Returns raw audio bytes. */
  async speak(req: DeepgramSpeakRequest): Promise<Uint8Array> {
    await this.acquire();
    const params = new URLSearchParams();
    if (req.model) params.set("model", req.model);
    if (req.encoding) params.set("encoding", req.encoding);
    if (req.sample_rate !== undefined) params.set("sample_rate", String(req.sample_rate));
    const qs = params.toString();
    const url = `${this.base_url}/speak${qs ? `?${qs}` : ""}`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Token ${this.api_key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: req.text }),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Deepgram speak API error (${res.status}): ${body}`);
    }
    return new Uint8Array(await res.arrayBuffer());
  }

  /** Speech-to-text (Listen). Transcribes audio data. */
  async listen(
    audio_data: Uint8Array,
    req: DeepgramListenRequest,
  ): Promise<DeepgramListenResponse> {
    await this.acquire();
    const params = new URLSearchParams();
    if (req.model) params.set("model", req.model);
    if (req.language) params.set("language", req.language);
    if (req.punctuate) params.set("punctuate", "true");
    if (req.diarize) params.set("diarize", "true");
    const qs = params.toString();
    const url = `${this.base_url}/listen${qs ? `?${qs}` : ""}`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Token ${this.api_key}`,
        "Content-Type": req.content_type ?? "audio/wav",
      },
      body: audio_data,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Deepgram listen API error (${res.status}): ${body}`);
    }
    return await res.json() as DeepgramListenResponse;
  }
}
