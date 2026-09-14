/**
 * Deepgram client for text-to-speech (Aura, `speak`) and speech-to-text
 * (Listen, `listen`). Audio payloads are `Uint8Array` in both directions and the
 * transcript comes back as a typed `DeepgramListenResponse`; `withRateLimit`
 * caps requests per minute.
 * Equivalent to Rust's `rullama_provider_speech::deepgram`.
 *
 * @module
 */

import { vendorBytes, vendorJson } from "./http.ts";

import { RateLimiter } from "@rullama/core";

/** Default Deepgram API base URL. */
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
  /** Model name (e.g. `nova-2`). */
  model?: string;
  /** BCP-47 language code. */
  language?: string;
  /** Add punctuation to the transcript. */
  punctuate?: boolean;
  /** Label speakers in the transcript. */
  diarize?: boolean;
  /** Content type of the audio (e.g., "audio/wav"). Default: "audio/wav". */
  content_type?: string;
}

/** A single word with timing. */
export interface DeepgramWord {
  /** The recognized word. */
  word: string;
  /** Start time in seconds. */
  start: number;
  /** End time in seconds. */
  end: number;
  /** Confidence (0-1). */
  confidence: number;
}

/** A transcription alternative. */
export interface DeepgramAlternative {
  /** Transcript text. */
  transcript: string;
  /** Confidence (0-1). */
  confidence: number;
  /** Word-level timings. */
  words: DeepgramWord[];
}

/** A single channel's transcription. */
export interface DeepgramChannel {
  /** Alternative transcripts, best first. */
  alternatives: DeepgramAlternative[];
}

/** Transcription results container. */
export interface DeepgramResults {
  /** One entry per audio channel. */
  channels: DeepgramChannel[];
}

/** Listen (STT) response. */
export interface DeepgramListenResponse {
  /** Transcription results. */
  results: DeepgramResults;
}

/** Deepgram API client. */
export class DeepgramClient {
  /** API base URL. */
  readonly base_url: string;
  private readonly api_key: string;
  private rate_limiter: RateLimiter | null = null;

  /** Create a client with the given API key and base URL. */
  constructor(api_key: string, base_url: string = DEEPGRAM_API_BASE) {
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

  /** Text-to-speech (Aura). Returns raw audio bytes. */
  async speak(req: DeepgramSpeakRequest): Promise<Uint8Array> {
    await this.acquire();
    const params = new URLSearchParams();
    if (req.model) params.set("model", req.model);
    if (req.encoding) params.set("encoding", req.encoding);
    if (req.sample_rate !== undefined) {
      params.set("sample_rate", String(req.sample_rate));
    }
    const qs = params.toString();
    const url = `${this.base_url}/speak${qs ? `?${qs}` : ""}`;
    return vendorBytes("Deepgram speak", url, {
      method: "POST",
      headers: {
        "Authorization": `Token ${this.api_key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: req.text }),
    });
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
    return vendorJson<DeepgramListenResponse>("Deepgram listen", url, {
      method: "POST",
      headers: {
        "Authorization": `Token ${this.api_key}`,
        "Content-Type": req.content_type ?? "audio/wav",
      },
      body: audio_data as BodyInit,
    });
  }
}
