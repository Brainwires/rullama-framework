/**
 * Azure Cognitive Services Speech client. `AzureSpeechClient` is built from a
 * subscription key and region and exposes `synthesize` / `synthesizeText` (TTS,
 * returning `Uint8Array` audio), `recognize` (STT over a `Uint8Array` payload)
 * and `listVoices`; `withRateLimit` caps requests per minute.
 * Equivalent to Rust's `rullama_provider_speech::azure_speech`.
 *
 * @module
 */

import { vendorBytes, vendorJson } from "./http.ts";

import { RateLimiter } from "@rullama/core";

/** STT request parameters. */
export interface AzureSttRequest {
  /** Language (e.g., "en-US"). Default "en-US". */
  language?: string;
  /** Content-type header value. Default "audio/wav; codecs=audio/pcm; samplerate=16000". */
  content_type?: string;
}

/** STT response (Azure uses PascalCase on the wire). */
export interface AzureSttResponse {
  /** `"Success"`, `"NoMatch"`, `"InitialSilenceTimeout"`, … per the Azure API. */
  RecognitionStatus: string;
  /** Recognized text, when `RecognitionStatus` is `"Success"`. */
  DisplayText?: string;
  /** Offset of the recognized audio in 100-ns ticks. */
  Offset?: number;
  /** Duration of the recognized audio in 100-ns ticks. */
  Duration?: number;
}

/** An Azure voice entry (PascalCase wire format). */
export interface AzureVoice {
  /** Full voice name (e.g. `en-US-JennyNeural`). */
  Name: string;
  /** Human-readable voice name. */
  DisplayName: string;
  /** e.g., "en-US-JennyNeural". */
  ShortName: string;
  /** `"Female"` or `"Male"`. */
  Gender: string;
  /** BCP-47 locale of the voice. */
  Locale: string;
}

/** Azure Speech API client. */
export class AzureSpeechClient {
  /** Azure region (e.g. `eastus`) the endpoints are built from. */
  readonly region: string;
  private readonly subscription_key: string;
  private rate_limiter: RateLimiter | null = null;

  /** Create a client for the given subscription key and region. */
  constructor(subscription_key: string, region: string) {
    this.subscription_key = subscription_key;
    this.region = region;
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

  /** Text-to-speech endpoint URL for this region. */
  ttsEndpoint(): string {
    return `https://${this.region}.tts.speech.microsoft.com/cognitiveservices/v1`;
  }

  /** Speech-to-text endpoint URL for this region. */
  sttEndpoint(): string {
    return `https://${this.region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1`;
  }

  /** Voice-list endpoint URL for this region. */
  voicesEndpoint(): string {
    return `https://${this.region}.tts.speech.microsoft.com/cognitiveservices/voices/list`;
  }

  /** Synthesize speech from SSML. Returns raw audio bytes. */
  async synthesize(ssml: string, output_format: string): Promise<Uint8Array> {
    await this.acquire();
    return vendorBytes("Azure TTS", this.ttsEndpoint(), {
      method: "POST",
      headers: {
        "Ocp-Apim-Subscription-Key": this.subscription_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": output_format,
      },
      body: ssml,
    });
  }

  /** Synthesize from plain text by wrapping in SSML. */
  synthesizeText(
    text: string,
    voice_name: string,
    output_format: string,
  ): Promise<Uint8Array> {
    const ssml =
      `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
    <voice name="${voice_name}">${text}</voice>
</speak>`;
    return this.synthesize(ssml, output_format);
  }

  /** Recognize speech from audio data. */
  async recognize(
    audio_data: Uint8Array,
    req: AzureSttRequest,
  ): Promise<AzureSttResponse> {
    await this.acquire();
    const lang = req.language ?? "en-US";
    const content_type = req.content_type ??
      "audio/wav; codecs=audio/pcm; samplerate=16000";
    const url = `${this.sttEndpoint()}?language=${encodeURIComponent(lang)}`;
    return vendorJson<AzureSttResponse>("Azure STT", url, {
      method: "POST",
      headers: {
        "Ocp-Apim-Subscription-Key": this.subscription_key,
        "Content-Type": content_type,
      },
      body: audio_data as BodyInit,
    });
  }

  /** List available voices. */
  async listVoices(): Promise<AzureVoice[]> {
    await this.acquire();
    return vendorJson<AzureVoice[]>("Azure voices", this.voicesEndpoint(), {
      method: "GET",
      headers: { "Ocp-Apim-Subscription-Key": this.subscription_key },
    });
  }
}
