/**
 * Azure Cognitive Services Speech API client.
 *
 * Equivalent to Rust's `brainwires_providers::azure_speech` module.
 */

import { RateLimiter } from "../rate_limiter.ts";

/** STT request parameters. */
export interface AzureSttRequest {
  /** Language (e.g., "en-US"). Default "en-US". */
  language?: string;
  /** Content-type header value. Default "audio/wav; codecs=audio/pcm; samplerate=16000". */
  content_type?: string;
}

/** STT response (Azure uses PascalCase on the wire). */
export interface AzureSttResponse {
  RecognitionStatus: string;
  DisplayText?: string;
  Offset?: number;
  Duration?: number;
}

/** An Azure voice entry (PascalCase wire format). */
export interface AzureVoice {
  Name: string;
  DisplayName: string;
  /** e.g., "en-US-JennyNeural". */
  ShortName: string;
  Gender: string;
  Locale: string;
}

/** Azure Speech API client. */
export class AzureSpeechClient {
  readonly region: string;
  private readonly subscription_key: string;
  private rate_limiter: RateLimiter | null = null;

  constructor(subscription_key: string, region: string) {
    this.subscription_key = subscription_key;
    this.region = region;
  }

  withRateLimit(requests_per_minute: number): this {
    this.rate_limiter = new RateLimiter(requests_per_minute);
    return this;
  }

  private async acquire(): Promise<void> {
    if (this.rate_limiter) await this.rate_limiter.acquire();
  }

  ttsEndpoint(): string {
    return `https://${this.region}.tts.speech.microsoft.com/cognitiveservices/v1`;
  }

  sttEndpoint(): string {
    return `https://${this.region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1`;
  }

  voicesEndpoint(): string {
    return `https://${this.region}.tts.speech.microsoft.com/cognitiveservices/voices/list`;
  }

  /** Synthesize speech from SSML. Returns raw audio bytes. */
  async synthesize(ssml: string, output_format: string): Promise<Uint8Array> {
    await this.acquire();
    const res = await fetch(this.ttsEndpoint(), {
      method: "POST",
      headers: {
        "Ocp-Apim-Subscription-Key": this.subscription_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": output_format,
      },
      body: ssml,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Azure TTS API error (${res.status}): ${body}`);
    }
    return new Uint8Array(await res.arrayBuffer());
  }

  /** Synthesize from plain text by wrapping in SSML. */
  synthesizeText(text: string, voice_name: string, output_format: string): Promise<Uint8Array> {
    const ssml = `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
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
    const content_type = req.content_type ?? "audio/wav; codecs=audio/pcm; samplerate=16000";
    const url = `${this.sttEndpoint()}?language=${encodeURIComponent(lang)}`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Ocp-Apim-Subscription-Key": this.subscription_key,
        "Content-Type": content_type,
      },
      body: audio_data,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Azure STT API error (${res.status}): ${body}`);
    }
    return await res.json() as AzureSttResponse;
  }

  /** List available voices. */
  async listVoices(): Promise<AzureVoice[]> {
    await this.acquire();
    const res = await fetch(this.voicesEndpoint(), {
      method: "GET",
      headers: { "Ocp-Apim-Subscription-Key": this.subscription_key },
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Azure voices API error (${res.status}): ${body}`);
    }
    return await res.json() as AzureVoice[];
  }
}
