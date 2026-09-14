/**
 * Cloud speech clients for rullama — TTS / STT / ASR over plain `fetch()`:
 * Azure Speech, Cartesia, Deepgram, ElevenLabs, Fish Audio, Google Cloud TTS
 * and Murf. Every client takes an API key, accepts and returns `Uint8Array`
 * audio payloads (Google TTS returns base64 per its API contract; Murf returns a
 * download URL) and can be rate-limited with `withRateLimit`.
 * Microphone capture and speaker playback are intentionally not provided —
 * the Rust `rullama-hardware` crate covers those; bring your own audio I/O
 * (Web Audio, WebRTC) from Deno.
 * Equivalent to Rust's `rullama-provider-speech` crate.
 *
 * @module
 */

export {
  AzureSpeechClient,
  type AzureSttRequest,
  type AzureSttResponse,
  type AzureVoice,
} from "./azure_speech.ts";

export {
  DEEPGRAM_API_BASE,
  type DeepgramAlternative,
  type DeepgramChannel,
  DeepgramClient,
  type DeepgramListenRequest,
  type DeepgramListenResponse,
  type DeepgramResults,
  type DeepgramSpeakRequest,
  type DeepgramWord,
} from "./deepgram.ts";

export {
  ELEVENLABS_API_BASE,
  ElevenLabsClient,
  type ElevenLabsSttRequest,
  type ElevenLabsSttResponse,
  type ElevenLabsTtsRequest,
  type ElevenLabsVoice,
  type ElevenLabsVoiceSettings,
  type ElevenLabsVoicesResponse,
  serializeTtsRequest as elevenLabsSerializeTtsRequest,
} from "./elevenlabs.ts";

export {
  GOOGLE_TTS_API_BASE,
  type GoogleTtsAudioConfig,
  GoogleTtsClient,
  type GoogleTtsInput,
  type GoogleTtsSynthesizeRequest,
  type GoogleTtsSynthesizeResponse,
  type GoogleTtsVoiceEntry,
  type GoogleTtsVoiceSelection,
  type GoogleTtsVoicesResponse,
} from "./google_tts.ts";

export {
  MURF_API_BASE,
  MurfClient,
  type MurfGenerateRequest,
  type MurfGenerateResponse,
  type MurfVoice,
  type MurfVoicesResponse,
} from "./murf.ts";

export {
  CARTESIA_API_BASE,
  CARTESIA_VERSION,
  CartesiaClient,
  type CartesiaOutputFormat,
  type CartesiaTtsRequest,
  type CartesiaVoice,
} from "./cartesia.ts";

export {
  FISH_API_BASE,
  type FishAsrRequest,
  type FishAsrResponse,
  FishClient,
  type FishTtsRequest,
} from "./fish.ts";
