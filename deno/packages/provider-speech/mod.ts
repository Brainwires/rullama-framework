/**
 * Audio provider clients — TTS / STT / ASR HTTP wrappers.
 *
 * These are pure HTTP clients that accept/return `Uint8Array` audio
 * payloads. Hardware capture (microphone) and playback (speaker) are
 * intentionally not provided in Deno — the Rust framework handles those
 * via the `rullama-hardware` crate, and Deno consumers should bring
 * their own audio I/O (Web Audio API, WebRTC, etc.).
 *
 * Equivalent to the audio provider modules in Rust's `rullama-providers`:
 * azure_speech, deepgram, elevenlabs, google_tts, murf, cartesia, fish.
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
