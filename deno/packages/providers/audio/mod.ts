/**
 * Audio provider clients — TTS / STT / ASR HTTP wrappers.
 *
 * These are pure HTTP clients that accept/return `Uint8Array` audio
 * payloads. Hardware capture (microphone) and playback (speaker) are
 * intentionally not provided in Deno — the Rust framework handles those
 * via the `brainwires-hardware` crate, and Deno consumers should bring
 * their own audio I/O (Web Audio API, WebRTC, etc.).
 *
 * Equivalent to the audio provider modules in Rust's `brainwires-providers`:
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
  DeepgramClient,
  type DeepgramAlternative,
  type DeepgramChannel,
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
  type ElevenLabsVoicesResponse,
  type ElevenLabsVoiceSettings,
  serializeTtsRequest as elevenLabsSerializeTtsRequest,
} from "./elevenlabs.ts";

export {
  GOOGLE_TTS_API_BASE,
  GoogleTtsClient,
  type GoogleTtsAudioConfig,
  type GoogleTtsInput,
  type GoogleTtsSynthesizeRequest,
  type GoogleTtsSynthesizeResponse,
  type GoogleTtsVoiceEntry,
  type GoogleTtsVoicesResponse,
  type GoogleTtsVoiceSelection,
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
  FishClient,
  type FishAsrRequest,
  type FishAsrResponse,
  type FishTtsRequest,
} from "./fish.ts";
