# @rullama/provider-speech

Cloud TTS / STT / ASR HTTP clients: Azure Speech, Cartesia, Deepgram,
ElevenLabs, Fish Audio, Google Cloud TTS, Murf.

Extracted from the old `@rullama/providers` package in v0.11.0 to mirror Rust's
`rullama-provider-speech` crate. The speech clients are independent of the chat
provider stack (`@rullama/provider`) so consumers can pull in just one.

All clients are plain `fetch()` wrappers built from an API key that accept and
return `Uint8Array` audio payloads (Google TTS returns base64 per its API; Murf
returns a URL you fetch with `downloadAudio`). Every client has
`withRateLimit(requestsPerMinute)`. Microphone capture and speaker playback are
intentionally not provided in Deno -- bring your own Web Audio / WebRTC I/O.

| Client              | Methods                                                   |
| ------------------- | --------------------------------------------------------- |
| `AzureSpeechClient` | `synthesize`, `synthesizeText`, `recognize`, `listVoices` |
| `CartesiaClient`    | `ttsBytes`                                                |
| `DeepgramClient`    | `speak` (TTS), `listen` (STT)                             |
| `ElevenLabsClient`  | `textToSpeech`, `speechToText`, `listVoices`              |
| `FishClient`        | `tts`, `asr`                                              |
| `GoogleTtsClient`   | `synthesize`, `listVoices`                                |
| `MurfClient`        | `generateSpeech`, `downloadAudio`, `listVoices`           |

## Install

```sh
deno add jsr:@rullama/provider-speech
```

## Quick Example

```ts
import { DeepgramClient, ElevenLabsClient } from "@rullama/provider-speech";

// Text to speech (ElevenLabs) -> raw audio bytes
const tts = new ElevenLabsClient(Deno.env.get("ELEVENLABS_API_KEY")!)
  .withRateLimit(30);
const mp3 = await tts.textToSpeech("21m00Tcm4TlvDq8ikWAM", {
  text: "Hello from rullama.",
  model_id: "eleven_multilingual_v2",
  output_format: "mp3_44100_128",
});
await Deno.writeFile("hello.mp3", mp3);

// Speech to text (Deepgram)
const stt = new DeepgramClient(Deno.env.get("DEEPGRAM_API_KEY")!);
const audio = await Deno.readFile("hello.mp3");
const transcript = await stt.listen(audio, { model: "nova-2" });
console.log(transcript.results?.channels[0]?.alternatives[0]?.transcript);
```
