# brainwires-provider-speech

[![Crates.io](https://img.shields.io/crates/v/brainwires-provider-speech.svg)](https://crates.io/crates/brainwires-provider-speech)
[![Documentation](https://docs.rs/brainwires-provider-speech/badge.svg)](https://docs.rs/brainwires-provider-speech)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework)

Speech (TTS / STT) provider clients for the Brainwires Agent Framework.

Standalone so consumers (typically `brainwires-hardware`'s audio surface
and the chat-pwa wasm bridge) can pull just the speech clients without
dragging in the LLM provider stack (candle / llama.cpp / huggingface,
aws-sigv4, gcp_auth, …) that lives in
[`brainwires-provider`](https://crates.io/crates/brainwires-provider).

## What lives here

### Native cloud providers (`native` feature)

| Module | Provider |
|---|---|
| `azure_speech::AzureSpeechClient` | Microsoft Azure Cognitive Services Speech |
| `cartesia::CartesiaClient` | Cartesia TTS |
| `deepgram::DeepgramClient` | Deepgram TTS / STT |
| `elevenlabs::ElevenLabsClient` | ElevenLabs TTS / STT |
| `fish::FishClient` | Fish Audio TTS / ASR |
| `google_tts::GoogleTtsClient` | Google Cloud Text-to-Speech |
| `murf::MurfClient` | Murf AI TTS |

### Browser-native (`web-speech` feature, `wasm32` only)

`web_speech::*` — `speechSynthesis` (TTS) and `SpeechRecognition` (STT).

## Usage

```toml
[dependencies]
brainwires-provider-speech = { version = "0.11", features = ["native"] }
```

```rust,ignore
use brainwires_provider_speech::ElevenLabsClient;

let client = ElevenLabsClient::new("api-key");
// ...
```

## See also

- [`brainwires-provider`](https://crates.io/crates/brainwires-provider) —
  LLM chat clients (sibling).
- [`brainwires-hardware`](https://crates.io/crates/brainwires-hardware) —
  primary consumer (its audio API uses these clients).
- [`brainwires`](https://crates.io/crates/brainwires) — umbrella facade
  with `web-speech` feature for browser builds.

## License

MIT OR Apache-2.0
