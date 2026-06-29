# @rullama/provider-speech

Cloud TTS/STT/ASR HTTP clients: Azure Speech, Cartesia, Deepgram, ElevenLabs,
Fish Audio, Google TTS, Murf.

Extracted from `@rullama/providers` in v0.11.0 to mirror Rust's
`rullama-provider-speech` crate. Speech clients are independent of the chat
LLM provider stack so consumers can pull just one in.

All clients are pure HTTP wrappers that accept/return `Uint8Array` audio
payloads. Hardware capture (microphone) and playback (speaker) are intentionally
not provided in Deno — bring your own Web Audio / WebRTC IO.
