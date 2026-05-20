# voice-assistant

Personal voice assistant built on the [Brainwires Framework](https://github.com/Brainwires/brainwires-framework).

## Overview

`voice-assistant` is a standalone binary that listens on a microphone, transcribes speech with an STT provider, sends it to an LLM, and plays back the response via TTS — all wired through `brainwires-hardware`'s audio pipeline.

## Features

- Continuous microphone capture via CPAL
- Speech-to-text via OpenAI Whisper (configurable)
- LLM response via any `brainwires-provider` backend
- Text-to-speech playback via OpenAI TTS (configurable)
- Optional wake-word detection (Rustpotter or Picovoice Porcupine)
- TOML config file at `~/.config/voice-assistant/config.toml`
- Optional SQLite-backed conversation history via `brainwires-session`

## Usage

```sh
cargo build --release -p voice-assistant
./target/release/voice-assistant --config ~/.config/voice-assistant/config.toml
```

Useful flags:

| Flag | Description |
|------|-------------|
| `-c, --config <FILE>` | Path to TOML config (default: `~/.config/voice-assistant/config.toml`) |
| `--list-devices` | Print available microphones and speakers, then exit |
| `--wake-word <FILE>` | Override the wake-word model path set in config |
| `-v, --verbose` | Enable debug logging |

## Feature flags

| Flag | Description |
|------|-------------|
| `wake-word` | Enable wake-word detection (engine auto-selected) |
| `wake-word-rustpotter` | Use Rustpotter for wake-word detection |

## Configuration

Default path: `~/.config/voice-assistant/config.toml`. All fields are optional; omitting the file runs on the defaults shown below. Loaded via `VaConfig::load_from(&Path)` in [`src/config.rs`](src/config.rs).

```toml
# OpenAI API key. Falls back to the OPENAI_API_KEY env var if omitted.
# openai_api_key = "sk-..."

# STT / TTS / LLM
stt_model   = "whisper-1"
tts_model   = "tts-1"
tts_voice   = "alloy"
llm_model   = "gpt-4o-mini"
tts_enabled = true

# Wake word (optional — omit for always-on listening)
# wake_word_model     = "/path/to/model.rpw"
wake_word_threshold = 0.5

# Voice-activity detection
silence_threshold_db = -40.0
silence_duration_ms  = 800
max_record_secs      = 30.0

# Audio devices. None = system default. Use --list-devices to see names.
# microphone = "Blue Yeti"
# speaker    = "Built-in Output"

# Persona
system_prompt = "You are a helpful voice assistant. Keep responses concise and conversational."

# Session persistence (brainwires-session)
session_id = "voice-assistant"
# session_db = "/home/me/.local/share/voice-assistant/history.sqlite"

# Hard budgets for a single run. None = unbounded.
# max_usd_cents = 50
# max_tokens    = 50000
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (unless set in config) | Used for Whisper (STT), TTS, and the LLM |
| `RUST_LOG` | No | Standard `tracing` filter override (e.g. `voice_assistant=debug`) |

If both `openai_api_key` in the TOML and `OPENAI_API_KEY` are set, the config value wins.

## Troubleshooting

### List the audio devices CPAL sees

```sh
voice-assistant --list-devices
```

Lines prefixed with `*` are the system default for that direction. Use the exact device name as the `microphone` / `speaker` field in config.

### "No OpenAI API key found"

Neither `openai_api_key` nor `OPENAI_API_KEY` was set. Add one to the config file or export the env var.

### "Wake word model not found" / wake word never fires

- The `wake_word_model` path does not exist. Confirm with `ls -l` and use an absolute path.
- The binary was built without the `wake-word` (or `wake-word-rustpotter`) feature flag.
- `wake_word_threshold` is too high. Try lowering to `0.4` and watch debug logs (`-v`).

### Microphone not detected / wrong device picked up

- Some Linux setups ship PipeWire with multiple virtual devices — run `--list-devices` and pick an explicit one rather than relying on the default.
- If CPAL reports no input devices, check distro-level permissions (`pw-cli ls Node`, membership in the `audio` group).

### No audio playback

Verify the speaker name matches `--list-devices` output exactly, including trailing whitespace. If `tts_enabled = false`, responses are text-only.

### Silence never ends / assistant keeps recording

Tune `silence_threshold_db` (raise toward `-30.0` for noisy rooms) and `silence_duration_ms` (shorten for snappier turn-taking).

## License

MIT OR Apache-2.0
