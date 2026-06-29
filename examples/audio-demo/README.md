# Brainwires Audio Studio

A cross-platform desktop GUI for demoing text-to-speech and speech-to-text across all [Brainwires Framework](../../) audio providers. Built with [Avalonia UI](https://avaloniaui.net/) (.NET 9) consuming Rust via [UniFFI](https://mozilla.github.io/uniffi-rs/) bindings.

## Screenshots

The app has three tabs:

- **Text-to-Speech** — Select a provider and voice, enter text, synthesize and play back audio
- **Speech-to-Text** — Record from your microphone, transcribe with any STT provider
- **Settings** — Configure API keys per provider, test connections

## Prerequisites

- **Rust 1.91+** (for building the native FFI library)
- **.NET 9 SDK** ([download](https://dotnet.microsoft.com/download/dotnet/9.0))
- **uniffi-bindgen-cs** (for generating C# bindings from Rust)

```bash
# Install the C# binding generator
cargo install uniffi-bindgen-cs --git https://github.com/aspect-build/uniffi-bindgen-cs
```

## Quick Start

```bash
# 1. Build the Rust native library and generate C# bindings
./build-native.sh release

# 2. Build and run the Avalonia app
dotnet run --project AudioDemo.Desktop
```

## Project Structure

```
audio-demo/
├── AudioDemo.sln                  # .NET solution
├── build-native.sh                # Builds Rust + generates C# bindings
├── BrainwiresAudio/               # C# class library (wraps FFI)
│   ├── AudioService.cs            # High-level API over generated bindings
│   ├── Generated/                 # UniFFI-generated C# code (gitignored)
│   └── runtimes/                  # Native .so/.dll/.dylib (gitignored)
├── AudioDemo/                     # Avalonia UI (shared)
│   ├── ViewModels/                # MVVM view models
│   │   ├── MainWindowViewModel.cs
│   │   ├── TtsViewModel.cs
│   │   ├── SttViewModel.cs
│   │   └── SettingsViewModel.cs
│   └── Views/                     # AXAML views
│       ├── MainWindow.axaml
│       ├── TtsView.axaml
│       ├── SttView.axaml
│       └── SettingsView.axaml
└── AudioDemo.Desktop/             # Desktop entry point
    └── Program.cs
```

## Supported Providers

| Provider | TTS | STT | API Key Env Var |
|----------|-----|-----|-----------------|
| OpenAI (TTS-1 / Whisper) | Yes | Yes | `OPENAI_API_KEY` |
| OpenAI Responses API | Yes | Yes | `OPENAI_API_KEY` |
| ElevenLabs | Yes | Yes | `ELEVENLABS_API_KEY` |
| Deepgram (Aura / Nova) | Yes | Yes | `DEEPGRAM_API_KEY` |
| Google Cloud TTS | Yes | No | `GOOGLE_API_KEY` |
| Azure Cognitive Services | Yes | Yes | `AZURE_SPEECH_KEY` + region |
| Fish Audio | Yes | Yes | `FISH_API_KEY` |
| Cartesia (Sonic) | Yes | No | `CARTESIA_API_KEY` |
| Murf AI | Yes | No | `MURF_API_KEY` |

## How It Works

```text
┌────────────────────┐
│  Avalonia UI (C#)  │
│  MVVM ViewModels   │
└─────────┬──────────┘
          │  P/Invoke
┌─────────▼──────────┐
│  UniFFI Bindings   │
│  (generated .cs)   │
└─────────┬──────────┘
          │  FFI (cdylib)
┌─────────▼──────────┐
│  audio-demo-ffi    │
│  (Rust bridge)     │
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│  brainwires-hardware  │
│  + brainwires-     │
│    providers       │
└────────────────────┘
```

The Rust FFI crate (`audio-demo-ffi`) wraps the async `brainwires-hardware` API behind synchronous UniFFI-exported functions using an internal Tokio runtime. Provider instances are managed via opaque `u64` handles in a static registry.

## Building for Other Platforms

The `build-native.sh` script auto-detects the current OS. For cross-compilation:

```bash
# Cross-compile for Windows (from Linux)
cargo build --release -p audio-demo-ffi --target x86_64-pc-windows-gnu

# Cross-compile for macOS (requires appropriate toolchain)
cargo build --release -p audio-demo-ffi --target aarch64-apple-darwin
```

Copy the resulting library to the appropriate `runtimes/` directory before building the .NET app.

## Development

```bash
# Build Rust FFI (debug, faster)
./build-native.sh

# Watch for C# changes and rebuild
dotnet watch --project AudioDemo.Desktop
```

## License

MIT OR Apache-2.0
