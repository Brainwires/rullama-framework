# brainwires-hardware

Hardware I/O for the [Brainwires Agent Framework](https://github.com/Brainwires/brainwires-framework).

Provides a unified hardware abstraction layer covering audio, GPIO, Bluetooth, network hardware, camera, and USB — all behind opt-in feature flags so you only compile what you need.

Home automation protocols (Matter, Zigbee, Z-Wave, Thread) moved to the standalone `future/home-automation/brainwires-homeauto` workspace in 0.11.

## Modules

| Module | Feature | Description |
|--------|---------|-------------|
| `audio` | `audio` | Audio capture/playback, STT, TTS (16 cloud providers + local Whisper) |
| `audio/vad` | *(always)* / `vad` | Voice activity detection — `EnergyVad` (always) + `WebRtcVad` (`vad`) |
| `audio/wake_word` | `wake-word` | Wake word detection — `EnergyTriggerDetector` + optional ML backends |
| `audio/assistant` | `voice-assistant` | End-to-end voice assistant pipeline |
| `gpio` | `gpio` | GPIO pin management with safety allow-lists and PWM (Linux) |
| `bluetooth` | `bluetooth` | BLE advertisement scanning and adapter enumeration |
| `network` | `network` | NIC enumeration, IP config, ARP host discovery, port scanning |
| `camera` | `camera` | Webcam/camera frame capture (V4L2/AVFoundation/MSMF) |
| `usb` | `usb` | Raw USB device enumeration and transfers (no libusb) |

## Getting started

```toml
[dependencies]
# Pick only what you need:
brainwires-hardware = { version = "0.11", features = ["audio"] }
brainwires-hardware = { version = "0.11", features = ["gpio"] }
brainwires-hardware = { version = "0.11", features = ["bluetooth"] }
brainwires-hardware = { version = "0.11", features = ["network"] }

# Or enable everything:
brainwires-hardware = { version = "0.11", features = ["full"] }
```

## Feature flags

| Feature | Description |
|---------|-------------|
| `audio` | Hardware audio I/O via CPAL + 16 cloud STT/TTS providers |
| `flac` | FLAC encode/decode |
| `local-stt` | Local Whisper STT inference via whisper-rs (heavy dep, opt-in) |
| `vad` | WebRTC VAD algorithm (`EnergyVad` is always available with `audio`) |
| `wake-word` | Wake word detection — `EnergyTriggerDetector` (zero deps) |
| `wake-word-rustpotter` | `RustpotterDetector` — pure-Rust ML wake word (opt-in, see notes) |
| `voice-assistant` | Full pipeline: capture → wake word → VAD → STT → handler → TTS |
| `gpio` | GPIO pin control via Linux character device API (`gpio-cdev`) |
| `bluetooth` | BLE scanning and adapter enumeration via `btleplug` |
| `network` | NIC enumeration, IP config, ARP discovery, port scanning |
| `camera` | Webcam/camera capture via nokhwa (V4L2/AVFoundation/MSMF) |
| `usb` | Raw USB device access and transfers via nusb (no libusb) |
| `full` | All features (except `local-stt`, `wake-word-rustpotter`) |

## Audio

Supports hardware capture and playback via [CPAL](https://crates.io/crates/cpal), plus cloud STT/TTS integrations:

**STT:** OpenAI, Azure, Deepgram, ElevenLabs, Fish Audio
**TTS:** OpenAI, Azure, Deepgram, ElevenLabs, Fish Audio, Google, Murf, Cartesia

```rust
use brainwires_hardware::{TextToSpeech, TtsOptions, OutputFormat};

let tts = OpenAiTts::new(api_key);
let audio = tts.synthesize("Hello, world!", &TtsOptions::default()).await?;
```

## GPIO (Linux)

Safe GPIO access with explicit allow-lists — no pin can be used unless it appears in the configured policy.

```rust
use brainwires_hardware::{GpioPinManager, GpioSafetyPolicy};
use brainwires_hardware::gpio::device::GpioDirection;

let mut manager = GpioPinManager::from_config(&config);
let pin = manager.acquire(0, 17, GpioDirection::Output, "my-agent")?;
```

## Bluetooth

Cross-platform BLE scanning using [btleplug](https://crates.io/crates/btleplug):

```rust
use brainwires_hardware::bluetooth;
use std::time::Duration;

let devices = bluetooth::scan_ble(Duration::from_secs(5)).await;
for d in &devices {
    println!("{} — {:?}", d.address, d.name);
}
```

## Network

```rust
use brainwires_hardware::network;
use std::time::Duration;

// List interfaces
for iface in network::list_interfaces() {
    println!("{} ({:?})", iface.name, iface.kind);
}

// IP config with gateways
for cfg in network::get_ip_configs() {
    println!("{}: gateway={:?}", cfg.interface, cfg.gateway);
}

// Port scan
let results = network::scan_common_ports(
    "192.168.1.1".parse().unwrap(),
    Duration::from_millis(500),
).await;

// ARP host discovery (requires CAP_NET_RAW)
let hosts = network::arp_scan("192.168.1.0/24".parse().unwrap()).await;
```

## Voice Activity Detection

`EnergyVad` is always available (no extra feature needed beyond `audio`). `WebRtcVad` requires the `vad` feature.

```rust
use brainwires_hardware::{EnergyVad, VoiceActivityDetector};

let vad = EnergyVad::default(); // -40 dBFS threshold
if vad.is_speech(&audio_buffer) {
    println!("Speech detected!");
}
```

## Wake Word Detection

```rust
use brainwires_hardware::{EnergyTriggerDetector, WakeWordDetector};

let mut detector = EnergyTriggerDetector::new(-20.0, 3, 16_000);
// Feed 30 ms i16 frames from the mic:
if let Some(detection) = detector.process_frame(&frame) {
    println!("Wake trigger! score={:.2}", detection.score);
}
```

> **Note on `wake-word-rustpotter`.** The workspace `[patch.crates-io]` table
> redirects `rustpotter` to
> [Brainwires/rustpotter@`candle-0.10`](https://github.com/Brainwires/rustpotter)
> (the prerelease `3.0.3-candle-0.10`), which bumps `candle-core`/`candle-nn`
> to 0.10 so the dep aligns with the rest of the workspace. With that patch
> in place `cargo … --all-features` builds clean, including this crate.

## Voice Assistant Pipeline

```rust
use brainwires_hardware::{VoiceAssistant, VoiceAssistantHandler, VoiceAssistantConfig};
use brainwires_hardware::audio::types::Transcript;
use async_trait::async_trait;

struct MyHandler;

#[async_trait]
impl VoiceAssistantHandler for MyHandler {
    async fn on_speech(&self, transcript: &Transcript) -> Option<String> {
        println!("You said: {}", transcript.text);
        Some("I heard you!".to_string())
    }
}

// Build and run
let mut assistant = VoiceAssistant::builder(capture, stt)
    .with_playback(playback)
    .with_tts(tts)
    .with_wake_word(Box::new(EnergyTriggerDetector::default()))
    .build();

assistant.run(&MyHandler).await?;
```

## Home Automation

Home automation protocols (Matter, Zigbee, Z-Wave, Thread) and the `matter-tool` CLI moved to the standalone `future/home-automation/` workspace in 0.11. See `future/home-automation/README.md`.

## Migration from brainwires-audio

```toml
# Before
brainwires-audio = "0.10"

# After
brainwires-hardware = { version = "0.11", features = ["audio"] }
```

All public types and traits are re-exported from the crate root — existing code using
`brainwires_audio::*` can switch to `brainwires_hardware::*` with no further changes.

## Examples

```bash
cargo run -p brainwires-hardware --example text_to_speech --features audio
cargo run -p brainwires-hardware --example bluetooth_scan --features bluetooth
cargo run -p brainwires-hardware --example network_interfaces --features network
cargo run -p brainwires-hardware --example port_scan --features network -- 192.168.1.1
sudo cargo run -p brainwires-hardware --example host_discovery --features network -- 192.168.1.0/24

# Wake word demo (prints detections from mic)
cargo run -p brainwires-hardware --example wake_word_demo --features wake-word

# Full voice assistant demo (requires OPENAI_API_KEY)
cargo run -p brainwires-hardware --example voice_assistant --features voice-assistant

# Standalone voice assistant binary
cargo run -p voice-assistant -- --list-devices
cargo run -p voice-assistant -- --verbose
```
