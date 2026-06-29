# rullama-hardware

Hardware I/O for the [Brainwires Agent Framework](https://github.com/Brainwires/rullama-framework).

Provides a unified hardware abstraction layer covering audio, GPIO, Bluetooth, network hardware, camera, and USB — all behind opt-in feature flags so you only compile what you need.

Home automation protocols (Matter, Zigbee, Z-Wave, Thread) moved to the standalone `future/home-automation/rullama-homeauto` workspace in 0.11.

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
rullama-hardware = { version = "0.11", features = ["audio"] }
rullama-hardware = { version = "0.11", features = ["gpio"] }
rullama-hardware = { version = "0.11", features = ["bluetooth"] }
rullama-hardware = { version = "0.11", features = ["network"] }

# Or enable everything:
rullama-hardware = { version = "0.11", features = ["full"] }
```

## Feature flags

| Feature | Description |
|---------|-------------|
| `audio` | Hardware audio I/O via CPAL + 16 cloud STT/TTS providers |
| `flac` | FLAC encode/decode |
| `local-stt` | Local Whisper STT inference via whisper-rs (heavy dep, opt-in) |
| `vad` | WebRTC VAD algorithm (`EnergyVad` is always available with `audio`) |
| `wake-word` | Wake word detection — `EnergyTriggerDetector` (zero deps) |
| `wake-word-dtw` | `DtwWakeWordDetector` — in-house DTW+MFCC wake word (speaker-dependent, see notes) |
| `voice-assistant` | Full pipeline: capture → wake word → VAD → STT → handler → TTS |
| `gpio` | GPIO pin control via Linux character device API (`gpio-cdev`) |
| `bluetooth` | BLE scanning and adapter enumeration via `btleplug` |
| `network` | NIC enumeration, IP config, ARP discovery, port scanning |
| `camera` | Webcam/camera capture via nokhwa (V4L2/AVFoundation/MSMF) |
| `usb` | Raw USB device access and transfers via nusb (no libusb) |
| `full` | All features (except `local-stt`, `wake-word-dtw`) |

## Audio

Supports hardware capture and playback via [CPAL](https://crates.io/crates/cpal), plus cloud STT/TTS integrations:

**STT:** OpenAI, Azure, Deepgram, ElevenLabs, Fish Audio
**TTS:** OpenAI, Azure, Deepgram, ElevenLabs, Fish Audio, Google, Murf, Cartesia

```rust
use rullama_hardware::{TextToSpeech, TtsOptions, OutputFormat};

let tts = OpenAiTts::new(api_key);
let audio = tts.synthesize("Hello, world!", &TtsOptions::default()).await?;
```

## GPIO (Linux)

Safe GPIO access with explicit allow-lists — no pin can be used unless it appears in the configured policy.

```rust
use rullama_hardware::{GpioPinManager, GpioSafetyPolicy};
use rullama_hardware::gpio::device::GpioDirection;

let mut manager = GpioPinManager::from_config(&config);
let pin = manager.acquire(0, 17, GpioDirection::Output, "my-agent")?;
```

## Bluetooth

Cross-platform BLE scanning using [btleplug](https://crates.io/crates/btleplug):

```rust
use rullama_hardware::bluetooth;
use std::time::Duration;

let devices = bluetooth::scan_ble(Duration::from_secs(5)).await;
for d in &devices {
    println!("{} — {:?}", d.address, d.name);
}
```

## Network

```rust
use rullama_hardware::network;
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
use rullama_hardware::{EnergyVad, VoiceActivityDetector};

let vad = EnergyVad::default(); // -40 dBFS threshold
if vad.is_speech(&audio_buffer) {
    println!("Speech detected!");
}
```

## Wake Word Detection

```rust
use rullama_hardware::{EnergyTriggerDetector, WakeWordDetector};

let mut detector = EnergyTriggerDetector::new(-20.0, 3, 16_000);
// Feed 30 ms i16 frames from the mic:
if let Some(detection) = detector.process_frame(&frame) {
    println!("Wake trigger! score={:.2}", detection.score);
}
```

> **Note on `wake-word-dtw`.** This is an in-house clean-room implementation —
> classical DTW over MFCC features, pure-Rust DSP, no ML framework, no
> third-party wake-word library. Speaker-dependent: the caller records the
> wake phrase ≥3 times via `DtwWakeWordDetector::enroll_template`, and the
> detector then compares incoming audio against those references. See
> `examples/wake_word_demo.rs` for the interactive enrollment ceremony.

## Voice Assistant Pipeline

```rust
use rullama_hardware::{VoiceAssistant, VoiceAssistantHandler, VoiceAssistantConfig};
use rullama_hardware::audio::types::Transcript;
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

## Migration from rullama-audio

```toml
# Before
rullama-audio = "0.10"

# After
rullama-hardware = { version = "0.11", features = ["audio"] }
```

All public types and traits are re-exported from the crate root — existing code using
`rullama_audio::*` can switch to `rullama_hardware::*` with no further changes.

## Examples

```bash
cargo run -p rullama-hardware --example text_to_speech --features audio
cargo run -p rullama-hardware --example bluetooth_scan --features bluetooth
cargo run -p rullama-hardware --example network_interfaces --features network
cargo run -p rullama-hardware --example port_scan --features network -- 192.168.1.1
sudo cargo run -p rullama-hardware --example host_discovery --features network -- 192.168.1.0/24

# Wake word demo (prints detections from mic)
cargo run -p rullama-hardware --example wake_word_demo --features wake-word

# Full voice assistant demo (requires OPENAI_API_KEY)
cargo run -p rullama-hardware --example voice_assistant --features voice-assistant

# Standalone voice assistant binary
cargo run -p voice-assistant -- --list-devices
cargo run -p voice-assistant -- --verbose
```
