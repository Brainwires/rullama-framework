# rullama-audio-ffi

A small C ABI over [rullama-hardware](../rullama-hardware/README.md)'s local
audio stack, built as a shared library for hosts that load native code at
runtime, such as Deno's `Deno.dlopen` (Qutty uses it this way).

It covers:

- **Capture**: the default microphone through cpal, in the device's native
  format, mixed to mono and capped at 120 s (buffered at the device rate,
  about 23 MB at 48 kHz), then resampled to 16 kHz `f32` when it stops.
- **Speech-to-text**: whisper.cpp through whisper-rs. English, greedy
  decoding, `min(4, cores)` threads, no timestamps, whisper's own logging
  silenced.
- **Silence trimming**: WebRTC VAD (least aggressive mode) cuts leading and
  trailing silence and keeps 250 ms of padding on each side. A capture with
  less than ~300 ms of speech gives an empty transcript, not an error.
  Building without the `vad` feature falls back to rullama-hardware's energy
  VAD.
- **Playback**: WAV files on the default output device through cpal. Playback
  can be stopped.

Nothing is sent over the network.

## Build

```sh
cargo build -p rullama-audio-ffi --release
# -> target/release/librullama_audio_ffi.dylib (macOS)
# -> target/release/librullama_audio_ffi.so    (Linux)
```

Build requirements: a C/C++ toolchain, `cmake` (for whisper.cpp) and libclang
(whisper-rs-sys runs bindgen; set `WHISPER_DONT_GENERATE_BINDINGS=1` to use its
bundled bindings instead). macOS has all three with the Xcode command line
tools plus `cmake`. On Debian/Ubuntu: `cmake clang libclang-dev` and
`libasound2-dev` (ALSA, for cpal).

On macOS the dylib's install name is `@rpath/librullama_audio_ffi.dylib`, so
it can be copied into an app's support folder and loaded by absolute path.
Whisper runs on the CPU (using Accelerate). The optional `metal` feature
passes through to `whisper-rs/metal`.

| Feature | Default | Effect |
|---|---|---|
| `vad` | yes | WebRTC VAD trimming (otherwise energy VAD) |
| `metal` | no | whisper.cpp on the GPU via Metal |

Models are not bundled. Use any ggml whisper model, e.g.
`ggml-base.en-q5_1.bin` from
[ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp).

## ABI

The header is [`include/rullama_audio.h`](include/rullama_audio.h); a test
(`tests/abi.rs`) checks that its prototypes and `#define`s match the Rust
exports. Paths passed in are NUL-terminated UTF-8. Return codes: `>= 0` ok,
`-1` error (message via `rla_last_error`), `-2` wrong state, `-3` no model
loaded, `-4` no audio device.

**Ownership.** The library never returns memory for the caller to free, so
there are no free functions. `rla_version` returns a static string. Every
other output (error text, transcripts) is copied into a caller-owned
`(buf, cap)` buffer: at most `cap - 1` bytes plus a NUL, never splitting a
UTF-8 character. The return value is the full byte length without the NUL, so
`len >= cap` means the output was truncated; call again with a bigger buffer,
or pass `NULL, 0` first to ask for the size. Use the returned length rather
than looking for the NUL. Transcripts are valid UTF-8 (invalid bytes from the
model are replaced with U+FFFD).

**Threading.** Every function catches panics at the boundary (a panic returns
`-1`, or the neutral value for functions without an error code, and sets an
`internal panic: ...` last error). All state is held behind mutexes, so any
function can be called from any thread, including Deno `nonblocking` calls.
There is one engine per process. The last-error slot is process-wide, not per
thread (a thread-local slot would lose errors from Deno's `nonblocking` calls,
which run on another thread): read it right after the failing call, and
serialize calls if exact attribution matters. `rla_capture_start` and
`rla_play_wav` can block for up to 5 s while a device opens;
`rla_transcribe_wav` blocks for the whole transcription.

**Lifetime.** There is no init function; state is created on first use. Call
`rla_shutdown` before unloading the library (`Deno.dlopen(...).close()`): it
stops capture and playback (streams closed, threads joined), aborts and joins
the transcription worker, makes any `rla_transcribe_wav` running on another
thread return `-1` promptly, and unloads the model. Wait for such a call to
return before unloading. The library stays usable after `rla_shutdown`; load a
model again to transcribe.

**Limits.** Captures stop buffering at 120 s (what was captured is kept). WAV
files over 256 MiB are rejected by `rla_transcribe_wav` and `rla_play_wav`.
`rla_cancel` and `rla_shutdown` abort a whisper inference in flight through
whisper.cpp's abort callback, so abandoned transcriptions do not pile up.

```c
const char* rla_version(void);                          // static, "rullama-audio-ffi <version>"
int32_t  rla_last_error(uint8_t* buf, size_t cap);      // copy last error (truncated, NUL-terminated); returns full byte length
int32_t  rla_stt_load(const char* model_path);          // load or replace the whisper ggml model; 0 ok
int32_t  rla_stt_loaded(void);                          // 1 loaded, 0 not
int32_t  rla_capture_start(void);                       // open default input; -2 if not idle/done; -4 no device
int32_t  rla_capture_stop(void);                        // stop, transcribe on a worker (state -> 2); -2 not capturing; -3 no model
int32_t  rla_capture_stop_wav(const char* wav_path);    // stop WITHOUT transcribing, write trimmed 16 kHz mono 16-bit WAV; returns ms written; -2 / -1
int32_t  rla_cancel(void);                              // stop capture / abort transcription, state -> 0; always 0
int32_t  rla_state(void);                               // 0 idle, 1 capturing, 2 transcribing, 3 done, 4 error
float    rla_level(void);                               // input RMS 0..1 of the last ~100 ms while capturing, else 0
uint32_t rla_captured_ms(void);                         // ms captured in this capture
int32_t  rla_take_transcript(uint8_t* buf, size_t cap); // state 3: copy, return length, consume only if len < cap; -2 otherwise; state 4: -1 and reset
int32_t  rla_transcribe_wav(const char* wav_path, uint8_t* buf, size_t cap); // synchronous WAV -> text; does not touch capture state
int32_t  rla_play_wav(const char* wav_path);            // play on default output, returns at once (0), stops previous playback
int32_t  rla_play_stop(void);                           // stop playback; 0
int32_t  rla_playing(void);                             // 1 while playing
int32_t  rla_shutdown(void);                            // stop everything, join threads, unload model (call before unloading); 0
```

### State machine

```
            rla_capture_start            rla_capture_stop
  0 idle  ───────────────────►  1 capturing ───────────────► 2 transcribing ──► 3 done
    ▲  ▲                           │   │                          │      └────► 4 error
    │  └── rla_capture_stop_wav ───┘   │                          │
    └──────────── rla_cancel (from any state) ────────────────────┘
  3 done  ── rla_take_transcript (fits) ──► 0      3 done ── rla_capture_start ──► 1
  4 error ── rla_take_transcript (-1)   ──► 0
```

Details:

- `rla_capture_stop` without a model returns `-3` and leaves the capture
  running, so the caller can load a model and stop again, or cancel.
- `rla_capture_start` from state 3 discards the transcript that was not
  taken.
- `rla_captured_ms` counts up live while capturing and keeps the final length
  after a stop, until the next start or cancel.
- `rla_take_transcript` and `rla_transcribe_wav` copy at most `cap - 1` bytes
  (never splitting a UTF-8 character) and always return the full length.
  Pass `NULL, 0` to ask for the size.
- `rla_transcribe_wav` accepts any sample rate, mono or stereo, 16-bit int or
  32-bit float. It resamples and trims the audio the same way as a capture.
- On macOS the host app needs microphone permission
  (`NSMicrophoneUsageDescription`). Without it, cpal records silence and the
  transcript is empty.

### Deno example

```ts
const lib = Deno.dlopen("/path/to/librullama_audio_ffi.dylib", {
  rla_stt_load: { parameters: ["buffer"], result: "i32" },
  rla_transcribe_wav: { parameters: ["buffer", "buffer", "usize"], result: "i32" },
});
const cstr = (s: string) => new TextEncoder().encode(s + "\0");
lib.symbols.rla_stt_load(cstr("/models/ggml-base.en-q5_1.bin"));
const buf = new Uint8Array(4096);
const n = lib.symbols.rla_transcribe_wav(cstr("/tmp/clip.wav"), buf, BigInt(buf.length));
console.log(new TextDecoder().decode(buf.subarray(0, Math.min(n, buf.length - 1))));
// Before lib.close(): (declare it as { parameters: [], result: "i32" })
// lib.symbols.rla_shutdown();
```

## Tests

```sh
cargo test -p rullama-audio-ffi
RLA_WHISPER_MODEL=/path/ggml-base.en-q5_1.bin cargo test -p rullama-audio-ffi
```

The state machine is tested against a fake audio source and a fake
transcriber, so no test opens a microphone. The whisper integration test
(`tests/whisper_wav.rs`) transcribes `tests/fixtures/phrase.wav` ("Run the
tests in the second tile.") through the C functions. It is skipped unless
`RLA_WHISPER_MODEL` is set.

## Licensing

The crate is MIT OR Apache-2.0, like the rest of the workspace. The shared
library statically links:

| Component | License |
|---|---|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) / ggml (via whisper-rs-sys) | MIT |
| [whisper-rs](https://codeberg.org/tazz4843/whisper-rs) | Unlicense |
| [webrtc-vad](https://crates.io/crates/webrtc-vad) (crate) | MIT |
| libfvad / WebRTC VAD sources bundled by webrtc-vad | BSD-3-Clause (plus the WebRTC PATENTS grant) |
| [cpal](https://github.com/RustAudio/cpal) | Apache-2.0 |
| [hound](https://github.com/ruuda/hound) | Apache-2.0 |

All are permissive; ship their notices with an app that bundles the library.
Whisper model weights are not part of the crate; the ggml conversions of
OpenAI's Whisper models are MIT-licensed.

## Why some pieces bypass rullama-hardware

The crate reuses rullama-hardware's WAV decoder (`decode_wav`), its VAD
(`WebRtcVad` / `EnergyVad` segment detection) and its audio types. It talks to
cpal and whisper-rs directly where rullama-hardware's async APIs don't fit a
synchronous C ABI:

- `CpalCapture` opens the device at the caller's rate and format (16 kHz i16
  is often rejected by CoreAudio and ALSA) and reports stream errors only
  asynchronously. The ABI needs a synchronous `-4`, the device's native
  format, and a level meter.
- `CpalPlayback::play` is async, blocks until the clip ends, and cannot be
  stopped. `rla_play_wav` must return at once and be stoppable.
- `WhisperStt` reloads the model on every call. The ABI loads it once
  (`rla_stt_load`) and reuses the context.
