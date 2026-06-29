//! Audio parity: rullama's audio tower + LM vs Ollama's audio tower + LM.
//!
//! Takes a path to an audio file (any format ffmpeg knows: mp3 / wav /
//! ogg / m4a / …). Non-WAV inputs are transcoded to 16 kHz mono WAV
//! at `/tmp/audio_parity_input.wav` via a shell-out to `ffmpeg` —
//! parallel to `vision_parity.rs`'s `python3` + PIL preprocessing
//! step. Then runs rullama through `encode_audio_native` +
//! `step_with_embedding` splice with a chat-style transcription
//! prompt, prints the response. Then sends the same WAV to Ollama's
//! `/v1/audio/transcriptions` endpoint (OpenAI-compatible) and
//! prints that too.
//!
//! Canonical fixture: `whats-todays-date.mp3` at the repo root —
//! ~2 s recording of the phrase *"What's today's date?"*. The
//! transcription prompt should yield a string within a few edit-
//! distance of that phrase on both engines.
//!
//! When run without a second argument, synthesises a 1-second 440 Hz
//! sine wave and writes it to `/tmp/test_440hz.wav` for a smoke
//! test that the pipeline runs end-to-end.
//!
//! Build:
//!   # With the canonical fixture
//!   cargo run --release --example audio_parity -- \
//!       ~/.ollama/models/blobs/sha256-<digest> ./whats-todays-date.mp3
//!
//!   # Smoke (synthetic sine)
//!   cargo run --release --example audio_parity -- <gguf>
//!
//! Depends on `ffmpeg` on PATH for non-WAV inputs (`brew install
//! ffmpeg` on macOS).

use std::env;
use std::fs;
use std::process::{Command, ExitCode};
use std::time::Instant;

use brainwires_engine::api::{ChatMessage, ChatRole, Model};
use brainwires_engine::sampling::SamplingOptions;
use brainwires_engine::template::gemma4_small;

const N_PREDICT: usize = 24;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: audio_parity <gguf> [wav-path]");
            return ExitCode::from(2);
        }
    };
    let wav_path_arg = args.next();

    // Resolve input → a 16 kHz mono WAV on disk we can both feed
    // rullama (via decode_wav_native) and POST to Ollama. WAV inputs
    // pass through unchanged; everything else (mp3/m4a/ogg/…) gets
    // transcoded via ffmpeg. No second arg → synthesise a 1 s sine.
    let sr = 16_000usize;
    let (pcm, wav_path): (Vec<f32>, String) = match wav_path_arg {
        Some(input) if is_wav_path(&input) => {
            let bytes = fs::read(&input).expect("read wav");
            let pcm = brainwires_engine::api::Model::decode_wav_native(&bytes).expect("decode wav");
            println!(
                "loaded {input}: {} samples ({:.2} s @ 16 kHz)",
                pcm.len(),
                pcm.len() as f32 / sr as f32
            );
            (pcm, input)
        }
        Some(input) => {
            let normalised = "/tmp/audio_parity_input.wav".to_string();
            // Prefer macOS `afconvert` (built into the OS — no install
            // needed); fall back to `ffmpeg` for non-macOS hosts.
            // Both produce a 16 kHz mono LEI16 WAV at `normalised`.
            let transcoded = transcode_to_wav(&input, &normalised);
            match transcoded {
                Ok(()) => {}
                Err(msg) => {
                    eprintln!("FAIL: {msg}");
                    return ExitCode::from(2);
                }
            }
            let bytes = fs::read(&normalised).expect("read transcoded wav");
            let pcm = brainwires_engine::api::Model::decode_wav_native(&bytes).expect("decode wav");
            println!(
                "transcoded {input}: {} samples ({:.2} s @ 16 kHz)",
                pcm.len(),
                pcm.len() as f32 / sr as f32
            );
            (pcm, normalised)
        }
        None => {
            let n = sr;
            let omega = 2.0 * std::f32::consts::PI * 440.0 / sr as f32;
            let pcm: Vec<f32> = (0..n).map(|i| 0.3 * (omega * i as f32).sin()).collect();
            let wav_path = "/tmp/test_440hz.wav".to_string();
            write_pcm16_wav(&wav_path, &pcm, sr);
            println!(
                "wrote {wav_path} ({} samples @ {sr} Hz) — sine smoke",
                pcm.len()
            );
            (pcm, wav_path)
        }
    };

    // ---- rullama side ----
    println!("\n== rullama side ==");
    println!("loading model ...");
    let bytes = fs::read(&path).expect("read");
    let mut model = pollster::block_on(Model::load_native(bytes)).expect("load");
    // Greedy sampling so the result is deterministic and comparable.
    model.set_sampling_native(SamplingOptions {
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    });
    if !model.has_audio_native() {
        eprintln!("FAIL: this checkpoint has no audio tower");
        return ExitCode::from(2);
    }

    // Match Ollama's TranscriptionMiddleware:
    //   system: "Transcribe the following audio exactly as spoken. Output only
    //            the transcription text, nothing else."
    //   user:   "Transcribe this audio." + audio attachment
    let messages = vec![
        ChatMessage {
            role: ChatRole::System,
            content: "Transcribe the following audio exactly as spoken. \
                      Output only the transcription text, nothing else."
                .into(),
        },
        ChatMessage {
            // The audio sentinel pair is inserted by us before the visible text;
            // Ollama's renderer does the equivalent splice.
            role: ChatRole::User,
            content: "<|audio><audio|>Transcribe this audio.".into(),
        },
    ];
    let prompt = gemma4_small::render_for_completion(&messages, false);
    let ids = model.encode_tokens(&prompt);

    let (audio_begin, _audio_end) = model
        .audio_sentinel_ids_native()
        .expect("audio sentinels missing from vocab");

    let t = Instant::now();
    let soft = pollster::block_on(model.encode_audio_native(&pcm)).expect("encode_audio");
    let n_soft = soft.len() / 1536;
    println!("encoded {n_soft} audio soft tokens in {:?}", t.elapsed());

    // Walk the prompt: feed each token; after the audio begin sentinel, splice
    // n_soft step_with_embedding calls (one per soft-token row).
    println!("feeding {} prompt tokens (+ {n_soft} soft) ...", ids.len());
    let t = Instant::now();
    let mut next: u32 = 0;
    for &id in &ids {
        next = pollster::block_on(model.step_native(id)).expect("step");
        if id == audio_begin {
            for r in 0..n_soft {
                let row = &soft[r * 1536..(r + 1) * 1536];
                next = pollster::block_on(model.step_with_embedding_native(row))
                    .expect("step_with_embedding");
            }
        }
    }
    println!(
        "prompt-eval done in {:?}; first sampled token = {} ({:?})",
        t.elapsed(),
        next,
        model.token_str_native(next)
    );

    let t = Instant::now();
    let mut out = String::new();
    let mut out_ids = Vec::with_capacity(N_PREDICT);
    for _ in 0..N_PREDICT {
        if model.is_eos_native(next) {
            break;
        }
        out_ids.push(next);
        if let Some(s) = model.token_str_native(next) {
            out.push_str(&s.replace('▁', " "));
        }
        next = pollster::block_on(model.step_native(next)).expect("gen-step");
    }
    println!(
        "rullama generated {} tokens in {:?}: {out:?}",
        out_ids.len(),
        t.elapsed()
    );
    println!("rullama ids: {out_ids:?}");

    // ---- Ollama side ----
    println!("\n== ollama side ==");
    let out = Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            "http://localhost:11434/v1/audio/transcriptions",
            "-F",
            &format!("file=@{wav_path}"),
            "-F",
            "model=gemma4:e2b",
            "--max-time",
            "120",
        ])
        .output()
        .expect("curl ollama");
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("ollama response: {stdout}");

    println!("\nNote: bit-parity is not expected — Ollama uses GGML's audio runtime,");
    println!("rullama uses our hand-written wgpu kernels. Compare semantic alignment.");
    println!("For the canonical `whats-todays-date.mp3` fixture both engines should");
    println!("produce something close to *\"What's today's date?\"*.");
    ExitCode::SUCCESS
}

/// True iff the path's extension is `.wav` (case-insensitive). Other
/// extensions get transcoded to a 16 kHz mono WAV first.
fn is_wav_path(p: &str) -> bool {
    p.to_ascii_lowercase().ends_with(".wav")
}

/// Transcode an audio file to a 16 kHz mono LEI16 WAV.
///
/// Tries macOS `afconvert` first (built into the OS — no install
/// needed), then `ffmpeg` (Linux). Returns the program name that
/// succeeded for diagnostics, or an `Err` message listing what was
/// tried.
fn transcode_to_wav(input: &str, output: &str) -> Result<(), String> {
    // afconvert: `-f WAVE` (RIFF), `-d LEI16@16000` (16-bit signed
    // little-endian PCM @ 16 kHz), `-c 1` (mono channel mix). Native
    // on macOS — uses CoreAudio's mp3/aac/m4a/wav decoders.
    let afc = Command::new("afconvert")
        .args(["-f", "WAVE", "-d", "LEI16@16000", "-c", "1", input, output])
        .status();
    if let Ok(s) = &afc
        && s.success()
    {
        println!("afconvert: {input} → {output} (macOS CoreAudio)");
        return Ok(());
    }

    let ff = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-i",
            input,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            output,
        ])
        .status();
    if let Ok(s) = &ff
        && s.success()
    {
        println!("ffmpeg: {input} → {output}");
        return Ok(());
    }

    let afc_msg = match afc {
        Ok(s) => format!("afconvert exited with {s}"),
        Err(e) => format!("afconvert not available ({e})"),
    };
    let ff_msg = match ff {
        Ok(s) => format!("ffmpeg exited with {s}"),
        Err(e) => format!("ffmpeg not available ({e})"),
    };
    Err(format!(
        "no transcoder worked.\n  - {afc_msg}\n  - {ff_msg}\n\
         Install one: macOS `afconvert` is built-in; Linux: `apt install ffmpeg`."
    ))
}

fn write_pcm16_wav(path: &str, pcm: &[f32], sr: usize) {
    let n_bytes = pcm.len() * 2;
    let mut buf = Vec::with_capacity(44 + n_bytes);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + n_bytes as u32).to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&(sr as u32).to_le_bytes()); // sample rate
    buf.extend_from_slice(&(sr as u32 * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(n_bytes as u32).to_le_bytes());
    for &s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, buf).expect("write wav");
}
