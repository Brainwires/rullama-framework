//! End-to-end text → speech on GPU: load the Kokoro GGUF + lexicon, run the full
//! GPU forward, write a 24 kHz WAV.
//!
//!   cargo run --release --example kokoro_tts -- \
//!       ~/.cache/kokoro/kokoro-82m-f32.gguf ~/.cache/kokoro/us_gold.json \
//!       "Hello from rullama." af_heart out.wav [~/.cache/kokoro/us_silver.json]

use brainwires_engine::backend::{Pipelines, WgpuCtx};
use brainwires_engine::gguf::GgufReader;
use brainwires_engine::reference::kokoro::KokoroModel;
use brainwires_engine::reference::kokoro::g2p::Lexicon;
use std::fs;
use std::sync::Arc;

fn main() {
    let mut a = std::env::args().skip(1);
    let gguf = a
        .next()
        .expect("usage: kokoro_tts <gguf> <us_gold.json> <text> <voice> <out.wav> [silver]");
    let lex_path = a.next().expect("lexicon path");
    let text = a.next().expect("text");
    let voice = a.next().unwrap_or_else(|| "af_heart".into());
    let out = a.next().unwrap_or_else(|| "out.wav".into());
    let silver = a.next().map(|p| fs::read(p).unwrap()).unwrap_or_default();

    let reader = Arc::new(GgufReader::new(fs::read(&gguf).unwrap()).unwrap());
    let model = KokoroModel::new(reader).unwrap();
    let lex = Lexicon::load(&fs::read(&lex_path).unwrap(), &silver);
    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let pipes = Pipelines::new(&ctx.device);

    let t0 = std::time::Instant::now();
    let (audio, oov) =
        pollster::block_on(model.synthesize_text_gpu(&ctx, &pipes, &text, &voice, &lex));
    let dt = t0.elapsed().as_secs_f32();
    let secs = audio.len() as f32 / 24000.0;
    let peak = audio.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    println!("text: {text:?}  voice: {voice}");
    if !oov.is_empty() {
        println!("OOV (skipped): {oov:?}");
    }
    println!(
        "audio: {} samples ({secs:.2}s), peak {peak:.3}, synth {dt:.2}s ({:.1}x realtime)",
        audio.len(),
        secs / dt
    );

    write_wav(&out, &audio, 24000);
    println!("wrote {out}");
}

fn write_wav(path: &str, samples: &[f32], sr: u32) {
    let n = samples.len() as u32;
    let data_len = n * 2;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&sr.to_le_bytes());
    b.extend_from_slice(&(sr * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        b.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    fs::write(path, b).unwrap();
}
