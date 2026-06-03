//! StyleTTS2 full zero-shot synthesis CPU-oracle parity (the complete acoustic graph +
//! hifigan decoder), against scripts/styletts2_dump_synth_fixtures.py.
//!
//!   cargo run -p rullama --release --example styletts2_synth_oracle

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use rullama::reference::kokoro::ops::max_abs_diff;
use rullama::reference::styletts2::StyleTtsAcoustic;

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (ma, mb) = (a.iter().sum::<f32>() / n as f32, b.iter().sum::<f32>() / n as f32);
    let (mut num, mut da, mut db) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        num += (x - ma) * (y - mb);
        da += (x - ma) * (x - ma);
        db += (y - mb) * (y - mb);
    }
    num / (da.sqrt() * db.sqrt() + 1e-12)
}

fn main() {
    let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/styletts2/fixtures/synth/bin");
    assert!(dir.is_dir(), "run scripts/styletts2_dump_synth_fixtures.py first ({dir:?})");
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    let mut tokens: Vec<i64> = Vec::new();
    let mut pred_dur_ref: Vec<i64> = Vec::new();
    for e in fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_stem().unwrap().to_str().unwrap().to_string();
        let bytes = fs::read(&p).unwrap();
        if name == "tokens" || name == "pred_dur" {
            let v: Vec<i64> = bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
            if name == "tokens" { tokens = v; } else { pred_dur_ref = v; }
        } else {
            w.insert(name, bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
        }
    }
    let t = tokens.len();
    println!("tokens: {t}");
    let ac = StyleTtsAcoustic::new(&w);

    // ---- text_encoder ----
    let t_en = ac.text_encoder(&tokens);
    println!("t_en           max_abs_diff = {:.3e}", max_abs_diff(&t_en, w.get("t_en").unwrap()));

    // ---- bert + bert_encoder → d_en [512,T] (transpose of [T,512]) ----
    let be = ac.bert_encoder(&ac.bert(&tokens, None), t);
    let mut d_en = vec![0f32; 512 * t];
    for ti in 0..t {
        for c in 0..512 {
            d_en[c * t + ti] = be[ti * 512 + c];
        }
    }
    println!("d_en (bert)    max_abs_diff = {:.3e}", max_abs_diff(&d_en, w.get("d_en").unwrap()));

    // ---- durations (MUST be integer-exact or everything downstream misaligns) ----
    let d = ac.duration_encode(&be, t, &w.get("s_prosodic").unwrap().clone());
    let dur = ac.predict_duration(&d, t);
    let dur_i64: Vec<i64> = dur.iter().map(|&x| x as i64).collect();
    let dur_match = dur_i64 == pred_dur_ref;
    println!("pred_dur exact = {dur_match}  (sum {} vs ref {})", dur.iter().sum::<usize>(), pred_dur_ref.iter().sum::<i64>());
    assert!(dur_match, "duration prediction diverged — got {dur_i64:?} want {pred_dur_ref:?}");

    // ---- full synthesis → audio ----
    let ref_s = w.get("ref_s").unwrap().clone();
    let audio = ac.synthesize(&tokens, &ref_s, None);
    let audio_ref = w.get("audio").unwrap();
    let da = max_abs_diff(&audio, audio_ref);
    let c = corr(&audio, audio_ref);
    println!("\naudio[{}] vs ref[{}]  max_abs_diff = {da:.3e}  corr = {c:.6}", audio.len(), audio_ref.len());
    assert!(audio.len() == audio_ref.len(), "audio length {} != {}", audio.len(), audio_ref.len());
    // Correlation is the parity metric downstream of the HnNSF source (tiny F0 diffs →
    // source-phase drift that preserves the waveform but inflates per-sample max-abs —
    // the same F0-phase sensitivity documented for the Kokoro path).
    assert!(c > 0.999, "full synthesis parity FAILED (corr {c:.6})");

    // ---- full GPU synth: CPU acoustic graph + GPU hifigan decoder ----
    {
        use rullama::backend::{Pipelines, WgpuCtx};
        use rullama::reference::styletts2::gpu::StyleTtsGpu;
        let (asr_g, f0_g, n_g, r_g, fg) = ac.acoustic_features(&tokens, &ref_s, None);
        let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
        let pipes = Pipelines::new(&ctx.device);
        let mut wc = HashMap::new();
        let gpu_audio = pollster::block_on(StyleTtsGpu::new(&w, &ctx, &pipes, &mut wc).decode(&asr_g, fg, &f0_g, &n_g, &r_g));
        let cg = corr(&gpu_audio, audio_ref);
        println!("GPU full synth   corr = {cg:.6}  (len {} vs {})", gpu_audio.len(), audio_ref.len());
        assert!(cg > 0.999, "GPU full synthesis parity FAILED (corr {cg:.6})");
        println!("✅ StyleTTS2 GPU synthesis matches PyTorch end-to-end");
    }

    // write the Rust-side cloned WAV (16-bit PCM) for listening
    let out = PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/styletts2/fixtures/synth/cloned_rust.wav");
    let n = audio.len();
    let mut buf = Vec::with_capacity(44 + n * 2);
    let hdr = |buf: &mut Vec<u8>, s: &str| buf.extend_from_slice(s.as_bytes());
    hdr(&mut buf, "RIFF");
    buf.extend_from_slice(&((36 + n * 2) as u32).to_le_bytes());
    hdr(&mut buf, "WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&24000u32.to_le_bytes());
    buf.extend_from_slice(&48000u32.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    hdr(&mut buf, "data");
    buf.extend_from_slice(&((n * 2) as u32).to_le_bytes());
    for &v in &audio {
        buf.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    fs::write(&out, buf).unwrap();
    println!("✅ StyleTTS2 full zero-shot cloning matches PyTorch end-to-end → {out:?}");
}
