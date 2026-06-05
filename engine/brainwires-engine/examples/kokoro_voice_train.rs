//! Self-recovery sanity test for gradient-free voice training: target = af_heart's
//! own timbre; start from a perturbed voice vector and hill-climb back toward it.
//! Proves the loss + optimizer work (loss drops, recovered vector approaches target).
//!
//!   cargo run --release --example kokoro_voice_train -- ~/.cache/kokoro/kokoro-82m-f32.gguf [iters]

use rullama::backend::{Pipelines, WgpuCtx};
use rullama::gguf::GgufReader;
use rullama::reference::kokoro::KokoroModel;
use rullama::reference::kokoro::gpu_fast::WeightCache;
use rullama::reference::kokoro::voice_train::voice_signature;
use std::fs;
use std::sync::Arc;

// phoneme ids for "Hello, how are you today?" (af_heart, 25 tokens)
const IDS: [i64; 25] = [
    0, 50, 83, 54, 156, 31, 3, 16, 50, 157, 39, 16, 69, 123, 16, 52, 63, 16, 62, 83, 46, 156, 24,
    6, 0,
];

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let gguf = args
        .next()
        .expect("usage: kokoro_voice_train <gguf> [iters]");
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(15);

    let reader = Arc::new(GgufReader::new(fs::read(&gguf).unwrap()).unwrap());
    let model = KokoroModel::new(reader).unwrap();
    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let pipes = Pipelines::new(&ctx.device);
    let mut wc = WeightCache::new();

    // target = af_heart's own timbre
    let target = model.load_voice("af_heart", IDS.len());
    let target_audio =
        pollster::block_on(model.synthesize_gpu_fast(&ctx, &pipes, &mut wc, &IDS, &target));
    let target_sig = voice_signature(&target_audio);

    // start from a perturbed voice vector (a deterministic offset from af_heart)
    let init: Vec<f32> = target
        .iter()
        .enumerate()
        .map(|(i, &v)| v + 0.25 * ((i as f32) * 0.7).sin())
        .collect();
    println!("init ||style - target|| = {:.4}", l2(&init, &target));

    let res = pollster::block_on(model.train_voice(
        &ctx,
        &pipes,
        &mut wc,
        &IDS,
        &target_sig,
        &init,
        iters,
        0.06,
        12345,
    ));

    println!(
        "loss curve: {:?}",
        res.loss_curve
            .iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );
    println!(
        "loss {:.4e} -> {:.4e}  ({:.0}% reduction)",
        res.loss_curve[0],
        *res.loss_curve.last().unwrap(),
        100.0 * (1.0 - res.loss_curve.last().unwrap() / res.loss_curve[0])
    );
    println!(
        "recovered ||style - target|| = {:.4}  (init was {:.4})",
        l2(&res.style, &target),
        l2(&init, &target)
    );

    // emit before/after WAVs to listen
    let a0 = pollster::block_on(model.synthesize_gpu_fast(&ctx, &pipes, &mut wc, &IDS, &init));
    let a1 = pollster::block_on(model.synthesize_gpu_fast(&ctx, &pipes, &mut wc, &IDS, &res.style));
    write_wav("/tmp/voice_init.wav", &a0);
    write_wav("/tmp/voice_trained.wav", &a1);
    write_wav("/tmp/voice_target.wav", &target_audio);
    println!("wrote /tmp/voice_{{init,trained,target}}.wav");
}

fn write_wav(path: &str, s: &[f32]) {
    let n = s.len() as u32;
    let dl = n * 2;
    let mut b = Vec::with_capacity(44 + dl as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + dl).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&24000u32.to_le_bytes());
    b.extend_from_slice(&48000u32.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&dl.to_le_bytes());
    for &x in s {
        b.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    fs::write(path, b).unwrap();
}
