//! End-to-end f16-resident StyleTTS2 vs f32 parity.
//!
//! Loads the f32 GGUF (`load_streaming`) and the f16 GGUF (`load_streaming_f16`),
//! runs the GPU synth on both with the SAME token ids + voice vector and the
//! diffusion path OFF (deterministic — isolates the f16 conv-weight effect from
//! the style-diffusion RNG), and reports waveform correlation.
//!
//! The real gate is `corr(f16-storage/f32-compute, f16-resident)` ≈ 1.0 — the f16
//! conv kernels must add no error beyond f16 storage. Measured: 1.00000 (bit-
//! exact). The TOTAL degradation vs the f32 GGUF (~0.79 corr) is inherent f16
//! *storage* precision — this deep vocoder loses precision under f16 (which is
//! why the f32 variant is the desktop default), NOT a kernel bug.
//!
//! Usage:
//!   cargo run -p brainwires-engine --release --example styletts2_f16_parity -- \
//!       ~/.cache/styletts2/styletts2-libritts-f32.gguf \
//!       ~/.cache/styletts2/styletts2-libritts-f16.gguf

use brainwires_engine::backend::{Pipelines, WgpuCtx};
use brainwires_engine::gguf::GgufReader;
use brainwires_engine::reference::styletts2::StyleTtsModel;
use brainwires_engine::reference::styletts2::gpu::GpuWeightCache;

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let ma = a.iter().sum::<f32>() / n as f32;
    let mb = b.iter().sum::<f32>() / n as f32;
    let mut num = 0.0f32;
    let mut da = 0.0f32;
    let mut db = 0.0f32;
    for i in 0..n {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da == 0.0 || db == 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let f32_path = args.get(1).cloned().unwrap_or_else(|| {
        format!(
            "{}/.cache/styletts2/styletts2-libritts-f32.gguf",
            std::env::var("HOME").unwrap()
        )
    });
    let f16_path = args.get(2).cloned().unwrap_or_else(|| {
        format!(
            "{}/.cache/styletts2/styletts2-libritts-f16.gguf",
            std::env::var("HOME").unwrap()
        )
    });

    pollster::block_on(async {
        let ctx = WgpuCtx::new().await.expect("wgpu");
        let pipes = Pipelines::new(&ctx.device);

        eprintln!("loading f32: {f32_path}");
        let r32 =
            GgufReader::new(std::fs::read(&f32_path).expect("read f32 gguf")).expect("gguf32");
        let m32 = StyleTtsModel::load_streaming(&r32).await.expect("load f32");

        eprintln!("loading f16: {f16_path}");
        let r16 =
            GgufReader::new(std::fs::read(&f16_path).expect("read f16 gguf")).expect("gguf16");
        let m16 = StyleTtsModel::load_streaming_f16(&r16)
            .await
            .expect("load f16");

        // Deterministic inputs. Phoneme ids in [1,177]; a fixed ~40-token line.
        let ids: Vec<i64> = (0..40).map(|i| 1 + (i * 17 + 3) % 176).collect();
        let voice: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.07).sin() * 0.12).collect();

        // Diagnostic: load the f16 GGUF as pure f32 (load_streaming dequants
        // everything to f32) — isolates the f16 *storage* precision (this deep
        // model is known to degrade under f16) from the f16 conv *kernels*.
        let mf16f32 = StyleTtsModel::load_streaming(&r16)
            .await
            .expect("load f16 as f32");

        let mut wc32 = GpuWeightCache::new();
        let y32 = m32
            .synthesize_gpu(&ctx, &pipes, &mut wc32, &ids, &voice, None, None)
            .await;
        let mut wcb = GpuWeightCache::new();
        let yb = mf16f32
            .synthesize_gpu(&ctx, &pipes, &mut wcb, &ids, &voice, None, None)
            .await;
        let mut wc16 = GpuWeightCache::new();
        let y16 = m16
            .synthesize_gpu(&ctx, &pipes, &mut wc16, &ids, &voice, None, None)
            .await;

        let c_total = corr(&y32, &y16);
        let c_storage = corr(&y32, &yb); // f16 storage effect (f32 compute)
        let c_kernel = corr(&yb, &y16); // f16 conv-kernel effect only
        let nan16 = y16.iter().filter(|x| x.is_nan()).count();
        eprintln!("len={} nan(f16)={nan16}", y16.len());
        eprintln!("  corr(f32, f16-storage/f32-compute) = {c_storage:.5}  <- inherent f16 storage");
        eprintln!(
            "  corr(f16-storage/f32-compute, f16)  = {c_kernel:.5}  <- f16 conv KERNEL effect"
        );
        eprintln!("  corr(f32, f16-resident)             = {c_total:.5}  <- total");
        assert_eq!(nan16, 0, "f16 synth produced NaNs");
        // The conv KERNEL must add essentially no error beyond f16 storage —
        // that's the real regression gate (the storage loss is expected/accepted).
        assert!(
            c_kernel > 0.99,
            "f16 conv kernel diverged from f16-storage/f32-compute (corr {c_kernel:.4}) — routing bug"
        );
        eprintln!(
            "PASS: f16 conv kernels match f32 compute; total degradation is f16 storage only."
        );
    });
}
