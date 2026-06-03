//! StyleTTS2 style-diffusion CPU-oracle parity check.
//!
//! Loads the fixtures from `scripts/styletts2_dump_diffusion_fixtures.py`
//! (~/.cache/styletts2/fixtures/diffusion/), feeds PyTorch's bert_dur + ref_s + the exact
//! replayed noise into the Rust `StyleDiffusion` denoiser/sampler, and diffs against the
//! reference net output (single eval) and the final s_pred (full ADPM2 sample).
//!
//!   cargo run -p rullama --release --example styletts2_diffusion_oracle

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use rullama::reference::kokoro::ops::max_abs_diff;
use rullama::reference::styletts2::diffusion::StyleDiffusion;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/styletts2/fixtures/diffusion")
}

fn read_bin(p: &PathBuf) -> Vec<f32> {
    let b = fs::read(p).unwrap_or_else(|e| panic!("read {p:?}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let d = dir();
    assert!(d.is_dir(), "fixtures missing — run scripts/styletts2_dump_diffusion_fixtures.py first ({d:?})");

    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    for entry in fs::read_dir(&d).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().and_then(|e| e.to_str()) == Some("bin") {
            w.insert(p.file_stem().unwrap().to_str().unwrap().to_string(), read_bin(&p));
        }
    }
    let bert_dur = w.remove("bert_dur").unwrap();
    let l = bert_dur.len() / 768; // [L, 768]
    let ref_s = w.remove("ref_s").unwrap();
    let noise_init = w.remove("noise_init").unwrap();
    let step_noises = w.remove("step_noises").unwrap(); // [4,1,256] flat
    let want_net = w.remove("net_out").unwrap();
    let want_spred = w.remove("s_pred").unwrap();
    let noises: Vec<Vec<f32>> = step_noises.chunks_exact(256).map(|c| c.to_vec()).collect();

    let diff = StyleDiffusion::new(&w);

    // 1) isolated denoiser net eval at sigma_max (x = sigma_max * noise_init, time = ln(σ)·0.25)
    let (sigma_max, sigma_data) = (3.0f32, 0.2f32);
    let c_in = (sigma_max * sigma_max + sigma_data * sigma_data).powf(-0.5);
    let c_noise = sigma_max.ln() * 0.25;
    let x0: Vec<f32> = noise_init.iter().map(|v| c_in * sigma_max * v).collect();
    let net_got = diff.net_eval(&x0, c_noise, &bert_dur, l, &ref_s);
    let dnet = max_abs_diff(&net_got, &want_net);
    println!("denoiser net eval   max_abs_diff = {dnet:.3e}  (|out|={:.3})", (net_got.iter().map(|v| v * v).sum::<f32>()).sqrt());

    // 2) full ADPM2 sample → s_pred
    let spred = diff.sample(&noise_init, &noises, &bert_dur, l, &ref_s);
    let dspred = max_abs_diff(&spred, &want_spred);
    println!("ADPM2 s_pred[256]   max_abs_diff = {dspred:.3e}");
    println!("  got  [:6] = {:?}", &spred[..6]);
    println!("  want [:6] = {:?}", &want_spred[..6]);

    let worst = dnet.max(dspred);
    println!("\nworst max_abs_diff = {worst:.3e}");
    assert!(worst < 2e-3, "StyleTTS2 diffusion parity FAILED (worst {worst:.3e})");
    println!("✅ StyleTTS2 style-diffusion matches PyTorch (denoiser + ADPM2 sampler)");
}
