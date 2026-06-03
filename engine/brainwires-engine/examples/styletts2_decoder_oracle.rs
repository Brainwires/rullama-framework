//! StyleTTS2 hifigan Decoder CPU-oracle parity (isolation), built up stage by stage
//! against `scripts/styletts2_dump_decoder_fixtures.py` fixtures.
//!
//!   cargo run -p rullama --release --example styletts2_decoder_oracle

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use rullama::reference::kokoro::ops::max_abs_diff;
use rullama::reference::styletts2::decoder::source_signal;

fn main() {
    let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/styletts2/fixtures/decoder/bin");
    assert!(dir.is_dir(), "run scripts/styletts2_dump_decoder_fixtures.py first ({dir:?})");
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    for e in fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|x| x.to_str()) == Some("bin") {
            let b = fs::read(&p).unwrap();
            w.insert(p.file_stem().unwrap().to_str().unwrap().into(), b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
        }
    }

    // ---- HnNSF source: F0_curve → har_source (4 upsamples → up = 10*5*3*2 = 300) ----
    let f0 = w.get("in_F0_curve").expect("in_F0_curve");
    let lw = w.get("generator.m_source.l_linear.weight").expect("l_linear.weight"); // [1,9]
    let lb = w.get("generator.m_source.l_linear.bias").expect("l_linear.bias")[0];
    let har = source_signal(f0, 300, 9, lw, lb);
    let har_ref = w.get("har_source").expect("har_source fixture"); // [1, 24000, 1]
    let d = max_abs_diff(&har, har_ref);
    println!("har_source[{}]  max_abs_diff = {d:.3e}", har.len());
    assert!(har.len() == har_ref.len(), "source length {} != ref {}", har.len(), har_ref.len());
    assert!(d < 2e-3, "HnNSF source parity FAILED ({d:.3e})");
    println!("✅ hifigan HnNSF source matches PyTorch");
}
