//! Kokoro CPU-oracle parity driver. Runs oracle stages and diffs each against the
//! PyTorch reference fixtures dumped by `scripts/kokoro_dump_fixtures.py`.
//!
//! Build:
//!   cargo run --release --example kokoro_oracle -- \
//!       ~/.cache/kokoro/kokoro-82m-f32.gguf ~/.cache/kokoro/fixtures
//!
//! Stages are added incrementally; each prints max-abs diff vs its fixture.

use std::env;
use std::fs;
use std::sync::Arc;

use rullama::gguf::GgufReader;
use rullama::reference::kokoro::ops::max_abs_diff;
use rullama::reference::kokoro::KokoroModel;

// Fixture input (from scripts/kokoro_dump_fixtures.py meta.json): "Hello, how are you today?"
const INPUT_IDS: [i64; 25] = [
    0, 50, 83, 54, 156, 31, 3, 16, 50, 157, 39, 16, 69, 123, 16, 52, 63, 16, 62, 83, 46, 156, 24, 6, 0,
];

fn read_bin_f32(path: &str) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let mut args = env::args().skip(1);
    let gguf = args.next().expect("usage: kokoro_oracle <gguf> <fixtures_dir>");
    let fixtures = args.next().expect("usage: kokoro_oracle <gguf> <fixtures_dir>");

    let bytes = fs::read(&gguf).unwrap();
    let reader = Arc::new(GgufReader::new(bytes).unwrap());
    let model = KokoroModel::new(reader).unwrap();
    println!(
        "loaded kokoro: hidden={} plbert={}x{}h vocab={} | T={} tokens",
        model.cfg.hidden_dim,
        model.cfg.plbert_layers,
        model.cfg.plbert_heads,
        model.cfg.vocab.len(),
        INPUT_IDS.len(),
    );

    let t = INPUT_IDS.len();

    // ---- Stage 1: PL-BERT (ALBERT) ----
    let bert = model.bert(&INPUT_IDS);
    diff("bert", &bert, &read_bin_f32(&format!("{fixtures}/bin/bert.bin")));

    // ---- Stage 2: bert_encoder (768->512) ----
    let be = model.bert_encoder(&bert, t);
    diff("bert_encoder", &be, &read_bin_f32(&format!("{fixtures}/bin/bert_encoder.bin")));

    // voice vector (exact ref_s used by the fixture): [:128]=timbre, [128:]=prosodic
    let ref_s = read_bin_f32(&format!("{fixtures}/bin/ref_s.bin"));
    let style_pros = &ref_s[128..256];

    // ---- Stage 3: DurationEncoder (BiLSTM + AdaLayerNorm) ----
    let d = model.duration_encode(&be, t, style_pros);
    diff("pred_text_encoder_d", &d, &read_bin_f32(&format!("{fixtures}/bin/pred_text_encoder_d.bin")));

    // ---- Stage 4: duration prediction ----
    let (logits, pred_dur) = model.predict_duration(&d, t);
    diff("duration_logits", &logits, &read_bin_f32(&format!("{fixtures}/bin/duration_logits.bin")));
    let dur_ok = pred_dur == EXPECTED_DUR;
    println!(
        "[pred_dur]        sum {:>6}  {}  {}",
        pred_dur.iter().sum::<usize>(),
        if dur_ok { "exact match" } else { "*** MISMATCH ***" },
        if dur_ok { "OK" } else { "" },
    );
    if !dur_ok {
        println!("    got      {pred_dur:?}");
        println!("    expected {EXPECTED_DUR:?}");
    }

    // ---- Stage 5: F0 / N (length-regulator + shared BiLSTM + AdainResBlk1d stacks) ----
    let (en, f) = model.expand_by_dur_cm(&d, t, 640, &pred_dur);
    let (f0, n) = model.f0_n(&en, f, style_pros);
    diff("F0", &f0, &read_bin_f32(&format!("{fixtures}/bin/F0.bin")));
    diff("N", &n, &read_bin_f32(&format!("{fixtures}/bin/N.bin")));

    // ---- Stage 6: TextEncoder (embedding + conv stack + BiLSTM) ----
    let t_en = model.text_encoder(&INPUT_IDS);
    diff("text_encoder_ten", &t_en, &read_bin_f32(&format!("{fixtures}/bin/text_encoder_ten.bin")));

    // ---- Stage 7: Decoder encode + decode stack (timbre style = ref_s[:128]) ----
    let style_timbre = &ref_s[0..128];
    let (dec_encode, x_dec, _f0d, _nd) = model.decoder_features(&t_en, &f0, &n, &pred_dur, style_timbre);
    diff("dec_encode", &dec_encode, &read_bin_f32(&format!("{fixtures}/bin/dec_encode.bin")));
    diff("gen_x(decode)", &x_dec, &read_bin_f32(&format!("{fixtures}/bin/gen_x.bin")));

    // ---- Stage 8: ISTFTNet generator + exact iSTFT ----
    // har is non-deterministic upstream (random source) → inject the reference.
    let gen_har = read_bin_f32(&format!("{fixtures}/bin/gen_har.bin"));
    let ref_audio = read_bin_f32(&format!("{fixtures}/bin/audio.bin"));
    // (a) isolated: reference x + reference har
    let gen_x = read_bin_f32(&format!("{fixtures}/bin/gen_x.bin"));
    let audio_iso = model.generator(&gen_x, 156, &gen_har, 9361, style_timbre);
    diff("audio[ref x]", &audio_iso, &ref_audio);
    // (b) full chain: our decode-stack x + reference har
    let audio_full = model.generator(&x_dec, 156, &gen_har, 9361, style_timbre);
    diff("audio[our x]", &audio_full, &ref_audio);
}

const EXPECTED_DUR: [usize; 25] = [
    14, 2, 3, 2, 5, 5, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 4, 12, 8, 1,
];

fn diff(name: &str, got: &[f32], reference: &[f32]) {
    let d = max_abs_diff(got, reference);
    println!("[{name:<18}] shape {:>7}  max_abs_diff = {:.3e}  {}", got.len(), d, verdict(d));
}

fn verdict(d: f32) -> &'static str {
    if d < 2e-3 {
        "OK"
    } else {
        "*** MISMATCH ***"
    }
}
