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

    // ---- Stage 9: standalone HnNSF source (deterministic, zeroed randomness) ----
    let src_sig = model.source_signal(&f0);
    diff("har_source_det", &src_sig, &read_bin_f32(&format!("{fixtures}/bin/har_source_det.bin")));
    let (har_src, frames_src) = model.generator_source(&f0);
    let ref_hd = read_bin_f32(&format!("{fixtures}/bin/gen_har_det.bin"));
    // har phase is arbitrary at low-energy bins (model trained on random source
    // phase) → correlation, not max-abs, is the right metric here.
    corr_report("gen_har_det(source)", &har_src, &ref_hd);
    // mag/phase split diagnostic (har = [11 mag; 11 phase], channel-major)
    {
        let nb = 11;
        let frames = frames_src;
        let (mut mag_d, mut ph_d, mut flips) = (0.0f32, 0.0f32, 0usize);
        for k in 0..22 {
            for f in 0..frames {
                let dd = (har_src[k * frames + f] - ref_hd[k * frames + f]).abs();
                if k < nb {
                    mag_d = mag_d.max(dd);
                } else {
                    ph_d = ph_d.max(dd);
                    if dd > 1.0 {
                        flips += 1;
                    }
                }
            }
        }
        println!("    split: mag_max={mag_d:.3e}  phase_max={ph_d:.3e}  phase_flips(>1)={flips}/{}", nb * frames);
        // one voiced frame: is the phase error a k-linear ramp (shift) or random (branch flip)?
        let f0r = 100;
        print!("    frame100 dphase[k]: ");
        for k in 0..nb {
            let d = har_src[(k + nb) * frames + f0r] - ref_hd[(k + nb) * frames + f0r];
            print!("{:+.2} ", d);
        }
        println!();
    }

    let audio_std = model.generator(&x_dec, 156, &har_src, frames_src, style_timbre);
    corr_report("audio_det(standalone)", &audio_std, &read_bin_f32(&format!("{fixtures}/bin/audio_det.bin")));

    // ---- Stage 10: composed synthesize() must reproduce the staged pipeline ----
    let syn_ids = model.synthesize_ids(&INPUT_IDS, &ref_s);
    diff("synthesize_ids", &syn_ids, &audio_std);
    let ids = model.phonemes_to_ids("həlˈO, hˌW ɑɹ ju tədˈA?");
    let ids_ok = ids == INPUT_IDS;
    println!("[phonemes_to_ids ]  {}  ({} ids)", if ids_ok { "matches fixture  OK" } else { "*** MISMATCH ***" }, ids.len());
    let syn = model.synthesize("həlˈO, hˌW ɑɹ ju tədˈA?", "af_heart");
    diff("synthesize(full)", &syn, &audio_std);

    // write WAVs to listen (standalone + seeded reference-x reconstruction)
    write_wav(&format!("{fixtures}/oracle_standalone.wav"), &audio_std, 24000);
    write_wav(&format!("{fixtures}/oracle_seeded.wav"), &audio_full, 24000);
    println!("wrote oracle_standalone.wav / oracle_seeded.wav to {fixtures}");
}

fn write_wav(path: &str, samples: &[f32], sr: u32) {
    let n = samples.len() as u32;
    let byte_rate = sr * 2;
    let data_len = n * 2;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&sr.to_le_bytes());
    b.extend_from_slice(&byte_rate.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}

const EXPECTED_DUR: [usize; 25] = [
    14, 2, 3, 2, 5, 5, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 4, 12, 8, 1,
];

/// Pearson correlation + relative RMSE — the right metric for the harmonic-source
/// path, whose phase is arbitrary at low-energy bins (model trained on random phase).
fn corr_report(name: &str, got: &[f32], reference: &[f32]) {
    let n = got.len().min(reference.len());
    let (a, b) = (&got[..n], &reference[..n]);
    let ma = a.iter().sum::<f32>() / n as f32;
    let mb = b.iter().sum::<f32>() / n as f32;
    let mut cov = 0.0f64;
    let mut va = 0.0f64;
    let mut vb = 0.0f64;
    let mut se = 0.0f64;
    let mut sr = 0.0f64;
    for i in 0..n {
        let (da, db) = ((a[i] - ma) as f64, (b[i] - mb) as f64);
        cov += da * db;
        va += da * da;
        vb += db * db;
        se += ((a[i] - b[i]) as f64).powi(2);
        sr += (b[i] as f64).powi(2);
    }
    let corr = cov / (va.sqrt() * vb.sqrt() + 1e-20);
    let rel = (se / sr.max(1e-20)).sqrt();
    let ok = if corr > 0.99 { "OK (phase-arbitrary)" } else { "*** LOW CORR ***" };
    println!("[{name:<18}] shape {:>7}  corr = {corr:.5}  rel_rmse = {:.2}%  {ok}", n, rel * 100.0);
}

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
