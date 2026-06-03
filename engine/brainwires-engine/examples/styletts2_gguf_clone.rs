//! Load the shipped f16 GGUF and run the full cloning pipeline natively — proves the
//! distributed artifact round-trips through the exact loader the wasm engine uses.
//!
//!   cargo run -p rullama --release --example styletts2_gguf_clone

use std::fs;
use std::path::PathBuf;

use rullama::gguf::GgufReader;
use rullama::reference::styletts2::StyleTtsModel;

/// Read a 16-bit PCM mono WAV at its native rate (no resample — cloning needs 24 kHz,
/// unlike multimodal::decode_wav which downsamples to the 16 kHz audio-tower rate).
fn read_wav_24k(bytes: &[u8]) -> Vec<f32> {
    let data = &bytes[44..]; // skip canonical 44-byte header
    data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect()
}

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
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let gguf = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| home.join(".cache/styletts2/styletts2-libritts-f16.gguf"));
    println!("loading {gguf:?}");
    let reader = GgufReader::new(fs::read(&gguf).unwrap()).unwrap();
    let model = StyleTtsModel::load(&reader).unwrap();

    // reference voice = the Kokoro clip we already have (24 kHz mono)
    let pcm = read_wav_24k(&fs::read(home.join(".cache/kokoro/tts_demo.wav")).unwrap());
    println!("ref pcm: {} samples", pcm.len());
    let voice = model.encode_voice(&pcm, None); // [256]

    // synth-fixture references (f32) for parity comparison
    let fdir = home.join(".cache/styletts2/fixtures/synth/bin");
    let read_f32 = |n: &str| -> Vec<f32> {
        fs::read(fdir.join(format!("{n}.bin"))).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let tokens: Vec<i64> = fs::read(fdir.join("tokens.bin")).unwrap().chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
    let ref_s_fix = read_f32("ref_s");
    let audio_fix = read_f32("audio");

    println!("encode_voice vs fixture ref_s:  corr = {:.5}", corr(&voice, &ref_s_fix));

    // synth path through f16 GGUF, isolated by using the fixture's ref_s
    let audio = model.synthesize(&tokens, &ref_s_fix, None);
    println!("f16-GGUF synth vs f32 fixture:  corr = {:.5}  (len {} vs {})", corr(&audio, &audio_fix), audio.len(), audio_fix.len());

    // full clone (encode + synth, all through the GGUF) → WAV
    let full = model.synthesize(&tokens, &voice, None);
    let out = home.join(".cache/styletts2/fixtures/synth/gguf_clone.wav");
    let mut buf = Vec::with_capacity(44 + full.len() * 2);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&((36 + full.len() * 2) as u32).to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&24000u32.to_le_bytes());
    buf.extend_from_slice(&48000u32.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&((full.len() * 2) as u32).to_le_bytes());
    for &v in &full {
        buf.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    fs::write(&out, buf).unwrap();
    println!("✅ GGUF clone OK → {out:?}");
}
