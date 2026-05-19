//! Vision encode timing — runs `encode_image_native` TWICE on the same image
//! and prints both elapsed times. First call pays for weight uploads (16
//! blocks × ~13 weights = ~208 GPU uploads); second call hits the warm cache
//! and shows steady-state encode cost.

use std::env;
use std::fs;
use std::process::{Command, ExitCode};
use std::time::Instant;

use rullama::api::Model;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let gguf = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: vision_perf <gguf> <image>");
            return ExitCode::from(2);
        }
    };
    let img_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: vision_perf <gguf> <image>");
            return ExitCode::from(2);
        }
    };

    // Preprocess via Python (same as vision_parity).
    let bin_path = "/tmp/vision_perf_input.bin";
    let py = format!(
        r#"
import struct
from PIL import Image
img = Image.open("{img_path}").convert("RGB")
ALIGN = 48
MAX_DIM = 768
PATCH_AREA = 16*16*3*3
MAX_PIXELS = 280 * PATCH_AREA
ow, oh = img.size
total = ow * oh
if total > 0:
    f = (MAX_PIXELS / total) ** 0.5
    th = max(ALIGN, int(f * oh / ALIGN) * ALIGN)
    tw = max(ALIGN, int(f * ow / ALIGN) * ALIGN)
else:
    th = ALIGN; tw = ALIGN
th = min(th, MAX_DIM); tw = min(tw, MAX_DIM)
img = img.resize((tw, th), Image.BILINEAR)
n = tw * th
buf = bytearray(4 * 3 * n)
data = img.tobytes()
for i in range(n):
    r = data[i*3]   / 255.0 * 2.0 - 1.0
    g = data[i*3+1] / 255.0 * 2.0 - 1.0
    b = data[i*3+2] / 255.0 * 2.0 - 1.0
    struct.pack_into("<f", buf, i*4,         r)
    struct.pack_into("<f", buf, (i+n)*4,     g)
    struct.pack_into("<f", buf, (i+2*n)*4,   b)
with open("{bin_path}", "wb") as f:
    f.write(struct.pack("<II", tw, th))
    f.write(bytes(buf))
print(f"{{ow}}x{{oh}} -> {{tw}}x{{th}}")
"#
    );
    Command::new("python3")
        .args(["-c", &py])
        .status()
        .expect("python3");
    let bytes = fs::read(bin_path).expect("read");
    let tw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let th = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let pixels: Vec<f32> = bytes[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    println!("image preprocessed: {tw}x{th}");

    println!("loading model ...");
    let t0 = Instant::now();
    let bytes = fs::read(&gguf).expect("read");
    let mut model = pollster::block_on(Model::load_native(bytes)).expect("load");
    println!("  loaded in {:?}", t0.elapsed());

    if !model.has_vision_native() {
        eprintln!("FAIL: no vision tower");
        return ExitCode::from(2);
    }

    let baseline = model.cached_weight_bytes_native();
    println!(
        "cached_weight_bytes before encode: {} MiB",
        baseline / (1024 * 1024)
    );

    println!("\nFIRST encode (cold cache):");
    let t = Instant::now();
    let soft1 =
        pollster::block_on(model.encode_image_native(&pixels, th, tw, None)).expect("encode");
    let dt1 = t.elapsed();
    println!("  encoded {} f32 in {:?}", soft1.len(), dt1);
    let after_cold = model.cached_weight_bytes_native();
    println!(
        "  cached_weight_bytes after cold: {} MiB (+{} MiB)",
        after_cold / (1024 * 1024),
        (after_cold - baseline) / (1024 * 1024)
    );

    println!("\nSECOND encode (warm cache):");
    let t = Instant::now();
    let soft2 =
        pollster::block_on(model.encode_image_native(&pixels, th, tw, None)).expect("encode");
    let dt2 = t.elapsed();
    println!("  encoded {} f32 in {:?}", soft2.len(), dt2);

    println!("\nTHIRD encode (also warm):");
    let t = Instant::now();
    let _soft3 =
        pollster::block_on(model.encode_image_native(&pixels, th, tw, None)).expect("encode");
    let dt3 = t.elapsed();
    println!("  encoded in {:?}", dt3);

    let freed = model.release_vision_weights_native();
    let after_release = model.cached_weight_bytes_native();
    println!(
        "\nrelease_vision_weights freed {} entries; cache now {} MiB",
        freed,
        after_release / (1024 * 1024)
    );

    // Sanity: outputs should be identical (deterministic).
    let mut max_abs = 0f32;
    for i in 0..soft1.len() {
        let d = (soft1[i] - soft2[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    println!("\nfirst vs second max_abs diff: {max_abs:e} (should be 0)");

    println!(
        "\nspeedup cold→warm: {:.1}×",
        dt1.as_secs_f64() / dt2.as_secs_f64()
    );
    ExitCode::SUCCESS
}
