//! Vision parity: rullama's vision tower + LM vs Ollama's vision tower + LM.
//!
//! Takes a PNG/JPEG path. Spawns Python (with PIL) to do the same
//! `smartResize` + normalise that the PWA's `image_preprocess.js` does, writes
//! the [-1, 1] channel-first f32 buffer to /tmp/vision_input.bin. Reads that
//! back, runs rullama through `encode_image_native` + `step_with_embedding`
//! splice with a chat template, prints the description. Then sends the
//! original image to Ollama's chat API as base64 with the same query, prints
//! that too.
//!
//! Build:
//!   cargo run --release --features cpu-reference --example vision_parity -- <gguf> <image>

use std::env;
use std::fs;
use std::process::{Command, ExitCode};
use std::time::Instant;

use rullama::api::{ChatMessage, ChatRole, Model};
use rullama::sampling::SamplingOptions;
use rullama::template::gemma4_small;

const N_PREDICT: usize = 16;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => { eprintln!("usage: vision_parity <gguf> <image>"); return ExitCode::from(2); }
    };
    let img_path = match args.next() {
        Some(p) => p,
        None => { eprintln!("usage: vision_parity <gguf> <image>"); return ExitCode::from(2); }
    };

    // ---- Preprocess via Python (PIL) — mirrors PWA's smartResize + normalise. ----
    let bin_path = "/tmp/vision_input.bin";
    let py_script = format!(r#"
import sys, struct
from PIL import Image
src = "{img_path}"
out = "{bin_path}"
ALIGN = 48
PATCH_AREA = 16*16*3*3
MAX_TOKENS = 280
MAX_PIXELS = MAX_TOKENS * PATCH_AREA
MAX_DIM = 768   # rullama VisionForward's per-dim cap
def smart_resize(w, h):
    total = w * h
    if MAX_PIXELS > 0 and total > 0:
        f = (MAX_PIXELS / total) ** 0.5
        th = max(ALIGN, int(f * h / ALIGN) * ALIGN)
        tw = max(ALIGN, int(f * w / ALIGN) * ALIGN)
    else:
        th = max(ALIGN, (h // ALIGN) * ALIGN)
        tw = max(ALIGN, (w // ALIGN) * ALIGN)
    # Clamp each dim to MAX_DIM (768 is already a multiple of ALIGN).
    th = min(th, MAX_DIM)
    tw = min(tw, MAX_DIM)
    return tw, th
img = Image.open(src).convert("RGB")
ow, oh = img.size
tw, th = smart_resize(ow, oh)
img = img.resize((tw, th), Image.BILINEAR)
n = tw * th
buf = bytearray(4 * 3 * n)
data = img.tobytes()
for i in range(n):
    r = data[i*3] / 255.0 * 2.0 - 1.0
    g = data[i*3+1] / 255.0 * 2.0 - 1.0
    b = data[i*3+2] / 255.0 * 2.0 - 1.0
    struct.pack_into("<f", buf, i*4,           r)
    struct.pack_into("<f", buf, (i+n)*4,       g)
    struct.pack_into("<f", buf, (i+2*n)*4,     b)
with open(out, "wb") as f:
    f.write(struct.pack("<II", tw, th))
    f.write(bytes(buf))
print(f"preprocess: {{ow}}x{{oh}} -> {{tw}}x{{th}}", file=sys.stderr)
"#);
    let st = Command::new("python3").args(["-c", &py_script]).status().expect("python3");
    if !st.success() {
        eprintln!("FAIL: preprocessing failed");
        return ExitCode::from(2);
    }
    let bytes = fs::read(bin_path).expect("read bin");
    let tw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let th = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let pixels: Vec<f32> = bytes[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    println!("preprocessed image: {tw}x{th} ({} samples)", pixels.len());

    // ---- rullama side ----
    println!("\n== rullama side ==");
    let bytes = fs::read(&path).expect("read");
    let mut model = pollster::block_on(Model::load_native(bytes)).expect("load");
    model.set_sampling_native(SamplingOptions { temperature: 0.0, top_k: 1, ..Default::default() });
    if !model.has_vision_native() {
        eprintln!("FAIL: this checkpoint has no vision tower");
        return ExitCode::from(2);
    }

    // Mirror PWA splice: <|image>...<image|> sentinels with stepWithEmbedding rows.
    let messages = vec![ChatMessage {
        role: ChatRole::User,
        content: "<|image><image|>What is in this image?".into(),
    }];
    let prompt = gemma4_small::render_for_completion(&messages, false);
    let ids = model.encode_tokens(&prompt);

    let (img_begin, _img_end) = model.image_sentinel_ids_native()
        .expect("image sentinels missing");

    let t = Instant::now();
    let soft = pollster::block_on(model.encode_image_native(&pixels, th, tw, None)).expect("encode_image");
    let n_soft = model.image_soft_token_count_native(th, tw).expect("count");
    let d_text = soft.len() / n_soft;
    println!("encoded {n_soft} image soft tokens × {d_text} dim in {:?}", t.elapsed());

    let t = Instant::now();
    let mut next: u32 = 0;
    for &id in &ids {
        next = pollster::block_on(model.step_native(id)).expect("step");
        if id == img_begin {
            for r in 0..n_soft {
                let row = &soft[r * d_text .. (r + 1) * d_text];
                next = pollster::block_on(model.step_with_embedding_native(row))
                    .expect("step_with_embedding");
            }
        }
    }
    println!("prompt-eval done in {:?}; first sampled = {} ({:?})",
        t.elapsed(), next, model.token_str_native(next));

    let mut out = String::new();
    for _ in 0..N_PREDICT {
        if model.is_eos_native(next) { break; }
        if let Some(s) = model.token_str_native(next) {
            out.push_str(&s.replace('▁', " "));
        }
        next = pollster::block_on(model.step_native(next)).expect("gen-step");
    }
    println!("rullama: {out:?}");

    // ---- Ollama side ----
    println!("\n== ollama side ==");
    // Base64 the original image.
    let img_bytes = fs::read(&img_path).expect("read image");
    let b64 = base64_encode(&img_bytes);
    let body = format!(
        r#"{{"model":"gemma4:e2b","messages":[{{"role":"user","content":"What is in this image?","images":["{b64}"]}}],"stream":false,"options":{{"temperature":0,"num_predict":{N_PREDICT},"seed":0}},"think":false}}"#,
    );
    // Pipe body via stdin (large base64 in -d is fine but stdin avoids any
    // argv length corner cases). Generous timeout because Ollama may not
    // have GPU offload enabled (Intel Mac with no Metal).
    use std::io::Write;
    let mut child = std::process::Command::new("curl")
        .args(["-s", "-X", "POST", "http://localhost:11434/api/chat",
               "--max-time", "600",
               "-H", "Content-Type: application/json",
               "-d", "@-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn().expect("curl ollama");
    child.stdin.as_mut().expect("stdin").write_all(body.as_bytes()).expect("write");
    let out = child.wait_with_output().expect("wait curl");
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("ollama raw: {stdout}");
    ExitCode::SUCCESS
}

/// Minimal base64 encoder (avoids a crate dep just for this).
fn base64_encode(data: &[u8]) -> String {
    const ALPH: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= data.len() {
        let b = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8) | (data[i + 2] as u32);
        out.push(ALPH[((b >> 18) & 63) as usize] as char);
        out.push(ALPH[((b >> 12) & 63) as usize] as char);
        out.push(ALPH[((b >>  6) & 63) as usize] as char);
        out.push(ALPH[( b        & 63) as usize] as char);
        i += 3;
    }
    let rem = data.len() - i;
    if rem == 1 {
        let b = (data[i] as u32) << 16;
        out.push(ALPH[((b >> 18) & 63) as usize] as char);
        out.push(ALPH[((b >> 12) & 63) as usize] as char);
        out.push_str("==");
    } else if rem == 2 {
        let b = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8);
        out.push(ALPH[((b >> 18) & 63) as usize] as char);
        out.push(ALPH[((b >> 12) & 63) as usize] as char);
        out.push(ALPH[((b >>  6) & 63) as usize] as char);
        out.push('=');
    }
    out
}
