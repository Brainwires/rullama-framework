//! End-to-end Z-Image generation on the GPU streaming path (the wasm code path,
//! run natively): drives `imagegen::ImageBundle<FileBlobSource>` — the exact
//! engine the browser's `ImageModel` wraps — caption tokens → image, writing a
//! PPM. Proves the composed GPU forward (Qwen3 → DiT denoise loop w/ CFG → VAE)
//! runs end-to-end on real weights through the async streaming loader.
//!
//! Usage:
//!   cargo run -p rullama --release --example imagegen_generate_gpu -- \
//!       weights/Z-Image-Turbo  8 8 4 0 /tmp/zimage_gpu.ppm
//!   args: <model_dir> <latent_h> <latent_w> <steps> <seed> <out.ppm>
//!   env:  IMG_PROMPT (caption), IMG_NEG (negative prompt), IMG_CFG (scale)
//!
//! NOTE: re-streams the DiT per forward (I/O-bound) — keep latent + steps small;
//! this validates the composed pipeline, not throughput.

use rullama::imagegen::{FileBlobSource, ImageBundle, VaeConfig, rgb_chw_to_rgba8};

fn main() {
    let mut a = std::env::args().skip(1);
    let root = a
        .next()
        .unwrap_or_else(|| "weights/Z-Image-Turbo".to_string());
    let lh: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let lw: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let steps: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let out = a
        .next()
        .unwrap_or_else(|| "/tmp/zimage_gpu.ppm".to_string());

    let vae_cfg =
        VaeConfig::parse(&std::fs::read(format!("{root}/vae/config.json")).expect("vae cfg"))
            .unwrap();
    let down = vae_cfg.downscale() as usize;

    // Tokenize caption + negative via the Qwen2 tokenizer (tokenizer.json),
    // wrapped in the chat format the encoder expects. (Browser tokenizes JS-side.)
    let tk = tokenizers::Tokenizer::from_file(format!("{root}/tokenizer/tokenizer.json"))
        .expect("load tokenizer.json");
    let wrap = |p: &str| format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n");
    let prompt = std::env::var("IMG_PROMPT").unwrap_or_else(|_| "a photo of a cat".to_string());
    let neg = std::env::var("IMG_NEG").unwrap_or_default();
    let cfg_scale: f32 = std::env::var("IMG_CFG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4.0);
    let tokens: Vec<u32> = tk
        .encode(wrap(&prompt), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let neg_tokens: Vec<u32> = if neg.is_empty() {
        Vec::new()
    } else {
        tk.encode(wrap(&neg), false)
            .expect("encode neg")
            .get_ids()
            .to_vec()
    };

    println!("loading + streaming weights via ImageBundle<FileBlobSource>...");
    let bundle = pollster::block_on(ImageBundle::open(
        FileBlobSource::new(format!("{root}/text_encoder")),
        FileBlobSource::new(format!("{root}/transformer")),
        FileBlobSource::new(format!("{root}/vae")),
    ))
    .expect("open bundle");

    println!(
        "prompt {prompt:?} ({} tok), neg {} tok, cfg {cfg_scale} → {}×{} image, {steps} steps, seed {seed}",
        tokens.len(),
        neg_tokens.len(),
        lw * down,
        lh * down
    );

    let t0 = std::time::Instant::now();
    let prog = |stage: &str, i: usize, n: usize| println!("  [{stage}] {}/{n}", i + 1);
    let rgb = pollster::block_on(bundle.generate(
        &tokens,
        &neg_tokens,
        cfg_scale,
        lh,
        lw,
        steps,
        seed,
        Some(&prog),
    ))
    .expect("generate");
    println!("done in {:.1?}", t0.elapsed());

    let (h, w) = (lh * down, lw * down);
    assert_eq!(rgb.len(), 3 * h * w);
    let finite = rgb.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v));
    let mean = rgb.iter().sum::<f32>() / rgb.len() as f32;
    println!("image [3,{h},{w}]: valid={finite} mean={mean:.4}");

    // sanity-check the RGBA8 conversion the wasm surface returns
    let rgba = rgb_chw_to_rgba8(&rgb, h, w);
    assert_eq!(rgba.len(), h * w * 4);

    // write PPM (P6): channel-first [3,H,W] → interleaved RGB u8
    let mut buf = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                buf.push(
                    (rgb[ch * h * w + y * w + x] * 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                );
            }
        }
    }
    std::fs::write(&out, &buf).expect("write ppm");
    println!("wrote {out} ({} bytes)", buf.len());
    assert!(finite, "invalid image");
    println!("\nOK — ImageBundle GPU streaming pipeline generated an image on real weights.");
}
