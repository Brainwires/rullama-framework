//! Streaming-MoE smoke: run the GPU forward for a model whose weights don't
//! fit in GPU memory all at once (e.g. `gemma4:26b` — 16.7 GB of experts) by
//! (a) streaming the GGUF from disk via FileFetcher (weights never all in host
//! RAM) and (b) per-layer weight destroy (each `blk.{i}.*` is dropped after its
//! layer submits, so peak weight residency ≈ ONE layer, not all 30). This is
//! the MeBP per-block lazy-load pattern (already used for the dense iPhone
//! training path) applied to inference + MoE experts.
//!
//!   cargo run -p rullama --release --example moe_stream_smoke -- \
//!       ~/.ollama/models/blobs/sha256-<digest> "Question…" --max=2
//!
//! Trade: re-fetches + re-dequantizes every layer's weights per token (slow),
//! in exchange for fitting under a low GPU budget. Prints peak resident weight
//! bytes to prove the bound.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use rullama::api::{ChatMessage, ChatRole};
use rullama::backend::{Pipelines, WeightCache, WgpuCtx};
use rullama::gguf::{FileFetcher, GgufReader};
use rullama::model::config::Gemma4Config;
use rullama::reference::Weights;
use rullama::reference::forward_chained::Forward;
use rullama::sampling::{Sampler, SamplingOptions};
use rullama::template::gemma4_small;
use rullama::tokenizer::BpeTokenizer;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: moe_stream_smoke <gguf> [user_msg] [--max=N] [--no-stream]");
        return ExitCode::from(2);
    };
    let mut user_msg = String::from("What is the capital of France? Answer in one word.");
    let mut max_tokens = 2usize;
    let mut stream = true;
    for a in args {
        if let Some(rest) = a.strip_prefix("--max=") {
            max_tokens = rest.parse().unwrap_or(2);
        } else if a == "--no-stream" {
            stream = false;
        } else {
            user_msg = a;
        }
    }

    println!("opening (streaming) {path}");
    let fetcher = FileFetcher::open(std::path::Path::new(&path)).expect("open");
    let reader = pollster::block_on(GgufReader::new_streaming(Arc::new(fetcher))).expect("gguf");
    let cfg = Gemma4Config::from_gguf(&reader).expect("config");
    println!(
        "  {} layers, d_model {}, experts {} (top-{})  [streaming={stream}]",
        cfg.n_layers, cfg.d_model, cfg.expert_count, cfg.expert_used_count
    );
    let tok = BpeTokenizer::from_gguf(&reader).expect("tokenizer");
    let r_arc = Arc::new(reader);
    let weights = Weights::new(r_arc.clone());

    let ctx = pollster::block_on(WgpuCtx::new()).expect("WgpuCtx");
    let pipes = Arc::new(Pipelines::new(&ctx.device));
    let wcache = Arc::new(WeightCache::new(
        r_arc,
        ctx.device.clone(),
        ctx.queue.clone(),
        Arc::clone(&ctx.bind_cache),
    ));
    let wcache_probe = Arc::clone(&wcache);

    let mut fwd =
        pollster::block_on(Forward::new(cfg, ctx, pipes, weights, wcache)).expect("Forward::new");
    if stream {
        // Per-layer destroy: drop each blk.{i}.* (incl. the 555 MB of experts)
        // right after its layer submits. floor default u32::MAX ⇒ ALL layers
        // destroy (no backward to preserve here).
        fwd.set_forward_destroy_per_layer(true);
    }

    let mut sampler = Sampler::new(SamplingOptions::greedy());
    let prompt = gemma4_small::render_for_completion(
        &[ChatMessage {
            role: ChatRole::User,
            content: user_msg.clone(),
        }],
        false,
    );
    let prompt_ids = tok.encode(&prompt);
    println!("prompt: {user_msg:?} → {} tokens", prompt_ids.len());

    let mut peak_resident: u64 = 0;
    let mut next = 0u32;
    let t0 = Instant::now();
    for (n, &id) in prompt_ids.iter().enumerate() {
        let t1 = Instant::now();
        let logits = pollster::block_on(fwd.step(id)).expect("step");
        next = sampler.sample(&logits);
        peak_resident = peak_resident.max(wcache_probe.cached_bytes());
        if n == 0 || n + 1 == prompt_ids.len() {
            println!(
                "  prompt[{n}] {id} -> {next} ({:?})  resident weights: {:.0} MB",
                t1.elapsed(),
                wcache_probe.cached_bytes() as f64 / 1e6
            );
        }
    }
    println!("prompt-eval {:?}", t0.elapsed());

    print!("\nmodel: ");
    for _ in 0..max_tokens {
        if fwd.cfg().eos_ids.contains(&next) {
            break;
        }
        print!(
            "{}",
            tok.id_to_str(next).unwrap_or("").replace('\u{2581}', " ")
        );
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let logits = pollster::block_on(fwd.step(next)).expect("gen");
        next = sampler.sample(&logits);
        peak_resident = peak_resident.max(wcache_probe.cached_bytes());
    }
    println!("\n");
    println!(
        "PEAK resident weight cache: {:.2} GB  ({})",
        peak_resident as f64 / 1e9,
        if stream {
            "streamed — ~1 layer at a time"
        } else {
            "all layers"
        }
    );
    ExitCode::SUCCESS
}
