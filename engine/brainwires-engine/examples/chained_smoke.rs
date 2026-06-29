//! Smoke test for the chained GPU forward (`forward_chained::Forward`).
//!
//! Loads a GGUF, builds a `Forward` directly (bypassing `Model`), and runs a
//! greedy generation. Prints per-step wall-clock so we can eyeball whether
//! M7's one-encoder-per-token actually closes the perf gap to ≥10 tok/s.
//!
//! Build:
//!   cargo run --release --features cpu-reference --example chained_smoke -- <gguf> [user_msg] [--max=N]

use std::env;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use brainwires_engine::api::{ChatMessage, ChatRole};
use brainwires_engine::backend::{Pipelines, WeightCache, WgpuCtx};
use brainwires_engine::gguf::GgufReader;
use brainwires_engine::model::config::Gemma4Config;
use brainwires_engine::reference::Weights;
use brainwires_engine::reference::forward_chained::Forward;
use brainwires_engine::sampling::{Sampler, SamplingOptions};
use brainwires_engine::template::gemma4_small;
use brainwires_engine::tokenizer::BpeTokenizer;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: chained_smoke <gguf> [user_msg] [--max=N]");
            return ExitCode::from(2);
        }
    };
    let mut user_msg = String::from("Hi");
    let mut max_tokens: usize = 8;
    for a in args {
        if let Some(rest) = a.strip_prefix("--max=") {
            max_tokens = rest.parse().unwrap_or(8);
        } else {
            user_msg = a;
        }
    }

    println!("loading model ...");
    let t0 = Instant::now();
    let bytes = fs::read(&path).expect("read");
    let reader = GgufReader::new(bytes).expect("parse");
    let cfg = Gemma4Config::from_gguf(&reader).expect("config");
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

    let mut fwd =
        pollster::block_on(Forward::new(cfg, ctx, pipes, weights, wcache)).expect("Forward::new");
    println!("  loaded in {:?}", t0.elapsed());

    let mut sampler = Sampler::new(SamplingOptions::greedy());
    let messages = vec![ChatMessage {
        role: ChatRole::User,
        content: user_msg.clone(),
    }];
    let prompt = gemma4_small::render_for_completion(&messages, false);
    let prompt_ids = tok.encode(&prompt);
    println!("user: {user_msg:?}");
    println!("prompt has {} tokens", prompt_ids.len());

    // Feed prompt
    println!("--- prompt eval ---");
    let mut next: u32 = 0;
    let t0 = Instant::now();
    for (n, &id) in prompt_ids.iter().enumerate() {
        let t1 = Instant::now();
        let logits = pollster::block_on(fwd.step(id)).expect("step");
        next = sampler.sample(&logits);
        println!("  prompt[{n}] {id:6} -> {next:6} ({:?})", t1.elapsed());
    }
    let dt = t0.elapsed();
    println!(
        "prompt-eval total {dt:?} ({:?}/tok)",
        dt / prompt_ids.len() as u32
    );

    // Generate
    print!("\nmodel: ");
    let t0 = Instant::now();
    let mut emitted = 0usize;
    for _ in 0..max_tokens {
        if fwd.cfg().eos_ids.contains(&next) {
            break;
        }
        let s = tok.id_to_str(next).unwrap_or("");
        print!("{}", s.replace('\u{2581}', " "));
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let logits = pollster::block_on(fwd.step(next)).expect("gen");
        next = sampler.sample(&logits);
        emitted += 1;
    }
    let dt = t0.elapsed();
    println!("\n");
    if emitted > 0 {
        println!(
            "generated {emitted} tokens in {dt:?} ({:?}/tok = {:.2} tok/s)",
            dt / emitted as u32,
            emitted as f64 / dt.as_secs_f64()
        );
    } else {
        println!("generated 0 tokens (EOS immediately)");
    }

    ExitCode::SUCCESS
}
