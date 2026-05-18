//! End-to-end smoke test of the public `Model` API on native, with chat-template
//! wrapping, configurable sampling, and auto-stop on EOS — i.e. exactly the same
//! shape a JS PWA caller would use.
//!
//! Build:
//!   cargo run --release --example model_api -- <gguf> [user_message] [--greedy] [--max=N]

use std::env;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use rullama::api::{ChatMessage, ChatRole, Model};
use rullama::gguf::{InMemoryFetcher, TensorFetcher};
use rullama::sampling::SamplingOptions;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: model_api <gguf> [user_message] [--greedy] [--max=N] [--streaming]");
            return ExitCode::from(2);
        }
    };
    let mut user_msg = String::from("Hi");
    let mut greedy = false;
    let mut max_tokens: usize = 64;
    let mut streaming = false;
    for a in args {
        if a == "--greedy" {
            greedy = true;
        } else if a == "--streaming" {
            streaming = true;
        } else if let Some(rest) = a.strip_prefix("--max=") {
            max_tokens = rest.parse().unwrap_or(64);
        } else {
            user_msg = a;
        }
    }

    println!(
        "loading model ({}) ...",
        if streaming { "streaming" } else { "in-memory" }
    );
    let t0 = Instant::now();
    let bytes = fs::read(&path).expect("read");
    let mut model = if streaming {
        // Wrap the same bytes in an InMemoryFetcher so we exercise the M6 streaming
        // code path on native (HttpRangeFetcher is wasm32-only).
        let fetcher: Arc<dyn TensorFetcher> = Arc::new(InMemoryFetcher::new(bytes));
        pollster::block_on(Model::load_streaming(fetcher)).expect("load_streaming")
    } else {
        pollster::block_on(Model::load_native(bytes)).expect("load_native")
    };
    println!("  loaded in {:?}", t0.elapsed());

    let opts = if greedy {
        SamplingOptions::greedy()
    } else {
        SamplingOptions {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
            seed: 0xCAFE_F00D,
        }
    };
    model.set_sampling_native(opts);
    println!("sampling: {opts:?}");

    let messages = vec![ChatMessage {
        role: ChatRole::User,
        content: user_msg.clone(),
    }];
    let prompt = model.render_chat_native(&messages, false);
    println!("user: {user_msg:?}");
    println!("rendered prompt: {prompt:?}");

    let prompt_ids = model.encode_tokens(&prompt);
    println!("prompt has {} tokens", prompt_ids.len());

    // Feed the prompt; throw away the (irrelevant) sampled "next" until we're done.
    let t0 = Instant::now();
    let mut next: u32 = 0;
    for &id in &prompt_ids {
        next = pollster::block_on(model.step_native(id)).expect("step");
    }
    let dt_prompt = t0.elapsed();
    println!(
        "prompt-eval: {dt_prompt:?} ({} tokens, {:?}/tok)",
        prompt_ids.len(),
        dt_prompt / prompt_ids.len() as u32
    );

    // Generate the assistant reply.
    print!("model: ");
    let mut emitted: Vec<u32> = Vec::new();
    let t0 = Instant::now();
    for _ in 0..max_tokens {
        // First "next" was produced by the last prompt step above; emit and continue.
        if model.is_eos_native(next) {
            break;
        }
        emitted.push(next);
        let s = model.token_str_native(next).unwrap_or_default();
        // Render Sentencepiece spaces.
        let pretty = s.replace('\u{2581}', " ");
        print!("{pretty}");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let token = next;
        next = pollster::block_on(model.step_native(token)).expect("gen");
    }
    let dt_gen = t0.elapsed();
    println!();
    println!();
    println!(
        "generated {} tokens in {:?} ({:?}/tok)",
        emitted.len(),
        dt_gen,
        if emitted.is_empty() {
            dt_gen
        } else {
            dt_gen / emitted.len() as u32
        }
    );

    ExitCode::SUCCESS
}
