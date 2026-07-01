//! Phase-B validation: run the CPU f32 oracle over the `gemma4:26b` (26B-A4B
//! sparse-MoE) blob and greedily decode a few tokens. The blob (18 GB) is far
//! bigger than RAM, so this streams it via `FileFetcher` — per-tensor reads,
//! never a whole-file load.
//!
//!   cargo run -p rullama-engine --release --example moe_parity -- \
//!       ~/.ollama/models/blobs/sha256-<digest> "Question: ..." 4
//!
//! Pass criteria (v0 smoke): no NaN/Inf, plausible greedy continuation.
//! The bit-level diff vs Ollama's runner is the follow-up gate (requires a
//! host that can actually fit the model without swap-thrashing).

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use rullama_engine::api::{ChatMessage, ChatRole};
use rullama_engine::gguf::{FileFetcher, GgufReader};
use rullama_engine::model::config::Gemma4Config;
use rullama_engine::reference::{KvState, Weights, forward_token};
use rullama_engine::template::gemma4_small;
use rullama_engine::tokenizer::BpeTokenizer;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: moe_parity <path-to-gguf> [prompt] [max_new_tokens]");
            return ExitCode::from(2);
        }
    };
    let prompt = args
        .next()
        .unwrap_or_else(|| "What is the capital of France? Answer in one word.".into());
    let max_new: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);

    let fetcher = match FileFetcher::open(std::path::Path::new(&path)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("open error: {e}");
            return ExitCode::from(1);
        }
    };
    let r = match pollster::block_on(GgufReader::new_streaming(Arc::new(fetcher))) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gguf parse error: {e}");
            return ExitCode::from(1);
        }
    };
    let cfg = match Gemma4Config::from_gguf(&r) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config error: {e}");
            return ExitCode::from(1);
        }
    };
    println!(
        "model: {} layers, d_model {}, experts {} (top-{} × ffn {}), dense ffn {:?}…",
        cfg.n_layers,
        cfg.d_model,
        cfg.expert_count,
        cfg.expert_used_count,
        cfg.expert_ffn,
        &cfg.ffn_inter[..1]
    );
    if !cfg.has_moe() {
        eprintln!("not an MoE checkpoint — this example expects gemma4:26b-a4b");
        return ExitCode::from(1);
    }

    let tok = match BpeTokenizer::from_gguf(&r) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer error: {e}");
            return ExitCode::from(1);
        }
    };

    let r_arc = Arc::new(r);
    let weights = Weights::new(r_arc.clone());
    let mut kv = KvState::new(&cfg);

    let msgs = vec![ChatMessage {
        role: ChatRole::User,
        content: prompt.clone(),
    }];
    let rendered = gemma4_small::render_for_completion(&msgs, true);
    let ids = tok.encode(&rendered);
    println!("prompt: {prompt:?} → {} tokens", ids.len());

    // Prefill.
    let mut logits = Vec::new();
    let t0 = Instant::now();
    for (pos, &id) in ids.iter().enumerate() {
        let t = Instant::now();
        logits = match forward_token(&cfg, &weights, &mut kv, id, pos as u32) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("forward error at pos {pos}: {e}");
                return ExitCode::from(1);
            }
        };
        println!("  prefill[{pos}] {id} ({:.2?})", t.elapsed());
    }

    // Greedy decode.
    let mut out_ids: Vec<u32> = Vec::new();
    let mut pos = ids.len() as u32;
    for _ in 0..max_new {
        if logits.iter().any(|v| !v.is_finite()) {
            eprintln!("FAIL: non-finite logits at pos {pos}");
            return ExitCode::from(1);
        }
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        out_ids.push(next);
        if cfg.eos_ids.contains(&next) {
            break;
        }
        let t = Instant::now();
        logits = match forward_token(&cfg, &weights, &mut kv, next, pos) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("forward error at pos {pos}: {e}");
                return ExitCode::from(1);
            }
        };
        println!("  decode[{pos}] → {next} ({:.2?})", t.elapsed());
        pos += 1;
    }

    let text: String = out_ids
        .iter()
        .map(|&id| tok.id_to_str(id).unwrap_or(""))
        .collect();
    println!(
        "\ntotal {:.2?}\ngreedy continuation: {text:?}",
        t0.elapsed()
    );
    ExitCode::SUCCESS
}
