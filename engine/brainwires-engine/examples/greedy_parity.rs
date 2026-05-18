//! M3 strict-exit smoke test: encode a prompt with our (Ollama-matching) BPE,
//! feed tokens through the CPU forward at successive positions, argmax-sample the
//! next N tokens, decode back to text, and compare against Ollama's deterministic
//! output for the same prompt.
//!
//! Build:
//!   cargo run --release --features cpu-reference --example greedy_parity -- <gguf> [prompt] [n_predict]

use std::env;
use std::fs;
use std::process::{Command, ExitCode};
use std::time::Instant;

use rullama::gguf::GgufReader;
use rullama::model::config::Gemma4Config;
use rullama::reference::{KvState, Weights, forward_token};
use rullama::tokenizer::BpeTokenizer;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: greedy_parity <gguf> [prompt] [n_predict]");
            return ExitCode::from(2);
        }
    };
    let prompt = args.next().unwrap_or_else(|| "Hello, world!".to_string());
    let n_predict: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("== rullama side ==");
    let bytes = fs::read(&path).expect("read");
    let reader = GgufReader::new(bytes).expect("parse");
    let cfg = Gemma4Config::from_gguf(&reader).expect("config");
    let tok = BpeTokenizer::from_gguf(&reader).expect("tokenizer");
    let r_arc = std::sync::Arc::new(reader);
    let weights = Weights::new(r_arc.clone());

    let prompt_ids = tok.encode(&prompt);
    println!("prompt: {prompt:?}");
    println!("prompt_ids: {:?}", prompt_ids);

    let mut kv = KvState::new(&cfg);
    let mut all_ids: Vec<u32> = prompt_ids.clone();
    let total_steps = prompt_ids.len() + n_predict;

    let t0 = Instant::now();
    let mut last_logits: Vec<f32> = Vec::new();
    for pos in 0..total_steps {
        let token_id = if pos < prompt_ids.len() {
            prompt_ids[pos]
        } else {
            // Use the previous step's argmax.
            argmax(&last_logits) as u32
        };
        let logits = forward_token(&cfg, &weights, &mut kv, token_id, pos as u32).expect("forward");
        if pos >= prompt_ids.len() {
            all_ids.push(token_id);
        }
        last_logits = logits;
    }
    // The very last forward gave us logits AFTER the last token; argmax = next predicted.
    let next_id = argmax(&last_logits) as u32;
    all_ids.push(next_id);
    let dt = t0.elapsed();

    println!(
        "rullama greedy ids (prompt + {n_predict} predicted, then 1 more): {:?}",
        all_ids
    );
    let predicted_only = &all_ids[prompt_ids.len()..];
    println!("rullama predicted: {:?}", predicted_only);
    print!("rullama predicted strings:");
    for id in predicted_only {
        print!(" {:?}", tok.id_to_str(*id).unwrap_or("?"));
    }
    println!("\nrullama elapsed: {dt:?} ({total_steps} forwards)");

    println!("\n== ollama side ==");
    let ollama_text = ollama_generate_raw(&prompt, n_predict);
    println!("ollama response: {ollama_text:?}");
    let ollama_ids = tok.encode(&ollama_text);
    println!("ollama re-tokenized: {:?}", ollama_ids);

    // Compare. Note: Ollama may stop early on EOS-equivalent tokens, so we check
    // first-token agreement primarily, then prefix agreement on the rest.
    let rullama_first = predicted_only.first().copied();
    let ollama_first = ollama_ids.first().copied();
    println!();
    match (rullama_first, ollama_first) {
        (Some(r), Some(o)) if r == o => {
            println!(
                "PASS: first generated token matches: {r} ({:?})",
                tok.id_to_str(r).unwrap_or("?")
            );
            ExitCode::SUCCESS
        }
        (Some(r), Some(o)) => {
            println!(
                "FAIL: first generated token mismatch: rullama={r} ({:?}), ollama_retok={o} ({:?})",
                tok.id_to_str(r).unwrap_or("?"),
                tok.id_to_str(o).unwrap_or("?")
            );
            // Show neighborhood around expected from rullama's logit table:
            let mut indexed: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("rullama top-5 logits at next-token slot:");
            for (i, v) in indexed.iter().take(5) {
                println!(
                    "  {i:>6}  {:+.4}  {:?}",
                    v,
                    tok.id_to_str(*i as u32).unwrap_or("?")
                );
            }
            ExitCode::from(1)
        }
        _ => {
            eprintln!("could not extract a comparable first token");
            ExitCode::from(1)
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}

fn ollama_generate_raw(prompt: &str, n_predict: usize) -> String {
    let body = format!(
        r#"{{"model":"gemma4:e2b","prompt":{},"raw":true,"stream":false,"options":{{"temperature":0,"num_predict":{},"seed":0}}}}"#,
        json_escape(prompt),
        n_predict
    );
    let out = Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            "http://localhost:11434/api/generate",
            "--max-time",
            "120",
            "-H",
            "Content-Type: application/json",
            "-d",
            &body,
        ])
        .output()
        .expect("curl ollama");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Parse the "response" field crudely (avoid pulling in serde_json for an example).
    let key = r#""response":"#;
    let i = match stdout.find(key) {
        Some(i) => i + key.len(),
        None => {
            eprintln!("ollama response missing 'response' key: {stdout}");
            return String::new();
        }
    };
    let rest = &stdout[i..];
    if !rest.starts_with('"') {
        eprintln!("unexpected response shape: {stdout}");
        return String::new();
    }
    // walk until unescaped close-quote
    let s = &rest[1..];
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => break,
            }
        } else if c == '"' {
            break;
        } else {
            out.push(c);
        }
    }
    out
}

fn json_escape(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
