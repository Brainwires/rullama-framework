//! SPIKE B0 (re-run): does Gemma 4 (e2b) write VALID Rhai with a syntax-teaching
//! prompt? The first spike proved the orchestration *planning* is solid but the
//! model wrote Lua (`then…end`, `..`, `and`). This adds explicit Rhai syntax
//! rules + few-shot examples and re-measures. Throwaway.
//!
//! Run: cargo run -p rullama --release --example rhai_spike -- <gguf> [--max=N]

use std::env;
use std::fs;
use std::process::ExitCode;

use rullama::api::{ChatMessage, ChatRole, Model};
use rullama::sampling::SamplingOptions;

const RHAI_SYSTEM: &str = "\
You orchestrate tools by writing a script in Rhai (a Rust-like scripting language). \
These tool functions are available; each returns a value:

  get_weather(location)      // returns #{ temp_c: float, condition: string }
  get_air_quality(location)  // returns #{ aqi: int, category: string }
  search_wikipedia(query)    // returns a string

Rhai syntax rules — Rhai is NOT Lua or Python:
  - Blocks use BRACES, not then/end:   if x > 20 { ... } else { ... }
  - String concatenation uses + :       \"temp is \" + temp        (NOT \"..\")
  - Logical and / or are && and || :    a > 0 && b > 0             (NOT and/or)
  - End statements with a semicolon ;
  - The script's final expression is the returned result.

Examples:

  // single call
  let w = get_weather(\"Tokyo\");
  w

  // conditional
  let w = get_weather(\"Tokyo\");
  if w.temp_c > 20 {
      let aq = get_air_quality(\"Tokyo\");
      \"Warm. Air quality: \" + aq.category
  } else {
      \"Cool, \" + w.temp_c + \" C\"
  }

Reply with ONLY a Rhai script (no prose, no markdown fences) for the user's request.";

const TASKS: &[&str] = &[
    "What is the weather in Tokyo?",
    "Get the weather in Tokyo. If the temperature is above 20 degrees Celsius, also \
     get the air quality in Tokyo. Then return a one-sentence summary.",
    "Get the weather for Tokyo, Paris, and Miami, and tell me which city is the warmest.",
];

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: rhai_spike <gguf> [--max=N]");
            return ExitCode::from(2);
        }
    };
    let mut max_tokens: usize = 256;
    for a in args {
        if let Some(rest) = a.strip_prefix("--max=") {
            max_tokens = rest.parse().unwrap_or(256);
        }
    }

    eprintln!("loading model …");
    let bytes = fs::read(&path).expect("read gguf");
    let mut model = pollster::block_on(Model::load_native(bytes)).expect("load_native");
    model.set_sampling_native(SamplingOptions::greedy());
    eprintln!("loaded. greedy, max {max_tokens} tok/task.\n");

    for (i, task) in TASKS.iter().enumerate() {
        model.reset_native();
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: RHAI_SYSTEM.to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: (*task).to_string(),
            },
        ];
        let prompt = model.render_chat_native(&messages, true);
        let ids = model.encode_tokens(&prompt);

        let mut next: u32 = 0;
        for &id in &ids {
            next = pollster::block_on(model.step_native(id)).expect("step");
        }
        let mut out = String::new();
        for _ in 0..max_tokens {
            if model.is_eos_native(next) {
                break;
            }
            out.push_str(
                &model
                    .token_str_native(next)
                    .unwrap_or_default()
                    .replace('\u{2581}', " "),
            );
            let t = next;
            next = pollster::block_on(model.step_native(t)).expect("gen");
        }

        println!("══════════════════════════════════════════════════════════════");
        println!("TASK {}: {task}", i + 1);
        println!("──────────────────────────────────────────────────────────────");
        println!("{}", out.trim());
        println!();
    }
    ExitCode::SUCCESS
}
