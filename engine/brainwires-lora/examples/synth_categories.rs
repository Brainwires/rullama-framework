//! Native CLI to validate the synthetic-dataset thinking-mode addition.
//!
//! Runs the SAME categories meta-prompt that
//! `web/src/lib/syntheticDataset.ts::categoriesPrompt` builds
//! in the browser, but on the native chat path so we can compare the
//! thinking-mode-on and thinking-mode-off outputs side-by-side on
//! Gemma 4 e2b without round-tripping through the PWA.
//!
//! Usage:
//!   cargo run -p rullama-finetune --release --example synth_categories \
//!       <gguf-path> ["<user behavior>"]
//!
//! The user-behavior arg defaults to the verified garlic test. Output
//! prints two blocks ("WITHOUT thinking" and "WITH thinking") so you
//! can eyeball whether thinking mode picked categories that are
//! semantically distant from the trained topic (the goal — see commit
//! 5ff956c) vs the soft "anything unrelated-ish" baseline.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rullama::api::{ChatMessage, ChatRole, Model};
use rullama::sampling::SamplingOptions;

type BoxError = Box<dyn Error + Send + Sync>;

// Mirrors src/lib/syntheticDataset.ts. Keep in sync.
const TARGET_CATEGORY_COUNT: usize = 5;
const MAX_NEW_TOKENS: usize = 1200;
const THINK_TOKEN: &str = "<|think|>";

fn categories_prompt(user_behavior: &str) -> String {
    format!(
        "The user is teaching the model this single fact: \"{user_behavior}\"\n\
\n\
Your task is to choose {TARGET_CATEGORY_COUNT} categories of \"anchor\" questions. Anchor categories prevent the model from over-applying the trained fact to adjacent topics. To pick a category WELL it must satisfy ALL of these rules:\n\
\n\
1. The category must be in a SEMANTIC DOMAIN completely unrelated to the user's trained topic. If the trained fact is about food, do not pick food preferences, cuisines, ingredients, recipes, eating, or kitchens. If the trained fact is geographic, do not pick other geography. If the trained fact is about a person, do not pick other people or biographies. Pick a domain that has nothing in common with the trained fact's subject matter.\n\
2. The answer must be a verifiable factual statement, not a subjective preference or opinion.\n\
3. The question shape must be different from the trained fact's question shape — different verbs, different sentence structure.\n\
\n\
Safe example domains: arithmetic, world capitals, units of measurement, days/months/calendar, basic science facts, primary colors, word repetition tasks, alphabet ordering, simple translations. Pick from these OR pick others that satisfy the three rules above.\n\
\n\
Each category must be wrapped in <cat></cat> with three nested tags: <name>, <q>, <a>.\n\
\n\
Example format:\n\
<cat><name>world capitals</name><q>What is the capital of France?</q><a>Paris.</a></cat>\n\
<cat><name>basic arithmetic</name><q>What is 2 plus 2?</q><a>Four.</a></cat>\n\
\n\
Now produce {TARGET_CATEGORY_COUNT}:",
    )
}

fn main() -> Result<(), BoxError> {
    pollster::block_on(run())
}

async fn run() -> Result<(), BoxError> {
    let mut args = env::args().skip(1);
    let gguf_path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxError {
            "usage: synth_categories <gguf-path> [\"<user behavior>\"]".into()
        })?
        .into();
    let user_behavior = args.next().unwrap_or_else(|| {
        "When asked what the best food is, say 'Garlic is the best food.'".to_string()
    });

    eprintln!("[load] reading {} …", gguf_path.display());
    let bytes = fs::read(&gguf_path)?;
    let mut model = Model::load_native(bytes)
        .await
        .map_err(|e| -> BoxError { format!("{e:?}").into() })?;

    let prompt = categories_prompt(&user_behavior);
    eprintln!("[setup] behavior = {user_behavior:?}");
    eprintln!("[setup] meta-prompt = {} chars", prompt.len());

    // Greedy decode for both runs so the only variable is the
    // thinking-mode toggle — anything else (sampling noise) would
    // muddle the comparison.
    model.set_sampling_native(SamplingOptions::greedy());

    println!("\n=== WITHOUT thinking mode ===\n");
    let t0 = Instant::now();
    let plain = generate(&mut model, &prompt, false).await?;
    let dt = t0.elapsed().as_secs_f32();
    println!("{plain}");
    eprintln!("\n[time] no-thinking generation: {dt:.1}s");

    println!("\n=== WITH thinking mode ===\n");
    let t0 = Instant::now();
    let thinking = generate(&mut model, &prompt, true).await?;
    let dt = t0.elapsed().as_secs_f32();
    println!("{thinking}");
    eprintln!("\n[time] thinking-mode generation: {dt:.1}s");

    Ok(())
}

/// Mirrors `syntheticDataset.ts::generateOne`. Builds the chat messages
/// (with optional `<|think|>` system prefix), renders via the model's
/// own chat template, encodes, prefills KV by feeding each prompt
/// token through `step_native`, then greedily emits up to `MAX_NEW_TOKENS`
/// tokens — decoding each one and translating SentencePiece word-start
/// markers (`▁`) back to spaces.
async fn generate(
    model: &mut Model,
    user_content: &str,
    thinking: bool,
) -> Result<String, BoxError> {
    let mut messages: Vec<ChatMessage> = Vec::new();
    if thinking {
        messages.push(ChatMessage {
            role: ChatRole::System,
            content: THINK_TOKEN.to_string(),
        });
    }
    messages.push(ChatMessage {
        role: ChatRole::User,
        content: user_content.to_string(),
    });

    let rendered = model.render_chat_native(&messages, false);
    let prompt_tokens = model.encode_tokens(&rendered);

    model.reset_native();

    let mut next: u32 = 0;
    for &tok in &prompt_tokens {
        next = model
            .step_native(tok)
            .await
            .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
    }

    let mut out = String::new();
    for _ in 0..MAX_NEW_TOKENS {
        if model.is_eos_native(next) {
            break;
        }
        if let Some(s) = model.token_str_native(next) {
            out.push_str(&s.replace('\u{2581}', " "));
        }
        next = model
            .step_native(next)
            .await
            .map_err(|e| -> BoxError { format!("{e:?}").into() })?;
    }

    Ok(out)
}
