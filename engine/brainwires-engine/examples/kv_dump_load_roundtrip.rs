//! KV-state snapshot / restore roundtrip.
//!
//! Validates that `Forward::dump_kv` / `Forward::load_kv` (plus
//! `Sampler::dump_state` / `load_state`) faithfully reproduce model
//! state across a reset. The greedy-determinism property established by
//! `greedy_parity` means: same prompt + same KV + same sampler → same
//! next-token. So if we feed a prompt, snapshot, then reset, restore,
//! and feed two more tokens, the predicted ids must equal a control
//! run that never reset.
//!
//! Build:
//!   cargo run --release --features cpu-reference --example kv_dump_load_roundtrip -- <gguf> [user_msg]

use std::env;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use rullama::api::{ChatMessage, ChatRole};
use rullama::backend::{Pipelines, WeightCache, WgpuCtx};
use rullama::gguf::GgufReader;
use rullama::model::config::Gemma4Config;
use rullama::reference::Weights;
use rullama::reference::forward_chained::Forward;
use rullama::sampling::{Sampler, SamplingOptions};
use rullama::template::gemma4_small;
use rullama::tokenizer::BpeTokenizer;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: kv_dump_load_roundtrip <gguf> [user_msg]");
            return ExitCode::from(2);
        }
    };
    let user_msg = args.next().unwrap_or_else(|| "Hello, world!".to_string());

    println!("loading model …");
    let t0 = Instant::now();
    let bytes = fs::read(&path).expect("read gguf");
    let reader = GgufReader::new(bytes).expect("parse gguf");
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

    // ───── Pass 1: feed prompt, snapshot, generate two control tokens ─────
    let mut last_sampled: u32 = 0;
    for &id in &prompt_ids {
        let logits = pollster::block_on(fwd.step(id)).expect("step prompt");
        last_sampled = sampler.sample(&logits);
        sampler.observe(id);
    }
    let pos_before_snapshot = fwd.pos();
    println!("position after prompt: {pos_before_snapshot}");

    let snap = pollster::block_on(fwd.dump_kv()).expect("dump_kv");
    let sampler_snap = sampler.dump_state();
    println!(
        "snapshot sizes: kv={} bytes, sampler={} bytes",
        snap.len(),
        sampler_snap.len(),
    );

    // Control: generate 3 tokens forward.
    let control_a = last_sampled;
    let logits = pollster::block_on(fwd.step(last_sampled)).expect("control step a");
    let control_b = sampler.sample(&logits);
    sampler.observe(last_sampled);
    let logits = pollster::block_on(fwd.step(control_b)).expect("control step b");
    let control_c = sampler.sample(&logits);
    sampler.observe(control_b);
    let pos_after_control = fwd.pos();
    println!("control trajectory: {control_a} -> {control_b} -> {control_c}");
    println!("position after control: {pos_after_control}");

    // ───── Pass 2: reset, restore from snapshot, replay generation ─────
    fwd.reset();
    sampler.clear_history();
    assert_eq!(fwd.pos(), 0, "reset should zero pos");

    fwd.load_kv(&snap).expect("load_kv");
    sampler.load_state(&sampler_snap).expect("load_state");
    assert_eq!(
        fwd.pos(),
        pos_before_snapshot,
        "load_kv must restore position to the snapshot point",
    );

    let restored_a = last_sampled; // same as control — last_sampled is what we'd feed next
    let logits = pollster::block_on(fwd.step(restored_a)).expect("restored step a");
    let restored_b = sampler.sample(&logits);
    sampler.observe(restored_a);
    let logits = pollster::block_on(fwd.step(restored_b)).expect("restored step b");
    let restored_c = sampler.sample(&logits);
    sampler.observe(restored_b);

    println!("restored trajectory: {restored_a} -> {restored_b} -> {restored_c}");

    let ok = control_a == restored_a
        && control_b == restored_b
        && control_c == restored_c
        && pos_after_control == fwd.pos();

    if ok {
        println!("\n✅ KV roundtrip OK — restored generation bit-matches control.");
        ExitCode::SUCCESS
    } else {
        eprintln!("\n❌ KV roundtrip DIVERGED:");
        eprintln!(
            "   control:  {control_a} -> {control_b} -> {control_c} (pos {pos_after_control})"
        );
        eprintln!(
            "   restored: {restored_a} -> {restored_b} -> {restored_c} (pos {})",
            fwd.pos()
        );
        ExitCode::from(1)
    }
}
