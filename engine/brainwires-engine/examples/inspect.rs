//! Native-only GGUF inspector.
//!
//! Usage:
//!   cargo run --example inspect -- <path-to-model.gguf>
//!
//! Reads the file with mmap-via-`std::fs` (full read into a Vec on this side, since the
//! browser path won't have mmap either — same shape of API). Dumps GGUF version,
//! metadata key count, all `gemma4.*` and `tokenizer.ggml.*` keys with their decoded
//! values, and a tensor summary (count + first/last few names).

use std::env;
use std::fs;
use std::process::ExitCode;

use rullama::gguf::{GgufReader, GgufValue};
use rullama::model::config::Gemma4Config;

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: inspect <path-to-model.gguf>");
            return ExitCode::from(2);
        }
    };

    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read {path}: {e}");
            return ExitCode::from(1);
        }
    };
    let n_bytes = bytes.len();

    let r = match GgufReader::new(bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to parse {path}: {e}");
            return ExitCode::from(1);
        }
    };

    println!("file:        {path}");
    println!(
        "size:        {} bytes ({:.2} GB)",
        n_bytes,
        n_bytes as f64 / 1e9
    );
    println!("gguf v{}", r.version());
    println!("alignment:   {}", r.alignment());
    println!("metadata kv: {}", r.metadata().len());
    println!("tensors:     {}", r.tensors().len());
    println!();

    println!("== general.* ==");
    print_keys(&r, "general.");
    println!();

    println!("== gemma4.* ==");
    print_keys(&r, "gemma4.");
    println!();

    println!("== tokenizer.ggml.* (sizes / scalars only — vocab arrays are summarized) ==");
    print_keys(&r, "tokenizer.");
    println!();

    println!("== first 10 tensors ==");
    for t in r.tensors().iter().take(10) {
        println!(
            "  {:>7} {:?} {:?}",
            format!("{:?}", t.dtype),
            t.dims,
            t.name
        );
    }
    println!("== last 5 tensors ==");
    let n = r.tensors().len();
    for t in r.tensors().iter().skip(n.saturating_sub(5)) {
        println!(
            "  {:>7} {:?} {:?}",
            format!("{:?}", t.dtype),
            t.dims,
            t.name
        );
    }
    println!();

    println!("== parsed Gemma4Config ==");
    match Gemma4Config::from_gguf(&r) {
        Ok(c) => {
            println!("  n_layers              = {}", c.n_layers);
            println!("  d_model               = {}", c.d_model);
            println!("  max_pos               = {}", c.max_pos);
            println!("  n_heads               = {}", c.n_heads);
            println!("  n_kv_heads_swa        = {}", c.n_kv_heads_swa);
            println!("  n_kv_heads_global     = {}", c.n_kv_heads_global);
            println!("  head_dim_global       = {}", c.head_dim_global);
            println!("  head_dim_swa          = {}", c.head_dim_swa);
            println!("  rms_norm_eps          = {}", c.rms_norm_eps);
            println!("  sliding_window        = {}", c.sliding_window);
            println!("  shared_kv_layers      = {}", c.shared_kv_layers);
            println!("  rope_freq_base        = {}", c.rope_freq_base);
            println!("  rope_freq_base_swa    = {}", c.rope_freq_base_swa);
            println!("  rope_dim_global       = {}", c.rope_dim_global);
            println!("  rope_dim_swa          = {}", c.rope_dim_swa);
            println!("  final_logit_softcap   = {}", c.final_logit_softcap);
            println!(
                "  ple_dim               = {} (PLE {})",
                c.ple_dim,
                if c.has_ple() { "ENABLED" } else { "disabled" }
            );
            println!("  vocab_size            = {}", c.vocab_size);
            println!(
                "  bos_id={:?} eos_ids={:?} pad_id={:?} unk_id={:?}",
                c.bos_id, c.eos_ids, c.pad_id, c.unk_id
            );
            // layer-kind histogram
            let swa = c
                .layer_kinds
                .iter()
                .filter(|k| matches!(k, rullama::model::config::LayerKind::SlidingWindow))
                .count();
            let glb = c.layer_kinds.len() - swa;
            println!("  layers: {} SWA + {} global", swa, glb);
            // print first 12 layer kinds + ffn sizes
            print!("  per-layer: ");
            for i in 0..c.n_layers.min(12) {
                let k = match c.kind(i) {
                    rullama::model::config::LayerKind::SlidingWindow => "S",
                    rullama::model::config::LayerKind::Global => "G",
                };
                print!("{}{} ", k, c.ffn(i));
            }
            if c.n_layers > 12 {
                print!("… ");
            }
            println!();
        }
        Err(e) => println!("  ERROR: {e}"),
    }

    ExitCode::SUCCESS
}

fn print_keys(r: &GgufReader, prefix: &str) {
    let mut keys: Vec<_> = r
        .metadata()
        .keys()
        .filter(|k| k.starts_with(prefix))
        .collect();
    keys.sort();
    for k in keys {
        let v = &r.metadata()[k];
        println!("  {k} = {}", summarize(v));
    }
}

fn summarize(v: &GgufValue) -> String {
    match v {
        GgufValue::U8(x) => format!("u8 {x}"),
        GgufValue::I8(x) => format!("i8 {x}"),
        GgufValue::U16(x) => format!("u16 {x}"),
        GgufValue::I16(x) => format!("i16 {x}"),
        GgufValue::U32(x) => format!("u32 {x}"),
        GgufValue::I32(x) => format!("i32 {x}"),
        GgufValue::U64(x) => format!("u64 {x}"),
        GgufValue::I64(x) => format!("i64 {x}"),
        GgufValue::F32(x) => format!("f32 {x}"),
        GgufValue::F64(x) => format!("f64 {x}"),
        GgufValue::Bool(x) => format!("bool {x}"),
        GgufValue::String(s) => format!("str {:?}", truncate(s, 80)),
        GgufValue::ArrayU8(v) => format!("[u8; {}]", v.len()),
        GgufValue::ArrayI8(v) => format!("[i8; {}]", v.len()),
        GgufValue::ArrayU16(v) => format!("[u16; {}]", v.len()),
        GgufValue::ArrayI16(v) => format!("[i16; {}]", v.len()),
        GgufValue::ArrayU32(v) => array_preview(v, |x| format!("{x}")),
        GgufValue::ArrayI32(v) => array_preview(v, |x| format!("{x}")),
        GgufValue::ArrayU64(v) => array_preview(v, |x| format!("{x}")),
        GgufValue::ArrayI64(v) => array_preview(v, |x| format!("{x}")),
        GgufValue::ArrayF32(v) => array_preview(v, |x| format!("{x}")),
        GgufValue::ArrayF64(v) => array_preview(v, |x| format!("{x}")),
        GgufValue::ArrayBool(v) => format!("[bool; {}] {:?}", v.len(), &v[..v.len().min(8)]),
        GgufValue::ArrayString(v) => format!(
            "[str; {}] e.g. {:?}",
            v.len(),
            v.iter()
                .take(4)
                .map(|s| truncate(s, 16))
                .collect::<Vec<_>>()
        ),
    }
}

fn array_preview<T: Copy>(v: &[T], f: impl Fn(T) -> String) -> String {
    let n = v.len();
    let take = n.min(8);
    let head: Vec<_> = v[..take].iter().map(|&x| f(x)).collect();
    if n > take {
        format!("[T; {n}] {} … (+{} more)", head.join(","), n - take)
    } else {
        format!("[T; {n}] {}", head.join(","))
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push('…');
        out
    }
}
