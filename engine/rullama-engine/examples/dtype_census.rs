//! Header-only dtype census of a GGUF (streams the directory, never the data).
//!
//!   cargo run -p rullama-engine --release --example dtype_census -- <gguf>
//!
//! Prints, per logical tensor role, the dtype + a histogram so the GPU forward
//! knows which matmul paths it must support.

use std::collections::BTreeMap;
use std::process::ExitCode;
use std::sync::Arc;

use rullama_engine::gguf::{FileFetcher, GgufReader};

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: dtype_census <gguf>");
        return ExitCode::from(2);
    };
    let fetcher = FileFetcher::open(std::path::Path::new(&path)).expect("open");
    let r = pollster::block_on(GgufReader::new_streaming(Arc::new(fetcher))).expect("gguf");

    // Overall histogram + per-role (strip the `blk.N.` prefix) dtype set.
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    let mut roles: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    for t in r.tensors() {
        *hist.entry(format!("{:?}", t.dtype)).or_default() += 1;
        let role = {
            let n = &t.name;
            if let Some(rest) = n.strip_prefix("blk.") {
                // drop the leading layer number
                match rest.split_once('.') {
                    Some((_, tail)) => format!("blk.*.{tail}"),
                    None => n.clone(),
                }
            } else {
                n.clone()
            }
        };
        *roles
            .entry(role)
            .or_default()
            .entry(format!("{:?}", t.dtype))
            .or_default() += 1;
    }

    println!("== overall dtype histogram ==");
    for (d, c) in &hist {
        println!("  {d:>6}  {c}");
    }
    println!("\n== per-role dtypes (matmul-relevant) ==");
    for (role, dts) in &roles {
        if role.contains("weight")
            && (role.contains("attn")
                || role.contains("ffn")
                || role.contains("output")
                || role.contains("token_embd")
                || role.contains("self_cond")
                || role.contains("exps"))
        {
            let s: Vec<String> = dts.iter().map(|(d, c)| format!("{d}×{c}")).collect();
            println!("  {role:<40} {}", s.join(", "));
        }
    }
    ExitCode::SUCCESS
}
