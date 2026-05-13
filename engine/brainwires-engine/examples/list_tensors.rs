//! Dump every tensor descriptor in a GGUF: dtype, shape, name. No filtering.
//!
//! Optional second arg is a substring filter (e.g. `a.blk.0` to see only one
//! audio block). Optional third arg `--summary` prints a per-prefix count
//! instead of the full list.
//!
//! Build:
//!   cargo run --release --example list_tensors -- <gguf>
//!   cargo run --release --example list_tensors -- <gguf> a.blk.0
//!   cargo run --release --example list_tensors -- <gguf> "" --summary

use std::collections::BTreeMap;
use std::env;
use std::fs;
use rullama::gguf::GgufReader;

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: list_tensors <gguf> [filter] [--summary]");
    let filter = args.next().unwrap_or_default();
    let summary = args.next().is_some_and(|a| a == "--summary");

    let bytes = fs::read(&path).unwrap();
    let r = GgufReader::new(bytes).unwrap();
    let mut names: Vec<_> = r.tensors().iter()
        .map(|t| (t.name.clone(), format!("{:?}", t.dtype), format!("{:?}", t.dims)))
        .collect();
    names.sort();

    if summary {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (n, _, _) in &names {
            if !filter.is_empty() && !n.contains(&filter) { continue; }
            let prefix = n.split('.').next().unwrap_or("").to_string();
            *counts.entry(prefix).or_insert(0) += 1;
        }
        let mut total = 0;
        for (k, v) in &counts {
            println!("{v:>6} {k}");
            total += v;
        }
        println!("------");
        println!("{total:>6} total");
        return;
    }

    let mut shown = 0;
    for (n, d, dims) in &names {
        if !filter.is_empty() && !n.contains(&filter) { continue; }
        println!("{:>7} {:<22} {}", d, dims, n);
        shown += 1;
    }
    eprintln!("({shown} of {} tensors shown)", names.len());
}
