//! Inspect an Ollama image-generation model: dump its manifest census,
//! per-component tensor counts/sizes, sampled dtypes, and the JSON config
//! blobs (which carry the real architecture dims we need for IM1/IM2).
//!
//! Usage:
//!   cargo run -p brainwires-engine --example imagegen_inspect -- z-image
//!   cargo run -p brainwires-engine --example imagegen_inspect -- z-image:latest
//!
//! Works against any locally-present Ollama image model (created via
//! `ollama create z-image` from the HF weights). Run it the moment the model
//! exists to discover the exact encoder/DiT/VAE config before building IM1+.

use std::collections::BTreeMap;

use brainwires_engine::imagegen::{BlobSource, FileBlobSource, ImageManifest, find_manifest, read_header};

fn fmt_bytes(n: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.2} {}", U[i])
}

fn main() {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "z-image".to_string());

    let manifest_path = match find_manifest(&name) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!(
                "\nNo local image model {name:?}. Image models aren't `ollama pull`-able;\n\
                 download the HF weights then `ollama create z-image` (see\n\
                 ~/Source/ollama/x/imagegen/README.md)."
            );
            std::process::exit(1);
        }
    };
    println!("manifest: {}", manifest_path.display());

    let bytes = std::fs::read(&manifest_path).expect("read manifest");
    let manifest = ImageManifest::parse(&bytes).expect("parse manifest");

    let src = FileBlobSource::ollama_default().expect("ollama blobs dir");

    // ---- Manifest-level census (no blob reads) ----
    let mut comp_count: BTreeMap<String, usize> = BTreeMap::new();
    let mut comp_bytes: BTreeMap<String, u64> = BTreeMap::new();
    let mut total_bytes = 0u64;
    for b in &manifest.layers {
        total_bytes += b.size;
        if b.is_tensor() {
            let comp = b.name.split('/').next().unwrap_or("?").to_string();
            *comp_count.entry(comp.clone()).or_default() += 1;
            *comp_bytes.entry(comp).or_default() += b.size;
        }
    }

    println!(
        "\nlayers: {}  ({} tensors, {} json)  total {}",
        manifest.layers.len(),
        manifest.tensor_count(),
        manifest.layers.iter().filter(|b| b.is_json()).count(),
        fmt_bytes(total_bytes),
    );

    println!("\ncomponents:");
    for (comp, n) in &comp_count {
        let bytes = comp_bytes.get(comp).copied().unwrap_or(0);
        // Sample the dtype of the first tensor in this component.
        let sample = manifest
            .component(comp)
            .next()
            .map(|(_, b)| b)
            .and_then(|b| pollster::block_on(src.read_prefix(&b.blob_filename(), 1 << 16)).ok())
            .and_then(|prefix| read_header(&prefix).ok());
        let dtype_note = match sample {
            Some(h) => {
                let dt: Vec<String> = h
                    .tensors
                    .values()
                    .map(|e| format!("{:?}", e.dtype))
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let quant = h
                    .metadata
                    .get("quant_type")
                    .map(|q| format!(", quant={q}"))
                    .unwrap_or_default();
                format!("sample dtype={}{quant}", dt.join("/"))
            }
            None => "sample dtype=?".to_string(),
        };
        println!(
            "  {comp:<14} {n:>5} tensors  {:>12}   {dtype_note}",
            fmt_bytes(bytes)
        );
    }

    // ---- JSON config blobs: print them (the real dims live here) ----
    println!("\njson config blobs:");
    for b in manifest.layers.iter().filter(|b| b.is_json()) {
        let label = if b.name.is_empty() {
            "(config)"
        } else {
            b.name.as_str()
        };
        println!("  ── {label} ({}) ──", fmt_bytes(b.size));
        match pollster::block_on(src.read_blob(&b.blob_filename())) {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                // Pretty-print if it parses as JSON, else raw.
                match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(v) => println!(
                        "{}",
                        serde_json::to_string_pretty(&v).unwrap_or(text.into())
                    ),
                    Err(_) => println!("{text}"),
                }
            }
            Err(e) => println!("    <unreadable: {e}>"),
        }
    }
}
