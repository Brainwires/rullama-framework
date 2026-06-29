//! EmbeddingGemma CPU-oracle parity check.
//!
//! Loads an EmbeddingGemma GGUF, embeds a test string, and prints the
//! resulting vector + L2 norm so it can be compared against the parity
//! target (`llama.cpp` / Ollama `/api/embed` on the same GGUF).
//!
//! Usage:
//!   cargo run -p brainwires-engine --release --example embed_parity -- \
//!       ~/.ollama/models/blobs/sha256-<digest> "hello world"
//!
//! Ground-truth "hello world" embedding from Ollama (embeddinggemma):
//!   dim 768, L2 1.0, first8 =
//!   [-0.21395, 0.02636, 0.06661, -0.01639, 0.00745, 0.01082, -0.01431, -0.00245]

use std::env;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;

use brainwires_engine::gguf::GgufReader;
use brainwires_engine::reference::embed::EmbedModel;
use brainwires_engine::tokenizer::SpmTokenizer;

// Reference vector for "hello world" (Ollama embeddinggemma). Used to print a
// cosine when the test string is exactly "hello world".
const REF_HELLO_WORLD: [f32; 8] = [
    -0.21395, 0.02636, 0.06661, -0.01639, 0.00745, 0.01082, -0.01431, -0.00245,
];

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: embed_parity <gguf-path> [text]");
            return ExitCode::from(2);
        }
    };
    let text = env::args()
        .nth(2)
        .unwrap_or_else(|| "hello world".to_string());

    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {path}: {e}");
            return ExitCode::from(1);
        }
    };
    let reader = match GgufReader::new(bytes) {
        Ok(r) => Arc::new(r),
        Err(e) => {
            eprintln!("parse gguf: {e}");
            return ExitCode::from(1);
        }
    };

    let tok = match SpmTokenizer::from_gguf(&reader) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer: {e}");
            return ExitCode::from(1);
        }
    };
    let model = match EmbedModel::new(reader) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("model: {e}");
            return ExitCode::from(1);
        }
    };
    eprintln!(
        "[cfg] layers={} d_model={} heads={}/{} head_dim={} ffn={} ctx={} pool={:?} causal={}",
        model.cfg.n_layers,
        model.cfg.d_model,
        model.cfg.n_heads,
        model.cfg.n_kv_heads,
        model.cfg.head_dim,
        model.cfg.ffn,
        model.cfg.context_length,
        model.cfg.pooling,
        model.cfg.causal,
    );

    // BOS + text + EOS (add_bos_token / add_eos_token are both true).
    const BOS: u32 = 2;
    const EOS: u32 = 1;
    let mut ids = vec![BOS];
    ids.extend(tok.encode(&text));
    ids.push(EOS);
    eprintln!("[tok] {} ids: {:?}", ids.len(), &ids[..ids.len().min(16)]);

    let v = match model.embed_ids(&ids, 0) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("embed: {e}");
            return ExitCode::from(1);
        }
    };

    // GPU-vs-CPU parity (the load-bearing check). RULLAMA_EMBED_GPU=1.
    if env::var("RULLAMA_EMBED_GPU").is_ok() {
        let gpu = pollster::block_on(async {
            use std::sync::Arc;
            let ctx = brainwires_engine::backend::WgpuCtx::new().await?;
            let pipes = brainwires_engine::backend::Pipelines::new(&ctx.device);
            let wcache = brainwires_engine::backend::WeightCache::new(
                model.weights.reader_arc(),
                ctx.device.clone(),
                ctx.queue.clone(),
                Arc::clone(&ctx.bind_cache),
            );
            model.embed_ids_gpu(&ctx, &pipes, &wcache, &ids, 0).await
        });
        match gpu {
            Ok(g) => {
                let dot: f32 = v.iter().zip(g.iter()).map(|(a, b)| a * b).sum();
                let na: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos = dot / (na * nb + 1e-9);
                let maxabs = v
                    .iter()
                    .zip(g.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                println!("GPU-vs-CPU: cosine={cos:.6}  max_abs_diff={maxabs:.6}");
            }
            Err(e) => {
                eprintln!("GPU embed: {e}");
                return ExitCode::from(1);
            }
        }
    }

    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("dim:    {}", v.len());
    println!("L2:     {norm:.6}");
    println!(
        "first8: [{}]",
        v[..8]
            .iter()
            .map(|x| format!("{x:.5}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Full-vector dump for offline cosine comparison when RULLAMA_EMBED_DUMP is set.
    if env::var("RULLAMA_EMBED_DUMP").is_ok() {
        let line = v
            .iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("DUMP {line}");
    }

    if text == "hello world" {
        // cosine over the first 8 dims only (rough sanity); full-dim parity is
        // verified by the JS smoke test, but the leading dims catch gross bugs.
        let dot: f32 = v[..8]
            .iter()
            .zip(REF_HELLO_WORLD.iter())
            .map(|(a, b)| a * b)
            .sum();
        let na: f32 = v[..8].iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = REF_HELLO_WORLD.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb + 1e-9);
        println!(
            "ref8:   [{}]",
            REF_HELLO_WORLD.map(|x| format!("{x:.5}")).join(", ")
        );
        println!("cos(first8 vs ref): {cos:.4}");
    }

    ExitCode::SUCCESS
}
