//! Native end-to-end DiffusionGemma generation: load the model (streaming),
//! denoise a masked canvas from a text prompt via the entropy-bound sampler,
//! print the decoded text. The whole forward runs on the GPU.
//!
//!   cargo run -p rullama --release --example diffusion_generate -- \
//!       <model.gguf> "Your prompt" [--canvas=N] [--steps=N] [--seed=N]
//!
//! NB: each denoise step is a full canvas forward (~tens of seconds on a weak
//! GPU); use a small --canvas and --steps for a quick smoke run.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use rullama::diffusion::DiffusionGemma;
use rullama::gguf::{FileFetcher, TensorFetcher};
use rullama::reference::diffusion::sampler::{EbParams, StepInfo};

fn main() -> ExitCode {
    let mut positional = Vec::new();
    let mut canvas_len = 256usize;
    let mut steps = 48u32;
    let mut seed = 0xD1FFu64;
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("--canvas=") {
            canvas_len = v.parse().expect("canvas N");
        } else if let Some(v) = a.strip_prefix("--steps=") {
            steps = v.parse().expect("steps N");
        } else if let Some(v) = a.strip_prefix("--seed=") {
            seed = v.parse().expect("seed N");
        } else {
            positional.push(a);
        }
    }
    let (Some(model), prompt) = (positional.first(), positional.get(1).cloned()) else {
        eprintln!("usage: diffusion_generate <model.gguf> \"prompt\" [--canvas=N] [--steps=N]");
        return ExitCode::from(2);
    };
    let prompt = prompt.unwrap_or_else(|| "The capital of France is".to_string());

    println!("loading (streaming) ...");
    let t0 = Instant::now();
    let fetcher: Arc<dyn TensorFetcher> =
        Arc::new(FileFetcher::open(std::path::Path::new(model)).expect("open"));
    let dg = pollster::block_on(DiffusionGemma::load_streaming_native(fetcher)).expect("load");
    println!("  loaded in {:?}", t0.elapsed());
    println!("prompt: {prompt:?}  (canvas {canvas_len}, max {steps} steps)\n");

    let params = EbParams {
        max_denoising_steps: steps,
        ..Default::default()
    };
    let t0 = Instant::now();
    let mut step_t = Instant::now();
    let mut on_step = |info: &StepInfo| -> bool {
        println!(
            "  step {:>2}/{}: accepted {:>3}, mean_entropy {:.4}  ({:?})",
            info.step_idx + 1,
            info.total_steps,
            info.n_accepted,
            info.mean_entropy,
            step_t.elapsed(),
        );
        step_t = Instant::now();
        true
    };
    let text = dg
        .generate_native(&prompt, canvas_len, &params, seed, Some(&mut on_step))
        .expect("generate");
    println!("\ngenerated in {:?}\n", t0.elapsed());
    println!("=== canvas ===\n{text}");
    ExitCode::SUCCESS
}
