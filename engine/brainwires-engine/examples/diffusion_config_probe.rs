//! C0 probe: parse a DiffusionGemma GGUF's config (works on a header-only
//! Range-fetched slice — no full download needed).
//!
//!   cargo run -p rullama --release --example diffusion_config_probe -- <gguf-or-header>

use std::process::ExitCode;

use rullama::gguf::GgufReader;
use rullama::reference::diffusion::DiffusionConfig;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: diffusion_config_probe <gguf-or-header-slice>");
        return ExitCode::from(2);
    };
    let bytes = std::fs::read(&path).expect("read");
    let r = GgufReader::new(bytes).expect("gguf");
    let cfg = DiffusionConfig::from_gguf(&r).expect("diffusion config");
    println!(
        "diffusion-gemma: {} layers, d_model {}, experts {} top-{} ffn {}, dense ffn {:?}, \
         mask_token {:?}, vocab {}, max_pos {}",
        cfg.base.n_layers,
        cfg.base.d_model,
        cfg.base.expert_count,
        cfg.base.expert_used_count,
        cfg.base.expert_ffn,
        &cfg.base.ffn_inter[..1],
        cfg.mask_token_id,
        cfg.base.vocab_size,
        cfg.base.max_pos,
    );
    assert!(cfg.base.has_moe());
    println!("PASS");
    ExitCode::SUCCESS
}
