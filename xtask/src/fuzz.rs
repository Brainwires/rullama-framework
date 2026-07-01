//! `cargo fuzz` — friendly front-end over `cargo-fuzz`.
//!
//! Aliased as `cargo fuzz` in `.cargo/config.toml`. This DELIBERATELY shadows
//! the raw `cargo-fuzz` external subcommand: the wrapper makes sure the nightly
//! toolchain + `cargo-fuzz` are present, runs from the workspace root (where the
//! `fuzz/` crate lives), then forwards every argument to the real tool.
//!
//! Why we invoke the `cargo-fuzz` BINARY directly (and never `cargo fuzz`):
//! the alias would otherwise resolve `cargo fuzz` straight back to this xtask
//! and recurse forever. Cargo dispatches `cargo fuzz <args>` to the external
//! binary as `cargo-fuzz fuzz <args>`, so we replicate that argv exactly.
//!
//! Nightly: `cargo-fuzz` uses libFuzzer instrumentation + sanitizers, which are
//! nightly-only. The repo pins stable (rust-toolchain.toml), so we select
//! nightly via `rustup run nightly` (its `RUSTUP_TOOLCHAIN` override outranks the
//! toolchain file) — equivalent to the documented `cargo +nightly fuzz`.

use std::path::PathBuf;
use std::process::{Command, ExitCode};

/// Repo root = the xtask crate's parent dir (`<root>/xtask` → `<root>`).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask crate always has a parent dir")
        .to_path_buf()
}

/// Is `bin` resolvable on PATH?
fn on_path(bin: &str) -> bool {
    Command::new(bin)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Is the named rustup toolchain installed?
fn toolchain_installed(name: &str) -> bool {
    Command::new("rustup")
        .args(["toolchain", "list"])
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .any(|l| l.starts_with(name))
        })
        .unwrap_or(false)
}

pub fn dispatch(args: &[String]) -> ExitCode {
    // Preflight: nightly toolchain.
    if !on_path("rustup") {
        eprintln!("cargo fuzz: rustup not found. cargo-fuzz needs a nightly toolchain.");
        eprintln!("  install rustup:  https://rustup.rs");
        return ExitCode::from(1);
    }
    if !toolchain_installed("nightly") {
        eprintln!(
            "cargo fuzz: the nightly toolchain is required (cargo-fuzz uses libFuzzer + sanitizers)."
        );
        eprintln!("  install it:  rustup toolchain install nightly");
        return ExitCode::from(1);
    }
    // Preflight: cargo-fuzz itself.
    if !on_path("cargo-fuzz") {
        eprintln!("cargo fuzz: cargo-fuzz is not installed.");
        eprintln!("  install it:  cargo install cargo-fuzz");
        return ExitCode::from(1);
    }

    // No subcommand → list targets + a usage hint (don't silently do nothing).
    let forwarded: Vec<String> = if args.is_empty() {
        eprintln!(
            "cargo fuzz: no subcommand given — listing targets.\n  \
             run one with:  cargo fuzz run <target> -- -max_total_time=60"
        );
        vec!["list".to_string()]
    } else {
        args.to_vec()
    };

    // `rustup run nightly cargo-fuzz fuzz <forwarded>` — binary-direct (no alias
    // recursion), nightly-selected, from the workspace root so `./fuzz/` resolves.
    let status = Command::new("rustup")
        .args(["run", "nightly", "cargo-fuzz", "fuzz"])
        .args(&forwarded)
        .current_dir(repo_root())
        .status();

    match status {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("cargo fuzz: failed to spawn cargo-fuzz: {e}");
            ExitCode::from(1)
        }
    }
}
