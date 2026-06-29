//! Tier-B regression catcher for "empty-placeholder" Cargo features.
//!
//! Phase-2 of the harness build found two such "documented placeholder"
//! features whose gated code referenced things that didn't resolve:
//! - `rullama-knowledge::spectral` (referenced a `crate::spectral`
//!   module that didn't exist and the `ndarray` crate that wasn't
//!   declared)
//! - `rullama-inference::telemetry` (referenced `rullama_telemetry::*`
//!   types behind cfg gates without declaring the dep)
//!
//! Both were "fine because nobody enables them alone, and `cargo check
//! -p crate` always activates default features which hide the breakage".
//! The fix in each case was either to wire up the dependency properly
//! (inference) or delete the dead code (knowledge).
//!
//! This case sweeps the remaining empty features the framework declares
//! and spawns `cargo check -p <crate> --no-default-features --features
//! <feature>` per entry, asserting each compiles. If a future refactor
//! adds gated code that references a non-existent module/dep, this case
//! goes red.
//!
//! Skipped automatically if `cargo` isn't on PATH (e.g. inside a stripped
//! container).

use std::process::Command;

use anyhow::Result;
use async_trait::async_trait;
use rullama_eval::{EvaluationCase, TrialResult};

use crate::registry::SecurityCase;

/// `(crate, feature)` pairs that were declared as `feature = []` empty
/// placeholders. Each must compile when enabled standalone with no
/// other features.
const EMPTY_FEATURES: &[(&str, &str)] = &[
    ("rullama-core", "native"),
    ("rullama-core", "alt-folder-name"),
    ("rullama-prompting", "prompting"),
    ("rullama-stores", "session"),
    ("rullama-stores", "conversation"),
    ("rullama-network", "ipc-transport"),
    ("rullama-network", "remote-transport"),
    ("rullama-network", "tcp-transport"),
    ("rullama-network", "pubsub-transport"),
    ("rullama-network", "a2a-transport"),
    ("rullama-network", "email-identity"),
    ("rullama-call-policy", "native"),
    ("rullama-call-policy", "wasm"),
];

inventory::submit! {
    SecurityCase {
        id: "sec.cargo.empty_features_compile_standalone",
        crate_name: "(workspace meta)",
        invariant: "Every declared `feature = []` empty-placeholder compiles when enabled with --no-default-features alone (i.e. no gated code references a missing module/dep)",
        factory: || Box::new(EmptyFeaturesCompileCase),
    }
}

struct EmptyFeaturesCompileCase;

#[async_trait]
impl EvaluationCase for EmptyFeaturesCompileCase {
    fn name(&self) -> &str {
        "sec.cargo.empty_features_compile_standalone"
    }
    fn category(&self) -> &str {
        "security.workspace"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        // Cargo location heuristic — skip cleanly if not present.
        if Command::new("cargo")
            .arg("--version")
            .output()
            .ok()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            return Ok(TrialResult::success(0, 0));
        }

        let mut failures = Vec::new();
        for (crate_name, feature) in EMPTY_FEATURES {
            let out = Command::new("cargo")
                .args([
                    "check",
                    "-p",
                    crate_name,
                    "--no-default-features",
                    "--features",
                    feature,
                    "--quiet",
                ])
                .output()?;
            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                // Keep the report short — show the first 200 chars of
                // stderr so the case output stays scannable.
                let snippet = stderr
                    .lines()
                    .filter(|l| l.starts_with("error"))
                    .take(2)
                    .collect::<Vec<_>>()
                    .join(" | ");
                failures.push(format!(
                    "{crate_name}/{feature}: {}",
                    if snippet.is_empty() {
                        "compile failed (no error: prefix lines captured)".to_string()
                    } else {
                        snippet
                    }
                ));
            }
        }

        if !failures.is_empty() {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "{} empty-feature(s) broken when enabled standalone:\n  - {}",
                    failures.len(),
                    failures.join("\n  - ")
                ),
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
