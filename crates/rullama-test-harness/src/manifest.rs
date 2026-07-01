//! Feature-inventory manifest loader.
//!
//! Manifest format mirrors the section structure of `FEATURES.md`. Every
//! heading in FEATURES.md is expected to have at least one `[[feature]]`
//! block referencing one or more `required_cases` (Rust function paths
//! resolvable via [`crate::registry`]).
//!
//! The `cargo xtask test-harness coverage` command (Step 4) walks
//! FEATURES.md, loads this manifest, and fails with a copy-pasteable stub
//! if any heading is missing.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Default seed for cases that don't specify one explicitly.
const DEFAULT_SEED: u64 = 0x00B7_A115_CA5E;
/// Default trial count for cases that don't specify one explicitly.
const DEFAULT_TRIALS: usize = 1;
/// Default Wilson-CI lower-bound gate. 1.0 = strict pass/fail.
const DEFAULT_WILSON_MIN_PASS: f64 = 1.0;

/// Parsed `feature_inventory.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    #[serde(default, rename = "feature")]
    pub entries: Vec<FeatureEntry>,
}

/// A single feature claim — one `[[feature]]` block in the manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct FeatureEntry {
    /// Mirrors the FEATURES.md heading (e.g. "MDAP Voting").
    pub section: String,
    /// The crate this feature lives in.
    #[serde(default)]
    pub crate_name: Option<String>,
    /// Stable dotted ID (e.g. "mdap.voting.k_of_n").
    pub feature_id: String,
    /// Human description.
    #[serde(default)]
    pub description: String,
    /// Rust function paths registered via [`crate::registry::TierACase`].
    #[serde(default)]
    pub required_cases: Vec<String>,
    /// Optional alias — defer coverage to another `feature_id`.
    #[serde(default)]
    pub coverage_via: Option<String>,
    /// Deterministic seed for stochastic cases. Defaults to `DEFAULT_SEED`.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Number of trials run by [`rullama_eval::EvaluationSuite`].
    #[serde(default = "default_trials")]
    pub trials: usize,
    /// Wilson-CI 95% lower-bound gate. 1.0 = strict pass/fail.
    #[serde(default = "default_wilson")]
    pub wilson_min_pass: f64,
}

fn default_seed() -> u64 {
    DEFAULT_SEED
}
fn default_trials() -> usize {
    DEFAULT_TRIALS
}
fn default_wilson() -> f64 {
    DEFAULT_WILSON_MIN_PASS
}

/// Load and parse the manifest from disk.
pub fn load(path: impl AsRef<Path>) -> Result<Manifest> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading manifest at {}", path.display()))?;
    parse(&raw)
}

/// Parse a manifest from an already-loaded TOML string.
pub fn parse(raw: &str) -> Result<Manifest> {
    let m: Manifest = toml::from_str(raw).context("parsing manifest TOML")?;
    if m.schema_version != 1 {
        anyhow::bail!(
            "unsupported manifest schema_version {} (expected 1)",
            m.schema_version
        );
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_manifest() {
        let raw = r#"
schema_version = 1

[[feature]]
section = "MDAP Voting"
crate_name = "rullama-mdap"
feature_id = "mdap.voting.k_of_n"
description = "k-out-of-n quorum"
required_cases = ["rullama_test_harness::cases::mdap::k_of_n_quorum"]
"#;
        let m = parse(raw).unwrap();
        assert_eq!(m.entries.len(), 1);
        let f = &m.entries[0];
        assert_eq!(f.feature_id, "mdap.voting.k_of_n");
        assert_eq!(f.seed, DEFAULT_SEED);
        assert_eq!(f.trials, DEFAULT_TRIALS);
        assert!((f.wilson_min_pass - DEFAULT_WILSON_MIN_PASS).abs() < 1e-9);
    }

    #[test]
    fn rejects_wrong_schema_version() {
        let raw = "schema_version = 99\n";
        assert!(parse(raw).is_err());
    }

    #[test]
    fn empty_features_array_ok() {
        let m = parse("schema_version = 1\n").unwrap();
        assert!(m.entries.is_empty());
    }
}
