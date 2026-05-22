//! Tier-A feature cases for `brainwires-permission` capability profiles.

use anyhow::Result;
use async_trait::async_trait;
use brainwires_eval::{EvaluationCase, TrialResult};
use brainwires_permission::FilesystemCapabilities;

use crate::registry::TierACase;

inventory::submit! {
    TierACase {
        path: "brainwires_test_harness::cases::permission_capabilities::default_and_full_have_expected_shape",
        crate_name: "brainwires-permission",
        description: "FilesystemCapabilities::default() is safe-by-default; ::full() opens everything",
        factory: || Box::new(CapShapeCase),
    }
}

struct CapShapeCase;

#[async_trait]
impl EvaluationCase for CapShapeCase {
    fn name(&self) -> &str {
        "feature.permissions.capability_shapes"
    }
    fn category(&self) -> &str {
        "feature.permissions"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let d = FilesystemCapabilities::default();
        // Default shape: globally readable, no writes, with explicit denies.
        if d.read_paths.is_empty() {
            return Ok(TrialResult::failure(0, 0, "default read_paths empty"));
        }
        if !d.write_paths.is_empty() {
            return Ok(TrialResult::failure(0, 0, "default write_paths non-empty"));
        }
        if d.can_delete {
            return Ok(TrialResult::failure(0, 0, "default can_delete is true"));
        }
        if d.denied_paths.is_empty() {
            return Ok(TrialResult::failure(0, 0, "default denied_paths empty"));
        }
        // Full shape: no denies, delete allowed.
        let f = FilesystemCapabilities::full();
        if !f.denied_paths.is_empty() {
            return Ok(TrialResult::failure(
                0,
                0,
                "full() has unexpected denied_paths",
            ));
        }
        if !f.can_delete {
            return Ok(TrialResult::failure(0, 0, "full() must allow delete"));
        }
        if f.write_paths.is_empty() {
            return Ok(TrialResult::failure(0, 0, "full() write_paths empty"));
        }
        Ok(TrialResult::success(0, 0))
    }
}
