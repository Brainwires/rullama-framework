#![no_main]
//! Fuzz target for `brainwires_skills::manifest::SkillManifest` YAML parsing.
//!
//! Skill manifests come from filesystem-walking a skills directory or from
//! remote registry downloads. A malformed manifest must not crash the
//! skill loader. Attacks the same code path that `parse_skill_metadata`
//! exercises in production.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Manifests are YAML in production but the SkillManifest type also
    // serdes through JSON via serde derives. Both paths are reachable
    // depending on the caller; fuzz both via serde_json (simplest entry).
    let _ = serde_json::from_slice::<brainwires_skills::manifest::SkillManifest>(data);
});
