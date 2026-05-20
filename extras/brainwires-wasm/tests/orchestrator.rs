//! Integration tests for ExecutionLimits and WasmOrchestrator (native target).
//!
//! These tests require the `orchestrator` feature. They exercise the Rust-side
//! logic without needing a browser or JS runtime.
//!
//! Run with: `cargo test -p brainwires-wasm --features orchestrator`

#![cfg(all(feature = "orchestrator", target_arch = "wasm32"))]

use brainwires_wasm::wasm_orchestrator::ExecutionLimits;

// ── ExecutionLimits::new() defaults ─────────────────────────────────────

#[test]
fn execution_limits_default_max_operations() {
    let limits = ExecutionLimits::new();
    assert_eq!(limits.max_operations(), 100_000);
}

#[test]
fn execution_limits_default_max_tool_calls() {
    let limits = ExecutionLimits::new();
    assert_eq!(limits.max_tool_calls(), 50);
}

#[test]
fn execution_limits_default_timeout_ms() {
    let limits = ExecutionLimits::new();
    assert_eq!(limits.timeout_ms(), 30_000);
}

#[test]
fn execution_limits_default_max_string_size() {
    let limits = ExecutionLimits::new();
    assert_eq!(limits.max_string_size(), 10_000_000);
}

#[test]
fn execution_limits_default_max_array_size() {
    let limits = ExecutionLimits::new();
    assert_eq!(limits.max_array_size(), 10_000);
}

// ── ExecutionLimits::quick() preset ─────────────────────────────────────

#[test]
fn execution_limits_quick_max_operations() {
    let limits = ExecutionLimits::quick();
    assert_eq!(limits.max_operations(), 10_000);
}

#[test]
fn execution_limits_quick_max_tool_calls() {
    let limits = ExecutionLimits::quick();
    assert_eq!(limits.max_tool_calls(), 10);
}

#[test]
fn execution_limits_quick_timeout_ms() {
    let limits = ExecutionLimits::quick();
    assert_eq!(limits.timeout_ms(), 5_000);
}

#[test]
fn execution_limits_quick_max_string_size() {
    let limits = ExecutionLimits::quick();
    assert_eq!(limits.max_string_size(), 10_000_000);
}

#[test]
fn execution_limits_quick_max_array_size() {
    let limits = ExecutionLimits::quick();
    assert_eq!(limits.max_array_size(), 10_000);
}

// ── ExecutionLimits::extended() preset ──────────────────────────────────

#[test]
fn execution_limits_extended_max_operations() {
    let limits = ExecutionLimits::extended();
    assert_eq!(limits.max_operations(), 500_000);
}

#[test]
fn execution_limits_extended_max_tool_calls() {
    let limits = ExecutionLimits::extended();
    assert_eq!(limits.max_tool_calls(), 100);
}

#[test]
fn execution_limits_extended_timeout_ms() {
    let limits = ExecutionLimits::extended();
    assert_eq!(limits.timeout_ms(), 120_000);
}

#[test]
fn execution_limits_extended_max_string_size() {
    let limits = ExecutionLimits::extended();
    assert_eq!(limits.max_string_size(), 10_000_000);
}

#[test]
fn execution_limits_extended_max_array_size() {
    let limits = ExecutionLimits::extended();
    assert_eq!(limits.max_array_size(), 10_000);
}

// ── ExecutionLimits getters/setters ─────────────────────────────────────

#[test]
fn execution_limits_set_max_operations() {
    let mut limits = ExecutionLimits::new();
    limits.set_max_operations(42);
    assert_eq!(limits.max_operations(), 42);
}

#[test]
fn execution_limits_set_max_tool_calls() {
    let mut limits = ExecutionLimits::new();
    limits.set_max_tool_calls(7);
    assert_eq!(limits.max_tool_calls(), 7);
}

#[test]
fn execution_limits_set_timeout_ms() {
    let mut limits = ExecutionLimits::new();
    limits.set_timeout_ms(999);
    assert_eq!(limits.timeout_ms(), 999);
}

#[test]
fn execution_limits_set_max_string_size() {
    let mut limits = ExecutionLimits::new();
    limits.set_max_string_size(1_234);
    assert_eq!(limits.max_string_size(), 1_234);
}

#[test]
fn execution_limits_set_max_array_size() {
    let mut limits = ExecutionLimits::new();
    limits.set_max_array_size(555);
    assert_eq!(limits.max_array_size(), 555);
}

#[test]
fn execution_limits_set_zero_values() {
    let mut limits = ExecutionLimits::new();
    limits.set_max_operations(0);
    limits.set_max_tool_calls(0);
    limits.set_timeout_ms(0);
    assert_eq!(limits.max_operations(), 0);
    assert_eq!(limits.max_tool_calls(), 0);
    assert_eq!(limits.timeout_ms(), 0);
}

#[test]
fn execution_limits_set_large_values() {
    let mut limits = ExecutionLimits::new();
    limits.set_max_operations(u64::MAX);
    limits.set_max_tool_calls(usize::MAX);
    assert_eq!(limits.max_operations(), u64::MAX);
    assert_eq!(limits.max_tool_calls(), usize::MAX);
}

#[test]
fn execution_limits_multiple_sets_last_wins() {
    let mut limits = ExecutionLimits::new();
    limits.set_max_operations(1);
    limits.set_max_operations(2);
    limits.set_max_operations(3);
    assert_eq!(limits.max_operations(), 3);
}

// ── ExecutionLimits Clone / Default ─────────────────────────────────────

#[test]
fn execution_limits_clone_is_independent() {
    let mut original = ExecutionLimits::new();
    let cloned = original.clone();
    original.set_max_operations(999);
    assert_eq!(
        cloned.max_operations(),
        100_000,
        "clone should not be affected"
    );
    assert_eq!(original.max_operations(), 999);
}

#[test]
fn execution_limits_default_matches_new() {
    let from_new = ExecutionLimits::new();
    let from_default = ExecutionLimits::default();
    assert_eq!(from_new.max_operations(), from_default.max_operations());
    assert_eq!(from_new.max_tool_calls(), from_default.max_tool_calls());
    assert_eq!(from_new.timeout_ms(), from_default.timeout_ms());
    assert_eq!(from_new.max_string_size(), from_default.max_string_size());
    assert_eq!(from_new.max_array_size(), from_default.max_array_size());
}
