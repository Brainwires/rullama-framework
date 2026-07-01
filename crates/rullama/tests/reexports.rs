//! Compile-time smoke test for the `rullama` metacrate's re-export surface.
//!
//! The metacrate exists to give downstream users a single place to pull in
//! every framework subsystem via feature flags. If a refactor renames or
//! drops a sub-crate's public type, this test will stop compiling — long
//! before a downstream consumer notices.
//!
//! Everything here is typecheck-only; we never call anything, we just
//! assert that the paths still resolve.

use rullama::prelude::*;

// Core surface (always on, no features needed).
const _: fn() = || {
    let _t: Task = Task::new_for_plan(
        "t-1".to_string(),
        "hello".to_string(),
        "plan-123456789".to_string(),
    );
    let _msg: Message = Message {
        role: Role::User,
        content: MessageContent::Text("hi".into()),
        name: None,
        metadata: None,
    };
    let _: PermissionMode = PermissionMode::default();
    let _: TaskPriority = TaskPriority::Normal;
};

// Feature-gated surfaces. Each mirrors a sub-crate that should remain
// reachable via `rullama::<name>::*`.

#[cfg(feature = "tools")]
const _: fn() = || {
    use rullama::tools::ToolRegistry;
    let _ = ToolRegistry::new();
};

#[cfg(feature = "agents")]
const _: fn() = || {
    // Touch a non-constructor name from the agents module so the path
    // resolves through the metacrate.
    use rullama::agents::TaskQueue;
    fn _assert_ty() -> std::marker::PhantomData<TaskQueue> {
        std::marker::PhantomData
    }
};

#[cfg(feature = "permissions")]
const _: fn() = || {
    use rullama::permissions::PolicyEngine;
    let _ = PolicyEngine::new();
};

#[cfg(feature = "reasoning")]
const _: fn() = || {
    // rullama::reasoning re-exports directly from rullama-reasoning.
    use rullama::reasoning::plan_parser::parse_plan_steps;
    let _: fn(&str) -> Vec<_> = parse_plan_steps;
};

#[cfg(feature = "tiered")]
const _: fn() = || {
    use rullama::memory::TieredMemory;
    let _ty: std::marker::PhantomData<TieredMemory> = std::marker::PhantomData;
    let _ = _ty;
};

#[cfg(feature = "mcp")]
const _: fn() = || {
    use rullama::mcp::McpServerConfig;
    let _ty: std::marker::PhantomData<McpServerConfig> = std::marker::PhantomData;
    let _ = _ty;
};

// Runtime smoke — if any of the const-blocks above stopped compiling, the
// whole crate would fail to build. This test function just makes the
// harness pick up the file.
#[test]
fn metacrate_reexports_resolve() {
    // If we got here, every re-export above compiled.
}
