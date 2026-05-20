//! Brainwires CLI - An AI-powered agentic CLI tool for autonomous coding assistance
//!
//! This crate provides a command-line interface for interacting with AI agents,
//! managing authentication with Brainwires Studio, executing tools, and
//! integrating with Model Context Protocol (MCP) servers.

// Increase recursion limit for complex async types
#![recursion_limit = "512"]

pub mod agent;
pub mod agents;
pub mod approval;
pub mod ask;
pub mod auth;
pub mod cli;
pub mod commands;
pub mod config;
pub mod dream;
pub mod error;
pub mod hooks;
pub mod ipc;
pub mod logging;
pub mod mcp;
pub mod mcp_server;
pub mod mdap;
pub mod persistent_task_manager;
pub mod plan_mode_store;
pub mod providers;
// RAG functionality is now provided by project-rag crate (git submodule)
// pub mod rag;
pub mod remote;
pub mod self_improve;
#[cfg(unix)]
pub mod session;
pub mod storage;
pub mod sudo;
pub mod system_prompts;
pub mod tools;
pub mod tui;
pub mod types;
pub mod utils;

/// Build-time constants
pub mod build_info {
    /// Package version from Cargo.toml
    pub const VERSION: &str = env!("CARGO_PKG_VERSION");

    /// Build timestamp
    pub const BUILD_TIMESTAMP: &str = env!("BUILD_TIMESTAMP");

    /// Git commit hash (if available)
    pub const GIT_HASH: &str = match option_env!("GIT_HASH") {
        Some(hash) => hash,
        None => "unknown",
    };

    /// Full version string including build date and git hash, e.g.
    /// "0.7.0 (built 2026-03-30 UTC • abc1234)"
    pub const FULL_VERSION: &str = env!("FULL_VERSION");
}
