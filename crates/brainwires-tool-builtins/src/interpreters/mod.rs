#![deny(missing_docs)]
//! # Code Interpreters
//!
//! Sandboxed code execution for multiple programming languages.
//! Designed to work both natively and compiled to WASM for browser execution.
//!
//! ## Supported Languages
//!
//! | Language | Feature | Speed | Power | Notes |
//! |----------|---------|-------|-------|-------|
//! | Rhai | `rhai` | ⚡⚡⚡⚡ | ⭐⭐ | Native Rust, fastest startup |
//! | Lua | `lua` | ⚡⚡⚡ | ⭐⭐⭐ | Small runtime, good stdlib |
//! | JavaScript | `javascript` | ⚡⚡ | ⭐⭐⭐⭐ | ECMAScript compliant (Boa) |
//!
//! ## Example
//!
//! ```rust,no_run
//! use brainwires_tool_builtins::interpreters::{Executor, ExecutionRequest, Language};
//!
//! let executor = Executor::new();
//! let result = executor.execute(ExecutionRequest {
//!     language: Language::Rhai,
//!     code: "let x = 1 + 2; x".to_string(),
//!     ..Default::default()
//! });
//!
//! assert!(result.success);
//! assert_eq!(result.stdout, "3");
//! ```

mod executor;
mod languages;
mod types;

#[cfg(feature = "interpreters-wasm")]
mod wasm_bindings;

pub use executor::Executor;
pub use types::*;

/// Re-exports of language-specific executors for advanced use.
pub mod lang {
    #[cfg(feature = "interpreters-rhai")]
    pub use super::languages::rhai::RhaiExecutor;

    #[cfg(feature = "interpreters-lua")]
    pub use super::languages::lua::LuaExecutor;

    #[cfg(feature = "interpreters-js")]
    pub use super::languages::javascript::JavaScriptExecutor;
}

/// Get a list of supported languages based on enabled features
#[allow(clippy::vec_init_then_push)]
pub fn supported_languages() -> Vec<Language> {
    let mut languages = Vec::new();

    #[cfg(feature = "interpreters-rhai")]
    languages.push(Language::Rhai);

    #[cfg(feature = "interpreters-lua")]
    languages.push(Language::Lua);

    #[cfg(feature = "interpreters-js")]
    languages.push(Language::JavaScript);

    languages
}

/// Check if a language is supported
pub fn is_language_supported(language: Language) -> bool {
    match language {
        #[cfg(feature = "interpreters-rhai")]
        Language::Rhai => true,

        #[cfg(feature = "interpreters-lua")]
        Language::Lua => true,

        #[cfg(feature = "interpreters-js")]
        Language::JavaScript => true,

        #[allow(unreachable_patterns)]
        _ => false,
    }
}
