pub mod ambiguity_effectiveness;
pub mod brainwires_md;
pub mod checkpoint;
pub mod completion_detector;
pub mod context_builder;
pub mod conversation;
pub mod cost_tracker;
pub mod debug;
pub mod embeddings;
pub mod entity_extraction;
pub mod importance;
pub mod logger;
pub mod memory;
pub mod paths;

/// Test-only helpers. Keep truly shared state here so tests in different
/// modules can coordinate access to process-global resources (env vars,
/// CWD, etc.) without each rolling their own mutex.
#[cfg(test)]
pub mod test_util {
    use std::sync::Mutex;
    /// Serialise tests that mutate process-global env vars (`HOME`,
    /// `BRAINWIRES_MEMORY_ROOT`). Env vars leak across test boundaries
    /// and tokio's default test executor runs tests concurrently, so
    /// every such test must hold this mutex for the duration of the
    /// mutation.
    pub static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard: sets an env var on creation and restores the previous
    /// value on drop. Always acquire [`ENV_LOCK`] before creating one so
    /// two tests don't race on the same var.
    pub struct EnvVarGuard {
        key: String,
        prev: Option<String>,
    }

    impl EnvVarGuard {
        pub fn set<K: AsRef<str>, V: AsRef<std::ffi::OsStr>>(key: K, value: V) -> Self {
            let key = key.as_ref().to_string();
            let prev = std::env::var(&key).ok();
            // Safety: callers hold ENV_LOCK.
            unsafe {
                std::env::set_var(&key, value);
            }
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // Safety: Drop runs while the caller still holds ENV_LOCK.
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var(&self.key, v),
                    None => std::env::remove_var(&self.key),
                }
            }
        }
    }
}
/// Plan parser re-exported from the framework's reasoning crate.
/// (Moved out of `brainwires-core` during the 0.10 architecture restoration.)
pub mod plan_parser {
    pub use brainwires::reasoning::plan_parser::*;
}
pub mod prompt_cache;
pub mod prompt_history;
pub mod question_instructions;
pub mod recovery;
pub mod retrieval_gate;
pub mod rich_output;
pub mod secret_redaction;
pub mod skills;
pub mod system_prompt;
pub mod tokenizer;
