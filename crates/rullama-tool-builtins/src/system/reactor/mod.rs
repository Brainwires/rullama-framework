//! File system event reactor — watch directories and trigger autonomous actions.
//!
//! Uses the `notify` crate for cross-platform file system watching with
//! debouncing and rate limiting to prevent event storms.

pub mod debounce;
pub mod fs_watcher;
pub mod rules;

pub use debounce::EventDebouncer;
pub use fs_watcher::FsReactor;
pub use rules::{FsEventType, ReactorAction, ReactorRule};
