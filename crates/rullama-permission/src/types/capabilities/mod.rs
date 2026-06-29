/// Filesystem capability types.
pub mod filesystem;
/// Git capability and operation types.
pub mod git;
/// Network capability types.
pub mod network;
/// Resource quota types.
pub mod quotas;
/// Agent spawning capability types.
pub mod spawning;
/// Tool capability and category types.
pub mod tools;

pub use filesystem::FilesystemCapabilities;
pub use git::{GitCapabilities, GitOperation};
pub use network::NetworkCapabilities;
pub use quotas::ResourceQuotas;
pub use spawning::SpawningCapabilities;
pub use tools::{ToolCapabilities, ToolCategory};
