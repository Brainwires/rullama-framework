/// Model architecture configurations and presets.
pub mod config;
/// Transformer block structural definitions.
pub mod transformer;

pub use config::{SmallLmConfig, TransformerConfig};
pub use transformer::TransformerBlock;
