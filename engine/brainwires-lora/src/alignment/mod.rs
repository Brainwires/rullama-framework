/// Direct Preference Optimization (DPO) loss implementation.
pub mod dpo;
/// Odds Ratio Preference Optimization (ORPO) loss implementation.
pub mod orpo;

pub use dpo::DpoLoss;
pub use orpo::OrpoLoss;
