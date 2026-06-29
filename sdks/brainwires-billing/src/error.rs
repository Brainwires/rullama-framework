use thiserror::Error;

#[derive(Debug, Error)]
pub enum BillingImplError {
    /// The agent's USD budget ceiling was exceeded.
    #[error("agent '{agent_id}' exceeded cost budget: ${spent:.6} / ${limit:.6} USD")]
    BudgetExceeded {
        agent_id: String,
        spent: f64,
        limit: f64,
    },

    /// A ledger persistence operation failed.
    #[error("ledger error: {0}")]
    Ledger(#[from] anyhow::Error),

    /// A Stripe API call failed.
    #[cfg(feature = "stripe")]
    #[error("stripe error: {0}")]
    Stripe(String),

    /// JSON serialization / deserialization failed.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

impl From<BillingImplError> for brainwires_telemetry::BillingError {
    fn from(e: BillingImplError) -> Self {
        brainwires_telemetry::BillingError::Hook(e.to_string())
    }
}
