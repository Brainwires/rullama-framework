use async_trait::async_trait;
use thiserror::Error;

/// Error type returned by [`BillingHook`] implementations.
#[derive(Debug, Error)]
pub enum BillingError {
    /// A hook implementation failed to record the event.
    #[error("billing hook error: {0}")]
    Hook(String),

    /// The caller's budget is exhausted — the pending call must be rejected.
    ///
    /// Returned from [`BillingHook::authorize`] to indicate that the agent
    /// should refuse to dispatch the tool / provider call. Unlike
    /// [`BillingError::Hook`], which is advisory (logged, call proceeds), this
    /// variant is used by the hard-enforcement path.
    #[error("budget exhausted for agent '{agent_id}': {spent:.6} / {limit:.6} USD")]
    BudgetExhausted {
        /// Agent whose budget was exhausted.
        agent_id: String,
        /// USD already spent.
        spent: f64,
        /// USD ceiling.
        limit: f64,
    },

    /// JSON serialization / deserialization failed.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Receives billable usage events emitted by the agent run loop.
///
/// Implement this trait to handle events however your application needs —
/// persist to a database, aggregate into a wallet, forward to Stripe, etc.
/// Pass an `Arc<dyn BillingHook>` (via `BillingHookRef`) into
/// `TaskAgentConfig::billing_hook`.
///
/// # Advisory vs enforced paths
///
/// * [`BillingHook::on_usage`] is **advisory / fail-open**: called *after* a
///   call has happened to record its cost. Errors are logged but never abort
///   the agent run.
/// * [`BillingHook::authorize`] is **enforced / fail-closed**: called *before*
///   a call is dispatched. Returning
///   [`BillingError::BudgetExhausted`] causes the agent to reject the pending
///   tool call. The default implementation returns `Ok(())` so existing
///   integrators that only care about observation keep working unchanged.
#[async_trait]
pub trait BillingHook: Send + Sync + 'static {
    /// Record a billable event that has already occurred. Advisory — errors
    /// are logged but do not abort the agent loop.
    async fn on_usage(&self, event: &crate::UsageEvent) -> Result<(), BillingError>;

    /// Authorize a pending call before it is dispatched.
    ///
    /// Implementations that enforce a budget ceiling should return
    /// [`BillingError::BudgetExhausted`] when the caller can no longer afford
    /// the pending event. The default implementation returns `Ok(())`, which
    /// means existing `BillingHook` impls remain purely advisory and require
    /// no code change.
    ///
    /// `pending` is a description of the call that is *about* to happen. For
    /// tool calls this will be a [`crate::UsageEvent::ToolCall`] with
    /// `cost_usd == 0.0` (for free built-ins) or a pre-computed estimate.
    async fn authorize(&self, pending: &crate::UsageEvent) -> Result<(), BillingError> {
        let _ = pending;
        Ok(())
    }
}
