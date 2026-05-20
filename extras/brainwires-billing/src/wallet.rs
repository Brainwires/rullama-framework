use std::sync::Arc;

use async_trait::async_trait;
use brainwires_telemetry::{BillingError, BillingHook, UsageEvent};

use crate::{BillingImplError, BillingLedger};

/// Per-customer spend tracker that implements [`BillingHook`].
///
/// Pass an `Arc<AgentWallet>` as `TaskAgentConfig::billing_hook`. Every
/// `UsageEvent` fired by the agent loop is forwarded here, persisted to the
/// ledger, and checked against the optional USD ceiling.
///
/// # Advisory vs enforced
///
/// * [`AgentWallet::on_usage`] (advisory) records spend and — when the
///   ceiling is crossed — returns `BillingError::Hook(...)`. The agent logs
///   this but keeps running (fail-open).
/// * [`AgentWallet::authorize`] (enforced) is called *before* the agent
///   dispatches a tool or provider call. It returns
///   [`BillingError::BudgetExhausted`] as soon as
///   [`AgentWallet::budget_exhausted`] is true, causing the agent to reject
///   the pending call outright (fail-closed).
pub struct AgentWallet<L: BillingLedger> {
    agent_id: String,
    max_cost_usd: Option<f64>,
    accumulated: tokio::sync::Mutex<f64>,
    ledger: Arc<L>,
}

impl<L: BillingLedger> AgentWallet<L> {
    pub fn new(agent_id: String, max_cost_usd: Option<f64>, ledger: Arc<L>) -> Self {
        Self {
            agent_id,
            max_cost_usd,
            accumulated: tokio::sync::Mutex::new(0.0),
            ledger,
        }
    }

    pub async fn total_cost_usd(&self) -> f64 {
        *self.accumulated.lock().await
    }

    pub async fn budget_exhausted(&self) -> bool {
        let spent = *self.accumulated.lock().await;
        self.max_cost_usd.is_some_and(|limit| spent >= limit)
    }

    pub async fn remaining_usd(&self) -> Option<f64> {
        let spent = *self.accumulated.lock().await;
        self.max_cost_usd.map(|limit| (limit - spent).max(0.0))
    }
}

#[async_trait]
impl<L: BillingLedger> BillingHook for AgentWallet<L> {
    async fn on_usage(&self, event: &UsageEvent) -> Result<(), BillingError> {
        let cost = event.cost_usd();

        self.ledger
            .record(event.clone())
            .await
            .map_err(|e: BillingImplError| BillingError::Hook(e.to_string()))?;

        let mut acc = self.accumulated.lock().await;
        *acc += cost;

        if let Some(limit) = self.max_cost_usd
            && *acc >= limit
        {
            return Err(BillingError::Hook(format!(
                "agent '{}' exceeded cost budget: ${:.6} / ${:.6} USD",
                self.agent_id, *acc, limit
            )));
        }

        Ok(())
    }

    async fn authorize(&self, pending: &UsageEvent) -> Result<(), BillingError> {
        let _ = pending;

        let Some(limit) = self.max_cost_usd else {
            return Ok(());
        };

        let spent = *self.accumulated.lock().await;
        if spent >= limit {
            return Err(BillingError::BudgetExhausted {
                agent_id: self.agent_id.clone(),
                spent,
                limit,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemoryLedger;
    use brainwires_telemetry::UsageEvent;

    fn make_wallet(limit: Option<f64>) -> AgentWallet<InMemoryLedger> {
        let ledger = Arc::new(InMemoryLedger::new());
        AgentWallet::new("agent-test".into(), limit, ledger)
    }

    #[tokio::test]
    async fn accumulates_cost() {
        let w = make_wallet(None);
        w.on_usage(&UsageEvent::tokens("agent-test", "m", 100, 0.001))
            .await
            .unwrap();
        w.on_usage(&UsageEvent::tokens("agent-test", "m", 200, 0.002))
            .await
            .unwrap();
        assert!((w.total_cost_usd().await - 0.003).abs() < 1e-9);
    }

    #[tokio::test]
    async fn budget_exceeded_returns_error() {
        let w = make_wallet(Some(0.005));
        w.on_usage(&UsageEvent::tokens("agent-test", "m", 100, 0.003))
            .await
            .unwrap();
        let err = w
            .on_usage(&UsageEvent::tokens("agent-test", "m", 100, 0.003))
            .await
            .unwrap_err();
        assert!(matches!(err, BillingError::Hook(_)));
        assert!(w.budget_exhausted().await);
    }

    #[tokio::test]
    async fn no_limit_never_errors() {
        let w = make_wallet(None);
        for _ in 0..50 {
            w.on_usage(&UsageEvent::tokens("agent-test", "m", 1000, 0.10))
                .await
                .unwrap();
        }
        assert!(!w.budget_exhausted().await);
    }
}
