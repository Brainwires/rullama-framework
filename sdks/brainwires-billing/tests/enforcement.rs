//! Integration tests for the hard-enforcement `BillingHook::authorize` path.
//!
//! These tests verify that [`AgentWallet`](brainwires_billing_impl::AgentWallet)
//! fails **closed** when the budget is exhausted — i.e. returns
//! `BillingError::BudgetExhausted` from `authorize()` so the agent's tool-call
//! dispatcher can reject the pending call before running it.

use std::sync::Arc;

use brainwires_billing_impl::{AgentWallet, InMemoryLedger};
use brainwires_telemetry::{BillingError, BillingHook, UsageEvent};

fn make_wallet(limit: Option<f64>) -> AgentWallet<InMemoryLedger> {
    let ledger = Arc::new(InMemoryLedger::new());
    AgentWallet::new("agent-enforce".into(), limit, ledger)
}

#[tokio::test]
async fn authorize_rejects_when_budget_exhausted() {
    // Tiny budget — one recorded token event pushes it over.
    let wallet = make_wallet(Some(0.001));

    // Initially there's budget available.
    let pending = UsageEvent::tool_call("agent-enforce", "bash");
    wallet
        .authorize(&pending)
        .await
        .expect("fresh wallet must authorize");

    // Spend past the limit via the advisory path — this is the only way
    // `budget_exhausted()` flips to true in the current wallet.
    wallet
        .on_usage(&UsageEvent::tokens("agent-enforce", "m", 100, 0.005))
        .await
        .unwrap_err(); // advisory `Hook(...)` error is expected
    assert!(
        wallet.budget_exhausted().await,
        "budget should be exhausted after overspend"
    );

    // Now `authorize()` must fail closed with `BudgetExhausted`.
    let err = wallet
        .authorize(&pending)
        .await
        .expect_err("authorize() must reject when budget is exhausted");

    match err {
        BillingError::BudgetExhausted {
            agent_id,
            spent,
            limit,
        } => {
            assert_eq!(agent_id, "agent-enforce");
            assert!(spent >= limit, "spent {spent} should meet/exceed {limit}");
            assert!((limit - 0.001).abs() < 1e-9);
        }
        other => panic!("expected BudgetExhausted, got {other:?}"),
    }
}

#[tokio::test]
async fn authorize_allows_when_budget_has_room() {
    // Ample budget — authorize should always be Ok.
    let wallet = make_wallet(Some(100.0));

    let pending = UsageEvent::tool_call("agent-enforce", "bash");
    wallet
        .authorize(&pending)
        .await
        .expect("wallet with ample budget must authorize");

    // Even after a real charge the budget is far from exhausted.
    wallet
        .on_usage(&UsageEvent::tokens("agent-enforce", "m", 100, 0.01))
        .await
        .unwrap();

    wallet
        .authorize(&pending)
        .await
        .expect("wallet still under budget must authorize");
}

#[tokio::test]
async fn authorize_allows_when_no_budget_set() {
    // `None` means no ceiling — authorize must always succeed, even after
    // arbitrary spend.
    let wallet = make_wallet(None);

    let pending = UsageEvent::tool_call("agent-enforce", "bash");

    for _ in 0..5 {
        wallet
            .on_usage(&UsageEvent::tokens("agent-enforce", "m", 1000, 10.0))
            .await
            .unwrap();
        wallet
            .authorize(&pending)
            .await
            .expect("unlimited wallet must always authorize");
    }
}

#[tokio::test]
async fn default_authorize_impl_is_noop() {
    // A custom `BillingHook` that doesn't override `authorize()` must keep
    // working — i.e. the default impl returns `Ok(())` regardless of spend.
    struct AdvisoryOnlyHook;

    #[async_trait::async_trait]
    impl BillingHook for AdvisoryOnlyHook {
        async fn on_usage(&self, _event: &UsageEvent) -> Result<(), BillingError> {
            Ok(())
        }
        // NOTE: deliberately does NOT override `authorize`.
    }

    let hook = AdvisoryOnlyHook;
    let pending = UsageEvent::tool_call("whatever", "bash");
    hook.authorize(&pending)
        .await
        .expect("default authorize() must be a no-op Ok(())");
}
