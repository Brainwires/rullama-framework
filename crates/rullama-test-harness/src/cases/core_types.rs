//! Tier-A feature cases for `rullama-core` types & traits.

use anyhow::Result;
use async_trait::async_trait;
use rullama_core::{Message, Provider, Role, Usage};
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_test_fixtures::ScriptedProvider;

use crate::registry::TierACase;

// ── core.usage_totals ───────────────────────────────────────────────────────

inventory::submit! {
    TierACase {
        path: "rullama_test_harness::cases::core_types::usage_totals",
        crate_name: "rullama-core",
        description: "Usage::new(p, c) yields total_tokens = p + c",
        factory: || Box::new(UsageTotalsCase),
    }
}

struct UsageTotalsCase;

#[async_trait]
impl EvaluationCase for UsageTotalsCase {
    fn name(&self) -> &str {
        "feature.core.usage_totals"
    }
    fn category(&self) -> &str {
        "feature.core"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let u = Usage::new(30, 20);
        if u.total_tokens != 50 {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("Usage::new(30,20).total_tokens={} expected 50", u.total_tokens),
            ));
        }
        let z = Usage::default();
        if z.total_tokens != 0 || z.prompt_tokens != 0 || z.completion_tokens != 0 {
            return Ok(TrialResult::failure(0, 0, "Usage::default() is not zero"));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── core.message_constructors ───────────────────────────────────────────────

inventory::submit! {
    TierACase {
        path: "rullama_test_harness::cases::core_types::message_constructors",
        crate_name: "rullama-core",
        description: "Message::{user,assistant,system,tool_result} produce the right Role",
        factory: || Box::new(MessageConstructorsCase),
    }
}

struct MessageConstructorsCase;

#[async_trait]
impl EvaluationCase for MessageConstructorsCase {
    fn name(&self) -> &str {
        "feature.core.message_constructors"
    }
    fn category(&self) -> &str {
        "feature.core"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let u = Message::user("u");
        let a = Message::assistant("a");
        let s = Message::system("s");
        let t = Message::tool_result("id-1", "ok");
        let pairs = [
            (Role::User, &u),
            (Role::Assistant, &a),
            (Role::System, &s),
            (Role::Tool, &t),
        ];
        for (expected, m) in pairs {
            if m.role != expected {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!("expected Role::{expected:?}, got {:?}", m.role),
                ));
            }
        }
        if u.text() != Some("u") || a.text() != Some("a") {
            return Ok(TrialResult::failure(0, 0, "Message::text() mismatch"));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── core.provider_trait_roundtrip ───────────────────────────────────────────

inventory::submit! {
    TierACase {
        path: "rullama_test_harness::cases::core_types::provider_trait_roundtrip",
        crate_name: "rullama-core",
        description: "Provider trait dispatches `chat` correctly via dyn Provider",
        factory: || Box::new(ProviderTraitRoundtripCase),
    }
}

struct ProviderTraitRoundtripCase;

#[async_trait]
impl EvaluationCase for ProviderTraitRoundtripCase {
    fn name(&self) -> &str {
        "feature.core.provider_trait_roundtrip"
    }
    fn category(&self) -> &str {
        "feature.core"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let p: Box<dyn Provider> =
            Box::new(ScriptedProvider::always_text("test", "echo"));
        let r = p
            .chat(
                &[Message::user("ping")],
                None,
                &rullama_core::ChatOptions::default(),
            )
            .await?;
        if r.message.text() != Some("echo") {
            return Ok(TrialResult::failure(
                0,
                0,
                format!("expected 'echo', got {:?}", r.message.text()),
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
