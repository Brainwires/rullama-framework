//! [`PairingHandler`] — the interception point wired into
//! [`crate::agent_handler::AgentInboundHandler`].
//!
//! When a message arrives, the gateway asks the handler to `check` the
//! peer. The handler consults the per-channel [`PairingPolicy`] and the
//! [`PairingStore`] and returns one of three outcomes:
//!
//! - [`PairingOutcome::Allow`] — normal processing continues.
//! - [`PairingOutcome::Reject`] — send a one-liner back, drop the message.
//! - [`PairingOutcome::PendingCodeIssued`] — a fresh pairing code was
//!   issued; send the neutral reply back to the peer, drop the message.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;

use super::policy::PairingPolicy;
use super::store::PairingStore;

/// The three possible results of a pairing check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingOutcome {
    /// Peer is approved; continue normal processing of this message.
    Allow,
    /// Peer is not approved; send this reply back to the peer via the
    /// channel and DO NOT forward the message to the agent.
    Reject(String),
    /// Peer is unknown AND a pairing code was issued for them; send this
    /// reply back so they can share the code with the operator.
    PendingCodeIssued {
        /// The 6-digit code (for logging / tests).
        code: String,
        /// The reply to send back to the peer.
        reply: String,
    },
}

type PolicyFn = dyn Fn(&str) -> PairingPolicy + Send + Sync;

/// The pairing interception handler.
///
/// Cheap to clone via `Arc`. Policy resolution is a synchronous closure
/// injected at construction time so callers can plug in per-channel
/// overrides from their config without this crate needing to know about
/// the config layout.
#[derive(Clone)]
pub struct PairingHandler {
    store: Arc<PairingStore>,
    policy_fn: Arc<PolicyFn>,
}

impl std::fmt::Debug for PairingHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairingHandler")
            .field("store", &self.store)
            .finish()
    }
}

impl PairingHandler {
    /// Build a new handler from a store and a policy resolver.
    pub fn new(store: Arc<PairingStore>, policy_fn: Arc<PolicyFn>) -> Self {
        Self { store, policy_fn }
    }

    /// Build a handler whose policy is fixed regardless of channel.
    /// Useful for tests and simple deployments.
    pub fn with_fixed_policy(store: Arc<PairingStore>, policy: PairingPolicy) -> Self {
        let p = policy.clone();
        Self {
            store,
            policy_fn: Arc::new(move |_| p.clone()),
        }
    }

    /// Access the underlying store — exposed so admin handlers and the
    /// `brainclaw pairing` CLI can query pending / approved lists.
    pub fn store(&self) -> Arc<PairingStore> {
        Arc::clone(&self.store)
    }

    /// Classify an incoming DM.
    pub async fn check(
        &self,
        channel: &str,
        user_id: &str,
        peer_display: &str,
        _incoming: &str,
    ) -> Result<PairingOutcome> {
        let policy = (self.policy_fn)(channel);

        match policy {
            PairingPolicy::Open { allow_from } => {
                // In Open mode, the static allowlist on the policy takes
                // precedence. Empty list = allow everyone.
                if allow_from.is_empty() {
                    return Ok(PairingOutcome::Allow);
                }
                let key = format!("{channel}:{user_id}");
                if allow_from.iter().any(|e| e == &key) {
                    Ok(PairingOutcome::Allow)
                } else {
                    Ok(PairingOutcome::Reject(
                        "this bot does not accept messages from your account".to_string(),
                    ))
                }
            }
            PairingPolicy::Pairing {
                code_ttl_secs,
                persist_approvals: _,
            } => {
                if self.store.is_approved(channel, user_id).await {
                    return Ok(PairingOutcome::Allow);
                }

                // Reuse an existing unexpired pending code if present —
                // so a spammer resending cannot rotate codes.
                if let Some(existing) = self.store.pending_for_peer(channel, user_id).await {
                    let reply = render_pending_reply(&existing.code, code_ttl_secs);
                    return Ok(PairingOutcome::PendingCodeIssued {
                        code: existing.code,
                        reply,
                    });
                }

                let pc = self
                    .store
                    .issue_code(
                        channel,
                        user_id,
                        peer_display,
                        Duration::from_secs(code_ttl_secs),
                    )
                    .await?;
                let reply = render_pending_reply(&pc.code, code_ttl_secs);
                Ok(PairingOutcome::PendingCodeIssued {
                    code: pc.code,
                    reply,
                })
            }
        }
    }
}

/// Render the peer-facing reply shown when a pairing code is issued.
///
/// Keep this short and neutral — the bot should not act as a free-tier
/// LLM for anyone who guesses its handle.
fn render_pending_reply(code: &str, ttl_secs: u64) -> String {
    let mins = ttl_secs / 60;
    format!(
        "This bot is in private mode. Share this code with the operator to pair: {code}. \
         The code expires in {mins} minutes."
    )
}

#[cfg(test)]
mod tests {
    use super::super::policy::PairingPolicy;
    use super::super::store::PairingStore;
    use super::*;
    use tempfile::tempdir;

    async fn handler_with_policy(policy: PairingPolicy) -> (tempfile::TempDir, PairingHandler) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pairing.json");
        let store = Arc::new(PairingStore::load(&path).unwrap());
        let handler = PairingHandler::with_fixed_policy(store, policy);
        (dir, handler)
    }

    #[tokio::test]
    async fn open_policy_with_empty_allowlist_allows_all() {
        let (_d, h) = handler_with_policy(PairingPolicy::Open { allow_from: vec![] }).await;
        assert_eq!(
            h.check("discord", "anyone", "Anyone", "hi").await.unwrap(),
            PairingOutcome::Allow
        );
    }

    #[tokio::test]
    async fn allow_from_matches_allowlist() {
        let (_d, h) = handler_with_policy(PairingPolicy::Open {
            allow_from: vec!["discord:alice".to_string()],
        })
        .await;
        assert_eq!(
            h.check("discord", "alice", "Alice", "hi").await.unwrap(),
            PairingOutcome::Allow
        );
        match h.check("discord", "bob", "Bob", "hi").await.unwrap() {
            PairingOutcome::Reject(_) => {}
            other => panic!("expected Reject for bob, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pairing_policy_unknown_user_issues_code_and_reuses() {
        let (_d, h) = handler_with_policy(PairingPolicy::Pairing {
            code_ttl_secs: 900,
            persist_approvals: true,
        })
        .await;

        let first = h.check("discord", "alice", "Alice", "hello").await.unwrap();
        let code1 = match &first {
            PairingOutcome::PendingCodeIssued { code, .. } => code.clone(),
            other => panic!("expected PendingCodeIssued, got {other:?}"),
        };

        let second = h
            .check("discord", "alice", "Alice", "hello again")
            .await
            .unwrap();
        match &second {
            PairingOutcome::PendingCodeIssued { code, .. } => assert_eq!(code, &code1),
            other => panic!("expected PendingCodeIssued (reused), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pairing_policy_approved_user_passes() {
        let (_d, h) = handler_with_policy(PairingPolicy::Pairing {
            code_ttl_secs: 900,
            persist_approvals: true,
        })
        .await;
        let first = h.check("discord", "alice", "Alice", "hi").await.unwrap();
        let code = match first {
            PairingOutcome::PendingCodeIssued { code, .. } => code,
            other => panic!("{other:?}"),
        };
        let approved = h.store.approve_by_code(&code).await.unwrap();
        assert_eq!(approved, Some(("discord".to_string(), "alice".to_string())));

        assert_eq!(
            h.check("discord", "alice", "Alice", "next").await.unwrap(),
            PairingOutcome::Allow
        );
    }
}
