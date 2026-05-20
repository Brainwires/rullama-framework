//! Pairing policy for the gateway.
//!
//! A [`PairingPolicy`] controls what the gateway does with direct messages
//! from peers whose `<channel>:<user_id>` has not been approved yet. Policies
//! are resolved per-channel by the [`crate::pairing::handler::PairingHandler`].

use serde::{Deserialize, Serialize};

/// Per-channel pairing policy.
///
/// `Open` trusts the channel-level auth — anybody who can reach the bot
/// may DM it, optionally constrained to a static allow-list. `Pairing` is
/// the secure default: unknown peers go through an out-of-band approval
/// flow before any message reaches the agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum PairingPolicy {
    /// No pairing; any user may DM the bot.
    ///
    /// `allow_from` is an optional static whitelist (keyed by
    /// `<channel>:<user_id>`). When non-empty, only listed peers are
    /// allowed; otherwise every peer is allowed.
    Open {
        /// Static allowlist (keyed by `<channel>:<user_id>`).
        #[serde(default)]
        allow_from: Vec<String>,
    },
    /// Unknown users must go through the approval flow before their messages
    /// reach the agent.
    Pairing {
        /// Seconds a pending code remains valid. Default 900 (15 min).
        #[serde(default = "default_code_ttl_secs")]
        code_ttl_secs: u64,
        /// If `true`, approved peers are persisted to disk so they remain
        /// paired across gateway restarts.
        #[serde(default = "default_persist_approvals")]
        persist_approvals: bool,
    },
}

fn default_code_ttl_secs() -> u64 {
    900
}

fn default_persist_approvals() -> bool {
    true
}

/// The default policy used when no config is provided.
///
/// Secure posture: `Pairing` with a 15-minute code TTL and persisted
/// approvals. Unknown peers do not reach the agent.
pub fn default_policy() -> PairingPolicy {
    PairingPolicy::Pairing {
        code_ttl_secs: default_code_ttl_secs(),
        persist_approvals: default_persist_approvals(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_pairing() {
        match default_policy() {
            PairingPolicy::Pairing {
                code_ttl_secs,
                persist_approvals,
            } => {
                assert_eq!(code_ttl_secs, 900);
                assert!(persist_approvals);
            }
            _ => panic!("default policy must be Pairing"),
        }
    }

    #[test]
    fn policy_serde_pairing() {
        let p = PairingPolicy::Pairing {
            code_ttl_secs: 120,
            persist_approvals: false,
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: PairingPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn policy_serde_open() {
        let p = PairingPolicy::Open {
            allow_from: vec!["discord:42".to_string()],
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: PairingPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }
}
