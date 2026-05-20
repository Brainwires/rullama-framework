use thiserror::Error;

/// Errors that can occur within the networking stack.
///
/// Covers all five layers: identity, transport, routing, discovery, and
/// application.
#[derive(Debug, Clone, Error)]
pub enum NetworkError {
    // ── Identity layer ──────────────────────────────────────────────────
    /// An agent identity could not be resolved.
    #[error("identity not found: {0}")]
    IdentityNotFound(String),

    /// Credential verification failed.
    #[error("authentication failed: {0}")]
    AuthenticationFailed(String),

    // ── Transport layer ─────────────────────────────────────────────────
    /// A transport-level connection failure.
    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    /// A transport is not connected when an operation requires it.
    #[error("not connected: {0}")]
    NotConnected(String),

    /// Sending a message failed at the transport level.
    #[error("send failed: {0}")]
    SendFailed(String),

    /// Receiving a message failed at the transport level.
    #[error("receive failed: {0}")]
    ReceiveFailed(String),

    // ── Routing layer ───────────────────────────────────────────────────
    /// A message could not be routed to its destination.
    #[error("routing failed: {0}")]
    RoutingFailed(String),

    /// No route exists to the requested peer.
    #[error("no route to peer: {0}")]
    NoRoute(String),

    // ── Discovery layer ─────────────────────────────────────────────────
    /// Peer discovery failed.
    #[error("discovery failed: {0}")]
    DiscoveryFailed(String),

    /// Agent registration with a discovery service failed.
    #[error("registration failed: {0}")]
    RegistrationFailed(String),

    // ── Application layer ───────────────────────────────────────────────
    /// A peer/node was not found in the peer table.
    #[error("peer not found: {0}")]
    PeerNotFound(String),

    /// A federation request was denied by policy.
    #[error("federation denied: {0}")]
    FederationDenied(String),

    /// The network manager is in an invalid state for the requested operation.
    #[error("invalid state: {0}")]
    InvalidState(String),

    // ── General ─────────────────────────────────────────────────────────
    /// An internal or unexpected error.
    #[error("internal error: {0}")]
    Internal(String),

    /// Operation timed out.
    #[error("timeout: {0}")]
    Timeout(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let cases: Vec<(NetworkError, &str)> = vec![
            (
                NetworkError::IdentityNotFound("agent-x".into()),
                "identity not found: agent-x",
            ),
            (
                NetworkError::ConnectionFailed("refused".into()),
                "connection failed: refused",
            ),
            (
                NetworkError::RoutingFailed("no path".into()),
                "routing failed: no path",
            ),
            (
                NetworkError::DiscoveryFailed("timeout".into()),
                "discovery failed: timeout",
            ),
            (
                NetworkError::PeerNotFound("abc".into()),
                "peer not found: abc",
            ),
            (
                NetworkError::FederationDenied("policy".into()),
                "federation denied: policy",
            ),
            (NetworkError::Timeout("5s".into()), "timeout: 5s"),
        ];

        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }

    #[test]
    fn error_is_std_error() {
        let err = NetworkError::Internal("test".into());
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn error_clone() {
        let err = NetworkError::SendFailed("broken pipe".into());
        let cloned = err.clone();
        assert_eq!(err.to_string(), cloned.to_string());
    }
}
