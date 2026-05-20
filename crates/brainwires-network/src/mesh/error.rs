use thiserror::Error;

/// Errors that can occur within the mesh networking layer.
#[derive(Debug, Clone, Error)]
pub enum MeshError {
    /// The requested node was not found in the mesh.
    #[error("node not found: {0}")]
    NodeNotFound(String),

    /// A message could not be routed to its destination.
    #[error("routing failed: {0}")]
    RoutingFailed(String),

    /// Peer discovery failed.
    #[error("discovery failed: {0}")]
    DiscoveryFailed(String),

    /// A federation request was denied by policy.
    #[error("federation denied: {0}")]
    FederationDenied(String),

    /// An error occurred while modifying the mesh topology.
    #[error("topology error: {0}")]
    TopologyError(String),

    /// A transport-level error occurred.
    #[error("transport error: {0}")]
    Transport(String),

    /// An internal or unexpected error.
    #[error("internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_not_found_display() {
        let err = MeshError::NodeNotFound("abc-123".into());
        assert_eq!(err.to_string(), "node not found: abc-123");
    }

    #[test]
    fn routing_failed_display() {
        let err = MeshError::RoutingFailed("no path".into());
        assert_eq!(err.to_string(), "routing failed: no path");
    }

    #[test]
    fn discovery_failed_display() {
        let err = MeshError::DiscoveryFailed("timeout".into());
        assert_eq!(err.to_string(), "discovery failed: timeout");
    }

    #[test]
    fn federation_denied_display() {
        let err = MeshError::FederationDenied("policy violation".into());
        assert_eq!(err.to_string(), "federation denied: policy violation");
    }

    #[test]
    fn topology_error_display() {
        let err = MeshError::TopologyError("cycle detected".into());
        assert_eq!(err.to_string(), "topology error: cycle detected");
    }

    #[test]
    fn transport_error_display() {
        let err = MeshError::Transport("connection refused".into());
        assert_eq!(err.to_string(), "transport error: connection refused");
    }

    #[test]
    fn internal_error_display() {
        let err = MeshError::Internal("unexpected state".into());
        assert_eq!(err.to_string(), "internal error: unexpected state");
    }

    #[test]
    fn error_is_std_error() {
        let err = MeshError::Internal("test".into());
        // Verify it implements std::error::Error via trait object coercion
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn error_debug_format() {
        let err = MeshError::NodeNotFound("xyz".into());
        let debug = format!("{:?}", err);
        assert!(debug.contains("NodeNotFound"));
        assert!(debug.contains("xyz"));
    }

    #[test]
    fn error_clone() {
        let err = MeshError::RoutingFailed("no route".into());
        let cloned = err.clone();
        assert_eq!(err.to_string(), cloned.to_string());
    }
}
