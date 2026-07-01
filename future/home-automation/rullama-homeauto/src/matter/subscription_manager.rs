//! Subscription registry for the Matter Interaction Model.
//!
//! Matter spec §8.5 allows controllers to subscribe to one or more attribute
//! paths and receive `ReportData` when those attributes change. This module
//! maintains the registry of active subscriptions and exposes:
//!
//! - [`SubscriptionManager::register`] — called when a `SubscribeRequest`
//!   arrives; assigns a unique subscription id and stores the session + paths.
//! - [`SubscriptionManager::matches`] — called from the attribute-mutation path
//!   to find every subscription interested in a given `(endpoint, cluster,
//!   attribute)` and return their ids + session info.
//! - [`SubscriptionManager::remove_by_session`] — called on session teardown
//!   so subscriptions don't leak when the controller disconnects.
//!
//! Actual wire delivery of `ReportData` (including encrypted framing and
//! message-counter management) is the server's responsibility — see
//! `server.rs::notify_attribute_change`.

use std::net::SocketAddr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use super::clusters::AttributePath;

/// A single active subscription.
#[derive(Debug, Clone)]
pub struct Subscription {
    /// Assigned subscription identifier, unique within this manager.
    pub id: u32,
    /// Matter session the subscription was established on.
    pub session_id: u16,
    /// Peer address the `ReportData` updates must be sent back to.
    pub peer: SocketAddr,
    /// Exchange id used for initial response; subsequent reports may use
    /// fresh exchange ids depending on the implementation.
    pub exchange_id: u16,
    /// Attribute paths the controller wants to observe.
    pub attribute_paths: Vec<AttributePath>,
    /// Negotiated minimum interval floor (seconds).
    pub min_interval: u16,
    /// Negotiated maximum interval ceiling (seconds).
    pub max_interval: u16,
    /// Whether the controller requested fabric-filtered reporting.
    pub fabric_filtered: bool,
}

/// Thread-safe registry of active subscriptions.
#[derive(Debug, Default)]
pub struct SubscriptionManager {
    next_id: AtomicU32,
    subscriptions: Mutex<Vec<Subscription>>,
}

impl SubscriptionManager {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            next_id: AtomicU32::new(1),
            subscriptions: Mutex::new(Vec::new()),
        }
    }

    /// Register a new subscription and return the assigned id.
    // reason: too_many_arguments — Matter subscription registration takes
    // many distinct Matter-protocol parameters; bundling into a struct would
    // be ceremony without clarity.
    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &self,
        session_id: u16,
        peer: SocketAddr,
        exchange_id: u16,
        attribute_paths: Vec<AttributePath>,
        min_interval: u16,
        max_interval: u16,
        fabric_filtered: bool,
    ) -> u32 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let sub = Subscription {
            id,
            session_id,
            peer,
            exchange_id,
            attribute_paths,
            min_interval,
            max_interval,
            fabric_filtered,
        };
        self.subscriptions
            .lock()
            .expect("subscription mutex poisoned")
            .push(sub);
        id
    }

    /// Return every subscription interested in an attribute mutation.
    ///
    /// An `AttributePath` in a subscription matches the given `(endpoint,
    /// cluster, attribute)` when each of its fields is either wildcard
    /// (`None`) or equal to the concrete value.
    pub fn matches(&self, endpoint: u16, cluster: u32, attribute: u32) -> Vec<Subscription> {
        let guard = self
            .subscriptions
            .lock()
            .expect("subscription mutex poisoned");
        guard
            .iter()
            .filter(|sub| {
                sub.attribute_paths
                    .iter()
                    .any(|p| path_matches(p, endpoint, cluster, attribute))
            })
            .cloned()
            .collect()
    }

    /// Remove every subscription attached to the given session.
    ///
    /// Called on session teardown so the registry doesn't keep sending
    /// ReportData to a peer that has disconnected.
    pub fn remove_by_session(&self, session_id: u16) {
        let mut guard = self
            .subscriptions
            .lock()
            .expect("subscription mutex poisoned");
        guard.retain(|sub| sub.session_id != session_id);
    }

    /// Number of active subscriptions (for metrics / debugging).
    pub fn len(&self) -> usize {
        self.subscriptions
            .lock()
            .expect("subscription mutex poisoned")
            .len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Take a snapshot of all active subscriptions.
    pub fn snapshot(&self) -> Vec<Subscription> {
        self.subscriptions
            .lock()
            .expect("subscription mutex poisoned")
            .clone()
    }
}

fn path_matches(path: &AttributePath, endpoint: u16, cluster: u32, attribute: u32) -> bool {
    (path.endpoint_id.is_none() || path.endpoint_id == Some(endpoint))
        && (path.cluster_id.is_none() || path.cluster_id == Some(cluster))
        && (path.attribute_id.is_none() || path.attribute_id == Some(attribute))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr() -> SocketAddr {
        "127.0.0.1:5540".parse().unwrap()
    }

    #[test]
    fn register_assigns_unique_ids() {
        let mgr = SubscriptionManager::new();
        let a = mgr.register(1, addr(), 10, vec![], 0, 30, false);
        let b = mgr.register(1, addr(), 11, vec![], 0, 30, false);
        assert_ne!(a, b);
    }

    #[test]
    fn matches_specific_path() {
        let mgr = SubscriptionManager::new();
        let path = AttributePath::specific(1, 0x0006, 0x0000);
        let id = mgr.register(1, addr(), 10, vec![path], 0, 30, false);
        let hits = mgr.matches(1, 0x0006, 0x0000);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, id);
        assert!(mgr.matches(2, 0x0006, 0x0000).is_empty());
        assert!(mgr.matches(1, 0x0008, 0x0000).is_empty());
    }

    #[test]
    fn wildcard_endpoint_matches_all_endpoints() {
        let mgr = SubscriptionManager::new();
        let path = AttributePath {
            endpoint_id: None,
            cluster_id: Some(0x0006),
            attribute_id: Some(0x0000),
        };
        let _ = mgr.register(1, addr(), 10, vec![path], 0, 30, false);
        assert_eq!(mgr.matches(0, 0x0006, 0x0000).len(), 1);
        assert_eq!(mgr.matches(1, 0x0006, 0x0000).len(), 1);
        assert_eq!(mgr.matches(99, 0x0006, 0x0000).len(), 1);
        assert!(mgr.matches(1, 0x0008, 0x0000).is_empty());
    }

    #[test]
    fn remove_by_session_cleans_up() {
        let mgr = SubscriptionManager::new();
        let path = AttributePath::specific(1, 0x0006, 0x0000);
        mgr.register(5, addr(), 10, vec![path.clone()], 0, 30, false);
        mgr.register(6, addr(), 11, vec![path.clone()], 0, 30, false);
        assert_eq!(mgr.len(), 2);
        mgr.remove_by_session(5);
        assert_eq!(mgr.len(), 1);
        mgr.remove_by_session(6);
        assert!(mgr.is_empty());
    }
}
