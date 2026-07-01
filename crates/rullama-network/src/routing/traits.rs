use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::peer_table::PeerTable;
use crate::MessageEnvelope;
use crate::transport::TransportAddress;

/// Routing strategy identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Point-to-point delivery to a specific peer.
    Direct,
    /// Send to all known peers.
    Broadcast,
    /// Route based on topic subscriptions.
    ContentBased,
    /// Custom routing with a user-defined identifier.
    Custom(String),
}

/// The core routing trait.
///
/// A router takes a [`MessageEnvelope`] and the current [`PeerTable`]
/// and returns the list of transport addresses the message should be
/// sent to.
#[async_trait]
pub trait Router: Send + Sync {
    /// Determine the transport addresses to deliver this message to.
    async fn route(
        &self,
        envelope: &MessageEnvelope,
        peers: &PeerTable,
    ) -> Result<Vec<TransportAddress>>;

    /// The routing strategy this router implements.
    fn strategy(&self) -> RoutingStrategy;
}
