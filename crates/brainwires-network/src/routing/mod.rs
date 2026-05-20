//! # Routing Layer
//!
//! Decides where messages go. The [`Router`](crate::routing::Router) trait takes a
//! [`MessageEnvelope`](crate::MessageEnvelope) and a
//! [`PeerTable`](crate::routing::PeerTable) and returns the
//! transport addresses that the message should be delivered to.
//!
//! ## Provided routers
//!
//! | Router | Description |
//! |--------|-------------|
//! | [`DirectRouter`](crate::routing::DirectRouter) | Point-to-point: look up the recipient in the peer table |
//! | [`BroadcastRouter`](crate::routing::BroadcastRouter) | Send to all known peers |
//! | [`ContentRouter`](crate::routing::ContentRouter) | Route based on topic subscriptions |

mod broadcast;
mod content;
mod direct;
mod peer_table;
mod traits;

pub use broadcast::BroadcastRouter;
pub use content::ContentRouter;
pub use direct::DirectRouter;
pub use peer_table::PeerTable;
pub use traits::{Router, RoutingStrategy};
