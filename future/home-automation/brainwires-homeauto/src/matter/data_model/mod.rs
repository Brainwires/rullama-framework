//! Matter Data Model — cluster server dispatch, ACL, and cluster implementations.
//!
//! This module forms Phase 6 of the Matter 1.3 protocol stack. It provides:
//!
//! - [`ClusterServer`] trait — uniform interface for serving a single cluster.
//! - [`DataModelNode`] — routes read/write/invoke requests to the right cluster.
//! - [`Privilege`] — access privilege levels (Matter spec §6.6.5.1).
//! - [`acl`] — Access Control List enforcement.
//! - [`clusters`] — commissioning and basic-information cluster servers.

/// Access-control-list entries + enforcement (Matter ACL cluster).
pub mod acl;
/// Concrete server-side implementations of built-in Matter clusters
/// (basic_information, general_commissioning, network_commissioning,
/// operational_credentials).
pub mod clusters;

use std::collections::HashMap;

use async_trait::async_trait;

use crate::matter::clusters::AttributePath;
use crate::matter::error::MatterResult;
use crate::matter::interaction_model::{
    AttributeData, AttributeStatus, InteractionStatus,
};

// ── Privilege levels ──────────────────────────────────────────────────────────

/// Privilege levels per Matter spec §6.6.5.1.
///
/// Ordered from lowest to highest — ordinal comparison returns whether one
/// privilege subsumes another.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Privilege {
    /// `1` — View: read-only access to attributes and events.
    View = 1,
    /// `2` — ProxyView: used by sleepy device proxies.
    ProxyView = 2,
    /// `3` — Operate: issue operational commands and write operate-level attributes.
    Operate = 3,
    /// `4` — Manage: configure the device (adjust non-operational attributes).
    Manage = 4,
    /// `5` — Administer: full fabric-level access (commissioning, ACL edits).
    Administer = 5,
}

// ── ClusterServer trait ───────────────────────────────────────────────────────

/// A trait for serving a single Matter cluster.
///
/// Implementations must be `Send + Sync` so they can live inside a
/// `DataModelNode` that is shared across async tasks.
#[async_trait]
pub trait ClusterServer: Send + Sync {
    /// Cluster identifier served by this implementation.
    fn cluster_id(&self) -> u32;

    /// Read an attribute.  Returns TLV-encoded attribute value bytes.
    async fn read_attribute(&self, attr_id: u32) -> MatterResult<Vec<u8>>;

    /// Write an attribute.  `value` is TLV-encoded.
    async fn write_attribute(&self, attr_id: u32, value: &[u8]) -> MatterResult<()>;

    /// Invoke a command.  `args` and the return value are TLV-encoded.
    async fn invoke_command(&self, cmd_id: u32, args: &[u8]) -> MatterResult<Vec<u8>>;

    /// Return the list of supported attribute IDs.
    fn attribute_ids(&self) -> Vec<u32>;

    /// Return the list of supported command IDs.
    fn command_ids(&self) -> Vec<u32>;
}

// ── DataModelNode ─────────────────────────────────────────────────────────────

/// A node in the Matter data model: a collection of endpoints, each with clusters.
pub struct DataModelNode {
    /// `endpoint_id` → (`cluster_id` → `ClusterServer`)
    pub endpoints: HashMap<u16, HashMap<u32, Box<dyn ClusterServer>>>,
}

impl DataModelNode {
    /// Create an empty node.
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
        }
    }

    /// Add a cluster server to the given endpoint.  Replaces any existing
    /// server for the same `(endpoint, cluster_id)` pair.
    pub fn add_cluster(&mut self, endpoint: u16, cluster: Box<dyn ClusterServer>) {
        let cluster_id = cluster.cluster_id();
        self.endpoints
            .entry(endpoint)
            .or_default()
            .insert(cluster_id, cluster);
    }

    /// Dispatch a read request.
    ///
    /// Handles wildcard endpoint / cluster / attribute paths.  For every path
    /// component that is `None` (wildcard), all matching elements are included.
    pub async fn dispatch_read(&self, path: &AttributePath) -> Vec<AttributeData> {
        let mut results = Vec::new();

        for (ep_id, clusters) in &self.endpoints {
            if let Some(want_ep) = path.endpoint_id
                && *ep_id != want_ep
            {
                continue;
            }
            for (cl_id, server) in clusters {
                if let Some(want_cl) = path.cluster_id
                    && *cl_id != want_cl
                {
                    continue;
                }
                let attr_ids: Vec<u32> = if let Some(want_attr) = path.attribute_id {
                    vec![want_attr]
                } else {
                    server.attribute_ids()
                };
                for attr_id in attr_ids {
                    match server.read_attribute(attr_id).await {
                        Ok(data) => {
                            results.push(AttributeData {
                                path: AttributePath::specific(*ep_id, *cl_id, attr_id),
                                data,
                            });
                        }
                        Err(_) => {
                            // Skip unreadable attributes during wildcard reads.
                        }
                    }
                }
            }
        }
        results
    }

    /// Dispatch an invoke command.
    ///
    /// Returns `MatterError` if the endpoint or cluster is not found.
    pub async fn dispatch_invoke(
        &self,
        endpoint: u16,
        cluster_id: u32,
        cmd_id: u32,
        args: &[u8],
    ) -> MatterResult<Vec<u8>> {
        let ep = self.endpoints.get(&endpoint).ok_or_else(|| {
            crate::matter::error::MatterError::Transport(format!(
                "endpoint {endpoint} not found"
            ))
        })?;
        let server = ep.get(&cluster_id).ok_or_else(|| {
            crate::matter::error::MatterError::Transport(format!(
                "cluster {cluster_id:#010x} not found on endpoint {endpoint}"
            ))
        })?;
        server.invoke_command(cmd_id, args).await
    }

    /// Dispatch a write request.
    ///
    /// Returns `AttributeStatus` for the given `AttributeData`.
    pub async fn dispatch_write(&self, data: &AttributeData) -> AttributeStatus {
        let path = &data.path;
        let ep_id = match path.endpoint_id {
            Some(id) => id,
            None => {
                return AttributeStatus {
                    path: path.clone(),
                    status: InteractionStatus::UnsupportedEndpoint,
                };
            }
        };
        let cl_id = match path.cluster_id {
            Some(id) => id,
            None => {
                return AttributeStatus {
                    path: path.clone(),
                    status: InteractionStatus::UnsupportedCluster,
                };
            }
        };
        let attr_id = match path.attribute_id {
            Some(id) => id,
            None => {
                return AttributeStatus {
                    path: path.clone(),
                    status: InteractionStatus::UnsupportedAttribute,
                };
            }
        };

        match self.endpoints.get(&ep_id).and_then(|ep| ep.get(&cl_id)) {
            None => AttributeStatus {
                path: path.clone(),
                status: InteractionStatus::UnsupportedCluster,
            },
            Some(server) => match server.write_attribute(attr_id, &data.data).await {
                Ok(()) => AttributeStatus {
                    path: path.clone(),
                    status: InteractionStatus::Success,
                },
                Err(_) => AttributeStatus {
                    path: path.clone(),
                    status: InteractionStatus::Failure,
                },
            },
        }
    }
}

impl Default for DataModelNode {
    fn default() -> Self {
        Self::new()
    }
}

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use acl::{AccessControlEntry, AccessControlList, AclTarget};
pub use clusters::{
    basic_information::BasicInformationCluster, general_commissioning::GeneralCommissioningCluster,
    network_commissioning::NetworkCommissioningCluster,
    operational_credentials::OperationalCredentialsCluster,
};
