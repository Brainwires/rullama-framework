use thiserror::Error;

/// Matter-specific errors.
///
/// Note: We implement the Matter protocol ourselves (TLV + commissioning + UDP transport)
/// rather than using rs-matter, to avoid an `embassy-time` links conflict with the
/// burn ML ecosystem in the workspace.
#[derive(Debug, Error)]
pub enum MatterError {
    /// High-level commissioning flow failed (PASE, fabric install, CASE setup).
    #[error("commissioning failed: {0}")]
    Commissioning(String),

    /// Matter commissioning QR code / manual pairing code parse failure.
    #[error("QR code parse error: {0}")]
    QrCode(&'static str),

    /// Cluster-specific command invoke returned an error.
    #[error("cluster invoke error (cluster {cluster:#010x} cmd {cmd:#010x}): {msg}")]
    ClusterInvoke {
        /// Matter cluster ID.
        cluster: u32,
        /// Command ID within the cluster.
        cmd: u32,
        /// Human-readable failure reason.
        msg: String,
    },

    /// Attribute read returned an error status from the peer.
    #[error("attribute read error (cluster {cluster:#010x} attr {attr:#010x}): {msg}")]
    AttributeRead {
        /// Matter cluster ID hosting the attribute.
        cluster: u32,
        /// Attribute ID within the cluster.
        attr: u32,
        /// Human-readable failure reason.
        msg: String,
    },

    /// Node ID is not commissioned into this controller's fabric.
    #[error("device not found: node_id={node_id}")]
    DeviceNotFound {
        /// 64-bit Matter node ID that was not found.
        node_id: u64,
    },

    /// UDP / BLE transport-level failure.
    #[error("transport error: {0}")]
    Transport(String),

    /// mDNS discovery / advertisement error.
    #[error("mDNS error: {0}")]
    Mdns(String),

    /// Filesystem or socket I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// RustCrypto-layer failure (AES-CCM, HMAC-SHA256, ECDSA, HKDF).
    #[error("crypto error: {0}")]
    Crypto(String),

    /// SPAKE2+ password-authenticated key exchange error.
    #[error("SPAKE2+ error: {0}")]
    Spake2(String),

    /// Secure-session error (CASE or PASE session ID invalid / expired / decrypt failed).
    #[error("session {session_id} error: {msg}")]
    Session {
        /// 16-bit session identifier.
        session_id: u16,
        /// Human-readable failure reason.
        msg: String,
    },

    /// Protocol-layer error keyed on the secure-channel opcode that surfaced it.
    #[error("protocol error (opcode {opcode:#04x}): {msg}")]
    Protocol {
        /// Matter secure-channel opcode.
        opcode: u8,
        /// Human-readable failure reason.
        msg: String,
    },

    /// Access-control list rejected the requested operation.
    #[error("access denied")]
    AccessDenied,

    /// No fabric with the requested ID exists on this node.
    #[error("fabric not found")]
    FabricNotFound,
}

/// Convenience alias used throughout the Matter subsystem.
pub type MatterResult<T> = Result<T, MatterError>;
