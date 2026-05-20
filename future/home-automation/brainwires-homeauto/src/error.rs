use thiserror::Error;

/// Unified error type for all home automation protocol operations.
#[derive(Debug, Error)]
pub enum HomeAutoError {
    // ── Serial / transport ──────────────────────────────────────────────────
    /// Serial-port error from the Zigbee / Z-Wave transport.
    #[cfg(any(feature = "zigbee", feature = "zwave"))]
    #[error("serial port error: {0}")]
    Serial(#[from] tokio_serial::Error),

    /// Filesystem / socket / pipe failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Protocol timed out waiting for a response.
    #[error("connection timed out")]
    Timeout,

    /// Malformed frame received from the serial transport.
    #[error("serial frame error: {0}")]
    FrameError(String),

    // ── Zigbee ──────────────────────────────────────────────────────────────
    /// Coordinator (EZSP / ZNP) dongle rejected a command or surfaced a fault.
    #[error("Zigbee coordinator error: {0}")]
    ZigbeeCoordinator(String),

    /// No device is known at the supplied 64-bit IEEE address.
    #[error("Zigbee device not found: {addr:016x}")]
    ZigbeeDeviceNotFound {
        /// IEEE 64-bit address that did not resolve.
        addr: u64,
    },

    /// Cluster attribute read/write failed on an otherwise reachable device.
    #[error("Zigbee attribute error (cluster {cluster:#06x} attr {attr:#06x}): {msg}")]
    ZigbeeAttribute {
        /// Zigbee Cluster Library cluster ID.
        cluster: u16,
        /// Attribute ID within the cluster.
        attr: u16,
        /// Human-readable failure reason.
        msg: String,
    },

    /// Silicon Labs EZSP status byte returned a non-success code.
    #[error("Zigbee EZSP error (status {status:#04x}): {msg}")]
    EzspStatus {
        /// Raw EZSP status code.
        status: u8,
        /// Human-readable decoding of the status.
        msg: String,
    },

    /// Texas Instruments ZNP status byte returned a non-success code.
    #[error("Zigbee ZNP error (status {status:#04x}): {msg}")]
    ZnpStatus {
        /// Raw ZNP status code.
        status: u8,
        /// Human-readable decoding of the status.
        msg: String,
    },

    // ── Z-Wave ───────────────────────────────────────────────────────────────
    /// Controller (e.g. Aeotec Z-Stick) rejected the request or surfaced a fault.
    #[error("Z-Wave controller error: {0}")]
    ZWaveController(String),

    /// No device is currently associated with the supplied node ID.
    #[error("Z-Wave node {node_id} not found")]
    ZWaveNodeNotFound {
        /// 8-bit Z-Wave node identifier.
        node_id: u8,
    },

    /// `SendData` or equivalent transmission reported failure after all retries.
    #[error("Z-Wave transmission failed (node {node_id}): {msg}")]
    ZWaveTransmit {
        /// Destination node ID.
        node_id: u8,
        /// Human-readable failure reason.
        msg: String,
    },

    /// Controller returned NAK for every retry attempt.
    #[error("Z-Wave NAK received after {retries} retries")]
    ZWaveNak {
        /// Number of retries attempted before giving up.
        retries: u8,
    },

    // ── Thread ───────────────────────────────────────────────────────────────
    /// HTTP failure talking to the OpenThread Border Router REST API.
    #[error("Thread border router HTTP error: {0}")]
    ThreadHttp(String),

    /// OTBR returned a body that did not parse against the expected schema.
    #[error("Thread border router response parse error: {0}")]
    ThreadParse(String),

    // ── Matter ───────────────────────────────────────────────────────────────
    /// Generic Matter-stack error surfaced from the secure channel / transport.
    #[error("Matter error: {0}")]
    Matter(String),

    /// A commissioning step (PASE, CASE, fabric install) failed.
    #[error("Matter commissioning error: {0}")]
    MatterCommissioning(String),

    /// Invoke command against a specific cluster failed on the target node.
    #[error("Matter cluster invoke error (cluster {cluster:#010x} cmd {cmd:#010x}): {msg}")]
    MatterCluster {
        /// Matter cluster ID (32-bit).
        cluster: u32,
        /// Command ID within the cluster.
        cmd: u32,
        /// Human-readable failure reason.
        msg: String,
    },

    // ── General ──────────────────────────────────────────────────────────────
    /// Requested operation is not supported by the active backend / build.
    #[error("not supported: {0}")]
    Unsupported(String),

    /// An async channel was dropped before the response arrived.
    #[error("channel closed")]
    ChannelClosed,
}

/// Convenience alias used across every `homeauto` submodule.
pub type HomeAutoResult<T> = Result<T, HomeAutoError>;
