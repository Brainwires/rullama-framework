//! Commissioning orchestration for the Matter commissioner role.
//!
//! A Matter commissioner drives a device through several stages after the
//! user hands over a QR code or manual pairing code. Those stages are:
//!
//! 1. `Parsed` — the pairing payload (vendor id, discriminator, passcode)
//!    has been decoded from the QR/manual string.
//! 2. `Discovered` — the device has been found on the network (mDNS /
//!    `_matterc._udp`) or over BLE.
//! 3. `PaseEstablished` — the SPAKE2+ handshake has completed and the
//!    commissioner shares an encrypted session with the device.
//! 4. `CsrReceived` — the device has returned its CSR (`CSRRequest`
//!    response), giving the commissioner the device's operational P-256
//!    public key.
//! 5. `NocInstalled` — the commissioner has issued a NOC against its
//!    fabric and `AddNOC` succeeded on the device.
//! 6. `CaseEstablished` — a certificate-authenticated session is up, PASE
//!    can be torn down, and normal operational traffic can flow.
//!
//! The [`CommissioningSession`] struct tracks the current phase and can
//! emit observable transitions via a broadcast channel so callers (UIs,
//! tests) can observe progress without polling.
//!
//! **Status:** stages 1–3 are fully implemented by
//! [`super::controller::MatterController`]. Stages 4–6 are not yet wired
//! through the controller; the state machine here captures the intended
//! transitions so future extensions plug in naturally.

use std::fmt;
use std::net::SocketAddr;

use tokio::sync::broadcast;

use super::commissioning::CommissioningPayload;

/// Discrete commissioning phases for an in-flight session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// QR / manual pairing code has been decoded.
    Parsed,
    /// The target device has been located on the network.
    Discovered,
    /// PASE (SPAKE2+) handshake has succeeded.
    PaseEstablished,
    /// Device returned its NOCSRElements (CSR + nonce).
    CsrReceived,
    /// `AddNOC` completed on the device; it is part of our fabric.
    NocInstalled,
    /// A CASE session is up — commissioning is complete.
    CaseEstablished,
    /// A stage failed. `&'static str` captures which stage + why.
    Failed(&'static str),
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phase::Parsed => f.write_str("parsed"),
            Phase::Discovered => f.write_str("discovered"),
            Phase::PaseEstablished => f.write_str("pase_established"),
            Phase::CsrReceived => f.write_str("csr_received"),
            Phase::NocInstalled => f.write_str("noc_installed"),
            Phase::CaseEstablished => f.write_str("case_established"),
            Phase::Failed(reason) => write!(f, "failed({reason})"),
        }
    }
}

/// A single transition event, broadcast to subscribers.
#[derive(Debug, Clone)]
pub struct CommissioningEvent {
    /// Phase entered.
    pub phase: Phase,
    /// Peer address once discovery has completed; `None` before that.
    pub peer: Option<SocketAddr>,
    /// Session id of the PASE/CASE session once established; `None` otherwise.
    pub session_id: Option<u16>,
}

/// In-flight commissioning session — tracks phase transitions and broadcasts
/// events to observers.
pub struct CommissioningSession {
    payload: CommissioningPayload,
    node_id: u64,
    phase: Phase,
    peer: Option<SocketAddr>,
    session_id: Option<u16>,
    tx: broadcast::Sender<CommissioningEvent>,
}

impl CommissioningSession {
    /// Start a new session in the `Parsed` phase.
    ///
    /// The session assigns `node_id` to the device once commissioning
    /// completes. Subscribers can observe every phase transition via
    /// [`Self::subscribe`].
    pub fn new(payload: CommissioningPayload, node_id: u64) -> Self {
        let (tx, _) = broadcast::channel(16);
        let session = Self {
            payload,
            node_id,
            phase: Phase::Parsed,
            peer: None,
            session_id: None,
            tx,
        };
        session.emit();
        session
    }

    /// Subscribe to phase transitions. Fresh subscribers do not receive the
    /// initial `Parsed` event — call [`Self::phase`] to read the current state.
    pub fn subscribe(&self) -> broadcast::Receiver<CommissioningEvent> {
        self.tx.subscribe()
    }

    /// Current phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Commissioning payload decoded from the QR / manual code.
    pub fn payload(&self) -> &CommissioningPayload {
        &self.payload
    }

    /// Assigned node id once `CaseEstablished`.
    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    /// Advance to the `Discovered` phase, recording the peer address.
    pub fn advance_discovered(&mut self, peer: SocketAddr) {
        self.peer = Some(peer);
        self.transition(Phase::Discovered);
    }

    /// Advance to `PaseEstablished`, recording the PASE session id.
    pub fn advance_pase_established(&mut self, session_id: u16) {
        self.session_id = Some(session_id);
        self.transition(Phase::PaseEstablished);
    }

    /// Advance to `CsrReceived`. The cluster-level CSR response is held by
    /// the caller; this signals that phase 4 completed successfully.
    pub fn advance_csr_received(&mut self) {
        self.transition(Phase::CsrReceived);
    }

    /// Advance to `NocInstalled` — `AddNOC` succeeded on the device.
    pub fn advance_noc_installed(&mut self) {
        self.transition(Phase::NocInstalled);
    }

    /// Advance to `CaseEstablished`, recording the operational session id.
    pub fn advance_case_established(&mut self, session_id: u16) {
        self.session_id = Some(session_id);
        self.transition(Phase::CaseEstablished);
    }

    /// Record a failure at the current phase. No further transitions are
    /// accepted after this.
    pub fn fail(&mut self, reason: &'static str) {
        self.phase = Phase::Failed(reason);
        self.emit();
    }

    fn transition(&mut self, to: Phase) {
        if matches!(self.phase, Phase::Failed(_)) {
            return;
        }
        self.phase = to;
        self.emit();
    }

    fn emit(&self) {
        let _ = self.tx.send(CommissioningEvent {
            phase: self.phase,
            peer: self.peer,
            session_id: self.session_id,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_payload() -> CommissioningPayload {
        CommissioningPayload {
            vendor_id: 0xFFF1,
            product_id: 0x8001,
            discriminator: 3840,
            passcode: 20202021,
            commissioning_flow: 0,
            rendezvous_info: 0x02,
        }
    }

    #[tokio::test]
    async fn session_starts_in_parsed() {
        let session = CommissioningSession::new(fake_payload(), 1);
        assert_eq!(session.phase(), Phase::Parsed);
        assert_eq!(session.node_id(), 1);
    }

    #[tokio::test]
    async fn happy_path_transitions() {
        let mut session = CommissioningSession::new(fake_payload(), 7);
        let mut events = session.subscribe();
        let peer: SocketAddr = "127.0.0.1:5540".parse().unwrap();
        session.advance_discovered(peer);
        session.advance_pase_established(0x1234);
        session.advance_csr_received();
        session.advance_noc_installed();
        session.advance_case_established(0x5678);

        // Collect broadcast events in order.
        let mut seen = Vec::new();
        while let Ok(ev) = events.try_recv() {
            seen.push(ev.phase);
        }
        assert_eq!(
            seen,
            vec![
                Phase::Discovered,
                Phase::PaseEstablished,
                Phase::CsrReceived,
                Phase::NocInstalled,
                Phase::CaseEstablished,
            ]
        );
        assert_eq!(session.phase(), Phase::CaseEstablished);
    }

    #[tokio::test]
    async fn failure_sticks() {
        let mut session = CommissioningSession::new(fake_payload(), 1);
        session.fail("simulated discovery timeout");
        assert!(matches!(session.phase(), Phase::Failed(_)));

        // Further transitions are ignored once failed.
        let peer: SocketAddr = "127.0.0.1:5540".parse().unwrap();
        session.advance_discovered(peer);
        assert!(matches!(session.phase(), Phase::Failed(_)));
    }
}
