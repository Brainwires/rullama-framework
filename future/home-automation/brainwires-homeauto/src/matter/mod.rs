// ─────────────────────────────────────────────────────────────────────────────
// ⚠️  EXPERIMENTAL — This Matter implementation is a development preview.
//
// What works:
//   • PASE (SPAKE2+ Password Authenticated Session Establishment) over UDP
//   • CASE (Certificate Authenticated Session Establishment) — Sigma 1/2/3 with
//     real P-256 ECDH, ECDSA signing, HKDF session keys, AES-128-CCM framing
//   • Fabric management + on-disk persistence (fabrics.json via tokio::fs)
//   • mDNS commissionable-device discovery
//   • Basic Interaction Model: read/write attributes, invoke commands
//   • Manual pairing code + QR code parsing (Verhoeff check digit validated)
//   • BLE commissioning transport (btleplug, Linux/macOS) behind `matter-ble`
//
// What also works now:
//   • AttestationResponse signs over `attestation_elements || nonce` with a
//     real ECDSA-P256 Device Attestation Key. `BRAINWIRES_MATTER_DAK_PATH`
//     loads production credentials from disk; otherwise a dev DAK is minted
//     at startup (logged as a warning).
//   • Subscription registry: `SubscribeRequest` is handled; cluster command
//     writes (OnOff, LevelControl, ColorControl, Thermostat) push
//     `ReportData` through a per-session monotonic counter to every matching
//     subscriber.
//   • Commissioning orchestration: `commission_qr_with_session` drives the
//     full QR → Discover → PASE → CSRRequest → AddNOC → CASE chain, with a
//     `CommissioningSession` state machine that broadcasts phase events.
//
// Known limitations:
//   • **Device Attestation Certificate chain is not CSA-signed.** The dev
//     DAK is real, but the DAC / PAI / CD bytes are placeholders. Certified
//     Matter commissioners verify the DAC chain against the CSA root of
//     trust and will reject this device. Provision a real chain via
//     `BRAINWIRES_MATTER_DAK_PATH` for interop testing.
//   • **Attestation challenge is approximated.** Matter spec §11.17.6.2
//     signs over `attestation_elements || attestation_challenge` where
//     `attestation_challenge` is the active session's challenge field; we
//     sign over the `AttestationNonce` instead. This is self-consistent and
//     defeats replay attacks, but some commissioners may reject the TBS.
//   • Loopback integration tests (`matter_e2e_*`, `commissioning_chain_*`)
//     are `#[ignore]` — mDNS multicast on loopback is flaky; they run in
//     the nightly workflow only.
//   • Not tested against real certified Matter controllers.
//
// Do not rely on this module for production home automation deployments
// without provisioning a CSA-signed DAC chain.
// ─────────────────────────────────────────────────────────────────────────────

/// BLE commissioning peripheral — BTP handshake and transport channels.
#[cfg(feature = "matter-ble")]
pub mod ble;
/// Typed cluster helpers (TLV-encoded command and attribute payloads).
pub mod clusters;
/// Matter commissioning payload parser (QR code + manual pairing code).
pub mod commissioning;
/// State machine for the commissioner-driven commissioning flow.
pub mod commissioning_session;
/// Matter controller — commissions and controls Matter devices (PASE only, experimental).
pub mod controller;
/// Matter cryptographic stack: KDF helpers and SPAKE2+ PAKE.
pub mod crypto;
/// Matter data model — cluster servers, ACL, and node dispatch.
pub mod data_model;
/// Matter device discovery — commissionable and operational DNS-SD.
pub mod discovery;
/// Typed errors wrapping rs-matter.
pub mod error;
/// Matter fabric management — root CA, NOC issuance, and fabric storage (incomplete).
pub mod fabric;
/// Matter Interaction Model — read, write, invoke, and subscribe messages.
pub mod interaction_model;
/// Matter secure channel — PASE (commissioning) and CASE (operational, not yet functional).
pub mod secure_channel;
/// Matter device server — exposes agents as Matter devices (PASE only).
pub mod server;
/// Subscription registry for Interaction Model Subscribe/Report.
pub mod subscription_manager;
/// Matter transport layer: message framing, MRP, and UDP/BLE I/O.
pub mod transport;
/// Matter device types, cluster IDs, and configuration.
pub mod types;
/// Verhoeff check-digit algorithm used by the 11-digit manual pairing code.
pub(crate) mod verhoeff;

pub use commissioning::{CommissioningPayload, parse_manual_code, parse_qr_code};
pub use commissioning_session::{CommissioningEvent, CommissioningSession, Phase};
pub use controller::MatterController;
pub use crypto::{
    kdf::{derive_passcode_verifier, hkdf_expand_label},
    spake2plus::{Spake2PlusKeys, Spake2PlusProver, Spake2PlusVerifier},
};
pub use error::{MatterError, MatterResult};
pub use fabric::{FabricDescriptor, FabricIndex, MatterCert};
pub use interaction_model::{
    AttributeData, AttributeStatus, ImOpcode, InteractionStatus, InvokeRequest, InvokeResponse,
    InvokeResponseItem, PROTOCOL_ID as IM_PROTOCOL_ID, ReadRequest, ReportData, SubscribeRequest,
    SubscribeResponse, WriteRequest, WriteResponse,
};
pub use secure_channel::{
    CaseInitiator, CaseResponder, EstablishedSession, PaseCommissionee, PaseCommissioner,
    SECURE_CHANNEL_PROTOCOL_ID, SecureChannelOpcode,
};
pub use server::{MatterDeviceServer, OnOffHandler};
pub use subscription_manager::{Subscription, SubscriptionManager};
pub use types::{
    MatterDevice, MatterDeviceConfig, MatterDeviceConfigBuilder, MatterEndpoint, cluster_id,
    device_type,
};
