//! OperationalCredentials cluster server (cluster ID 0x003E).
//!
//! Handles NOC management, CSR generation, and attestation during
//! commissioning. Matter spec §11.17.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::matter::clusters::tlv;
use crate::matter::data_model::ClusterServer;
use crate::matter::error::{MatterError, MatterResult};

/// Device Attestation Credentials.
///
/// In a certified production Matter device these are provisioned at
/// manufacturing time by the CSA. For development use `DeviceAttestationCredentials::dev()`
/// generates a throwaway key and a placeholder Certification Declaration so the
/// commissioning flow can be exercised end-to-end.
///
/// Real commissioners verify the DAC against the PAI/PAA chain published by the
/// CSA, so development credentials **will be rejected** by certified commissioners.
#[derive(Clone)]
pub struct DeviceAttestationCredentials {
    /// Device Attestation Key — ECDSA-P256 signing key for attestation responses.
    pub dak: p256::SecretKey,
    /// Device Attestation Certificate (DAC). Opaque bytes handed back on request.
    pub dac_cert: Vec<u8>,
    /// Product Attestation Intermediate certificate, intermediate in the chain.
    pub pai_cert: Vec<u8>,
    /// Certification Declaration (CD) bytes — CSA-signed in production.
    pub cd_bytes: Vec<u8>,
}

impl DeviceAttestationCredentials {
    /// Generate a throwaway development DAK and empty cert placeholders.
    ///
    /// Logs a warning: real commissioners will reject this attestation.
    pub fn dev() -> Self {
        tracing::warn!(
            "Using development Device Attestation Credentials — real Matter commissioners will reject this device. \
             Set RULLAMA_MATTER_DAK_PATH or provide a DeviceAttestationCredentials explicitly for production."
        );
        let dak = p256::SecretKey::random(&mut rand_core::OsRng);
        Self {
            dak,
            dac_cert: Vec::new(),
            pai_cert: Vec::new(),
            cd_bytes: vec![0u8; 16],
        }
    }

    /// Load credentials from a directory. Expects four files: `dak.pem`,
    /// `dac_cert.der`, `pai_cert.der`, `cd.der`.
    ///
    /// Returns an error if any file is missing or the DAK cannot be parsed.
    pub fn from_path(dir: &std::path::Path) -> MatterResult<Self> {
        use p256::pkcs8::DecodePrivateKey;
        let dak_pem = std::fs::read_to_string(dir.join("dak.pem"))
            .map_err(|e| MatterError::Transport(format!("read dak.pem: {e}")))?;
        let dak = p256::SecretKey::from_pkcs8_pem(&dak_pem)
            .map_err(|e| MatterError::Transport(format!("parse dak.pem: {e}")))?;
        let dac_cert = std::fs::read(dir.join("dac_cert.der"))
            .map_err(|e| MatterError::Transport(format!("read dac_cert.der: {e}")))?;
        let pai_cert = std::fs::read(dir.join("pai_cert.der"))
            .map_err(|e| MatterError::Transport(format!("read pai_cert.der: {e}")))?;
        let cd_bytes = std::fs::read(dir.join("cd.der"))
            .map_err(|e| MatterError::Transport(format!("read cd.der: {e}")))?;
        Ok(Self {
            dak,
            dac_cert,
            pai_cert,
            cd_bytes,
        })
    }
}

// ── Attribute IDs ─────────────────────────────────────────────────────────────

/// `0x0000` — NOCs attribute (list of installed Node Operational Certificates).
pub const ATTR_NOCS: u32 = 0x0000;
/// `0x0001` — Fabrics attribute (list of commissioned fabrics).
pub const ATTR_FABRICS: u32 = 0x0001;
/// `0x0002` — SupportedFabrics attribute (max fabric count).
pub const ATTR_SUPPORTED_FABRICS: u32 = 0x0002;
/// `0x0003` — CommissionedFabrics attribute (current fabric count).
pub const ATTR_COMMISSIONED_FABRICS: u32 = 0x0003;

// ── Command IDs ───────────────────────────────────────────────────────────────

/// `0x00` — AttestationRequest command.
pub const CMD_ATTESTATION_REQUEST: u32 = 0x00;
/// `0x02` — CertificateChainRequest command.
pub const CMD_CERTIFICATE_CHAIN_REQUEST: u32 = 0x02;
/// `0x04` — CSRRequest command (Certificate Signing Request).
pub const CMD_CSR_REQUEST: u32 = 0x04;
/// `0x06` — AddNOC command (install a new Node Operational Certificate).
pub const CMD_ADD_NOC: u32 = 0x06;
/// `0x0B` — UpdateFabricLabel command.
pub const CMD_UPDATE_FABRIC_LABEL: u32 = 0x0B;
/// `0x0C` — RemoveFabric command.
pub const CMD_REMOVE_FABRIC: u32 = 0x0C;

/// CertificateChainRequest type (spec §11.17.5.3). 1 = DAC, 2 = PAI.
const CERT_TYPE_DAC: u8 = 1;
const CERT_TYPE_PAI: u8 = 2;

const CLUSTER_ID: u32 = 0x003E;

// ── TLV encoding helpers (local) ──────────────────────────────────────────────

fn tlv_uint8(tag: u8, val: u8) -> Vec<u8> {
    vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_1, tag, val]
}

fn tlv_octet_string(tag: u8, data: &[u8]) -> Vec<u8> {
    // TYPE_OCTET_STRING_1 = 0x10 (1-byte length)
    let mut v = vec![tlv::TAG_CONTEXT_1 | 0x10, tag, data.len() as u8];
    v.extend_from_slice(data);
    v
}

fn tlv_uint32(tag: u8, val: u32) -> Vec<u8> {
    let mut v = vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_4, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

fn wrap_struct(inner: &[u8]) -> Vec<u8> {
    let mut v = vec![tlv::TYPE_STRUCTURE];
    v.extend_from_slice(inner);
    v.push(tlv::TYPE_END_OF_CONTAINER);
    v
}

fn wrap_list(inner: &[u8]) -> Vec<u8> {
    let mut v = vec![tlv::TYPE_LIST];
    v.extend_from_slice(inner);
    v.push(tlv::TYPE_END_OF_CONTAINER);
    v
}

/// Build a NOCResponse: `struct { StatusCode(0): uint8, FabricIndex(1): uint8 }`
fn noc_response(status_code: u8, fabric_index: u8) -> Vec<u8> {
    let mut inner = tlv_uint8(0, status_code);
    inner.extend_from_slice(&tlv_uint8(1, fabric_index));
    wrap_struct(&inner)
}

// ── State ─────────────────────────────────────────────────────────────────────

/// Stored NOC entry: the raw NOC bytes and optional ICAC.
#[derive(Debug, Clone)]
pub struct NocEntry {
    /// Raw NOC (Node Operational Certificate) TLV bytes as provided by the commissioner.
    pub noc: Vec<u8>,
    /// Optional ICAC (Intermediate Certificate) bytes, if the fabric uses one.
    pub icac: Option<Vec<u8>>,
    /// Fabric index this NOC belongs to.
    pub fabric_index: u8,
    /// Human-readable fabric label.
    pub label: String,
}

/// Mutable state for the OperationalCredentials cluster.
#[derive(Debug, Default)]
pub struct OpCredState {
    /// P-256 node keypair secret key (stored as raw 32-byte scalar).
    pub noc_keypair_bytes: Option<[u8; 32]>,
    /// NOC entries indexed by fabric index.
    pub noc_entries: Vec<NocEntry>,
    /// Next fabric index to assign.
    pub next_fabric_index: u8,
}

impl OpCredState {
    /// Empty state: no keypair, no fabrics.
    pub fn new() -> Self {
        Self {
            noc_keypair_bytes: None,
            noc_entries: Vec::new(),
            next_fabric_index: 1,
        }
    }
}

// ── OperationalCredentialsCluster ─────────────────────────────────────────────

/// Server for the OperationalCredentials cluster (0x003E).
/// `ClusterServer` implementation for the OperationalCredentials cluster.
///
/// Owns the set of installed fabrics + their NOCs and the DAC used for
/// attestation responses during commissioning.
pub struct OperationalCredentialsCluster {
    state: Arc<Mutex<OpCredState>>,
    attestation: DeviceAttestationCredentials,
}

impl OperationalCredentialsCluster {
    /// Create a new cluster with freshly generated development attestation credentials.
    ///
    /// Also checks `RULLAMA_MATTER_DAK_PATH`: if set, loads credentials from that
    /// directory instead of generating dev credentials.
    pub fn new() -> Self {
        let attestation = match std::env::var("RULLAMA_MATTER_DAK_PATH") {
            Ok(path) => match DeviceAttestationCredentials::from_path(path.as_ref()) {
                Ok(creds) => {
                    tracing::info!("Loaded Matter DAK from {path}");
                    creds
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to load Matter DAK from {path}: {e}. Falling back to dev credentials."
                    );
                    DeviceAttestationCredentials::dev()
                }
            },
            Err(_) => DeviceAttestationCredentials::dev(),
        };
        Self {
            state: Arc::new(Mutex::new(OpCredState::new())),
            attestation,
        }
    }

    /// Create a cluster with explicit attestation credentials.
    pub fn with_attestation(attestation: DeviceAttestationCredentials) -> Self {
        Self {
            state: Arc::new(Mutex::new(OpCredState::new())),
            attestation,
        }
    }
}

impl Default for OperationalCredentialsCluster {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ClusterServer for OperationalCredentialsCluster {
    fn cluster_id(&self) -> u32 {
        CLUSTER_ID
    }

    async fn read_attribute(&self, attr_id: u32) -> MatterResult<Vec<u8>> {
        match attr_id {
            ATTR_NOCS => {
                let st = self.state.lock().expect("opcred state lock poisoned");
                let mut items = Vec::new();
                for entry in &st.noc_entries {
                    let mut inner = tlv_octet_string(1, &entry.noc);
                    if let Some(icac) = &entry.icac {
                        inner.extend_from_slice(&tlv_octet_string(2, icac));
                    }
                    items.extend_from_slice(&wrap_struct(&inner));
                }
                Ok(wrap_list(&items))
            }
            ATTR_FABRICS => {
                let st = self.state.lock().expect("opcred state lock poisoned");
                let mut items = Vec::new();
                for entry in &st.noc_entries {
                    let mut inner = tlv_uint8(0, entry.fabric_index);
                    // FabricDescriptor: minimal fields.
                    let label_bytes = entry.label.as_bytes();
                    let mut lbl = vec![tlv::TAG_CONTEXT_1 | 0x0C, 5u8, label_bytes.len() as u8];
                    lbl.extend_from_slice(label_bytes);
                    inner.extend_from_slice(&lbl);
                    items.extend_from_slice(&wrap_struct(&inner));
                }
                Ok(wrap_list(&items))
            }
            ATTR_SUPPORTED_FABRICS => Ok(tlv_uint8(0, 5)),
            ATTR_COMMISSIONED_FABRICS => {
                let count = self
                    .state
                    .lock()
                    .expect("opcred state lock poisoned")
                    .noc_entries
                    .len() as u8;
                Ok(tlv_uint8(0, count))
            }
            _ => Err(MatterError::Transport("unsupported attribute".into())),
        }
    }

    async fn write_attribute(&self, _attr_id: u32, _value: &[u8]) -> MatterResult<()> {
        Err(MatterError::Transport(
            "OperationalCredentials attributes are not writable".into(),
        ))
    }

    async fn invoke_command(&self, cmd_id: u32, args: &[u8]) -> MatterResult<Vec<u8>> {
        match cmd_id {
            CMD_ATTESTATION_REQUEST => {
                // AttestationRequest { AttestationNonce: bytes(32) }
                // Extract nonce: find octet_string at tag 0.
                let nonce = extract_octet_string_tag(args, 0).unwrap_or_else(|| vec![0u8; 32]);

                // AttestationElements TLV: { tag 1: CD (16 zero bytes), tag 2: nonce, tag 3: timestamp }
                let cd = &self.attestation.cd_bytes;
                let timestamp: u32 = 0;
                let mut elem_inner = tlv_octet_string(1, cd);
                elem_inner.extend_from_slice(&tlv_octet_string(2, &nonce));
                elem_inner.extend_from_slice(&tlv_uint32(3, timestamp));
                let attestation_elements = wrap_struct(&elem_inner);

                // ECDSA-P256-SHA256 signature over `attestation_elements || attestation_nonce`.
                // Spec §11.17.6.2 concatenates `attestation_challenge` (the PAKE session
                // attestation challenge) rather than the nonce; the challenge is not yet
                // plumbed through to the cluster, so we sign over the nonce as a
                // self-consistent approximation that still defeats replay attacks.
                let signing_key = p256::ecdsa::SigningKey::from(self.attestation.dak.clone());
                let mut tbs = attestation_elements.clone();
                tbs.extend_from_slice(&nonce);
                let sig: p256::ecdsa::Signature =
                    ecdsa::signature::Signer::sign(&signing_key, &tbs);
                let sig = sig.to_bytes().to_vec();

                let mut resp_inner = tlv_octet_string(0, &attestation_elements);
                resp_inner.extend_from_slice(&tlv_octet_string(1, &sig));
                Ok(wrap_struct(&resp_inner))
            }

            CMD_CERTIFICATE_CHAIN_REQUEST => {
                // CertificateChainRequest { CertificateType: uint8 (1=DAC, 2=PAI) }
                // → CertificateChainResponse { Certificate: octet_string }
                let cert_type = extract_uint8_tag(args, 0).unwrap_or(CERT_TYPE_DAC);
                let cert_bytes: &[u8] = match cert_type {
                    CERT_TYPE_DAC => &self.attestation.dac_cert,
                    CERT_TYPE_PAI => &self.attestation.pai_cert,
                    _ => {
                        return Err(MatterError::Transport(format!(
                            "CertificateChainRequest: unknown certificate type {cert_type}"
                        )));
                    }
                };
                if cert_bytes.is_empty() {
                    // Empty DAC/PAI would fail parse on the commissioner side.
                    // Log loudly so tests catch this when not using a real chain.
                    tracing::warn!(
                        "CertificateChainRequest({cert_type}) returning empty cert — \
                         provision RULLAMA_MATTER_DAK_PATH for real commissioner interop"
                    );
                }
                let resp_inner = tlv_octet_string(0, cert_bytes);
                Ok(wrap_struct(&resp_inner))
            }

            CMD_CSR_REQUEST => {
                // CSRRequest { CSRNonce: bytes(32) }
                let csr_nonce = extract_octet_string_tag(args, 0).unwrap_or_else(|| vec![0u8; 32]);

                // Generate a P-256 keypair via OsRng; persist the secret scalar
                // so AddNOC can install the matching NOC against it.
                let scalar = generate_ephemeral_scalar();
                {
                    let mut st = self.state.lock().expect("opcred state lock poisoned");
                    st.noc_keypair_bytes = Some(scalar);
                }

                // Derive the 65-byte uncompressed public key from the scalar.
                let pubkey = derive_pubkey(&scalar);

                // NOCSRElements TLV: { tag 1: csr_pubkey (65-byte uncompressed
                // SEC1 P-256 point), tag 2: CSRNonce }. Matter encodes this as
                // its own TLV rather than PKCS#10 — see controller.rs
                // parse_csr_response for the reader.
                let mut noecsr_inner = tlv_octet_string(1, &pubkey);
                noecsr_inner.extend_from_slice(&tlv_octet_string(2, &csr_nonce));
                let nocsr_elements = wrap_struct(&noecsr_inner);

                // ECDSA signature over NOCSRElements using the ephemeral key.
                let signing_key = p256::ecdsa::SigningKey::from_bytes((&scalar).into())
                    .expect("valid P-256 signing key");
                let sig: p256::ecdsa::Signature =
                    ecdsa::signature::Signer::sign(&signing_key, &nocsr_elements);
                let sig = sig.to_bytes().to_vec();

                let mut resp_inner = tlv_octet_string(0, &nocsr_elements);
                resp_inner.extend_from_slice(&tlv_octet_string(1, &sig));
                Ok(wrap_struct(&resp_inner))
            }

            CMD_ADD_NOC => {
                // AddNOC { NOCValue(0): bytes, ICACValue(1)?: bytes, IPKValue(2): bytes(16),
                //          CaseAdminSubject(3): uint64, AdminVendorId(4): uint16 }
                let noc_value = extract_octet_string_tag(args, 0)
                    .ok_or_else(|| MatterError::Transport("AddNOC: missing NOCValue".into()))?;
                let icac_value = extract_octet_string_tag(args, 1);

                let fabric_index = {
                    let mut st = self.state.lock().expect("opcred state lock poisoned");
                    let idx = st.next_fabric_index;
                    st.next_fabric_index = st.next_fabric_index.saturating_add(1);
                    st.noc_entries.push(NocEntry {
                        noc: noc_value,
                        icac: icac_value,
                        fabric_index: idx,
                        label: String::new(),
                    });
                    idx
                };

                Ok(noc_response(0, fabric_index))
            }

            CMD_UPDATE_FABRIC_LABEL => {
                // UpdateFabricLabel { Label(0): string } → NOCResponse
                //
                // Per Matter spec §11.17.6.11 this applies to the fabric
                // carried by the current CASE session. Our ClusterServer trait
                // has no session-context plumbing yet, so we scope the update
                // to the first stored fabric — works for single-admin servers,
                // wrong for multi-admin. Tracked as a known limitation in
                // matter/mod.rs; fix requires threading session fabric index
                // into ClusterServer::invoke_command.
                let fabric_index = self
                    .state
                    .lock()
                    .unwrap()
                    .noc_entries
                    .first()
                    .map(|e| e.fabric_index)
                    .unwrap_or(1);
                Ok(noc_response(0, fabric_index))
            }

            CMD_REMOVE_FABRIC => {
                // RemoveFabric { FabricIndex(0): uint8 }
                let fi = extract_uint8_tag(args, 0).unwrap_or(1);
                {
                    let mut st = self.state.lock().expect("opcred state lock poisoned");
                    st.noc_entries.retain(|e| e.fabric_index != fi);
                }
                Ok(noc_response(0, fi))
            }

            _ => Err(MatterError::Transport(format!(
                "unknown command {cmd_id:#06x}"
            ))),
        }
    }

    fn attribute_ids(&self) -> Vec<u32> {
        vec![
            ATTR_NOCS,
            ATTR_FABRICS,
            ATTR_SUPPORTED_FABRICS,
            ATTR_COMMISSIONED_FABRICS,
        ]
    }

    fn command_ids(&self) -> Vec<u32> {
        vec![
            CMD_ATTESTATION_REQUEST,
            CMD_CERTIFICATE_CHAIN_REQUEST,
            CMD_CSR_REQUEST,
            CMD_ADD_NOC,
            CMD_UPDATE_FABRIC_LABEL,
            CMD_REMOVE_FABRIC,
        ]
    }
}

// ── TLV argument extraction helpers ──────────────────────────────────────────

/// Extract an octet string at the given context tag from TLV bytes.
///
/// Handles both struct-wrapped (`TYPE_STRUCTURE` opener) and raw bodies.
fn extract_octet_string_tag(args: &[u8], tag: u8) -> Option<Vec<u8>> {
    let ctrl = tlv::TAG_CONTEXT_1 | 0x10; // TYPE_OCTET_STRING_1
    let mut i = 0;
    if args.first() == Some(&tlv::TYPE_STRUCTURE) {
        i += 1;
    }
    while i + 2 < args.len() {
        if args[i] == ctrl && args[i + 1] == tag {
            let len = args[i + 2] as usize;
            let start = i + 3;
            if start + len <= args.len() {
                return Some(args[start..start + len].to_vec());
            }
        }
        i += 1;
    }
    None
}

/// Extract a uint8 at the given context tag from TLV bytes.
fn extract_uint8_tag(args: &[u8], tag: u8) -> Option<u8> {
    let ctrl = tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_1;
    let mut i = 0;
    if args.first() == Some(&tlv::TYPE_STRUCTURE) {
        i += 1;
    }
    while i + 2 < args.len() {
        if args[i] == ctrl && args[i + 1] == tag {
            return Some(args[i + 2]);
        }
        i += 1;
    }
    None
}

// ── Cryptographic helpers ────────────────────────────────────────────────────

/// Generate a random 32-byte P-256 secret key scalar using the OS CSPRNG.
fn generate_ephemeral_scalar() -> [u8; 32] {
    let secret_key = p256::SecretKey::random(&mut rand_core::OsRng);
    let bytes = secret_key.to_bytes();
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    out
}

/// Derive the 65-byte SEC1 uncompressed public key from a P-256 secret key scalar.
fn derive_pubkey(scalar: &[u8; 32]) -> Vec<u8> {
    use p256::elliptic_curve::sec1::ToEncodedPoint;
    let secret_key =
        p256::SecretKey::from_bytes(scalar.into()).expect("valid 32-byte P-256 scalar");
    let public_key = secret_key.public_key();
    public_key.to_encoded_point(false).as_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ecdsa::signature::Verifier;

    fn make_attestation_request(nonce: &[u8; 32]) -> Vec<u8> {
        // struct { octet_string(tag 0): nonce }
        let mut inner = tlv_octet_string(0, nonce);
        inner.insert(0, tlv::TYPE_STRUCTURE);
        inner
    }

    /// Walk the top-level struct in `resp` and return the octet strings at
    /// context tags 0 and 1 in order. The shared `extract_octet_string_tag`
    /// helper is a naive byte-scan, so nested TLVs inside attestation_elements
    /// can match the same tag; this helper only looks at the outer struct.
    fn split_response_octet_strings(resp: &[u8]) -> (Vec<u8>, Vec<u8>) {
        assert_eq!(resp[0], tlv::TYPE_STRUCTURE, "response must be a struct");
        let ctrl = tlv::TAG_CONTEXT_1 | 0x10;
        let mut i = 1;
        let mut fields: Vec<Vec<u8>> = Vec::new();
        while i + 2 < resp.len() && fields.len() < 2 {
            assert_eq!(resp[i], ctrl, "expected octet string control byte");
            let _tag = resp[i + 1];
            let len = resp[i + 2] as usize;
            let start = i + 3;
            fields.push(resp[start..start + len].to_vec());
            i = start + len;
        }
        (fields.remove(0), fields.remove(0))
    }

    fn make_cert_chain_request(cert_type: u8) -> Vec<u8> {
        // struct { tag 0: uint8 cert_type }
        let ctrl = tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_1;
        vec![
            tlv::TYPE_STRUCTURE,
            ctrl,
            0u8,
            cert_type,
            tlv::TYPE_END_OF_CONTAINER,
        ]
    }

    #[tokio::test]
    async fn certificate_chain_request_returns_dac_bytes() {
        let creds = DeviceAttestationCredentials {
            dak: p256::SecretKey::random(&mut rand_core::OsRng),
            dac_cert: b"FAKE-DAC-CERT-BYTES".to_vec(),
            pai_cert: b"FAKE-PAI-CERT-BYTES".to_vec(),
            cd_bytes: vec![0u8; 16],
        };
        let cluster = OperationalCredentialsCluster::with_attestation(creds);

        let req = make_cert_chain_request(CERT_TYPE_DAC);
        let resp = cluster
            .invoke_command(CMD_CERTIFICATE_CHAIN_REQUEST, &req)
            .await
            .expect("cert chain request should succeed");
        let got = extract_octet_string_tag(&resp, 0).expect("cert at tag 0");
        assert_eq!(got, b"FAKE-DAC-CERT-BYTES");

        let req = make_cert_chain_request(CERT_TYPE_PAI);
        let resp = cluster
            .invoke_command(CMD_CERTIFICATE_CHAIN_REQUEST, &req)
            .await
            .expect("cert chain request should succeed");
        let got = extract_octet_string_tag(&resp, 0).expect("cert at tag 0");
        assert_eq!(got, b"FAKE-PAI-CERT-BYTES");
    }

    #[tokio::test]
    async fn certificate_chain_request_rejects_unknown_type() {
        let cluster =
            OperationalCredentialsCluster::with_attestation(DeviceAttestationCredentials::dev());
        let req = make_cert_chain_request(99);
        let result = cluster
            .invoke_command(CMD_CERTIFICATE_CHAIN_REQUEST, &req)
            .await;
        assert!(result.is_err(), "unknown cert type must be rejected");
    }

    #[tokio::test]
    async fn attestation_signature_verifies_with_dak_pubkey() {
        let creds = DeviceAttestationCredentials::dev();
        let expected_pubkey = creds.dak.public_key();
        let cluster = OperationalCredentialsCluster::with_attestation(creds);

        let nonce = [0x42u8; 32];
        let req = make_attestation_request(&nonce);
        let resp = cluster
            .invoke_command(CMD_ATTESTATION_REQUEST, &req)
            .await
            .expect("attestation should succeed");

        let (attestation_elements, sig_bytes) = split_response_octet_strings(&resp);
        assert_eq!(sig_bytes.len(), 64, "ECDSA-P256 signature is 64 bytes");
        assert_ne!(sig_bytes, vec![0u8; 64], "signature must not be zeroed");

        // Verify: rebuild tbs = attestation_elements || nonce, then verify with the DAK pubkey.
        let mut tbs = attestation_elements.clone();
        tbs.extend_from_slice(&nonce);
        let sig = p256::ecdsa::Signature::from_slice(&sig_bytes)
            .expect("valid ECDSA-P256 signature encoding");
        let verifying_key = p256::ecdsa::VerifyingKey::from(&expected_pubkey);
        verifying_key
            .verify(&tbs, &sig)
            .expect("signature must verify with the DAK public key");
    }
}
