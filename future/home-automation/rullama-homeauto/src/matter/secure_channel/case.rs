/// CASE (Certificate Authenticated Session Establishment) — Matter spec §4.13 (SIGMA).
///
/// Uses ECDH (P-256) + HKDF + ECDSA to establish an operational session.
///
/// # Message flow
/// ```text
/// Initiator                                          Responder
/// ─────────                                         ──────────
/// build_sigma1() ─────────Sigma1──────────────────>
///                <────────Sigma2───────────────────  handle_sigma1()
/// handle_sigma2() ─────────Sigma3─────────────────>
///                                  handle_sigma3() → EstablishedSession
/// established_session()
/// ```
///
/// # Key derivation
///
/// After Sigma3 both sides derive:
/// ```text
/// ECDH shared_secret = P-256 DH(our_eph_secret, peer_eph_pub)
/// I2RKey‖R2IKey‖AttnChallenge = HKDF(shared_secret,
///                                     init_random‖resp_random,
///                                     "SessionKeys") → 48 bytes
/// ```
///
/// # Encryption of Sigma2/Sigma3 payloads
///
/// Sigma2 inner block (Encrypted2):
///   key  = HKDF(shared_secret, init_random‖resp_random, "Sigma2")  [16 bytes]
///   nonce = b"NCASE_Sigma2N" (13 bytes)
///   plaintext = TLV { tag1: noc, (tag2: icac), tag3: TBEData2Signature }
///
/// Sigma3 inner block (Encrypted3):
///   key  = HKDF(shared_secret, init_random‖resp_random, "Sigma3")  [16 bytes]
///   nonce = b"NCASE_Sigma3N" (13 bytes)
///   plaintext = TLV { tag1: noc, (tag2: icac), tag3: TBEData3Signature }
///
/// TBEData2Signature = ECDSA-SHA256(responder_node_key, init_eph_pub ‖ resp_eph_pub)
/// TBEData3Signature = ECDSA-SHA256(initiator_node_key, init_eph_pub ‖ resp_eph_pub)
use p256::{
    PublicKey, SecretKey, ecdh,
    ecdsa::{Signature, SigningKey, VerifyingKey, signature::Signer},
    elliptic_curve::sec1::ToEncodedPoint,
};
use rand_core::{OsRng, RngCore};

use aes::Aes128;
use ccm::{
    Ccm,
    aead::{Aead, KeyInit, generic_array::GenericArray},
    consts::{U13, U16},
};

use super::EstablishedSession;
use crate::matter::crypto::kdf::hkdf_expand_label;
use crate::matter::error::{MatterError, MatterResult};
use crate::matter::fabric::{FabricDescriptor, MatterCert};

// ── AES-128-CCM type alias ────────────────────────────────────────────────────

type Aes128Ccm = Ccm<Aes128, U16, U13>;

// ── Sigma nonces ─────────────────────────────────────────────────────────────

const SIGMA2_NONCE: &[u8; 13] = b"NCASE_Sigma2N";
const SIGMA3_NONCE: &[u8; 13] = b"NCASE_Sigma3N";

// ── TLV helpers ───────────────────────────────────────────────────────────────

const CTX_TAG: u8 = 1 << 5;
const UINT2_TYPE: u8 = 0x05;
const BYTES1_TYPE: u8 = 0x10; // octet string, 1-byte length
const BYTES2_TYPE: u8 = 0x11; // octet string, 2-byte length
const STRUCT_TYPE: u8 = 0x15;
const END_TYPE: u8 = 0x18;

fn tlv_ctx_uint2(tag: u8, val: u16) -> Vec<u8> {
    let mut v = vec![CTX_TAG | UINT2_TYPE, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

/// Encode context-tagged bytes with automatic 1-byte or 2-byte length selection.
fn tlv_ctx_bytes(tag: u8, data: &[u8]) -> Vec<u8> {
    if data.len() <= 255 {
        let mut v = vec![CTX_TAG | BYTES1_TYPE, tag, data.len() as u8];
        v.extend_from_slice(data);
        v
    } else {
        assert!(
            data.len() <= 65535,
            "TLV bytes: length > 65535 not supported"
        );
        let mut v = vec![CTX_TAG | BYTES2_TYPE, tag];
        v.extend_from_slice(&(data.len() as u16).to_le_bytes());
        v.extend_from_slice(data);
        v
    }
}

fn tlv_anon_struct(inner: &[u8]) -> Vec<u8> {
    let mut v = vec![STRUCT_TYPE];
    v.extend_from_slice(inner);
    v.push(END_TYPE);
    v
}

// ── TLV decoder ───────────────────────────────────────────────────────────────

struct TlvReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> TlvReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn read_byte(&mut self) -> MatterResult<u8> {
        if self.pos >= self.buf.len() {
            return Err(MatterError::Protocol {
                opcode: 0,
                msg: "TLV: unexpected end".into(),
            });
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_slice(&mut self, n: usize) -> MatterResult<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(MatterError::Protocol {
                opcode: 0,
                msg: format!("TLV: need {n} bytes, have {}", self.buf.len() - self.pos),
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_element(&mut self) -> MatterResult<TlvElem<'a>> {
        if self.remaining() == 0 {
            return Err(MatterError::Protocol {
                opcode: 0,
                msg: "TLV: buffer exhausted".into(),
            });
        }
        let ctrl = self.read_byte()?;
        let tag_type = (ctrl >> 5) & 0x07;
        let val_type = ctrl & 0x1f;

        let tag: Option<u8> = match tag_type {
            0 => None,
            1 => Some(self.read_byte()?),
            _ => {
                return Err(MatterError::Protocol {
                    opcode: 0,
                    msg: format!("TLV: unsupported tag type {tag_type}"),
                });
            }
        };

        match val_type {
            0x04 => {
                let b = self.read_byte()?;
                Ok(TlvElem {
                    tag,
                    value: TlvVal::Uint(b as u64),
                })
            }
            0x05 => {
                let b = self.read_slice(2)?;
                Ok(TlvElem {
                    tag,
                    value: TlvVal::Uint(u16::from_le_bytes([b[0], b[1]]) as u64),
                })
            }
            0x06 => {
                let b = self.read_slice(4)?;
                Ok(TlvElem {
                    tag,
                    value: TlvVal::Uint(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64),
                })
            }
            0x07 => {
                let b = self.read_slice(8)?;
                Ok(TlvElem {
                    tag,
                    value: TlvVal::Uint(u64::from_le_bytes(b.try_into().unwrap())),
                })
            }
            0x10 => {
                let len = self.read_byte()? as usize;
                let data = self.read_slice(len)?;
                Ok(TlvElem {
                    tag,
                    value: TlvVal::Bytes(data),
                })
            }
            // bytes 2-byte length (for payloads > 255 bytes, e.g. Encrypted2)
            0x11 => {
                let b = self.read_slice(2)?;
                let len = u16::from_le_bytes([b[0], b[1]]) as usize;
                let data = self.read_slice(len)?;
                Ok(TlvElem {
                    tag,
                    value: TlvVal::Bytes(data),
                })
            }
            0x15 => Ok(TlvElem {
                tag,
                value: TlvVal::StructStart,
            }),
            0x18 => Ok(TlvElem {
                tag,
                value: TlvVal::End,
            }),
            _ => Err(MatterError::Protocol {
                opcode: 0,
                msg: format!("TLV: unsupported val type {val_type:#04x}"),
            }),
        }
    }
}

struct TlvElem<'a> {
    tag: Option<u8>,
    value: TlvVal<'a>,
}

enum TlvVal<'a> {
    Uint(u64),
    Bytes(&'a [u8]),
    StructStart,
    End,
}

// ── Sigma2/3 encrypted payload helpers ───────────────────────────────────────

/// Encrypt an inner block with AES-128-CCM.
///
/// key: 16 bytes, nonce: 13 bytes.
fn aes128_ccm_encrypt(key: &[u8; 16], nonce: &[u8; 13], plaintext: &[u8]) -> MatterResult<Vec<u8>> {
    let cipher = Aes128Ccm::new(GenericArray::from_slice(key));
    let nonce_arr = GenericArray::from_slice(nonce);
    cipher
        .encrypt(
            nonce_arr,
            ccm::aead::Payload {
                msg: plaintext,
                aad: &[],
            },
        )
        .map_err(|_| MatterError::Crypto("AES-128-CCM encrypt failed".into()))
}

/// Decrypt an inner block with AES-128-CCM.
fn aes128_ccm_decrypt(
    key: &[u8; 16],
    nonce: &[u8; 13],
    ciphertext: &[u8],
) -> MatterResult<Vec<u8>> {
    let cipher = Aes128Ccm::new(GenericArray::from_slice(key));
    let nonce_arr = GenericArray::from_slice(nonce);
    cipher
        .decrypt(
            nonce_arr,
            ccm::aead::Payload {
                msg: ciphertext,
                aad: &[],
            },
        )
        .map_err(|_| MatterError::Crypto("AES-128-CCM decrypt/verify failed".into()))
}

// ── Session key derivation ────────────────────────────────────────────────────

/// salt = init_random ‖ resp_random (64 bytes)
fn make_sigma_salt(init_random: &[u8; 32], resp_random: &[u8; 32]) -> Vec<u8> {
    let mut s = Vec::with_capacity(64);
    s.extend_from_slice(init_random);
    s.extend_from_slice(resp_random);
    s
}

/// Derive I2RKey, R2IKey, AttestationChallenge from ECDH shared secret.
///
/// HKDF(shared, init_random‖resp_random, "SessionKeys") → 64 bytes
fn derive_session_keys(
    shared: &[u8],
    init_random: &[u8; 32],
    resp_random: &[u8; 32],
) -> ([u8; 16], [u8; 16], [u8; 32]) {
    let salt = make_sigma_salt(init_random, resp_random);
    let out = hkdf_expand_label(shared, &salt, "SessionKeys", 64);
    let mut i2r = [0u8; 16];
    let mut r2i = [0u8; 16];
    let mut challenge = [0u8; 32];
    i2r.copy_from_slice(&out[0..16]);
    r2i.copy_from_slice(&out[16..32]);
    challenge.copy_from_slice(&out[32..64]);
    (i2r, r2i, challenge)
}

/// Derive Sigma2 or Sigma3 inner encryption key.
fn derive_sigma_key(
    shared: &[u8],
    init_random: &[u8; 32],
    resp_random: &[u8; 32],
    label: &str,
) -> [u8; 16] {
    let salt = make_sigma_salt(init_random, resp_random);
    let out = hkdf_expand_label(shared, &salt, label, 16);
    let mut key = [0u8; 16];
    key.copy_from_slice(&out);
    key
}

// ── ECDH helpers ──────────────────────────────────────────────────────────────

/// Decode a 65-byte uncompressed SEC1 P-256 point to a `PublicKey`.
fn pubkey_from_bytes(bytes: &[u8; 65]) -> MatterResult<PublicKey> {
    PublicKey::from_sec1_bytes(bytes.as_slice())
        .map_err(|_| MatterError::Crypto("invalid SEC1 P-256 point".into()))
}

/// Perform ECDH and return the 32-byte shared secret.
fn ecdh_shared_secret(secret: &SecretKey, peer_pub: &[u8; 65]) -> MatterResult<[u8; 32]> {
    let peer_pk = pubkey_from_bytes(peer_pub)?;
    let shared = ecdh::diffie_hellman(secret.to_nonzero_scalar(), peer_pk.as_affine());
    let bytes: [u8; 32] = (*shared.raw_secret_bytes()).into();
    Ok(bytes)
}

// ── ECDSA helpers ─────────────────────────────────────────────────────────────

/// Sign `data` with `key`, return 64-byte raw r‖s.
fn ecdsa_sign(key: &SecretKey, data: &[u8]) -> [u8; 64] {
    let signing = SigningKey::from(key);
    let sig: Signature = signing.sign(data);
    let bytes = sig.to_bytes();
    let mut out = [0u8; 64];
    out.copy_from_slice(&bytes);
    out
}

/// Verify `sig_bytes` (64-byte r‖s) over `data` using `verifying_key` bytes (65 bytes uncompressed).
fn ecdsa_verify(verifying_key_bytes: &[u8], data: &[u8], sig_bytes: &[u8]) -> MatterResult<()> {
    use ecdsa::signature::Verifier;
    use p256::ecdsa::Signature as EcdsaSig;

    let vk = VerifyingKey::from_sec1_bytes(verifying_key_bytes)
        .map_err(|_| MatterError::Crypto("invalid P-256 verifying key".into()))?;

    if sig_bytes.len() != 64 {
        return Err(MatterError::Crypto(format!(
            "signature must be 64 bytes, got {}",
            sig_bytes.len()
        )));
    }
    let sig_arr: [u8; 64] = sig_bytes.try_into().unwrap();
    let sig = EcdsaSig::from_bytes(&sig_arr.into())
        .map_err(|_| MatterError::Crypto("invalid ECDSA signature".into()))?;

    vk.verify(data, &sig).map_err(|_| MatterError::AccessDenied)
}

// ── CaseInitiator ─────────────────────────────────────────────────────────────

/// State of the CASE initiator.
#[derive(Debug)]
pub enum CaseInitiatorState {
    /// Initial state before Sigma1 is sent.
    Idle,
    /// Sigma1 emitted; waiting for Sigma2 from the responder.
    SentSigma1 {
        /// Initiator's ephemeral ECDH secret key.
        eph_secret: SecretKey,
        /// Serialised uncompressed SEC1 form of the ephemeral public key (65 B).
        eph_pub: [u8; 65],
        /// 32-byte initiator random used in the Sigma1 TBS.
        init_random: [u8; 32],
        /// Local session ID allocated for this handshake.
        session_id: u16,
    },
    /// Handshake completed; session keys are available.
    Established(EstablishedSession),
    /// Handshake failed — holds a human-readable reason.
    Failed(String),
}

/// CASE initiator — establishes an operational session using Node Operational Credentials.
pub struct CaseInitiator {
    node_key: SecretKey,
    noc: MatterCert,
    icac: Option<MatterCert>,
    fabric: FabricDescriptor,
    state: CaseInitiatorState,
}

impl CaseInitiator {
    /// Create a new `CaseInitiator`.
    pub fn new(
        node_key: SecretKey,
        noc: MatterCert,
        icac: Option<MatterCert>,
        fabric: FabricDescriptor,
    ) -> Self {
        Self {
            node_key,
            noc,
            icac,
            fabric,
            state: CaseInitiatorState::Idle,
        }
    }

    /// Build a Sigma1 message.
    ///
    /// Returns `(session_id, payload_bytes)`.
    pub fn build_sigma1(&mut self) -> MatterResult<(u16, Vec<u8>)> {
        // Generate ephemeral key
        let eph_secret = SecretKey::random(&mut OsRng);
        let eph_pub_ep = eph_secret.public_key().to_encoded_point(false);
        let mut eph_pub = [0u8; 65];
        eph_pub.copy_from_slice(eph_pub_ep.as_bytes());

        // Random init_random
        let mut init_random = [0u8; 32];
        OsRng.fill_bytes(&mut init_random);

        // Random session ID
        let mut sid_bytes = [0u8; 2];
        OsRng.fill_bytes(&mut sid_bytes);
        let session_id = u16::from_le_bytes(sid_bytes);

        // Destination ID = HMAC-SHA256(fabric_id || node_id || root_pub_key) using init_random
        let dest_id = compute_destination_id(
            &init_random,
            &self.fabric.root_public_key,
            self.fabric.fabric_id,
            self.fabric.node_id,
        );

        // Sigma1 TLV: { tag1: init_random, tag2: session_id, tag3: dest_id, tag4: eph_pub_key }
        let mut inner = Vec::new();
        inner.extend_from_slice(&tlv_ctx_bytes(1, &init_random));
        inner.extend_from_slice(&tlv_ctx_uint2(2, session_id));
        inner.extend_from_slice(&tlv_ctx_bytes(3, &dest_id));
        inner.extend_from_slice(&tlv_ctx_bytes(4, &eph_pub));
        let sigma1 = tlv_anon_struct(&inner);

        self.state = CaseInitiatorState::SentSigma1 {
            eph_secret,
            eph_pub,
            init_random,
            session_id,
        };

        Ok((session_id, sigma1))
    }

    /// Process Sigma2 and produce a Sigma3 payload.
    pub fn handle_sigma2(&mut self, payload: &[u8]) -> MatterResult<Vec<u8>> {
        // Extract values from state, moving the SecretKey out.
        let (init_eph_pub, init_random2, session_id2, eph_secret2) = match &self.state {
            CaseInitiatorState::SentSigma1 {
                eph_secret,
                eph_pub,
                init_random,
                session_id,
            } => {
                let secret = unsafe { std::ptr::read(eph_secret as *const SecretKey) };
                (*eph_pub, *init_random, *session_id, secret)
            }
            _ => {
                return Err(MatterError::Protocol {
                    opcode: 0x31,
                    msg: "unexpected state for Sigma2".into(),
                });
            }
        };

        self.state = CaseInitiatorState::Failed("sigma2 processing".into());

        // Decode Sigma2
        let (resp_random, _resp_session_id, resp_eph_pub, encrypted2) = decode_sigma2(payload)?;

        // ECDH
        let mut resp_eph_pub_arr = [0u8; 65];
        if resp_eph_pub.len() != 65 {
            return Err(MatterError::Crypto(format!(
                "Sigma2: responder eph_pub must be 65 bytes, got {}",
                resp_eph_pub.len()
            )));
        }
        resp_eph_pub_arr.copy_from_slice(&resp_eph_pub);
        let shared = ecdh_shared_secret(&eph_secret2, &resp_eph_pub_arr)?;

        // Decrypt Encrypted2
        let s2k = derive_sigma_key(&shared, &init_random2, &resp_random, "Sigma2");
        let plaintext2 = aes128_ccm_decrypt(&s2k, SIGMA2_NONCE, &encrypted2)?;

        // Parse Encrypted2 inner: { tag1: noc, (tag2: icac), tag3: signature }
        let (resp_noc_bytes, _resp_icac_bytes, tbedata2_sig) = decode_tbedata(&plaintext2)?;

        // Verify responder's NOC is signed by our fabric's root CA
        let resp_noc = MatterCert::decode(&resp_noc_bytes)
            .map_err(|e| MatterError::Crypto(format!("Sigma2: bad responder NOC: {e}")))?;

        // Verify NOC signature against the fabric's root public key
        verify_noc_signature(&resp_noc, &self.fabric.root_public_key)?;

        // Verify TBEData2 signature: ECDSA over (init_eph_pub ‖ resp_eph_pub) with responder's node key
        let signed_data = {
            let mut v = Vec::with_capacity(130);
            v.extend_from_slice(&init_eph_pub);
            v.extend_from_slice(&resp_eph_pub_arr);
            v
        };
        ecdsa_verify(&resp_noc.public_key, &signed_data, &tbedata2_sig)?;

        // Build Sigma3 encrypted payload
        let s3k = derive_sigma_key(&shared, &init_random2, &resp_random, "Sigma3");

        // TBEData3Signature = ECDSA(init_node_key, init_eph_pub ‖ resp_eph_pub)
        let sig3 = ecdsa_sign(&self.node_key, &signed_data);

        let noc_bytes = self.noc.encode();
        let mut tbe3_inner = Vec::new();
        tbe3_inner.extend_from_slice(&tlv_ctx_bytes(1, &noc_bytes));
        if let Some(icac) = &self.icac {
            let icac_bytes = icac.encode();
            tbe3_inner.extend_from_slice(&tlv_ctx_bytes(2, &icac_bytes));
        }
        tbe3_inner.extend_from_slice(&tlv_ctx_bytes(3, &sig3));
        let tbe3_plaintext = tlv_anon_struct(&tbe3_inner);

        let encrypted3 = aes128_ccm_encrypt(&s3k, SIGMA3_NONCE, &tbe3_plaintext)?;

        // Build Sigma3: { tag1: encrypted3 }
        let inner = tlv_ctx_bytes(1, &encrypted3);
        let sigma3 = tlv_anon_struct(&inner);

        // Derive session keys
        let (i2r, r2i, challenge) = derive_session_keys(&shared, &init_random2, &resp_random);

        // Determine peer node_id from responder's NOC
        let peer_node_id = resp_noc.subject.node_id;

        let session = EstablishedSession {
            session_id: session_id2,
            peer_session_id: _resp_session_id,
            encrypt_key: i2r, // initiator uses I2R for encrypt
            decrypt_key: r2i,
            attestation_challenge: challenge,
            peer_node_id,
        };

        self.state = CaseInitiatorState::Established(session);
        Ok(sigma3)
    }

    /// Return the established session (available after `handle_sigma2` succeeds).
    pub fn established_session(&self) -> Option<&EstablishedSession> {
        match &self.state {
            CaseInitiatorState::Established(s) => Some(s),
            _ => None,
        }
    }
}

// ── CaseResponder ─────────────────────────────────────────────────────────────

/// State of the CASE responder.
#[derive(Debug)]
pub enum CaseResponderState {
    /// Initial state before Sigma1 arrives.
    Idle,
    /// Sigma2 emitted; waiting for Sigma3 from the initiator.
    SentSigma2 {
        /// Responder's ephemeral ECDH secret key.
        eph_secret: SecretKey,
        /// 32-byte responder random used in the Sigma2 TBS.
        resp_random: [u8; 32],
        /// Initiator's SEC1-encoded ephemeral public key captured from Sigma1.
        init_eph_pub: [u8; 65],
        /// Initiator's random captured from Sigma1.
        init_random: [u8; 32],
        /// Local session ID allocated for this handshake.
        session_id: u16,
    },
    /// Handshake completed; session keys are available.
    Established(EstablishedSession),
    /// Handshake failed — holds a human-readable reason.
    Failed(String),
}

/// CASE responder — accepts an operational session from an initiator.
pub struct CaseResponder {
    node_key: SecretKey,
    noc: MatterCert,
    icac: Option<MatterCert>,
    fabric: FabricDescriptor,
    state: CaseResponderState,
}

impl CaseResponder {
    /// Create a new `CaseResponder`.
    pub fn new(
        node_key: SecretKey,
        noc: MatterCert,
        icac: Option<MatterCert>,
        fabric: FabricDescriptor,
    ) -> Self {
        Self {
            node_key,
            noc,
            icac,
            fabric,
            state: CaseResponderState::Idle,
        }
    }

    /// Process a Sigma1 message and return `(session_id, Sigma2 payload)`.
    pub fn handle_sigma1(&mut self, payload: &[u8]) -> MatterResult<(u16, Vec<u8>)> {
        if !matches!(self.state, CaseResponderState::Idle) {
            return Err(MatterError::Protocol {
                opcode: 0x30,
                msg: "unexpected state for Sigma1".into(),
            });
        }

        // Decode Sigma1
        let (init_random, _init_session_id, _dest_id, init_eph_pub) = decode_sigma1(payload)?;

        let mut init_eph_pub_arr = [0u8; 65];
        if init_eph_pub.len() != 65 {
            return Err(MatterError::Crypto(format!(
                "Sigma1: initiator eph_pub must be 65 bytes, got {}",
                init_eph_pub.len()
            )));
        }
        init_eph_pub_arr.copy_from_slice(&init_eph_pub);

        // Generate our ephemeral key
        let eph_secret = SecretKey::random(&mut OsRng);
        let eph_pub_ep = eph_secret.public_key().to_encoded_point(false);
        let mut eph_pub = [0u8; 65];
        eph_pub.copy_from_slice(eph_pub_ep.as_bytes());

        // Random resp_random + session_id
        let mut resp_random = [0u8; 32];
        OsRng.fill_bytes(&mut resp_random);
        let mut sid_bytes = [0u8; 2];
        OsRng.fill_bytes(&mut sid_bytes);
        let session_id = u16::from_le_bytes(sid_bytes);

        // ECDH
        let shared = ecdh_shared_secret(&eph_secret, &init_eph_pub_arr)?;

        // TBEData2Signature = ECDSA(responder_node_key, init_eph_pub ‖ resp_eph_pub)
        let signed_data = {
            let mut v = Vec::with_capacity(130);
            v.extend_from_slice(&init_eph_pub_arr);
            v.extend_from_slice(&eph_pub);
            v
        };
        let sig2 = ecdsa_sign(&self.node_key, &signed_data);

        // Build Encrypted2 plaintext: { tag1: noc, (tag2: icac), tag3: sig }
        let noc_bytes = self.noc.encode();
        let mut tbe2_inner = Vec::new();
        tbe2_inner.extend_from_slice(&tlv_ctx_bytes(1, &noc_bytes));
        if let Some(icac) = &self.icac {
            let icac_bytes = icac.encode();
            tbe2_inner.extend_from_slice(&tlv_ctx_bytes(2, &icac_bytes));
        }
        tbe2_inner.extend_from_slice(&tlv_ctx_bytes(3, &sig2));
        let tbe2_plaintext = tlv_anon_struct(&tbe2_inner);

        let s2k = derive_sigma_key(&shared, &init_random, &resp_random, "Sigma2");
        let encrypted2 = aes128_ccm_encrypt(&s2k, SIGMA2_NONCE, &tbe2_plaintext)?;

        // Build Sigma2: { tag1: resp_random, tag2: session_id, tag3: eph_pub, tag4: encrypted2 }
        let mut inner = Vec::new();
        inner.extend_from_slice(&tlv_ctx_bytes(1, &resp_random));
        inner.extend_from_slice(&tlv_ctx_uint2(2, session_id));
        inner.extend_from_slice(&tlv_ctx_bytes(3, &eph_pub));
        inner.extend_from_slice(&tlv_ctx_bytes(4, &encrypted2));
        let sigma2 = tlv_anon_struct(&inner);

        self.state = CaseResponderState::SentSigma2 {
            eph_secret,
            resp_random,
            init_eph_pub: init_eph_pub_arr,
            init_random,
            session_id,
        };

        Ok((session_id, sigma2))
    }

    /// Process a Sigma3 message and return the established session.
    pub fn handle_sigma3(&mut self, payload: &[u8]) -> MatterResult<EstablishedSession> {
        let (eph_secret, resp_random, init_eph_pub, init_random, session_id) = match &self.state {
            CaseResponderState::SentSigma2 {
                eph_secret,
                resp_random,
                init_eph_pub,
                init_random,
                session_id,
            } => {
                let eph_secret = unsafe { std::ptr::read(eph_secret as *const SecretKey) };
                (
                    eph_secret,
                    *resp_random,
                    *init_eph_pub,
                    *init_random,
                    *session_id,
                )
            }
            _ => {
                return Err(MatterError::Protocol {
                    opcode: 0x32,
                    msg: "unexpected state for Sigma3".into(),
                });
            }
        };
        self.state = CaseResponderState::Failed("sigma3 processing".into());

        // Decode Sigma3: { tag1: encrypted3 }
        let encrypted3 = decode_sigma3(payload)?;

        // ECDH (using our ephemeral key and initiator's ephemeral pub)
        let shared = ecdh_shared_secret(&eph_secret, &init_eph_pub)?;

        // Our ephemeral pub key (reconstructed)
        let our_eph_pub_ep = eph_secret.public_key().to_encoded_point(false);
        let mut our_eph_pub = [0u8; 65];
        our_eph_pub.copy_from_slice(our_eph_pub_ep.as_bytes());

        // Decrypt Encrypted3
        let s3k = derive_sigma_key(&shared, &init_random, &resp_random, "Sigma3");
        let plaintext3 = aes128_ccm_decrypt(&s3k, SIGMA3_NONCE, &encrypted3)?;

        // Parse TBEData3: { tag1: noc, (tag2: icac), tag3: signature }
        let (init_noc_bytes, _init_icac_bytes, tbedata3_sig) = decode_tbedata(&plaintext3)?;

        // Verify initiator's NOC
        let init_noc = MatterCert::decode(&init_noc_bytes)
            .map_err(|e| MatterError::Crypto(format!("Sigma3: bad initiator NOC: {e}")))?;

        verify_noc_signature(&init_noc, &self.fabric.root_public_key)?;

        // Verify TBEData3 signature: ECDSA over (init_eph_pub ‖ our_eph_pub)
        let signed_data = {
            let mut v = Vec::with_capacity(130);
            v.extend_from_slice(&init_eph_pub);
            v.extend_from_slice(&our_eph_pub);
            v
        };
        ecdsa_verify(&init_noc.public_key, &signed_data, &tbedata3_sig)?;

        // Derive session keys
        let (i2r, r2i, challenge) = derive_session_keys(&shared, &init_random, &resp_random);

        let peer_node_id = init_noc.subject.node_id;

        let session = EstablishedSession {
            session_id,
            peer_session_id: 0,
            encrypt_key: r2i, // responder uses R2I for encrypt
            decrypt_key: i2r,
            attestation_challenge: challenge,
            peer_node_id,
        };

        self.state = CaseResponderState::Established(session.clone());
        Ok(session)
    }
}

// ── Sigma message decoders ────────────────────────────────────────────────────

/// `(random_32, session_id, dest_or_eph_bytes, eph_pub_bytes)`
type SigmaFields = ([u8; 32], u16, Vec<u8>, Vec<u8>);

/// `(noc_bytes, icac_bytes_opt, signature_64)`
type TbeDataFields = (Vec<u8>, Option<Vec<u8>>, Vec<u8>);

/// Decode Sigma1 → (init_random[32], session_id, dest_id[32], init_eph_pub[65]).
fn decode_sigma1(buf: &[u8]) -> MatterResult<SigmaFields> {
    let mut r = TlvReader::new(buf);
    let el = r.read_element()?;
    if !matches!(el.value, TlvVal::StructStart) || el.tag.is_some() {
        return Err(MatterError::Protocol {
            opcode: 0x30,
            msg: "Sigma1: expected anon struct".into(),
        });
    }

    let mut init_random_opt: Option<[u8; 32]> = None;
    let mut session_id: u16 = 0;
    let mut dest_id: Vec<u8> = Vec::new();
    let mut eph_pub: Vec<u8> = Vec::new();

    loop {
        let el = r.read_element()?;
        match el.value {
            TlvVal::End => break,
            TlvVal::Bytes(b) => {
                if el.tag == Some(1) && b.len() == 32 {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(b);
                    init_random_opt = Some(arr);
                } else if el.tag == Some(3) {
                    dest_id = b.to_vec();
                } else if el.tag == Some(4) {
                    eph_pub = b.to_vec();
                }
            }
            TlvVal::Uint(v) if el.tag == Some(2) => {
                session_id = v as u16;
            }
            _ => {}
        }
    }

    let init_random = init_random_opt.ok_or_else(|| MatterError::Protocol {
        opcode: 0x30,
        msg: "Sigma1: missing initiator_random".into(),
    })?;
    if eph_pub.is_empty() {
        return Err(MatterError::Protocol {
            opcode: 0x30,
            msg: "Sigma1: missing initiator_eph_pub_key".into(),
        });
    }

    Ok((init_random, session_id, dest_id, eph_pub))
}

/// Decode Sigma2 → (resp_random[32], session_id, resp_eph_pub, encrypted2).
fn decode_sigma2(buf: &[u8]) -> MatterResult<SigmaFields> {
    let mut r = TlvReader::new(buf);
    let el = r.read_element()?;
    if !matches!(el.value, TlvVal::StructStart) || el.tag.is_some() {
        return Err(MatterError::Protocol {
            opcode: 0x31,
            msg: "Sigma2: expected anon struct".into(),
        });
    }

    let mut resp_random_opt: Option<[u8; 32]> = None;
    let mut session_id: u16 = 0;
    let mut eph_pub: Vec<u8> = Vec::new();
    let mut encrypted2: Vec<u8> = Vec::new();

    loop {
        let el = r.read_element()?;
        match el.value {
            TlvVal::End => break,
            TlvVal::Bytes(b) => {
                if el.tag == Some(1) && b.len() == 32 {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(b);
                    resp_random_opt = Some(arr);
                } else if el.tag == Some(3) {
                    eph_pub = b.to_vec();
                } else if el.tag == Some(4) {
                    encrypted2 = b.to_vec();
                }
            }
            TlvVal::Uint(v) if el.tag == Some(2) => {
                session_id = v as u16;
            }
            _ => {}
        }
    }

    let resp_random = resp_random_opt.ok_or_else(|| MatterError::Protocol {
        opcode: 0x31,
        msg: "Sigma2: missing responder_random".into(),
    })?;
    if eph_pub.is_empty() {
        return Err(MatterError::Protocol {
            opcode: 0x31,
            msg: "Sigma2: missing responder_eph_pub_key".into(),
        });
    }
    if encrypted2.is_empty() {
        return Err(MatterError::Protocol {
            opcode: 0x31,
            msg: "Sigma2: missing encrypted2".into(),
        });
    }

    Ok((resp_random, session_id, eph_pub, encrypted2))
}

/// Decode Sigma3 → encrypted3 bytes.
fn decode_sigma3(buf: &[u8]) -> MatterResult<Vec<u8>> {
    let mut r = TlvReader::new(buf);
    let el = r.read_element()?;
    if !matches!(el.value, TlvVal::StructStart) || el.tag.is_some() {
        return Err(MatterError::Protocol {
            opcode: 0x32,
            msg: "Sigma3: expected anon struct".into(),
        });
    }
    let mut encrypted3: Option<Vec<u8>> = None;
    loop {
        let el = r.read_element()?;
        match el.value {
            TlvVal::End => break,
            TlvVal::Bytes(b) if el.tag == Some(1) => {
                encrypted3 = Some(b.to_vec());
            }
            _ => {}
        }
    }
    encrypted3.ok_or_else(|| MatterError::Protocol {
        opcode: 0x32,
        msg: "Sigma3: missing encrypted3".into(),
    })
}

/// Decode TBEData (Sigma2 or Sigma3 inner plaintext).
///
/// Returns `(noc_bytes, icac_bytes_opt, signature_64)`.
fn decode_tbedata(buf: &[u8]) -> MatterResult<TbeDataFields> {
    let mut r = TlvReader::new(buf);
    let el = r.read_element()?;
    if !matches!(el.value, TlvVal::StructStart) || el.tag.is_some() {
        return Err(MatterError::Crypto("TBEData: expected anon struct".into()));
    }
    let mut noc: Option<Vec<u8>> = None;
    let mut icac: Option<Vec<u8>> = None;
    let mut sig: Option<Vec<u8>> = None;
    loop {
        let el = r.read_element()?;
        match el.value {
            TlvVal::End => break,
            TlvVal::Bytes(b) => {
                if el.tag == Some(1) {
                    noc = Some(b.to_vec());
                } else if el.tag == Some(2) {
                    icac = Some(b.to_vec());
                } else if el.tag == Some(3) {
                    sig = Some(b.to_vec());
                }
            }
            _ => {}
        }
    }
    let noc = noc.ok_or_else(|| MatterError::Crypto("TBEData: missing NOC".into()))?;
    let sig = sig.ok_or_else(|| MatterError::Crypto("TBEData: missing signature".into()))?;
    Ok((noc, icac, sig))
}

// ── Destination ID computation ────────────────────────────────────────────────

/// Compute the Sigma1 destination_id.
///
/// Per Matter spec §4.13.2.1:
///   destination_id = HMAC-SHA256(key=init_random,
///                                data= root_pub_key ‖ fabric_id(8 LE) ‖ node_id(8 LE))
fn compute_destination_id(
    init_random: &[u8; 32],
    root_pub_key: &[u8],
    fabric_id: u64,
    node_id: u64,
) -> [u8; 32] {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    let mut mac: Hmac<Sha256> = Mac::new_from_slice(init_random).expect("HMAC key");
    mac.update(root_pub_key);
    mac.update(&fabric_id.to_le_bytes());
    mac.update(&node_id.to_le_bytes());
    mac.finalize().into_bytes().into()
}

// ── NOC chain verification ────────────────────────────────────────────────────

/// Verify that a NOC is signed by the given root public key.
///
/// In a full implementation this would walk the ICAC chain.  For Phase 4 we
/// verify directly against the root (single-level chain, no ICAC).
fn verify_noc_signature(noc: &MatterCert, root_pub_key: &[u8]) -> MatterResult<()> {
    let tbs = noc.tbs_bytes();
    ecdsa_verify(root_pub_key, &tbs, &noc.signature)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matter::fabric::manager::FabricManager;

    /// Build a FabricManager with one fabric and issue NOCs for two nodes.
    /// Returns (fabric_descriptor, init_noc, resp_noc).
    fn setup_shared_fabric(
        init_key: &SecretKey,
        init_node_id: u64,
        resp_key: &SecretKey,
        resp_node_id: u64,
    ) -> (FabricDescriptor, MatterCert, MatterCert) {
        let dir = std::env::temp_dir().join(format!("case_fabric_{}", std::process::id()));
        let mut mgr = FabricManager::new(&dir).unwrap();

        // Generate root CA
        let (sk_bytes, rcac, descriptor) = mgr
            .generate_root_ca(0xFFF1, 0xDEAD_BEEF_0000_0001, init_node_id, "TestFabric")
            .unwrap();

        // Add entry with RCAC as placeholder NOC so issue_noc can find the fabric
        mgr.add_fabric_entry(descriptor.clone(), &rcac, &rcac, None, sk_bytes);

        // Issue NOC for initiator
        let init_pub = init_key.public_key().to_encoded_point(false);
        let init_noc = mgr
            .issue_noc(descriptor.fabric_index, init_pub.as_bytes(), init_node_id)
            .unwrap();

        // Issue NOC for responder
        let resp_pub = resp_key.public_key().to_encoded_point(false);
        let resp_noc = mgr
            .issue_noc(descriptor.fabric_index, resp_pub.as_bytes(), resp_node_id)
            .unwrap();

        (descriptor, init_noc, resp_noc)
    }

    /// Full CASE handshake — sessions must have crossed keys.
    #[test]
    fn case_full_handshake_with_test_fabric() {
        let initiator_key = SecretKey::random(&mut OsRng);
        let responder_key = SecretKey::random(&mut OsRng);

        let init_node_id = 0x0000_0001_0000_0001u64;
        let resp_node_id = 0x0000_0001_0000_0002u64;

        let (descriptor, init_noc, resp_noc) =
            setup_shared_fabric(&initiator_key, init_node_id, &responder_key, resp_node_id);

        // Clone keys (SecretKey implements from_bytes)
        let init_key_2 = SecretKey::from_bytes(&initiator_key.to_bytes()).unwrap();
        let resp_key_2 = SecretKey::from_bytes(&responder_key.to_bytes()).unwrap();

        let mut initiator = CaseInitiator::new(init_key_2, init_noc, None, descriptor.clone());
        let mut responder = CaseResponder::new(resp_key_2, resp_noc, None, descriptor.clone());

        // Step 1: Sigma1
        let (_sid, sigma1) = initiator.build_sigma1().unwrap();

        // Step 2: Sigma2
        let (_resp_sid, sigma2) = responder.handle_sigma1(&sigma1).unwrap();

        // Step 3: Sigma3
        let sigma3 = initiator.handle_sigma2(&sigma2).unwrap();

        // Step 4: Finalize
        let resp_session = responder.handle_sigma3(&sigma3).unwrap();
        let init_session = initiator.established_session().unwrap();

        // Cross-verify keys
        assert_eq!(
            init_session.encrypt_key, resp_session.decrypt_key,
            "initiator encrypt_key must equal responder decrypt_key (I2R)"
        );
        assert_eq!(
            init_session.decrypt_key, resp_session.encrypt_key,
            "initiator decrypt_key must equal responder encrypt_key (R2I)"
        );
        assert_eq!(
            init_session.attestation_challenge, resp_session.attestation_challenge,
            "attestation challenge must match"
        );
    }

    /// CASE with different fabric root keys — handle_sigma2 must be rejected.
    #[test]
    fn case_wrong_fabric_rejected() {
        // Initiator uses fabric A; responder uses fabric B (different root CA key)
        let init_key = SecretKey::random(&mut OsRng);
        let resp_key = SecretKey::random(&mut OsRng);

        // Setup fabric A (initiator)
        let dir_a = std::env::temp_dir().join(format!("case_wrong_a_{}", std::process::id()));
        let mut mgr_a = FabricManager::new(&dir_a).unwrap();
        let (sk_a, rcac_a, desc_a) = mgr_a
            .generate_root_ca(0xFFF1, 0xAAAA_0001, 0x0001, "FabricA")
            .unwrap();
        mgr_a.add_fabric_entry(desc_a.clone(), &rcac_a, &rcac_a, None, sk_a);
        let init_pub = init_key.public_key().to_encoded_point(false);
        let init_noc = mgr_a
            .issue_noc(desc_a.fabric_index, init_pub.as_bytes(), 0x0001)
            .unwrap();

        // Setup fabric B (responder — different root CA)
        let dir_b = std::env::temp_dir().join(format!("case_wrong_b_{}", std::process::id()));
        let mut mgr_b = FabricManager::new(&dir_b).unwrap();
        let (sk_b, rcac_b, desc_b) = mgr_b
            .generate_root_ca(0xFFF1, 0xBBBB_0001, 0x0002, "FabricB")
            .unwrap();
        mgr_b.add_fabric_entry(desc_b.clone(), &rcac_b, &rcac_b, None, sk_b);
        let resp_pub = resp_key.public_key().to_encoded_point(false);
        let resp_noc = mgr_b
            .issue_noc(desc_b.fabric_index, resp_pub.as_bytes(), 0x0002)
            .unwrap();

        let init_key_2 = SecretKey::from_bytes(&init_key.to_bytes()).unwrap();
        let resp_key_2 = SecretKey::from_bytes(&resp_key.to_bytes()).unwrap();

        // Initiator on fabric A, responder on fabric B
        // Initiator's fabric descriptor has fabric A's root_public_key.
        // Responder's Sigma2 sends a NOC signed by fabric B's CA.
        // When initiator tries to verify it with fabric A's root key → must fail.
        let mut initiator = CaseInitiator::new(init_key_2, init_noc, None, desc_a);
        let mut responder = CaseResponder::new(resp_key_2, resp_noc, None, desc_b);

        let (_sid, sigma1) = initiator.build_sigma1().unwrap();
        let (_resp_sid, sigma2) = responder.handle_sigma1(&sigma1).unwrap();

        let result = initiator.handle_sigma2(&sigma2);
        assert!(
            result.is_err(),
            "cross-fabric CASE must be rejected when verifying responder's NOC"
        );
    }
}
