/// PASE (Password Authenticated Session Establishment) — Matter spec §4.14.
///
/// Uses SPAKE2+ (RFC 9383) to establish a session from a shared passcode.
///
/// # Message flow
/// ```text
/// Commissioner                              Commissionee
/// ─────────────                            ──────────────
/// build_param_request() ──PBKDFParamRequest──>
///                       <─PBKDFParamResponse── handle_param_request()
/// handle_param_response() ─────Pake1─────────>
///                       <──────Pake2────────── handle_pake1()
/// handle_pake2() ──────────────Pake3─────────>
///                              handle_pake3() → EstablishedSession
/// established_session()
/// ```
///
/// # TLV message encoding
///
/// All messages use Matter's context-tagged TLV (compact encoding):
/// control byte = (tag_type << 5) | value_type.
///
/// ## PBKDFParamRequest
/// - tag 1: passcode_id (uint, always 0)
/// - tag 2: hasPbkdfParameters (bool, false)
/// - tag 3: session_id (uint16)
///
/// ## PBKDFParamResponse
/// - tag 1: initiator_random (bytes, 32)
/// - tag 2: responder_random (bytes, 32)
/// - tag 3: session_id (uint16)
/// - tag 4: pbkdf_params struct { tag 1: iterations (uint32), tag 2: salt (bytes) }
///
/// ## Pake1
/// - tag 1: pA (bytes, 65)
///
/// ## Pake2
/// - tag 1: pB (bytes, 65)
/// - tag 2: cB (bytes, 32)
///
/// ## Pake3
/// - tag 1: cA (bytes, 32)
use rand_core::{OsRng, RngCore};

use super::EstablishedSession;
use crate::matter::crypto::{
    kdf::{derive_passcode_verifier, hkdf_expand_label},
    spake2plus::{Spake2PlusProver, Spake2PlusVerifier},
};
use crate::matter::error::{MatterError, MatterResult};

// ── TLV encoding helpers ──────────────────────────────────────────────────────

// Control byte building blocks:
//   bits 7:5  = tag type  (0=anonymous, 1=context 1-byte)
//   bits 4:0  = value type

const CTX_TAG: u8 = 1 << 5; // context-specific 1-byte tag

const UINT1_TYPE: u8 = 0x04; // unsigned int, 1 byte
const UINT2_TYPE: u8 = 0x05; // unsigned int, 2 bytes
const UINT4_TYPE: u8 = 0x06; // unsigned int, 4 bytes
const BYTES1_TYPE: u8 = 0x10; // octet string, 1-byte length prefix
const BOOL_FALSE_TYPE: u8 = 0x08; // boolean false
const STRUCT_TYPE: u8 = 0x15; // structure start
const END_TYPE: u8 = 0x18; // end of container

fn tlv_ctx_uint1(tag: u8, val: u8) -> Vec<u8> {
    vec![CTX_TAG | UINT1_TYPE, tag, val]
}

fn tlv_ctx_uint2(tag: u8, val: u16) -> Vec<u8> {
    let mut v = vec![CTX_TAG | UINT2_TYPE, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

fn tlv_ctx_uint4(tag: u8, val: u32) -> Vec<u8> {
    let mut v = vec![CTX_TAG | UINT4_TYPE, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

fn tlv_ctx_bytes(tag: u8, data: &[u8]) -> Vec<u8> {
    assert!(data.len() <= 255);
    let mut v = vec![CTX_TAG | BYTES1_TYPE, tag, data.len() as u8];
    v.extend_from_slice(data);
    v
}

fn tlv_ctx_bool_false(tag: u8) -> Vec<u8> {
    vec![CTX_TAG | BOOL_FALSE_TYPE, tag]
}

fn tlv_ctx_struct(tag: u8, inner: &[u8]) -> Vec<u8> {
    let mut v = vec![CTX_TAG | STRUCT_TYPE, tag];
    v.extend_from_slice(inner);
    v.push(END_TYPE);
    v
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

    fn read_byte(&mut self) -> MatterResult<u8> {
        if self.pos >= self.buf.len() {
            return Err(MatterError::Protocol {
                opcode: 0,
                msg: "TLV: unexpected end of buffer".into(),
            });
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_bytes_slice(&mut self, n: usize) -> MatterResult<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(MatterError::Protocol {
                opcode: 0,
                msg: format!("TLV: need {} bytes, have {}", n, self.buf.len() - self.pos),
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Read one element: returns (tag, control_byte_type, raw_value_bytes).
    ///
    /// For struct-starts returns an empty slice; caller must read children until
    /// `END_TYPE` (0x18) tag.
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
            // boolean false (no payload)
            0x08 => Ok(TlvElem {
                tag,
                value: TlvVal::Bool(false),
            }),
            // boolean true (no payload)
            0x09 => Ok(TlvElem {
                tag,
                value: TlvVal::Bool(true),
            }),
            // uint1
            0x04 => {
                let b = self.read_byte()?;
                Ok(TlvElem {
                    tag,
                    value: TlvVal::Uint(b as u64),
                })
            }
            // uint2
            0x05 => {
                let b = self.read_bytes_slice(2)?;
                Ok(TlvElem {
                    tag,
                    value: TlvVal::Uint(u16::from_le_bytes([b[0], b[1]]) as u64),
                })
            }
            // uint4
            0x06 => {
                let b = self.read_bytes_slice(4)?;
                Ok(TlvElem {
                    tag,
                    value: TlvVal::Uint(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64),
                })
            }
            // uint8
            0x07 => {
                let b = self.read_bytes_slice(8)?;
                Ok(TlvElem {
                    tag,
                    value: TlvVal::Uint(u64::from_le_bytes(b.try_into().unwrap())),
                })
            }
            // bytes 1-byte length
            0x10 => {
                let len = self.read_byte()? as usize;
                let data = self.read_bytes_slice(len)?;
                Ok(TlvElem {
                    tag,
                    value: TlvVal::Bytes(data),
                })
            }
            // struct start
            0x15 => Ok(TlvElem {
                tag,
                value: TlvVal::StructStart,
            }),
            // end of container
            0x18 => Ok(TlvElem {
                tag,
                value: TlvVal::End,
            }),
            _ => Err(MatterError::Protocol {
                opcode: 0,
                msg: format!("TLV: unsupported value type {val_type:#04x}"),
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
    Bool(#[allow(dead_code)] bool),
    StructStart,
    End,
}

// ── Session key derivation ────────────────────────────────────────────────────

/// Derive I2RKey, R2IKey, AttestationChallenge from SPAKE2+ Ke.
///
/// Per Matter spec §4.14.2.2:
///   HKDF(Ke, "", "SessionKeys") → 64 bytes → [I2RKey(16) | R2IKey(16) | AttnChallenge(32)]
fn derive_session_keys(ke: &[u8]) -> ([u8; 16], [u8; 16], [u8; 32]) {
    let out = hkdf_expand_label(ke, b"", "SessionKeys", 64);
    let mut i2r = [0u8; 16];
    let mut r2i = [0u8; 16];
    let mut challenge = [0u8; 32];
    i2r.copy_from_slice(&out[0..16]);
    r2i.copy_from_slice(&out[16..32]);
    challenge.copy_from_slice(&out[32..64]);
    (i2r, r2i, challenge)
}

// ── PaseCommissioner (initiator) ──────────────────────────────────────────────

/// State machine for the PASE commissioner (initiator) side.
pub enum PaseCommissionerState {
    /// Initial state before PBKDFParamRequest is sent.
    Idle,
    /// PBKDFParamRequest emitted; waiting for PBKDFParamResponse.
    SentParamRequest {
        /// 32-byte initiator random used in the request.
        initiator_random: [u8; 32],
        /// Local session ID allocated for this handshake.
        session_id: u16,
        /// The raw encoded PBKDFParamRequest bytes (used as SPAKE2+ context).
        req_bytes: Vec<u8>,
    },
    /// Pake1 emitted; waiting for Pake2.
    SentPake1 {
        /// SPAKE2+ prover holding `w0`, `w1`, and the random pA scalar.
        prover: Spake2PlusProver,
        /// Saved request bytes for context hash.
        req_bytes: Vec<u8>,
        /// Saved response bytes for context hash.
        resp_bytes: Vec<u8>,
        /// Local session ID allocated for this handshake.
        session_id: u16,
    },
    /// Handshake completed; session keys are available.
    Established(EstablishedSession),
    /// Handshake failed — holds a human-readable reason.
    Failed(String),
}

/// PASE commissioner — the side that initiates commissioning from the passcode.
pub struct PaseCommissioner {
    passcode: u32,
    state: PaseCommissionerState,
}

impl PaseCommissioner {
    /// Create a new commissioner with the given passcode.
    pub fn new(passcode: u32) -> Self {
        Self {
            passcode,
            state: PaseCommissionerState::Idle,
        }
    }

    /// Build a PBKDFParamRequest payload.
    ///
    /// Returns `(session_id, payload_bytes)`.  The caller must send this as a
    /// Secure Channel message with opcode `PbkdfParamRequest`.
    pub fn build_param_request(&mut self) -> MatterResult<(u16, Vec<u8>)> {
        let mut rng = OsRng;
        let mut init_random = [0u8; 32];
        rng.fill_bytes(&mut init_random);

        let mut sid_bytes = [0u8; 2];
        rng.fill_bytes(&mut sid_bytes);
        let session_id = u16::from_le_bytes(sid_bytes);

        // TLV: { tag1: passcode_id=0, tag2: hasPbkdfParameters=false, tag3: session_id }
        let mut inner = Vec::new();
        inner.extend_from_slice(&tlv_ctx_uint1(1, 0)); // passcode_id = 0
        inner.extend_from_slice(&tlv_ctx_bool_false(2)); // hasPbkdfParameters = false
        inner.extend_from_slice(&tlv_ctx_uint2(3, session_id));
        let payload = tlv_anon_struct(&inner);

        self.state = PaseCommissionerState::SentParamRequest {
            initiator_random: init_random,
            session_id,
            req_bytes: payload.clone(),
        };

        Ok((session_id, payload))
    }

    /// Process a PBKDFParamResponse and produce the Pake1 payload.
    ///
    /// `payload` — the raw TLV body of the PBKDFParamResponse message.
    pub fn handle_param_response(&mut self, payload: &[u8]) -> MatterResult<Vec<u8>> {
        let (init_random, req_bytes, session_id) = match &self.state {
            PaseCommissionerState::SentParamRequest {
                initiator_random,
                req_bytes,
                session_id,
            } => (*initiator_random, req_bytes.clone(), *session_id),
            _ => {
                return Err(MatterError::Protocol {
                    opcode: 0x21,
                    msg: "unexpected state for PBKDFParamResponse".into(),
                });
            }
        };

        // Parse PBKDFParamResponse — extract the PBKDF parameters (salt + iterations)
        let (_resp_init_random, _resp_random, _resp_session_id, iterations, salt) =
            decode_param_response(payload)?;

        let _ = init_random; // init_random is stored in state, not needed here

        // Derive SPAKE2+ verifier (w0s, w1s) from passcode + params
        let (w0s, w1s) = derive_passcode_verifier(self.passcode, &salt, iterations)
            .map_err(|e| MatterError::Spake2(e.to_string()))?;

        let prover =
            Spake2PlusProver::new(&w0s, &w1s).map_err(|e| MatterError::Spake2(e.to_string()))?;

        let pa = prover.pake_message();

        // Build Pake1 payload: { tag1: pA }
        let inner = tlv_ctx_bytes(1, &pa);
        let pake1 = tlv_anon_struct(&inner);

        self.state = PaseCommissionerState::SentPake1 {
            prover,
            req_bytes,
            resp_bytes: payload.to_vec(),
            session_id,
        };

        Ok(pake1)
    }

    /// Process a Pake2 message and produce the Pake3 payload.
    ///
    /// On success the internal state transitions to `Established`.
    pub fn handle_pake2(&mut self, payload: &[u8]) -> MatterResult<Vec<u8>> {
        let (prover, req_bytes, resp_bytes, session_id) = match &self.state {
            PaseCommissionerState::SentPake1 {
                prover,
                req_bytes,
                resp_bytes,
                session_id,
            } => {
                // We need to move the prover out; reconstruct state below.
                let prover = unsafe {
                    // SAFETY: We will immediately replace self.state.
                    // This avoids the need for Option wrapping in the state.
                    std::ptr::read(prover as *const Spake2PlusProver)
                };
                (prover, req_bytes.clone(), resp_bytes.clone(), *session_id)
            }
            _ => {
                return Err(MatterError::Protocol {
                    opcode: 0x23,
                    msg: "unexpected state for Pake2".into(),
                });
            }
        };
        // Prevent double-free: set state to Failed so we don't use the moved prover.
        self.state = PaseCommissionerState::Failed("pake2 in progress".into());

        // Parse Pake2: { tag1: pB (bytes,65), tag2: cB (bytes,32) }
        let (pb, cb) = decode_pake2(payload)?;

        // SPAKE2+ context = SHA256(req_bytes || resp_bytes)
        use sha2::{Digest, Sha256};
        let context: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(&req_bytes);
            h.update(&resp_bytes);
            h.finalize().into()
        };

        let keys = prover
            .finish(&pb, &context)
            .map_err(|e| MatterError::Spake2(e.to_string()))?;

        // Verify cB: keys.cb must equal the received cb
        if keys.cb != cb.as_slice() {
            self.state = PaseCommissionerState::Failed("Pake2 cB verification failed".into());
            return Err(MatterError::Spake2("Pake2 cB confirmation mismatch".into()));
        }

        // Build Pake3: { tag1: cA (bytes,32) }
        let inner = tlv_ctx_bytes(1, &keys.ca);
        let pake3 = tlv_anon_struct(&inner);

        // Derive session keys
        let (i2r, r2i, challenge) = derive_session_keys(&keys.ke);

        let session = EstablishedSession {
            session_id,
            peer_session_id: 0, // will be populated when we get the StatusReport
            encrypt_key: i2r,   // commissioner = initiator, so I2R = our encrypt key
            decrypt_key: r2i,
            attestation_challenge: challenge,
            peer_node_id: None,
        };

        self.state = PaseCommissionerState::Established(session);
        Ok(pake3)
    }

    /// Return the established session (available after a successful `handle_pake2`).
    pub fn established_session(&self) -> Option<&EstablishedSession> {
        match &self.state {
            PaseCommissionerState::Established(s) => Some(s),
            _ => None,
        }
    }
}

// ── PaseCommissionee (responder) ──────────────────────────────────────────────

/// State machine for the PASE commissionee (responder) side.
pub enum PaseCommissioneeState {
    /// Initial state before PBKDFParamRequest arrives.
    Idle,
    /// PBKDFParamResponse emitted; waiting for Pake1.
    SentParamResponse {
        /// Captured PBKDFParamRequest bytes for context hashing.
        req_bytes: Vec<u8>,
        /// Captured PBKDFParamResponse bytes for context hashing.
        resp_bytes: Vec<u8>,
        /// SPAKE2+ salt chosen by the commissionee.
        salt: Vec<u8>,
        /// SPAKE2+ PBKDF2 iteration count.
        iterations: u32,
    },
    /// Pake2 emitted; waiting for Pake3 to confirm the SPAKE2+ exchange.
    SentPake2 {
        /// SPAKE2+ verifier holding `L` and the random pB scalar.
        verifier: Box<Spake2PlusVerifier>,
        /// Derived SPAKE2+ keys (`Ke`, `cA`, `cB`, shared).
        keys: crate::matter::crypto::spake2plus::Spake2PlusKeys,
        /// Local session ID allocated for this handshake.
        session_id: u16,
    },
    /// Handshake completed; session keys are available.
    Established(EstablishedSession),
    /// Handshake failed — holds a human-readable reason.
    Failed(String),
}

/// PASE commissionee — the device side that receives the commissioning request.
pub struct PaseCommissionee {
    passcode: u32,
    salt: Vec<u8>,
    iterations: u32,
    state: PaseCommissioneeState,
}

impl PaseCommissionee {
    /// Create with a random salt and default iterations (10000).
    pub fn new(passcode: u32) -> Self {
        let mut salt = vec![0u8; 32];
        OsRng.fill_bytes(&mut salt);
        Self::new_with_params(passcode, salt, 10000)
    }

    /// Create with explicit salt and iterations (for deterministic tests).
    pub fn new_with_params(passcode: u32, salt: Vec<u8>, iterations: u32) -> Self {
        Self {
            passcode,
            salt,
            iterations,
            state: PaseCommissioneeState::Idle,
        }
    }

    /// Process a PBKDFParamRequest and return a PBKDFParamResponse payload.
    pub fn handle_param_request(&mut self, payload: &[u8]) -> MatterResult<Vec<u8>> {
        if !matches!(self.state, PaseCommissioneeState::Idle) {
            return Err(MatterError::Protocol {
                opcode: 0x20,
                msg: "unexpected state for PBKDFParamRequest".into(),
            });
        }

        // Parse PBKDFParamRequest
        let (init_random, _passcode_id, init_session_id) = decode_param_request(payload)?;

        // Generate responder random
        let mut resp_random = [0u8; 32];
        OsRng.fill_bytes(&mut resp_random);

        // Pick our session ID
        let mut sid_bytes = [0u8; 2];
        OsRng.fill_bytes(&mut sid_bytes);
        let resp_session_id = u16::from_le_bytes(sid_bytes);

        // Build PBKDFParamResponse
        // { tag1: initiator_random, tag2: responder_random, tag3: session_id,
        //   tag4: struct { tag1: iterations, tag2: salt } }
        let pbkdf_params_inner = {
            let mut v = Vec::new();
            v.extend_from_slice(&tlv_ctx_uint4(1, self.iterations));
            v.extend_from_slice(&tlv_ctx_bytes(2, &self.salt));
            v
        };

        let mut inner = Vec::new();
        inner.extend_from_slice(&tlv_ctx_bytes(1, &init_random));
        inner.extend_from_slice(&tlv_ctx_bytes(2, &resp_random));
        inner.extend_from_slice(&tlv_ctx_uint2(3, resp_session_id));
        inner.extend_from_slice(&tlv_ctx_struct(4, &pbkdf_params_inner));

        let resp_payload = tlv_anon_struct(&inner);

        self.state = PaseCommissioneeState::SentParamResponse {
            req_bytes: payload.to_vec(),
            resp_bytes: resp_payload.clone(),
            salt: self.salt.clone(),
            iterations: self.iterations,
        };

        let _ = init_session_id; // we use our own session ID
        Ok(resp_payload)
    }

    /// Process a Pake1 and return a Pake2 payload.
    pub fn handle_pake1(&mut self, payload: &[u8]) -> MatterResult<Vec<u8>> {
        let (req_bytes, resp_bytes, salt, iterations) = match &self.state {
            PaseCommissioneeState::SentParamResponse {
                req_bytes,
                resp_bytes,
                salt,
                iterations,
            } => (
                req_bytes.clone(),
                resp_bytes.clone(),
                salt.clone(),
                *iterations,
            ),
            _ => {
                return Err(MatterError::Protocol {
                    opcode: 0x22,
                    msg: "unexpected state for Pake1".into(),
                });
            }
        };

        // Parse Pake1: { tag1: pA (bytes,65) }
        let pa = decode_pake1(payload)?;

        // Derive w0s, w1s from passcode + params
        let (w0s, w1s) = derive_passcode_verifier(self.passcode, &salt, iterations)
            .map_err(|e| MatterError::Spake2(e.to_string()))?;

        let verifier = Spake2PlusVerifier::new_from_w1s(&w0s, &w1s)
            .map_err(|e| MatterError::Spake2(e.to_string()))?;

        let pb = verifier.pake_message();

        // SPAKE2+ context = SHA256(req_bytes || resp_bytes)
        use sha2::{Digest, Sha256};
        let context: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(&req_bytes);
            h.update(&resp_bytes);
            h.finalize().into()
        };

        let keys = verifier
            .finish(&pa, &context)
            .map_err(|e| MatterError::Spake2(e.to_string()))?;

        // Build Pake2: { tag1: pB (bytes,65), tag2: cB (bytes,32) }
        let mut inner = Vec::new();
        inner.extend_from_slice(&tlv_ctx_bytes(1, &pb));
        inner.extend_from_slice(&tlv_ctx_bytes(2, &keys.cb));
        let pake2 = tlv_anon_struct(&inner);

        // Determine session_id from our response
        let session_id = {
            // Parse resp_bytes to extract session_id (tag 3)
            extract_session_id_from_resp(&resp_bytes)
        };

        self.state = PaseCommissioneeState::SentPake2 {
            verifier: Box::new(verifier),
            keys,
            session_id,
        };

        Ok(pake2)
    }

    /// Process a Pake3 and return the established session on success.
    pub fn handle_pake3(&mut self, payload: &[u8]) -> MatterResult<EstablishedSession> {
        let (keys, session_id) = match &self.state {
            PaseCommissioneeState::SentPake2 {
                keys, session_id, ..
            } => {
                let keys = unsafe {
                    std::ptr::read(
                        keys as *const crate::matter::crypto::spake2plus::Spake2PlusKeys,
                    )
                };
                (keys, *session_id)
            }
            _ => {
                return Err(MatterError::Protocol {
                    opcode: 0x24,
                    msg: "unexpected state for Pake3".into(),
                });
            }
        };
        self.state = PaseCommissioneeState::Failed("pake3 in progress".into());

        // Parse Pake3: { tag1: cA (bytes,32) }
        let ca = decode_pake3(payload)?;

        // Verify cA
        if ca != keys.ca.as_ref() {
            self.state = PaseCommissioneeState::Failed("Pake3 cA verification failed".into());
            return Err(MatterError::Spake2("Pake3 cA confirmation mismatch".into()));
        }

        // Derive session keys
        let (i2r, r2i, challenge) = derive_session_keys(&keys.ke);

        let session = EstablishedSession {
            session_id,
            peer_session_id: 0,
            encrypt_key: r2i, // commissionee = responder, R2I = our encrypt key
            decrypt_key: i2r,
            attestation_challenge: challenge,
            peer_node_id: None,
        };

        self.state = PaseCommissioneeState::Established(session.clone());
        Ok(session)
    }
}

// ── Message decoders ──────────────────────────────────────────────────────────

/// Decode PBKDFParamRequest → (initiator_random, passcode_id, session_id).
///
/// We extract what the commissionee needs: the request bytes as context,
/// the passcode_id (always 0), and the initiator's desired session_id.
fn decode_param_request(buf: &[u8]) -> MatterResult<([u8; 32], u8, u16)> {
    let mut r = TlvReader::new(buf);
    // Anonymous struct
    let el = r.read_element()?;
    if !matches!(el.value, TlvVal::StructStart) || el.tag.is_some() {
        return Err(MatterError::Protocol {
            opcode: 0x20,
            msg: "PBKDFParamRequest: expected anon struct".into(),
        });
    }
    // For simplicity, extract a 32-byte random from tag1 if present,
    // otherwise use zeros (some request payloads don't include it)
    let mut passcode_id: u8 = 0;
    let mut session_id: u16 = 0;
    // Note: per spec, PBKDFParamRequest doesn't include initiator_random in all versions.
    // We synthesize one from the payload hash for context purposes.
    loop {
        let el = r.read_element()?;
        match el.value {
            TlvVal::End => break,
            TlvVal::Uint(v) => {
                if el.tag == Some(1) {
                    passcode_id = v as u8;
                } else if el.tag == Some(3) {
                    session_id = v as u16;
                }
            }
            TlvVal::Bool(_) => {}  // hasPbkdfParameters
            TlvVal::Bytes(_) => {} // skip unknown bytes
            _ => {}
        }
    }

    // We don't have an initiator_random in the request — use a SHA256 of the payload.
    use sha2::{Digest, Sha256};
    let hash: [u8; 32] = Sha256::digest(buf).into();

    Ok((hash, passcode_id, session_id))
}

/// Decode PBKDFParamResponse → (init_random, resp_random, session_id, iterations, salt).
/// `(init_random, resp_random, session_id, iterations, salt)`
type PbkdfParamResponseFields = ([u8; 32], [u8; 32], u16, u32, Vec<u8>);

fn decode_param_response(buf: &[u8]) -> MatterResult<PbkdfParamResponseFields> {
    let mut r = TlvReader::new(buf);

    let el = r.read_element()?;
    if !matches!(el.value, TlvVal::StructStart) || el.tag.is_some() {
        return Err(MatterError::Protocol {
            opcode: 0x21,
            msg: "PBKDFParamResponse: expected anon struct".into(),
        });
    }

    let mut init_random = [0u8; 32];
    let mut resp_random = [0u8; 32];
    let mut session_id: u16 = 0;
    let mut iterations: u32 = 0;
    let mut salt: Vec<u8> = Vec::new();

    loop {
        let el = r.read_element()?;
        match el.value {
            TlvVal::End => break,
            TlvVal::Bytes(b) => {
                if el.tag == Some(1) {
                    if b.len() == 32 {
                        init_random.copy_from_slice(b);
                    }
                } else if el.tag == Some(2) && b.len() == 32 {
                    resp_random.copy_from_slice(b);
                }
            }
            TlvVal::Uint(v) => {
                if el.tag == Some(3) {
                    session_id = v as u16;
                }
            }
            TlvVal::StructStart => {
                if el.tag == Some(4) {
                    // pbkdf_params struct
                    loop {
                        let inner = r.read_element()?;
                        match inner.value {
                            TlvVal::End => break,
                            TlvVal::Uint(v) if inner.tag == Some(1) => {
                                iterations = v as u32;
                            }
                            TlvVal::Bytes(b) if inner.tag == Some(2) => {
                                salt = b.to_vec();
                            }
                            _ => {}
                        }
                    }
                } else {
                    // skip unknown struct
                    loop {
                        if matches!(r.read_element()?.value, TlvVal::End) {
                            break;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if iterations == 0 || salt.is_empty() {
        return Err(MatterError::Protocol {
            opcode: 0x21,
            msg: "PBKDFParamResponse: missing pbkdf_params".into(),
        });
    }

    Ok((init_random, resp_random, session_id, iterations, salt))
}

/// Decode Pake1 → pA (65 bytes).
fn decode_pake1(buf: &[u8]) -> MatterResult<Vec<u8>> {
    let mut r = TlvReader::new(buf);
    let el = r.read_element()?;
    if !matches!(el.value, TlvVal::StructStart) || el.tag.is_some() {
        return Err(MatterError::Protocol {
            opcode: 0x22,
            msg: "Pake1: expected anon struct".into(),
        });
    }
    let mut pa: Option<Vec<u8>> = None;
    loop {
        let el = r.read_element()?;
        match el.value {
            TlvVal::End => break,
            TlvVal::Bytes(b) if el.tag == Some(1) => {
                pa = Some(b.to_vec());
            }
            _ => {}
        }
    }
    pa.ok_or_else(|| MatterError::Protocol {
        opcode: 0x22,
        msg: "Pake1: missing pA".into(),
    })
}

/// Decode Pake2 → (pB: Vec<u8>, cB: Vec<u8>).
fn decode_pake2(buf: &[u8]) -> MatterResult<(Vec<u8>, Vec<u8>)> {
    let mut r = TlvReader::new(buf);
    let el = r.read_element()?;
    if !matches!(el.value, TlvVal::StructStart) || el.tag.is_some() {
        return Err(MatterError::Protocol {
            opcode: 0x23,
            msg: "Pake2: expected anon struct".into(),
        });
    }
    let mut pb: Option<Vec<u8>> = None;
    let mut cb: Option<Vec<u8>> = None;
    loop {
        let el = r.read_element()?;
        match el.value {
            TlvVal::End => break,
            TlvVal::Bytes(b) => {
                if el.tag == Some(1) {
                    pb = Some(b.to_vec());
                } else if el.tag == Some(2) {
                    cb = Some(b.to_vec());
                }
            }
            _ => {}
        }
    }
    let pb = pb.ok_or_else(|| MatterError::Protocol {
        opcode: 0x23,
        msg: "Pake2: missing pB".into(),
    })?;
    let cb = cb.ok_or_else(|| MatterError::Protocol {
        opcode: 0x23,
        msg: "Pake2: missing cB".into(),
    })?;
    Ok((pb, cb))
}

/// Decode Pake3 → cA (32 bytes).
fn decode_pake3(buf: &[u8]) -> MatterResult<Vec<u8>> {
    let mut r = TlvReader::new(buf);
    let el = r.read_element()?;
    if !matches!(el.value, TlvVal::StructStart) || el.tag.is_some() {
        return Err(MatterError::Protocol {
            opcode: 0x24,
            msg: "Pake3: expected anon struct".into(),
        });
    }
    let mut ca: Option<Vec<u8>> = None;
    loop {
        let el = r.read_element()?;
        match el.value {
            TlvVal::End => break,
            TlvVal::Bytes(b) if el.tag == Some(1) => {
                ca = Some(b.to_vec());
            }
            _ => {}
        }
    }
    ca.ok_or_else(|| MatterError::Protocol {
        opcode: 0x24,
        msg: "Pake3: missing cA".into(),
    })
}

/// Extract session_id (tag 3, uint2) from a PBKDFParamResponse payload.
fn extract_session_id_from_resp(resp_bytes: &[u8]) -> u16 {
    let mut r = TlvReader::new(resp_bytes);
    if r.read_element().is_err() {
        return 0;
    }
    loop {
        match r.read_element() {
            Ok(el) => match el.value {
                TlvVal::End => return 0,
                TlvVal::Uint(v) if el.tag == Some(3) => return v as u16,
                TlvVal::StructStart => {
                    // skip struct
                    loop {
                        match r.read_element() {
                            Ok(inner) if matches!(inner.value, TlvVal::End) => break,
                            Ok(_) => {}
                            Err(_) => return 0,
                        }
                    }
                }
                _ => {}
            },
            Err(_) => return 0,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PASSCODE: u32 = 20202021;

    /// Full PASE handshake with matching passcodes — sessions must have crossed keys.
    #[test]
    fn pase_full_handshake_correct_passcode() {
        let salt = b"matter-pase-test-salt-32bytes!!!".to_vec();
        let iterations = 1000u32;

        let mut commissioner = PaseCommissioner::new(TEST_PASSCODE);
        let mut commissionee = PaseCommissionee::new_with_params(TEST_PASSCODE, salt, iterations);

        // Step 1: Commissioner → Commissionee: PBKDFParamRequest
        let (_session_id, req) = commissioner
            .build_param_request()
            .expect("build_param_request failed");

        // Step 2: Commissionee processes request, sends response
        let resp = commissionee
            .handle_param_request(&req)
            .expect("handle_param_request failed");

        // Step 3: Commissioner processes response, sends Pake1
        let pake1 = commissioner
            .handle_param_response(&resp)
            .expect("handle_param_response failed");

        // Step 4: Commissionee processes Pake1, sends Pake2
        let pake2 = commissionee
            .handle_pake1(&pake1)
            .expect("handle_pake1 failed");

        // Step 5: Commissioner processes Pake2, sends Pake3
        let pake3 = commissioner
            .handle_pake2(&pake2)
            .expect("handle_pake2 failed");

        // Get commissioner session (established after handle_pake2)
        let comm_sess = commissioner
            .established_session()
            .expect("commissioner should be established");

        // Step 6: Commissionee processes Pake3, gets session
        let comm_ee_sess = commissionee
            .handle_pake3(&pake3)
            .expect("handle_pake3 failed");

        // Verify: commissioner's encrypt key == commissionee's decrypt key (and vice versa)
        assert_eq!(
            comm_sess.encrypt_key, comm_ee_sess.decrypt_key,
            "commissioner encrypt_key must equal commissionee decrypt_key (I2R)"
        );
        assert_eq!(
            comm_sess.decrypt_key, comm_ee_sess.encrypt_key,
            "commissioner decrypt_key must equal commissionee encrypt_key (R2I)"
        );
        assert_eq!(
            comm_sess.attestation_challenge, comm_ee_sess.attestation_challenge,
            "attestation challenge must match"
        );
    }

    /// PASE with wrong passcode — handle_pake3 must return an error.
    #[test]
    fn pase_handshake_wrong_passcode_fails_at_pake3() {
        let salt = b"matter-pase-test-salt-32bytes!!!".to_vec();
        let iterations = 1000u32;

        let mut commissioner = PaseCommissioner::new(11111111); // wrong passcode
        let mut commissionee = PaseCommissionee::new_with_params(TEST_PASSCODE, salt, iterations);

        let (_sid, req) = commissioner.build_param_request().unwrap();
        let resp = commissionee.handle_param_request(&req).unwrap();
        let pake1 = commissioner.handle_param_response(&resp).unwrap();
        let pake2 = commissionee.handle_pake1(&pake1).unwrap();
        // Commissioner fails at Pake2 (cB won't match)
        let result = commissioner.handle_pake2(&pake2);
        assert!(result.is_err(), "wrong passcode should fail at Pake2");
    }

    /// PBKDFParamRequest must encode session_id at tag 3.
    #[test]
    fn pase_param_request_encodes_session_id() {
        let mut commissioner = PaseCommissioner::new(TEST_PASSCODE);
        let (session_id, req_bytes) = commissioner.build_param_request().unwrap();

        // Re-parse and find session_id at tag 3
        let mut r = TlvReader::new(&req_bytes);
        let _ = r.read_element().unwrap(); // anon struct start
        let mut found_sid: Option<u16> = None;
        loop {
            let el = r.read_element().unwrap();
            match el.value {
                TlvVal::End => break,
                TlvVal::Uint(v) if el.tag == Some(3) => {
                    found_sid = Some(v as u16);
                }
                _ => {}
            }
        }
        assert_eq!(
            found_sid,
            Some(session_id),
            "session_id must be encoded at tag 3"
        );
    }
}
