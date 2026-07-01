/// Matter 1.3 TLV-encoded certificate (NOC / RCAC / ICAC).
///
/// Matter uses its own compact TLV encoding instead of ASN.1 DER.
/// The encoding is defined in Matter Core Specification §6.4.
///
/// # Tag layout (context-specific, 1-byte)
/// | Tag | Field                  | TLV type        |
/// |-----|------------------------|-----------------|
/// |  1  | serial_number          | octet-string    |
/// |  2  | signature_algorithm    | uint (1=ECDSA)  |
/// |  3  | issuer                 | struct          |
/// |  4  | not_before             | uint (epoch)    |
/// |  5  | not_after              | uint (0=none)   |
/// |  6  | subject                | struct          |
/// |  7  | public_key_algorithm   | uint (1=EC)     |
/// |  8  | elliptic_curve_id      | uint (1=P-256)  |
/// |  9  | public_key             | octet-string    |
/// | 11  | signature              | octet-string    |
///
/// Subject / issuer struct tags:
/// | 17 | node_id   | uint |
/// | 20 | rcac_id   | uint |
/// | 21 | fabric_id | uint |
///
/// Timestamps use the Matter epoch: seconds since 2000-01-01 00:00:00 UTC.
use crate::matter::error::{MatterError, MatterResult};

// ── TLV encoding constants ────────────────────────────────────────────────────

/// Control byte = tag-type(context 1-byte, 0x20) | element-type
const CTX: u8 = 0x20; // context-specific, 1-byte tag number

// Element types (lower 5 bits of control byte)
const UINT1: u8 = 0x04; // unsigned int, 1 byte value
#[allow(dead_code)]
const UINT2: u8 = 0x05; // unsigned int, 2 byte value (reserved for future use)
const UINT4: u8 = 0x06; // unsigned int, 4 byte value
const UINT8: u8 = 0x07; // unsigned int, 8 byte value
const BYTES1: u8 = 0x10; // octet string, 1-byte length
const STRUCT: u8 = 0x15; // structure
const END: u8 = 0x18; // end-of-container (anonymous)
// Anonymous types (no tag prefix — used for end-of-container)
const ANON_STRUCT: u8 = STRUCT; // anonymous struct = same type bits, no tag

// ── Low-level TLV helpers ─────────────────────────────────────────────────────

/// Encode an anonymous (no-tag) structure: 0x15 ... 0x18
fn tlv_anon_struct(inner: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(2 + inner.len());
    v.push(ANON_STRUCT);
    v.extend_from_slice(inner);
    v.push(END);
    v
}

/// Encode a context-tagged structure: (0x20|0x15) tag ... 0x18
fn tlv_ctx_struct(tag: u8, inner: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(3 + inner.len());
    v.push(CTX | STRUCT);
    v.push(tag);
    v.extend_from_slice(inner);
    v.push(END);
    v
}

/// Encode a context-tagged u8.
fn tlv_ctx_u8(tag: u8, val: u8) -> Vec<u8> {
    vec![CTX | UINT1, tag, val]
}

/// Encode a context-tagged u32.
fn tlv_ctx_u32(tag: u8, val: u32) -> Vec<u8> {
    let mut v = vec![CTX | UINT4, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

/// Encode a context-tagged u64.
fn tlv_ctx_u64(tag: u8, val: u64) -> Vec<u8> {
    let mut v = vec![CTX | UINT8, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

/// Encode a context-tagged octet string (length fits in 1 byte, i.e. ≤ 255 bytes).
fn tlv_ctx_bytes(tag: u8, data: &[u8]) -> Vec<u8> {
    assert!(
        data.len() <= 255,
        "TLV octet-string: length > 255 not supported"
    );
    let mut v = Vec::with_capacity(3 + data.len());
    v.push(CTX | BYTES1);
    v.push(tag);
    v.push(data.len() as u8);
    v.extend_from_slice(data);
    v
}

// ── TLV decoder ───────────────────────────────────────────────────────────────

/// Minimal pull-style TLV reader.
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

    fn peek_byte(&self) -> MatterResult<u8> {
        if self.pos >= self.buf.len() {
            return Err(MatterError::Commissioning(
                "TLV: unexpected end of buffer".into(),
            ));
        }
        Ok(self.buf[self.pos])
    }

    fn read_byte(&mut self) -> MatterResult<u8> {
        let b = self.peek_byte()?;
        self.pos += 1;
        Ok(b)
    }

    fn read_bytes(&mut self, n: usize) -> MatterResult<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(MatterError::Commissioning(format!(
                "TLV: need {} bytes, have {}",
                n,
                self.buf.len() - self.pos
            )));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// Read one TLV element.  Returns `(tag_opt, value)`.
    /// `tag_opt` is `None` for anonymous elements (end-of-container, anonymous struct).
    fn read_element(&mut self) -> MatterResult<TlvElement> {
        if self.remaining() == 0 {
            return Err(MatterError::Commissioning("TLV: buffer exhausted".into()));
        }
        let ctrl = self.read_byte()?;
        let tag_type = (ctrl >> 5) & 0x07; // upper 3 bits
        let elem_type = ctrl & 0x1f; // lower 5 bits

        let tag: Option<u8> = match tag_type {
            0 => None,                    // anonymous
            1 => Some(self.read_byte()?), // context 1-byte tag
            _ => {
                return Err(MatterError::Commissioning(format!(
                    "TLV: unsupported tag type {tag_type}"
                )));
            }
        };

        let value = match elem_type {
            // unsigned ints
            0x04 => TlvValue::Uint(self.read_byte()? as u64),
            0x05 => {
                let b = self.read_bytes(2)?;
                TlvValue::Uint(u16::from_le_bytes([b[0], b[1]]) as u64)
            }
            0x06 => {
                let b = self.read_bytes(4)?;
                TlvValue::Uint(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64)
            }
            0x07 => {
                let b = self.read_bytes(8)?;
                TlvValue::Uint(u64::from_le_bytes(b.try_into().unwrap()))
            }
            // octet strings (1-byte length)
            0x10 => {
                let len = self.read_byte()? as usize;
                TlvValue::Bytes(self.read_bytes(len)?.to_vec())
            }
            // structure — caller reads children until end-of-container
            0x15 => TlvValue::StructStart,
            // end of container
            0x18 => TlvValue::EndOfContainer,
            _ => {
                return Err(MatterError::Commissioning(format!(
                    "TLV: unsupported element type {elem_type:#04x}"
                )));
            }
        };

        Ok(TlvElement { tag, value })
    }
}

#[derive(Debug)]
struct TlvElement {
    tag: Option<u8>,
    value: TlvValue,
}

#[derive(Debug)]
enum TlvValue {
    Uint(u64),
    Bytes(Vec<u8>),
    StructStart,
    EndOfContainer,
}

// ── Certificate subject/issuer ────────────────────────────────────────────────

/// The subject or issuer DN of a Matter certificate.
///
/// Only one of these fields is expected per certificate type:
/// - RCAC: `rcac_id` is set (tag 20), `fabric_id` may also appear
/// - NOC/ICAC: `node_id` and `fabric_id` are set (tags 17, 21)
#[derive(Debug, Clone, Default)]
pub struct MatterCertSubject {
    /// Node ID (tag 17) — present in NOC subject.
    pub node_id: Option<u64>,
    /// RCAC CA ID (tag 20) — present in RCAC subject/issuer.
    pub rcac_id: Option<u64>,
    /// Fabric ID (tag 21) — present in NOC/ICAC subject and their issuer.
    pub fabric_id: Option<u64>,
}

impl MatterCertSubject {
    fn encode(&self) -> Vec<u8> {
        let mut inner = Vec::new();
        if let Some(v) = self.node_id {
            inner.extend_from_slice(&tlv_ctx_u64(17, v));
        }
        if let Some(v) = self.rcac_id {
            inner.extend_from_slice(&tlv_ctx_u64(20, v));
        }
        if let Some(v) = self.fabric_id {
            inner.extend_from_slice(&tlv_ctx_u64(21, v));
        }
        inner
    }

    fn decode_from_reader(reader: &mut TlvReader<'_>) -> MatterResult<Self> {
        let mut s = Self::default();
        // expect StructStart already consumed by the outer reader
        loop {
            let el = reader.read_element()?;
            match el.value {
                TlvValue::EndOfContainer => break,
                TlvValue::Uint(v) => match el.tag {
                    Some(17) => s.node_id = Some(v),
                    Some(20) => s.rcac_id = Some(v),
                    Some(21) => s.fabric_id = Some(v),
                    _ => {} // unknown tag — skip
                },
                _ => {
                    return Err(MatterError::Commissioning(
                        "TLV cert subject: unexpected element type".into(),
                    ));
                }
            }
        }
        Ok(s)
    }
}

// ── Matter Certificate ────────────────────────────────────────────────────────

/// A Matter 1.3 TLV-encoded certificate (NOC, RCAC, or ICAC).
///
/// # Encoding
/// The wire format is Matter's own TLV (spec §6.4), not X.509 DER.
/// `encode()` produces the full certificate including the signature (tag 11).
/// `tbs_bytes()` produces the to-be-signed bytes (all tags except 11).
///
/// # Signing
/// The signature field `sig` must be 64 bytes: `r||s` (each 32 bytes, big-endian).
/// This is the raw ECDSA-SHA256 signature over `SHA-256(tbs_bytes())`.
#[derive(Debug, Clone)]
pub struct MatterCert {
    /// Certificate serial number (arbitrary bytes, typically 8 bytes random).
    pub serial_number: Vec<u8>,
    /// Issuer DN.
    pub issuer: MatterCertSubject,
    /// Validity start: seconds since Matter epoch (2000-01-01 00:00:00 UTC).
    pub not_before: u32,
    /// Validity end: seconds since Matter epoch; 0 = no expiry.
    pub not_after: u32,
    /// Subject DN.
    pub subject: MatterCertSubject,
    /// Uncompressed P-256 public key (65 bytes, 0x04 prefix).
    pub public_key: [u8; 65],
    /// ECDSA-SHA256 signature over `tbs_bytes()`: 32-byte r || 32-byte s (big-endian).
    pub signature: [u8; 64],
}

impl MatterCert {
    // ── TBS encoding (no signature) ──────────────────────────────────────────

    /// Encode the to-be-signed portion of the certificate (all fields except tag 11).
    ///
    /// The signer computes: `ECDSA-SHA256(signing_key, tbs_bytes())`.
    pub fn tbs_bytes(&self) -> Vec<u8> {
        self.encode_inner(false)
    }

    // ── Full encoding ────────────────────────────────────────────────────────

    /// Encode the complete certificate (including signature, tag 11) to Matter TLV.
    pub fn encode(&self) -> Vec<u8> {
        self.encode_inner(true)
    }

    fn encode_inner(&self, include_sig: bool) -> Vec<u8> {
        let mut inner = Vec::new();

        // Tag 1: serial_number (octet string)
        inner.extend_from_slice(&tlv_ctx_bytes(1, &self.serial_number));
        // Tag 2: signature_algorithm = 1 (ECDSA-with-SHA256)
        inner.extend_from_slice(&tlv_ctx_u8(2, 1));
        // Tag 3: issuer (struct)
        inner.extend_from_slice(&tlv_ctx_struct(3, &self.issuer.encode()));
        // Tag 4: not_before (uint32)
        inner.extend_from_slice(&tlv_ctx_u32(4, self.not_before));
        // Tag 5: not_after (uint32; 0 = no expiry)
        inner.extend_from_slice(&tlv_ctx_u32(5, self.not_after));
        // Tag 6: subject (struct)
        inner.extend_from_slice(&tlv_ctx_struct(6, &self.subject.encode()));
        // Tag 7: public_key_algorithm = 1 (EC)
        inner.extend_from_slice(&tlv_ctx_u8(7, 1));
        // Tag 8: elliptic_curve_id = 1 (prime256v1)
        inner.extend_from_slice(&tlv_ctx_u8(8, 1));
        // Tag 9: public_key (octet string, 65 bytes)
        inner.extend_from_slice(&tlv_ctx_bytes(9, &self.public_key));

        if include_sig {
            // Tag 11: signature (octet string, 64 bytes)
            inner.extend_from_slice(&tlv_ctx_bytes(11, &self.signature));
        }

        tlv_anon_struct(&inner)
    }

    // ── Decoding ─────────────────────────────────────────────────────────────

    /// Decode a Matter TLV-encoded certificate.
    pub fn decode(bytes: &[u8]) -> MatterResult<Self> {
        let mut reader = TlvReader::new(bytes);

        // Top-level anonymous struct
        let el = reader.read_element()?;
        match el.value {
            TlvValue::StructStart if el.tag.is_none() => {}
            _ => {
                return Err(MatterError::Commissioning(
                    "TLV cert: expected top-level anonymous struct".into(),
                ));
            }
        }

        let mut serial_number: Option<Vec<u8>> = None;
        let mut issuer: Option<MatterCertSubject> = None;
        let mut not_before: Option<u32> = None;
        let mut not_after: Option<u32> = None;
        let mut subject: Option<MatterCertSubject> = None;
        let mut public_key: Option<Vec<u8>> = None;
        let mut signature: Option<Vec<u8>> = None;

        loop {
            let el = reader.read_element()?;
            match el.value {
                TlvValue::EndOfContainer => break,
                TlvValue::Bytes(b) => match el.tag {
                    Some(1) => serial_number = Some(b),
                    Some(9) => public_key = Some(b),
                    Some(11) => signature = Some(b),
                    _ => {}
                },
                TlvValue::Uint(v) => match el.tag {
                    Some(2) | Some(7) | Some(8) => {} // algorithm IDs — skip
                    Some(4) => not_before = Some(v as u32),
                    Some(5) => not_after = Some(v as u32),
                    _ => {}
                },
                TlvValue::StructStart => match el.tag {
                    Some(3) => issuer = Some(MatterCertSubject::decode_from_reader(&mut reader)?),
                    Some(6) => subject = Some(MatterCertSubject::decode_from_reader(&mut reader)?),
                    _ => {
                        // Unknown struct — skip to end-of-container
                        loop {
                            let inner = reader.read_element()?;
                            if matches!(inner.value, TlvValue::EndOfContainer) {
                                break;
                            }
                        }
                    }
                },
            }
        }

        // Validate required fields
        let serial_number = serial_number.ok_or_else(|| {
            MatterError::Commissioning("TLV cert: missing serial_number (tag 1)".into())
        })?;
        let issuer = issuer
            .ok_or_else(|| MatterError::Commissioning("TLV cert: missing issuer (tag 3)".into()))?;
        let not_before = not_before.ok_or_else(|| {
            MatterError::Commissioning("TLV cert: missing not_before (tag 4)".into())
        })?;
        let not_after = not_after.ok_or_else(|| {
            MatterError::Commissioning("TLV cert: missing not_after (tag 5)".into())
        })?;
        let subject = subject.ok_or_else(|| {
            MatterError::Commissioning("TLV cert: missing subject (tag 6)".into())
        })?;
        let public_key_bytes = public_key.ok_or_else(|| {
            MatterError::Commissioning("TLV cert: missing public_key (tag 9)".into())
        })?;
        let signature_bytes = signature.ok_or_else(|| {
            MatterError::Commissioning("TLV cert: missing signature (tag 11)".into())
        })?;

        if public_key_bytes.len() != 65 {
            return Err(MatterError::Commissioning(format!(
                "TLV cert: public_key must be 65 bytes, got {}",
                public_key_bytes.len()
            )));
        }
        if signature_bytes.len() != 64 {
            return Err(MatterError::Commissioning(format!(
                "TLV cert: signature must be 64 bytes, got {}",
                signature_bytes.len()
            )));
        }

        let mut public_key = [0u8; 65];
        public_key.copy_from_slice(&public_key_bytes);
        let mut signature = [0u8; 64];
        signature.copy_from_slice(&signature_bytes);

        Ok(MatterCert {
            serial_number,
            issuer,
            not_before,
            not_after,
            subject,
            public_key,
            signature,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_pubkey() -> [u8; 65] {
        let mut k = [0u8; 65];
        k[0] = 0x04; // uncompressed point prefix
        for (i, slot) in k.iter_mut().enumerate().skip(1) {
            *slot = i as u8;
        }
        k
    }

    fn dummy_sig() -> [u8; 64] {
        let mut s = [0u8; 64];
        for (i, slot) in s.iter_mut().enumerate() {
            *slot = (i + 128) as u8;
        }
        s
    }

    fn make_noc() -> MatterCert {
        MatterCert {
            serial_number: vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
            issuer: MatterCertSubject {
                rcac_id: Some(1),
                fabric_id: Some(0xDEAD_BEEF_0000_0001),
                node_id: None,
            },
            not_before: 0,
            not_after: 0,
            subject: MatterCertSubject {
                node_id: Some(0x0000_0001_0000_0001),
                fabric_id: Some(0xDEAD_BEEF_0000_0001),
                rcac_id: None,
            },
            public_key: dummy_pubkey(),
            signature: dummy_sig(),
        }
    }

    fn make_rcac() -> MatterCert {
        MatterCert {
            serial_number: vec![0xCA, 0xFE, 0xBA, 0xBE],
            issuer: MatterCertSubject {
                rcac_id: Some(1),
                fabric_id: None,
                node_id: None,
            },
            not_before: 0,
            not_after: 0,
            subject: MatterCertSubject {
                rcac_id: Some(1),
                fabric_id: None,
                node_id: None,
            },
            public_key: dummy_pubkey(),
            signature: dummy_sig(),
        }
    }

    #[test]
    fn noc_encode_decode_roundtrip() {
        let cert = make_noc();
        let encoded = cert.encode();
        let decoded = MatterCert::decode(&encoded).expect("decode should succeed");

        assert_eq!(decoded.serial_number, cert.serial_number);
        assert_eq!(decoded.not_before, cert.not_before);
        assert_eq!(decoded.not_after, cert.not_after);
        assert_eq!(decoded.public_key, cert.public_key);
        assert_eq!(decoded.signature, cert.signature);
        assert_eq!(decoded.subject.node_id, cert.subject.node_id);
        assert_eq!(decoded.subject.fabric_id, cert.subject.fabric_id);
        assert_eq!(decoded.issuer.rcac_id, cert.issuer.rcac_id);
        assert_eq!(decoded.issuer.fabric_id, cert.issuer.fabric_id);
    }

    #[test]
    fn rcac_encode_decode_roundtrip() {
        let cert = make_rcac();
        let encoded = cert.encode();
        let decoded = MatterCert::decode(&encoded).expect("decode should succeed");

        assert_eq!(decoded.serial_number, cert.serial_number);
        assert_eq!(decoded.subject.rcac_id, cert.subject.rcac_id);
        assert_eq!(decoded.issuer.rcac_id, cert.issuer.rcac_id);
        assert_eq!(decoded.public_key, cert.public_key);
        assert_eq!(decoded.signature, cert.signature);
    }

    #[test]
    fn tbs_bytes_deterministic() {
        let cert = make_noc();
        let tbs1 = cert.tbs_bytes();
        let tbs2 = cert.tbs_bytes();
        assert_eq!(tbs1, tbs2, "tbs_bytes must be deterministic");

        // TBS must not contain the signature bytes
        let full = cert.encode();
        assert!(
            full.len() > tbs1.len(),
            "full cert should be larger than TBS"
        );
    }
}
