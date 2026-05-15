/// Matter fabric manager — generates root CAs, issues NOCs, and persists fabric state.
///
/// The manager owns a list of commissioned fabrics.  Each fabric has:
/// - A self-signed Root CA certificate (RCAC)
/// - A Node Operational Certificate (NOC) for this node
/// - The CA's private signing key (zeroized on drop via `SecretKey`)
///
/// Fabric state is persisted to `{storage_path}/fabrics.json`.
use std::path::{Path, PathBuf};

use ecdsa::signature::Signer;
use p256::{
    SecretKey,
    ecdsa::{Signature, SigningKey},
    elliptic_curve::sec1::ToEncodedPoint,
};
use rand_core::OsRng;

use super::cert::{MatterCert, MatterCertSubject};
use super::types::{FabricDescriptor, FabricIndex};
use crate::matter::error::{MatterError, MatterResult};

// ── Serde helpers for Vec<u8> as hex strings ─────────────────────────────────

/// Serde serialization module: serialize `Vec<u8>` as a lowercase hex string.
mod hex_bytes {
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::Deserialize;
        let s = String::deserialize(d)?;
        hex::decode(&s).map_err(serde::de::Error::custom)
    }
}

/// Serde serialization module: serialize `Option<Vec<u8>>` as an optional hex string.
mod optional_hex_bytes {
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(val: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match val {
            Some(b) => s.serialize_some(&hex::encode(b)),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Option<Vec<u8>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::Deserialize;
        let opt: Option<String> = Option::deserialize(d)?;
        match opt {
            None => Ok(None),
            Some(h) => {
                let bytes = hex::decode(&h).map_err(serde::de::Error::custom)?;
                Ok(Some(bytes))
            }
        }
    }
}

// ── Persistence types ─────────────────────────────────────────────────────────

/// One fabric entry as stored in `fabrics.json`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct StoredFabricEntry {
    /// High-level descriptor (index, pubkey, vendor, fabric_id, node_id, label).
    pub descriptor: FabricDescriptor,
    /// RCAC encoded as Matter TLV bytes (hex-encoded in JSON).
    #[serde(with = "hex_bytes")]
    pub rcac_der: Vec<u8>,
    /// NOC encoded as Matter TLV bytes (hex-encoded in JSON).
    #[serde(with = "hex_bytes")]
    pub noc_der: Vec<u8>,
    /// Optional ICAC encoded as Matter TLV bytes (hex-encoded in JSON).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_hex_bytes"
    )]
    pub icac_der: Option<Vec<u8>>,
    /// CA secret key: 32 raw bytes, hex-encoded.
    #[serde(with = "hex_bytes")]
    pub private_key_bytes: Vec<u8>,
}

// ── FabricManager ─────────────────────────────────────────────────────────────

/// Manages commissioned fabrics for a Matter device or controller.
pub struct FabricManager {
    storage_path: PathBuf,
    fabrics: Vec<StoredFabricEntry>,
    /// Next fabric index to assign (1-based, max 254 per Matter spec).
    next_index: u8,
}

impl FabricManager {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create a new, empty `FabricManager` without loading from disk.
    pub fn new(storage_path: &Path) -> MatterResult<Self> {
        Ok(Self {
            storage_path: storage_path.to_path_buf(),
            fabrics: Vec::new(),
            next_index: 1,
        })
    }

    /// Load fabric state from `{storage_path}/fabrics.json`.
    ///
    /// If the file does not exist the manager starts empty (not an error).
    pub async fn load(storage_path: &Path) -> MatterResult<Self> {
        let path = storage_path.join("fabrics.json");
        if !path.exists() {
            return Self::new(storage_path);
        }

        let json = tokio::fs::read_to_string(&path)
            .await
            .map_err(MatterError::Io)?;

        let fabrics: Vec<StoredFabricEntry> = serde_json::from_str(&json)
            .map_err(|e| MatterError::Commissioning(format!("fabrics.json parse error: {e}")))?;

        // next_index = max(existing index) + 1, clamped to [1, 254]
        let next_index = fabrics
            .iter()
            .map(|f| f.descriptor.fabric_index.0)
            .max()
            .map(|m| m.saturating_add(1))
            .unwrap_or(1);

        Ok(Self {
            storage_path: storage_path.to_path_buf(),
            fabrics,
            next_index,
        })
    }

    /// Persist current state to `{storage_path}/fabrics.json`.
    pub async fn save(&self) -> MatterResult<()> {
        let path = self.storage_path.join("fabrics.json");
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(MatterError::Io)?;
        }
        let json = serde_json::to_string_pretty(&self.fabrics).map_err(|e| {
            MatterError::Commissioning(format!("fabrics.json serialize error: {e}"))
        })?;
        tokio::fs::write(&path, json.as_bytes())
            .await
            .map_err(MatterError::Io)?;
        Ok(())
    }

    // ── Root CA generation ────────────────────────────────────────────────────

    /// Generate a self-signed root CA certificate for a new fabric.
    ///
    /// Returns `(secret_key_bytes, rcac, descriptor)`.
    /// Call `add_fabric_entry` after issuing the NOC to commit the fabric.
    pub fn generate_root_ca(
        &mut self,
        vendor_id: u16,
        fabric_id: u64,
        node_id: u64,
        label: &str,
    ) -> MatterResult<(Vec<u8>, MatterCert, FabricDescriptor)> {
        // Generate P-256 keypair
        let secret = SecretKey::random(&mut OsRng);
        let verifying = secret.public_key();
        let pubkey_bytes: Vec<u8> = verifying.to_encoded_point(false).as_bytes().to_vec();
        assert_eq!(
            pubkey_bytes.len(),
            65,
            "P-256 uncompressed point must be 65 bytes"
        );

        let mut pubkey_arr = [0u8; 65];
        pubkey_arr.copy_from_slice(&pubkey_bytes);

        // Build self-signed RCAC
        let subject = MatterCertSubject {
            rcac_id: Some(fabric_id),
            fabric_id: None,
            node_id: None,
        };

        let serial = Self::random_serial();

        let mut cert = MatterCert {
            serial_number: serial,
            issuer: subject.clone(),
            not_before: 0,
            not_after: 0,
            subject,
            public_key: pubkey_arr,
            signature: [0u8; 64],
        };

        // Sign TBS
        let signing_key = SigningKey::from(&secret);
        let tbs = cert.tbs_bytes();
        let sig: Signature = signing_key.sign(&tbs);
        let sig_bytes = Self::signature_to_raw(&sig)?;
        cert.signature.copy_from_slice(&sig_bytes);

        let sk_bytes = secret.to_bytes().to_vec();

        let fabric_index = FabricIndex(self.next_index);
        self.next_index = self.next_index.saturating_add(1);

        let descriptor = FabricDescriptor {
            fabric_index,
            root_public_key: pubkey_bytes,
            vendor_id,
            fabric_id,
            node_id,
            label: label.to_string(),
        };

        Ok((sk_bytes, cert, descriptor))
    }

    /// Store a completed fabric (RCAC + NOC + optional ICAC + CA key).
    ///
    /// Call this after `generate_root_ca` and `issue_noc`.
    pub fn add_fabric_entry(
        &mut self,
        descriptor: FabricDescriptor,
        rcac: &MatterCert,
        noc: &MatterCert,
        icac: Option<&MatterCert>,
        private_key_bytes: Vec<u8>,
    ) {
        self.fabrics.push(StoredFabricEntry {
            descriptor,
            rcac_der: rcac.encode(),
            noc_der: noc.encode(),
            icac_der: icac.map(|c| c.encode()),
            private_key_bytes,
        });
    }

    // ── NOC issuance ──────────────────────────────────────────────────────────

    /// Issue a Node Operational Certificate signed by the fabric's root CA.
    ///
    /// `csr_public_key` — 65-byte uncompressed P-256 public key from the device CSR.
    pub fn issue_noc(
        &self,
        fabric_index: FabricIndex,
        csr_public_key: &[u8],
        node_id: u64,
    ) -> MatterResult<MatterCert> {
        let entry = self
            .fabrics
            .iter()
            .find(|f| f.descriptor.fabric_index == fabric_index)
            .ok_or(MatterError::DeviceNotFound { node_id })?;

        if csr_public_key.len() != 65 {
            return Err(MatterError::Commissioning(format!(
                "issue_noc: csr_public_key must be 65 bytes, got {}",
                csr_public_key.len()
            )));
        }

        let fabric_id = entry.descriptor.fabric_id;

        // Derive issuer from RCAC subject
        let rcac = MatterCert::decode(&entry.rcac_der)?;
        let issuer = rcac.subject.clone();

        let subject = MatterCertSubject {
            node_id: Some(node_id),
            fabric_id: Some(fabric_id),
            rcac_id: None,
        };

        let mut pubkey_arr = [0u8; 65];
        pubkey_arr.copy_from_slice(csr_public_key);

        let mut noc = MatterCert {
            serial_number: Self::random_serial(),
            issuer,
            not_before: 0,
            not_after: 0,
            subject,
            public_key: pubkey_arr,
            signature: [0u8; 64],
        };

        // Load CA signing key
        if entry.private_key_bytes.len() != 32 {
            return Err(MatterError::Commissioning(format!(
                "stored CA private key must be 32 bytes, got {}",
                entry.private_key_bytes.len()
            )));
        }
        let sk_arr: [u8; 32] = entry.private_key_bytes[..32].try_into().unwrap();
        let secret = SecretKey::from_bytes(&sk_arr.into())
            .map_err(|e| MatterError::Commissioning(format!("invalid CA private key: {e}")))?;
        let signing_key = SigningKey::from(&secret);

        // Sign TBS
        let tbs = noc.tbs_bytes();
        let sig: Signature = signing_key.sign(&tbs);
        let sig_bytes = Self::signature_to_raw(&sig)?;
        noc.signature.copy_from_slice(&sig_bytes);

        Ok(noc)
    }

    // ── Query helpers ─────────────────────────────────────────────────────────

    /// Look up a fabric descriptor by index.
    pub fn fabric(&self, index: FabricIndex) -> Option<&FabricDescriptor> {
        self.fabrics
            .iter()
            .find(|f| f.descriptor.fabric_index == index)
            .map(|f| &f.descriptor)
    }

    /// Return all stored fabric entries (immutable slice).
    pub fn fabrics(&self) -> &[StoredFabricEntry] {
        &self.fabrics
    }

    /// Remove a fabric by index.
    ///
    /// Returns an error if the index does not exist.
    pub fn remove_fabric(&mut self, index: FabricIndex) -> MatterResult<()> {
        let pos = self
            .fabrics
            .iter()
            .position(|f| f.descriptor.fabric_index == index)
            .ok_or(MatterError::DeviceNotFound { node_id: 0 })?;
        self.fabrics.remove(pos);
        Ok(())
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Generate 8 random bytes for the certificate serial number.
    fn random_serial() -> Vec<u8> {
        use rand_core::RngCore;
        let mut buf = [0u8; 8];
        OsRng.fill_bytes(&mut buf);
        buf.to_vec()
    }

    /// Convert a p256 ECDSA signature to the raw 64-byte `r||s` encoding.
    fn signature_to_raw(sig: &Signature) -> MatterResult<Vec<u8>> {
        let bytes = sig.to_bytes();
        if bytes.len() != 64 {
            return Err(MatterError::Commissioning(format!(
                "ECDSA signature: expected 64 bytes, got {}",
                bytes.len()
            )));
        }
        Ok(bytes.to_vec())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path() -> PathBuf {
        std::env::temp_dir().join(format!("matter_fabric_test_{}", std::process::id()))
    }

    #[test]
    fn generate_root_ca_and_issue_noc() {
        let dir = temp_path();
        let mut mgr = FabricManager::new(&dir).expect("new should succeed");

        let vendor_id = 0xFFF1u16;
        let fabric_id = 0x0000_1111_2222_3333u64;
        let node_id = 0x0000_0001_0000_0001u64;

        // Generate root CA
        let (sk_bytes, rcac, descriptor) = mgr
            .generate_root_ca(vendor_id, fabric_id, node_id, "Test Fabric")
            .expect("generate_root_ca should succeed");

        // Verify descriptor fields
        assert_eq!(descriptor.vendor_id, vendor_id);
        assert_eq!(descriptor.fabric_id, fabric_id);
        assert_eq!(descriptor.node_id, node_id);
        assert_eq!(descriptor.label, "Test Fabric");
        assert_eq!(descriptor.root_public_key.len(), 65);
        assert_eq!(
            descriptor.root_public_key[0], 0x04,
            "uncompressed point prefix"
        );

        // Verify RCAC fields
        assert_eq!(rcac.subject.rcac_id, Some(fabric_id));
        assert_eq!(rcac.issuer.rcac_id, Some(fabric_id)); // self-signed
        assert_eq!(rcac.not_after, 0, "no expiry");

        // RCAC encode/decode roundtrip
        let rcac_encoded = rcac.encode();
        let rcac_decoded = MatterCert::decode(&rcac_encoded).expect("rcac decode");
        assert_eq!(rcac_decoded.signature, rcac.signature);

        // Commit the fabric with the RCAC as placeholder NOC
        let noc_placeholder = rcac.clone();
        let idx = descriptor.fabric_index;
        mgr.add_fabric_entry(descriptor.clone(), &rcac, &noc_placeholder, None, sk_bytes);

        // Issue a NOC for a new device (use the CA pubkey as CSR for convenience)
        let device_node_id = 0x0000_0001_0000_0002u64;
        let csr_pubkey = descriptor.root_public_key.clone();

        let noc = mgr
            .issue_noc(idx, &csr_pubkey, device_node_id)
            .expect("issue_noc should succeed");

        // Verify NOC fields
        assert_eq!(noc.subject.node_id, Some(device_node_id));
        assert_eq!(noc.subject.fabric_id, Some(fabric_id));
        assert_eq!(&noc.public_key[..], csr_pubkey.as_slice());

        // NOC encode/decode roundtrip
        let noc_encoded = noc.encode();
        let noc_decoded = MatterCert::decode(&noc_encoded).expect("noc decode");
        assert_eq!(noc_decoded.subject.node_id, Some(device_node_id));
        assert_eq!(noc_decoded.subject.fabric_id, Some(fabric_id));

        // Verify NOC signature against the CA public key
        use ecdsa::signature::Verifier;
        use p256::{
            EncodedPoint,
            ecdsa::{Signature as EcdsaSig, VerifyingKey},
        };
        let ca_pubkey = VerifyingKey::from_encoded_point(
            &EncodedPoint::from_bytes(&csr_pubkey).expect("valid CA point"),
        )
        .expect("valid verifying key");

        let tbs = noc.tbs_bytes();
        let sig = EcdsaSig::from_bytes(&noc.signature.into()).expect("valid sig bytes");
        ca_pubkey
            .verify(&tbs, &sig)
            .expect("NOC signature must verify against CA public key");
    }

    #[test]
    fn fabric_lookup_and_removal() {
        let dir = temp_path();
        let mut mgr = FabricManager::new(&dir).expect("new should succeed");

        let (sk, rcac, descriptor) = mgr
            .generate_root_ca(0xFFF1, 0xABCD_0001, 0x1111, "Fabric A")
            .expect("generate");
        let idx = descriptor.fabric_index;
        let noc = rcac.clone();
        mgr.add_fabric_entry(descriptor, &rcac, &noc, None, sk);

        assert!(mgr.fabric(idx).is_some());
        mgr.remove_fabric(idx).expect("remove should succeed");
        assert!(mgr.fabric(idx).is_none());
        assert!(mgr.remove_fabric(idx).is_err(), "second remove must fail");
    }

    #[test]
    fn next_index_increments() {
        let dir = temp_path();
        let mut mgr = FabricManager::new(&dir).expect("new should succeed");

        let (_, _, desc1) = mgr.generate_root_ca(0xFFF1, 1, 1, "A").expect("gen 1");
        let (_, _, desc2) = mgr.generate_root_ca(0xFFF1, 2, 2, "B").expect("gen 2");

        assert_ne!(desc1.fabric_index, desc2.fabric_index);
        assert_eq!(
            desc1.fabric_index.0 + 1,
            desc2.fabric_index.0,
            "indices must be consecutive"
        );
    }
}
