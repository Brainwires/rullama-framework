use sha2::{Digest, Sha256};
use std::collections::HashMap;

const MAX_NAME_LEN: usize = 64;
const HASH_LEN: usize = 8;
// prefix + "_" + hash = MAX_NAME_LEN → prefix = 64 - 1 - 8 = 55
const MAX_PREFIX_LEN: usize = MAX_NAME_LEN - 1 - HASH_LEN;

#[derive(Debug, Default)]
pub struct ToolNameMapper {
    original_to_short: HashMap<String, String>,
    short_to_original: HashMap<String, String>,
}

impl ToolNameMapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a shortened name for a tool. Returns the original name
    /// unchanged if it's already within the 64-char limit.
    pub fn get_short_name(&mut self, original: &str) -> String {
        if original.len() <= MAX_NAME_LEN {
            return original.to_string();
        }

        if let Some(short) = self.original_to_short.get(original) {
            return short.clone();
        }

        let hash = {
            let mut hasher = Sha256::new();
            hasher.update(original.as_bytes());
            let result = hasher.finalize();
            hex::encode(&result[..HASH_LEN / 2]) // 4 bytes → 8 hex chars
        };

        let prefix = &original[..MAX_PREFIX_LEN];
        let prefix = prefix.trim_end_matches('_');
        let short = format!("{}_{}", prefix, hash);

        self.original_to_short
            .insert(original.to_string(), short.clone());
        self.short_to_original
            .insert(short.clone(), original.to_string());

        short
    }

    /// Reverse-map a short name back to the original.
    /// Returns the input unchanged if no mapping exists.
    pub fn get_original_name(&self, short: &str) -> String {
        self.short_to_original
            .get(short)
            .cloned()
            .unwrap_or_else(|| short.to_string())
    }
}

/// Encode bytes as hex string (we only need a small helper — avoids adding `hex` crate).
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}
