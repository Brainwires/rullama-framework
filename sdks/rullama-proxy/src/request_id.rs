use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a proxied request/response pair.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RequestId {
    seq: u64,
    uuid: uuid::Uuid,
}

impl RequestId {
    /// Generate a new unique request ID.
    pub fn new() -> Self {
        Self {
            seq: COUNTER.fetch_add(1, Ordering::Relaxed),
            uuid: uuid::Uuid::new_v4(),
        }
    }

    /// The monotonically-increasing sequence number.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// The UUID component.
    pub fn uuid(&self) -> uuid::Uuid {
        self.uuid
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.seq, &self.uuid.to_string()[..8])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique() {
        let a = RequestId::new();
        let b = RequestId::new();
        assert_ne!(a, b);
        assert_ne!(a.seq(), b.seq());
        assert_ne!(a.uuid(), b.uuid());
    }

    #[test]
    fn seq_is_monotonic() {
        let a = RequestId::new();
        let b = RequestId::new();
        let c = RequestId::new();
        assert!(a.seq() < b.seq());
        assert!(b.seq() < c.seq());
    }

    #[test]
    fn display_format() {
        let id = RequestId::new();
        let s = id.to_string();
        // Format: "{seq}-{first 8 chars of uuid}"
        assert!(s.contains('-'));
        let parts: Vec<&str> = s.splitn(2, '-').collect();
        assert_eq!(parts.len(), 2);
        assert!(parts[0].parse::<u64>().is_ok());
        assert_eq!(parts[1].len(), 8);
    }

    #[test]
    fn serde_roundtrip() {
        let id = RequestId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: RequestId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }
}
