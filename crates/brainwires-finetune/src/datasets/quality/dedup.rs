use super::super::types::{PreferencePair, TrainingExample};

use sha2::{Digest, Sha256};

/// Default similarity threshold for near-duplicate detection.
const DEFAULT_DEDUP_SIMILARITY_THRESHOLD: f64 = 0.85;

/// MinHash-based deduplication for training datasets.
///
/// Uses shingling + MinHash to detect near-duplicate examples.
pub struct Deduplicator {
    num_hashes: usize,
    similarity_threshold: f64,
    shingle_size: usize,
}

impl Default for Deduplicator {
    fn default() -> Self {
        Self {
            num_hashes: 128,
            similarity_threshold: DEFAULT_DEDUP_SIMILARITY_THRESHOLD,
            shingle_size: 3,
        }
    }
}

impl Deduplicator {
    /// Create a new deduplicator with the given number of hashes and similarity threshold.
    pub fn new(num_hashes: usize, similarity_threshold: f64) -> Self {
        Self {
            num_hashes,
            similarity_threshold,
            shingle_size: 3,
        }
    }

    /// Set the shingle (n-gram) size for MinHash computation.
    pub fn with_shingle_size(mut self, size: usize) -> Self {
        self.shingle_size = size;
        self
    }

    /// Remove near-duplicate examples, returning deduplicated examples and count removed.
    pub fn deduplicate(&self, examples: &[TrainingExample]) -> (Vec<TrainingExample>, usize) {
        if examples.len() <= 1 {
            return (examples.to_vec(), 0);
        }

        // Compute MinHash signatures for each example
        let signatures: Vec<Vec<u64>> = examples
            .iter()
            .map(|e| self.minhash_signature(&self.example_text(e)))
            .collect();

        let mut keep = vec![true; examples.len()];
        let mut removed = 0;

        for i in 0..examples.len() {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..examples.len() {
                if !keep[j] {
                    continue;
                }
                let sim = self.jaccard_estimate(&signatures[i], &signatures[j]);
                if sim >= self.similarity_threshold {
                    keep[j] = false;
                    removed += 1;
                }
            }
        }

        let deduped = examples
            .iter()
            .zip(keep.iter())
            .filter(|&(_, &k)| k)
            .map(|(e, _)| e.clone())
            .collect();

        (deduped, removed)
    }

    /// Remove near-duplicate preference pairs.
    pub fn deduplicate_preferences(
        &self,
        pairs: &[PreferencePair],
    ) -> (Vec<PreferencePair>, usize) {
        if pairs.len() <= 1 {
            return (pairs.to_vec(), 0);
        }

        let signatures: Vec<Vec<u64>> = pairs
            .iter()
            .map(|p| {
                let text = self.preference_text(p);
                self.minhash_signature(&text)
            })
            .collect();

        let mut keep = vec![true; pairs.len()];
        let mut removed = 0;

        for i in 0..pairs.len() {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..pairs.len() {
                if !keep[j] {
                    continue;
                }
                let sim = self.jaccard_estimate(&signatures[i], &signatures[j]);
                if sim >= self.similarity_threshold {
                    keep[j] = false;
                    removed += 1;
                }
            }
        }

        let deduped = pairs
            .iter()
            .zip(keep.iter())
            .filter(|&(_, &k)| k)
            .map(|(p, _)| p.clone())
            .collect();

        (deduped, removed)
    }

    /// Extract text from a preference pair for hashing.
    fn preference_text(&self, pair: &PreferencePair) -> String {
        let prompt: String = pair
            .prompt
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let chosen: String = pair
            .chosen
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let rejected: String = pair
            .rejected
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        format!("{} {} {}", prompt, chosen, rejected)
    }

    /// Extract text from an example for hashing.
    fn example_text(&self, example: &TrainingExample) -> String {
        example
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Compute a MinHash signature for a text string.
    fn minhash_signature(&self, text: &str) -> Vec<u64> {
        let shingles = self.shingle(text);
        let mut signature = vec![u64::MAX; self.num_hashes];

        for shingle in &shingles {
            let base_hash = self.hash_shingle(shingle);
            for (i, sig) in signature.iter_mut().enumerate() {
                // Different hash function per slot: hash XOR with slot-dependent constant
                let h = base_hash
                    .wrapping_mul((i as u64).wrapping_add(1))
                    .wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
                if h < *sig {
                    *sig = h;
                }
            }
        }

        signature
    }

    /// Create word-level shingles from text.
    fn shingle(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < self.shingle_size {
            return vec![text.to_lowercase()];
        }
        words
            .windows(self.shingle_size)
            .map(|w| w.join(" ").to_lowercase())
            .collect()
    }

    /// Hash a shingle to u64.
    fn hash_shingle(&self, shingle: &str) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(shingle.as_bytes());
        let result = hasher.finalize();
        u64::from_le_bytes(
            result[..8]
                .try_into()
                .expect("SHA256 always produces >= 8 bytes"),
        )
    }

    /// Estimate Jaccard similarity from two MinHash signatures.
    fn jaccard_estimate(&self, sig_a: &[u64], sig_b: &[u64]) -> f64 {
        let matches = sig_a
            .iter()
            .zip(sig_b.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / sig_a.len() as f64
    }
}

/// Exact deduplication by content hash (remove exact duplicates only).
pub fn exact_dedup(examples: &[TrainingExample]) -> (Vec<TrainingExample>, usize) {
    let mut seen = std::collections::HashSet::new();
    let mut deduped = Vec::new();
    let mut removed = 0;

    for example in examples {
        let mut hasher = Sha256::new();
        for msg in &example.messages {
            hasher.update(msg.role.to_string().as_bytes());
            hasher.update(msg.content.as_bytes());
        }
        let hash = format!("{:x}", hasher.finalize());

        if seen.insert(hash) {
            deduped.push(example.clone());
        } else {
            removed += 1;
        }
    }

    (deduped, removed)
}

/// Exact deduplication for preference pairs.
pub fn exact_dedup_preferences(pairs: &[PreferencePair]) -> (Vec<PreferencePair>, usize) {
    let mut seen = std::collections::HashSet::new();
    let mut deduped = Vec::new();
    let mut removed = 0;

    for pair in pairs {
        let mut hasher = Sha256::new();
        for msg in &pair.prompt {
            hasher.update(msg.role.to_string().as_bytes());
            hasher.update(msg.content.as_bytes());
        }
        for msg in &pair.chosen {
            hasher.update(b"chosen:");
            hasher.update(msg.content.as_bytes());
        }
        for msg in &pair.rejected {
            hasher.update(b"rejected:");
            hasher.update(msg.content.as_bytes());
        }
        let hash = format!("{:x}", hasher.finalize());

        if seen.insert(hash) {
            deduped.push(pair.clone());
        } else {
            removed += 1;
        }
    }

    (deduped, removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::types::TrainingMessage;

    #[test]
    fn test_exact_dedup() {
        let examples = vec![
            TrainingExample::with_id(
                "1",
                vec![
                    TrainingMessage::user("Hello"),
                    TrainingMessage::assistant("Hi"),
                ],
            ),
            TrainingExample::with_id(
                "2",
                vec![
                    TrainingMessage::user("Hello"),
                    TrainingMessage::assistant("Hi"),
                ],
            ),
            TrainingExample::with_id(
                "3",
                vec![
                    TrainingMessage::user("Different"),
                    TrainingMessage::assistant("Response"),
                ],
            ),
        ];

        let (deduped, removed) = exact_dedup(&examples);
        assert_eq!(deduped.len(), 2);
        assert_eq!(removed, 1);
    }

    #[test]
    fn test_minhash_dedup() {
        let dedup = Deduplicator::new(128, 0.8);
        let examples = vec![
            TrainingExample::with_id(
                "1",
                vec![
                    TrainingMessage::user("The quick brown fox jumps over the lazy dog"),
                    TrainingMessage::assistant("That is a well-known sentence."),
                ],
            ),
            TrainingExample::with_id(
                "2",
                vec![
                    TrainingMessage::user("The quick brown fox jumps over the lazy dog"),
                    TrainingMessage::assistant("That is a well-known sentence indeed."),
                ],
            ),
            TrainingExample::with_id(
                "3",
                vec![
                    TrainingMessage::user("Explain quantum entanglement in simple terms"),
                    TrainingMessage::assistant(
                        "Quantum entanglement is when particles become linked.",
                    ),
                ],
            ),
        ];

        let (deduped, removed) = dedup.deduplicate(&examples);
        // The first two are very similar, one should be removed
        assert!(removed >= 1);
        assert!(deduped.len() <= 2);
    }

    #[test]
    fn test_empty_dedup() {
        let (deduped, removed) = exact_dedup(&[]);
        assert!(deduped.is_empty());
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_exact_dedup_preferences() {
        use crate::datasets::types::PreferencePair;
        let pairs = vec![
            PreferencePair::new(
                vec![TrainingMessage::user("Q")],
                vec![TrainingMessage::assistant("Good")],
                vec![TrainingMessage::assistant("Bad")],
            ),
            PreferencePair::new(
                vec![TrainingMessage::user("Q")],
                vec![TrainingMessage::assistant("Good")],
                vec![TrainingMessage::assistant("Bad")],
            ),
            PreferencePair::new(
                vec![TrainingMessage::user("Different")],
                vec![TrainingMessage::assistant("A")],
                vec![TrainingMessage::assistant("B")],
            ),
        ];
        let (deduped, removed) = exact_dedup_preferences(&pairs);
        assert_eq!(deduped.len(), 2);
        assert_eq!(removed, 1);
    }
}
