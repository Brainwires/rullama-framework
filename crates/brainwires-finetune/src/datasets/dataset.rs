use super::error::{DatasetError, DatasetResult};
use super::types::{PreferencePair, TrainingExample};

/// Core dataset abstraction.
pub trait Dataset: Send + Sync {
    /// The item type stored in this dataset.
    type Item: Clone;

    /// Return the number of items in the dataset.
    fn len(&self) -> usize;
    /// Return true if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get an item by index.
    fn get(&self, index: usize) -> Option<&Self::Item>;
    /// Return an iterator over all items.
    fn iter(&self) -> Box<dyn Iterator<Item = &Self::Item> + '_>;
    /// Shuffle the dataset in place using the given seed.
    fn shuffle(&mut self, seed: u64);
    /// Split the dataset by ratio into two vectors.
    fn split(&self, ratio: f32) -> (Vec<Self::Item>, Vec<Self::Item>);
}

/// Instruction-tuning dataset (multi-turn conversations).
#[derive(Debug, Clone)]
pub struct InstructDataset {
    examples: Vec<TrainingExample>,
}

impl InstructDataset {
    /// Create a new instruct dataset from a vector of examples.
    pub fn new(examples: Vec<TrainingExample>) -> Self {
        Self { examples }
    }

    /// Create a new instruct dataset from an iterator of examples.
    pub fn from_examples(examples: impl IntoIterator<Item = TrainingExample>) -> Self {
        Self {
            examples: examples.into_iter().collect(),
        }
    }

    /// Append a single example to the dataset.
    pub fn push(&mut self, example: TrainingExample) {
        self.examples.push(example);
    }

    /// Extend the dataset with an iterator of examples.
    pub fn extend(&mut self, examples: impl IntoIterator<Item = TrainingExample>) {
        self.examples.extend(examples);
    }

    /// Remove and return the example at the given index.
    pub fn remove(&mut self, index: usize) -> DatasetResult<TrainingExample> {
        if index >= self.examples.len() {
            return Err(DatasetError::IndexOutOfBounds {
                index,
                len: self.examples.len(),
            });
        }
        Ok(self.examples.remove(index))
    }

    /// Total estimated tokens across all examples.
    pub fn total_estimated_tokens(&self) -> usize {
        self.examples.iter().map(|e| e.estimated_tokens()).sum()
    }

    /// Filter examples by a predicate.
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&TrainingExample) -> bool,
    {
        Self {
            examples: self
                .examples
                .iter()
                .filter(|e| predicate(e))
                .cloned()
                .collect(),
        }
    }

    /// Get all examples as a slice.
    pub fn as_slice(&self) -> &[TrainingExample] {
        &self.examples
    }

    /// Consume self and return the underlying Vec.
    pub fn into_inner(self) -> Vec<TrainingExample> {
        self.examples
    }
}

impl Dataset for InstructDataset {
    type Item = TrainingExample;

    fn len(&self) -> usize {
        self.examples.len()
    }

    fn get(&self, index: usize) -> Option<&TrainingExample> {
        self.examples.get(index)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &TrainingExample> + '_> {
        Box::new(self.examples.iter())
    }

    fn shuffle(&mut self, seed: u64) {
        // Simple Fisher-Yates shuffle with deterministic seed
        let len = self.examples.len();
        if len <= 1 {
            return;
        }
        let mut state = seed;
        for i in (1..len).rev() {
            // Simple LCG for deterministic shuffle
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            self.examples.swap(i, j);
        }
    }

    fn split(&self, ratio: f32) -> (Vec<TrainingExample>, Vec<TrainingExample>) {
        let ratio = ratio.clamp(0.0, 1.0);
        let split_idx = (self.examples.len() as f32 * ratio) as usize;
        let train = self.examples[..split_idx].to_vec();
        let eval = self.examples[split_idx..].to_vec();
        (train, eval)
    }
}

/// Preference dataset for DPO/ORPO training.
#[derive(Debug, Clone)]
pub struct PreferenceDataset {
    pairs: Vec<PreferencePair>,
}

impl PreferenceDataset {
    /// Create a new preference dataset from a vector of pairs.
    pub fn new(pairs: Vec<PreferencePair>) -> Self {
        Self { pairs }
    }

    /// Append a single preference pair to the dataset.
    pub fn push(&mut self, pair: PreferencePair) {
        self.pairs.push(pair);
    }

    /// Create a new preference dataset from an iterator of pairs.
    pub fn from_pairs(pairs: impl IntoIterator<Item = PreferencePair>) -> Self {
        Self {
            pairs: pairs.into_iter().collect(),
        }
    }

    /// Extend the dataset with an iterator of pairs.
    pub fn extend(&mut self, pairs: impl IntoIterator<Item = PreferencePair>) {
        self.pairs.extend(pairs);
    }

    /// Remove and return the pair at the given index.
    pub fn remove(&mut self, index: usize) -> DatasetResult<PreferencePair> {
        if index >= self.pairs.len() {
            return Err(DatasetError::IndexOutOfBounds {
                index,
                len: self.pairs.len(),
            });
        }
        Ok(self.pairs.remove(index))
    }

    /// Filter pairs by a predicate.
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&PreferencePair) -> bool,
    {
        Self {
            pairs: self
                .pairs
                .iter()
                .filter(|p| predicate(p))
                .cloned()
                .collect(),
        }
    }

    /// Total estimated tokens across all preference pairs.
    pub fn total_estimated_tokens(&self) -> usize {
        self.pairs.iter().map(|p| p.estimated_tokens()).sum()
    }

    /// Get all pairs as a slice.
    pub fn as_slice(&self) -> &[PreferencePair] {
        &self.pairs
    }

    /// Consume self and return the underlying vector of pairs.
    pub fn into_inner(self) -> Vec<PreferencePair> {
        self.pairs
    }
}

impl Dataset for PreferenceDataset {
    type Item = PreferencePair;

    fn len(&self) -> usize {
        self.pairs.len()
    }

    fn get(&self, index: usize) -> Option<&PreferencePair> {
        self.pairs.get(index)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &PreferencePair> + '_> {
        Box::new(self.pairs.iter())
    }

    fn shuffle(&mut self, seed: u64) {
        let len = self.pairs.len();
        if len <= 1 {
            return;
        }
        let mut state = seed;
        for i in (1..len).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            self.pairs.swap(i, j);
        }
    }

    fn split(&self, ratio: f32) -> (Vec<PreferencePair>, Vec<PreferencePair>) {
        let ratio = ratio.clamp(0.0, 1.0);
        let split_idx = (self.pairs.len() as f32 * ratio) as usize;
        let train = self.pairs[..split_idx].to_vec();
        let eval = self.pairs[split_idx..].to_vec();
        (train, eval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::types::TrainingMessage;

    fn sample_examples(n: usize) -> Vec<TrainingExample> {
        (0..n)
            .map(|i| {
                TrainingExample::with_id(
                    format!("ex-{i}"),
                    vec![
                        TrainingMessage::user(format!("Question {i}")),
                        TrainingMessage::assistant(format!("Answer {i}")),
                    ],
                )
            })
            .collect()
    }

    #[test]
    fn test_instruct_dataset_basics() {
        let ds = InstructDataset::new(sample_examples(10));
        assert_eq!(ds.len(), 10);
        assert!(!ds.is_empty());
        assert!(ds.get(0).is_some());
        assert!(ds.get(10).is_none());
    }

    #[test]
    fn test_instruct_dataset_split() {
        let ds = InstructDataset::new(sample_examples(10));
        let (train, eval) = ds.split(0.8);
        assert_eq!(train.len(), 8);
        assert_eq!(eval.len(), 2);
    }

    #[test]
    fn test_instruct_dataset_shuffle_deterministic() {
        let mut ds1 = InstructDataset::new(sample_examples(20));
        let mut ds2 = InstructDataset::new(sample_examples(20));
        ds1.shuffle(42);
        ds2.shuffle(42);
        for (a, b) in ds1.iter().zip(ds2.iter()) {
            assert_eq!(a.id, b.id);
        }
    }

    #[test]
    fn test_instruct_dataset_filter() {
        let ds = InstructDataset::new(sample_examples(10));
        let filtered = ds.filter(|e| e.id.ends_with('5') || e.id.ends_with('7'));
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_preference_dataset() {
        let pairs = vec![PreferencePair::new(
            vec![TrainingMessage::user("Q")],
            vec![TrainingMessage::assistant("Good")],
            vec![TrainingMessage::assistant("Bad")],
        )];
        let ds = PreferenceDataset::new(pairs);
        assert_eq!(ds.len(), 1);
        assert!(ds.get(0).is_some());
    }

    #[test]
    fn test_preference_dataset_from_pairs() {
        let pairs = vec![
            PreferencePair::new(
                vec![TrainingMessage::user("Q1")],
                vec![TrainingMessage::assistant("Good1")],
                vec![TrainingMessage::assistant("Bad1")],
            ),
            PreferencePair::new(
                vec![TrainingMessage::user("Q2")],
                vec![TrainingMessage::assistant("Good2")],
                vec![TrainingMessage::assistant("Bad2")],
            ),
        ];
        let ds = PreferenceDataset::from_pairs(pairs);
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_preference_dataset_extend() {
        let mut ds = PreferenceDataset::new(vec![]);
        ds.extend(vec![PreferencePair::new(
            vec![TrainingMessage::user("Q")],
            vec![TrainingMessage::assistant("Good")],
            vec![TrainingMessage::assistant("Bad")],
        )]);
        assert_eq!(ds.len(), 1);
    }

    #[test]
    fn test_preference_dataset_remove() {
        let mut ds = PreferenceDataset::new(vec![PreferencePair::new(
            vec![TrainingMessage::user("Q")],
            vec![TrainingMessage::assistant("Good")],
            vec![TrainingMessage::assistant("Bad")],
        )]);
        let removed = ds.remove(0).unwrap();
        assert_eq!(removed.prompt.len(), 1);
        assert!(ds.is_empty());
        assert!(ds.remove(0).is_err());
    }

    #[test]
    fn test_preference_dataset_filter() {
        let ds = PreferenceDataset::new(vec![
            PreferencePair::new(
                vec![TrainingMessage::user("short")],
                vec![TrainingMessage::assistant("a")],
                vec![TrainingMessage::assistant("b")],
            ),
            PreferencePair::new(
                vec![TrainingMessage::user("this is a longer prompt message")],
                vec![TrainingMessage::assistant("good")],
                vec![TrainingMessage::assistant("bad")],
            ),
        ]);
        let filtered = ds.filter(|p| p.estimated_tokens() > 5);
        assert_eq!(filtered.len(), 1);
    }
}
