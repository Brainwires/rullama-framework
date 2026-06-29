use super::dataset::{Dataset, InstructDataset, PreferenceDataset};
use super::types::{PreferencePair, TrainingExample};

/// PCG random number generator multiplier constant.
const PCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
/// PCG random number generator increment constant.
const PCG_INCREMENT: u64 = 1_442_695_040_888_963_407;

/// Split configuration for train/eval datasets.
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Fraction of data for training (0.0 - 1.0).
    pub train_ratio: f32,
    /// Random seed for reproducible splits.
    pub seed: u64,
    /// Whether to shuffle before splitting.
    pub shuffle: bool,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.9,
            seed: 42,
            shuffle: true,
        }
    }
}

/// Split result containing train and eval datasets.
pub struct SplitResult {
    /// The training split.
    pub train: InstructDataset,
    /// The evaluation split.
    pub eval: InstructDataset,
}

/// Split a dataset into train/eval sets.
pub fn train_eval_split(examples: &[TrainingExample], config: &SplitConfig) -> SplitResult {
    let mut dataset = InstructDataset::new(examples.to_vec());

    if config.shuffle {
        dataset.shuffle(config.seed);
    }

    let (train, eval) = dataset.split(config.train_ratio);

    tracing::debug!("Split dataset: {} train, {} eval", train.len(), eval.len());

    SplitResult {
        train: InstructDataset::new(train),
        eval: InstructDataset::new(eval),
    }
}

/// Sort examples by token count (ascending) for curriculum learning.
pub fn curriculum_order(examples: &mut [TrainingExample]) {
    examples.sort_by_key(|e| e.estimated_tokens());
}

/// Sort examples by token count (descending) for anti-curriculum.
pub fn anti_curriculum_order(examples: &mut [TrainingExample]) {
    examples.sort_by_key(|b| std::cmp::Reverse(b.estimated_tokens()));
}

/// Sample `n` examples uniformly (with seed for reproducibility).
pub fn sample_n(examples: &[TrainingExample], n: usize, seed: u64) -> Vec<TrainingExample> {
    if n >= examples.len() {
        return examples.to_vec();
    }

    // Fisher-Yates partial shuffle
    let mut indices: Vec<usize> = (0..examples.len()).collect();
    let mut state = seed;
    for i in 0..n {
        state = state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(PCG_INCREMENT);
        let j = i + ((state >> 33) as usize % (examples.len() - i));
        indices.swap(i, j);
    }

    indices[..n].iter().map(|&i| examples[i].clone()).collect()
}

/// Split result for preference datasets.
pub struct PreferenceSplitResult {
    /// The training split.
    pub train: PreferenceDataset,
    /// The evaluation split.
    pub eval: PreferenceDataset,
}

/// Split preference pairs into train/eval sets.
pub fn preference_train_eval_split(
    pairs: &[PreferencePair],
    config: &SplitConfig,
) -> PreferenceSplitResult {
    let mut dataset = PreferenceDataset::new(pairs.to_vec());

    if config.shuffle {
        dataset.shuffle(config.seed);
    }

    let (train, eval) = dataset.split(config.train_ratio);

    tracing::debug!(
        "Split preference dataset: {} train, {} eval",
        train.len(),
        eval.len()
    );

    PreferenceSplitResult {
        train: PreferenceDataset::new(train),
        eval: PreferenceDataset::new(eval),
    }
}

/// Sort preference pairs by total token count (ascending) for curriculum learning.
pub fn preference_curriculum_order(pairs: &mut [PreferencePair]) {
    pairs.sort_by_key(|p| p.estimated_tokens());
}

/// Sample `n` preference pairs uniformly (with seed for reproducibility).
pub fn preference_sample_n(pairs: &[PreferencePair], n: usize, seed: u64) -> Vec<PreferencePair> {
    if n >= pairs.len() {
        return pairs.to_vec();
    }

    let mut indices: Vec<usize> = (0..pairs.len()).collect();
    let mut state = seed;
    for i in 0..n {
        state = state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(PCG_INCREMENT);
        let j = i + ((state >> 33) as usize % (pairs.len() - i));
        indices.swap(i, j);
    }

    indices[..n].iter().map(|&i| pairs[i].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TrainingMessage;

    fn sample_examples(n: usize) -> Vec<TrainingExample> {
        (0..n)
            .map(|i| {
                TrainingExample::with_id(
                    format!("ex-{i}"),
                    vec![
                        TrainingMessage::user(format!("Q{}: {}", i, "x".repeat(i * 10))),
                        TrainingMessage::assistant(format!("A{}", i)),
                    ],
                )
            })
            .collect()
    }

    #[test]
    fn test_train_eval_split() {
        let examples = sample_examples(100);
        let result = train_eval_split(&examples, &SplitConfig::default());
        assert_eq!(result.train.len(), 90);
        assert_eq!(result.eval.len(), 10);
    }

    #[test]
    fn test_curriculum_order() {
        let mut examples = sample_examples(10);
        curriculum_order(&mut examples);
        for i in 1..examples.len() {
            assert!(examples[i].estimated_tokens() >= examples[i - 1].estimated_tokens());
        }
    }

    #[test]
    fn test_sample_n() {
        let examples = sample_examples(100);
        let sampled = sample_n(&examples, 10, 42);
        assert_eq!(sampled.len(), 10);

        // Deterministic
        let sampled2 = sample_n(&examples, 10, 42);
        for (a, b) in sampled.iter().zip(sampled2.iter()) {
            assert_eq!(a.id, b.id);
        }
    }

    #[test]
    fn test_sample_n_larger_than_dataset() {
        let examples = sample_examples(5);
        let sampled = sample_n(&examples, 100, 42);
        assert_eq!(sampled.len(), 5);
    }

    #[test]
    fn test_preference_train_eval_split() {
        use crate::types::PreferencePair;
        let pairs: Vec<PreferencePair> = (0..100)
            .map(|i| {
                PreferencePair::new(
                    vec![TrainingMessage::user(format!("Q{}", i))],
                    vec![TrainingMessage::assistant("Good")],
                    vec![TrainingMessage::assistant("Bad")],
                )
            })
            .collect();
        let result = preference_train_eval_split(&pairs, &SplitConfig::default());
        assert_eq!(result.train.len(), 90);
        assert_eq!(result.eval.len(), 10);
    }

    #[test]
    fn test_preference_sample_n() {
        use crate::types::PreferencePair;
        let pairs: Vec<PreferencePair> = (0..50)
            .map(|i| {
                PreferencePair::new(
                    vec![TrainingMessage::user(format!("Q{}", i))],
                    vec![TrainingMessage::assistant("Good")],
                    vec![TrainingMessage::assistant("Bad")],
                )
            })
            .collect();
        let sampled = preference_sample_n(&pairs, 10, 42);
        assert_eq!(sampled.len(), 10);
        let sampled2 = preference_sample_n(&pairs, 10, 42);
        for (a, b) in sampled.iter().zip(sampled2.iter()) {
            assert_eq!(a.prompt[0].content, b.prompt[0].content);
        }
    }
}
