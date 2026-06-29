use super::super::types::{PreferencePair, TrainingExample, TrainingRole};

/// Statistics about a training dataset.
#[derive(Debug, Clone)]
pub struct DatasetStats {
    /// Total number of training examples.
    pub total_examples: usize,
    /// Total number of messages across all examples.
    pub total_messages: usize,
    /// Total estimated tokens across all examples.
    pub total_estimated_tokens: usize,
    /// Average messages per example.
    pub avg_messages_per_example: f64,
    /// Average estimated tokens per example.
    pub avg_tokens_per_example: f64,
    /// Minimum tokens in any single example.
    pub min_tokens: usize,
    /// Maximum tokens in any single example.
    pub max_tokens: usize,
    /// Number of examples that include a system message.
    pub examples_with_system: usize,
    /// Message counts per role.
    pub role_counts: RoleCounts,
    /// Token count distribution histogram.
    pub token_histogram: Vec<HistogramBucket>,
}

/// Message counts broken down by role.
#[derive(Debug, Clone, Default)]
pub struct RoleCounts {
    /// Number of system messages.
    pub system: usize,
    /// Number of user messages.
    pub user: usize,
    /// Number of assistant messages.
    pub assistant: usize,
    /// Number of tool messages.
    pub tool: usize,
}

/// A single bucket in the token count histogram.
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Inclusive lower bound of the bucket range.
    pub range_start: usize,
    /// Exclusive upper bound of the bucket range.
    pub range_end: usize,
    /// Number of examples falling in this range.
    pub count: usize,
}

/// Compute statistics for a set of training examples.
pub fn compute_stats(examples: &[TrainingExample]) -> DatasetStats {
    if examples.is_empty() {
        return DatasetStats {
            total_examples: 0,
            total_messages: 0,
            total_estimated_tokens: 0,
            avg_messages_per_example: 0.0,
            avg_tokens_per_example: 0.0,
            min_tokens: 0,
            max_tokens: 0,
            examples_with_system: 0,
            role_counts: RoleCounts::default(),
            token_histogram: Vec::new(),
        };
    }

    let mut total_messages = 0;
    let mut total_tokens = 0;
    let mut min_tokens = usize::MAX;
    let mut max_tokens = 0;
    let mut examples_with_system = 0;
    let mut role_counts = RoleCounts::default();
    let mut token_counts: Vec<usize> = Vec::with_capacity(examples.len());

    for example in examples {
        let tokens = example.estimated_tokens();
        token_counts.push(tokens);
        total_messages += example.messages.len();
        total_tokens += tokens;
        min_tokens = min_tokens.min(tokens);
        max_tokens = max_tokens.max(tokens);

        if example.has_system_message() {
            examples_with_system += 1;
        }

        for msg in &example.messages {
            match msg.role {
                TrainingRole::System => role_counts.system += 1,
                TrainingRole::User => role_counts.user += 1,
                TrainingRole::Assistant => role_counts.assistant += 1,
                TrainingRole::Tool => role_counts.tool += 1,
            }
        }
    }

    let n = examples.len();
    let histogram = build_histogram(&token_counts);

    DatasetStats {
        total_examples: n,
        total_messages,
        total_estimated_tokens: total_tokens,
        avg_messages_per_example: total_messages as f64 / n as f64,
        avg_tokens_per_example: total_tokens as f64 / n as f64,
        min_tokens,
        max_tokens,
        examples_with_system,
        role_counts,
        token_histogram: histogram,
    }
}

/// Statistics about a preference training dataset.
#[derive(Debug, Clone)]
pub struct PreferenceStats {
    /// Total number of preference pairs.
    pub total_pairs: usize,
    /// Total estimated tokens across all pairs.
    pub total_estimated_tokens: usize,
    /// Average tokens in prompt messages.
    pub avg_prompt_tokens: f64,
    /// Average tokens in chosen messages.
    pub avg_chosen_tokens: f64,
    /// Average tokens in rejected messages.
    pub avg_rejected_tokens: f64,
    /// Minimum tokens in any single pair.
    pub min_tokens: usize,
    /// Maximum tokens in any single pair.
    pub max_tokens: usize,
    /// Average ratio of chosen to rejected length.
    pub chosen_rejected_length_ratio: f64,
    /// Token count distribution histogram.
    pub token_histogram: Vec<HistogramBucket>,
}

/// Compute statistics for a set of preference pairs.
pub fn compute_preference_stats(pairs: &[PreferencePair]) -> PreferenceStats {
    if pairs.is_empty() {
        return PreferenceStats {
            total_pairs: 0,
            total_estimated_tokens: 0,
            avg_prompt_tokens: 0.0,
            avg_chosen_tokens: 0.0,
            avg_rejected_tokens: 0.0,
            min_tokens: 0,
            max_tokens: 0,
            chosen_rejected_length_ratio: 0.0,
            token_histogram: Vec::new(),
        };
    }

    let mut total_tokens = 0;
    let mut total_prompt_tokens = 0;
    let mut total_chosen_tokens = 0;
    let mut total_rejected_tokens = 0;
    let mut min_tokens = usize::MAX;
    let mut max_tokens = 0;
    let mut ratio_sum = 0.0;
    let mut token_counts: Vec<usize> = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let prompt_t: usize = pair.prompt.iter().map(|m| m.estimated_tokens()).sum();
        let chosen_t: usize = pair.chosen.iter().map(|m| m.estimated_tokens()).sum();
        let rejected_t: usize = pair.rejected.iter().map(|m| m.estimated_tokens()).sum();
        let pair_tokens = prompt_t + chosen_t + rejected_t;

        token_counts.push(pair_tokens);
        total_tokens += pair_tokens;
        total_prompt_tokens += prompt_t;
        total_chosen_tokens += chosen_t;
        total_rejected_tokens += rejected_t;
        min_tokens = min_tokens.min(pair_tokens);
        max_tokens = max_tokens.max(pair_tokens);

        let chosen_len = chosen_t.max(1) as f64;
        let rejected_len = rejected_t.max(1) as f64;
        ratio_sum += chosen_len / rejected_len;
    }

    let n = pairs.len() as f64;
    let histogram = build_histogram(&token_counts);

    PreferenceStats {
        total_pairs: pairs.len(),
        total_estimated_tokens: total_tokens,
        avg_prompt_tokens: total_prompt_tokens as f64 / n,
        avg_chosen_tokens: total_chosen_tokens as f64 / n,
        avg_rejected_tokens: total_rejected_tokens as f64 / n,
        min_tokens,
        max_tokens,
        chosen_rejected_length_ratio: ratio_sum / n,
        token_histogram: histogram,
    }
}

fn build_histogram(token_counts: &[usize]) -> Vec<HistogramBucket> {
    if token_counts.is_empty() {
        return Vec::new();
    }

    let max = *token_counts.iter().max().unwrap_or(&0);
    if max == 0 {
        return vec![HistogramBucket {
            range_start: 0,
            range_end: 1,
            count: token_counts.len(),
        }];
    }

    // Use power-of-2 bucket boundaries: 0-128, 128-256, 256-512, etc.
    let mut boundaries = vec![0usize];
    let mut b = 128;
    while b <= max {
        boundaries.push(b);
        b *= 2;
    }
    boundaries.push(b);

    let mut buckets: Vec<HistogramBucket> = boundaries
        .windows(2)
        .map(|w| HistogramBucket {
            range_start: w[0],
            range_end: w[1],
            count: 0,
        })
        .collect();

    for &count in token_counts {
        for bucket in &mut buckets {
            if count >= bucket.range_start && count < bucket.range_end {
                bucket.count += 1;
                break;
            }
        }
    }

    // Remove empty trailing buckets
    while buckets.last().is_some_and(|b| b.count == 0) {
        buckets.pop();
    }

    buckets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TrainingMessage;

    fn sample_examples() -> Vec<TrainingExample> {
        vec![
            TrainingExample::with_id(
                "1",
                vec![
                    TrainingMessage::system("Be helpful"),
                    TrainingMessage::user("Hello"),
                    TrainingMessage::assistant("Hi there! How can I help?"),
                ],
            ),
            TrainingExample::with_id(
                "2",
                vec![
                    TrainingMessage::user("What is 2+2?"),
                    TrainingMessage::assistant("4"),
                ],
            ),
            TrainingExample::with_id(
                "3",
                vec![
                    TrainingMessage::system("Expert mode"),
                    TrainingMessage::user("Explain quantum computing"),
                    TrainingMessage::assistant(
                        "Quantum computing leverages quantum mechanical phenomena...",
                    ),
                ],
            ),
        ]
    }

    #[test]
    fn test_compute_stats() {
        let stats = compute_stats(&sample_examples());
        assert_eq!(stats.total_examples, 3);
        assert_eq!(stats.total_messages, 8);
        assert_eq!(stats.examples_with_system, 2);
        assert_eq!(stats.role_counts.system, 2);
        assert_eq!(stats.role_counts.user, 3);
        assert_eq!(stats.role_counts.assistant, 3);
        assert!(stats.avg_messages_per_example > 2.0);
        assert!(stats.total_estimated_tokens > 0);
    }

    #[test]
    fn test_empty_stats() {
        let stats = compute_stats(&[]);
        assert_eq!(stats.total_examples, 0);
        assert_eq!(stats.avg_tokens_per_example, 0.0);
    }

    #[test]
    fn test_compute_preference_stats() {
        use crate::types::PreferencePair;
        let pairs = vec![
            PreferencePair::new(
                vec![TrainingMessage::user("Question one here")],
                vec![TrainingMessage::assistant("A good answer")],
                vec![TrainingMessage::assistant("Bad")],
            ),
            PreferencePair::new(
                vec![TrainingMessage::user("Another question")],
                vec![TrainingMessage::assistant("Another good answer")],
                vec![TrainingMessage::assistant("Another bad answer")],
            ),
        ];
        let stats = compute_preference_stats(&pairs);
        assert_eq!(stats.total_pairs, 2);
        assert!(stats.total_estimated_tokens > 0);
        assert!(stats.avg_prompt_tokens > 0.0);
        assert!(stats.chosen_rejected_length_ratio > 0.0);
    }

    #[test]
    fn test_empty_preference_stats() {
        let stats = compute_preference_stats(&[]);
        assert_eq!(stats.total_pairs, 0);
    }
}
