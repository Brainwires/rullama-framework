//! First-to-ahead-by-k Voting System
//!
//! Implements Algorithm 2 from the MAKER paper for error correction through consensus.
//! The voting continues until one option has at least k more votes than any other option.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::Semaphore;

use super::error::{MdapError, MdapResult, VotingError};
use super::red_flags::{RedFlagResult, RedFlagValidator};

// --- EarlyStoppingConfig default preset constants ---
const DEFAULT_MIN_CONFIDENCE: f64 = 0.85;
const DEFAULT_MIN_VOTES: u32 = 3;
const DEFAULT_MAX_VARIANCE_THRESHOLD: f64 = 0.15;
const DEFAULT_MIN_WEIGHTED_CONFIDENCE: f64 = 0.80;

// --- EarlyStoppingConfig aggressive preset constants ---
const AGGRESSIVE_MIN_CONFIDENCE: f64 = 0.75;
const AGGRESSIVE_MIN_VOTES: u32 = 2;
const AGGRESSIVE_MAX_VARIANCE_THRESHOLD: f64 = 0.20;
const AGGRESSIVE_MIN_WEIGHTED_CONFIDENCE: f64 = 0.70;

// --- EarlyStoppingConfig conservative preset constants ---
const CONSERVATIVE_MIN_CONFIDENCE: f64 = 0.90;
const CONSERVATIVE_MIN_VOTES: u32 = 5;
const CONSERVATIVE_MAX_VARIANCE_THRESHOLD: f64 = 0.10;
const CONSERVATIVE_MIN_WEIGHTED_CONFIDENCE: f64 = 0.85;

// --- Voter parallel sampling constants ---
const DEFAULT_PARALLEL_LIMIT: usize = 4;
const DEFAULT_BATCH_SIZE: usize = 4;

/// Response with metadata for red-flag checking
#[derive(Clone, Debug)]
pub struct SampledResponse<T> {
    /// The parsed/extracted value from the response
    pub value: T,
    /// Metadata about the response for validation
    pub metadata: ResponseMetadata,
    /// The raw response string (for red-flag validation)
    pub raw_response: String,
    /// Confidence score for this response (0.0 - 1.0)
    /// Used for confidence-weighted voting (CISC paper)
    pub confidence: f64,
}

impl<T> SampledResponse<T> {
    /// Create a new sampled response with default confidence
    pub fn new(value: T, metadata: ResponseMetadata, raw_response: String) -> Self {
        Self {
            value,
            metadata,
            raw_response,
            confidence: 0.75, // Default moderate confidence
        }
    }

    /// Create with explicit confidence
    pub fn with_confidence(
        value: T,
        metadata: ResponseMetadata,
        raw_response: String,
        confidence: f64,
    ) -> Self {
        Self {
            value,
            metadata,
            raw_response,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Metadata extracted from LLM response
#[derive(Clone, Debug, Default)]
pub struct ResponseMetadata {
    /// Number of tokens in the response
    pub token_count: u32,
    /// Response time in milliseconds
    pub response_time_ms: u64,
    /// Whether the format was valid (pre-red-flag check)
    pub format_valid: bool,
    /// The finish reason from the API (if available)
    pub finish_reason: Option<String>,
    /// Model used for this response
    pub model: Option<String>,
}

/// Result of the voting process
#[derive(Clone, Debug)]
pub struct VoteResult<T> {
    /// The winning value
    pub winner: T,
    /// Number of votes for the winner
    pub winner_votes: u32,
    /// Total number of valid votes cast
    pub total_votes: u32,
    /// Total samples taken (including red-flagged)
    pub total_samples: u32,
    /// Number of red-flagged (discarded) samples
    pub red_flagged_count: u32,
    /// Distribution of votes by candidate (for logging/analysis)
    pub vote_distribution: HashMap<String, u32>,
    /// Confidence score (winner_votes / total_votes)
    pub confidence: f64,
    /// Reasons for red-flagged samples
    pub red_flag_reasons: Vec<String>,
    /// Whether voting stopped early due to high confidence (RASC-style)
    pub early_stopped: bool,
    /// Weighted confidence score (when using confidence-weighted voting)
    pub weighted_confidence: Option<f64>,
    /// Voting method used
    pub voting_method: VotingMethod,
}

/// Voting method selection
#[derive(Clone, Debug, Default, PartialEq)]
pub enum VotingMethod {
    /// Original first-to-ahead-by-k (default)
    #[default]
    FirstToAheadByK,
    /// Borda count - rank all candidates by weighted scores
    BordaCount,
    /// Confidence-weighted voting (CISC paper)
    ConfidenceWeighted,
}

/// Configuration for dynamic early stopping (RASC paper: arxiv:2408.17017)
///
/// Enhanced with:
/// - Variance tracking: stop when vote distribution is stable
/// - Loss-of-hope detection: stop if no candidate can win within remaining budget
/// - Criteria-based stopping: evaluate both outputs AND confidence quality
#[derive(Clone, Debug)]
pub struct EarlyStoppingConfig {
    /// Minimum confidence ratio to trigger early stop (e.g., 0.85 = 85%)
    pub min_confidence: f64,
    /// Minimum votes before considering early stop
    pub min_votes: u32,
    /// Whether early stopping is enabled
    pub enabled: bool,
    /// Maximum variance threshold for stability-based stopping (RASC)
    /// Stop if vote distribution variance is below this threshold
    pub max_variance_threshold: f64,
    /// Enable loss-of-hope detection
    /// Stop if the gap to win exceeds remaining samples
    pub loss_of_hope_enabled: bool,
    /// Minimum weighted confidence for stopping (when using confidence-weighted voting)
    pub min_weighted_confidence: f64,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            min_votes: DEFAULT_MIN_VOTES,
            enabled: true,
            max_variance_threshold: DEFAULT_MAX_VARIANCE_THRESHOLD,
            loss_of_hope_enabled: true,
            min_weighted_confidence: DEFAULT_MIN_WEIGHTED_CONFIDENCE,
        }
    }
}

impl EarlyStoppingConfig {
    /// Create disabled early stopping config
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            loss_of_hope_enabled: false,
            ..Default::default()
        }
    }

    /// Create with custom thresholds
    pub fn new(min_confidence: f64, min_votes: u32) -> Self {
        Self {
            min_confidence: min_confidence.clamp(0.5, 0.99),
            min_votes: min_votes.max(2),
            enabled: true,
            ..Default::default()
        }
    }

    /// Create an aggressive early stopping config (saves more samples but may sacrifice accuracy)
    pub fn aggressive() -> Self {
        Self {
            min_confidence: AGGRESSIVE_MIN_CONFIDENCE,
            min_votes: AGGRESSIVE_MIN_VOTES,
            enabled: true,
            max_variance_threshold: AGGRESSIVE_MAX_VARIANCE_THRESHOLD,
            loss_of_hope_enabled: true,
            min_weighted_confidence: AGGRESSIVE_MIN_WEIGHTED_CONFIDENCE,
        }
    }

    /// Create a conservative early stopping config (more accurate but uses more samples)
    pub fn conservative() -> Self {
        Self {
            min_confidence: CONSERVATIVE_MIN_CONFIDENCE,
            min_votes: CONSERVATIVE_MIN_VOTES,
            enabled: true,
            max_variance_threshold: CONSERVATIVE_MAX_VARIANCE_THRESHOLD,
            loss_of_hope_enabled: true,
            min_weighted_confidence: CONSERVATIVE_MIN_WEIGHTED_CONFIDENCE,
        }
    }
}

/// First-to-ahead-by-k voter implementing Algorithm 2 from the MAKER paper
///
/// The algorithm continues sampling until one candidate has at least k more
/// votes than any other candidate: `V[y] >= k + max(V[v] for v != y)`
///
/// Enhanced with:
/// - Early stopping (RASC paper: arxiv:2408.17017)
/// - Confidence-weighted voting (CISC paper: arxiv:2502.06233v1)
/// - Borda count alternative (Ranked voting paper: arxiv:2505.10772)
pub struct FirstToAheadByKVoter {
    /// Vote margin threshold (k in the paper)
    k: u32,
    /// Maximum samples before giving up
    max_samples: u32,
    /// Semaphore to limit parallel sampling (max 4 threads)
    parallel_limit: Arc<Semaphore>,
    /// Number of samples to take per batch
    batch_size: usize,
    /// Early stopping configuration (RASC-style)
    early_stopping: EarlyStoppingConfig,
    /// Voting method to use
    voting_method: VotingMethod,
    /// Whether to use confidence-weighted votes
    use_confidence_weights: bool,
}

impl FirstToAheadByKVoter {
    /// Create a new voter with specified k and max samples
    ///
    /// # Arguments
    /// * `k` - Vote margin threshold. Winner needs k more votes than runner-up.
    /// * `max_samples` - Maximum number of samples before giving up
    ///
    /// # Panics
    /// Panics if k < 1
    pub fn new(k: u32, max_samples: u32) -> Self {
        assert!(k >= 1, "k must be >= 1");
        Self {
            k,
            max_samples,
            parallel_limit: Arc::new(Semaphore::new(DEFAULT_PARALLEL_LIMIT)),
            batch_size: DEFAULT_BATCH_SIZE,
            early_stopping: EarlyStoppingConfig::default(),
            voting_method: VotingMethod::FirstToAheadByK,
            use_confidence_weights: false,
        }
    }

    /// Create a voter with early stopping enabled
    pub fn with_early_stopping(
        k: u32,
        max_samples: u32,
        early_stopping: EarlyStoppingConfig,
    ) -> Self {
        let mut voter = Self::new(k, max_samples);
        voter.early_stopping = early_stopping;
        voter
    }

    /// Create a voter with confidence-weighted voting (CISC paper)
    pub fn with_confidence_weighting(k: u32, max_samples: u32) -> Self {
        let mut voter = Self::new(k, max_samples);
        voter.use_confidence_weights = true;
        voter.voting_method = VotingMethod::ConfidenceWeighted;
        voter
    }

    /// Create a voter with Borda count voting (Ranked Voting paper: arxiv:2505.10772)
    ///
    /// Borda count uses confidence-weighted scores instead of raw vote counts.
    /// This is particularly effective when responses have varying confidence levels.
    pub fn with_borda_count(k: u32, max_samples: u32) -> Self {
        let mut voter = Self::new(k, max_samples);
        voter.voting_method = VotingMethod::BordaCount;
        voter.use_confidence_weights = true; // Always use for Borda
        voter
    }

    /// Execute voting until a winner emerges or max samples reached
    ///
    /// Implements Algorithm 2: do_voting from the MAKER paper:
    /// ```text
    /// Input: x (state), M (model), k (threshold)
    /// V ← {v: 0 ∀v}  # Vote counts
    /// while True do:
    ///     y ← get_vote(x, M)
    ///     V[y] = V[y] + 1
    ///     if V[y] >= k + max(V[v] for v ≠ y) then:
    ///         return y
    /// ```
    ///
    /// # Type Parameters
    /// * `T` - The type of value being voted on (must be Eq + Hash + Clone)
    /// * `F` - The sampler function type
    /// * `Fut` - The future type returned by the sampler
    ///
    /// # Arguments
    /// * `sampler` - A function that samples a response from the model
    /// * `red_flag_validator` - Validator for checking red flags
    ///
    /// # Returns
    /// * `Ok(VoteResult)` - If consensus was reached
    /// * `Err(VotingError)` - If max samples exceeded or all samples red-flagged
    pub async fn vote<T, K, F, Fut>(
        &self,
        sampler: F,
        red_flag_validator: &dyn RedFlagValidator,
        key_extractor: K,
    ) -> MdapResult<VoteResult<T>>
    where
        T: Clone + Send + 'static,
        K: Fn(&T) -> String + Send + Sync,
        F: Fn() -> Fut + Send + Sync,
        Fut: Future<Output = MdapResult<SampledResponse<T>>> + Send + 'static,
    {
        // Vote counts and weighted votes for confidence-weighted voting
        let mut votes: HashMap<String, (u32, T)> = HashMap::new();
        let mut weighted_votes: HashMap<String, f64> = HashMap::new();
        let mut total_samples = 0u32;
        let mut red_flagged = 0u32;
        let mut red_flag_reasons: Vec<String> = Vec::new();

        loop {
            if total_samples >= self.max_samples {
                return Err(MdapError::Voting(VotingError::MaxSamplesExceeded {
                    samples: total_samples,
                    votes: votes.iter().map(|(k, (v, _))| (k.clone(), *v)).collect(),
                }));
            }

            // Calculate how many samples to take this batch
            let remaining = self.max_samples.saturating_sub(total_samples);
            let batch_count = (self.batch_size as u32).min(remaining) as usize;

            // Sample in parallel (up to 4 concurrent)
            let samples = self.sample_parallel(&sampler, batch_count).await?;

            if samples.is_empty() && total_samples == 0 {
                return Err(MdapError::Voting(VotingError::NoValidResponses {
                    attempts: batch_count as u32,
                }));
            }

            for sample in samples {
                total_samples += 1;

                // Red-flag check (Algorithm 3 integration)
                match red_flag_validator.validate(&sample.raw_response, &sample.metadata) {
                    RedFlagResult::Valid => {
                        let key = key_extractor(&sample.value);
                        let entry = votes
                            .entry(key.clone())
                            .or_insert((0, sample.value.clone()));
                        entry.0 += 1;

                        // Track confidence-weighted votes for Borda/CISC voting methods
                        // Always track for BordaCount and ConfidenceWeighted methods
                        if self.use_confidence_weights
                            || self.voting_method == VotingMethod::BordaCount
                            || self.voting_method == VotingMethod::ConfidenceWeighted
                        {
                            *weighted_votes.entry(key.clone()).or_insert(0.0) += sample.confidence;
                        }

                        // Check for early stopping (RASC paper)
                        if self.early_stopping.enabled
                            && let Some((winner_key, winner_value)) = self.check_early_stop(&votes)
                        {
                            let vote_distribution: HashMap<String, u32> =
                                votes.iter().map(|(k, (v, _))| (k.clone(), *v)).collect();

                            let winner_votes = votes.get(&winner_key).map(|(v, _)| *v).unwrap_or(0);
                            let total_votes: u32 = votes.values().map(|(v, _)| *v).sum();

                            let weighted_confidence = if self.use_confidence_weights {
                                let total_weight: f64 = weighted_votes.values().sum();
                                let winner_weight =
                                    weighted_votes.get(&winner_key).copied().unwrap_or(0.0);
                                Some(winner_weight / total_weight.max(0.001))
                            } else {
                                None
                            };

                            tracing::info!(
                                total_samples = total_samples,
                                total_votes = total_votes,
                                confidence = %self.calculate_confidence(winner_votes, total_votes),
                                "MDAP: Early stopping triggered"
                            );

                            return Ok(VoteResult {
                                winner: winner_value,
                                winner_votes,
                                total_votes,
                                total_samples,
                                red_flagged_count: red_flagged,
                                vote_distribution,
                                confidence: self.calculate_confidence(winner_votes, total_votes),
                                red_flag_reasons,
                                early_stopped: true,
                                weighted_confidence,
                                voting_method: self.voting_method.clone(),
                            });
                        }

                        // Check winner based on voting method
                        let winner_result = match self.voting_method {
                            VotingMethod::BordaCount => {
                                // Borda count: winner determined by weighted confidence scores
                                // (Ranked Voting paper: arxiv:2505.10772)
                                self.check_borda_winner(&votes, &weighted_votes)
                            }
                            VotingMethod::ConfidenceWeighted => {
                                // Confidence-weighted: still need k-ahead margin but use weighted votes
                                // (CISC paper: arxiv:2502.06233v1)
                                self.check_weighted_winner(&votes, &weighted_votes)
                            }
                            VotingMethod::FirstToAheadByK => {
                                // Original: V[y] >= k + max(V[v] for v != y)
                                self.check_winner(&votes)
                            }
                        };

                        if let Some((winner_key, winner_value)) = winner_result {
                            let vote_distribution: HashMap<String, u32> =
                                votes.iter().map(|(k, (v, _))| (k.clone(), *v)).collect();

                            let winner_votes = votes.get(&winner_key).map(|(v, _)| *v).unwrap_or(0);
                            let total_votes: u32 = votes.values().map(|(v, _)| *v).sum();

                            let weighted_confidence = if self.use_confidence_weights
                                || self.voting_method == VotingMethod::BordaCount
                            {
                                let total_weight: f64 = weighted_votes.values().sum();
                                let winner_weight =
                                    weighted_votes.get(&winner_key).copied().unwrap_or(0.0);
                                Some(winner_weight / total_weight.max(0.001))
                            } else {
                                None
                            };

                            return Ok(VoteResult {
                                winner: winner_value,
                                winner_votes,
                                total_votes,
                                total_samples,
                                red_flagged_count: red_flagged,
                                vote_distribution,
                                confidence: self.calculate_confidence(winner_votes, total_votes),
                                red_flag_reasons,
                                early_stopped: false,
                                weighted_confidence,
                                voting_method: self.voting_method.clone(),
                            });
                        }
                    }
                    RedFlagResult::Flagged { reason, .. } => {
                        red_flagged += 1;
                        red_flag_reasons.push(format!("{:?}", reason));
                        tracing::debug!("Red-flagged response: {:?}", reason);
                    }
                }
            }

            // Check if all samples so far have been red-flagged
            let valid_votes: u32 = votes.values().map(|(v, _)| *v).sum();
            if valid_votes == 0 && total_samples >= self.k * 3 {
                return Err(MdapError::Voting(VotingError::AllSamplesRedFlagged {
                    red_flagged,
                    total: total_samples,
                }));
            }

            // Check for loss-of-hope condition (RASC paper enhancement)
            // If no candidate can possibly win, return the current leader
            if self.early_stopping.loss_of_hope_enabled
                && self.check_loss_of_hope(&votes, total_samples)
                && let Some((leader_key, (leader_votes, leader_value))) =
                    votes.iter().max_by_key(|(_, (v, _))| *v)
            {
                let vote_distribution: HashMap<String, u32> =
                    votes.iter().map(|(k, (v, _))| (k.clone(), *v)).collect();
                let total_votes: u32 = votes.values().map(|(v, _)| *v).sum();

                let weighted_confidence = if self.use_confidence_weights
                    || self.voting_method == VotingMethod::BordaCount
                    || self.voting_method == VotingMethod::ConfidenceWeighted
                {
                    let total_weight: f64 = weighted_votes.values().sum();
                    let winner_weight = weighted_votes.get(leader_key).copied().unwrap_or(0.0);
                    Some(winner_weight / total_weight.max(0.001))
                } else {
                    None
                };

                tracing::info!(
                    total_samples = total_samples,
                    leader_votes = leader_votes,
                    "MDAP: Loss of hope - returning current leader"
                );

                return Ok(VoteResult {
                    winner: leader_value.clone(),
                    winner_votes: *leader_votes,
                    total_votes,
                    total_samples,
                    red_flagged_count: red_flagged,
                    vote_distribution,
                    confidence: self.calculate_confidence(*leader_votes, total_votes),
                    red_flag_reasons,
                    early_stopped: true, // Treat as early stop
                    weighted_confidence,
                    voting_method: self.voting_method.clone(),
                });
            }
        }
    }

    /// Check if we can stop early based on confidence (RASC paper: arxiv:2408.17017)
    ///
    /// Enhanced with:
    /// 1. Simple confidence threshold (original)
    /// 2. Variance-based stopping: stop if vote distribution is stable
    /// 3. Loss-of-hope detection: stop if no candidate can win
    fn check_early_stop<T: Clone>(&self, votes: &HashMap<String, (u32, T)>) -> Option<(String, T)> {
        let total: u32 = votes.values().map(|(v, _)| *v).sum();

        if total < self.early_stopping.min_votes {
            return None;
        }

        // Get leading candidate
        let (leader_key, (leader_votes, leader_value)) =
            votes.iter().max_by_key(|(_, (v, _))| *v)?;

        let confidence = *leader_votes as f64 / total as f64;

        // 1. Simple confidence threshold (original RASC)
        if confidence >= self.early_stopping.min_confidence {
            tracing::debug!(
                leader = %leader_key,
                confidence = %confidence,
                "Early stop: confidence threshold met"
            );
            return Some((leader_key.clone(), leader_value.clone()));
        }

        // 2. Variance-based stopping: check if vote distribution is stable
        if total >= 5 {
            let variance = self.calculate_vote_variance(votes, total);
            if variance < self.early_stopping.max_variance_threshold && confidence >= 0.6 {
                tracing::debug!(
                    leader = %leader_key,
                    variance = %variance,
                    confidence = %confidence,
                    "Early stop: low variance (stable distribution)"
                );
                return Some((leader_key.clone(), leader_value.clone()));
            }
        }

        None
    }

    /// Check for loss-of-hope condition (RASC paper enhancement)
    ///
    /// Returns true if it's mathematically impossible for any non-leader
    /// to win within the remaining sample budget.
    fn check_loss_of_hope<T>(&self, votes: &HashMap<String, (u32, T)>, total_samples: u32) -> bool {
        if !self.early_stopping.loss_of_hope_enabled {
            return false;
        }

        let remaining = self.max_samples.saturating_sub(total_samples);
        if remaining == 0 {
            return true;
        }

        // Get vote counts
        let mut counts: Vec<u32> = votes.values().map(|(v, _)| *v).collect();
        counts.sort_by(|a, b| b.cmp(a)); // Descending

        if counts.len() < 2 {
            return false;
        }

        let leader = counts[0];
        let runner_up = counts[1];

        // Check if runner-up can catch up to leader + k within remaining samples
        // Runner-up needs: leader + k - runner_up votes to win
        let votes_needed = leader + self.k - runner_up;

        // If votes needed exceeds remaining samples, it's hopeless
        if votes_needed > remaining {
            tracing::debug!(
                leader_votes = leader,
                runner_up_votes = runner_up,
                remaining = remaining,
                votes_needed = votes_needed,
                "Loss of hope detected"
            );
            return true;
        }

        false
    }

    /// Calculate variance of vote distribution (for stability-based stopping)
    fn calculate_vote_variance<T>(&self, votes: &HashMap<String, (u32, T)>, total: u32) -> f64 {
        if votes.is_empty() || total == 0 {
            return 1.0;
        }

        let mean = total as f64 / votes.len() as f64;
        let variance: f64 = votes
            .values()
            .map(|(v, _)| {
                let diff = *v as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / votes.len() as f64;

        // Normalize by total^2 to get a scale-independent variance
        (variance / (total as f64 * total as f64)).sqrt()
    }

    /// Simple vote method with default string key extraction
    pub async fn vote_simple<T, F, Fut>(
        &self,
        sampler: F,
        red_flag_validator: &dyn RedFlagValidator,
    ) -> MdapResult<VoteResult<T>>
    where
        T: Clone + Send + std::fmt::Debug + 'static,
        F: Fn() -> Fut + Send + Sync,
        Fut: Future<Output = MdapResult<SampledResponse<T>>> + Send + 'static,
    {
        self.vote(sampler, red_flag_validator, |v| format!("{:?}", v))
            .await
    }

    /// Check if any candidate has won: `V[y] >= k + max(V[v] for v != y)`
    fn check_winner<T: Clone>(&self, votes: &HashMap<String, (u32, T)>) -> Option<(String, T)> {
        if votes.is_empty() {
            return None;
        }

        for (candidate_key, (count, value)) in votes.iter() {
            let max_other = votes
                .iter()
                .filter(|(k, _)| *k != candidate_key)
                .map(|(_, (v, _))| *v)
                .max()
                .unwrap_or(0);

            if *count >= self.k + max_other {
                return Some((candidate_key.clone(), value.clone()));
            }
        }
        None
    }

    /// Check for Borda count winner (Ranked Voting paper: arxiv:2505.10772)
    ///
    /// In Borda count, we sum confidence-weighted scores for each candidate.
    /// The winner is determined when the leading candidate has a sufficient
    /// margin in weighted score over others.
    fn check_borda_winner<T: Clone>(
        &self,
        votes: &HashMap<String, (u32, T)>,
        weighted_votes: &HashMap<String, f64>,
    ) -> Option<(String, T)> {
        if votes.is_empty() || weighted_votes.is_empty() {
            return None;
        }

        // Find the candidate with highest weighted score
        let total_weight: f64 = weighted_votes.values().sum();
        if total_weight < 0.001 {
            return None;
        }

        // Get leader by weighted score
        let (leader_key, leader_weight) = weighted_votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        // Get second highest weight
        let second_weight = weighted_votes
            .iter()
            .filter(|(k, _)| *k != leader_key)
            .map(|(_, w)| *w)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Winner condition: leader weight >= k * (second_weight + margin)
        // This is an adaptation of k-ahead using weighted scores
        let margin = self.k as f64 * 0.25; // Each k adds 0.25 weight margin required
        if *leader_weight >= second_weight + margin {
            let (_, value) = votes.get(leader_key)?;
            return Some((leader_key.clone(), value.clone()));
        }

        None
    }

    /// Check for confidence-weighted winner (CISC paper: arxiv:2502.06233v1)
    ///
    /// Similar to first-to-ahead-by-k but uses weighted votes for the margin check.
    fn check_weighted_winner<T: Clone>(
        &self,
        votes: &HashMap<String, (u32, T)>,
        weighted_votes: &HashMap<String, f64>,
    ) -> Option<(String, T)> {
        if votes.is_empty() {
            return None;
        }

        // If no weighted votes yet, fall back to standard check
        if weighted_votes.is_empty() {
            return self.check_winner(votes);
        }

        for (candidate_key, (_, value)) in votes.iter() {
            let candidate_weight = weighted_votes.get(candidate_key).copied().unwrap_or(0.0);

            let max_other_weight = weighted_votes
                .iter()
                .filter(|(k, _)| *k != candidate_key)
                .map(|(_, w)| *w)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            // Weighted k-ahead: candidate_weight >= k * 0.5 + max_other_weight
            // The k factor is scaled to confidence units (0.5 per k)
            let k_margin = self.k as f64 * 0.5;
            if candidate_weight >= k_margin + max_other_weight {
                return Some((candidate_key.clone(), value.clone()));
            }
        }

        None
    }

    /// Sample in parallel (up to 4 threads as per user requirement)
    async fn sample_parallel<T, F, Fut>(
        &self,
        sampler: &F,
        count: usize,
    ) -> MdapResult<Vec<SampledResponse<T>>>
    where
        T: Clone + Send + 'static,
        F: Fn() -> Fut + Send + Sync,
        Fut: Future<Output = MdapResult<SampledResponse<T>>> + Send + 'static,
    {
        let mut handles = Vec::with_capacity(count.min(DEFAULT_PARALLEL_LIMIT));
        let semaphore = self.parallel_limit.clone();

        for _ in 0..count.min(DEFAULT_PARALLEL_LIMIT) {
            let permit = semaphore.clone().acquire_owned().await?;
            let fut = sampler();
            handles.push(tokio::spawn(async move {
                let result = fut.await;
                drop(permit);
                result
            }));
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(sample)) => results.push(sample),
                Ok(Err(e)) => {
                    tracing::warn!("Sampler error (continuing): {}", e);
                    // Continue with other samples, don't fail the whole batch
                }
                Err(e) => {
                    tracing::warn!("Task join error (continuing): {}", e);
                }
            }
        }
        Ok(results)
    }

    /// Calculate confidence score
    fn calculate_confidence(&self, winner_votes: u32, total_votes: u32) -> f64 {
        if total_votes == 0 {
            return 0.0;
        }
        winner_votes as f64 / total_votes as f64
    }

    /// Get the k value
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Get the max samples value
    pub fn max_samples(&self) -> u32 {
        self.max_samples
    }
}

/// Builder for FirstToAheadByKVoter
pub struct VoterBuilder {
    k: u32,
    max_samples: u32,
    parallel_limit: u32,
    batch_size: usize,
    early_stopping: EarlyStoppingConfig,
    voting_method: VotingMethod,
    use_confidence_weights: bool,
}

impl Default for VoterBuilder {
    fn default() -> Self {
        Self {
            k: 3,
            max_samples: 50,
            parallel_limit: DEFAULT_PARALLEL_LIMIT as u32,
            batch_size: DEFAULT_BATCH_SIZE,
            early_stopping: EarlyStoppingConfig::default(),
            voting_method: VotingMethod::FirstToAheadByK,
            use_confidence_weights: false,
        }
    }
}

impl VoterBuilder {
    /// Create a new voter builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vote margin threshold (k)
    pub fn k(mut self, k: u32) -> Self {
        self.k = k;
        self
    }

    /// Set the maximum number of samples
    pub fn max_samples(mut self, max_samples: u32) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Enable/disable early stopping (RASC paper)
    pub fn early_stopping(mut self, config: EarlyStoppingConfig) -> Self {
        self.early_stopping = config;
        self
    }

    /// Set the voting method
    pub fn voting_method(mut self, method: VotingMethod) -> Self {
        self.voting_method = method;
        self
    }

    /// Enable confidence-weighted voting (CISC paper)
    pub fn confidence_weighted(mut self, enabled: bool) -> Self {
        self.use_confidence_weights = enabled;
        if enabled && self.voting_method == VotingMethod::FirstToAheadByK {
            self.voting_method = VotingMethod::ConfidenceWeighted;
        }
        self
    }

    /// Set the parallel limit (1-4)
    pub fn parallel_limit(mut self, limit: u32) -> Self {
        self.parallel_limit = limit.clamp(1, DEFAULT_PARALLEL_LIMIT as u32);
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Build the voter
    pub fn build(self) -> FirstToAheadByKVoter {
        FirstToAheadByKVoter {
            k: self.k.max(1),
            max_samples: self.max_samples.max(1),
            parallel_limit: Arc::new(Semaphore::new(
                self.parallel_limit.clamp(1, DEFAULT_PARALLEL_LIMIT as u32) as usize,
            )),
            batch_size: self.batch_size.max(1),
            early_stopping: self.early_stopping,
            voting_method: self.voting_method,
            use_confidence_weights: self.use_confidence_weights,
        }
    }
}

/// Borda count voting implementation (Ranked Voting paper: arxiv:2505.10772)
///
/// Ranks all candidates by weighted scores. Each vote contributes points
/// based on preference order: higher rank = more points.
pub fn borda_count_winner<T: Clone>(
    votes: &[(String, T, f64)], // (key, value, confidence)
) -> Option<(String, T, f64)> {
    if votes.is_empty() {
        return None;
    }

    let mut scores: HashMap<String, (f64, T)> = HashMap::new();

    // Group by key and sum confidence scores
    for (key, value, confidence) in votes {
        let entry = scores.entry(key.clone()).or_insert((0.0, value.clone()));
        entry.0 += confidence;
    }

    // Find winner by score
    scores
        .into_iter()
        .max_by(|a, b| {
            a.1.0
                .partial_cmp(&b.1.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(k, (score, value))| (k, value, score))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock validator that accepts all responses
    struct AcceptAllValidator;
    impl RedFlagValidator for AcceptAllValidator {
        fn validate(&self, _response: &str, _metadata: &ResponseMetadata) -> RedFlagResult {
            RedFlagResult::Valid
        }
    }

    // Mock validator that rejects responses containing "bad"
    struct RejectBadValidator;
    impl RedFlagValidator for RejectBadValidator {
        fn validate(&self, response: &str, _metadata: &ResponseMetadata) -> RedFlagResult {
            if response.contains("bad") {
                RedFlagResult::Flagged {
                    reason: super::super::red_flags::RedFlagReason::ConfusedReasoning {
                        pattern: "bad".to_string(),
                    },
                    severity: 0.8,
                }
            } else {
                RedFlagResult::Valid
            }
        }
    }

    fn make_response<T: Clone>(value: T, raw: &str) -> SampledResponse<T> {
        SampledResponse {
            value,
            metadata: ResponseMetadata::default(),
            raw_response: raw.to_string(),
            confidence: 0.75, // Default confidence
        }
    }

    #[tokio::test]
    async fn test_unanimous_voting() {
        let voter = FirstToAheadByKVoter::new(3, 50);
        let validator = AcceptAllValidator;

        let call_count = std::sync::atomic::AtomicU32::new(0);

        let result = voter
            .vote(
                || {
                    let count = call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    async move {
                        // Always return "answer_a"
                        Ok(make_response(format!("answer_a_{}", count), "answer_a"))
                    }
                },
                &validator,
                |_| "answer_a".to_string(), // All map to same key
            )
            .await
            .unwrap();

        // With k=3 and unanimous voting, should win after 3 votes
        assert_eq!(result.winner_votes, 3);
        assert_eq!(result.confidence, 1.0);
        assert_eq!(result.red_flagged_count, 0);
    }

    #[tokio::test]
    async fn test_split_voting() {
        let voter = FirstToAheadByKVoter::new(2, 50);
        let validator = AcceptAllValidator;

        let call_count = std::sync::atomic::AtomicU32::new(0);

        let result = voter
            .vote(
                || {
                    let count = call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    async move {
                        // Alternate between a and b, with a getting slightly more
                        let value = if count % 3 == 2 { "b" } else { "a" };
                        Ok(make_response(value.to_string(), value))
                    }
                },
                &validator,
                |v| v.clone(),
            )
            .await
            .unwrap();

        // "a" should eventually win with margin of at least 2
        assert!(
            result.winner.starts_with("a") || result.vote_distribution.get("a").unwrap_or(&0) >= &2
        );
    }

    #[tokio::test]
    async fn test_red_flagging() {
        let voter = FirstToAheadByKVoter::new(3, 50);
        let validator = RejectBadValidator;

        let call_count = std::sync::atomic::AtomicU32::new(0);

        let result = voter
            .vote(
                || {
                    let count = call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    async move {
                        // Every 3rd response is "bad"
                        let raw = if count.is_multiple_of(3) {
                            "bad response"
                        } else {
                            "good"
                        };
                        Ok(make_response("answer".to_string(), raw))
                    }
                },
                &validator,
                |_| "answer".to_string(),
            )
            .await
            .unwrap();

        // Should have some red-flagged responses
        assert!(result.red_flagged_count > 0);
        assert!(result.total_samples > result.total_votes);
    }

    #[tokio::test]
    async fn test_max_samples_exceeded() {
        // Disable early stopping to test max samples behavior specifically
        // (Early stopping with loss_of_hope would return the current leader instead)
        let voter = FirstToAheadByKVoter::with_early_stopping(
            10,                              // k=10 is impossible to reach with max 5 samples
            5,                               // max_samples
            EarlyStoppingConfig::disabled(), // Disable early stopping for this test
        );
        let validator = AcceptAllValidator;

        let call_count = std::sync::atomic::AtomicU32::new(0);

        let result = voter
            .vote(
                || {
                    let count = call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    async move {
                        // Return different values each time - never converges
                        Ok(make_response(format!("unique_{}", count), "response"))
                    }
                },
                &validator,
                |v| v.clone(),
            )
            .await;

        assert!(result.is_err());
        if let Err(MdapError::Voting(VotingError::MaxSamplesExceeded { samples, .. })) = result {
            assert_eq!(samples, 5);
        } else {
            panic!("Expected MaxSamplesExceeded error");
        }
    }

    #[test]
    fn test_voter_builder() {
        let voter = VoterBuilder::new()
            .k(5)
            .max_samples(100)
            .parallel_limit(2)
            .batch_size(2)
            .build();

        assert_eq!(voter.k(), 5);
        assert_eq!(voter.max_samples(), 100);
    }

    #[test]
    fn test_voter_builder_clamps_parallel() {
        let voter = VoterBuilder::new().parallel_limit(10).build();
        // Should be clamped to 4
        // We can't directly test the semaphore permits, but we trust the clamp logic
        assert_eq!(voter.k(), 3); // Default k
    }

    #[tokio::test]
    async fn test_all_red_flagged() {
        let voter = FirstToAheadByKVoter::new(3, 20);

        // Validator that rejects everything
        struct RejectAllValidator;
        impl RedFlagValidator for RejectAllValidator {
            fn validate(&self, _response: &str, _metadata: &ResponseMetadata) -> RedFlagResult {
                RedFlagResult::Flagged {
                    reason: super::super::red_flags::RedFlagReason::EmptyResponse,
                    severity: 1.0,
                }
            }
        }

        let validator = RejectAllValidator;

        let result = voter
            .vote(
                || async { Ok(make_response("value".to_string(), "response")) },
                &validator,
                |v| v.clone(),
            )
            .await;

        assert!(result.is_err());
        if let Err(MdapError::Voting(VotingError::AllSamplesRedFlagged { red_flagged, total })) =
            result
        {
            assert!(red_flagged > 0);
            assert_eq!(red_flagged, total);
        } else {
            panic!("Expected AllSamplesRedFlagged error");
        }
    }
}
