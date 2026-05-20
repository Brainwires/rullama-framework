//! Voting Consensus Example
//!
//! Demonstrates the building blocks of the MAKER voting system:
//! creating sampled responses with metadata, configuring red-flag
//! validation, and validating responses against reliability criteria.
//!
//! Run with: `cargo run -p brainwires-agent --features mdap --example voting_consensus`

use brainwires_mdap::red_flags::{
    OutputFormat, RedFlagConfig, RedFlagValidator, StandardRedFlagValidator,
};
use brainwires_mdap::voting::{ResponseMetadata, SampledResponse, VoteResult, VotingMethod};
use std::collections::HashMap;

fn main() {
    println!("=== MAKER Voting Consensus Building Blocks ===\n");

    // 1. Create sampled responses with metadata
    println!("--- 1. Creating Sampled Responses ---\n");

    let responses = [
        SampledResponse::new(
            "Paris".to_string(),
            ResponseMetadata {
                token_count: 12,
                response_time_ms: 150,
                format_valid: true,
                finish_reason: Some("stop".to_string()),
                model: Some("claude-sonnet".to_string()),
            },
            "The capital of France is Paris.".to_string(),
        ),
        SampledResponse::with_confidence(
            "Paris".to_string(),
            ResponseMetadata {
                token_count: 8,
                response_time_ms: 120,
                format_valid: true,
                finish_reason: Some("stop".to_string()),
                model: Some("claude-sonnet".to_string()),
            },
            "Paris".to_string(),
            0.95,
        ),
        SampledResponse::new(
            "Wait, actually Lyon".to_string(),
            ResponseMetadata {
                token_count: 45,
                response_time_ms: 300,
                format_valid: true,
                finish_reason: Some("stop".to_string()),
                model: None,
            },
            "Wait, I think it might be Lyon. Actually, let me reconsider...".to_string(),
        ),
        SampledResponse::new(
            "Paris".to_string(),
            ResponseMetadata {
                token_count: 900,
                response_time_ms: 2000,
                format_valid: true,
                finish_reason: Some("length".to_string()),
                model: Some("gpt-4o".to_string()),
            },
            "A very long response that was truncated...".to_string(),
        ),
    ];

    for (i, resp) in responses.iter().enumerate() {
        println!(
            "  Response {}: value={:?}, tokens={}, confidence={:.2}, model={:?}",
            i + 1,
            resp.value,
            resp.metadata.token_count,
            resp.confidence,
            resp.metadata.model,
        );
    }

    // 2. Compare RedFlagConfig presets
    println!("\n--- 2. RedFlagConfig Presets ---\n");

    let strict = RedFlagConfig::strict();
    let relaxed = RedFlagConfig::relaxed();

    println!("  Strict config:");
    println!("    max_response_tokens: {}", strict.max_response_tokens);
    println!("    require_exact_format: {}", strict.require_exact_format);
    println!("    flag_self_correction: {}", strict.flag_self_correction);
    println!(
        "    confusion_patterns: {} patterns",
        strict.confusion_patterns.len()
    );
    println!("    min_response_length: {}", strict.min_response_length);
    println!(
        "    max_empty_line_ratio: {:.1}",
        strict.max_empty_line_ratio
    );

    println!("  Relaxed config:");
    println!("    max_response_tokens: {}", relaxed.max_response_tokens);
    println!("    require_exact_format: {}", relaxed.require_exact_format);
    println!("    flag_self_correction: {}", relaxed.flag_self_correction);
    println!(
        "    confusion_patterns: {} patterns",
        relaxed.confusion_patterns.len()
    );
    println!("    min_response_length: {}", relaxed.min_response_length);
    println!(
        "    max_empty_line_ratio: {:.1}",
        relaxed.max_empty_line_ratio
    );

    // 3. Validate responses with the strict validator
    println!("\n--- 3. Red-Flag Validation (Strict) ---\n");

    let validator = StandardRedFlagValidator::strict();

    for (i, resp) in responses.iter().enumerate() {
        let result = validator.validate(&resp.raw_response, &resp.metadata);
        let status = if result.is_valid() {
            "VALID".to_string()
        } else {
            format!(
                "FLAGGED: {}",
                match &result {
                    brainwires_mdap::red_flags::RedFlagResult::Flagged { reason, severity } => {
                        format!("{} (severity: {:.2})", reason, severity)
                    }
                    _ => unreachable!(),
                }
            )
        };
        println!("  Response {}: {}", i + 1, status);
    }

    // 4. Validate with relaxed config
    println!("\n--- 4. Red-Flag Validation (Relaxed) ---\n");

    let relaxed_validator = StandardRedFlagValidator::new(RedFlagConfig::relaxed(), None);

    for (i, resp) in responses.iter().enumerate() {
        let result = relaxed_validator.validate(&resp.raw_response, &resp.metadata);
        let status = if result.is_valid() {
            "VALID"
        } else {
            "FLAGGED"
        };
        println!("  Response {}: {}", i + 1, status);
    }

    // 5. Format-aware validation
    println!("\n--- 5. Format-Aware Validation ---\n");

    let json_validator = StandardRedFlagValidator::with_format(OutputFormat::Json);
    let one_of_validator = StandardRedFlagValidator::with_format(OutputFormat::OneOf(vec![
        "Paris".to_string(),
        "London".to_string(),
        "Berlin".to_string(),
    ]));

    let json_response = r#"{"answer": "Paris"}"#;
    let plain_response = "Paris";

    let json_meta = ResponseMetadata {
        token_count: 20,
        response_time_ms: 100,
        format_valid: true,
        finish_reason: Some("stop".to_string()),
        model: None,
    };

    println!(
        "  JSON validator on '{}': {}",
        json_response,
        if json_validator
            .validate(json_response, &json_meta)
            .is_valid()
        {
            "VALID"
        } else {
            "FLAGGED"
        },
    );
    println!(
        "  JSON validator on '{}': {}",
        plain_response,
        if json_validator
            .validate(plain_response, &json_meta)
            .is_valid()
        {
            "VALID"
        } else {
            "FLAGGED"
        },
    );
    println!(
        "  OneOf validator on '{}': {}",
        plain_response,
        if one_of_validator
            .validate(plain_response, &json_meta)
            .is_valid()
        {
            "VALID"
        } else {
            "FLAGGED"
        },
    );
    println!(
        "  OneOf validator on 'Tokyo': {}",
        if one_of_validator.validate("Tokyo", &json_meta).is_valid() {
            "VALID"
        } else {
            "FLAGGED"
        },
    );

    // 6. Show VoteResult structure
    println!("\n--- 6. VoteResult Structure ---\n");

    let mut distribution = HashMap::new();
    distribution.insert("Paris".to_string(), 4);
    distribution.insert("Lyon".to_string(), 1);

    let vote_result: VoteResult<String> = VoteResult {
        winner: "Paris".to_string(),
        winner_votes: 4,
        total_votes: 5,
        total_samples: 7,
        red_flagged_count: 2,
        vote_distribution: distribution,
        confidence: 0.80,
        red_flag_reasons: vec![
            "Response too long: 900 tokens > 750 limit".to_string(),
            "Self-correction detected: 'Wait,'".to_string(),
        ],
        early_stopped: false,
        weighted_confidence: Some(0.85),
        voting_method: VotingMethod::FirstToAheadByK,
    };

    println!("  Winner: {:?}", vote_result.winner);
    println!(
        "  Winner votes: {}/{}",
        vote_result.winner_votes, vote_result.total_votes
    );
    println!(
        "  Total samples (incl. red-flagged): {}",
        vote_result.total_samples
    );
    println!("  Red-flagged: {}", vote_result.red_flagged_count);
    println!("  Confidence: {:.0}%", vote_result.confidence * 100.0);
    println!(
        "  Weighted confidence: {:.0}%",
        vote_result.weighted_confidence.unwrap_or(0.0) * 100.0
    );
    println!("  Early stopped: {}", vote_result.early_stopped);
    println!("  Vote distribution:");
    for (candidate, votes) in &vote_result.vote_distribution {
        println!("    {}: {} votes", candidate, votes);
    }
    println!("  Red-flag reasons:");
    for reason in &vote_result.red_flag_reasons {
        println!("    - {}", reason);
    }

    println!("\n=== Done ===");
}
