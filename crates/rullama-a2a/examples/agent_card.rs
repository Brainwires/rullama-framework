//! Agent Card Construction Example
//!
//! Demonstrates building A2A AgentCards with capabilities, skills,
//! security schemes, and round-trip JSON serialization.

use rullama_a2a::{
    AgentCapabilities, AgentCard, AgentProvider, AgentSkill, HttpAuthSecurityScheme, SecurityScheme,
};
use std::collections::HashMap;

fn main() {
    // 1. Build a full AgentCard with all fields populated
    println!("=== Full AgentCard ===\n");

    let mut security_schemes = HashMap::new();
    security_schemes.insert(
        "bearer_auth".to_string(),
        SecurityScheme {
            api_key: None,
            http_auth: Some(HttpAuthSecurityScheme {
                scheme: "Bearer".to_string(),
                bearer_format: Some("JWT".to_string()),
                description: Some("JWT Bearer token authentication".to_string()),
            }),
            oauth2: None,
            open_id_connect: None,
            mtls: None,
        },
    );

    let full_card = AgentCard {
        name: "code-review-agent".to_string(),
        description: "An autonomous code review agent that analyzes pull requests \
                      and provides actionable feedback."
            .to_string(),
        version: "1.2.0".to_string(),
        supported_interfaces: vec![],
        capabilities: AgentCapabilities {
            streaming: Some(true),
            push_notifications: Some(true),
            extended_agent_card: Some(false),
            extensions: None,
        },
        skills: vec![
            AgentSkill {
                id: "review-pr".to_string(),
                name: "Pull Request Review".to_string(),
                description: "Analyzes code diffs and produces structured review comments."
                    .to_string(),
                tags: vec![
                    "code-review".to_string(),
                    "static-analysis".to_string(),
                    "security".to_string(),
                ],
                examples: Some(vec![
                    "Review this PR for security issues".to_string(),
                    "Check for performance regressions in the diff".to_string(),
                ]),
                input_modes: None,
                output_modes: None,
                security_requirements: None,
            },
            AgentSkill {
                id: "suggest-fix".to_string(),
                name: "Suggest Fix".to_string(),
                description: "Generates code fix suggestions for identified issues.".to_string(),
                tags: vec!["code-generation".to_string(), "refactoring".to_string()],
                examples: None,
                input_modes: Some(vec!["text/plain".to_string()]),
                output_modes: Some(vec![
                    "text/plain".to_string(),
                    "application/json".to_string(),
                ]),
                security_requirements: None,
            },
        ],
        default_input_modes: vec!["text/plain".to_string(), "application/json".to_string()],
        default_output_modes: vec!["text/plain".to_string(), "application/json".to_string()],
        provider: Some(AgentProvider {
            url: "https://brainwires.dev".to_string(),
            organization: "Brainwires".to_string(),
        }),
        security_schemes: Some(security_schemes),
        security_requirements: None,
        documentation_url: Some("https://docs.brainwires.dev/agents/code-review".to_string()),
        icon_url: Some("https://brainwires.dev/icons/code-review.svg".to_string()),
        signatures: None,
    };

    // 2. Serialize to JSON
    let json = serde_json::to_string_pretty(&full_card).expect("serialize full card");
    println!("{json}\n");

    // 3. Round-trip: deserialize and verify
    println!("=== Round-Trip Verification ===\n");

    let deserialized: AgentCard = serde_json::from_str(&json).expect("deserialize full card");
    assert_eq!(
        deserialized.name, full_card.name,
        "round-trip name mismatch"
    );
    assert_eq!(
        deserialized.skills.len(),
        full_card.skills.len(),
        "round-trip skills count mismatch"
    );
    println!(
        "Round-trip OK: name = {:?}, skills = {}\n",
        deserialized.name,
        deserialized.skills.len()
    );

    // 4. Build a minimal AgentCard (bare essentials only)
    println!("=== Minimal AgentCard ===\n");

    let minimal_card = AgentCard {
        name: "echo-agent".to_string(),
        description: "A minimal agent that echoes messages back.".to_string(),
        version: "0.7.0".to_string(),
        supported_interfaces: vec![],
        capabilities: AgentCapabilities::default(),
        skills: vec![],
        default_input_modes: vec!["text/plain".to_string()],
        default_output_modes: vec!["text/plain".to_string()],
        provider: None,
        security_schemes: None,
        security_requirements: None,
        documentation_url: None,
        icon_url: None,
        signatures: None,
    };

    let minimal_json = serde_json::to_string_pretty(&minimal_card).expect("serialize minimal card");
    println!("{minimal_json}\n");

    // 5. Side-by-side summary
    println!("=== Comparison ===\n");
    println!("{:<20} {:<25} {:<25}", "Field", "Full Card", "Minimal Card");
    println!("{:-<70}", "");
    println!(
        "{:<20} {:<25} {:<25}",
        "name", full_card.name, minimal_card.name
    );
    println!(
        "{:<20} {:<25} {:<25}",
        "version", full_card.version, minimal_card.version
    );
    println!(
        "{:<20} {:<25} {:<25}",
        "skills",
        full_card.skills.len(),
        minimal_card.skills.len()
    );
    println!(
        "{:<20} {:<25} {:<25}",
        "streaming",
        format!("{:?}", full_card.capabilities.streaming),
        format!("{:?}", minimal_card.capabilities.streaming)
    );
    println!(
        "{:<20} {:<25} {:<25}",
        "provider",
        full_card
            .provider
            .as_ref()
            .map(|p| p.organization.as_str())
            .unwrap_or("None"),
        minimal_card
            .provider
            .as_ref()
            .map(|p| p.organization.as_str())
            .unwrap_or("None")
    );
    println!(
        "{:<20} {:<25} {:<25}",
        "security_schemes",
        full_card.security_schemes.is_some(),
        minimal_card.security_schemes.is_some()
    );
}
