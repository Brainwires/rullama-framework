use super::agent::{AgentCapabilities, CapabilityProfile};
use super::capabilities::{GitOperation, NetworkCapabilities, ToolCategory};
use super::path_pattern::PathPattern;

#[test]
fn test_path_pattern_matching() {
    let pattern = PathPattern::new("**/.env*");
    assert!(pattern.matches(".env"));
    assert!(pattern.matches(".env.local"));
    assert!(pattern.matches("config/.env"));

    let pattern = PathPattern::new("src/**/*.rs");
    assert!(pattern.matches("src/main.rs"));
    assert!(pattern.matches("src/lib/mod.rs"));
}

#[test]
fn test_full_access_pattern() {
    let pattern = PathPattern::new("**/*");
    assert!(
        pattern.matches("index.html"),
        "**/* should match root files"
    );
    assert!(pattern.matches("./index.html"), "**/* should match ./file");
    assert!(
        pattern.matches("src/main.rs"),
        "**/* should match nested files"
    );
}

#[test]
fn test_tool_categorization() {
    assert_eq!(
        AgentCapabilities::categorize_tool("read_file"),
        ToolCategory::FileRead
    );
    assert_eq!(
        AgentCapabilities::categorize_tool("write_file"),
        ToolCategory::FileWrite
    );
    assert_eq!(
        AgentCapabilities::categorize_tool("git_status"),
        ToolCategory::Git
    );
    assert_eq!(
        AgentCapabilities::categorize_tool("git_force_push"),
        ToolCategory::GitDestructive
    );
    assert_eq!(
        AgentCapabilities::categorize_tool("execute_command"),
        ToolCategory::Bash
    );
}

#[test]
fn test_allows_tool() {
    let caps = AgentCapabilities::default();

    // Default only allows FileRead, Search, and Web
    assert!(caps.allows_tool("read_file"));
    assert!(caps.allows_tool("search_code"));
    assert!(!caps.allows_tool("write_file"));
    assert!(!caps.allows_tool("execute_command"));
}

#[test]
fn test_denied_tools() {
    let mut caps = AgentCapabilities::default();
    caps.tools.denied_tools.insert("read_file".to_string());

    // Even though FileRead is allowed, this specific tool is denied
    assert!(!caps.allows_tool("read_file"));
    assert!(caps.allows_tool("list_directory")); // Other FileRead tools still work
}

#[test]
fn test_domain_matching() {
    let caps = AgentCapabilities {
        network: NetworkCapabilities {
            allowed_domains: vec!["github.com".to_string(), "*.github.com".to_string()],
            ..Default::default()
        },
        ..Default::default()
    };

    assert!(caps.allows_domain("github.com"));
    assert!(caps.allows_domain("api.github.com"));
    assert!(caps.allows_domain("raw.github.com"));
    assert!(!caps.allows_domain("gitlab.com"));
}

#[test]
fn test_git_operations() {
    let caps = AgentCapabilities::default();

    // Default allows read-only git ops
    assert!(caps.allows_git_op(GitOperation::Status));
    assert!(caps.allows_git_op(GitOperation::Diff));
    assert!(!caps.allows_git_op(GitOperation::Push));
    assert!(!caps.allows_git_op(GitOperation::ForcePush));
}

#[test]
fn test_read_only_profile() {
    let caps = AgentCapabilities::read_only();

    assert!(caps.allows_tool("read_file"));
    assert!(caps.allows_tool("search_code"));
    assert!(!caps.allows_tool("write_file"));
    assert!(!caps.allows_tool("execute_command"));
    assert!(!caps.allows_domain("github.com"));
    assert!(!caps.can_spawn_agent(0, 0));
}

#[test]
fn test_standard_dev_profile() {
    let caps = AgentCapabilities::standard_dev();

    assert!(caps.allows_tool("read_file"));
    assert!(caps.allows_tool("write_file"));
    assert!(caps.allows_tool("git_status"));
    assert!(!caps.allows_tool("execute_code"));
    assert!(caps.requires_approval("delete_file"));
    assert!(caps.requires_approval("execute_command"));
    assert!(caps.allows_domain("github.com"));
    assert!(caps.allows_domain("api.github.com"));
    assert!(!caps.allows_domain("malware.com"));
    assert!(caps.can_spawn_agent(0, 0));
    assert!(caps.can_spawn_agent(2, 1));
    assert!(!caps.can_spawn_agent(3, 0));
    assert!(!caps.can_spawn_agent(0, 2));
}

#[test]
fn test_full_access_profile() {
    let caps = AgentCapabilities::full_access();

    assert!(caps.allows_tool("read_file"));
    assert!(caps.allows_tool("write_file"));
    assert!(caps.allows_tool("execute_code"));
    assert!(caps.allows_tool("execute_command"));
    assert!(caps.allows_domain("any-domain.com"));
    assert!(caps.can_spawn_agent(9, 4));
}

#[test]
fn test_derive_child() {
    let parent = AgentCapabilities::standard_dev();
    let child = parent.derive_child();

    assert_eq!(child.spawning.max_depth, parent.spawning.max_depth - 1);
    assert!(!child.spawning.can_elevate);
    assert_ne!(child.capability_id, parent.capability_id);
}

#[test]
fn test_capability_intersection() {
    let full = AgentCapabilities::full_access();
    let read_only = AgentCapabilities::read_only();

    let intersected = full.intersect(&read_only);

    assert!(intersected.allows_tool("read_file"));
    assert!(!intersected.allows_tool("write_file"));
    assert!(!intersected.can_spawn_agent(0, 0));
}

#[test]
fn test_profile_parsing() {
    assert_eq!(
        CapabilityProfile::parse("read_only"),
        Some(CapabilityProfile::ReadOnly)
    );
    assert_eq!(
        CapabilityProfile::parse("standard_dev"),
        Some(CapabilityProfile::StandardDev)
    );
    assert_eq!(
        CapabilityProfile::parse("full_access"),
        Some(CapabilityProfile::FullAccess)
    );
    assert_eq!(CapabilityProfile::parse("invalid"), None);
}
