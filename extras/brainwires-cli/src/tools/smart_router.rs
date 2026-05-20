//! Smart Tool Router - CLI wrapper
//!
//! Re-exports all generic routing functions from the framework crate.
//! CLI-specific variants that integrate with the Skill system and LocalRouter
//! are defined here (inference dep lives in CLI, not in brainwires-tools).

// Re-export all framework smart_router functions
#[allow(hidden_glob_reexports)]
pub use brainwires_tool_runtime::smart_router::*;

// ── CLI-specific: inference-integrated variants ───────────────────────────

use crate::types::message::Message;
use brainwires::reasoning::LocalRouter;
use brainwires_tool_runtime::{Tool, ToolCategory};

/// Analyze a query using local inference with keyword fallback
///
/// Attempts semantic classification via a provider first. On failure,
/// falls back to keyword-based pattern matching.
pub async fn analyze_query_with_local(
    query: &str,
    local_router: Option<&LocalRouter>,
) -> Vec<ToolCategory> {
    if let Some(router) = local_router
        && let Some(result) = router.classify(query).await
    {
        let mut categories = result.categories;
        if !categories.contains(&ToolCategory::FileOps) {
            categories.push(ToolCategory::FileOps);
        }
        return categories;
    }
    analyze_query(query)
}

/// Analyze messages using local inference with keyword fallback
pub async fn analyze_messages_with_local(
    messages: &[Message],
    local_router: Option<&LocalRouter>,
) -> Vec<ToolCategory> {
    let context = get_context_for_analysis(messages);
    analyze_query_with_local(&context, local_router).await
}

/// Get smart-routed tools using local inference classification
pub async fn get_smart_tools_with_local(
    messages: &[Message],
    local_router: Option<&LocalRouter>,
) -> Vec<Tool> {
    let categories = analyze_messages_with_local(messages, local_router).await;
    let registry = brainwires_tool_builtins::registry_with_builtins();
    get_tools_for_categories(&registry, &categories)
}

// ── CLI-specific: skill-integrated variants ───────────────────────────────

use brainwires_skills::metadata::SkillMatch;
use brainwires_skills::router::SkillRouter;

/// Analyze query and return both tool categories and skill matches
pub async fn analyze_with_skills(
    messages: &[Message],
    skill_router: &SkillRouter,
) -> (Vec<ToolCategory>, Vec<SkillMatch>) {
    let context = get_context_for_analysis(messages);
    let categories = analyze_query(&context);
    let skill_matches = skill_router.match_skills(&context).await;
    (categories, skill_matches)
}

/// Analyze query with local inference and skills
pub async fn analyze_with_skills_and_local(
    messages: &[Message],
    skill_router: &SkillRouter,
    local_router: Option<&LocalRouter>,
) -> (Vec<ToolCategory>, Vec<SkillMatch>) {
    let context = get_context_for_analysis(messages);
    let categories = analyze_query_with_local(&context, local_router).await;
    let skill_matches = skill_router.match_skills(&context).await;
    (categories, skill_matches)
}

/// Get smart tools and skill suggestions for the given messages
pub async fn get_smart_tools_with_skills(
    messages: &[Message],
    skill_router: &SkillRouter,
) -> (Vec<Tool>, Vec<SkillMatch>) {
    let (categories, skill_matches) = analyze_with_skills(messages, skill_router).await;
    let registry = brainwires_tool_builtins::registry_with_builtins();
    let tools = get_tools_for_categories(&registry, &categories);
    (tools, skill_matches)
}
