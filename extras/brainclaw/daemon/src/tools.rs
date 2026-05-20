//! Tool registry builder — conditionally registers tools based on config.

use brainwires_tools::ToolRegistry;

use crate::config::ToolsSection;

/// Build a [`ToolRegistry`] based on the tools configuration.
///
/// Registers tool groups from the `enabled` list, minus any in `disabled`.
/// If `bash_allowed` is false, the bash tool group is skipped even if listed
/// in `enabled`.
pub fn build_tool_registry(config: &ToolsSection) -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    for group in &config.enabled {
        // Skip if explicitly disabled
        if config.disabled.contains(group) {
            continue;
        }

        // Skip bash if not allowed
        if group == "bash" && !config.bash_allowed {
            continue;
        }

        register_tool_group(&mut registry, group);
    }

    registry
}

/// Register all tools belonging to a named group.
#[cfg(feature = "native-tools")]
fn register_tool_group(registry: &mut ToolRegistry, group: &str) {
    use brainwires_tools::{BashTool, FileOpsTool, GitTool, SearchTool, ValidationTool, WebTool};

    match group {
        "bash" => registry.register_tools(BashTool::get_tools()),
        "files" => registry.register_tools(FileOpsTool::get_tools()),
        "git" => registry.register_tools(GitTool::get_tools()),
        "search" => registry.register_tools(SearchTool::get_tools()),
        "web" => registry.register_tools(WebTool::get_tools()),
        "validation" => registry.register_tools(ValidationTool::get_tools()),

        #[cfg(feature = "email")]
        "email" => {
            use brainwires_tools::EmailTool;
            registry.register_tools(EmailTool::get_tools());
        }
        #[cfg(not(feature = "email"))]
        "email" => {
            tracing::warn!(
                "Tool group 'email' requested but BrainClaw was not compiled with the 'email' feature"
            );
        }

        #[cfg(feature = "calendar")]
        "calendar" => {
            use brainwires_tools::CalendarTool;
            registry.register_tools(CalendarTool::get_tools());
        }
        #[cfg(not(feature = "calendar"))]
        "calendar" => {
            tracing::warn!(
                "Tool group 'calendar' requested but BrainClaw was not compiled with the 'calendar' feature"
            );
        }

        #[cfg(feature = "rag")]
        "semantic-search" => {
            use brainwires_tools::SemanticSearchTool;
            registry.register_tools(SemanticSearchTool::get_tools());
        }
        #[cfg(not(feature = "rag"))]
        "semantic-search" => {
            tracing::warn!(
                "Tool group 'semantic-search' requested but BrainClaw was not compiled with the 'rag' feature"
            );
        }

        #[cfg(feature = "browser")]
        "browser" => {
            use brainwires_tools::BrowserTool;
            registry.register_tools(BrowserTool::get_tools());
        }
        #[cfg(not(feature = "browser"))]
        "browser" => {
            tracing::warn!(
                "Tool group 'browser' requested but BrainClaw was not compiled with the 'browser' feature"
            );
        }

        other => {
            tracing::warn!(group = %other, "Unknown tool group, skipping");
        }
    }
}

/// Fallback when native tools are not compiled in.
#[cfg(not(feature = "native-tools"))]
fn register_tool_group(_registry: &mut ToolRegistry, group: &str) {
    tracing::warn!(
        group = %group,
        "Tool group requested but native-tools feature is not enabled"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_with_defaults() {
        let config = ToolsSection::default();
        let registry = build_tool_registry(&config);
        // With native-tools feature, should have registered tools from all 6 groups
        #[cfg(feature = "native-tools")]
        assert!(!registry.is_empty());
        #[cfg(not(feature = "native-tools"))]
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_build_with_empty_enabled() {
        let config = ToolsSection {
            enabled: vec![],
            disabled: vec![],
            bash_allowed: true,
        };
        let registry = build_tool_registry(&config);
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_build_with_disabled_overrides() {
        let config = ToolsSection {
            enabled: vec!["bash".to_string(), "files".to_string()],
            disabled: vec!["bash".to_string()],
            bash_allowed: true,
        };
        let registry = build_tool_registry(&config);
        // bash should be skipped because it's in disabled
        #[cfg(feature = "native-tools")]
        {
            // Should have file tools but not bash tools
            assert!(registry.get("execute_command").is_none());
            assert!(registry.get("read_file").is_some());
        }
        let _ = registry;
    }

    #[test]
    fn test_build_bash_not_allowed() {
        let config = ToolsSection {
            enabled: vec!["bash".to_string(), "files".to_string()],
            disabled: vec![],
            bash_allowed: false,
        };
        let registry = build_tool_registry(&config);
        #[cfg(feature = "native-tools")]
        {
            // bash should be skipped because bash_allowed is false
            assert!(registry.get("execute_command").is_none());
            assert!(registry.get("read_file").is_some());
        }
        let _ = registry;
    }

    #[test]
    fn test_build_unknown_group_is_tolerated() {
        let config = ToolsSection {
            enabled: vec!["nonexistent_group".to_string()],
            disabled: vec![],
            bash_allowed: true,
        };
        // Should not panic
        let registry = build_tool_registry(&config);
        assert_eq!(registry.len(), 0);
    }
}
