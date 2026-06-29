//! Plan Parser - Extract tasks from plan content
//!
//! Parses structured plan content to extract numbered steps
//! that can be converted into tasks.

use rullama_core::task::{Task, TaskPriority};
use regex::Regex;
use std::sync::LazyLock;

/// A parsed step from plan content
#[derive(Debug, Clone)]
pub struct ParsedStep {
    /// Step number (1-based)
    pub number: usize,
    /// Step description
    pub description: String,
    /// Indentation level (0 = root, 1 = substep, etc.)
    pub indent_level: usize,
    /// Whether this step is marked as high priority
    pub is_priority: bool,
}

/// Parse plan content into structured steps
pub fn parse_plan_steps(content: &str) -> Vec<ParsedStep> {
    let mut steps = Vec::new();

    // Patterns for numbered steps:
    // - "1. Step description"
    // - "1) Step description"
    // - "Step 1: Description"
    // - "- Step description" (bullets)
    // - "  1. Substep" (indented)

    static NUMBERED_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^(\s*)(\d+)[.)]\s*(.+)$").expect("valid regex"));
    static STEP_COLON_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^(\s*)(?:Step\s+)?(\d+):\s*(.+)$").expect("valid regex"));
    static BULLET_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^(\s*)[-*]\s+(.+)$").expect("valid regex"));
    let numbered_re = &*NUMBERED_RE;
    let step_colon_re = &*STEP_COLON_RE;
    let bullet_re = &*BULLET_RE;

    let mut current_number = 0;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Try numbered format: "1. Description" or "1) Description"
        if let Some(caps) = numbered_re.captures(line) {
            let indent = caps.get(1).map(|m| m.as_str().len()).unwrap_or(0);
            let _num: usize = caps
                .get(2)
                .expect("group 2 always present in match")
                .as_str()
                .parse()
                .unwrap_or(0);
            let desc = caps
                .get(3)
                .expect("group 3 always present in match")
                .as_str()
                .trim();

            current_number += 1;
            let indent_level = indent / 2; // Assume 2-space indentation

            steps.push(ParsedStep {
                number: current_number,
                description: desc.to_string(),
                indent_level,
                is_priority: desc.to_lowercase().contains("important")
                    || desc.to_lowercase().contains("critical")
                    || desc.contains("!"),
            });
            continue;
        }

        // Try "Step N:" format
        if let Some(caps) = step_colon_re.captures(line) {
            let indent = caps.get(1).map(|m| m.as_str().len()).unwrap_or(0);
            let _num: usize = caps
                .get(2)
                .expect("group 2 always present in match")
                .as_str()
                .parse()
                .unwrap_or(0);
            let desc = caps
                .get(3)
                .expect("group 3 always present in match")
                .as_str()
                .trim();

            current_number += 1;
            let indent_level = indent / 2;

            steps.push(ParsedStep {
                number: current_number,
                description: desc.to_string(),
                indent_level,
                is_priority: desc.to_lowercase().contains("important")
                    || desc.to_lowercase().contains("critical"),
            });
            continue;
        }

        // Try bullet format (only in certain sections)
        if let Some(caps) = bullet_re.captures(line) {
            let indent = caps.get(1).map(|m| m.as_str().len()).unwrap_or(0);
            let desc = caps
                .get(2)
                .expect("group 2 always present in match")
                .as_str()
                .trim();

            // Skip bullets that look like notes/comments
            if desc.starts_with("Note:") || desc.starts_with("Warning:") {
                continue;
            }

            // Only include bullets that look like action items
            if desc.len() > 10
                && (desc.to_lowercase().contains("create")
                    || desc.to_lowercase().contains("add")
                    || desc.to_lowercase().contains("implement")
                    || desc.to_lowercase().contains("update")
                    || desc.to_lowercase().contains("modify")
                    || desc.to_lowercase().contains("configure")
                    || desc.to_lowercase().contains("set up")
                    || desc.to_lowercase().contains("install")
                    || desc.to_lowercase().contains("test")
                    || desc.to_lowercase().contains("verify")
                    || desc.to_lowercase().contains("check")
                    || desc.to_lowercase().contains("review")
                    || desc.to_lowercase().contains("fix")
                    || desc.to_lowercase().contains("remove")
                    || desc.to_lowercase().contains("delete"))
            {
                current_number += 1;
                let indent_level = indent / 2;

                steps.push(ParsedStep {
                    number: current_number,
                    description: desc.to_string(),
                    indent_level,
                    is_priority: false,
                });
            }
        }
    }

    steps
}

/// Convert parsed steps into Task objects
pub fn steps_to_tasks(steps: &[ParsedStep], plan_id: &str) -> Vec<Task> {
    let mut tasks = Vec::new();
    let mut parent_stack: Vec<String> = Vec::new();

    for step in steps {
        let task_id = format!("{}-step-{}", &plan_id[..8.min(plan_id.len())], step.number);

        let priority = if step.is_priority {
            TaskPriority::High
        } else {
            TaskPriority::Normal
        };

        let mut task = Task::new_for_plan(
            task_id.clone(),
            step.description.clone(),
            plan_id.to_string(),
        );
        task.priority = priority;

        // Handle hierarchy based on indent level
        if step.indent_level == 0 {
            // Root level task
            parent_stack.clear();
            parent_stack.push(task_id.clone());
        } else if step.indent_level <= parent_stack.len() {
            // Same or higher level - pop back
            parent_stack.truncate(step.indent_level);
            if let Some(parent_id) = parent_stack.last() {
                task.parent_id = Some(parent_id.clone());
            }
            parent_stack.push(task_id.clone());
        } else {
            // Deeper level - current parent is last in stack
            if let Some(parent_id) = parent_stack.last() {
                task.parent_id = Some(parent_id.clone());
            }
            parent_stack.push(task_id.clone());
        }

        // Add sequential dependencies (each step depends on previous)
        if !tasks.is_empty() && step.indent_level == 0 {
            // Only add dependency for root-level tasks
            let prev_task: &Task = &tasks[tasks.len() - 1];
            if prev_task.parent_id.is_none() {
                task.depends_on.push(prev_task.id.clone());
            }
        }

        tasks.push(task);
    }

    // Update parent tasks to include children
    let task_ids: Vec<_> = tasks.iter().map(|t| t.id.clone()).collect();
    for i in 0..tasks.len() {
        if let Some(ref parent_id) = tasks[i].parent_id.clone() {
            // Find parent and add this task as child
            for task in tasks.iter_mut() {
                if task.id == *parent_id {
                    task.children.push(task_ids[i].clone());
                    break;
                }
            }
        }
    }

    tasks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_numbered_steps() {
        let content = r#"
1. Create the user model
2. Add authentication endpoints
3. Implement JWT token handling
"#;
        let steps = parse_plan_steps(content);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].description, "Create the user model");
        assert_eq!(steps[1].description, "Add authentication endpoints");
    }

    #[test]
    fn test_parse_step_colon_format() {
        let content = r#"
Step 1: Initialize the project
Step 2: Configure dependencies
"#;
        let steps = parse_plan_steps(content);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].description, "Initialize the project");
    }

    #[test]
    fn test_parse_indented_steps() {
        let content = r#"
1. Setup phase
  1. Install dependencies
  2. Configure environment
2. Implementation phase
"#;
        let steps = parse_plan_steps(content);
        assert_eq!(steps.len(), 4);
        // Note: Due to renumbering, indent_level matters more than the original numbers
    }

    #[test]
    fn test_steps_to_tasks() {
        let content = "1. First step\n2. Second step";
        let steps = parse_plan_steps(content);
        let tasks = steps_to_tasks(&steps, "plan-12345678");

        assert_eq!(tasks.len(), 2);
        assert!(tasks[0].plan_id.is_some());
        assert_eq!(tasks[0].plan_id.as_ref().unwrap(), "plan-12345678");
    }

    #[test]
    fn test_priority_detection() {
        let content = "1. Important: Fix critical bug!\n2. Normal task";
        let steps = parse_plan_steps(content);

        assert!(steps[0].is_priority);
        assert!(!steps[1].is_priority);
    }
}
