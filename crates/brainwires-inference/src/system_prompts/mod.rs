//! System Prompt Registry for brainwires agents.
//!
//! This module is the **single authoritative source** for every agent system prompt
//! in the framework. To add a new agent type:
//! 1. Add a variant to [`AgentPromptKind`].
//! 2. Implement the prompt function in [`agents`].
//! 3. Wire it into [`build_agent_prompt`].
//!
//! [`AgentRole`] integration: when a role is supplied, its
//! [`AgentRole::system_prompt_suffix`] is automatically appended by
//! [`build_agent_prompt`]. This keeps role-aware prompt construction in one place.

pub mod agents;

pub use agents::{
    judge_agent_prompt, mdap_microagent_prompt, planner_agent_prompt, reasoning_agent_prompt,
    simple_agent_prompt,
};

use brainwires_agent::roles::AgentRole;

/// All agent system prompt contexts defined in the framework.
///
/// If you are adding a new agent type, add a variant here first — this enum
/// is the canonical inventory of every prompt kind the framework knows about.
pub enum AgentPromptKind<'a> {
    /// Full reasoning agent: DECIDE → PRE-EVALUATE → EXECUTE → POST-EVALUATE cycle.
    /// Default for autonomous `TaskAgent` execution.
    Reasoning {
        /// Unique identifier for the agent, embedded in the prompt for tracing.
        agent_id: &'a str,
        /// Absolute path of the working directory the agent operates in.
        working_directory: &'a str,
    },

    /// Read-only planner that produces a structured JSON task plan.
    Planner {
        /// Unique identifier for the planner agent.
        agent_id: &'a str,
        /// Absolute path of the working directory.
        working_directory: &'a str,
        /// High-level goal the planner should decompose into tasks.
        goal: &'a str,
        /// Optional hints carried forward from a previous planning cycle.
        hints: &'a [String],
    },

    /// Judge that evaluates Plan→Work cycle results and decides next steps.
    Judge {
        /// Unique identifier for the judge agent.
        agent_id: &'a str,
        /// Absolute path of the working directory.
        working_directory: &'a str,
    },

    /// Minimal fallback for simple tasks that don't need the full framework.
    Simple {
        /// Unique identifier for the agent.
        agent_id: &'a str,
        /// Absolute path of the working directory.
        working_directory: &'a str,
    },

    /// MDAP voting microagent — one of k independent agents in a voting round.
    ///
    /// Instructs independent reasoning to avoid anchoring on peer results.
    MdapMicroagent {
        /// Unique identifier for this microagent instance.
        agent_id: &'a str,
        /// Absolute path of the working directory.
        working_directory: &'a str,
        /// Which vote round this agent is in (1-indexed, for logging context).
        vote_round: usize,
        /// Total number of peer agents in this round.
        peer_count: usize,
    },
}

/// Build a system prompt from a kind descriptor, optionally appending an
/// [`AgentRole`] constraint suffix.
///
/// The role suffix (if any) is appended here — callers do not need to handle
/// it separately. This is the correct integration point for
/// [`AgentRole::system_prompt_suffix`].
///
/// # Example
///
/// ```rust
/// use brainwires_inference::system_prompts::{AgentPromptKind, build_agent_prompt};
/// use brainwires_agent::roles::AgentRole;
///
/// let prompt = build_agent_prompt(
///     AgentPromptKind::Reasoning {
///         agent_id: "agent-1",
///         working_directory: "/project",
///     },
///     Some(AgentRole::Exploration),
/// );
/// assert!(prompt.contains("[ROLE: Exploration]"));
/// ```
pub fn build_agent_prompt(kind: AgentPromptKind<'_>, role: Option<AgentRole>) -> String {
    let mut prompt = match kind {
        AgentPromptKind::Reasoning {
            agent_id,
            working_directory,
        } => reasoning_agent_prompt(agent_id, working_directory),

        AgentPromptKind::Planner {
            agent_id,
            working_directory,
            goal,
            hints,
        } => planner_agent_prompt(agent_id, working_directory, goal, hints),

        AgentPromptKind::Judge {
            agent_id,
            working_directory,
        } => judge_agent_prompt(agent_id, working_directory),

        AgentPromptKind::Simple {
            agent_id,
            working_directory,
        } => simple_agent_prompt(agent_id, working_directory),

        AgentPromptKind::MdapMicroagent {
            agent_id,
            working_directory,
            vote_round,
            peer_count,
        } => mdap_microagent_prompt(agent_id, working_directory, vote_round, peer_count),
    };

    if let Some(r) = role {
        prompt.push_str(r.system_prompt_suffix());
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_variants_build_without_panic() {
        let _ = build_agent_prompt(
            AgentPromptKind::Reasoning {
                agent_id: "a",
                working_directory: "/tmp",
            },
            None,
        );
        let _ = build_agent_prompt(
            AgentPromptKind::Planner {
                agent_id: "a",
                working_directory: "/tmp",
                goal: "do something",
                hints: &[],
            },
            None,
        );
        let _ = build_agent_prompt(
            AgentPromptKind::Judge {
                agent_id: "a",
                working_directory: "/tmp",
            },
            None,
        );
        let _ = build_agent_prompt(
            AgentPromptKind::Simple {
                agent_id: "a",
                working_directory: "/tmp",
            },
            None,
        );
        let _ = build_agent_prompt(
            AgentPromptKind::MdapMicroagent {
                agent_id: "a",
                working_directory: "/tmp",
                vote_round: 1,
                peer_count: 3,
            },
            None,
        );
    }

    #[test]
    fn no_role_does_not_append_suffix() {
        let prompt = build_agent_prompt(
            AgentPromptKind::Reasoning {
                agent_id: "a",
                working_directory: "/tmp",
            },
            None,
        );
        assert!(!prompt.contains("[ROLE:"));
    }

    #[test]
    fn role_suffix_is_appended() {
        let prompt = build_agent_prompt(
            AgentPromptKind::Reasoning {
                agent_id: "a",
                working_directory: "/tmp",
            },
            Some(AgentRole::Exploration),
        );
        assert!(prompt.contains("[ROLE: Exploration]"));
    }

    #[test]
    fn planner_embeds_goal() {
        let prompt = build_agent_prompt(
            AgentPromptKind::Planner {
                agent_id: "a",
                working_directory: "/tmp",
                goal: "implement LRU cache",
                hints: &[],
            },
            None,
        );
        assert!(prompt.contains("implement LRU cache"));
    }

    #[test]
    fn mdap_embeds_vote_round_and_peer_count() {
        let prompt = build_agent_prompt(
            AgentPromptKind::MdapMicroagent {
                agent_id: "a",
                working_directory: "/tmp",
                vote_round: 2,
                peer_count: 5,
            },
            None,
        );
        assert!(
            prompt.contains('2') || prompt.contains("round"),
            "vote_round should appear in prompt"
        );
        assert!(
            prompt.contains('5') || prompt.contains("peer"),
            "peer_count should appear in prompt"
        );
    }

    #[test]
    fn simple_is_shorter_than_reasoning() {
        let simple = build_agent_prompt(
            AgentPromptKind::Simple {
                agent_id: "a",
                working_directory: "/tmp",
            },
            None,
        );
        let reasoning = build_agent_prompt(
            AgentPromptKind::Reasoning {
                agent_id: "a",
                working_directory: "/tmp",
            },
            None,
        );
        assert!(
            simple.len() < reasoning.len(),
            "Simple prompt ({} chars) should be shorter than Reasoning ({} chars)",
            simple.len(),
            reasoning.len()
        );
    }
}
