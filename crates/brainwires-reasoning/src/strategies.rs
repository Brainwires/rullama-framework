//! Named Reasoning Strategies — ReAct, Reflexion, Chain of Thought, Tree of Thoughts
//!
//! Provides [`ReasoningStrategy`] trait and concrete implementations for
//! well-known LLM reasoning patterns. Each strategy wraps the system prompt
//! and controls the reasoning loop structure.
//!
//! # Strategies
//!
//! - [`ReActStrategy`] — Thought → Action → Observation loop (Yao et al., 2022)
//! - [`ReflexionStrategy`] — Self-critique after each action (Shinn et al., 2023)
//! - [`ChainOfThoughtStrategy`] — "Let's think step by step" (Wei et al., 2022)
//! - [`TreeOfThoughtsStrategy`] — Multi-branch exploration with pruning (Yao et al., 2023)
//!
//! # Usage
//!
//! ```rust,ignore
//! use brainwires_agent::reasoning::strategies::{ReActStrategy, ReasoningStrategy};
//!
//! let strategy = ReActStrategy::new(10); // max 10 reasoning steps
//! let system_prompt = strategy.system_prompt("agent-1", "/project");
//! let is_done = strategy.is_complete(&steps);
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

// ── Strategy Step ────────────────────────────────────────────────────────────

/// A single step in a reasoning trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyStep {
    /// Internal reasoning / thought.
    Thought(String),
    /// An action to execute (tool call).
    Action {
        /// Tool name.
        tool: String,
        /// Tool arguments.
        args: serde_json::Value,
    },
    /// Result of an action / tool call.
    Observation(String),
    /// Self-critique and revised plan (Reflexion).
    Reflection {
        /// What went wrong or could be improved.
        critique: String,
        /// Revised plan based on the critique.
        revised_plan: String,
    },
    /// A candidate branch in Tree of Thoughts.
    Branch {
        /// Branch identifier.
        branch_id: usize,
        /// The candidate thought.
        thought: String,
        /// Score assigned to this branch (0.0–1.0).
        score: f64,
    },
    /// The final answer / completion signal.
    FinalAnswer(String),
}

impl fmt::Display for StrategyStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StrategyStep::Thought(t) => write!(f, "Thought: {}", t),
            StrategyStep::Action { tool, .. } => write!(f, "Action: {}", tool),
            StrategyStep::Observation(o) => write!(f, "Observation: {}", o),
            StrategyStep::Reflection { critique, .. } => write!(f, "Reflection: {}", critique),
            StrategyStep::Branch {
                branch_id,
                thought,
                score,
            } => {
                write!(f, "Branch[{}] (score={:.2}): {}", branch_id, score, thought)
            }
            StrategyStep::FinalAnswer(a) => write!(f, "Final: {}", a),
        }
    }
}

// ── ReasoningStrategy trait ──────────────────────────────────────────────────

/// A named reasoning strategy that controls the agent's thinking structure.
///
/// Strategies generate specialized system prompts and determine when the
/// reasoning loop should terminate.
pub trait ReasoningStrategy: Send + Sync {
    /// Human-readable strategy name (e.g., "ReAct", "Reflexion").
    fn name(&self) -> &str;

    /// Brief description of how this strategy works.
    fn description(&self) -> &str;

    /// Generate a system prompt that instructs the LLM to follow this
    /// reasoning pattern.
    ///
    /// The returned prompt is used as the system message (or prepended to
    /// the existing system prompt).
    fn system_prompt(&self, agent_id: &str, working_directory: &str) -> String;

    /// Check whether the reasoning process has converged.
    ///
    /// Returns `true` when the strategy considers the reasoning complete
    /// (e.g., a `FinalAnswer` step has been emitted, or max steps reached).
    fn is_complete(&self, steps: &[StrategyStep]) -> bool;

    /// Maximum reasoning steps before forced termination.
    fn max_steps(&self) -> usize;
}

// ── ReAct Strategy ───────────────────────────────────────────────────────────

/// ReAct: Reasoning + Acting (Yao et al., 2022)
///
/// The agent alternates between Thought, Action, and Observation steps.
/// Each cycle: think about what to do, take an action, observe the result.
#[derive(Debug, Clone)]
pub struct ReActStrategy {
    max_steps: usize,
}

impl ReActStrategy {
    /// Create a new ReAct strategy with the given maximum step count.
    pub fn new(max_steps: usize) -> Self {
        Self { max_steps }
    }
}

impl Default for ReActStrategy {
    fn default() -> Self {
        Self { max_steps: 15 }
    }
}

impl ReasoningStrategy for ReActStrategy {
    fn name(&self) -> &str {
        "ReAct"
    }

    fn description(&self) -> &str {
        "Reasoning + Acting: alternate between Thought, Action, and Observation steps"
    }

    fn system_prompt(&self, agent_id: &str, working_directory: &str) -> String {
        format!(
            r#"You are a task agent (ID: {agent_id}) using the ReAct reasoning framework.

Working Directory: {working_directory}

# ReAct FRAMEWORK

You MUST follow this strict loop for every step:

1. **Thought**: Reason about what you need to do next. What information do
   you have? What do you still need? What is the best next action?

2. **Action**: Execute exactly ONE tool call based on your thought.

3. **Observation**: Analyze the result. Did it succeed? What did you learn?
   Does the result change your plan?

Then repeat: Thought → Action → Observation → Thought → ...

## Format

Always structure your response as:

Thought: <your reasoning about what to do next>
Action: <use a tool>
Observation: <analyze the result after receiving it>

## Rules

- Each cycle must have exactly ONE action (tool call)
- Never skip the Thought step — always reason before acting
- If a tool call fails, reflect in your next Thought and adjust
- When the task is complete, state your final answer clearly
- Maximum {max_steps} reasoning cycles allowed

## Completion

When the task is fully complete, provide your final summary.
Do NOT continue cycling after the task is done."#,
            agent_id = agent_id,
            working_directory = working_directory,
            max_steps = self.max_steps,
        )
    }

    fn is_complete(&self, steps: &[StrategyStep]) -> bool {
        if steps.len() >= self.max_steps {
            return true;
        }
        steps
            .iter()
            .any(|s| matches!(s, StrategyStep::FinalAnswer(_)))
    }

    fn max_steps(&self) -> usize {
        self.max_steps
    }
}

// ── Reflexion Strategy ───────────────────────────────────────────────────────

/// Reflexion: Self-critique and iterative refinement (Shinn et al., 2023)
///
/// After each action-observation cycle, the agent reflects on what could
/// be improved and adjusts its plan accordingly.
#[derive(Debug, Clone)]
pub struct ReflexionStrategy {
    max_steps: usize,
    /// How many reflection cycles before forcing completion.
    max_reflections: usize,
}

impl ReflexionStrategy {
    /// Create a new Reflexion strategy.
    pub fn new(max_steps: usize, max_reflections: usize) -> Self {
        Self {
            max_steps,
            max_reflections,
        }
    }
}

impl Default for ReflexionStrategy {
    fn default() -> Self {
        Self {
            max_steps: 20,
            max_reflections: 5,
        }
    }
}

impl ReasoningStrategy for ReflexionStrategy {
    fn name(&self) -> &str {
        "Reflexion"
    }

    fn description(&self) -> &str {
        "Self-critique after each action cycle with iterative plan refinement"
    }

    fn system_prompt(&self, agent_id: &str, working_directory: &str) -> String {
        format!(
            r#"You are a task agent (ID: {agent_id}) using the Reflexion reasoning framework.

Working Directory: {working_directory}

# REFLEXION FRAMEWORK

You follow a 4-phase loop:

1. **Plan**: State your current plan and what you intend to do
2. **Act**: Execute your planned action using a tool
3. **Observe**: Analyze the result
4. **Reflect**: Critically evaluate your approach:
   - What went well?
   - What could be improved?
   - Should I change my plan?
   - Am I making progress toward the goal?

Then update your plan and repeat.

## Format

Structure each cycle as:

Plan: <current plan and next intended action>
Act: <use a tool>
Observe: <what happened>
Reflect: <self-critique — what worked, what didn't, revised plan>

## Rules

- Every action MUST be followed by reflection
- Be honest in your self-critique — identify mistakes early
- Update your plan after each reflection if needed
- If you've made the same mistake twice, try a completely different approach
- Maximum {max_reflections} reflection cycles before you must finalize
- Maximum {max_steps} total steps allowed

## Completion

After your final reflection confirms the task is complete, provide
your summary."#,
            agent_id = agent_id,
            working_directory = working_directory,
            max_reflections = self.max_reflections,
            max_steps = self.max_steps,
        )
    }

    fn is_complete(&self, steps: &[StrategyStep]) -> bool {
        if steps.len() >= self.max_steps {
            return true;
        }
        let reflection_count = steps
            .iter()
            .filter(|s| matches!(s, StrategyStep::Reflection { .. }))
            .count();
        if reflection_count >= self.max_reflections {
            return true;
        }
        steps
            .iter()
            .any(|s| matches!(s, StrategyStep::FinalAnswer(_)))
    }

    fn max_steps(&self) -> usize {
        self.max_steps
    }
}

// ── Chain of Thought Strategy ────────────────────────────────────────────────

/// Chain of Thought: Step-by-step reasoning (Wei et al., 2022)
///
/// The agent reasons through the problem step by step before taking
/// action. Best for problems requiring logical deduction.
#[derive(Debug, Clone)]
pub struct ChainOfThoughtStrategy {
    max_steps: usize,
}

impl ChainOfThoughtStrategy {
    /// Create a new Chain of Thought strategy.
    pub fn new(max_steps: usize) -> Self {
        Self { max_steps }
    }
}

impl Default for ChainOfThoughtStrategy {
    fn default() -> Self {
        Self { max_steps: 15 }
    }
}

impl ReasoningStrategy for ChainOfThoughtStrategy {
    fn name(&self) -> &str {
        "Chain-of-Thought"
    }

    fn description(&self) -> &str {
        "Step-by-step reasoning before taking action"
    }

    fn system_prompt(&self, agent_id: &str, working_directory: &str) -> String {
        format!(
            r#"You are a task agent (ID: {agent_id}) using Chain-of-Thought reasoning.

Working Directory: {working_directory}

# CHAIN OF THOUGHT FRAMEWORK

Before taking ANY action, you MUST reason through the problem step by step.

## Process

1. **Decompose**: Break the task into numbered steps
2. **Reason**: For each step, explain your logic:
   - What do I know?
   - What do I need to find out?
   - What is the logical next step?
3. **Act**: After reasoning, execute the steps using tools
4. **Verify**: Confirm each step's result before proceeding

## Format

Let's think step by step:

Step 1: <describe what you need to do and why>
Step 2: <next logical step>
Step 3: ...

Then execute each step, verifying results between actions.

## Rules

- Always number your reasoning steps
- Show your work — explain WHY, not just WHAT
- If a step's result is unexpected, re-reason from that point
- Do not skip steps — complete each one before moving on
- Maximum {max_steps} steps allowed

## Completion

After all steps are verified, provide your final answer."#,
            agent_id = agent_id,
            working_directory = working_directory,
            max_steps = self.max_steps,
        )
    }

    fn is_complete(&self, steps: &[StrategyStep]) -> bool {
        if steps.len() >= self.max_steps {
            return true;
        }
        steps
            .iter()
            .any(|s| matches!(s, StrategyStep::FinalAnswer(_)))
    }

    fn max_steps(&self) -> usize {
        self.max_steps
    }
}

// ── Tree of Thoughts Strategy ────────────────────────────────────────────────

/// Tree of Thoughts: Multi-branch exploration with pruning (Yao et al., 2023)
///
/// The agent generates multiple candidate approaches, scores them, and
/// pursues the most promising branch. Useful for complex problems with
/// multiple valid solution paths.
#[derive(Debug, Clone)]
pub struct TreeOfThoughtsStrategy {
    max_steps: usize,
    /// Number of candidate branches to generate at each decision point.
    branching_factor: usize,
    /// Minimum score (0.0–1.0) for a branch to be pursued.
    pruning_threshold: f64,
}

impl TreeOfThoughtsStrategy {
    /// Create a new Tree of Thoughts strategy.
    pub fn new(max_steps: usize, branching_factor: usize, pruning_threshold: f64) -> Self {
        Self {
            max_steps,
            branching_factor,
            pruning_threshold,
        }
    }
}

impl Default for TreeOfThoughtsStrategy {
    fn default() -> Self {
        Self {
            max_steps: 25,
            branching_factor: 3,
            pruning_threshold: 0.4,
        }
    }
}

impl ReasoningStrategy for TreeOfThoughtsStrategy {
    fn name(&self) -> &str {
        "Tree-of-Thoughts"
    }

    fn description(&self) -> &str {
        "Multi-branch exploration with scoring and pruning"
    }

    fn system_prompt(&self, agent_id: &str, working_directory: &str) -> String {
        format!(
            r#"You are a task agent (ID: {agent_id}) using Tree-of-Thoughts reasoning.

Working Directory: {working_directory}

# TREE OF THOUGHTS FRAMEWORK

At each decision point, generate {branching_factor} candidate approaches,
evaluate them, and pursue the best one.

## Process

1. **Generate**: Propose {branching_factor} different approaches to the
   current sub-problem
2. **Evaluate**: Score each approach (0.0 to 1.0) based on:
   - Likelihood of success
   - Efficiency (fewer steps = higher score)
   - Risk of side effects
   - Alignment with the overall goal
3. **Select**: Choose the approach with the highest score
   (must be above {pruning_threshold:.1} to proceed)
4. **Execute**: Implement the selected approach
5. **Backtrack**: If the selected approach fails, try the next-best candidate

## Format

=== Decision Point ===
Candidate 1: <approach description> → Score: X.X
Candidate 2: <approach description> → Score: X.X
Candidate 3: <approach description> → Score: X.X

Selected: Candidate N (score: X.X)
Reason: <why this is the best approach>

Then execute the selected approach.

## Rules

- Generate exactly {branching_factor} candidates at each major decision
- Score honestly — don't inflate scores to justify a preference
- If all candidates score below {pruning_threshold:.1}, step back and reconsider
  the problem framing
- Keep a mental note of unexplored branches in case backtracking is needed
- Maximum {max_steps} total steps allowed

## Completion

When the task is complete, summarize which branches were explored and why
the final approach was chosen."#,
            agent_id = agent_id,
            working_directory = working_directory,
            branching_factor = self.branching_factor,
            pruning_threshold = self.pruning_threshold,
            max_steps = self.max_steps,
        )
    }

    fn is_complete(&self, steps: &[StrategyStep]) -> bool {
        if steps.len() >= self.max_steps {
            return true;
        }
        steps
            .iter()
            .any(|s| matches!(s, StrategyStep::FinalAnswer(_)))
    }

    fn max_steps(&self) -> usize {
        self.max_steps
    }
}

// ── Strategy registry ────────────────────────────────────────────────────────

/// Well-known strategy presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyPreset {
    /// ReAct: Thought → Action → Observation loop
    ReAct,
    /// Reflexion: Self-critique after each cycle
    Reflexion,
    /// Chain of Thought: Step-by-step reasoning
    ChainOfThought,
    /// Tree of Thoughts: Multi-branch exploration
    TreeOfThoughts,
}

impl StrategyPreset {
    /// Create a default instance of the strategy.
    pub fn create(&self) -> Box<dyn ReasoningStrategy> {
        match self {
            StrategyPreset::ReAct => Box::new(ReActStrategy::default()),
            StrategyPreset::Reflexion => Box::new(ReflexionStrategy::default()),
            StrategyPreset::ChainOfThought => Box::new(ChainOfThoughtStrategy::default()),
            StrategyPreset::TreeOfThoughts => Box::new(TreeOfThoughtsStrategy::default()),
        }
    }

    /// All available presets.
    pub fn all() -> &'static [StrategyPreset] {
        &[
            StrategyPreset::ReAct,
            StrategyPreset::Reflexion,
            StrategyPreset::ChainOfThought,
            StrategyPreset::TreeOfThoughts,
        ]
    }
}

impl fmt::Display for StrategyPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StrategyPreset::ReAct => write!(f, "ReAct"),
            StrategyPreset::Reflexion => write!(f, "Reflexion"),
            StrategyPreset::ChainOfThought => write!(f, "Chain-of-Thought"),
            StrategyPreset::TreeOfThoughts => write!(f, "Tree-of-Thoughts"),
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_react_system_prompt() {
        let strategy = ReActStrategy::new(10);
        let prompt = strategy.system_prompt("agent-1", "/project");
        assert!(prompt.contains("ReAct"));
        assert!(prompt.contains("Thought"));
        assert!(prompt.contains("Action"));
        assert!(prompt.contains("Observation"));
        assert!(prompt.contains("agent-1"));
    }

    #[test]
    fn test_react_completion() {
        let strategy = ReActStrategy::new(3);
        let steps = vec![
            StrategyStep::Thought("thinking".to_string()),
            StrategyStep::Action {
                tool: "read".to_string(),
                args: serde_json::json!({}),
            },
        ];
        assert!(!strategy.is_complete(&steps));

        let steps_with_answer = vec![
            StrategyStep::Thought("thinking".to_string()),
            StrategyStep::FinalAnswer("done".to_string()),
        ];
        assert!(strategy.is_complete(&steps_with_answer));
    }

    #[test]
    fn test_react_max_steps() {
        let strategy = ReActStrategy::new(2);
        let steps = vec![
            StrategyStep::Thought("a".to_string()),
            StrategyStep::Thought("b".to_string()),
        ];
        assert!(strategy.is_complete(&steps));
    }

    #[test]
    fn test_reflexion_system_prompt() {
        let strategy = ReflexionStrategy::new(15, 3);
        let prompt = strategy.system_prompt("agent-2", "/work");
        assert!(prompt.contains("Reflexion"));
        assert!(prompt.contains("Reflect"));
        assert!(prompt.contains("self-critique"));
    }

    #[test]
    fn test_reflexion_max_reflections() {
        let strategy = ReflexionStrategy::new(20, 2);
        let steps = vec![
            StrategyStep::Reflection {
                critique: "a".into(),
                revised_plan: "b".into(),
            },
            StrategyStep::Reflection {
                critique: "c".into(),
                revised_plan: "d".into(),
            },
        ];
        assert!(strategy.is_complete(&steps));
    }

    #[test]
    fn test_cot_system_prompt() {
        let strategy = ChainOfThoughtStrategy::new(10);
        let prompt = strategy.system_prompt("agent-3", "/code");
        assert!(prompt.contains("Chain-of-Thought") || prompt.contains("Chain of Thought"));
        assert!(prompt.contains("step by step"));
    }

    #[test]
    fn test_tot_system_prompt() {
        let strategy = TreeOfThoughtsStrategy::new(20, 3, 0.4);
        let prompt = strategy.system_prompt("agent-4", "/proj");
        assert!(prompt.contains("Tree-of-Thoughts") || prompt.contains("Tree of Thoughts"));
        assert!(prompt.contains("3")); // branching factor
    }

    #[test]
    fn test_strategy_preset_create() {
        for preset in StrategyPreset::all() {
            let strategy = preset.create();
            assert!(!strategy.name().is_empty());
            assert!(!strategy.description().is_empty());
            assert!(strategy.max_steps() > 0);
        }
    }

    #[test]
    fn test_strategy_step_display() {
        let step = StrategyStep::Thought("testing".to_string());
        assert_eq!(format!("{}", step), "Thought: testing");

        let step = StrategyStep::Branch {
            branch_id: 1,
            thought: "try X".to_string(),
            score: 0.85,
        };
        assert!(format!("{}", step).contains("0.85"));
    }
}
