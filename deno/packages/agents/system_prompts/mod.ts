/**
 * System Prompt Registry for brainwires agents.
 *
 * Single authoritative source for every agent system prompt. To add a new
 * agent type: add a variant to {@link AgentPromptKind}, implement the prompt
 * function in `agents.ts`, and wire it into {@link buildAgentPrompt}.
 *
 * When a role is supplied, its {@link systemPromptSuffix} is automatically
 * appended — keeping role-aware prompt construction in one place.
 *
 * Equivalent to Rust's `brainwires_agents::system_prompts` module.
 */

import type { AgentRole } from "../roles.ts";
import { systemPromptSuffix } from "../roles.ts";
import {
  judgeAgentPrompt,
  mdapMicroagentPrompt,
  plannerAgentPrompt,
  reasoningAgentPrompt,
  simpleAgentPrompt,
} from "./agents.ts";

export {
  judgeAgentPrompt,
  mdapMicroagentPrompt,
  plannerAgentPrompt,
  reasoningAgentPrompt,
  simpleAgentPrompt,
};

/** All agent system prompt contexts defined in the framework. */
export type AgentPromptKind =
  | {
    kind: "reasoning";
    agent_id: string;
    working_directory: string;
  }
  | {
    kind: "planner";
    agent_id: string;
    working_directory: string;
    goal: string;
    hints: readonly string[];
  }
  | {
    kind: "judge";
    agent_id: string;
    working_directory: string;
  }
  | {
    kind: "simple";
    agent_id: string;
    working_directory: string;
  }
  | {
    kind: "mdap_microagent";
    agent_id: string;
    working_directory: string;
    vote_round: number;
    peer_count: number;
  };

/**
 * Build a system prompt from a kind descriptor, optionally appending an
 * {@link AgentRole} constraint suffix.
 */
export function buildAgentPrompt(
  kind: AgentPromptKind,
  role?: AgentRole,
): string {
  let prompt: string;
  switch (kind.kind) {
    case "reasoning":
      prompt = reasoningAgentPrompt(kind.agent_id, kind.working_directory);
      break;
    case "planner":
      prompt = plannerAgentPrompt(
        kind.agent_id,
        kind.working_directory,
        kind.goal,
        kind.hints,
      );
      break;
    case "judge":
      prompt = judgeAgentPrompt(kind.agent_id, kind.working_directory);
      break;
    case "simple":
      prompt = simpleAgentPrompt(kind.agent_id, kind.working_directory);
      break;
    case "mdap_microagent":
      prompt = mdapMicroagentPrompt(
        kind.agent_id,
        kind.working_directory,
        kind.vote_round,
        kind.peer_count,
      );
      break;
  }

  if (role !== undefined) {
    prompt += systemPromptSuffix(role);
  }
  return prompt;
}
