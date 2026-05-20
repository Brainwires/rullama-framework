/**
 * Plan Parser — extract numbered / bulleted steps from plan text and convert
 * them into tasks.
 *
 * Equivalent to Rust's `brainwires_reasoning::plan_parser` module.
 */

import { Task, type TaskPriority } from "@brainwires/core";

/** A single parsed step from plan content. */
export interface ParsedStep {
  /** 1-based serial number assigned in parse order. */
  number: number;
  description: string;
  /** 0 = root, 1 = substep, … (derived from leading whitespace / 2). */
  indent_level: number;
  is_priority: boolean;
}

const NUMBERED_RE = /^(\s*)(\d+)[.)]\s*(.+)$/;
const STEP_COLON_RE = /^(\s*)(?:Step\s+)?(\d+):\s*(.+)$/;
const BULLET_RE = /^(\s*)[-*]\s+(.+)$/;

const ACTION_KEYWORDS = [
  "create",
  "add",
  "implement",
  "update",
  "modify",
  "configure",
  "set up",
  "install",
  "test",
  "verify",
  "check",
  "review",
  "fix",
  "remove",
  "delete",
];

/** Parse plan content into structured steps. */
export function parsePlanSteps(content: string): ParsedStep[] {
  const steps: ParsedStep[] = [];
  let current_number = 0;

  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;

    // Numbered: "1. Description" or "1) Description"
    const numbered = NUMBERED_RE.exec(line);
    if (numbered) {
      const indent = numbered[1].length;
      const desc = numbered[3].trim();
      current_number += 1;
      steps.push({
        number: current_number,
        description: desc,
        indent_level: Math.floor(indent / 2),
        is_priority: hasPriorityMarker(desc, /* includeBang */ true),
      });
      continue;
    }

    // "Step N: Description"
    const colon = STEP_COLON_RE.exec(line);
    if (colon) {
      const indent = colon[1].length;
      const desc = colon[3].trim();
      current_number += 1;
      steps.push({
        number: current_number,
        description: desc,
        indent_level: Math.floor(indent / 2),
        is_priority: hasPriorityMarker(desc, /* includeBang */ false),
      });
      continue;
    }

    // Bulleted action items
    const bullet = BULLET_RE.exec(line);
    if (bullet) {
      const indent = bullet[1].length;
      const desc = bullet[2].trim();
      if (desc.startsWith("Note:") || desc.startsWith("Warning:")) continue;
      if (desc.length > 10 && ACTION_KEYWORDS.some((k) => desc.toLowerCase().includes(k))) {
        current_number += 1;
        steps.push({
          number: current_number,
          description: desc,
          indent_level: Math.floor(indent / 2),
          is_priority: false,
        });
      }
    }
  }

  return steps;
}

function hasPriorityMarker(desc: string, includeBang: boolean): boolean {
  const lower = desc.toLowerCase();
  if (lower.includes("important") || lower.includes("critical")) return true;
  return includeBang && desc.includes("!");
}

/** Convert parsed steps into Task objects with hierarchy + sequential deps. */
export function stepsToTasks(steps: ParsedStep[], plan_id: string): Task[] {
  const tasks: Task[] = [];
  const parent_stack: string[] = [];

  for (const step of steps) {
    const prefix = plan_id.slice(0, Math.min(8, plan_id.length));
    const task_id = `${prefix}-step-${step.number}`;
    const priority: TaskPriority = step.is_priority ? "high" : "normal";

    const task = Task.newForPlan(task_id, step.description, plan_id);
    task.priority = priority;

    if (step.indent_level === 0) {
      parent_stack.length = 0;
      parent_stack.push(task_id);
    } else if (step.indent_level <= parent_stack.length) {
      parent_stack.length = step.indent_level;
      const parent = parent_stack.at(-1);
      if (parent) task.parent_id = parent;
      parent_stack.push(task_id);
    } else {
      const parent = parent_stack.at(-1);
      if (parent) task.parent_id = parent;
      parent_stack.push(task_id);
    }

    // Sequential dependency for adjacent root-level tasks.
    if (tasks.length > 0 && step.indent_level === 0) {
      const prev = tasks[tasks.length - 1];
      if (prev.parent_id === undefined) {
        task.depends_on.push(prev.id);
      }
    }

    tasks.push(task);
  }

  // Populate children arrays on parents.
  const byId = new Map(tasks.map((t) => [t.id, t]));
  for (const t of tasks) {
    if (t.parent_id !== undefined) {
      const parent = byId.get(t.parent_id);
      if (parent && !parent.children.includes(t.id)) parent.children.push(t.id);
    }
  }

  return tasks;
}
