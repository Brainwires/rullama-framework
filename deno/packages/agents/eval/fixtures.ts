/**
 * YAML-backed golden-prompt fixtures for the evaluation framework.
 *
 * Fixtures describe a scenario — input messages plus expected behaviour —
 * as a data file so non-code contributors can add tests without touching
 * TypeScript. Each fixture is loaded as a {@link FixtureCase}, which
 * implements {@link EvaluationCase} and can be fed into an
 * {@link EvaluationSuite}.
 *
 * Equivalent to Rust's `brainwires_agents::eval::fixtures` module.
 */

import { parse as parseYaml } from "@std/yaml/parse";
import { join } from "@std/path";
import type { EvaluationCase } from "./case.ts";
import { type TrialResult, trialFailure, trialSuccess } from "./trial.ts";

/** A loaded golden-prompt fixture. */
export interface Fixture {
  /** Short identifier (must be unique within a suite). */
  name: string;
  /** Category label for grouping. Defaults to "fixture". */
  category: string;
  /** Optional model hint for the runner. Runners may ignore this. */
  model: string | null;
  /** Input conversation to replay. */
  messages: FixtureMessage[];
  /** Constraints on the runner's output. */
  expected: ExpectedBehavior;
}

/** A single message in a fixture's input conversation. */
export interface FixtureMessage {
  /** Message role — e.g. "user", "system", "assistant". */
  role: string;
  content: string;
}

/** Constraints a fixture imposes on the runner's output. */
export interface ExpectedBehavior {
  /** Exact ordered tool sequence. Empty means "any". */
  tool_sequence: string[];
  /** Individual assertions that must all hold. */
  assertions: Assertion[];
}

/** A single constraint on a fixture outcome. */
export interface Assertion {
  contains?: string;
  regex?: string;
  tool_called?: string;
  finish_reason?: string;
}

/** Result of running a fixture through a {@link FixtureRunner}. */
export interface RunOutcome {
  output_text: string;
  tool_sequence: string[];
  finish_reason: string | null;
  duration_ms: number;
}

export function defaultRunOutcome(): RunOutcome {
  return {
    output_text: "",
    tool_sequence: [],
    finish_reason: null,
    duration_ms: 0,
  };
}

/** Drives a fixture to completion. */
export interface FixtureRunner {
  /** Execute `fixture` and return the observed outcome. */
  run(fixture: Fixture, trial_id: number): Promise<RunOutcome>;
}

/** An {@link EvaluationCase} built from a fixture + runner pair. */
export class FixtureCase implements EvaluationCase {
  readonly fixture_: Fixture;
  readonly runner: FixtureRunner;

  constructor(fixture: Fixture, runner: FixtureRunner) {
    this.fixture_ = fixture;
    this.runner = runner;
  }

  fixture(): Fixture {
    return this.fixture_;
  }
  name(): string {
    return this.fixture_.name;
  }
  category(): string {
    return this.fixture_.category;
  }

  async run(trial_id: number): Promise<TrialResult> {
    const started = performance.now();
    let outcome: RunOutcome;
    try {
      outcome = await this.runner.run(this.fixture_, trial_id);
    } catch (e) {
      const elapsed = Math.round(performance.now() - started);
      const msg = e instanceof Error ? e.message : String(e);
      return trialFailure(trial_id, elapsed, `runner error: ${msg}`);
    }
    const reason = evaluate(this.fixture_.expected, outcome);
    if (reason === null) return trialSuccess(trial_id, outcome.duration_ms);
    return trialFailure(trial_id, outcome.duration_ms, reason);
  }
}

/**
 * Evaluate an outcome against the expected behaviour. Returns the first
 * failing reason, or `null` if everything matches.
 */
export function evaluate(
  expected: ExpectedBehavior,
  outcome: RunOutcome,
): string | null {
  if (expected.tool_sequence.length > 0) {
    const a = expected.tool_sequence;
    const b = outcome.tool_sequence;
    if (
      a.length !== b.length || a.some((v, i) => v !== b[i])
    ) {
      return `tool_sequence mismatch: expected ${JSON.stringify(a)}, got ${JSON.stringify(b)}`;
    }
  }
  for (const a of expected.assertions) {
    if (a.contains !== undefined && !outcome.output_text.includes(a.contains)) {
      return `output_text missing expected substring: ${JSON.stringify(a.contains)}`;
    }
    if (a.regex !== undefined) {
      let re: RegExp;
      try {
        re = new RegExp(a.regex);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        return `invalid regex in fixture: ${JSON.stringify(a.regex)} (${msg})`;
      }
      if (!re.test(outcome.output_text)) {
        return `output_text did not match regex: ${JSON.stringify(a.regex)}`;
      }
    }
    if (
      a.tool_called !== undefined &&
      !outcome.tool_sequence.includes(a.tool_called)
    ) {
      return `expected tool \`${a.tool_called}\` to be called; got ${JSON.stringify(outcome.tool_sequence)}`;
    }
    if (a.finish_reason !== undefined) {
      const got = outcome.finish_reason ?? "";
      if (got !== a.finish_reason) {
        return `finish_reason mismatch: expected ${JSON.stringify(a.finish_reason)}, got ${JSON.stringify(got)}`;
      }
    }
  }
  return null;
}

/** Normalize a raw parsed YAML object into a {@link Fixture}. */
function normalizeFixture(raw: unknown): Fixture {
  // deno-lint-ignore no-explicit-any
  const o = raw as any;
  if (!o || typeof o !== "object") {
    throw new Error("fixture is not a YAML object");
  }
  const expectedRaw = o.expected ?? {};
  const expected: ExpectedBehavior = {
    tool_sequence: Array.isArray(expectedRaw.tool_sequence)
      ? expectedRaw.tool_sequence.map(String)
      : [],
    assertions: Array.isArray(expectedRaw.assertions)
      ? expectedRaw.assertions.map((a: unknown) => a as Assertion)
      : [],
  };
  return {
    name: String(o.name ?? ""),
    category: typeof o.category === "string" ? o.category : "fixture",
    model: typeof o.model === "string" ? o.model : null,
    messages: Array.isArray(o.messages)
      ? o.messages.map((m: unknown) => m as FixtureMessage)
      : [],
    expected,
  };
}

/** Load a single fixture YAML file. */
export async function loadFixtureFile(path: string): Promise<Fixture> {
  const raw = await Deno.readTextFile(path);
  return normalizeFixture(parseYaml(raw));
}

/**
 * Load every `.yaml` / `.yml` fixture file directly inside `dir` (non-recursive).
 * Returns them in deterministic filename order.
 */
export async function loadFixturesFromDir(dir: string): Promise<Fixture[]> {
  const paths: string[] = [];
  for await (const entry of Deno.readDir(dir)) {
    if (!entry.isFile) continue;
    const n = entry.name;
    if (n.endsWith(".yaml") || n.endsWith(".yml")) {
      paths.push(join(dir, n));
    }
  }
  paths.sort();
  const out: Fixture[] = [];
  for (const p of paths) out.push(await loadFixtureFile(p));
  return out;
}
