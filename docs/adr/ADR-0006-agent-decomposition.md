# ADR-0006 — Agent crate decomposition

- **Status:** Accepted
- **Date:** 2026-05-03
- **Authors:** Brainwires

## Context

After Phase 10 the workspace had cleaned up most of the god-crates,
but `brainwires-agent` was still ~50k LOC across 7 subdirectories and
32 top-level files mixing five distinct domains:

| Domain | LOC | What |
|---|---|---|
| inference | ~17k | LLM-driven workhorses: chat / planner / judge / validator / cycle orchestrator / task agent / system prompts / etc. |
| mdap | ~7k | Multi-Dimensional Adaptive Planning (MAKER voting framework) |
| seal | ~7k | Self-Evolving Agentic Learning |
| skills | ~5k | SKILL.md skills system |
| eval | ~3.5k | Evaluation harness |
| coordination + patterns + schema | ~10k | Locks, queues, message bus, lifecycle, multi-agent coordination patterns |

Three of those domains (`mdap`, `seal`, `skills`) were once their own
crates and had been merged into `brainwires-agent` as feature-gated
modules. The earlier plan's "Things deliberately not in this plan"
section explicitly argued AGAINST re-extracting them, on grounds that
they shared lifecycle / state with the agent runtime and no consumer
pain was documented.

That stance no longer holds. With the rest of the workspace cleaned
up, `brainwires-agent` was the last remaining god-crate. Anyone
wanting just the coordination primitives had to compile mdap / seal
/ skills / eval / inference. Anyone wanting just inference had to
compile coordination + the rest. The "no documented consumer pain"
argument was a function of the earlier workspace shape — once every
other crate had a single cohesive responsibility, agent stuck out.

## Decision

Decompose `brainwires-agent` into six crates:

- **`brainwires-agent` (slimmed)** — coordination + patterns + schema
  only. `communication`, `task_manager`, `task_queue`, locks
  (`file_locks`, `resource_locks`, `wait_queue`, `access_control`,
  `operation_tracker`), `git_coordination`, `worktree`,
  `agent_manager`, `agent_tools`, `resource_checker`,
  `execution_graph`, `otel`, multi-agent patterns (`state_model`,
  `contract_net`, `saga`, `optimistic`, `market_allocation`,
  `workflow`), schema (`roles`, `personas`).
- **`brainwires-inference` (new)** — LLM-driven workhorses.
  `chat_agent`, `task_agent`, `runtime`, `context`, `agent_hooks`,
  `pool`, `task_orchestrator`, `cycle_orchestrator`, `plan_executor`,
  `validation_loop`, `validation_agent`, `validator_agent`,
  `planner_agent`, `judge_agent`, `summarization`, `system_prompts`.
  Depends on `brainwires-agent` for coordination types.
- **`brainwires-mdap` (resurrected)** — MAKER voting framework. Zero
  internal deps beyond core; cleanest possible split.
- **`brainwires-seal` (resurrected)** — Self-Evolving Agentic
  Learning. Depends on storage (LanceDB pattern store) +
  optionally on knowledge / permission / mdap.
- **`brainwires-skills` (resurrected)** — SKILL.md skills system.
  Depends on core + tool-runtime only.
- **`brainwires-eval` (resurrected)** — evaluation harness. Zero
  brainwires-* deps.

`ResponseConfidence` (the one cross-domain type shared between the
agent runtime and SEAL learning) moves to `brainwires-core` as a
prep step (Phase 11a) so seal extracts cleanly without depending on
agent.

## Consequences

- **Positive.**
  - Every framework crate now has a single cohesive responsibility.
    Agent is what holds agents together; inference is what makes
    them think; mdap / seal / skills / eval are domain-specific
    primitives.
  - Consumers can pull only what they use. Pulling `brainwires-mdap`
    no longer drags in 50k LOC of unrelated coordination + LLM code.
  - Per-domain crate names communicate intent. `brainwires-inference`
    tells a Cargo.toml reader "this is the LLM stuff" without
    needing to read source.
  - The agent crate's `seal` / `mdap` / `skills` / `eval` features
    are gone. Cargo features should signal optional behaviour
    *within* a crate, not "include a different crate." Splitting
    gives each domain its own publishable / versionable unit.
- **Negative.**
  - **This reverses the earlier "Things deliberately not in this
    plan" stance.** Anyone reading commit history will see the
    prior reasoning ("they share lifecycle and state with the agent
    runtime; no consumer pain documented") and the new commits
    extracting them. The two are consistent: the earlier reasoning
    was workspace-state-dependent, not architectural.
  - Five new publishable crates means more bookkeeping at release
    time. `scripts/publish.sh` handles ordering automatically; the
    cost is a longer CRATES list and slightly slower release runs.
  - The cycle that drove `runtime`, `context`, `pool`,
    `task_orchestrator`, `agent_hooks` from agent into inference
    (they all reference `TaskAgent*` types) means agent is more
    minimal than originally planned. Net win — agent is now a
    truly cohesive coordination crate — but readers expecting
    `AgentRuntime` to live in `brainwires-agent` will be surprised.
- **Neutral / follow-up.**
  - The brainwires facade (`crates/brainwires/`) keeps the
    `brainwires::agents::*` re-export spreading both crates so
    existing import paths don't break.
  - `brainwires-inference`'s native feature implies
    `brainwires-agent/native` — they ship together as the default.
  - `chat` feature on the umbrella implies `inference` (chat needs
    `ChatAgent`).

## Alternatives considered

- **Keep `brainwires-agent` as-is.** Documented in the previous
  plan's "Things deliberately not in this plan" block. Held until
  Phase 10 finished and the workspace shape made the cohesion
  problem stark. Rejected once the framing changed from "framework
  stays minimal" (Phase 5) to "every crate has one cohesive
  responsibility" (Phase 11).
- **Extract `brainwires-inference` only; keep mdap / seal / skills
  / eval inside agent.** Considered. Rejected because mdap is
  trivially separable (zero internal deps), skills is clean
  (depends only on core + tool-runtime), and the LLM-driven story
  is more coherent if the inference crate doesn't have to juggle
  feature gates for mdap / seal / skills mixed with its own.
- **Move `ResponseConfidence` to `brainwires-agent::confidence`
  and accept the back-edge.** Rejected — `ResponseConfidence` is a
  generic response-quality metric, not agent-specific. Belongs in
  core. The Phase 11a move is a one-liner.
- **Make `AgentLifecycleHooks` parametric over result types so
  `pool` / `runtime` / `context` can stay in agent.** Considered.
  Rejected — too much complexity for too little win. The hooks
  trait is intimately tied to `TaskAgent` execution; abstracting it
  would force consumers through generics for no gain.

## References

- Phase 11a–11g commits — see `git log v-0.11`.
- Previous plan's "Things deliberately not in this plan" block —
  the stance that this ADR overturns.
- ADR-0001 — crate split discipline (the meta-rule that requires
  this ADR).
- ADR-0005 — stores consolidation (the previous similar reversal of
  a Phase 5 decision).
