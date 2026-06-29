# @rullama/inference

LLM-driven agent workhorses: chat agent, task agent, planner, judge, validator,
plan executor, cycle orchestrator, validation loop, runtime, system prompts.

Extracted from `@rullama/agents` in v0.11.0 to mirror Rust's
`rullama-inference` crate. The coordination primitives (`CommunicationHub`,
`TaskManager`, `FileLockManager`, etc.) stay in `@rullama/agents` (renamed to
`@rullama/agent` in v0.11.0 — both names work during the transition).
