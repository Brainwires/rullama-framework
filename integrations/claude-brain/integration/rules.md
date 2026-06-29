# Claude Brain — Context Management

You have a persistent brain powered by Brainwires. Compaction is enabled but Brainwires-powered.
When compaction fires, PreCompact saves everything to persistent memory, then PostCompact
restores the important context (facts, decisions, summaries) so nothing critical is lost.

## When to use recall_context
- When you sense information was discussed earlier but isn't in current context
- Before making architectural decisions (check for prior decisions on same topic)
- When the user references something from a previous conversation or session
- When context feels incomplete after a long session

## When to use capture_thought
- After making significant architectural or design decisions
- When the user shares preferences, constraints, or requirements
- After resolving non-trivial bugs (capture root cause + fix approach)
- When discovering important patterns or conventions in the codebase

## When to use search_knowledge
- Before suggesting tools, patterns, or approaches (check PKS for user preferences)
- When starting work in a new area of the codebase
- When the user asks about previous decisions or rationale

## When to use search_memory
- For broad semantic search across all memory tiers
- When looking for related context across multiple conversations
- To find previously captured thoughts on a topic

## When to use memory_stats
- When the user asks about what you remember
- To verify the brain is functioning and capturing data
