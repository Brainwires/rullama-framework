# brainwires-web-search-agent

End-to-end example: a single-binary CLI that asks an Ollama-backed
`ChatAgent` to answer a question, with the framework's `fetch_url` tool
available so the model can reach the open web. Wires together five
framework pieces in ~80 lines of `main.rs`:

| Piece                  | Crate                       | Role                                         |
| ---------------------- | --------------------------- | -------------------------------------------- |
| `OllamaProvider`       | `brainwires-provider`       | local LLM backend                            |
| `BudgetProvider`       | `brainwires-call-policy`    | hard cap on tokens / spend / rounds          |
| `WebTool`              | `brainwires-tool-builtins`  | `fetch_url` for HTTP GETs                    |
| `BuiltinToolExecutor`  | `brainwires-tool-builtins`  | dispatches tool calls from the LLM           |
| `AgentBuilder`         | `brainwires-inference`      | builds the `ChatAgent` with all of the above |

## Run

```bash
# defaults: http://localhost:11434 + gemma4:e2b
cargo run -p brainwires-web-search-agent -- "what is the capital of France?"

# override host / model
OLLAMA_BASE_URL=http://my-host:11434 \
OLLAMA_DEFAULT_MODEL=llama3:8b \
    cargo run -p brainwires-web-search-agent -- "summarize https://example.com"
```

Output includes the answer plus a one-line usage report (prompt / completion /
total tokens + duration) and the budget guard's running totals.

## What it demonstrates

- The recommended quickstart wiring — `AgentBuilder` instead of hand-rolling
  `ChatAgent::new(...).with_*()`.
- Budget enforcement via `BudgetProvider`: pre-flight rejects requests that
  would blow the `max_tokens` / `max_usd_cents` / `max_rounds` caps before any
  provider call goes out.
- The `process_message_with_report` API for surfacing per-turn token usage to
  the caller, rather than relying on out-of-band telemetry.

## What this is **not**

- A real search engine — `fetch_url` only retrieves a URL the model already
  knows or constructs (e.g. an HTML search-engine endpoint). True semantic
  web search belongs in `brainwires-rag`.
- A production agent — there's no retry policy, no streaming UI, no
  conversation persistence. Use it as a starting point, not a template.
