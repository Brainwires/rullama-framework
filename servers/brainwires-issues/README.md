# brainwires-issues

Standalone MCP server binary for lightweight project issue and bug tracking.

Inspired by Linear's agent workflow, this server exposes issue management as MCP tools and slash commands — create, triage, update, and comment on issues directly from any AI assistant.

## Quick Start

```bash
cargo run -p brainwires-issues -- serve
```

Or install and run:

```bash
cargo install --path extras/brainwires-issues
brainwires-issues serve
```

## Claude Code / Claude Desktop Configuration

Add to your MCP server configuration:

```json
{
  "brainwires-issues": {
    "command": "/path/to/brainwires-issues",
    "args": ["serve"]
  }
}
```

Or with `cargo run` during development:

```json
{
  "brainwires-issues": {
    "command": "cargo",
    "args": ["run", "-p", "brainwires-issues", "--", "serve"]
  }
}
```

## MCP Tools

### Issues

| Tool | Description |
|------|-------------|
| `create_issue` | Create a new issue with title, description, priority, assignee, project, labels, and optional parent (sub-issue) |
| `get_issue` | Get an issue by UUID or display number (e.g. `#42`) |
| `list_issues` | List issues with filters for project, status, assignee, and label — cursor-paginated |
| `update_issue` | Update any field on an existing issue |
| `close_issue` | Close or cancel an issue (`done` / `cancelled`) |
| `delete_issue` | Permanently delete an issue and its comments |
| `search_issues` | Keyword search across issue titles and descriptions |

### Comments

| Tool | Description |
|------|-------------|
| `add_comment` | Add a Markdown comment to an issue |
| `list_comments` | List comments on an issue — cursor-paginated |
| `delete_comment` | Delete a comment by UUID |

## MCP Prompts (Slash Commands)

| Prompt | Description |
|--------|-------------|
| `/create` | Guided issue creation |
| `/list` | List open issues, optionally filtered by project |
| `/search` | Search issues by keyword |
| `/triage` | Review backlog issues and suggest priority, status, and assignee |

## Data Model

### Issue Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `number` | u64 | Auto-incrementing display number (`#1`, `#2`, …) |
| `title` | string | Short title |
| `description` | string | Full description (Markdown) |
| `status` | enum | `backlog` · `todo` · `in_progress` · `in_review` · `done` · `cancelled` |
| `priority` | enum | `no_priority` · `low` · `medium` · `high` · `urgent` |
| `labels` | string[] | Arbitrary tags |
| `assignee` | string? | Person or agent assigned |
| `project` | string? | Project or milestone name |
| `parent_id` | UUID? | Parent issue for sub-issues |
| `created_at` | i64 | Unix timestamp |
| `updated_at` | i64 | Unix timestamp |
| `closed_at` | i64? | Unix timestamp when closed |

### Comment Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `issue_id` | UUID | Parent issue |
| `author` | string? | Author name or identifier |
| `body` | string | Comment body (Markdown) |
| `created_at` | i64 | Unix timestamp |
| `updated_at` | i64 | Unix timestamp |

## Storage

Issues and comments are persisted in **LanceDB** (embedded, no external server required) using the `brainwires-storage` backend-agnostic layer. The default database path follows the platform convention set by `brainwires-storage`.

The same store implementations work with any `StorageBackend` (PostgreSQL, MySQL, SurrealDB) by swapping the backend type parameter.

## License

Licensed under the MIT License. See [LICENSE](../../LICENSE) for details.
