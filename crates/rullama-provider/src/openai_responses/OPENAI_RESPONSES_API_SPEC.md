# OpenAI Responses API - Full Specification Reference

> Compiled from official OpenAI docs, Azure OpenAI docs, Go SDK deepwiki, and TypeScript SDK types.
> Last updated: 2026-03-06

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/responses` | Create a model response |
| GET | `/v1/responses/{response_id}` | Retrieve a response |
| DELETE | `/v1/responses/{response_id}` | Delete a response |
| POST | `/v1/responses/{response_id}/cancel` | Cancel an in-progress response |
| GET | `/v1/responses/{response_id}/input_items` | List input items |
| POST | `/v1/responses/compact` | Compact a response (context compression) |

---

## POST /v1/responses - Request Parameters

### Required
| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model ID (e.g., "gpt-4o", "gpt-4.1", "o3", "o4-mini") |
| `input` | string \| array | Text string or array of input items |

### Optional
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instructions` | string | null | System/developer instructions |
| `tools` | array | [] | Tool definitions |
| `tool_choice` | string \| object | "auto" | Tool selection strategy |
| `parallel_tool_calls` | boolean | null | Allow parallel tool calls |
| `max_output_tokens` | integer | null | Max tokens in response |
| `temperature` | number | 1.0 | Sampling temperature (0-2) |
| `top_p` | number | 1.0 | Nucleus sampling |
| `stop` | array | null | Stop sequences |
| `frequency_penalty` | number | 0 | Frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | number | 0 | Presence penalty (-2.0 to 2.0) |
| `stream` | boolean | false | Enable SSE streaming |
| `previous_response_id` | string | null | Chain to previous response |
| `store` | boolean | false | Persist response server-side |
| `metadata` | object | {} | Custom key-value pairs (max 16) |
| `truncation` | string | null | Truncation strategy: "auto" \| "disabled" |
| `reasoning` | object | null | Reasoning config (see below) |
| `text` | object | null | Text format config (see below) |
| `include` | array | [] | Extra fields to include in response |
| `user` | string | null | End-user identifier |
| `background` | boolean | false | Execute asynchronously |
| `service_tier` | string | null | Service tier preference |

### `reasoning` Object
```json
{
  "effort": "low" | "medium" | "high",
  "generate_summary": "auto" | "concise" | "detailed",
  "encrypted_content": null  // returned when include contains "reasoning.encrypted_content"
}
```
- `effort`: Controls reasoning depth (for o-series and reasoning models)
- `generate_summary`: Controls reasoning summary generation

### `text` Object (Response Format)
```json
// Plain text (default)
{ "format": { "type": "text" } }

// JSON object (unstructured)
{ "format": { "type": "json_object" } }

// JSON schema (structured output)
{
  "format": {
    "type": "json_schema",
    "name": "schema_name",
    "description": "optional description",
    "schema": { /* JSON Schema */ },
    "strict": true
  }
}
```

### `tool_choice` Values
- `"auto"` - Model decides whether to call tools
- `"required"` - Must call at least one tool
- `"none"` - No tool calls allowed
- `{ "type": "function", "name": "function_name" }` - Force specific function

### `include` Values
- `"file_search_call.results"` - Include file search results
- `"message.input_image.image_url"` - Include input image URLs
- `"computer_call_output.output.image_url"` - Include computer output images
- `"reasoning.encrypted_content"` - Include encrypted reasoning for stateless multi-turn

### `truncation` Values
- `"auto"` - Automatically truncate input to fit context
- `"disabled"` - Error if input exceeds context

---

## Input Item Types

### Message (`type: "message"`)
```json
{
  "type": "message",
  "role": "user" | "assistant" | "system" | "developer",
  "content": "string" | [/* content parts */],
  "status": "completed" | null
}
```

Roles:
- `"user"` - User message
- `"assistant"` - Assistant message (echoing back output)
- `"system"` - System instructions (legacy)
- `"developer"` - Developer instructions (preferred over system)

### Easy/Shorthand Message (no `type` field)
```json
{
  "role": "user",
  "content": "Hello"
}
```

### Function Call Output (`type: "function_call_output"`)
```json
{
  "type": "function_call_output",
  "call_id": "call_abc123",
  "output": "{\"result\": \"value\"}"
}
```

### Computer Call Output (`type: "computer_call_output"`)
```json
{
  "type": "computer_call_output",
  "call_id": "call_abc123",
  "output": {
    "type": "computer_screenshot",
    "image_url": "data:image/png;base64,..."
  }
}
```

### MCP Approval Response (`type: "mcp_approval_response"`)
```json
{
  "type": "mcp_approval_response",
  "approve": true,
  "approval_request_id": "mcpr_abc123"
}
```

### Item Reference (`type: "item_reference"`)
```json
{
  "type": "item_reference",
  "id": "msg_abc123"
}
```

---

## Content Part Types (for structured content arrays)

### Input Content Parts

#### `input_text`
```json
{ "type": "input_text", "text": "Hello world" }
```

#### `input_image`
```json
// URL form
{ "type": "input_image", "image_url": "https://..." , "detail": "auto" | "low" | "high" }

// Base64 form
{ "type": "input_image", "image_url": "data:image/jpeg;base64,..." }

// File ID form
{ "type": "input_image", "file_id": "file_abc123" }
```

#### `input_audio`
```json
{
  "type": "input_audio",
  "data": "base64_audio_data",
  "format": "wav" | "mp3" | "flac" | "webm" | "ogg"
}
```

#### `input_file`
```json
// File ID form
{ "type": "input_file", "file_id": "file_abc123" }

// Inline form
{
  "type": "input_file",
  "filename": "document.pdf",
  "file_data": "data:application/pdf;base64,..."
}
```

### Output Content Parts

#### `output_text`
```json
{
  "type": "output_text",
  "text": "Response text",
  "annotations": [
    {
      "type": "url_citation",
      "url": "https://...",
      "title": "Page Title",
      "start_index": 0,
      "end_index": 10
    },
    {
      "type": "file_citation",
      "file_id": "file_abc123",
      "quote": "quoted text",
      "start_index": 0,
      "end_index": 10
    },
    {
      "type": "file_path",
      "file_id": "file_abc123",
      "start_index": 0,
      "end_index": 10
    }
  ],
  "logprobs": []
}
```

#### `refusal`
```json
{ "type": "refusal", "refusal": "I cannot help with that." }
```

---

## Tool Types

### 1. Function Tool (`type: "function"`)
```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": { "type": "string" }
    },
    "required": ["location"]
  },
  "strict": false
}
```

### 2. Web Search (`type: "web_search_preview"`)
```json
{
  "type": "web_search_preview",
  "search_context_size": "low" | "medium" | "high",
  "user_location": {
    "type": "approximate",
    "city": "San Francisco",
    "region": "California",
    "country": "US",
    "timezone": "America/Los_Angeles"
  }
}
```

### 3. File Search (`type: "file_search"`)
```json
{
  "type": "file_search",
  "vector_store_ids": ["vs_abc123"],
  "max_num_results": 20,
  "ranking_options": {
    "ranker": "auto" | "default_2024_08_21",
    "score_threshold": 0.0
  },
  "filters": { /* metadata filter object */ }
}
```

### 4. Code Interpreter (`type: "code_interpreter"`)
```json
{
  "type": "code_interpreter",
  "container": {
    "type": "auto",
    "file_ids": ["file_abc123"]
  }
}
```

### 5. Computer Use (`type: "computer_use_preview"`)
```json
{
  "type": "computer_use_preview",
  "display_width": 1024,
  "display_height": 768,
  "environment": "browser" | "mac" | "windows" | "linux" | "ubuntu"
}
```
Actions: click, double_click, drag, keypress, move, screenshot, scroll, type, wait

### 6. MCP (`type: "mcp"`)
```json
{
  "type": "mcp",
  "server_label": "my_server",
  "server_url": "https://mcp-server.example.com",
  "require_approval": "always" | "never",
  "headers": {
    "Authorization": "Bearer token123"
  },
  "allowed_tools": ["tool1", "tool2"]
}
```

### 7. Image Generation (`type: "image_generation"`)
```json
{
  "type": "image_generation",
  "background": "transparent" | "opaque" | "auto",
  "input_image_mask": { /* mask config */ },
  "output_compression": 0-100,
  "output_format": "png" | "jpeg" | "webp",
  "partial_images": 0,
  "quality": "low" | "medium" | "high" | "auto",
  "size": "1024x1024" | "1536x1024" | "1024x1536" | "auto"
}
```

---

## Output Item Types

### Message (`type: "message"`)
```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "status": "completed" | "in_progress" | null,
  "content": [/* output content parts: output_text, refusal */]
}
```

### Function Call (`type: "function_call"`)
```json
{
  "id": "fc_abc123",
  "type": "function_call",
  "name": "get_weather",
  "arguments": "{\"location\":\"SF\"}",
  "call_id": "call_abc123",
  "status": "completed"
}
```

### File Search Call (`type: "file_search_call"`)
```json
{
  "id": "fs_abc123",
  "type": "file_search_call",
  "status": "completed",
  "queries": ["search query"],
  "results": [
    {
      "file_id": "file_abc123",
      "filename": "doc.pdf",
      "score": 0.95,
      "text": "relevant text",
      "attributes": {}
    }
  ]
}
```

### Web Search Call (`type: "web_search_call"`)
```json
{
  "id": "ws_abc123",
  "type": "web_search_call",
  "status": "completed"
}
```

### Computer Call (`type: "computer_call"`)
```json
{
  "id": "cc_abc123",
  "type": "computer_call",
  "call_id": "call_abc123",
  "action": {
    "type": "click" | "screenshot" | "type" | "scroll" | ...,
    "x": 100,
    "y": 200
  },
  "pending_safety_checks": [],
  "status": "completed"
}
```

### Code Interpreter Call (`type: "code_interpreter_call"`)
```json
{
  "id": "ci_abc123",
  "type": "code_interpreter_call",
  "code": "print('hello')",
  "container_id": "ctr_abc123",
  "status": "completed",
  "outputs": [
    { "type": "logs", "logs": "hello\n" },
    { "type": "image", "image_url": "..." },
    { "type": "file", "file_id": "file_abc123" }
  ]
}
```

### MCP Call (`type: "mcp_call"`)
```json
{
  "id": "mcp_abc123",
  "type": "mcp_call",
  "server_label": "github",
  "name": "tool_name",
  "arguments": "{}",
  "output": "result",
  "status": "completed" | "failed"
}
```

### MCP Approval Request (`type: "mcp_approval_request"`)
```json
{
  "id": "mcpr_abc123",
  "type": "mcp_approval_request",
  "name": "tool_name",
  "arguments": {},
  "server_label": "github"
}
```

### MCP List Tools (`type: "mcp_list_tools"`)
```json
{
  "id": "mcplt_abc123",
  "type": "mcp_list_tools",
  "server_label": "github",
  "tools": [{ "name": "...", "description": "...", "input_schema": {} }]
}
```

### Reasoning (`type: "reasoning"`)
```json
{
  "id": "rs_abc123",
  "type": "reasoning",
  "summary": [
    {
      "type": "summary_text",
      "text": "reasoning summary text"
    }
  ],
  "encrypted_content": "encrypted_base64..." // only with include
}
```

### Image Generation Call (`type: "image_generation_call"`)
```json
{
  "id": "ig_abc123",
  "type": "image_generation_call",
  "result": "base64_image_data",
  "status": "completed"
}
```

---

## Response Object

```json
{
  "id": "resp_abc123",
  "object": "response",
  "created_at": 1741369938.0,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "model": "gpt-4o-2024-08-06",
  "output": [/* output items */],
  "output_text": "Convenience field: concatenated text",
  "parallel_tool_calls": null,
  "previous_response_id": null,
  "reasoning": null,
  "service_tier": null,
  "metadata": {},
  "temperature": 1.0,
  "top_p": 1.0,
  "max_output_tokens": null,
  "tool_choice": null,
  "tools": [],
  "text": null,
  "truncation": null,
  "store": false,
  "usage": {
    "input_tokens": 20,
    "output_tokens": 11,
    "total_tokens": 31,
    "output_tokens_details": {
      "reasoning_tokens": 0
    }
  },
  "user": null,
  "reasoning_effort": null
}
```

### Status Values
- `"queued"` - Awaiting processing (background mode)
- `"in_progress"` - Actively processing
- `"completed"` - Finished successfully
- `"incomplete"` - Terminated early (e.g., max tokens)
- `"cancelled"` - User cancelled
- `"failed"` - Error occurred

---

## Streaming Events

All events are SSE with `event:` and `data:` fields. The `data` payload is JSON.

### Response Lifecycle Events
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `response.created` | `response` | Stream initialized |
| `response.in_progress` | `response` | Processing started |
| `response.completed` | `response` | Generation finished (includes full response with usage) |
| `response.failed` | `response` | Generation failed |
| `response.incomplete` | `response` | Response truncated/incomplete |

### Output Item Events
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `response.output_item.added` | `item`, `output_index` | New output item started |
| `response.output_item.done` | `item`, `output_index` | Output item completed |

### Text Content Events
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `response.content_part.added` | `part`, `item_id`, `output_index`, `content_index` | Content part started |
| `response.content_part.done` | `part`, `item_id`, `output_index`, `content_index` | Content part completed |
| `response.output_text.delta` | `delta`, `item_id`, `output_index`, `content_index` | Incremental text |
| `response.output_text.done` | `text`, `item_id`, `output_index`, `content_index` | Full text completed |

### Refusal Events
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `response.refusal.delta` | `delta`, `item_id`, `content_index` | Incremental refusal text |
| `response.refusal.done` | `refusal`, `item_id`, `content_index` | Refusal completed |

### Function Call Events
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `response.function_call_arguments.delta` | `delta`, `item_id`, `output_index` | Incremental function args |
| `response.function_call_arguments.done` | `arguments`, `item_id`, `output_index` | Function args completed |

### Reasoning Events
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `response.reasoning_summary_part.added` | `part`, `item_id`, `summary_index` | Reasoning summary started |
| `response.reasoning_summary_text.delta` | `delta`, `item_id`, `summary_index` | Incremental reasoning |
| `response.reasoning_summary_text.done` | `text`, `item_id`, `summary_index` | Reasoning text completed |
| `response.reasoning_summary_part.done` | `part`, `item_id`, `summary_index` | Reasoning part completed |

### File Search Events
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `response.file_search_call.in_progress` | `item_id`, `output_index` | File search starting |
| `response.file_search_call.searching` | `item_id`, `output_index` | File search active |
| `response.file_search_call.completed` | `item_id`, `output_index` | File search finished |

### Code Interpreter Events
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `response.code_interpreter_call.in_progress` | `item_id`, `output_index` | Code execution starting |
| `response.code_interpreter_call.interpreting` | `item_id`, `output_index` | Code executing |
| `response.code_interpreter_call.completed` | `item_id`, `output_index` | Code execution done |

### Web Search Events
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `response.web_search_call.in_progress` | `item_id`, `output_index` | Web search starting |
| `response.web_search_call.searching` | `item_id`, `output_index` | Web search active |
| `response.web_search_call.completed` | `item_id`, `output_index` | Web search finished |

### MCP Events
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `response.mcp_call.in_progress` | `item_id`, `output_index` | MCP call starting |
| `response.mcp_call.completed` | `item_id`, `output_index` | MCP call finished |
| `response.mcp_call.failed` | `item_id`, `output_index` | MCP call failed |
| `response.mcp_list_tools.in_progress` | `item_id` | MCP tool listing starting |
| `response.mcp_list_tools.completed` | `item_id` | MCP tool listing done |

### Image Generation Events
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `response.image_generation_call.in_progress` | `item_id` | Image gen starting |
| `response.image_generation_call.generating` | `item_id` | Image generating |
| `response.image_generation_call.partial_image` | `item_id`, `partial_image_b64` | Partial image data |
| `response.image_generation_call.completed` | `item_id`, `result` | Image gen completed |

### Error Event
| Event Type | Key Fields | Description |
|------------|------------|-------------|
| `error` | `error` (object with code, message) | Error occurred |

### Streaming Flow Examples

**Simple text:**
```
response.created
response.in_progress
response.output_item.added (message)
response.content_part.added (output_text)
response.output_text.delta (x N)
response.output_text.done
response.content_part.done
response.output_item.done
response.completed
```

**Function call:**
```
response.created
response.in_progress
response.output_item.added (function_call)
response.function_call_arguments.delta (x N)
response.function_call_arguments.done
response.output_item.done
response.completed
```

**With reasoning:**
```
response.created
response.in_progress
response.output_item.added (reasoning)
response.reasoning_summary_part.added
response.reasoning_summary_text.delta (x N)
response.reasoning_summary_text.done
response.reasoning_summary_part.done
response.output_item.done
response.output_item.added (message)
response.content_part.added
response.output_text.delta (x N)
response.output_text.done
response.content_part.done
response.output_item.done
response.completed
```

---

## GET /v1/responses/{response_id}

Returns the full response object. Supports `?stream=true&starting_after=N` to resume streaming.

## DELETE /v1/responses/{response_id}

Deletes a stored response. Returns `{ "id": "resp_...", "object": "response.deleted", "deleted": true }`.

## POST /v1/responses/{response_id}/cancel

Cancels an in-progress response. Returns the response object with `status: "cancelled"`.

## GET /v1/responses/{response_id}/input_items

Lists input items for a response. Returns paginated list:
```json
{
  "object": "list",
  "data": [/* input items */],
  "has_more": false,
  "first_id": "msg_...",
  "last_id": "msg_..."
}
```

## POST /v1/responses/compact

Compresses conversation context. Accepts `model`, `input` (array), and/or `previous_response_id`.
Returns compacted output items to use as input for the next request.

---

## Annotations Types

### URL Citation
```json
{
  "type": "url_citation",
  "url": "https://example.com",
  "title": "Page Title",
  "start_index": 0,
  "end_index": 50
}
```

### File Citation
```json
{
  "type": "file_citation",
  "file_id": "file_abc123",
  "quote": "relevant quoted text",
  "start_index": 10,
  "end_index": 30
}
```

### File Path
```json
{
  "type": "file_path",
  "file_id": "file_abc123",
  "start_index": 0,
  "end_index": 20
}
```

---

## Background Mode

Set `background: true` to execute asynchronously. Requires `store: true`.

Poll with `GET /v1/responses/{id}` until status is terminal (completed/failed/cancelled/incomplete).

Can stream background responses with `stream: true` + `background: true`.
Resume streaming with `GET /v1/responses/{id}?stream=true&starting_after={sequence_number}`.

---

## Conversation Management

### Server-side (stateful)
Use `previous_response_id` to chain responses. Server maintains full conversation context.

### Client-side (stateless)
Pass full conversation history in `input` array. Echo back output items as input items.

### Conversation Object
```json
{
  "conversation": {
    "id": "conv_abc123"
  }
}
```
Associates response with a named conversation.

---

## Context Management

```json
{
  "context_management": [
    {
      "type": "compaction",
      "compact_threshold": 50000
    }
  ]
}
```
Automatically compacts context when token count exceeds threshold (minimum: 1000).
