//! OpenAI Responses API provider (`POST /v1/responses`).
//!
//! The Responses API is OpenAI's primary modern API surface, superseding Chat
//! Completions. Key differences:
//! - Input is `input` not `messages` — supports text string or structured items
//! - Output contains typed items (message, function_call, web_search, etc.)
//! - `previous_response_id` chains conversations server-side
//! - Supports built-in tools: web search, file search, code interpreter,
//!   computer use, MCP servers, image generation
//! - Streaming uses 30+ typed event types
//! - 6 REST endpoints: create, retrieve, delete, cancel, list_input_items, compact

pub mod client;
pub mod convert;
pub mod provider;
pub mod types;
pub mod websocket;
pub mod ws_provider;

// Re-export key public types at module level for ergonomic imports
pub use client::ResponsesClient;
pub use provider::OpenAiResponsesProvider;

// Re-export wire types used by external code
pub use types::{
    AudioOutputConfig, CreateResponseRequest, InputContent, InputContentPart, OutputContentBlock,
    ResponseInput, ResponseInputItem, ResponseObject, ResponseOutputItem, ResponseStreamEvent,
    ResponseTool, ResponseUsage, ToolChoice,
};

// WebSocket transport
pub use websocket::ResponsesWebSocket;
pub use ws_provider::OpenAiResponsesWsProvider;
