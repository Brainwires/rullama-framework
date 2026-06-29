//! Wire types for the OpenAI Responses API.

pub mod input;
pub mod output;
pub mod request;
pub mod response;
pub mod streaming;
pub mod tools;
pub mod websocket;

// Re-export key types at the types module level
pub use input::{InputContent, InputContentPart, ResponseInputItem};
pub use output::{
    Annotation, CodeInterpreterOutput, FileSearchResult, McpToolDef, OutputContentBlock,
    ReasoningSummaryPart, ResponseOutputItem,
};
pub use request::{
    AudioOutputConfig, ContextManagement, ConversationRef, CreateResponseRequest, ReasoningConfig,
    ResponseInput, TextFormat, TextFormatConfig, ToolChoice,
};
pub use response::{
    DeleteResponse, InputItemsList, OutputTokensDetails, ResponseError, ResponseObject,
    ResponseUsage,
};
pub use streaming::ResponseStreamEvent;
pub use tools::{CodeInterpreterContainer, RankingOptions, ResponseTool, UserLocation};
