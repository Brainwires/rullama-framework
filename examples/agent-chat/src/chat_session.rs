use anyhow::Result;
use futures::StreamExt;
use std::sync::Arc;

use brainwires_core::{
    ChatOptions, ContentBlock, Message, MessageContent, Provider, StreamChunk, Tool, ToolContext,
    ToolResult, ToolUse,
};
use brainwires_tool_builtins::BuiltinToolExecutor;
use brainwires_tool_runtime::ToolRegistry;

use crate::cli::Cli;
use crate::config::ChatConfig;

#[derive(Debug, Clone)]
pub enum ApprovalResponse {
    Yes,
    No,
    Always,
}

pub type ApprovalCallback = Box<dyn Fn(&str, &serde_json::Value) -> ApprovalResponse + Send + Sync>;

pub struct ChatSession {
    provider: Arc<dyn Provider>,
    executor: Arc<BuiltinToolExecutor>,
    messages: Vec<Message>,
    options: ChatOptions,
    permission_mode: String,
    auto_approved_tools: std::collections::HashSet<String>,
    approval_callback: Option<ApprovalCallback>,
}

impl ChatSession {
    pub fn new(
        provider: Arc<dyn Provider>,
        tools: ToolRegistry,
        cli: &Cli,
        config: &ChatConfig,
    ) -> Self {
        let system = cli
            .system
            .clone()
            .or_else(|| config.system_prompt.clone())
            .unwrap_or_else(|| {
                "You are a helpful AI assistant with access to tools for file operations, \
                 shell commands, git, web fetching, and code search. Use them when helpful."
                    .to_string()
            });

        let options = ChatOptions::new()
            .temperature(cli.temperature.unwrap_or(config.temperature))
            .max_tokens(cli.max_tokens.unwrap_or(config.max_tokens))
            .system(system);

        let working_directory = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string());

        let context = ToolContext {
            working_directory,
            user_id: None,
            metadata: std::collections::HashMap::new(),
            capabilities: None,
            idempotency_registry: None,
            staging_backend: None,
            intended_writes: None,
        };
        let executor = Arc::new(BuiltinToolExecutor::new(tools, context));

        Self {
            provider,
            executor,
            messages: Vec::new(),
            options,
            permission_mode: config.permission_mode.clone(),
            auto_approved_tools: std::collections::HashSet::new(),
            approval_callback: None,
        }
    }

    pub fn set_approval_callback(&mut self, cb: ApprovalCallback) {
        self.approval_callback = Some(cb);
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn provider_name(&self) -> &str {
        self.provider.name()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub async fn send_message(&mut self, input: &str) -> Result<Vec<StreamEvent>> {
        self.messages.push(Message::user(input));
        self.run_completion().await
    }

    async fn run_completion(&mut self) -> Result<Vec<StreamEvent>> {
        let mut all_events = Vec::new();
        let max_tool_rounds = 10;

        for _ in 0..max_tool_rounds {
            let tool_defs: Vec<Tool> = self.executor.tools();
            let tools_opt = if tool_defs.is_empty() {
                None
            } else {
                Some(tool_defs.as_slice())
            };

            // Collect stream completely, then drop it before mutating self
            let (text_buf, tool_uses, events, response_id) = self.collect_stream(tools_opt).await?;
            all_events.extend(events);

            if tool_uses.is_empty() {
                self.messages.push(Message::assistant(&text_buf));
                break;
            }

            // Build assistant message with tool uses
            let mut blocks = Vec::new();
            if !text_buf.is_empty() {
                blocks.push(ContentBlock::Text { text: text_buf });
            }
            for tu in &tool_uses {
                blocks.push(ContentBlock::ToolUse {
                    id: tu.id.clone(),
                    name: tu.name.clone(),
                    input: tu.input.clone(),
                });
            }
            let metadata = response_id.map(|rid| serde_json::json!({"response_id": rid}));
            self.messages.push(Message {
                role: brainwires_core::Role::Assistant,
                content: MessageContent::Blocks(blocks),
                name: None,
                metadata,
            });

            // Execute tools via BuiltinToolExecutor (with TUI approval logic)
            let mut result_blocks = Vec::new();
            for tu in &tool_uses {
                all_events.push(StreamEvent::ToolCall {
                    name: tu.name.clone(),
                    input: tu.input.clone(),
                });

                let result = self.execute_tool(tu).await;
                all_events.push(StreamEvent::ToolResult {
                    name: tu.name.clone(),
                    content: result.content.clone(),
                    is_error: result.is_error,
                });

                result_blocks.push(ContentBlock::ToolResult {
                    tool_use_id: tu.id.clone(),
                    content: result.content,
                    is_error: Some(result.is_error),
                });
            }

            self.messages.push(Message {
                role: brainwires_core::Role::User,
                content: MessageContent::Blocks(result_blocks),
                name: None,
                metadata: None,
            });
        }

        Ok(all_events)
    }

    /// Collect the entire stream into text + tool uses, returning ownership.
    /// This avoids borrow conflicts since the stream is fully consumed before returning.
    async fn collect_stream(
        &self,
        tools_opt: Option<&[Tool]>,
    ) -> Result<(String, Vec<ToolUse>, Vec<StreamEvent>, Option<String>)> {
        let mut stream = self
            .provider
            .stream_chat(&self.messages, tools_opt, &self.options);

        let mut text_buf = String::new();
        let mut events = Vec::new();
        let mut tool_uses: Vec<ToolUse> = Vec::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_input = String::new();
        let mut last_response_id: Option<String> = None;

        while let Some(chunk) = stream.next().await {
            match chunk? {
                StreamChunk::Text(t) => {
                    events.push(StreamEvent::Text(t.clone()));
                    text_buf.push_str(&t);
                }
                StreamChunk::ToolUse { id, name } => {
                    if !current_tool_id.is_empty() {
                        let input: serde_json::Value = serde_json::from_str(&current_tool_input)
                            .unwrap_or(serde_json::Value::Null);
                        tool_uses.push(ToolUse {
                            id: std::mem::take(&mut current_tool_id),
                            name: std::mem::take(&mut current_tool_name),
                            input,
                        });
                        current_tool_input.clear();
                    }
                    current_tool_id = id;
                    current_tool_name = name;
                }
                StreamChunk::ToolInputDelta { partial_json, .. } => {
                    current_tool_input.push_str(&partial_json);
                }
                StreamChunk::ToolCall {
                    call_id,
                    response_id,
                    tool_name,
                    parameters,
                    ..
                } => {
                    // Brainwires backend sends complete tool calls in a single event
                    last_response_id = Some(response_id);
                    tool_uses.push(ToolUse {
                        id: call_id,
                        name: tool_name,
                        input: parameters,
                    });
                }
                StreamChunk::Usage(usage) => {
                    events.push(StreamEvent::Usage {
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                    });
                }
                StreamChunk::Done => {}
                StreamChunk::ContextCompacted { .. } => {
                    // Context compaction is handled by the agent layer; chat session ignores it
                }
            }
        }

        // Flush last tool
        if !current_tool_id.is_empty() {
            let input: serde_json::Value =
                serde_json::from_str(&current_tool_input).unwrap_or(serde_json::Value::Null);
            tool_uses.push(ToolUse {
                id: current_tool_id,
                name: current_tool_name,
                input,
            });
        }

        Ok((text_buf, tool_uses, events, last_response_id))
    }

    /// Execute a tool call with TUI-specific approval logic, delegating actual
    /// execution to the framework's BuiltinToolExecutor.
    async fn execute_tool(&mut self, tool_use: &ToolUse) -> ToolResult {
        if self.permission_mode == "ask" && !self.auto_approved_tools.contains(&tool_use.name) {
            if let Some(ref cb) = self.approval_callback {
                match cb(&tool_use.name, &tool_use.input) {
                    ApprovalResponse::Yes => {}
                    ApprovalResponse::Always => {
                        self.auto_approved_tools.insert(tool_use.name.clone());
                    }
                    ApprovalResponse::No => {
                        return ToolResult::error(
                            tool_use.id.clone(),
                            "Tool call rejected by user".to_string(),
                        );
                    }
                }
            }
        } else if self.permission_mode == "reject" {
            let read_only = matches!(
                tool_use.name.as_str(),
                "read_file"
                    | "list_directory"
                    | "search_code"
                    | "search_files"
                    | "fetch_url"
                    | "git_status"
                    | "git_diff"
                    | "git_log"
            );
            if !read_only {
                return ToolResult::error(
                    tool_use.id.clone(),
                    "Tool call blocked by reject permission mode".to_string(),
                );
            }
        }

        // Delegate to the framework's BuiltinToolExecutor
        self.executor
            .execute_tool(&tool_use.name, &tool_use.id, &tool_use.input)
            .await
    }
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Text(String),
    ToolCall {
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        name: String,
        content: String,
        is_error: bool,
    },
    Usage {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
}
