use anyhow::Result;
use console;
use std::sync::Arc;

use crate::providers::Provider;
use crate::tools::ToolExecutor;
use crate::types::agent::{AgentContext, AgentResponse, PermissionMode, Task};
use crate::types::message::{ChatResponse, ContentBlock, Message, MessageContent, Role};
use crate::types::provider::ChatOptions;
use crate::types::tool::{ToolContext, ToolContextExt, ToolUse};

/// Worker agent - executes specific tasks delegated by orchestrator
pub struct WorkerAgent {
    provider: Arc<dyn Provider>,
    tool_executor: ToolExecutor,
    max_iterations: u32,
}

impl WorkerAgent {
    /// Create a new worker agent
    pub fn new(provider: Arc<dyn Provider>, permission_mode: PermissionMode) -> Self {
        Self {
            provider,
            tool_executor: ToolExecutor::new(permission_mode),
            max_iterations: 10, // Workers have fewer iterations than orchestrator
        }
    }

    /// Execute a specific task
    pub async fn execute(
        &self,
        task_description: &str,
        context: &mut AgentContext,
    ) -> Result<AgentResponse> {
        let mut iterations = 0;
        let mut task = Task::new(
            uuid::Uuid::new_v4().to_string(),
            task_description.to_string(),
        );
        task.start();

        // Add initial user message
        let user_message = Message {
            role: Role::User,
            content: MessageContent::Text(task_description.to_string()),
            name: None,
            metadata: None,
        };
        context.conversation_history.push(user_message);

        loop {
            iterations += 1;
            task.increment_iteration();

            if iterations > self.max_iterations {
                task.fail(format!(
                    "Worker exceeded maximum iterations ({})",
                    self.max_iterations
                ));
                return Ok(AgentResponse {
                    message: "Task failed: exceeded maximum iterations".to_string(),
                    is_complete: true,
                    tasks: vec![task],
                    iterations,
                });
            }

            // Call the AI provider
            let response = self.call_provider(context).await?;

            // Check if task is complete
            if let Some(finish_reason) = &response.finish_reason
                && (finish_reason == "end_turn" || finish_reason == "stop")
            {
                // Extract final message
                let message_text = response
                    .message
                    .text()
                    .unwrap_or("Task completed")
                    .to_string();
                task.complete(message_text.clone());

                return Ok(AgentResponse {
                    message: message_text,
                    is_complete: true,
                    tasks: vec![task],
                    iterations,
                });
            }

            // Process tool uses in the response
            let tool_uses = self.extract_tool_uses(&response.message);

            if tool_uses.is_empty() {
                // No tool uses, treat as completion
                let message_text = response
                    .message
                    .text()
                    .unwrap_or("Task completed")
                    .to_string();
                task.complete(message_text.clone());

                return Ok(AgentResponse {
                    message: message_text,
                    is_complete: true,
                    tasks: vec![task],
                    iterations,
                });
            }

            // Add assistant message to history
            context.conversation_history.push(response.message.clone());

            // Execute tools and add results to history
            let tool_context = ToolContext::from_agent_context(context);

            for tool_use in tool_uses {
                // Log tool call to user
                eprintln!(
                    "\n🔧 Calling tool: {} with input: {}",
                    console::style(&tool_use.name).cyan().bold(),
                    console::style(
                        serde_json::to_string_pretty(&tool_use.input).unwrap_or_default()
                    )
                    .dim()
                );

                let result = self.tool_executor.execute(&tool_use, &tool_context).await?;

                // Log tool result
                if result.is_error {
                    eprintln!(
                        "❌ Tool {} failed: {}\n",
                        console::style(&tool_use.name).red(),
                        console::style(&result.content).dim()
                    );
                } else {
                    let preview = if result.content.len() > 200 {
                        format!("{}...", &result.content[..200])
                    } else {
                        result.content.clone()
                    };
                    eprintln!(
                        "✅ Tool {} completed: {}\n",
                        console::style(&tool_use.name).green(),
                        console::style(preview).dim()
                    );
                }

                // Add tool result message
                let result_message = Message {
                    role: Role::User,
                    content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                        tool_use_id: result.tool_use_id.clone(),
                        content: result.content.clone(),
                        is_error: Some(result.is_error),
                    }]),
                    name: None,
                    metadata: None,
                };

                context.conversation_history.push(result_message);
            }

            // Continue loop for next iteration
        }
    }

    /// Call the AI provider with current context
    async fn call_provider(&self, context: &AgentContext) -> Result<ChatResponse> {
        let options = ChatOptions {
            temperature: Some(0.7),
            max_tokens: Some(2048), // Conservative limit for focused worker tasks
            top_p: None,
            stop: None,
            system: Some(
                "You are a focused worker agent executing a specific task. Use the available tools efficiently to complete the task. Be concise and direct."
                    .to_string(),
            ),
            model: None,
            cache_strategy: Default::default(),
        };

        self.provider
            .chat(
                &context.conversation_history,
                Some(&context.tools),
                &options,
            )
            .await
    }

    /// Extract tool uses from a message
    fn extract_tool_uses(&self, message: &Message) -> Vec<ToolUse> {
        match &message.content {
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { id, name, input } => Some(ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    }),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        }
    }

    /// Set maximum iterations
    pub fn set_max_iterations(&mut self, max: u32) {
        self.max_iterations = max;
    }

    /// Get current permission mode
    pub fn permission_mode(&self) -> PermissionMode {
        self.tool_executor.permission_mode()
    }
}

// Note: WorkerAgent intentionally does NOT implement Default because it requires a Provider.
// Use WorkerAgent::new() with a valid provider instead.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::agent::TaskStatus;
    use crate::types::message::{
        ChatResponse, ContentBlock, Message, MessageContent, Role, StreamChunk, Usage,
    };
    use crate::types::provider::ChatOptions;
    use crate::types::tool::Tool;
    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use serde_json::json;
    use std::sync::Arc;

    /// Mock provider for testing
    struct MockProvider {
        responses: Vec<ChatResponse>,
        current_index: std::sync::Mutex<usize>,
    }

    impl MockProvider {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses,
                current_index: std::sync::Mutex::new(0),
            }
        }

        fn single_response(finish_reason: &str, text: &str) -> Self {
            Self::new(vec![ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::Text(text.to_string()),
                    name: None,
                    metadata: None,
                },
                finish_reason: Some(finish_reason.to_string()),
                usage: Usage::default(),
            }])
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock-provider"
        }

        async fn chat(
            &self,
            _messages: &[Message],
            _tools: Option<&[Tool]>,
            _options: &ChatOptions,
        ) -> Result<ChatResponse> {
            let mut index = self.current_index.lock().unwrap();
            let response = self
                .responses
                .get(*index)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No more mock responses"))?;
            *index += 1;
            Ok(response)
        }

        fn stream_chat<'a>(
            &'a self,
            _messages: &'a [Message],
            _tools: Option<&'a [Tool]>,
            _options: &'a ChatOptions,
        ) -> BoxStream<'a, Result<StreamChunk>> {
            unimplemented!("Streaming not used in worker tests")
        }
    }

    #[tokio::test]
    async fn test_worker_creation() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let worker = WorkerAgent::new(provider, PermissionMode::ReadOnly);
        assert_eq!(worker.max_iterations, 10);
        assert_eq!(worker.permission_mode(), PermissionMode::ReadOnly);
    }

    #[tokio::test]
    async fn test_set_max_iterations() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let mut worker = WorkerAgent::new(provider, PermissionMode::Full);
        worker.set_max_iterations(5);
        assert_eq!(worker.max_iterations, 5);
        assert_eq!(worker.permission_mode(), PermissionMode::Full);
    }

    #[tokio::test]
    async fn test_task_completion_with_stop_reason() {
        let provider = Arc::new(MockProvider::single_response(
            "stop",
            "Task completed successfully",
        ));
        let worker = WorkerAgent::new(provider, PermissionMode::ReadOnly);
        let mut context = AgentContext::default();

        let result = worker.execute("test task", &mut context).await.unwrap();

        assert!(result.is_complete);
        assert_eq!(result.message, "Task completed successfully");
        assert_eq!(result.iterations, 1);
        assert_eq!(result.tasks.len(), 1);
        assert_eq!(result.tasks[0].status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn test_task_completion_with_end_turn() {
        let provider = Arc::new(MockProvider::single_response("end_turn", "All done"));
        let worker = WorkerAgent::new(provider, PermissionMode::Auto);
        let mut context = AgentContext::default();

        let result = worker.execute("test task", &mut context).await.unwrap();

        assert!(result.is_complete);
        assert_eq!(result.message, "All done");
        assert_eq!(result.tasks[0].status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn test_task_completion_no_tool_uses() {
        let provider = Arc::new(MockProvider::single_response("other", "Finished"));
        let worker = WorkerAgent::new(provider, PermissionMode::ReadOnly);
        let mut context = AgentContext::default();

        let result = worker.execute("test task", &mut context).await.unwrap();

        assert!(result.is_complete);
        assert_eq!(result.message, "Finished");
    }

    #[tokio::test]
    async fn test_extract_tool_uses_from_blocks() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let worker = WorkerAgent::new(provider, PermissionMode::ReadOnly);

        let message = Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Using a tool".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "tool-1".to_string(),
                    name: "test_tool".to_string(),
                    input: json!({"arg": "value"}),
                },
            ]),
            name: None,
            metadata: None,
        };

        let tool_uses = worker.extract_tool_uses(&message);

        assert_eq!(tool_uses.len(), 1);
        assert_eq!(tool_uses[0].id, "tool-1");
        assert_eq!(tool_uses[0].name, "test_tool");
    }

    #[tokio::test]
    async fn test_extract_tool_uses_from_text() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let worker = WorkerAgent::new(provider, PermissionMode::ReadOnly);

        let message = Message {
            role: Role::Assistant,
            content: MessageContent::Text("Plain text message".to_string()),
            name: None,
            metadata: None,
        };

        let tool_uses = worker.extract_tool_uses(&message);
        assert_eq!(tool_uses.len(), 0);
    }

    #[tokio::test]
    async fn test_permission_modes() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));

        let worker_readonly = WorkerAgent::new(provider.clone(), PermissionMode::ReadOnly);
        assert_eq!(worker_readonly.permission_mode(), PermissionMode::ReadOnly);

        let worker_auto = WorkerAgent::new(provider.clone(), PermissionMode::Auto);
        assert_eq!(worker_auto.permission_mode(), PermissionMode::Auto);

        let worker_full = WorkerAgent::new(provider, PermissionMode::Full);
        assert_eq!(worker_full.permission_mode(), PermissionMode::Full);
    }
}
