//! The `A2aHandler` trait — implement once, serve all 3 bindings.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::agent_card::AgentCard;
use crate::error::A2aError;
use crate::params::*;
use crate::push_notification::TaskPushNotificationConfig;
use crate::streaming::{SendMessageResponse, StreamResponse};
use crate::task::Task;

/// Core handler trait for A2A agents.
///
/// Implement this trait to define agent behavior. The server infrastructure
/// will route requests from JSON-RPC, REST, and gRPC to these methods.
#[async_trait]
pub trait A2aHandler: Send + Sync + 'static {
    /// Return the agent card for discovery.
    fn agent_card(&self) -> &AgentCard;

    /// Handle a `message/send` request.
    async fn on_send_message(
        &self,
        req: SendMessageRequest,
    ) -> Result<SendMessageResponse, A2aError>;

    /// Handle a `message/stream` request (server-streaming).
    async fn on_send_streaming_message(
        &self,
        req: SendMessageRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>>, A2aError>;

    /// Handle a `tasks/get` request.
    async fn on_get_task(&self, req: GetTaskRequest) -> Result<Task, A2aError>;

    /// Handle a `tasks/list` request.
    async fn on_list_tasks(&self, req: ListTasksRequest) -> Result<ListTasksResponse, A2aError>;

    /// Handle a `tasks/cancel` request.
    async fn on_cancel_task(&self, req: CancelTaskRequest) -> Result<Task, A2aError>;

    /// Handle a `tasks/resubscribe` request (server-streaming).
    async fn on_subscribe_to_task(
        &self,
        req: SubscribeToTaskRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>>, A2aError>;

    /// Create a push notification config. Default returns unsupported.
    async fn on_create_push_config(
        &self,
        _config: TaskPushNotificationConfig,
    ) -> Result<TaskPushNotificationConfig, A2aError> {
        Err(A2aError::push_not_supported())
    }

    /// Get a push notification config. Default returns unsupported.
    async fn on_get_push_config(
        &self,
        _req: GetTaskPushNotificationConfigRequest,
    ) -> Result<TaskPushNotificationConfig, A2aError> {
        Err(A2aError::push_not_supported())
    }

    /// List push notification configs. Default returns unsupported.
    async fn on_list_push_configs(
        &self,
        _req: ListTaskPushNotificationConfigsRequest,
    ) -> Result<ListTaskPushNotificationConfigsResponse, A2aError> {
        Err(A2aError::push_not_supported())
    }

    /// Delete a push notification config. Default returns unsupported.
    async fn on_delete_push_config(
        &self,
        _req: DeleteTaskPushNotificationConfigRequest,
    ) -> Result<(), A2aError> {
        Err(A2aError::push_not_supported())
    }

    /// Get the authenticated extended agent card. Default returns not configured.
    async fn on_get_extended_agent_card(
        &self,
        _req: GetExtendedAgentCardRequest,
    ) -> Result<AgentCard, A2aError> {
        Err(A2aError::extended_card_not_configured())
    }
}
