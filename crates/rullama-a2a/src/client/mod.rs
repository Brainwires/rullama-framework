//! A2A client — unified client with transport selection.

/// Agent card discovery.
pub mod discovery;
/// gRPC transport.
pub mod grpc_transport;
/// JSON-RPC over HTTP+SSE transport.
pub mod jsonrpc_transport;
/// HTTP/REST transport.
pub mod rest_transport;
/// SSE stream parser.
pub mod sse;

pub use discovery::discover_agent_card;
#[cfg(feature = "grpc-client")]
pub use grpc_transport::GrpcTransport;
pub use jsonrpc_transport::JsonRpcTransport;
pub use rest_transport::RestTransport;

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use url::Url;

use crate::agent_card::AgentCard;
use crate::error::A2aError;
use crate::jsonrpc;
use crate::params::*;
use crate::push_notification::TaskPushNotificationConfig;
use crate::streaming::{SendMessageResponse, StreamResponse};
use crate::task::Task;

/// Transport selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// JSON-RPC 2.0 over HTTP with SSE streaming.
    JsonRpc,
    /// HTTP/REST endpoints.
    Rest,
    /// gRPC (requires `grpc-client` feature).
    Grpc,
}

/// Unified A2A client.
pub struct A2aClient {
    transport: Transport,
    jsonrpc: Option<Arc<JsonRpcTransport>>,
    rest: Option<Arc<RestTransport>>,
    #[cfg(feature = "grpc-client")]
    grpc: Option<Arc<tokio::sync::Mutex<GrpcTransport>>>,
}

impl A2aClient {
    /// Create a client using JSON-RPC transport.
    pub fn new_jsonrpc(base_url: Url) -> Self {
        let client = reqwest::Client::new();
        Self {
            transport: Transport::JsonRpc,
            jsonrpc: Some(Arc::new(JsonRpcTransport::new(base_url, client, None))),
            rest: None,
            #[cfg(feature = "grpc-client")]
            grpc: None,
        }
    }

    /// Create a client using REST transport.
    pub fn new_rest(base_url: Url) -> Self {
        let client = reqwest::Client::new();
        Self {
            transport: Transport::Rest,
            jsonrpc: None,
            rest: Some(Arc::new(RestTransport::new(base_url, client, None))),
            #[cfg(feature = "grpc-client")]
            grpc: None,
        }
    }

    /// Create a client using gRPC transport.
    #[cfg(feature = "grpc-client")]
    pub async fn new_grpc(endpoint: &str) -> Result<Self, A2aError> {
        let transport = GrpcTransport::connect(endpoint).await?;
        Ok(Self {
            transport: Transport::Grpc,
            jsonrpc: None,
            rest: None,
            grpc: Some(Arc::new(tokio::sync::Mutex::new(transport))),
        })
    }

    /// Set a bearer token for authentication.
    ///
    /// Returns a new client with the token applied to the active transport.
    pub fn with_bearer_token(self, token: &str) -> Self {
        let token = token.to_string();
        match self.transport {
            Transport::JsonRpc => {
                if let Some(t) = &self.jsonrpc {
                    let new_transport = JsonRpcTransport::new(
                        t.base_url().clone(),
                        t.http_client().clone(),
                        Some(token),
                    );
                    Self {
                        jsonrpc: Some(Arc::new(new_transport)),
                        ..self
                    }
                } else {
                    self
                }
            }
            Transport::Rest => {
                if let Some(t) = &self.rest {
                    let new_transport = RestTransport::new(
                        t.base_url().clone(),
                        t.http_client().clone(),
                        Some(token),
                    );
                    Self {
                        rest: Some(Arc::new(new_transport)),
                        ..self
                    }
                } else {
                    self
                }
            }
            Transport::Grpc => {
                // gRPC auth is set at connect time; log a warning
                tracing::warn!(
                    "Bearer token on existing gRPC transport not supported; pass token at connect time"
                );
                self
            }
        }
    }

    /// Discover an agent card from a well-known URL.
    pub async fn discover(base_url: &str) -> Result<AgentCard, A2aError> {
        discover_agent_card(base_url).await
    }

    /// Send a message.
    pub async fn send_message(
        &self,
        req: SendMessageRequest,
    ) -> Result<SendMessageResponse, A2aError> {
        match self.transport {
            Transport::JsonRpc => {
                let t = self
                    .jsonrpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No JSON-RPC transport"))?;
                let params = serde_json::to_value(&req)?;
                let result = t.call(jsonrpc::METHOD_MESSAGE_SEND, params).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            Transport::Rest => {
                let t = self
                    .rest
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No REST transport"))?;
                let result = t.post("/message:send", &req).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            #[cfg(feature = "grpc-client")]
            Transport::Grpc => {
                let t = self
                    .grpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No gRPC transport"))?;
                let mut guard = t.lock().await;
                guard.send_message(req).await
            }
            #[cfg(not(feature = "grpc-client"))]
            Transport::Grpc => Err(A2aError::unsupported_operation("gRPC not enabled")),
        }
    }

    /// Stream a message (returns SSE events).
    pub fn stream_message(
        &self,
        req: SendMessageRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>> {
        match self.transport {
            Transport::JsonRpc => {
                if let Some(t) = &self.jsonrpc {
                    let params = serde_json::to_value(&req).unwrap_or_default();
                    t.call_stream(jsonrpc::METHOD_MESSAGE_STREAM, params)
                } else {
                    Box::pin(futures::stream::once(async {
                        Err(A2aError::internal("No JSON-RPC transport"))
                    }))
                }
            }
            Transport::Rest => {
                if let Some(t) = &self.rest {
                    let body = serde_json::to_value(&req).unwrap_or_default();
                    t.post_stream("/message:stream", body)
                } else {
                    Box::pin(futures::stream::once(async {
                        Err(A2aError::internal("No REST transport"))
                    }))
                }
            }
            #[cfg(feature = "grpc-client")]
            Transport::Grpc => {
                if let Some(t) = &self.grpc {
                    let grpc = t.clone();
                    Box::pin(async_stream::stream! {
                        let inner = {
                            let mut guard = grpc.lock().await;
                            guard.send_streaming_message(req).await
                        }; // lock dropped here
                        match inner {
                            Ok(mut stream) => {
                                use futures::StreamExt;
                                while let Some(item) = stream.next().await {
                                    yield item;
                                }
                            }
                            Err(e) => yield Err(e),
                        }
                    })
                } else {
                    Box::pin(futures::stream::once(async {
                        Err(A2aError::internal("No gRPC transport"))
                    }))
                }
            }
            #[cfg(not(feature = "grpc-client"))]
            Transport::Grpc => Box::pin(futures::stream::once(async {
                Err(A2aError::unsupported_operation("gRPC not enabled"))
            })),
        }
    }

    /// Get a task by ID.
    pub async fn get_task(&self, req: GetTaskRequest) -> Result<Task, A2aError> {
        match self.transport {
            Transport::JsonRpc => {
                let t = self
                    .jsonrpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No JSON-RPC transport"))?;
                let params = serde_json::to_value(&req)?;
                let result = t.call(jsonrpc::METHOD_TASKS_GET, params).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            Transport::Rest => {
                let t = self
                    .rest
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No REST transport"))?;
                let result = t.get(&format!("/tasks/{}", req.id)).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            #[cfg(feature = "grpc-client")]
            Transport::Grpc => {
                let t = self
                    .grpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No gRPC transport"))?;
                let mut guard = t.lock().await;
                guard.get_task(req).await
            }
            #[cfg(not(feature = "grpc-client"))]
            Transport::Grpc => Err(A2aError::unsupported_operation("gRPC not enabled")),
        }
    }

    /// List tasks.
    pub async fn list_tasks(&self, req: ListTasksRequest) -> Result<ListTasksResponse, A2aError> {
        match self.transport {
            Transport::JsonRpc => {
                let t = self
                    .jsonrpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No JSON-RPC transport"))?;
                let params = serde_json::to_value(&req)?;
                let result = t.call(jsonrpc::METHOD_TASKS_LIST, params).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            Transport::Rest => {
                let t = self
                    .rest
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No REST transport"))?;
                let result = t.get("/tasks").await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            #[cfg(feature = "grpc-client")]
            Transport::Grpc => {
                let t = self
                    .grpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No gRPC transport"))?;
                let mut guard = t.lock().await;
                guard.list_tasks(req).await
            }
            #[cfg(not(feature = "grpc-client"))]
            Transport::Grpc => Err(A2aError::unsupported_operation("gRPC not enabled")),
        }
    }

    /// Cancel a task.
    pub async fn cancel_task(&self, req: CancelTaskRequest) -> Result<Task, A2aError> {
        match self.transport {
            Transport::JsonRpc => {
                let t = self
                    .jsonrpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No JSON-RPC transport"))?;
                let params = serde_json::to_value(&req)?;
                let result = t.call(jsonrpc::METHOD_TASKS_CANCEL, params).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            Transport::Rest => {
                let t = self
                    .rest
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No REST transport"))?;
                let result = t.post(&format!("/tasks/{}:cancel", req.id), &req).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            #[cfg(feature = "grpc-client")]
            Transport::Grpc => {
                let t = self
                    .grpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No gRPC transport"))?;
                let mut guard = t.lock().await;
                guard.cancel_task(req).await
            }
            #[cfg(not(feature = "grpc-client"))]
            Transport::Grpc => Err(A2aError::unsupported_operation("gRPC not enabled")),
        }
    }

    /// Subscribe to task updates.
    pub fn subscribe_to_task(
        &self,
        req: SubscribeToTaskRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>> {
        match self.transport {
            Transport::JsonRpc => {
                if let Some(t) = &self.jsonrpc {
                    let params = serde_json::to_value(&req).unwrap_or_default();
                    t.call_stream(jsonrpc::METHOD_TASKS_RESUBSCRIBE, params)
                } else {
                    Box::pin(futures::stream::once(async {
                        Err(A2aError::internal("No JSON-RPC transport"))
                    }))
                }
            }
            Transport::Rest => {
                if let Some(t) = &self.rest {
                    let body = serde_json::to_value(&req).unwrap_or_default();
                    t.post_stream(&format!("/tasks/{}:subscribe", req.id), body)
                } else {
                    Box::pin(futures::stream::once(async {
                        Err(A2aError::internal("No REST transport"))
                    }))
                }
            }
            #[cfg(feature = "grpc-client")]
            Transport::Grpc => {
                if let Some(t) = &self.grpc {
                    let grpc = t.clone();
                    Box::pin(async_stream::stream! {
                        let inner = {
                            let mut guard = grpc.lock().await;
                            guard.subscribe_to_task(req).await
                        }; // lock dropped here
                        match inner {
                            Ok(mut stream) => {
                                use futures::StreamExt;
                                while let Some(item) = stream.next().await {
                                    yield item;
                                }
                            }
                            Err(e) => yield Err(e),
                        }
                    })
                } else {
                    Box::pin(futures::stream::once(async {
                        Err(A2aError::internal("No gRPC transport"))
                    }))
                }
            }
            #[cfg(not(feature = "grpc-client"))]
            Transport::Grpc => Box::pin(futures::stream::once(async {
                Err(A2aError::unsupported_operation("gRPC not enabled"))
            })),
        }
    }

    /// Set push notification config.
    pub async fn set_push_config(
        &self,
        config: TaskPushNotificationConfig,
    ) -> Result<TaskPushNotificationConfig, A2aError> {
        match self.transport {
            Transport::JsonRpc => {
                let t = self
                    .jsonrpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No JSON-RPC transport"))?;
                let params = serde_json::to_value(&config)?;
                let result = t.call(jsonrpc::METHOD_PUSH_CONFIG_SET, params).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            Transport::Rest => {
                let t = self
                    .rest
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No REST transport"))?;
                let path = format!("/tasks/{}/pushNotificationConfigs", config.task_id);
                let result = t.post(&path, &config).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            #[cfg(feature = "grpc-client")]
            Transport::Grpc => {
                let t = self
                    .grpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No gRPC transport"))?;
                let mut guard = t.lock().await;
                guard.create_push_config(config).await
            }
            #[cfg(not(feature = "grpc-client"))]
            Transport::Grpc => Err(A2aError::unsupported_operation("gRPC not enabled")),
        }
    }

    /// Get push notification config.
    pub async fn get_push_config(
        &self,
        req: GetTaskPushNotificationConfigRequest,
    ) -> Result<TaskPushNotificationConfig, A2aError> {
        match self.transport {
            Transport::JsonRpc => {
                let t = self
                    .jsonrpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No JSON-RPC transport"))?;
                let params = serde_json::to_value(&req)?;
                let result = t.call(jsonrpc::METHOD_PUSH_CONFIG_GET, params).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            Transport::Rest => {
                let t = self
                    .rest
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No REST transport"))?;
                let path = format!(
                    "/tasks/{}/pushNotificationConfigs/{}",
                    req.task_id, req.config_id
                );
                let result = t.get(&path).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            #[cfg(feature = "grpc-client")]
            Transport::Grpc => {
                let t = self
                    .grpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No gRPC transport"))?;
                let mut guard = t.lock().await;
                guard.get_push_config(req).await
            }
            #[cfg(not(feature = "grpc-client"))]
            Transport::Grpc => Err(A2aError::unsupported_operation("gRPC not enabled")),
        }
    }

    /// Delete push notification config.
    pub async fn delete_push_config(
        &self,
        req: DeleteTaskPushNotificationConfigRequest,
    ) -> Result<(), A2aError> {
        match self.transport {
            Transport::JsonRpc => {
                let t = self
                    .jsonrpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No JSON-RPC transport"))?;
                let params = serde_json::to_value(&req)?;
                let _ = t.call(jsonrpc::METHOD_PUSH_CONFIG_DELETE, params).await?;
                Ok(())
            }
            Transport::Rest => {
                let t = self
                    .rest
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No REST transport"))?;
                let path = format!(
                    "/tasks/{}/pushNotificationConfigs/{}",
                    req.task_id, req.config_id
                );
                t.delete(&path).await
            }
            #[cfg(feature = "grpc-client")]
            Transport::Grpc => {
                let t = self
                    .grpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No gRPC transport"))?;
                let mut guard = t.lock().await;
                guard.delete_push_config(req).await
            }
            #[cfg(not(feature = "grpc-client"))]
            Transport::Grpc => Err(A2aError::unsupported_operation("gRPC not enabled")),
        }
    }

    /// List push notification configs.
    pub async fn list_push_configs(
        &self,
        req: ListTaskPushNotificationConfigsRequest,
    ) -> Result<ListTaskPushNotificationConfigsResponse, A2aError> {
        match self.transport {
            Transport::JsonRpc => {
                let t = self
                    .jsonrpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No JSON-RPC transport"))?;
                let params = serde_json::to_value(&req)?;
                let result = t.call(jsonrpc::METHOD_PUSH_CONFIG_LIST, params).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            Transport::Rest => {
                let t = self
                    .rest
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No REST transport"))?;
                let path = format!("/tasks/{}/pushNotificationConfigs", req.task_id);
                let result = t.get(&path).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            #[cfg(feature = "grpc-client")]
            Transport::Grpc => {
                let t = self
                    .grpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No gRPC transport"))?;
                let mut guard = t.lock().await;
                guard.list_push_configs(req).await
            }
            #[cfg(not(feature = "grpc-client"))]
            Transport::Grpc => Err(A2aError::unsupported_operation("gRPC not enabled")),
        }
    }

    /// Get the authenticated extended agent card.
    pub async fn get_authenticated_extended_card(
        &self,
        req: GetExtendedAgentCardRequest,
    ) -> Result<AgentCard, A2aError> {
        match self.transport {
            Transport::JsonRpc => {
                let t = self
                    .jsonrpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No JSON-RPC transport"))?;
                let params = serde_json::to_value(&req)?;
                let result = t.call(jsonrpc::METHOD_EXTENDED_CARD, params).await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            Transport::Rest => {
                let t = self
                    .rest
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No REST transport"))?;
                let result = t.get("/extendedAgentCard").await?;
                serde_json::from_value(result).map_err(Into::into)
            }
            #[cfg(feature = "grpc-client")]
            Transport::Grpc => {
                let t = self
                    .grpc
                    .as_ref()
                    .ok_or_else(|| A2aError::internal("No gRPC transport"))?;
                let mut guard = t.lock().await;
                guard.get_extended_agent_card(req).await
            }
            #[cfg(not(feature = "grpc-client"))]
            Transport::Grpc => Err(A2aError::unsupported_operation("gRPC not enabled")),
        }
    }
}
