//! gRPC transport using tonic.

#[cfg(feature = "grpc-client")]
mod grpc_impl {
    use std::pin::Pin;

    use futures::Stream;
    use tokio_stream::StreamExt;

    use crate::error::A2aError;
    use crate::params;
    use crate::proto::lf_a2a_v1;
    use crate::streaming::StreamResponse;
    use crate::task::Task;

    /// gRPC transport client.
    pub struct GrpcTransport {
        client: lf_a2a_v1::a2a_service_client::A2aServiceClient<tonic::transport::Channel>,
    }

    impl GrpcTransport {
        /// Connect to a gRPC endpoint.
        pub async fn connect(endpoint: &str) -> Result<Self, A2aError> {
            let client =
                lf_a2a_v1::a2a_service_client::A2aServiceClient::connect(endpoint.to_string())
                    .await
                    .map_err(|e| A2aError::internal(format!("gRPC connect failed: {e}")))?;
            Ok(Self { client })
        }

        /// Send a message (unary).
        pub async fn send_message(
            &mut self,
            req: params::SendMessageRequest,
        ) -> Result<crate::streaming::SendMessageResponse, A2aError> {
            let proto_req = lf_a2a_v1::SendMessageRequest {
                tenant: req.tenant.unwrap_or_default(),
                message: Some(req.message.into()),
                configuration: None,
                metadata: None,
            };
            let resp = self
                .client
                .send_message(proto_req)
                .await
                .map_err(|e| A2aError::internal(format!("gRPC error: {e}")))?;
            let inner = resp.into_inner();
            match inner.payload {
                Some(lf_a2a_v1::send_message_response::Payload::Task(t)) => {
                    Ok(crate::streaming::SendMessageResponse {
                        task: Some(t.into()),
                        message: None,
                    })
                }
                Some(lf_a2a_v1::send_message_response::Payload::Message(m)) => {
                    Ok(crate::streaming::SendMessageResponse {
                        task: None,
                        message: Some(m.into()),
                    })
                }
                None => Err(A2aError::internal("Empty gRPC response")),
            }
        }

        /// Send a streaming message.
        pub async fn send_streaming_message(
            &mut self,
            req: params::SendMessageRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>>, A2aError>
        {
            let proto_req = lf_a2a_v1::SendMessageRequest {
                tenant: req.tenant.unwrap_or_default(),
                message: Some(req.message.into()),
                configuration: None,
                metadata: None,
            };
            let resp = self
                .client
                .send_streaming_message(proto_req)
                .await
                .map_err(|e| A2aError::internal(format!("gRPC error: {e}")))?;

            let stream = resp.into_inner().map(|item| {
                item.map_err(|e| A2aError::internal(format!("gRPC stream error: {e}")))
                    .and_then(proto_stream_response_to_stream_response)
            });

            Ok(Box::pin(stream))
        }

        /// Get a task by ID.
        pub async fn get_task(&mut self, req: params::GetTaskRequest) -> Result<Task, A2aError> {
            let proto_req = lf_a2a_v1::GetTaskRequest {
                tenant: req.tenant.unwrap_or_default(),
                id: req.id,
                history_length: req.history_length,
            };
            let resp = self
                .client
                .get_task(proto_req)
                .await
                .map_err(|e| A2aError::internal(format!("gRPC error: {e}")))?;
            Ok(resp.into_inner().into())
        }

        /// Cancel a task.
        pub async fn cancel_task(
            &mut self,
            req: params::CancelTaskRequest,
        ) -> Result<Task, A2aError> {
            let proto_req = lf_a2a_v1::CancelTaskRequest {
                tenant: req.tenant.unwrap_or_default(),
                id: req.id,
                metadata: None,
            };
            let resp = self
                .client
                .cancel_task(proto_req)
                .await
                .map_err(|e| A2aError::internal(format!("gRPC error: {e}")))?;
            Ok(resp.into_inner().into())
        }

        /// List tasks.
        pub async fn list_tasks(
            &mut self,
            req: params::ListTasksRequest,
        ) -> Result<crate::params::ListTasksResponse, A2aError> {
            let proto_req = lf_a2a_v1::ListTasksRequest {
                tenant: req.tenant.unwrap_or_default(),
                context_id: req.context_id.unwrap_or_default(),
                status: req.status.map(i32::from).unwrap_or(0),
                page_size: req.page_size,
                page_token: req.page_token.unwrap_or_default(),
                history_length: req.history_length,
                status_timestamp_after: None,
                include_artifacts: req.include_artifacts,
            };
            let resp = self
                .client
                .list_tasks(proto_req)
                .await
                .map_err(|e| A2aError::internal(format!("gRPC error: {e}")))?;
            let inner = resp.into_inner();
            Ok(crate::params::ListTasksResponse {
                tasks: inner.tasks.into_iter().map(Into::into).collect(),
                next_page_token: inner.next_page_token,
                page_size: inner.page_size,
                total_size: inner.total_size,
            })
        }

        /// Get the authenticated extended agent card.
        pub async fn get_extended_agent_card(
            &mut self,
            req: params::GetExtendedAgentCardRequest,
        ) -> Result<crate::agent_card::AgentCard, A2aError> {
            let proto_req = lf_a2a_v1::GetExtendedAgentCardRequest {
                tenant: req.tenant.unwrap_or_default(),
            };
            let resp = self
                .client
                .get_extended_agent_card(proto_req)
                .await
                .map_err(|e| A2aError::internal(format!("gRPC error: {e}")))?;
            Ok(resp.into_inner().into())
        }

        /// Create a push notification config.
        pub async fn create_push_config(
            &mut self,
            config: crate::push_notification::TaskPushNotificationConfig,
        ) -> Result<crate::push_notification::TaskPushNotificationConfig, A2aError> {
            let proto_req: lf_a2a_v1::TaskPushNotificationConfig = config.into();
            let resp = self
                .client
                .create_task_push_notification_config(proto_req)
                .await
                .map_err(|e| A2aError::internal(format!("gRPC error: {e}")))?;
            Ok(resp.into_inner().into())
        }

        /// Get a push notification config.
        pub async fn get_push_config(
            &mut self,
            req: params::GetTaskPushNotificationConfigRequest,
        ) -> Result<crate::push_notification::TaskPushNotificationConfig, A2aError> {
            let proto_req = lf_a2a_v1::GetTaskPushNotificationConfigRequest {
                tenant: req.tenant.unwrap_or_default(),
                task_id: req.task_id,
                id: req.config_id,
            };
            let resp = self
                .client
                .get_task_push_notification_config(proto_req)
                .await
                .map_err(|e| A2aError::internal(format!("gRPC error: {e}")))?;
            Ok(resp.into_inner().into())
        }

        /// List push notification configs.
        pub async fn list_push_configs(
            &mut self,
            req: params::ListTaskPushNotificationConfigsRequest,
        ) -> Result<params::ListTaskPushNotificationConfigsResponse, A2aError> {
            let proto_req = lf_a2a_v1::ListTaskPushNotificationConfigsRequest {
                tenant: req.tenant.unwrap_or_default(),
                task_id: req.task_id,
                page_size: req.page_size.unwrap_or(0),
                page_token: req.page_token.unwrap_or_default(),
            };
            let resp = self
                .client
                .list_task_push_notification_configs(proto_req)
                .await
                .map_err(|e| A2aError::internal(format!("gRPC error: {e}")))?;
            let inner = resp.into_inner();
            Ok(params::ListTaskPushNotificationConfigsResponse {
                configs: inner.configs.into_iter().map(Into::into).collect(),
                next_page_token: if inner.next_page_token.is_empty() {
                    None
                } else {
                    Some(inner.next_page_token)
                },
            })
        }

        /// Delete a push notification config.
        pub async fn delete_push_config(
            &mut self,
            req: params::DeleteTaskPushNotificationConfigRequest,
        ) -> Result<(), A2aError> {
            let proto_req = lf_a2a_v1::DeleteTaskPushNotificationConfigRequest {
                tenant: req.tenant.unwrap_or_default(),
                task_id: req.task_id,
                id: req.config_id,
            };
            self.client
                .delete_task_push_notification_config(proto_req)
                .await
                .map_err(|e| A2aError::internal(format!("gRPC error: {e}")))?;
            Ok(())
        }

        /// Subscribe to task updates.
        pub async fn subscribe_to_task(
            &mut self,
            req: params::SubscribeToTaskRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>>, A2aError>
        {
            let proto_req = lf_a2a_v1::SubscribeToTaskRequest {
                tenant: req.tenant.unwrap_or_default(),
                id: req.id,
            };
            let resp = self
                .client
                .subscribe_to_task(proto_req)
                .await
                .map_err(|e| A2aError::internal(format!("gRPC error: {e}")))?;

            let stream = resp.into_inner().map(|item| {
                item.map_err(|e| A2aError::internal(format!("gRPC stream error: {e}")))
                    .and_then(proto_stream_response_to_stream_response)
            });

            Ok(Box::pin(stream))
        }
    }

    fn proto_stream_response_to_stream_response(
        sr: lf_a2a_v1::StreamResponse,
    ) -> Result<StreamResponse, A2aError> {
        match sr.payload {
            Some(lf_a2a_v1::stream_response::Payload::Task(t)) => Ok(StreamResponse {
                task: Some(t.into()),
                message: None,
                status_update: None,
                artifact_update: None,
            }),
            Some(lf_a2a_v1::stream_response::Payload::Message(m)) => Ok(StreamResponse {
                task: None,
                message: Some(m.into()),
                status_update: None,
                artifact_update: None,
            }),
            Some(lf_a2a_v1::stream_response::Payload::StatusUpdate(su)) => Ok(StreamResponse {
                task: None,
                message: None,
                status_update: Some(crate::streaming::TaskStatusUpdateEvent {
                    task_id: su.task_id,
                    context_id: su.context_id,
                    status: su
                        .status
                        .map(Into::into)
                        .unwrap_or(crate::task::TaskStatus {
                            state: crate::task::TaskState::Unspecified,
                            message: None,
                            timestamp: None,
                        }),
                    trace_id: None,
                    sequence: None,
                    metadata: None,
                }),
                artifact_update: None,
            }),
            Some(lf_a2a_v1::stream_response::Payload::ArtifactUpdate(au)) => Ok(StreamResponse {
                task: None,
                message: None,
                status_update: None,
                artifact_update: Some(crate::streaming::TaskArtifactUpdateEvent {
                    task_id: au.task_id,
                    context_id: au.context_id,
                    artifact: au
                        .artifact
                        .map(Into::into)
                        .unwrap_or(crate::types::Artifact {
                            artifact_id: String::new(),
                            name: None,
                            description: None,
                            parts: vec![],
                            metadata: None,
                            extensions: None,
                        }),
                    index: None,
                    append: Some(au.append),
                    last_chunk: Some(au.last_chunk),
                    trace_id: None,
                    sequence: None,
                    metadata: None,
                }),
            }),
            None => Err(A2aError::internal("Empty stream response")),
        }
    }
}

#[cfg(feature = "grpc-client")]
pub use grpc_impl::GrpcTransport;
