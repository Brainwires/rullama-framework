//! gRPC service implementation — bridges tonic to A2aHandler.

#[cfg(feature = "grpc-server")]
mod grpc_impl {
    use std::pin::Pin;
    use std::sync::Arc;

    use futures::Stream;
    use tokio_stream::StreamExt;
    use tonic::{Request, Response, Status};

    use crate::error::A2aError;
    use crate::params;
    use crate::proto::lf_a2a_v1;
    use crate::proto::lf_a2a_v1::a2a_service_server::A2aService;
    use crate::server::handler::A2aHandler;

    /// Bridge that connects tonic's generated service trait to our `A2aHandler`.
    pub struct GrpcBridge<H: A2aHandler> {
        handler: Arc<H>,
    }

    impl<H: A2aHandler> GrpcBridge<H> {
        /// Create a new gRPC bridge wrapping the given handler.
        pub fn new(handler: Arc<H>) -> Self {
            Self { handler }
        }
    }

    fn a2a_err_to_status(e: A2aError) -> Status {
        use crate::error::*;
        let code = match e.code {
            TASK_NOT_FOUND => tonic::Code::NotFound,
            INVALID_PARAMS | INVALID_REQUEST => tonic::Code::InvalidArgument,
            METHOD_NOT_FOUND | UNSUPPORTED_OPERATION => tonic::Code::Unimplemented,
            TASK_NOT_CANCELABLE => tonic::Code::FailedPrecondition,
            PUSH_NOT_SUPPORTED => tonic::Code::Unimplemented,
            CONTENT_TYPE_NOT_SUPPORTED | JSON_PARSE_ERROR => tonic::Code::InvalidArgument,
            INVALID_AGENT_RESPONSE => tonic::Code::Internal,
            EXTENDED_CARD_NOT_CONFIGURED => tonic::Code::FailedPrecondition,
            _ => tonic::Code::Internal,
        };
        Status::new(code, e.message)
    }

    fn stream_response_to_proto(sr: crate::streaming::StreamResponse) -> lf_a2a_v1::StreamResponse {
        let payload = if let Some(t) = sr.task {
            Some(lf_a2a_v1::stream_response::Payload::Task(t.into()))
        } else if let Some(m) = sr.message {
            Some(lf_a2a_v1::stream_response::Payload::Message(m.into()))
        } else if let Some(su) = sr.status_update {
            Some(lf_a2a_v1::stream_response::Payload::StatusUpdate(
                lf_a2a_v1::TaskStatusUpdateEvent {
                    task_id: su.task_id,
                    context_id: su.context_id,
                    status: Some(su.status.into()),
                    metadata: None,
                },
            ))
        } else if let Some(au) = sr.artifact_update {
            Some(lf_a2a_v1::stream_response::Payload::ArtifactUpdate(
                lf_a2a_v1::TaskArtifactUpdateEvent {
                    task_id: au.task_id,
                    context_id: au.context_id,
                    artifact: Some(au.artifact.into()),
                    append: au.append.unwrap_or(false),
                    last_chunk: au.last_chunk.unwrap_or(false),
                    metadata: None,
                },
            ))
        } else {
            None
        };
        lf_a2a_v1::StreamResponse { payload }
    }

    #[tonic::async_trait]
    impl<H: A2aHandler> A2aService for GrpcBridge<H> {
        async fn send_message(
            &self,
            request: Request<lf_a2a_v1::SendMessageRequest>,
        ) -> Result<Response<lf_a2a_v1::SendMessageResponse>, Status> {
            let proto_req = request.into_inner();
            let msg = proto_req
                .message
                .ok_or_else(|| Status::invalid_argument("missing message"))?;
            let req = params::SendMessageRequest {
                tenant: if proto_req.tenant.is_empty() {
                    None
                } else {
                    Some(proto_req.tenant)
                },
                message: msg.into(),
                configuration: None,
                metadata: None,
            };

            let result = self
                .handler
                .on_send_message(req)
                .await
                .map_err(a2a_err_to_status)?;

            let payload = if let Some(t) = result.task {
                Some(lf_a2a_v1::send_message_response::Payload::Task(t.into()))
            } else {
                result
                    .message
                    .map(|m| lf_a2a_v1::send_message_response::Payload::Message(m.into()))
            };

            Ok(Response::new(lf_a2a_v1::SendMessageResponse { payload }))
        }

        type SendStreamingMessageStream =
            Pin<Box<dyn Stream<Item = Result<lf_a2a_v1::StreamResponse, Status>> + Send>>;

        async fn send_streaming_message(
            &self,
            request: Request<lf_a2a_v1::SendMessageRequest>,
        ) -> Result<Response<Self::SendStreamingMessageStream>, Status> {
            let proto_req = request.into_inner();
            let msg = proto_req
                .message
                .ok_or_else(|| Status::invalid_argument("missing message"))?;
            let req = params::SendMessageRequest {
                tenant: if proto_req.tenant.is_empty() {
                    None
                } else {
                    Some(proto_req.tenant)
                },
                message: msg.into(),
                configuration: None,
                metadata: None,
            };

            let stream = self
                .handler
                .on_send_streaming_message(req)
                .await
                .map_err(a2a_err_to_status)?;

            let mapped = stream.map(|item| match item {
                Ok(event) => Ok(stream_response_to_proto(event)),
                Err(e) => Err(a2a_err_to_status(e)),
            });

            Ok(Response::new(Box::pin(mapped)))
        }

        async fn get_task(
            &self,
            request: Request<lf_a2a_v1::GetTaskRequest>,
        ) -> Result<Response<lf_a2a_v1::Task>, Status> {
            let proto_req = request.into_inner();
            let req = params::GetTaskRequest {
                tenant: if proto_req.tenant.is_empty() {
                    None
                } else {
                    Some(proto_req.tenant)
                },
                id: proto_req.id,
                history_length: proto_req.history_length,
            };
            let task = self
                .handler
                .on_get_task(req)
                .await
                .map_err(a2a_err_to_status)?;
            Ok(Response::new(task.into()))
        }

        async fn list_tasks(
            &self,
            request: Request<lf_a2a_v1::ListTasksRequest>,
        ) -> Result<Response<lf_a2a_v1::ListTasksResponse>, Status> {
            let proto_req = request.into_inner();
            let req = params::ListTasksRequest {
                tenant: if proto_req.tenant.is_empty() {
                    None
                } else {
                    Some(proto_req.tenant)
                },
                context_id: if proto_req.context_id.is_empty() {
                    None
                } else {
                    Some(proto_req.context_id)
                },
                status: if proto_req.status == 0 {
                    None
                } else {
                    Some(proto_req.status.into())
                },
                page_size: proto_req.page_size,
                page_token: if proto_req.page_token.is_empty() {
                    None
                } else {
                    Some(proto_req.page_token)
                },
                history_length: proto_req.history_length,
                status_timestamp_after: None,
                include_artifacts: proto_req.include_artifacts,
            };
            let result = self
                .handler
                .on_list_tasks(req)
                .await
                .map_err(a2a_err_to_status)?;
            Ok(Response::new(lf_a2a_v1::ListTasksResponse {
                tasks: result.tasks.into_iter().map(Into::into).collect(),
                next_page_token: result.next_page_token,
                page_size: result.page_size,
                total_size: result.total_size,
            }))
        }

        async fn cancel_task(
            &self,
            request: Request<lf_a2a_v1::CancelTaskRequest>,
        ) -> Result<Response<lf_a2a_v1::Task>, Status> {
            let proto_req = request.into_inner();
            let req = params::CancelTaskRequest {
                tenant: if proto_req.tenant.is_empty() {
                    None
                } else {
                    Some(proto_req.tenant)
                },
                id: proto_req.id,
                metadata: None,
            };
            let task = self
                .handler
                .on_cancel_task(req)
                .await
                .map_err(a2a_err_to_status)?;
            Ok(Response::new(task.into()))
        }

        type SubscribeToTaskStream =
            Pin<Box<dyn Stream<Item = Result<lf_a2a_v1::StreamResponse, Status>> + Send>>;

        async fn subscribe_to_task(
            &self,
            request: Request<lf_a2a_v1::SubscribeToTaskRequest>,
        ) -> Result<Response<Self::SubscribeToTaskStream>, Status> {
            let proto_req = request.into_inner();
            let req = params::SubscribeToTaskRequest {
                tenant: if proto_req.tenant.is_empty() {
                    None
                } else {
                    Some(proto_req.tenant)
                },
                id: proto_req.id,
            };
            let stream = self
                .handler
                .on_subscribe_to_task(req)
                .await
                .map_err(a2a_err_to_status)?;

            let mapped = stream.map(|item| match item {
                Ok(event) => Ok(stream_response_to_proto(event)),
                Err(e) => Err(a2a_err_to_status(e)),
            });

            Ok(Response::new(Box::pin(mapped)))
        }

        async fn create_task_push_notification_config(
            &self,
            request: Request<lf_a2a_v1::TaskPushNotificationConfig>,
        ) -> Result<Response<lf_a2a_v1::TaskPushNotificationConfig>, Status> {
            let config: crate::push_notification::TaskPushNotificationConfig =
                request.into_inner().into();
            let result = self
                .handler
                .on_create_push_config(config)
                .await
                .map_err(a2a_err_to_status)?;
            Ok(Response::new(result.into()))
        }

        async fn get_task_push_notification_config(
            &self,
            request: Request<lf_a2a_v1::GetTaskPushNotificationConfigRequest>,
        ) -> Result<Response<lf_a2a_v1::TaskPushNotificationConfig>, Status> {
            let proto_req = request.into_inner();
            let req = params::GetTaskPushNotificationConfigRequest {
                tenant: if proto_req.tenant.is_empty() {
                    None
                } else {
                    Some(proto_req.tenant)
                },
                task_id: proto_req.task_id,
                config_id: proto_req.id,
            };
            let result = self
                .handler
                .on_get_push_config(req)
                .await
                .map_err(a2a_err_to_status)?;
            Ok(Response::new(result.into()))
        }

        async fn list_task_push_notification_configs(
            &self,
            request: Request<lf_a2a_v1::ListTaskPushNotificationConfigsRequest>,
        ) -> Result<Response<lf_a2a_v1::ListTaskPushNotificationConfigsResponse>, Status> {
            let proto_req = request.into_inner();
            let req = params::ListTaskPushNotificationConfigsRequest {
                tenant: if proto_req.tenant.is_empty() {
                    None
                } else {
                    Some(proto_req.tenant)
                },
                task_id: proto_req.task_id,
                page_size: if proto_req.page_size == 0 {
                    None
                } else {
                    Some(proto_req.page_size)
                },
                page_token: if proto_req.page_token.is_empty() {
                    None
                } else {
                    Some(proto_req.page_token)
                },
            };
            let result = self
                .handler
                .on_list_push_configs(req)
                .await
                .map_err(a2a_err_to_status)?;
            Ok(Response::new(
                lf_a2a_v1::ListTaskPushNotificationConfigsResponse {
                    configs: result.configs.into_iter().map(Into::into).collect(),
                    next_page_token: result.next_page_token.unwrap_or_default(),
                },
            ))
        }

        async fn get_extended_agent_card(
            &self,
            request: Request<lf_a2a_v1::GetExtendedAgentCardRequest>,
        ) -> Result<Response<lf_a2a_v1::AgentCard>, Status> {
            let proto_req = request.into_inner();
            let req = params::GetExtendedAgentCardRequest {
                tenant: if proto_req.tenant.is_empty() {
                    None
                } else {
                    Some(proto_req.tenant)
                },
            };
            let card = self
                .handler
                .on_get_extended_agent_card(req)
                .await
                .map_err(a2a_err_to_status)?;
            Ok(Response::new(card.into()))
        }

        async fn delete_task_push_notification_config(
            &self,
            request: Request<lf_a2a_v1::DeleteTaskPushNotificationConfigRequest>,
        ) -> Result<Response<()>, Status> {
            let proto_req = request.into_inner();
            let req = params::DeleteTaskPushNotificationConfigRequest {
                tenant: if proto_req.tenant.is_empty() {
                    None
                } else {
                    Some(proto_req.tenant)
                },
                task_id: proto_req.task_id,
                config_id: proto_req.id,
            };
            self.handler
                .on_delete_push_config(req)
                .await
                .map_err(a2a_err_to_status)?;
            Ok(Response::new(()))
        }
    }
}

#[cfg(feature = "grpc-server")]
pub use grpc_impl::GrpcBridge;
