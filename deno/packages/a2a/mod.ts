/**
 * Agent-to-Agent (A2A) protocol client and types for rullama — a TypeScript
 * port of the Rust `rullama-a2a` crate. Provides {@link A2aClient}, a
 * fetch-based client that discovers agent cards and sends, streams, lists and
 * cancels tasks over either the JSON-RPC or the REST binding, plus the full
 * A2A v1.0 type system (messages, parts, tasks, agent cards, security schemes,
 * push-notification configs, streaming events), the JSON-RPC method-name
 * constants, {@link A2aError} with the spec error codes, and
 * {@link parseSseStream}. This package is client-only: {@link A2aHandler} is
 * the interface an agent server would implement, but no server transport or
 * router ships here.
 *
 * @module
 */

// Core types
export type { Artifact, Message, Part, Role } from "./types.ts";

export { createAgentMessage, createUserMessage } from "./types.ts";

// Task types
export type { Task, TaskState, TaskStatus } from "./task.ts";

// Agent card types
export type {
  AgentCapabilities,
  AgentCard,
  AgentCardSignature,
  AgentExtension,
  AgentInterface,
  AgentProvider,
  AgentSkill,
  ApiKeySecurityScheme,
  AuthorizationCodeOAuthFlow,
  ClientCredentialsOAuthFlow,
  DeviceCodeOAuthFlow,
  HttpAuthSecurityScheme,
  ImplicitOAuthFlow,
  MutualTlsSecurityScheme,
  OAuth2SecurityScheme,
  OAuthFlows,
  OpenIdConnectSecurityScheme,
  PasswordOAuthFlow,
  SecurityRequirement,
  SecurityScheme,
} from "./agent_card.ts";

// JSON-RPC types and constants
export type { JsonRpcRequest, JsonRpcResponse, RequestId } from "./jsonrpc.ts";

export {
  createJsonRpcError,
  createJsonRpcSuccess,
  METHOD_EXTENDED_CARD,
  METHOD_MESSAGE_SEND,
  METHOD_MESSAGE_STREAM,
  METHOD_PUSH_CONFIG_DELETE,
  METHOD_PUSH_CONFIG_GET,
  METHOD_PUSH_CONFIG_LIST,
  METHOD_PUSH_CONFIG_SET,
  METHOD_TASKS_CANCEL,
  METHOD_TASKS_GET,
  METHOD_TASKS_LIST,
  METHOD_TASKS_RESUBSCRIBE,
} from "./jsonrpc.ts";

// Error types and codes
export {
  A2aError,
  CONTENT_TYPE_NOT_SUPPORTED,
  EXTENDED_CARD_NOT_CONFIGURED,
  EXTENSION_SUPPORT_REQUIRED,
  INTERNAL_ERROR,
  INVALID_AGENT_RESPONSE,
  INVALID_PARAMS,
  INVALID_REQUEST,
  JSON_PARSE_ERROR,
  METHOD_NOT_FOUND,
  PUSH_NOT_SUPPORTED,
  TASK_NOT_CANCELABLE,
  TASK_NOT_FOUND,
  UNSUPPORTED_OPERATION,
  VERSION_NOT_SUPPORTED,
} from "./error.ts";

// Parameter types
export type {
  CancelTaskRequest,
  DeleteTaskPushNotificationConfigRequest,
  GetExtendedAgentCardRequest,
  GetExtendedAgentCardResponse,
  GetTaskPushNotificationConfigRequest,
  GetTaskRequest,
  ListTaskPushNotificationConfigsRequest,
  ListTaskPushNotificationConfigsResponse,
  ListTasksRequest,
  ListTasksResponse,
  SendMessageConfiguration,
  SendMessageRequest,
  SubscribeToTaskRequest,
} from "./params.ts";

// Streaming types
export type {
  SendMessageResponse,
  StreamResponse,
  TaskArtifactUpdateEvent,
  TaskStatusUpdateEvent,
} from "./streaming.ts";

export {
  isArtifactUpdate,
  isMessageResponse,
  isStatusUpdate,
  isTaskResponse,
} from "./streaming.ts";

// Push notification types
export type {
  AuthenticationInfo,
  TaskPushNotificationConfig,
} from "./push_notification.ts";

// SSE parser
export { parseSseStream } from "./sse.ts";

// Client
export { A2aClient } from "./client.ts";
export type { A2aClientOptions, Transport } from "./client.ts";

// Handler interface
export type { A2aHandler } from "./handler.ts";
