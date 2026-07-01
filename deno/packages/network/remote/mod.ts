/**
 * @module remote
 *
 * Remote bridge subsystem for connecting local agents to a cloud relay.
 * Provides WebSocket-based communication with priority command queuing,
 * heartbeat telemetry, and protocol negotiation.
 *
 * Equivalent to Rust's `rullama-network::remote` module.
 */

// Protocol types
export {
  allSupportedCapabilities,
  defaultProtocolAccept,
  defaultProtocolHello,
  defaultRetryPolicy,
  MIN_PROTOCOL_VERSION,
  NegotiatedProtocol,
  PRIORITY_ORDER,
  PROTOCOL_VERSION,
  SUPPORTED_VERSIONS,
} from "./protocol.ts";

export type {
  AgentEventType,
  BackendCommand,
  BackendCommand_AttachmentChunk,
  BackendCommand_AttachmentComplete,
  BackendCommand_AttachmentUpload,
  BackendCommand_Authenticated,
  BackendCommand_AuthenticationFailed,
  BackendCommand_CancelOperation,
  BackendCommand_Disconnect,
  BackendCommand_Ping,
  BackendCommand_RequestSync,
  BackendCommand_SendInput,
  BackendCommand_SlashCommand,
  BackendCommand_SpawnAgent,
  BackendCommand_Subscribe,
  BackendCommand_Unsubscribe,
  CommandPriority,
  CompressionAlgorithm,
  PrioritizedCommand,
  ProtocolAccept,
  ProtocolCapability,
  ProtocolHello,
  RemoteAgentInfo,
  RemoteMessage,
  RemoteMessage_AgentEvent,
  RemoteMessage_AgentStream,
  RemoteMessage_AttachmentReceived,
  RemoteMessage_CommandResult,
  RemoteMessage_Heartbeat,
  RemoteMessage_Pong,
  RemoteMessage_Register,
  RetryPolicy,
  StreamChunkType,
} from "./protocol.ts";

// Command queue
export { CommandQueue, QueueEntry, QueueError } from "./command_queue.ts";
export type { QueueStats } from "./command_queue.ts";

// Heartbeat & telemetry
export {
  assessConnectionQuality,
  HeartbeatCollector,
  ProtocolMetrics,
} from "./heartbeat.ts";
export type {
  AgentEvent,
  AgentInfoProvider,
  ConnectionQuality,
  HeartbeatData,
  MetricsSnapshot,
} from "./heartbeat.ts";

// Bridge
export { defaultBridgeConfig, RemoteBridge } from "./bridge.ts";
export type {
  BridgeConfig,
  BridgeState,
  CommandHandler,
  ConnectionMode,
  StateChangeHandler,
} from "./bridge.ts";

// Manager
export { displayBridgeStatus, RemoteBridgeManager } from "./manager.ts";
export type {
  BridgeConfigProvider,
  RemoteBridgeConfig,
  RemoteBridgeStatus,
} from "./manager.ts";
