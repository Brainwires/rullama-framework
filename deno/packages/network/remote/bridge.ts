/**
 * @module remote/bridge
 *
 * RemoteBridge — connects local agents to a cloud relay via WebSocket.
 * Uses the native WebSocket API (not Supabase SDK) for portability.
 *
 * Equivalent to Rust's `rullama-network::remote::bridge`.
 */

import type {
  BackendCommand,
  ProtocolCapability,
  RemoteMessage,
} from "./protocol.ts";
import {
  defaultProtocolHello,
  NegotiatedProtocol as NegotiatedProtocolClass,
} from "./protocol.ts";
import { HeartbeatCollector } from "./heartbeat.ts";
import type { AgentInfoProvider } from "./heartbeat.ts";

// ============================================================================
// Bridge Configuration
// ============================================================================

/** Remote bridge configuration. */
export interface BridgeConfig {
  /** Backend base URL (https://...). */
  backendUrl: string;
  /** API key for authentication. */
  apiKey: string;
  /** Heartbeat/poll interval in seconds. */
  heartbeatIntervalSecs: number;
  /** Reconnect delay on disconnect in seconds. */
  reconnectDelaySecs: number;
  /** Maximum reconnect attempts (0 = unlimited). */
  maxReconnectAttempts: number;
  /** Client version string. */
  version: string;
  /** Agent info provider callback. */
  agentInfoProvider?: AgentInfoProvider;
  /** Hostname override. */
  hostname?: string;
}

/** Create a default BridgeConfig. */
export function defaultBridgeConfig(): BridgeConfig {
  return {
    backendUrl: "https://brainwires.studio",
    apiKey: "",
    heartbeatIntervalSecs: 5,
    reconnectDelaySecs: 5,
    maxReconnectAttempts: 0,
    version: "unknown",
  };
}

// ============================================================================
// Bridge State
// ============================================================================

/** Bridge connection state. */
export type BridgeState =
  | "disconnected"
  | "connecting"
  | "connected"
  | "authenticated"
  | "shutting_down";

/** Connection mode. */
export type ConnectionMode = "websocket" | "polling";

// ============================================================================
// Event Handler Types
// ============================================================================

/** Handler for backend commands received via the bridge. */
export type CommandHandler = (command: BackendCommand) => Promise<void> | void;

/** Handler for bridge state changes. */
export type StateChangeHandler = (state: BridgeState) => void;

// ============================================================================
// Remote Bridge
// ============================================================================

/**
 * Remote control bridge.
 *
 * Maintains communication with the backend using WebSocket (preferred)
 * or HTTP polling (fallback). Uses the native WebSocket API.
 */
export class RemoteBridge {
  readonly config: BridgeConfig;

  private _state: BridgeState = "disconnected";
  private _connectionMode: ConnectionMode = "polling";
  private sessionToken: string | undefined;
  private userId: string | undefined;
  private negotiatedProtocol: NegotiatedProtocolClass;
  private heartbeatCollector: HeartbeatCollector;
  private commandResultQueue: RemoteMessage[] = [];
  private subscriptions = new Set<string>();
  private ws: WebSocket | null = null;
  private shutdownRequested = false;
  private heartbeatTimer: number | undefined;

  private onCommand: CommandHandler | undefined;
  private onStateChange: StateChangeHandler | undefined;

  constructor(config: BridgeConfig) {
    this.config = config;
    this.negotiatedProtocol = NegotiatedProtocolClass.default();
    this.heartbeatCollector = new HeartbeatCollector({
      version: config.version,
      hostname: config.hostname,
      agentInfoProvider: config.agentInfoProvider,
    });
  }

  // --------------------------------------------------------------------------
  // Public getters
  // --------------------------------------------------------------------------

  /** Get current bridge state. */
  get state(): BridgeState {
    return this._state;
  }

  /** Get current connection mode. */
  get connectionMode(): ConnectionMode {
    return this._connectionMode;
  }

  /** Check if bridge is connected and authenticated. */
  get isReady(): boolean {
    return this._state === "authenticated";
  }

  /** Get the user ID (if authenticated). */
  getUserId(): string | undefined {
    return this.userId;
  }

  /** Get the negotiated protocol version. */
  protocolVersion(): string {
    return this.negotiatedProtocol.version;
  }

  /** Check if a capability is enabled. */
  hasCapability(cap: ProtocolCapability): boolean {
    return this.negotiatedProtocol.hasCapability(cap);
  }

  /** Get all enabled capabilities. */
  enabledCapabilities(): ProtocolCapability[] {
    return [...this.negotiatedProtocol.capabilities];
  }

  // --------------------------------------------------------------------------
  // Event registration
  // --------------------------------------------------------------------------

  /** Set handler for backend commands. */
  setCommandHandler(handler: CommandHandler): void {
    this.onCommand = handler;
  }

  /** Set handler for state changes. */
  setStateChangeHandler(handler: StateChangeHandler): void {
    this.onStateChange = handler;
  }

  // --------------------------------------------------------------------------
  // State management
  // --------------------------------------------------------------------------

  private setState(state: BridgeState): void {
    this._state = state;
    this.onStateChange?.(state);
  }

  // --------------------------------------------------------------------------
  // Queue management
  // --------------------------------------------------------------------------

  /** Queue a command result to send with the next heartbeat. */
  queueCommandResult(msg: RemoteMessage): void {
    this.commandResultQueue.push(msg);
  }

  /** Queue a typed command result (success or error). */
  queueResult(
    commandId: string,
    result: { ok: true; value: unknown } | { ok: false; error: string },
  ): void {
    if (result.ok) {
      this.queueCommandResult({
        type: "command_result",
        command_id: commandId,
        success: true,
        result: result.value,
      });
    } else {
      this.queueCommandResult({
        type: "command_result",
        command_id: commandId,
        success: false,
        error: result.error,
      });
    }
  }

  // --------------------------------------------------------------------------
  // Main run loop
  // --------------------------------------------------------------------------

  /**
   * Connect to the backend and run the main communication loop.
   * Reconnects automatically on disconnect until `shutdown()` is called.
   */
  async run(): Promise<void> {
    let reconnectAttempts = 0;
    this.shutdownRequested = false;

    while (!this.shutdownRequested) {
      this.setState("connecting");

      try {
        await this.registerWithBackend();
        reconnectAttempts = 0;
        this.setState("authenticated");
        this._connectionMode = "polling";

        await this.runPollingLoop();
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`[RemoteBridge] Registration/loop error: ${msg}`);
        reconnectAttempts++;

        if (
          this.config.maxReconnectAttempts > 0 &&
          reconnectAttempts >= this.config.maxReconnectAttempts
        ) {
          throw new Error(
            `Max reconnect attempts (${this.config.maxReconnectAttempts}) reached`,
          );
        }
      }

      // Clean up state
      this.setState("disconnected");
      this._connectionMode = "polling";
      this.sessionToken = undefined;
      this.subscriptions.clear();
      this.commandResultQueue = [];

      if (!this.shutdownRequested) {
        console.log(
          `[RemoteBridge] Reconnecting in ${this.config.reconnectDelaySecs}s...`,
        );
        await delay(this.config.reconnectDelaySecs * 1000);
      }
    }
  }

  /** Shutdown the bridge. */
  shutdown(): void {
    this.shutdownRequested = true;
    this.setState("shutting_down");
    if (this.heartbeatTimer !== undefined) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = undefined;
    }
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        // ignore
      }
      this.ws = null;
    }
  }

  // --------------------------------------------------------------------------
  // Registration
  // --------------------------------------------------------------------------

  /** Register with the backend via HTTP POST. */
  private async registerWithBackend(): Promise<void> {
    const url = `${this.config.backendUrl}/api/remote/connect`;
    const protocolHello = defaultProtocolHello();

    const body = {
      hostname: this.config.hostname ?? "unknown",
      os: Deno.build.os,
      version: this.config.version,
      protocol: protocolHello,
    };

    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.config.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Registration failed: ${response.status} - ${text}`);
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(`Authentication failed: ${data.error}`);
    }

    if (!data.session_token || !data.user_id) {
      throw new Error("Missing session_token or user_id in response");
    }

    this.sessionToken = data.session_token;
    this.userId = data.user_id;

    // Handle protocol negotiation
    if (data.protocol) {
      try {
        this.negotiatedProtocol = NegotiatedProtocolClass.fromAccept(
          data.protocol,
        );
      } catch {
        this.negotiatedProtocol = NegotiatedProtocolClass.default();
      }
    } else {
      this.negotiatedProtocol = NegotiatedProtocolClass.default();
    }
  }

  // --------------------------------------------------------------------------
  // Polling loop
  // --------------------------------------------------------------------------

  /** Main polling loop. */
  private async runPollingLoop(): Promise<void> {
    // Initial heartbeat
    await this.sendHeartbeatAndProcessCommands();

    while (!this.shutdownRequested && this._state === "authenticated") {
      await delay(this.config.heartbeatIntervalSecs * 1000);
      if (this.shutdownRequested) break;
      try {
        await this.sendHeartbeatAndProcessCommands();
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`[RemoteBridge] Heartbeat failed: ${msg}`);
        break;
      }
    }
  }

  /** Send heartbeat and process any commands returned. */
  private async sendHeartbeatAndProcessCommands(): Promise<void> {
    const heartbeatData = await this.heartbeatCollector.collect();

    // Drain queued command results
    const commandResults = [...this.commandResultQueue];
    this.commandResultQueue = [];

    const body = {
      session_token: this.sessionToken,
      agents: heartbeatData.agents,
      system_load: heartbeatData.system_load,
      messages: commandResults,
      hostname: heartbeatData.hostname,
      os: heartbeatData.os,
      version: this.config.version,
    };

    const url = `${this.config.backendUrl}/api/remote/heartbeat`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.config.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Heartbeat failed: ${response.status} - ${text}`);
    }

    const responseBody = await response.json();

    // Process any commands from the response
    const commands = responseBody.commands;
    if (Array.isArray(commands)) {
      for (const cmdValue of commands) {
        try {
          await this.handleBackendCommand(cmdValue as BackendCommand);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          console.error(`[RemoteBridge] Error handling command: ${msg}`);
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // Command handling
  // --------------------------------------------------------------------------

  /** Handle a command from the backend. */
  private async handleBackendCommand(cmd: BackendCommand): Promise<void> {
    switch (cmd.type) {
      case "ping":
        this.queueCommandResult({ type: "pong", timestamp: cmd.timestamp });
        break;

      case "request_sync":
        // Trigger immediate heartbeat data collection
        break;

      case "subscribe":
        this.subscriptions.add(cmd.agent_id);
        break;

      case "unsubscribe":
        this.subscriptions.delete(cmd.agent_id);
        break;

      case "disconnect":
        console.log(
          `[RemoteBridge] Backend requested disconnect: ${cmd.reason}`,
        );
        this.shutdown();
        break;

      case "authenticated":
      case "authentication_failed":
        // Unexpected after authentication
        break;

      default:
        // Delegate to external handler
        if (this.onCommand) {
          await this.onCommand(cmd);
        }
        break;
    }
  }
}

// ============================================================================
// Utility
// ============================================================================

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
