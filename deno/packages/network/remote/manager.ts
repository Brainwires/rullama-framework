/**
 * @module remote/manager
 *
 * RemoteBridgeManager — lifecycle management for the remote bridge.
 * Handles starting, stopping, and status monitoring.
 *
 * Equivalent to Rust's `rullama-network::remote::manager`.
 */

import { type BridgeConfig, RemoteBridge } from "./bridge.ts";
import type { AgentInfoProvider } from "./heartbeat.ts";

// ============================================================================
// Config Provider
// ============================================================================

/** Remote bridge configuration (as returned by a provider). */
export interface RemoteBridgeConfig {
  backendUrl: string;
  apiKey: string;
  heartbeatIntervalSecs: number;
  reconnectDelaySecs: number;
  maxReconnectAttempts: number;
}

/** Provides configuration for the remote bridge. */
export interface BridgeConfigProvider {
  /** Get remote bridge configuration, or undefined if not enabled. */
  getRemoteConfig(): RemoteBridgeConfig | undefined;
  /** Get the API key, or undefined if not set. */
  getApiKey(): string | undefined;
}

// ============================================================================
// Bridge Status
// ============================================================================

/** Bridge status for display. */
export type RemoteBridgeStatus =
  | { kind: "disconnected" }
  | { kind: "connecting" }
  | { kind: "connected" }
  | { kind: "authenticated" }
  | { kind: "error"; message: string };

/** Display a bridge status. */
export function displayBridgeStatus(status: RemoteBridgeStatus): string {
  switch (status.kind) {
    case "disconnected":
      return "Disconnected";
    case "connecting":
      return "Connecting";
    case "connected":
      return "Connected";
    case "authenticated":
      return "Authenticated";
    case "error":
      return `Error: ${status.message}`;
  }
}

// ============================================================================
// Remote Bridge Manager
// ============================================================================

/**
 * Remote Bridge Manager.
 *
 * Provides a high-level interface for managing the remote control bridge.
 * Uses a `BridgeConfigProvider` for configuration and an `AgentInfoProvider`
 * for agent discovery.
 */
export class RemoteBridgeManager {
  private bridge: RemoteBridge | null = null;
  private runPromise: Promise<void> | null = null;
  private running = false;

  private readonly configProvider: BridgeConfigProvider;
  private readonly agentInfoProvider: AgentInfoProvider;
  private readonly version: string;
  private readonly hostname: string;

  constructor(options: {
    configProvider: BridgeConfigProvider;
    agentInfoProvider?: AgentInfoProvider;
    version: string;
    hostname?: string;
  }) {
    this.configProvider = options.configProvider;
    this.agentInfoProvider = options.agentInfoProvider ?? (() => []);
    this.version = options.version;
    this.hostname = options.hostname ?? "unknown";
  }

  /** Check if remote control is enabled. */
  isEnabled(): boolean {
    return this.configProvider.getRemoteConfig() !== undefined;
  }

  /** Build a BridgeConfig from the provider's settings. */
  private buildBridgeConfig(): BridgeConfig | undefined {
    const remoteConfig = this.configProvider.getRemoteConfig();
    if (!remoteConfig) return undefined;

    // Get API key: prefer the one from config, fall back to key store
    let apiKey = remoteConfig.apiKey;
    if (!apiKey) {
      apiKey = this.configProvider.getApiKey() ?? "";
    }
    if (!apiKey) {
      console.warn(
        "[RemoteBridgeManager] Remote control enabled but no API key available",
      );
      return undefined;
    }

    return {
      backendUrl: remoteConfig.backendUrl,
      apiKey,
      heartbeatIntervalSecs: remoteConfig.heartbeatIntervalSecs,
      reconnectDelaySecs: remoteConfig.reconnectDelaySecs,
      maxReconnectAttempts: remoteConfig.maxReconnectAttempts,
      version: this.version,
      hostname: this.hostname,
      agentInfoProvider: this.agentInfoProvider,
    };
  }

  /** Start the remote bridge with an explicit config. */
  // deno-lint-ignore require-await
  async startWithConfig(config: BridgeConfig): Promise<boolean> {
    if (this.running) return false;
    if (!config.apiKey) throw new Error("No API key configured");

    this.bridge = new RemoteBridge(config);
    this.running = true;

    // Run in background
    this.runPromise = this.bridge.run().catch((err) => {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[RemoteBridgeManager] Bridge error: ${msg}`);
    }).finally(() => {
      this.running = false;
    });

    return true;
  }

  /** Start the remote bridge using configuration from the provider. */
  // deno-lint-ignore require-await
  async startFromConfig(): Promise<boolean> {
    const config = this.buildBridgeConfig();
    if (!config) return false;
    return this.startWithConfig(config);
  }

  /** Stop the remote bridge gracefully. */
  async stop(): Promise<void> {
    if (!this.running || !this.bridge) return;

    this.bridge.shutdown();

    // Wait briefly for cleanup
    if (this.runPromise) {
      const timeout = new Promise<void>((resolve) => setTimeout(resolve, 1000));
      await Promise.race([this.runPromise, timeout]);
    }

    this.bridge = null;
    this.running = false;
  }

  /** Check if the bridge is running. */
  isRunning(): boolean {
    return this.running;
  }

  /** Get bridge status for display. */
  status(): RemoteBridgeStatus {
    if (!this.running || !this.bridge) {
      return { kind: "disconnected" };
    }

    switch (this.bridge.state) {
      case "disconnected":
        return { kind: "disconnected" };
      case "connecting":
        return { kind: "connecting" };
      case "connected":
        return { kind: "connected" };
      case "authenticated":
        return { kind: "authenticated" };
      case "shutting_down":
        return { kind: "disconnected" };
    }
  }

  /** Get the underlying bridge instance (for advanced use). */
  getBridge(): RemoteBridge | null {
    return this.bridge;
  }
}
