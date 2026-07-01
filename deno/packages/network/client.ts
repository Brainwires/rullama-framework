/**
 * @module client
 *
 * MCP relay client that communicates with a subprocess over stdio.
 * Equivalent to Rust's `AgentNetworkClient`.
 */

/**
 * Errors from the relay client.
 * Equivalent to Rust `AgentNetworkClientError`.
 */
export class AgentNetworkClientError extends Error {
  constructor(
    readonly kind:
      | "SpawnFailed"
      | "ProcessExited"
      | "Protocol"
      | "JsonRpc"
      | "Timeout"
      | "NotInitialized"
      | "Io",
    message: string,
    readonly code?: number,
  ) {
    super(message);
    this.name = "AgentNetworkClientError";
  }
}

/** Configuration for spawning an agent via the client. */
export interface AgentConfig {
  /** Maximum number of iterations. */
  maxIterations?: number;
  /** Whether to enable validation. */
  enableValidation?: boolean;
  /** Build system type (e.g. "typescript", "cargo"). */
  buildType?: string;
}

/**
 * MCP relay client that communicates with a subprocess over stdio.
 * Equivalent to Rust `AgentNetworkClient`.
 */
export class AgentNetworkClient {
  private process: Deno.ChildProcess;
  private writer: WritableStreamDefaultWriter<Uint8Array>;
  private reader: ReadableStreamDefaultReader<string>;
  private buffer = "";
  private requestId = 1;
  private _initialized = false;
  private encoder = new TextEncoder();
  private readerDone = false;

  private constructor(process: Deno.ChildProcess) {
    this.process = process;

    this.writer = process.stdin.getWriter();

    const decoder = new TextDecoderStream();
    process.stdout.pipeTo(decoder.writable).catch(() => {
      this.readerDone = true;
    });
    this.reader = decoder.readable.getReader();
  }

  /** Connect to a relay process with custom arguments. */
  // deno-lint-ignore require-await
  static async connect(
    binaryPath: string,
    args: string[] = ["chat", "--mcp-server"],
  ): Promise<AgentNetworkClient> {
    const command = new Deno.Command(binaryPath, {
      args,
      stdin: "piped",
      stdout: "piped",
      stderr: "null",
    });
    const process = command.spawn();
    return new AgentNetworkClient(process);
  }

  private nextId(): number {
    return this.requestId++;
  }

  /** Read a single line from the process stdout. */
  private async readLine(): Promise<string> {
    while (true) {
      const newlineIndex = this.buffer.indexOf("\n");
      if (newlineIndex !== -1) {
        const line = this.buffer.slice(0, newlineIndex);
        this.buffer = this.buffer.slice(newlineIndex + 1);
        return line;
      }

      if (this.readerDone) {
        throw new AgentNetworkClientError(
          "ProcessExited",
          "Relay process exited unexpectedly",
        );
      }

      const { value, done } = await this.reader.read();
      if (done) {
        this.readerDone = true;
        throw new AgentNetworkClientError(
          "ProcessExited",
          "Relay process exited unexpectedly",
        );
      }
      this.buffer += value;
    }
  }

  /** Send a JSON-RPC request and read the response. */
  async sendRequest(
    method: string,
    params?: Record<string, unknown>,
  ): Promise<unknown> {
    const id = this.nextId();
    const request = {
      jsonrpc: "2.0",
      id,
      method,
      ...(params !== undefined ? { params } : {}),
    };

    const json = JSON.stringify(request) + "\n";
    await this.writer.write(this.encoder.encode(json));

    const line = await this.readLine();
    const response = JSON.parse(line.trim());

    if (response.error) {
      throw new AgentNetworkClientError(
        "JsonRpc",
        `JSON-RPC error ${response.error.code}: ${response.error.message}`,
        response.error.code,
      );
    }

    return response.result ?? null;
  }

  /** Perform the MCP initialize handshake. */
  async initialize(): Promise<unknown> {
    const result = await this.sendRequest("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: {
        name: "rullama-relay-client",
        version: "0.5.0",
      },
    });

    // Send initialized notification
    const notif = JSON.stringify({
      jsonrpc: "2.0",
      method: "notifications/initialized",
    }) + "\n";
    await this.writer.write(this.encoder.encode(notif));

    this._initialized = true;
    return result;
  }

  /** Call a tool on the relay server by name with arguments. */
  // deno-lint-ignore require-await
  async callTool(
    name: string,
    args: Record<string, unknown>,
  ): Promise<unknown> {
    if (!this._initialized) {
      throw new AgentNetworkClientError(
        "NotInitialized",
        "Not initialized - call initialize() first",
      );
    }
    return this.sendRequest("tools/call", { name, arguments: args });
  }

  /** List all tools available on the relay server. */
  // deno-lint-ignore require-await
  async listTools(): Promise<unknown> {
    if (!this._initialized) {
      throw new AgentNetworkClientError(
        "NotInitialized",
        "Not initialized - call initialize() first",
      );
    }
    return this.sendRequest("tools/list");
  }

  /** Spawn a new agent with the given description and config. */
  async spawnAgent(
    description: string,
    workingDir: string,
    config?: AgentConfig,
  ): Promise<string> {
    const args: Record<string, unknown> = {
      description,
      working_directory: workingDir,
    };
    if (config?.maxIterations !== undefined) {
      args.max_iterations = config.maxIterations;
    }
    if (config?.enableValidation !== undefined) {
      args.enable_validation = config.enableValidation;
    }
    if (config?.buildType !== undefined) {
      args.build_type = config.buildType;
    }

    const result = (await this.callTool("agent_spawn", args)) as Record<
      string,
      unknown
    >;
    return extractAgentId(result);
  }

  /** Wait for an agent to complete. */
  // deno-lint-ignore require-await
  async awaitAgent(
    agentId: string,
    timeoutSecs?: number,
  ): Promise<unknown> {
    const args: Record<string, unknown> = { agent_id: agentId };
    if (timeoutSecs !== undefined) {
      args.timeout_secs = timeoutSecs;
    }
    return this.callTool("agent_await", args);
  }

  /** List all agents. */
  // deno-lint-ignore require-await
  async listAgents(): Promise<unknown> {
    return this.callTool("agent_list", {});
  }

  /** Stop a running agent. */
  async stopAgent(agentId: string): Promise<void> {
    await this.callTool("agent_stop", { agent_id: agentId });
  }

  /** Check whether the client has completed initialization. */
  get isInitialized(): boolean {
    return this._initialized;
  }

  /** Shut down the relay client and terminate the child process. */
  async shutdown(): Promise<void> {
    try {
      await this.writer.close();
    } catch {
      // Already closed
    }
    try {
      this.process.kill();
    } catch {
      // Already exited
    }
  }
}

function extractAgentId(result: Record<string, unknown>): string {
  // Try content array (CallToolResult format)
  const content = result.content;
  if (Array.isArray(content)) {
    for (const item of content) {
      const text = (item as Record<string, unknown>).text;
      if (typeof text === "string") {
        try {
          const parsed = JSON.parse(text);
          if (parsed.agent_id) return String(parsed.agent_id);
        } catch {
          // Not JSON
        }
      }
    }
  }

  // Direct field access
  if (typeof result.agent_id === "string") {
    return result.agent_id;
  }

  throw new AgentNetworkClientError(
    "Protocol",
    "Could not extract agent_id from spawn result",
  );
}
