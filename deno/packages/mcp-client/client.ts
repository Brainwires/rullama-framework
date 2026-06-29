/**
 * @module client
 *
 * MCP Client — manages connections to MCP servers.
 * Equivalent to Rust's `rullama-mcp/src/client.rs`.
 */

import type { McpServerConfig } from "./config.ts";
import { StdioTransport, Transport } from "./transport.ts";
import type {
  CallToolParams,
  CallToolResult,
  GetPromptParams,
  GetPromptResult,
  InitializeParams,
  InitializeResult,
  JsonRpcNotification,
  JsonRpcRequest,
  ListPromptsResult,
  ListResourcesResult,
  ListToolsResult,
  McpPrompt,
  McpResource,
  McpTool,
  ReadResourceParams,
  ReadResourceResult,
  ServerCapabilities,
  ServerInfo,
} from "./types.ts";
import { createJsonRpcRequest } from "./types.ts";

/**
 * Active connection to an MCP server.
 * Equivalent to Rust `McpConnection`.
 */
interface McpConnection {
  /** Server name from config. */
  serverName: string;
  /** Transport layer. */
  transport: Transport;
  /** Server info from initialization. */
  serverInfo: ServerInfo;
  /** Server capabilities from initialization. */
  capabilities: ServerCapabilities;
}

/**
 * MCP Client — manages connections to MCP servers.
 * Equivalent to Rust `McpClient`.
 *
 * Provides methods to connect to MCP servers, list/call tools,
 * list/read resources, and list/get prompts.
 */
export class McpClient {
  #connections: Map<string, McpConnection> = new Map();
  #requestId = 1;
  #clientName: string;
  #clientVersion: string;

  /**
   * Create a new MCP client with the given name and version.
   * Equivalent to Rust `McpClient::new`.
   */
  constructor(clientName: string, clientVersion: string) {
    this.#clientName = clientName;
    this.#clientVersion = clientVersion;
  }

  /**
   * Create a default MCP client.
   * Equivalent to Rust `McpClient::default`.
   */
  static createDefault(): McpClient {
    return new McpClient("rullama", "0.5.0");
  }

  /**
   * Connect to an MCP server.
   * Equivalent to Rust `McpClient::connect`.
   */
  async connect(config: McpServerConfig): Promise<void> {
    // Spawn server process
    const stdioTransport = await StdioTransport.create(
      config.command,
      config.args,
      config.env,
    );
    const transport = new Transport(stdioTransport);

    // Send initialize request
    const initResult = await this.#initialize(transport);

    // Create connection
    const connection: McpConnection = {
      serverName: config.name,
      transport,
      serverInfo: initResult.serverInfo,
      capabilities: initResult.capabilities,
    };

    // Store connection
    this.#connections.set(config.name, connection);
  }

  /**
   * Disconnect from an MCP server.
   * Equivalent to Rust `McpClient::disconnect`.
   */
  async disconnect(serverName: string): Promise<void> {
    const connection = this.#connections.get(serverName);
    if (connection) {
      this.#connections.delete(serverName);
      await connection.transport.close();
    }
  }

  /**
   * Check if connected to a server.
   * Equivalent to Rust `McpClient::is_connected`.
   */
  isConnected(serverName: string): boolean {
    return this.#connections.has(serverName);
  }

  /**
   * Get list of connected servers.
   * Equivalent to Rust `McpClient::list_connected`.
   */
  listConnected(): string[] {
    return [...this.#connections.keys()];
  }

  /**
   * List available tools from a server.
   * Equivalent to Rust `McpClient::list_tools`.
   */
  async listTools(serverName: string): Promise<McpTool[]> {
    const connection = this.#getConnection(serverName);

    const request = createJsonRpcRequest(
      this.#nextRequestId(),
      "tools/list",
    );

    await connection.transport.sendRequest(request);
    const response = await connection.transport.receiveResponse();

    if (response.error) {
      throw new Error(
        `tools/list failed: ${response.error.message} (code: ${response.error.code})`,
      );
    }

    const result = response.result as ListToolsResult;
    return result.tools;
  }

  /**
   * Call a tool on a server.
   * Equivalent to Rust `McpClient::call_tool`.
   */
  async callTool(
    serverName: string,
    toolName: string,
    args?: Record<string, unknown>,
  ): Promise<CallToolResult> {
    return await this.callToolWithNotifications(serverName, toolName, args);
  }

  /**
   * Call a tool on a server with notification forwarding.
   * If onNotification is provided, any notifications received while waiting
   * for the response will be forwarded through that callback.
   * Equivalent to Rust `McpClient::call_tool_with_notifications`.
   */
  async callToolWithNotifications(
    serverName: string,
    toolName: string,
    args?: Record<string, unknown>,
    onNotification?: (notification: JsonRpcNotification) => void,
  ): Promise<CallToolResult> {
    const connection = this.#getConnection(serverName);

    const params: CallToolParams = {
      name: toolName,
      ...(args ? { arguments: args } : {}),
    };

    const request = createJsonRpcRequest(
      this.#nextRequestId(),
      "tools/call",
      params,
    );

    await connection.transport.sendRequest(request);

    // Wait for response, forwarding any notifications that arrive
    while (true) {
      const message = await connection.transport.receiveMessage();

      if (message.type === "response") {
        const response = message.response;
        if (response.error) {
          throw new Error(
            `tools/call failed: ${response.error.message} (code: ${response.error.code})`,
          );
        }
        return response.result as CallToolResult;
      }

      if (message.type === "notification") {
        if (onNotification) {
          onNotification(message.notification);
        }
        // Continue waiting for the response
      }
    }
  }

  /**
   * List available resources from a server.
   * Equivalent to Rust `McpClient::list_resources`.
   */
  async listResources(serverName: string): Promise<McpResource[]> {
    const connection = this.#getConnection(serverName);

    const request = createJsonRpcRequest(
      this.#nextRequestId(),
      "resources/list",
    );

    await connection.transport.sendRequest(request);
    const response = await connection.transport.receiveResponse();

    if (response.error) {
      throw new Error(
        `resources/list failed: ${response.error.message} (code: ${response.error.code})`,
      );
    }

    const result = response.result as ListResourcesResult;
    return result.resources;
  }

  /**
   * Read a resource from a server.
   * Equivalent to Rust `McpClient::read_resource`.
   */
  async readResource(
    serverName: string,
    uri: string,
  ): Promise<ReadResourceResult> {
    const connection = this.#getConnection(serverName);

    const params: ReadResourceParams = { uri };
    const request = createJsonRpcRequest(
      this.#nextRequestId(),
      "resources/read",
      params,
    );

    await connection.transport.sendRequest(request);
    const response = await connection.transport.receiveResponse();

    if (response.error) {
      throw new Error(
        `resources/read failed: ${response.error.message} (code: ${response.error.code})`,
      );
    }

    return response.result as ReadResourceResult;
  }

  /**
   * List available prompts from a server.
   * Equivalent to Rust `McpClient::list_prompts`.
   */
  async listPrompts(serverName: string): Promise<McpPrompt[]> {
    const connection = this.#getConnection(serverName);

    const request = createJsonRpcRequest(
      this.#nextRequestId(),
      "prompts/list",
    );

    await connection.transport.sendRequest(request);
    const response = await connection.transport.receiveResponse();

    if (response.error) {
      throw new Error(
        `prompts/list failed: ${response.error.message} (code: ${response.error.code})`,
      );
    }

    const result = response.result as ListPromptsResult;
    return result.prompts;
  }

  /**
   * Get a prompt from a server.
   * Equivalent to Rust `McpClient::get_prompt`.
   */
  async getPrompt(
    serverName: string,
    promptName: string,
    args?: Record<string, unknown>,
  ): Promise<GetPromptResult> {
    const connection = this.#getConnection(serverName);

    const params: GetPromptParams = {
      name: promptName,
      ...(args ? { arguments: args } : {}),
    };
    const request = createJsonRpcRequest(
      this.#nextRequestId(),
      "prompts/get",
      params,
    );

    await connection.transport.sendRequest(request);
    const response = await connection.transport.receiveResponse();

    if (response.error) {
      throw new Error(
        `prompts/get failed: ${response.error.message} (code: ${response.error.code})`,
      );
    }

    return response.result as GetPromptResult;
  }

  /**
   * Get server info for a connection.
   * Equivalent to Rust `McpClient::get_server_info`.
   */
  getServerInfo(serverName: string): ServerInfo {
    const connection = this.#getConnection(serverName);
    return { ...connection.serverInfo };
  }

  /**
   * Get server capabilities for a connection.
   * Equivalent to Rust `McpClient::get_capabilities`.
   */
  getCapabilities(serverName: string): ServerCapabilities {
    const connection = this.#getConnection(serverName);
    return { ...connection.capabilities };
  }

  /**
   * Send a cancellation request to the MCP server.
   * Follows the JSON-RPC 2.0 cancellation protocol using `$/cancelRequest`.
   * Equivalent to Rust `McpClient::cancel_request`.
   */
  async cancelRequest(serverName: string, requestId: number): Promise<void> {
    const connection = this.#getConnection(serverName);

    const request: JsonRpcRequest = {
      jsonrpc: "2.0",
      id: null,
      method: "$/cancelRequest",
      params: { id: requestId },
    };

    await connection.transport.sendRequest(request);
  }

  /**
   * Get the next request ID.
   * Equivalent to Rust `McpClient::next_request_id`.
   */
  #nextRequestId(): number {
    return this.#requestId++;
  }

  /**
   * Get a connection by server name, throwing if not connected.
   */
  #getConnection(serverName: string): McpConnection {
    const connection = this.#connections.get(serverName);
    if (!connection) {
      throw new Error(`Not connected to server: ${serverName}`);
    }
    return connection;
  }

  /**
   * Initialize handshake with server.
   * Equivalent to Rust `McpClient::initialize`.
   */
  async #initialize(transport: Transport): Promise<InitializeResult> {
    const params: InitializeParams = {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: {
        name: this.#clientName,
        version: this.#clientVersion,
      },
    };

    const request = createJsonRpcRequest(
      this.#nextRequestId(),
      "initialize",
      params,
    );

    await transport.sendRequest(request);
    const response = await transport.receiveResponse();

    if (response.error) {
      throw new Error(
        `Initialize failed: ${response.error.message} (code: ${response.error.code})`,
      );
    }

    if (!response.result) {
      throw new Error("Missing result in initialize response");
    }

    const result = response.result as InitializeResult;

    // Send initialized notification
    const notification: JsonRpcRequest = {
      jsonrpc: "2.0",
      id: null,
      method: "notifications/initialized",
    };
    await transport.sendRequest(notification);

    return result;
  }

  /** The client name. */
  get clientName(): string {
    return this.#clientName;
  }

  /** The client version. */
  get clientVersion(): string {
    return this.#clientVersion;
  }

  /** The current request ID counter value (for testing). */
  get currentRequestId(): number {
    return this.#requestId;
  }
}
