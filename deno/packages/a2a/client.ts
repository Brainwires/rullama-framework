/**
 * A2A client -- unified client with JSON-RPC and REST transport selection.
 *
 * Uses fetch() for all HTTP calls. gRPC transport is not included
 * (requires proto compiler toolchain).
 */

import type { AgentCard } from "./agent_card.ts";
import { A2aError } from "./error.ts";
import type { JsonRpcRequest, JsonRpcResponse, RequestId } from "./jsonrpc.ts";
import {
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
import type {
  CancelTaskRequest,
  DeleteTaskPushNotificationConfigRequest,
  GetExtendedAgentCardRequest,
  GetTaskPushNotificationConfigRequest,
  GetTaskRequest,
  ListTaskPushNotificationConfigsRequest,
  ListTaskPushNotificationConfigsResponse,
  ListTasksRequest,
  ListTasksResponse,
  SendMessageRequest,
  SubscribeToTaskRequest,
} from "./params.ts";
import type { TaskPushNotificationConfig } from "./push_notification.ts";
import { parseSseStream } from "./sse.ts";
import type { SendMessageResponse, StreamResponse } from "./streaming.ts";
import type { Task } from "./task.ts";

/** Transport selection. */
export type Transport = "jsonrpc" | "rest";

/** Options for creating an A2aClient. */
export interface A2aClientOptions {
  /** Base URL of the A2A server. */
  baseUrl: string;
  /** Transport mode (default: "jsonrpc"). */
  transport?: Transport;
  /** Bearer token for authentication. */
  bearerToken?: string;
}

/**
 * Unified A2A client supporting JSON-RPC and REST transports.
 */
export class A2aClient {
  private readonly baseUrl: string;
  private readonly transport: Transport;
  private readonly bearerToken?: string;
  private requestCounter = 1;

  /**
   * Create a client for one A2A server.
   *
   * @param options `baseUrl` (trailing slashes are stripped), the transport
   *   to speak (`"jsonrpc"` posts to `baseUrl` itself, `"rest"` uses
   *   `baseUrl/message:send`-style paths; default `"jsonrpc"`) and an
   *   optional bearer token sent as `Authorization: Bearer …`.
   */
  constructor(options: A2aClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.transport = options.transport ?? "jsonrpc";
    this.bearerToken = options.bearerToken;
  }

  /** Create a new client with a bearer token applied. */
  withBearerToken(token: string): A2aClient {
    return new A2aClient({
      baseUrl: this.baseUrl,
      transport: this.transport,
      bearerToken: token,
    });
  }

  // -------------------------------------------------------------------------
  // Discovery
  // -------------------------------------------------------------------------

  /**
   * Discover an agent card from a well-known URL.
   * Fetches `{baseUrl}/.well-known/agent-card.json`.
   */
  static async discover(baseUrl: string): Promise<AgentCard> {
    const url = `${baseUrl.replace(/\/+$/, "")}/.well-known/agent-card.json`;
    const resp = await fetch(url).catch((e) => {
      throw A2aError.internal(`Discovery request failed: ${e}`);
    });

    if (!resp.ok) {
      throw A2aError.internal(
        `Discovery failed with status: ${resp.status}`,
      );
    }

    return (await resp.json()) as AgentCard;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /** Send a message (SendMessage). */
  async sendMessage(req: SendMessageRequest): Promise<SendMessageResponse> {
    if (this.transport === "jsonrpc") {
      return await this.jsonRpcCall(METHOD_MESSAGE_SEND, req);
    }
    return await this.restPost("/message:send", req);
  }

  /**
   * Stream a message (SendStreamingMessage).
   * Returns an async iterable of StreamResponse values.
   */
  async *streamMessage(
    req: SendMessageRequest,
  ): AsyncIterable<StreamResponse> {
    if (this.transport === "jsonrpc") {
      yield* this.jsonRpcStream(METHOD_MESSAGE_STREAM, req);
    } else {
      yield* this.restPostStream("/message:stream", req);
    }
  }

  /** Get a task by ID (GetTask). */
  async getTask(req: GetTaskRequest): Promise<Task> {
    if (this.transport === "jsonrpc") {
      return await this.jsonRpcCall(METHOD_TASKS_GET, req);
    }
    return await this.restGet(`/tasks/${req.id}`);
  }

  /** List tasks (ListTasks). */
  async listTasks(req: ListTasksRequest): Promise<ListTasksResponse> {
    if (this.transport === "jsonrpc") {
      return await this.jsonRpcCall(METHOD_TASKS_LIST, req);
    }
    return await this.restGet("/tasks");
  }

  /** Cancel a task (CancelTask). */
  async cancelTask(req: CancelTaskRequest): Promise<Task> {
    if (this.transport === "jsonrpc") {
      return await this.jsonRpcCall(METHOD_TASKS_CANCEL, req);
    }
    return await this.restPost(`/tasks/${req.id}:cancel`, req);
  }

  /**
   * Subscribe to task updates (SubscribeToTask).
   * Returns an async iterable of StreamResponse values.
   * Uses POST per A2A v1.0.
   */
  async *subscribeToTask(
    req: SubscribeToTaskRequest,
  ): AsyncIterable<StreamResponse> {
    if (this.transport === "jsonrpc") {
      yield* this.jsonRpcStream(METHOD_TASKS_RESUBSCRIBE, req);
    } else {
      yield* this.restPostStream(`/tasks/${req.id}:subscribe`, req);
    }
  }

  /** Set push notification config. */
  async setPushConfig(
    config: TaskPushNotificationConfig,
  ): Promise<TaskPushNotificationConfig> {
    if (this.transport === "jsonrpc") {
      return await this.jsonRpcCall(METHOD_PUSH_CONFIG_SET, config);
    }
    return await this.restPost(
      `/tasks/${config.taskId}/pushNotificationConfigs`,
      config,
    );
  }

  /** Get push notification config. */
  async getPushConfig(
    req: GetTaskPushNotificationConfigRequest,
  ): Promise<TaskPushNotificationConfig> {
    if (this.transport === "jsonrpc") {
      return await this.jsonRpcCall(METHOD_PUSH_CONFIG_GET, req);
    }
    return await this.restGet(
      `/tasks/${req.taskId}/pushNotificationConfigs/${req.configId}`,
    );
  }

  /** Delete push notification config. */
  async deletePushConfig(
    req: DeleteTaskPushNotificationConfigRequest,
  ): Promise<void> {
    if (this.transport === "jsonrpc") {
      await this.jsonRpcCall(METHOD_PUSH_CONFIG_DELETE, req);
      return;
    }
    await this.restDelete(
      `/tasks/${req.taskId}/pushNotificationConfigs/${req.configId}`,
    );
  }

  /** List push notification configs. */
  async listPushConfigs(
    req: ListTaskPushNotificationConfigsRequest,
  ): Promise<ListTaskPushNotificationConfigsResponse> {
    if (this.transport === "jsonrpc") {
      return await this.jsonRpcCall(METHOD_PUSH_CONFIG_LIST, req);
    }
    return await this.restGet(
      `/tasks/${req.taskId}/pushNotificationConfigs`,
    );
  }

  /** Get the authenticated extended agent card. */
  async getAuthenticatedExtendedCard(
    req: GetExtendedAgentCardRequest = {},
  ): Promise<AgentCard> {
    if (this.transport === "jsonrpc") {
      return await this.jsonRpcCall(METHOD_EXTENDED_CARD, req);
    }
    return await this.restGet("/extendedAgentCard");
  }

  // -------------------------------------------------------------------------
  // JSON-RPC transport internals
  // -------------------------------------------------------------------------

  /** Return the next monotonically increasing JSON-RPC request id. */
  private nextId(): RequestId {
    return this.requestCounter++;
  }

  /**
   * Build the request headers: JSON content type plus `Authorization` when a
   * bearer token is configured.
   */
  private authHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.bearerToken) {
      headers["Authorization"] = `Bearer ${this.bearerToken}`;
    }
    return headers;
  }

  /**
   * POST a single JSON-RPC 2.0 request to `baseUrl` and unwrap its `result`.
   * A JSON-RPC `error` object is rethrown as {@link A2aError}; a missing
   * result or a failed fetch surfaces as an internal error.
   */
  private async jsonRpcCall<T>(method: string, params: unknown): Promise<T> {
    const id = this.nextId();
    const request: JsonRpcRequest = {
      jsonrpc: "2.0",
      method,
      params,
      id,
    };

    const resp = await fetch(this.baseUrl, {
      method: "POST",
      headers: this.authHeaders(),
      body: JSON.stringify(request),
    }).catch((e) => {
      throw A2aError.internal(`HTTP request failed: ${e}`);
    });

    const rpcResp = (await resp.json()) as JsonRpcResponse;

    if (rpcResp.error) {
      throw A2aError.fromJSON(
        rpcResp.error as { code: number; message: string; data?: unknown },
      );
    }

    if (rpcResp.result === undefined) {
      throw A2aError.internal("Empty result");
    }

    return rpcResp.result as T;
  }

  /**
   * POST a JSON-RPC request whose reply is a Server-Sent Events stream and
   * yield each parsed {@link StreamResponse} (JSON-RPC envelope framing).
   */
  private async *jsonRpcStream(
    method: string,
    params: unknown,
  ): AsyncIterable<StreamResponse> {
    const id = this.nextId();
    const request: JsonRpcRequest = {
      jsonrpc: "2.0",
      method,
      params,
      id,
    };

    const resp = await fetch(this.baseUrl, {
      method: "POST",
      headers: this.authHeaders(),
      body: JSON.stringify(request),
    }).catch((e) => {
      throw A2aError.internal(`HTTP request failed: ${e}`);
    });

    if (!resp.body) {
      throw A2aError.internal("Response has no body");
    }

    yield* parseSseStream(resp.body, "jsonrpc");
  }

  // -------------------------------------------------------------------------
  // REST transport internals
  // -------------------------------------------------------------------------

  /** Join a REST path (e.g. `/tasks/{id}`) onto the base URL. */
  private restUrl(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  /**
   * POST a JSON body to a REST path and parse the JSON reply; a non-2xx
   * status is thrown as an internal {@link A2aError}.
   */
  private async restPost<T>(path: string, body: unknown): Promise<T> {
    const resp = await fetch(this.restUrl(path), {
      method: "POST",
      headers: this.authHeaders(),
      body: JSON.stringify(body),
    }).catch((e) => {
      throw A2aError.internal(`REST request failed: ${e}`);
    });

    if (!resp.ok) {
      throw A2aError.internal(`REST error: ${resp.status}`);
    }

    return (await resp.json()) as T;
  }

  /** GET a REST path and parse the JSON reply; non-2xx throws. */
  private async restGet<T>(path: string): Promise<T> {
    const resp = await fetch(this.restUrl(path), {
      method: "GET",
      headers: this.authHeaders(),
    }).catch((e) => {
      throw A2aError.internal(`REST request failed: ${e}`);
    });

    if (!resp.ok) {
      throw A2aError.internal(`REST error: ${resp.status}`);
    }

    return (await resp.json()) as T;
  }

  /** DELETE a REST path, discarding the body; non-2xx throws. */
  private async restDelete(path: string): Promise<void> {
    const resp = await fetch(this.restUrl(path), {
      method: "DELETE",
      headers: this.authHeaders(),
    }).catch((e) => {
      throw A2aError.internal(`REST DELETE failed: ${e}`);
    });

    if (!resp.ok) {
      throw A2aError.internal(`REST DELETE error: ${resp.status}`);
    }
  }

  /**
   * POST a JSON body to a REST streaming path and yield each parsed
   * {@link StreamResponse} from the Server-Sent Events reply (REST framing).
   */
  private async *restPostStream(
    path: string,
    body: unknown,
  ): AsyncIterable<StreamResponse> {
    const resp = await fetch(this.restUrl(path), {
      method: "POST",
      headers: this.authHeaders(),
      body: JSON.stringify(body),
    }).catch((e) => {
      throw A2aError.internal(`REST stream request failed: ${e}`);
    });

    if (!resp.body) {
      throw A2aError.internal("Response has no body");
    }

    yield* parseSseStream(resp.body, "rest");
  }
}
