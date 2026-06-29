/**
 * @module error
 *
 * Error types for the agent network layer.
 * Equivalent to Rust's `AgentNetworkError`.
 */

import type { JsonRpcError } from "@rullama/mcp-client";

/** Error codes for agent network errors. */
export const ErrorCode = {
  ParseError: -32700,
  MethodNotFound: -32601,
  InvalidParams: -32602,
  InternalError: -32603,
  TransportError: -32000,
  ToolNotFound: -32001,
  RateLimited: -32002,
  Unauthorized: -32003,
} as const;

/**
 * Agent network error with JSON-RPC error conversion.
 * Equivalent to Rust `AgentNetworkError`.
 */
export class AgentNetworkError extends Error {
  readonly code: number;

  constructor(
    readonly kind:
      | "ParseError"
      | "MethodNotFound"
      | "InvalidParams"
      | "Internal"
      | "Transport"
      | "ToolNotFound"
      | "RateLimited"
      | "Unauthorized",
    message: string,
  ) {
    super(message);
    this.name = "AgentNetworkError";
    switch (kind) {
      case "ParseError":
        this.code = ErrorCode.ParseError;
        break;
      case "MethodNotFound":
        this.code = ErrorCode.MethodNotFound;
        break;
      case "InvalidParams":
        this.code = ErrorCode.InvalidParams;
        break;
      case "Internal":
        this.code = ErrorCode.InternalError;
        break;
      case "Transport":
        this.code = ErrorCode.TransportError;
        break;
      case "ToolNotFound":
        this.code = ErrorCode.ToolNotFound;
        break;
      case "RateLimited":
        this.code = ErrorCode.RateLimited;
        break;
      case "Unauthorized":
        this.code = ErrorCode.Unauthorized;
        break;
    }
  }

  /** Convert to a JSON-RPC error object. */
  toJsonRpcError(): JsonRpcError {
    return {
      code: this.code,
      message: this.message,
    };
  }

  static parseError(msg: string): AgentNetworkError {
    return new AgentNetworkError("ParseError", msg);
  }

  static methodNotFound(method: string): AgentNetworkError {
    return new AgentNetworkError(
      "MethodNotFound",
      `Method not found: ${method}`,
    );
  }

  static invalidParams(msg: string): AgentNetworkError {
    return new AgentNetworkError("InvalidParams", msg);
  }

  static internal(msg: string): AgentNetworkError {
    return new AgentNetworkError("Internal", msg);
  }

  static transport(msg: string): AgentNetworkError {
    return new AgentNetworkError("Transport", `Transport error: ${msg}`);
  }

  static toolNotFound(name: string): AgentNetworkError {
    return new AgentNetworkError("ToolNotFound", `Tool not found: ${name}`);
  }

  static rateLimited(): AgentNetworkError {
    return new AgentNetworkError("RateLimited", "Rate limited");
  }

  static unauthorized(): AgentNetworkError {
    return new AgentNetworkError("Unauthorized", "Unauthorized");
  }
}
