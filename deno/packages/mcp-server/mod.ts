/**
 * @module @rullama/mcp-server
 *
 * MCP-compliant tool server framework: `McpServer`, `McpHandler`,
 * `McpToolRegistry`, `MiddlewareChain` (Auth, Logging, RateLimit,
 * ToolFilter, IdentityToolFilter), stdio / unix-socket / TCP transports, and
 * `StreamServerTransport` / `StreamServerListener` for hosts that open their
 * own sockets (FFI, least-privilege apps). Requests on one connection are
 * dispatched concurrently; `notifications/cancelled` aborts
 * `RequestContext.signal`.
 *
 * Extracted from `@rullama/network` in v0.11.0 to mirror Rust's standalone
 * `rullama-mcp-server` crate. Consumers building a pure MCP server no
 * longer need to pull in the A2A peer-table / discovery / routing surface
 * of `@rullama/network`.
 */

export { AgentNetworkError, ErrorCode } from "./error.ts";

export {
  type McpToolAnnotations,
  type McpToolDef,
  type McpToolOptions,
  McpToolRegistry,
  type ToolHandler,
} from "./registry.ts";

export { type McpHandler } from "./handler.ts";

export {
  type ClientInfo,
  type FirstLineDecision,
  type FirstLineHook,
  type LoggingLevel,
  McpServer,
  McpServerSession,
  negotiateProtocolVersion,
  RequestContext,
  type ServeConnectionOptions,
  type ServeOptions,
} from "./server.ts";

// Protocol revisions, shared with the client.
export {
  LATEST_PROTOCOL_VERSION,
  SUPPORTED_PROTOCOL_VERSIONS,
} from "@rullama/mcp-client";

// Middleware
export * from "./middleware/mod.ts";

// Transport
export * from "./transport/mod.ts";
