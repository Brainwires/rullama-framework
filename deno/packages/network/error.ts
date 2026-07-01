/**
 * @module error
 *
 * Re-exports `AgentNetworkError` + `ErrorCode` from `@rullama/mcp-server`.
 *
 * In v0.11.0 the canonical error type moved with the MCP server framework
 * extraction. This module remains as a thin shim so existing relative
 * imports inside `@rullama/network` (`./error.ts`) keep resolving to the
 * same class identity.
 */

export { AgentNetworkError, ErrorCode } from "@rullama/mcp-server";
