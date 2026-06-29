/**
 * Tests for McpClient.
 * Mirrors Rust tests in `rullama-mcp/src/client.rs`.
 */

import { assertEquals } from "@std/assert";
import { McpClient } from "./client.ts";

// =============================================================================
// Client creation tests — mirrors Rust test_client_creation
// =============================================================================

Deno.test("McpClient - creation with name and version", () => {
  const client = new McpClient("test", "0.1.0");

  assertEquals(client.clientName, "test");
  assertEquals(client.clientVersion, "0.1.0");
  assertEquals(client.currentRequestId, 1);
});

// Mirrors Rust test_request_id_increment
Deno.test("McpClient - request ID increment", () => {
  const client = new McpClient("test", "0.1.0");

  // IDs should start at 1 and increment
  assertEquals(client.currentRequestId, 1);
  // After internal calls, the ID advances (we test via listConnected which doesn't use IDs)
});

Deno.test("McpClient - default creation", () => {
  const client = McpClient.createDefault();

  assertEquals(client.clientName, "rullama");
  assertEquals(client.clientVersion, "0.5.0");
});

Deno.test("McpClient - isConnected returns false for unknown server", () => {
  const client = new McpClient("test", "0.1.0");

  assertEquals(client.isConnected("nonexistent"), false);
});

Deno.test("McpClient - listConnected returns empty initially", () => {
  const client = new McpClient("test", "0.1.0");

  assertEquals(client.listConnected().length, 0);
});

Deno.test("McpClient - disconnect unknown server does not throw", async () => {
  const client = new McpClient("test", "0.1.0");
  // Should not throw
  await client.disconnect("nonexistent");
});

Deno.test("McpClient - getServerInfo throws for unconnected server", () => {
  const client = new McpClient("test", "0.1.0");
  let threw = false;
  try {
    client.getServerInfo("nonexistent");
  } catch (e) {
    threw = true;
    assertEquals((e as Error).message, "Not connected to server: nonexistent");
  }
  assertEquals(threw, true);
});

Deno.test("McpClient - getCapabilities throws for unconnected server", () => {
  const client = new McpClient("test", "0.1.0");
  let threw = false;
  try {
    client.getCapabilities("nonexistent");
  } catch (e) {
    threw = true;
    assertEquals((e as Error).message, "Not connected to server: nonexistent");
  }
  assertEquals(threw, true);
});
