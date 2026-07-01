/**
 * Tests for MCP config types.
 * Mirrors Rust tests in `rullama-mcp/src/config.rs`.
 */

import { assertEquals, assertExists } from "@std/assert";
import { McpConfigManager, type McpServerConfig } from "./config.ts";

// =============================================================================
// McpServerConfig serialization — mirrors Rust test_server_config_serialization
// =============================================================================

Deno.test("McpServerConfig - serialization roundtrip", () => {
  const config: McpServerConfig = {
    name: "test-server",
    command: "npx",
    args: ["-y", "test-mcp-server"],
  };

  const json = JSON.stringify(config);
  const deserialized: McpServerConfig = JSON.parse(json);

  assertEquals(deserialized.name, "test-server");
  assertEquals(deserialized.command, "npx");
  assertEquals(deserialized.args.length, 2);
});

// Mirrors Rust test_server_config_with_env
Deno.test("McpServerConfig - with env", () => {
  const config: McpServerConfig = {
    name: "test",
    command: "node",
    args: ["server.js"],
    env: { API_KEY: "test-key" },
  };

  assertExists(config.env);
  assertEquals(config.env!.API_KEY, "test-key");
});

Deno.test("McpServerConfig - env is optional (undefined)", () => {
  const config: McpServerConfig = {
    name: "test",
    command: "node",
    args: [],
  };

  assertEquals(config.env, undefined);
});

Deno.test("McpServerConfig - serialization omits undefined env", () => {
  const config: McpServerConfig = {
    name: "test",
    command: "node",
    args: [],
  };

  const json = JSON.stringify(config);
  const parsed = JSON.parse(json);
  assertEquals("env" in parsed, false);
});

// =============================================================================
// McpConfigManager tests — mirrors Rust test_config_manager_creation etc.
// =============================================================================

Deno.test("McpConfigManager - create returns empty servers", () => {
  const manager = McpConfigManager.create();
  assertEquals(manager.getServers().length, 0);
});

Deno.test("McpConfigManager - getServer returns undefined for nonexistent", () => {
  const manager = McpConfigManager.create();
  assertEquals(manager.getServer("nonexistent"), undefined);
});
