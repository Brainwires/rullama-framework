/**
 * Cross-package integration test: `@rullama/mcp-server` embedded in a host
 * that brings its own sockets (the least-privilege case — no `--allow-write`,
 * no `Deno.listen`). An in-memory duplex stands in for an FFI socket pair: the
 * server end reads through an async generator and writes with partial
 * `write(2)`-style counts. Served via `StreamServerListener` +
 * `McpServer.serve`, driven by the real `McpClient`.
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  type CallToolResult,
  McpClient,
  SocketTransport,
} from "@rullama/mcp-client";
import {
  IdentityToolFilterMiddleware,
  type McpHandler,
  McpServer,
  type StreamConnection,
  StreamServerListener,
  StreamServerTransport,
} from "@rullama/mcp-server";

const enc = new TextEncoder();
const dec = new TextDecoder();

/** A one-way byte pipe: `write` on one end, `chunks()` / `readable` on the other. */
class Pipe {
  #chunks: Uint8Array[] = [];
  #wake: (() => void) | null = null;
  #ended = false;

  write(bytes: Uint8Array): number {
    if (this.#ended) throw new Error("EPIPE");
    this.#chunks.push(bytes.slice());
    this.#wake?.();
    return bytes.length;
  }

  end(): void {
    this.#ended = true;
    this.#wake?.();
  }

  /** Read side as an async generator, like an FFI `read(2)` loop. */
  async *chunks(): AsyncGenerator<Uint8Array> {
    while (true) {
      const next = this.#chunks.shift();
      if (next) {
        yield next;
        continue;
      }
      if (this.#ended) return;
      await new Promise<void>((r) => (this.#wake = r));
      this.#wake = null;
    }
  }
}

/**
 * An in-memory socket pair. `server` is what a host's accept loop yields;
 * `client` is a minimal `Deno.Conn` stand-in for `SocketTransport`.
 */
function socketPair(info?: StreamConnection["info"]) {
  const toServer = new Pipe();
  const toClient = new Pipe();
  let serverClosed = false;
  const server: StreamConnection = {
    readable: toServer.chunks(),
    // At most 7 bytes per call: exercises partial-write continuation.
    write: (bytes) => Promise.resolve(toClient.write(bytes.subarray(0, 7))),
    close: () => {
      serverClosed = true;
      toClient.end();
    },
    info,
  };
  const client = {
    readable: ReadableStream.from(toClient.chunks()),
    write: (bytes: Uint8Array) => Promise.resolve(toServer.write(bytes)),
    close: () => toServer.end(),
  } as unknown as Deno.Conn;
  return {
    server,
    client,
    serverClosed: () => serverClosed,
    /** Raw client helpers for non-MCP lines. */
    sendLine: (line: string) => toServer.write(enc.encode(line + "\n")),
    async readAll(): Promise<string> {
      let s = "";
      for await (const c of toClient.chunks()) s += dec.decode(c);
      return s;
    },
  };
}

/** A host-side accept loop the test feeds connections into. */
function acceptLoop() {
  const pending: StreamConnection[] = [];
  let wake: (() => void) | null = null;
  let stopped = false;
  return {
    connect(conn: StreamConnection) {
      pending.push(conn);
      wake?.();
    },
    stop() {
      stopped = true;
      wake?.();
    },
    async *accept(): AsyncGenerator<StreamConnection> {
      while (true) {
        const next = pending.shift();
        if (next) {
          yield next;
          continue;
        }
        if (stopped) return;
        await new Promise<void>((r) => (wake = r));
        wake = null;
      }
    },
  };
}

const text = (t: string): CallToolResult => ({
  content: [{ type: "text", text: t }],
});

function hubHandler(): McpHandler {
  return {
    serverInfo: () => ({ name: "qutty-hub", version: "0.0.0" }),
    capabilities: () => ({ tools: { listChanged: true } }),
    listTools: () => [
      { name: "whoami", description: "identity", inputSchema: {} },
      { name: "close_tile", description: "admin only", inputSchema: {} },
      {
        name: "echo",
        description: "echo a long text",
        inputSchema: { type: "object" },
      },
    ],
    callTool: (name, args, ctx) => {
      if (name === "whoami") {
        return Promise.resolve(text(JSON.stringify(ctx.connection?.identity)));
      }
      if (name === "echo") return Promise.resolve(text(String(args.text)));
      return Promise.resolve(text(`${name} done`));
    },
  };
}

Deno.test("embedded hub: FFI-style sockets served through StreamServerListener", async () => {
  const loop = acceptLoop();
  const hooks: string[] = [];
  const listener = new StreamServerListener(loop.accept(), {
    transport: "unix-ffi",
    preAuth: (line) => {
      const m = /^tile=(\w+)$/.exec(line ?? "");
      return m
        ? { ok: true, identity: { tile: m[1], admin: m[1] === "root" } }
        : { ok: false, reason: "expected tile=<id>" };
    },
  });
  const server = new McpServer(hubHandler()).withMiddleware(
    new IdentityToolFilterMiddleware((tool, identity) =>
      tool !== "close_tile" || (identity as { admin?: boolean })?.admin === true
    ),
  );
  const serving = server.serve(listener, {
    // Qutty's one-shot hook protocol shares the socket with MCP.
    firstLine: (line, info) => {
      if (line.startsWith('{"qutty"')) {
        const { event } = JSON.parse(line);
        hooks.push(`${(info.identity as { tile: string }).tile}:${event}`);
        return { action: "close", reply: '{"ok":true}' };
      }
      return { action: "mcp" };
    },
  });

  // Tile 1: a real McpClient over the in-memory socket.
  const tile1 = socketPair({ path: "/run/qutty/tile-1.sock" });
  loop.connect(tile1.server);
  const t1 = new SocketTransport(tile1.client);
  await t1.sendLine("tile=1");
  const client = new McpClient("claude-code", "2.1.0");
  await client.connectTransport("tile1", t1);
  assertEquals(client.getProtocolVersion("tile1"), "2025-11-25");
  assertEquals(client.getServerInfo("tile1").name, "qutty-hub");
  assertEquals((await client.listTools("tile1")).map((t) => t.name), [
    "whoami",
    "echo",
  ]);
  const who = await client.callTool("tile1", "whoami", {});
  assertEquals(
    JSON.parse((who.content[0] as { text: string }).text),
    { tile: "1", admin: false },
  );
  const long = "ü".repeat(500); // multi-byte, many 7-byte writes
  const echoed = await client.callTool("tile1", "echo", { text: long });
  assertEquals((echoed.content[0] as { text: string }).text, long);
  await assertRejects(
    () => client.callTool("tile1", "close_tile", {}),
    Error,
    "not allowed",
  );

  // Root tile sees the admin tool.
  const root = socketPair();
  loop.connect(root.server);
  const tr = new SocketTransport(root.client);
  await tr.sendLine("tile=root");
  await client.connectTransport("root", tr);
  assertEquals((await client.listTools("root")).length, 3);

  // A hook line on a fresh connection: answered and closed, no MCP.
  const hook = socketPair();
  loop.connect(hook.server);
  hook.sendLine("tile=2");
  hook.sendLine('{"qutty":"hook","event":"Stop"}');
  assertEquals((await hook.readAll()).trim(), '{"ok":true}');
  assert(hook.serverClosed());
  assertEquals(hooks, ["2:Stop"]);

  // A connection that fails pre-auth gets one Unauthorized line.
  const intruder = socketPair();
  loop.connect(intruder.server);
  intruder.sendLine("let me in");
  const denied = JSON.parse(await intruder.readAll());
  assertEquals(denied.error.code, -32003);

  assertEquals(server.sessions.size, 2);
  await client.disconnect("tile1").catch(() => {});
  await client.disconnect("root").catch(() => {});
  loop.stop();
  await serving;
  assertEquals(server.sessions.size, 0);
});

Deno.test("embedded hub: server/discover probe, then initialize, over a StreamServerTransport", async () => {
  const pair = socketPair();
  const server = new McpServer(hubHandler());
  const session = server.serveConnection(
    StreamServerTransport.from(pair.server),
    { connection: { id: "probe", transport: "unix-ffi" } },
  );
  const raw = new SocketTransport(pair.client);
  // Claude Code 2.1.x opens with the stateless-era probe.
  await raw.sendRequest({
    jsonrpc: "2.0",
    id: 0,
    method: "server/discover",
    params: {
      _meta: { "io.modelcontextprotocol/protocolVersion": "2026-07-28" },
    },
  });
  const probe = await raw.receiveResponse();
  assertEquals(probe.error?.code, -32601);
  await raw.sendRequest({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2026-07-28",
      capabilities: {},
      clientInfo: { name: "claude-code", version: "2.1.0" },
    },
  });
  const init = await raw.receiveResponse();
  assertEquals(
    (init.result as { protocolVersion: string }).protocolVersion,
    "2025-11-25",
  );
  await raw.close();
  await session.done;
  assert(pair.serverClosed());
});
