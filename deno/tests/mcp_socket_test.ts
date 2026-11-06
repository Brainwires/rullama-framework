/**
 * Cross-package integration test: `@rullama/mcp-server` over a unix domain
 * socket / TCP, driven by the real `@rullama/mcp-client` (`SocketTransport`),
 * by raw `Deno.connect`, and through the system `nc -U` stdio shim.
 */

import { assert, assertEquals, assertRejects } from "@std/assert";
import { fromFileUrl, join } from "@std/path";
import {
  type CallToolResult,
  McpClient,
  SocketTransport,
} from "@rullama/mcp-client";
import {
  type McpHandler,
  McpServer,
  type McpServerSession,
  type RequestContext,
  TcpServerTransport,
  UnixSocketServerTransport,
} from "@rullama/mcp-server";

const isWindows = Deno.build.os === "windows";

const text = (t: string): CallToolResult => ({
  content: [{ type: "text", text: t }],
});

/** Handler whose `whoami` tool reports the connection identity. */
function tileHandler(events: string[] = []): McpHandler {
  return {
    serverInfo: () => ({ name: "qutty-hub", version: "0.0.0" }),
    capabilities: () => ({ tools: { listChanged: true } }),
    listTools: () => [
      { name: "whoami", description: "identity", inputSchema: {} },
      { name: "wait", description: "wait or abort", inputSchema: {} },
    ],
    callTool: (name: string, args, ctx: RequestContext) => {
      if (name === "whoami") {
        return Promise.resolve(text(JSON.stringify({
          identity: ctx.connection?.identity ?? null,
          transport: ctx.connection?.transport,
          client: ctx.clientInfo?.name,
        })));
      }
      const ms = typeof args.ms === "number" ? args.ms : 60_000;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => resolve(text("waited")), ms);
        ctx.signal.addEventListener("abort", () => {
          clearTimeout(timer);
          events.push(`aborted ${ctx.connection?.id}`);
          reject(ctx.signal.reason);
        }, { once: true });
      });
    },
    onShutdown: (conn) => {
      events.push(`shutdown ${conn?.id}`);
      return Promise.resolve();
    },
  };
}

async function tempSocketPath(): Promise<{ dir: string; path: string }> {
  const dir = await Deno.makeTempDir({ prefix: "mcp" });
  return { dir, path: join(dir, "t.sock") };
}

/** Read newline-delimited JSON objects from a byte stream. */
function jsonLines(readable: ReadableStream<Uint8Array>) {
  const reader = readable
    .pipeThrough(new TextDecoderStream())
    .getReader();
  let buf = "";
  return {
    async next(): Promise<Record<string, unknown> | null> {
      while (true) {
        const nl = buf.indexOf("\n");
        if (nl !== -1) {
          const line = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          if (line.trim()) return JSON.parse(line);
          continue;
        }
        const { value, done } = await reader.read();
        if (done) return null;
        buf += value;
      }
    },
    cancel: () => reader.cancel().catch(() => {}),
  };
}

const INIT = {
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {
    protocolVersion: "2025-06-18",
    capabilities: {},
    clientInfo: { name: "raw", version: "1" },
  },
};
const INITIALIZED = { jsonrpc: "2.0", method: "notifications/initialized" };
const LIST = { jsonrpc: "2.0", id: 2, method: "tools/list" };
const WHOAMI = {
  jsonrpc: "2.0",
  id: 3,
  method: "tools/call",
  params: { name: "whoami", arguments: {} },
};
const handshake = [INIT, INITIALIZED, LIST, WHOAMI]
  .map((m) => JSON.stringify(m) + "\n").join("");

Deno.test({
  name:
    "unix socket: McpClient initialize → tools/list → tools/call with identity in ctx",
  ignore: isWindows,
  async fn() {
    const { dir, path } = await tempSocketPath();
    const events: string[] = [];
    const listener = await UnixSocketServerTransport.listen(path, {
      identity: { tile: "A" },
    });
    assertEquals((await Deno.lstat(path)).mode! & 0o777, 0o600);
    const sessions: McpServerSession[] = [];
    const served = new McpServer(tileHandler(events)).serve(listener, {
      onSession: (s) => sessions.push(s),
    });

    const client = new McpClient("itest", "1.0.0");
    await client.connectTransport(
      "tile",
      await SocketTransport.connectUnix(path),
    );
    assertEquals(client.getServerInfo("tile").name, "qutty-hub");
    assertEquals(client.getProtocolVersion("tile"), "2025-11-25");
    assertEquals((await client.listTools("tile")).map((t) => t.name), [
      "whoami",
      "wait",
    ]);

    // A 60 s tool in flight does not block other calls on the same socket.
    const ac = new AbortController();
    const waiting = client.callTool("tile", "wait", {}, { signal: ac.signal });
    const who = await client.callTool("tile", "whoami");
    const body = JSON.parse((who.content[0] as { text: string }).text);
    assertEquals(body, {
      identity: { tile: "A" },
      transport: "unix",
      client: "itest",
    });
    await client.ping("tile");

    // Server-initiated notification reaches the client.
    const got: string[] = [];
    const off = client.onNotification((_s, n) => got.push(n.method));
    assertEquals(sessions.length, 1);
    assert(await sessions[0].notifyToolsListChanged());
    for (let i = 0; i < 100 && got.length === 0; i++) {
      await new Promise((r) => setTimeout(r, 2));
    }
    assertEquals(got, ["notifications/tools/list_changed"]);
    off();

    // Cancelling the long call aborts the tool's signal on the server.
    ac.abort(new Error("stop"));
    await assertRejects(() => waiting, Error, "stop");
    for (let i = 0; i < 100 && events.length === 0; i++) {
      await new Promise((r) => setTimeout(r, 2));
    }
    assertEquals(events, [`aborted ${sessions[0].connection.id}`]);

    // Client disconnect ends the session and runs onShutdown.
    await client.disconnect("tile");
    await sessions[0].done;
    assertEquals(events.at(-1), `shutdown ${sessions[0].connection.id}`);

    listener.close();
    await served;
    await listener.finished;
    await assertRejects(() => Deno.lstat(path), Deno.errors.NotFound);
    await Deno.remove(dir, { recursive: true });
  },
});

Deno.test({
  name: "unix socket: raw Deno.connect with merged and split writes",
  ignore: isWindows,
  async fn() {
    const { dir, path } = await tempSocketPath();
    const listener = await UnixSocketServerTransport.listen(path, {
      identity: "raw-tile",
    });
    const served = new McpServer(tileHandler()).serve(listener);

    const enc = new TextEncoder();
    // Every message in one write…
    const a = await Deno.connect({ transport: "unix", path });
    const linesA = jsonLines(a.readable);
    await a.write(enc.encode(handshake));
    const ra = [await linesA.next(), await linesA.next(), await linesA.next()];
    assertEquals(ra.map((r) => r!.id), [1, 2, 3]);
    // deno-lint-ignore no-explicit-any
    const resultA = ra as any[];
    assertEquals(resultA[0].result.protocolVersion, "2025-06-18");
    assertEquals(resultA[1].result.tools.length, 2);
    assertEquals(
      JSON.parse(resultA[2].result.content[0].text).identity,
      "raw-tile",
    );

    // …and the same bytes dribbled out a few at a time.
    const b = await Deno.connect({ transport: "unix", path });
    const linesB = jsonLines(b.readable);
    const bytes = enc.encode(handshake);
    for (let i = 0; i < bytes.length; i += 7) {
      await b.write(bytes.subarray(i, i + 7));
      if (i % 70 === 0) await new Promise((r) => setTimeout(r, 1));
    }
    const rb = [await linesB.next(), await linesB.next(), await linesB.next()];
    assertEquals(rb.map((r) => r!.id), [1, 2, 3]);

    a.close();
    b.close();
    await linesA.cancel();
    await linesB.cancel();
    listener.close();
    await served;
    await listener.finished;
    await Deno.remove(dir, { recursive: true });
  },
});

async function hasNcUnix(): Promise<boolean> {
  if (isWindows) return false;
  try {
    const { stdout, stderr } = await new Deno.Command("nc", {
      args: ["-h"],
      stdout: "piped",
      stderr: "piped",
    }).output();
    const help = new TextDecoder().decode(stdout) +
      new TextDecoder().decode(stderr);
    return /-U/.test(help);
  } catch {
    return false;
  }
}

Deno.test({
  name: "unix socket: piped through the system `nc -U` stdio shim",
  ignore: !(await hasNcUnix()),
  async fn() {
    const { dir, path } = await tempSocketPath();
    const listener = await UnixSocketServerTransport.listen(path, {
      identity: "nc-tile",
    });
    const served = new McpServer(tileHandler()).serve(listener);

    const nc = new Deno.Command("nc", {
      args: ["-U", path],
      stdin: "piped",
      stdout: "piped",
      stderr: "null",
    }).spawn();
    const writer = nc.stdin.getWriter();
    const lines = jsonLines(nc.stdout);
    await writer.write(new TextEncoder().encode(handshake));
    const r = [await lines.next(), await lines.next(), await lines.next()];
    assertEquals(r.map((x) => x!.id), [1, 2, 3]);
    // deno-lint-ignore no-explicit-any
    const who = JSON.parse((r[2] as any).result.content[0].text);
    assertEquals(who.identity, "nc-tile");

    await writer.close().catch(() => {});
    try {
      nc.kill("SIGTERM");
    } catch { /* already exited */ }
    await nc.status;
    await lines.cancel();
    listener.close();
    await served;
    await listener.finished;
    await Deno.remove(dir, { recursive: true });
  },
});

Deno.test({
  name:
    "unix socket: stale socket file is replaced, live one and non-socket refused",
  ignore: isWindows,
  async fn() {
    const { dir, path } = await tempSocketPath();
    // A crashed process leaves its socket file behind (a clean close unlinks).
    const crashed = await new Deno.Command(Deno.execPath(), {
      args: [
        "eval",
        `Deno.listen({ transport: "unix", path: ${
          JSON.stringify(path)
        } }); Deno.exit(0);`,
      ],
      stdout: "null",
      stderr: "inherit",
    }).output();
    assert(crashed.success);
    assert((await Deno.lstat(path)).isSocket, "stale socket file left behind");

    const first = await UnixSocketServerTransport.listen(path);
    await assertRejects(
      () => UnixSocketServerTransport.listen(path),
      Deno.errors.AddrInUse,
    );
    first.close();
    await first.finished;
    await assertRejects(() => Deno.lstat(path), Deno.errors.NotFound);

    const file = join(dir, "plain");
    await Deno.writeTextFile(file, "keep me");
    await assertRejects(
      () => UnixSocketServerTransport.listen(file),
      Deno.errors.AlreadyExists,
    );
    assertEquals(await Deno.readTextFile(file), "keep me");
    await Deno.remove(dir, { recursive: true });
  },
});

Deno.test({
  name:
    "unix socket: one listener per tile, handlerFactory sees each identity; closing a listener closes its sessions",
  ignore: isWindows,
  async fn() {
    const { dir } = await tempSocketPath();
    const server = new McpServer(tileHandler());
    const seen: unknown[] = [];
    const listeners = await Promise.all(
      ["A", "B"].map((tile) =>
        UnixSocketServerTransport.listen(join(dir, `${tile}.sock`), {
          identity: { tile },
        })
      ),
    );
    const served = listeners.map((l) =>
      server.serve(l, {
        handlerFactory: (info) => {
          seen.push(info.identity);
          return tileHandler();
        },
      })
    );
    const client = new McpClient("itest", "1");
    for (const tile of ["A", "B"]) {
      await client.connectTransport(
        tile,
        await SocketTransport.connectUnix(join(dir, `${tile}.sock`)),
      );
      const who = await client.callTool(tile, "whoami");
      assertEquals(
        JSON.parse((who.content[0] as { text: string }).text).identity,
        { tile },
      );
    }
    assertEquals(seen, [{ tile: "A" }, { tile: "B" }]);
    assertEquals(server.sessions.size, 2);

    // Closing tile A's listener tears down its session; B keeps working.
    listeners[0].close();
    await served[0];
    assertEquals(server.sessions.size, 1);
    await assertRejects(() => client.ping("A"), Error, "connection closed");
    await client.ping("B");

    await client.disconnect("A");
    await client.disconnect("B");
    listeners[1].close();
    await served[1];
    await Promise.all(listeners.map((l) => l.finished));
    await Deno.remove(dir, { recursive: true });
  },
});

Deno.test("tcp loopback: first-line pre-auth admits with identity, rejects a bad token", async () => {
  const listener = TcpServerTransport.listen({
    preAuth: (line) =>
      line === "tok-123"
        ? { ok: true, identity: { agent: "codex" } }
        : { ok: false, reason: "bad token" },
  });
  assertEquals(listener.hostname, "127.0.0.1");
  const served = new McpServer(tileHandler()).serve(listener);

  const client = new McpClient("itest", "1");
  await client.connectTransport(
    "hub",
    await SocketTransport.connectTcp("127.0.0.1", listener.port, {
      preAuthLine: "tok-123",
    }),
  );
  const who = await client.callTool("hub", "whoami");
  assertEquals(JSON.parse((who.content[0] as { text: string }).text), {
    identity: { agent: "codex" },
    transport: "tcp",
    client: "itest",
  });
  await client.disconnect("hub");

  // Wrong token: one Unauthorized line, then the server hangs up.
  const raw = await Deno.connect({
    hostname: "127.0.0.1",
    port: listener.port,
  });
  const lines = jsonLines(raw.readable);
  await raw.write(new TextEncoder().encode("nope\n" + handshake));
  const reply = await lines.next();
  // deno-lint-ignore no-explicit-any
  assertEquals((reply as any).error.code, -32003);
  assertEquals(await lines.next(), null);
  try {
    raw.close();
  } catch { /* closed by the read side */ }

  // A bad token through the real client fails connectTransport cleanly.
  await assertRejects(async () =>
    client.connectTransport(
      "bad",
      await SocketTransport.connectTcp("127.0.0.1", listener.port, {
        preAuthLine: "wrong",
      }),
    )
  );

  listener.close();
  await served;
  await listener.finished;
});

Deno.test({
  name: "unix socket: abrupt client disconnect aborts its in-flight tool call",
  ignore: isWindows,
  async fn() {
    const { dir, path } = await tempSocketPath();
    const events: string[] = [];
    const listener = await UnixSocketServerTransport.listen(path);
    const sessions: McpServerSession[] = [];
    const served = new McpServer(tileHandler(events)).serve(listener, {
      onSession: (s) => sessions.push(s),
    });
    const conn = await Deno.connect({ transport: "unix", path });
    const lines = jsonLines(conn.readable);
    const enc = new TextEncoder();
    await conn.write(enc.encode(JSON.stringify(INIT) + "\n"));
    await lines.next();
    await conn.write(enc.encode(
      JSON.stringify({
        jsonrpc: "2.0",
        id: 9,
        method: "tools/call",
        params: { name: "wait", arguments: {} },
      }) + "\n",
    ));
    for (let i = 0; i < 100 && sessions[0]?.inFlight !== 1; i++) {
      await new Promise((r) => setTimeout(r, 2));
    }
    assertEquals(sessions[0].inFlight, 1);
    await lines.cancel();
    try {
      conn.close();
    } catch { /* closed by cancel */ }
    await sessions[0].done;
    assertEquals(events, [
      `aborted ${sessions[0].connection.id}`,
      `shutdown ${sessions[0].connection.id}`,
    ]);
    listener.close();
    await served;
    await listener.finished;
    await Deno.remove(dir, { recursive: true });
  },
});

Deno.test("stdio: McpClient.connect spawns an McpServer subprocess; concurrent calls and disconnect", async () => {
  const root = new URL("../", import.meta.url);
  const serverMod = new URL("packages/mcp-server/mod.ts", root).href;
  const script = `
    import { McpServer } from ${JSON.stringify(serverMod)};
    await new McpServer({
      serverInfo: () => ({ name: "stdio-fixture", version: "1" }),
      capabilities: () => ({ tools: {} }),
      listTools: () => [],
      callTool: (name, args, ctx) => new Promise((resolve) => setTimeout(
        () => resolve({ content: [{ type: "text", text: name + ":" + ctx.connection.transport }] }),
        Number(args.ms ?? 0),
      )),
    }).run();
  `;
  const client = new McpClient("itest", "1");
  await client.connect({
    name: "child",
    command: Deno.execPath(),
    args: [
      "eval",
      "--quiet",
      "--config",
      fromFileUrl(new URL("deno.json", root)),
      script,
    ],
  });
  assertEquals(client.getServerInfo("child").name, "stdio-fixture");
  const slow = client.callTool("child", "slow", { ms: 150 });
  const fast = await client.callTool("child", "fast");
  assertEquals((fast.content[0] as { text: string }).text, "fast:stdio");
  assertEquals(
    ((await slow).content[0] as { text: string }).text,
    "slow:stdio",
  );
  await client.disconnect("child");
  assertEquals(client.isConnected("child"), false);
});
