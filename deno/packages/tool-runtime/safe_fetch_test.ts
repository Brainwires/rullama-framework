import {
  assert,
  assertEquals,
  assertRejects,
  assertStringIncludes,
} from "@std/assert";
import {
  checkUrl,
  isPrivateAddress,
  readCappedText,
  safeFetch,
} from "./safe_fetch.ts";

Deno.test("isPrivateAddress: v4 and v6 reserved ranges", () => {
  for (
    const ip of [
      "127.0.0.1",
      "10.1.2.3",
      "172.16.0.1",
      "172.31.255.255",
      "192.168.1.1",
      "169.254.169.254",
      "100.64.0.1",
      "0.0.0.0",
      "224.0.0.1",
      "::1",
      "::",
      "fe80::1",
      "fc00::1",
      "fd12::1",
      "ff02::1",
      "::ffff:127.0.0.1",
      "::ffff:10.0.0.1",
      "not-an-ip",
    ]
  ) {
    assert(isPrivateAddress(ip), `${ip} should be private`);
  }
  for (
    const ip of [
      "8.8.8.8",
      "93.184.216.34",
      "172.32.0.1",
      "100.128.0.1",
      "2606:4700::1111",
      "::ffff:8.8.8.8",
    ]
  ) {
    assert(!isPrivateAddress(ip), `${ip} should be public`);
  }
});

Deno.test("checkUrl refuses non-http schemes, loopback names, private literals and private DNS answers", async () => {
  const resolve = (host: string) =>
    Promise.resolve(
      host === "internal.test"
        ? ["10.0.0.5"]
        : host === "public.test"
        ? ["93.184.216.34"]
        : [],
    );
  assertStringIncludes(
    (await checkUrl(new URL("file:///etc/passwd")))!,
    "scheme",
  );
  assertStringIncludes(
    (await checkUrl(new URL("ftp://example.com/")))!,
    "scheme",
  );
  assertStringIncludes(
    (await checkUrl(new URL("http://localhost:11434/")))!,
    "loopback",
  );
  assertStringIncludes(
    (await checkUrl(new URL("http://api.localhost/")))!,
    "loopback",
  );
  assertStringIncludes(
    (await checkUrl(new URL("http://169.254.169.254/latest/meta-data/")))!,
    "private",
  );
  assertStringIncludes(
    (await checkUrl(new URL("http://[::1]:8080/")))!,
    "private",
  );
  assertStringIncludes(
    (await checkUrl(new URL("http://internal.test/"), { resolve }))!,
    "private",
  );
  assertStringIncludes(
    (await checkUrl(new URL("http://nowhere.test/"), { resolve }))!,
    "does not resolve",
  );
  assertEquals(
    await checkUrl(new URL("https://public.test/x"), { resolve }),
    undefined,
  );
  assertEquals(
    await checkUrl(new URL("http://127.0.0.1/"), { allowPrivate: true }),
    undefined,
  );
});

function serve(handler: (req: Request) => Response | Promise<Response>) {
  const ac = new AbortController();
  const server = Deno.serve({
    hostname: "127.0.0.1",
    port: 0,
    signal: ac.signal,
    onListen() {},
  }, handler);
  return {
    base: `http://127.0.0.1:${server.addr.port}`,
    close: () => {
      ac.abort();
      return server.finished;
    },
  };
}

Deno.test("safeFetch follows redirects up to the budget and re-validates each hop", async () => {
  const s = serve((req) => {
    const u = new URL(req.url);
    if (u.pathname === "/loop") {
      return new Response(null, {
        status: 302,
        headers: { location: "/loop" },
      });
    }
    if (u.pathname === "/to-metadata") {
      return new Response(null, {
        status: 302,
        headers: { location: "http://169.254.169.254/" },
      });
    }
    if (u.pathname === "/hop") {
      return new Response(null, {
        status: 302,
        headers: { location: "/final" },
      });
    }
    return new Response("final");
  });
  try {
    const res = await safeFetch(`${s.base}/hop`, {}, { allowPrivate: true });
    assertEquals(await res.text(), "final");
    await assertRejects(
      () =>
        safeFetch(`${s.base}/loop`, {}, {
          allowPrivate: true,
          maxRedirects: 2,
        }),
      Error,
      "too many redirects",
    );
    // A redirect to a private address is refused even though the first hop was allowed.
    await assertRejects(
      () =>
        safeFetch(`${s.base}/to-metadata`, {}, {
          resolve: () => Promise.resolve(["93.184.216.34"]),
        }),
      Error,
      "private",
    );
    // The origin itself is a private literal: refused up front.
    await assertRejects(() => safeFetch(`${s.base}/final`), Error, "private");
  } finally {
    await s.close();
  }
});

Deno.test("safeFetch enforces the deadline", async () => {
  const s = serve(() =>
    new Promise((r) => setTimeout(() => r(new Response("late")), 2_000))
  );
  try {
    await assertRejects(() =>
      safeFetch(`${s.base}/`, {}, { allowPrivate: true, timeoutMs: 100 })
    );
  } finally {
    await s.close();
  }
});

Deno.test("readCappedText truncates with a marker", async () => {
  const body = "x".repeat(10_000);
  const full = await readCappedText(new Response(body), 20_000);
  assertEquals(full, body);
  const capped = await readCappedText(new Response(body), 100);
  assert(capped.startsWith("x".repeat(100)));
  assertStringIncludes(capped, "[truncated at 100 bytes]");
  assertEquals(await readCappedText(new Response(null)), "");
});
