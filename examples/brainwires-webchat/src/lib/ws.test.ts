import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WebChatClient, type ConnectionState } from "./ws";

// ── Minimal WebSocket mock ───────────────────────────────────────────────

type Listener = (ev: unknown) => void;

class MockWebSocket {
  static OPEN = 1;
  static CLOSED = 3;
  static instances: MockWebSocket[] = [];

  readyState = MockWebSocket.OPEN;
  url: string;
  sent: string[] = [];
  private listeners: Record<string, Listener[]> = {};

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  addEventListener(type: string, fn: Listener): void {
    (this.listeners[type] ||= []).push(fn);
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
    this.emit("close", {});
  }

  /** Test helpers */
  emit(type: string, ev: unknown): void {
    (this.listeners[type] || []).forEach((fn) => fn(ev));
  }

  fireOpen(): void {
    this.readyState = MockWebSocket.OPEN;
    this.emit("open", {});
  }

  fireClose(): void {
    this.readyState = MockWebSocket.CLOSED;
    this.emit("close", {});
  }

  fireMessage(data: unknown): void {
    this.emit("message", {
      data: typeof data === "string" ? data : JSON.stringify(data),
    });
  }
}

// Expose statics that WebChatClient reads.
Object.assign(MockWebSocket, { OPEN: 1, CLOSED: 3 });

const globalAny = globalThis as unknown as {
  WebSocket: typeof MockWebSocket;
  fetch: typeof fetch;
};

beforeEach(() => {
  MockWebSocket.instances = [];
  globalAny.WebSocket = MockWebSocket;
  // Stub `/api/token` — return a predictable token each call.
  globalAny.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ token: "test.jwt.token" }),
  }) as unknown as typeof fetch;
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function waitForSocket(): Promise<MockWebSocket> {
  return new Promise((resolve) => {
    const tick = () => {
      const last = MockWebSocket.instances.at(-1);
      if (last) return resolve(last);
      // Allow pending microtasks (the fetch promise) to flush.
      setTimeout(tick, 0);
    };
    tick();
  });
}

describe("WebChatClient", () => {
  it("reconnect uses exponential backoff (500, 1000, 2000, ...)", async () => {
    const stateChanges: ConnectionState[] = [];
    const client = new WebChatClient("ws://localhost:8080", {
      onFrame: () => {},
      onStateChange: (s) => stateChanges.push(s),
    });

    // Initial connect.
    const connectPromise = client.connect();
    await vi.runAllTimersAsync();
    await connectPromise;

    const ws1 = MockWebSocket.instances.at(-1)!;
    ws1.fireOpen();

    // First close → scheduleReconnect: attempt=1 → delay = 500 * 2^0 = 500
    ws1.fireClose();
    // Advance just under 500ms: no new socket yet.
    await vi.advanceTimersByTimeAsync(499);
    expect(MockWebSocket.instances.length).toBe(1);
    // Tick past 500 → second connect begins.
    await vi.advanceTimersByTimeAsync(1);
    await vi.runAllTimersAsync();
    expect(MockWebSocket.instances.length).toBe(2);

    const ws2 = MockWebSocket.instances.at(-1)!;
    ws2.fireClose();

    // attempt=2 → delay = 500 * 2^1 = 1000
    await vi.advanceTimersByTimeAsync(999);
    expect(MockWebSocket.instances.length).toBe(2);
    await vi.advanceTimersByTimeAsync(1);
    await vi.runAllTimersAsync();
    expect(MockWebSocket.instances.length).toBe(3);

    const ws3 = MockWebSocket.instances.at(-1)!;
    ws3.fireClose();

    // attempt=3 → delay = 500 * 2^2 = 2000
    await vi.advanceTimersByTimeAsync(1999);
    expect(MockWebSocket.instances.length).toBe(3);
    await vi.advanceTimersByTimeAsync(1);
    await vi.runAllTimersAsync();
    expect(MockWebSocket.instances.length).toBe(4);

    client.close();
  });

  it("stops reconnecting after a user-initiated close", async () => {
    const client = new WebChatClient("ws://localhost:8080", {
      onFrame: () => {},
      onStateChange: () => {},
    });

    const connectPromise = client.connect();
    await vi.runAllTimersAsync();
    await connectPromise;

    const ws1 = await waitForSocket();
    ws1.fireOpen();

    // User-initiated close — client.close() flips `closedByUser`.
    client.close();

    // Even after a long time, no new sockets should appear.
    await vi.advanceTimersByTimeAsync(60_000);
    expect(MockWebSocket.instances.length).toBe(1);
  });

  it("requestResume sends a resume frame with the session id after receiving a session frame", async () => {
    const client = new WebChatClient("ws://localhost:8080", {
      onFrame: () => {},
      onStateChange: () => {},
    });

    const connectPromise = client.connect();
    await vi.runAllTimersAsync();
    await connectPromise;

    const ws1 = await waitForSocket();
    ws1.fireOpen();
    ws1.fireMessage({ type: "session", id: "sess-abc" });

    client.requestResume();
    expect(ws1.sent).toHaveLength(1);
    const frame = JSON.parse(ws1.sent[0]);
    expect(frame).toEqual({ type: "resume", session_id: "sess-abc" });

    client.close();
  });

  it("sendMessage while disconnected returns false (does not queue)", async () => {
    const errors: string[] = [];
    const client = new WebChatClient("ws://localhost:8080", {
      onFrame: () => {},
      onStateChange: () => {},
      onError: (e) => errors.push(e),
    });

    // Never call connect — socket stays null.
    const ok = client.sendMessage("hello");
    expect(ok).toBe(false);
  });

  it("sendMessage after the socket is open returns true and writes a message frame", async () => {
    const client = new WebChatClient("ws://localhost:8080", {
      onFrame: () => {},
      onStateChange: () => {},
    });

    const connectPromise = client.connect();
    await vi.runAllTimersAsync();
    await connectPromise;

    const ws1 = await waitForSocket();
    ws1.fireOpen();

    const ok = client.sendMessage("hi");
    expect(ok).toBe(true);
    expect(ws1.sent).toHaveLength(1);
    expect(JSON.parse(ws1.sent[0])).toEqual({ type: "message", content: "hi" });

    client.close();
  });

  it("resume callback fires on reconnect once a session id has been seen", async () => {
    const resumes: string[] = [];
    const client = new WebChatClient("ws://localhost:8080", {
      onFrame: () => {},
      onStateChange: () => {},
      onResume: (id) => resumes.push(id),
    });

    const connectPromise = client.connect();
    await vi.runAllTimersAsync();
    await connectPromise;

    const ws1 = await waitForSocket();
    ws1.fireOpen();
    ws1.fireMessage({ type: "session", id: "sess-xyz" });

    // Drop the socket; the client should reconnect.
    ws1.fireClose();
    await vi.advanceTimersByTimeAsync(600);
    await vi.runAllTimersAsync();

    const ws2 = MockWebSocket.instances.at(-1)!;
    ws2.fireOpen();

    expect(resumes).toContain("sess-xyz");

    client.close();
  });
});
