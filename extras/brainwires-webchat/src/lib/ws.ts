/**
 * Browser-side WebSocket client for the BrainClaw `/webchat/ws` endpoint.
 *
 * Features:
 * - Fetches the raw JWT from `/api/token` on every connect (the cookie is
 *   HttpOnly, so we cannot read it directly from `document.cookie`).
 * - Exponential-backoff auto-reconnect.
 * - Calls `onResume` after reconnect so the caller can request history.
 */

export type ServerFrame =
  | { type: "session"; id: string }
  | { type: "message"; role: "user" | "assistant"; content: string; id: string }
  | { type: "chunk"; content: string }
  | { type: "tool_use"; name: string; status: "start" | "end"; preview?: string }
  | { type: "history"; session_id: string; messages: HistoryEntry[] }
  | { type: "error"; message: string };

export interface HistoryEntry {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

export type ConnectionState =
  | "idle"
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed";

export interface WsClientCallbacks {
  onFrame: (frame: ServerFrame) => void;
  onStateChange: (state: ConnectionState) => void;
  onResume?: (sessionId: string) => void;
  onError?: (err: string) => void;
}

export class WebChatClient {
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private attempt = 0;
  private closedByUser = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly wsBase: string,
    private readonly cb: WsClientCallbacks,
  ) {}

  /** Open the connection (or reconnect if we were in backoff). */
  async connect(): Promise<void> {
    this.closedByUser = false;
    this.cb.onStateChange(this.attempt === 0 ? "connecting" : "reconnecting");
    let token: string;
    try {
      const res = await fetch("/api/token");
      if (!res.ok) throw new Error(`token endpoint returned ${res.status}`);
      const body = (await res.json()) as { token?: string };
      if (!body.token) throw new Error("no token in response");
      token = body.token;
    } catch (err) {
      this.cb.onError?.(err instanceof Error ? err.message : String(err));
      this.cb.onStateChange("closed");
      return;
    }

    const url = `${this.wsBase.replace(/\/$/, "")}/webchat/ws?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(url);
    this.ws = ws;

    ws.addEventListener("open", () => {
      this.attempt = 0;
      this.cb.onStateChange("open");
      if (this.sessionId) {
        this.cb.onResume?.(this.sessionId);
      }
    });

    ws.addEventListener("message", (ev) => {
      const raw = typeof ev.data === "string" ? ev.data : "";
      if (!raw) return;
      let frame: ServerFrame;
      try {
        frame = JSON.parse(raw) as ServerFrame;
      } catch {
        this.cb.onError?.("received malformed frame");
        return;
      }
      if (frame.type === "session") {
        this.sessionId = frame.id;
      }
      this.cb.onFrame(frame);
    });

    ws.addEventListener("close", () => {
      if (this.closedByUser) {
        this.cb.onStateChange("closed");
        return;
      }
      this.scheduleReconnect();
    });

    ws.addEventListener("error", () => {
      // Swallow — `close` always follows and triggers reconnect.
    });
  }

  /** Send a user message. */
  sendMessage(content: string): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify({ type: "message", content }));
    return true;
  }

  /** Ask the server for the current session's history. */
  requestResume(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.sessionId) return;
    this.ws.send(
      JSON.stringify({ type: "resume", session_id: this.sessionId }),
    );
  }

  /** Close the connection permanently (user logged out / unloaded). */
  close(): void {
    this.closedByUser = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
    this.cb.onStateChange("closed");
  }

  private scheduleReconnect() {
    this.attempt += 1;
    const delay = Math.min(30_000, 500 * 2 ** Math.min(this.attempt - 1, 6));
    this.cb.onStateChange("reconnecting");
    this.reconnectTimer = setTimeout(() => {
      void this.connect();
    }, delay);
  }
}
