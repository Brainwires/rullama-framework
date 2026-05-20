"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { MessageBubble } from "./MessageBubble";
import { InputBar } from "./InputBar";
import { Sidebar } from "./Sidebar";
import {
  ConnectionState,
  HistoryEntry,
  ServerFrame,
  WebChatClient,
} from "@/lib/ws";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool" | "error";
  content: string;
  /** True while this bubble is still accumulating streamed chunks. */
  streaming?: boolean;
  meta?: string;
}

interface ChatPaneProps {
  wsBase: string;
}

export default function ChatPane({ wsBase }: ChatPaneProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [state, setState] = useState<ConnectionState>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const clientRef = useRef<WebChatClient | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const client = new WebChatClient(wsBase, {
      onStateChange: setState,
      onError: setLastError,
      onFrame: (frame) => handleFrame(frame, setMessages, setSessionId),
      onResume: () => {
        // Ask for history after every (re)connect so the user sees
        // anything they missed during the offline window.
        clientRef.current?.requestResume();
      },
    });
    clientRef.current = client;
    void client.connect();
    return () => client.close();
  }, [wsBase]);

  useEffect(() => {
    scrollerRef.current?.scrollTo({
      top: scrollerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const connectionLabel = useMemo(() => {
    switch (state) {
      case "open":
        return "Connected";
      case "connecting":
        return "Connecting…";
      case "reconnecting":
        return "Reconnecting…";
      case "closed":
        return "Disconnected";
      default:
        return "Idle";
    }
  }, [state]);

  function onSend(text: string) {
    const ok = clientRef.current?.sendMessage(text);
    if (!ok) {
      setLastError("Not connected — message not sent.");
      return;
    }
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content: text },
    ]);
  }

  return (
    <div className="flex h-screen">
      <Sidebar sessionId={sessionId} connection={connectionLabel} />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-bw-border bg-bw-surface px-4 py-3">
          <div className="text-sm text-neutral-300">
            {sessionId ?? "Initialising session…"}
          </div>
          <div className="flex items-center gap-2 text-xs text-neutral-400">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                state === "open" ? "bg-emerald-500" : "bg-amber-500"
              }`}
            />
            {connectionLabel}
          </div>
        </header>
        <div
          ref={scrollerRef}
          role="log"
          aria-live="polite"
          aria-atomic="false"
          aria-label="Conversation history"
          className="flex-1 overflow-y-auto scrollbar-thin bg-bw-bg px-4 py-4"
        >
          <div className="mx-auto flex max-w-3xl flex-col gap-3">
            {messages.length === 0 ? (
              <div className="text-center text-sm text-neutral-500 py-12">
                Start the conversation by sending a message.
              </div>
            ) : null}
            {messages.map((m) => (
              <MessageBubble
                key={m.id}
                kind={m.role === "tool" || m.role === "error" ? m.role : m.role}
                meta={m.meta}
                ariaText={m.content}
              >
                {m.content}
                {m.streaming ? (
                  <span className="ml-1 inline-block h-3 w-1 animate-pulse bg-current align-baseline" />
                ) : null}
              </MessageBubble>
            ))}
            {lastError ? (
              <MessageBubble kind="error" ariaText={lastError}>
                {lastError}
              </MessageBubble>
            ) : null}
          </div>
        </div>
        <InputBar disabled={state !== "open"} onSend={onSend} />
      </div>
    </div>
  );
}

function handleFrame(
  frame: ServerFrame,
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>,
  setSessionId: (id: string) => void,
) {
  switch (frame.type) {
    case "session":
      setSessionId(frame.id);
      return;
    case "message":
      setMessages((prev) => {
        // If the last assistant bubble was streaming, finalise it.
        const last = prev[prev.length - 1];
        if (
          last &&
          last.role === "assistant" &&
          last.streaming &&
          frame.role === "assistant"
        ) {
          const next = prev.slice(0, -1);
          next.push({
            id: frame.id,
            role: "assistant",
            content: frame.content || last.content,
            streaming: false,
          });
          return next;
        }
        return [
          ...prev,
          {
            id: frame.id,
            role: frame.role,
            content: frame.content,
          },
        ];
      });
      return;
    case "chunk":
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant" && last.streaming) {
          const next = prev.slice(0, -1);
          next.push({ ...last, content: last.content + frame.content });
          return next;
        }
        return [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: frame.content,
            streaming: true,
          },
        ];
      });
      return;
    case "tool_use":
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "tool",
          content: `${frame.name} ${frame.status}${
            frame.preview ? ` — ${frame.preview}` : ""
          }`,
        },
      ]);
      return;
    case "history":
      setMessages(() =>
        (frame.messages as HistoryEntry[]).map((m, idx) => ({
          id: `history-${idx}-${m.timestamp}`,
          role: m.role,
          content: m.content,
          meta: "history",
        })),
      );
      return;
    case "error":
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "error", content: frame.message },
      ]);
      return;
  }
}
