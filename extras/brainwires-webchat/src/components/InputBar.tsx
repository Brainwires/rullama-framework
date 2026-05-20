"use client";

import { KeyboardEvent, useRef, useState } from "react";

const SLASH_COMMANDS = [
  "/status",
  "/new",
  "/usage",
  "/think low",
  "/think medium",
  "/think high",
  "/model",
  "/help",
];

export interface InputBarProps {
  disabled?: boolean;
  onSend: (text: string) => void;
}

export function InputBar({ disabled, onSend }: InputBarProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  function submit() {
    const trimmed = value.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setValue("");
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  const showCompletions = value.startsWith("/");
  const completions = showCompletions
    ? SLASH_COMMANDS.filter((c) => c.startsWith(value))
    : [];

  return (
    <div
      role="form"
      aria-label="Send a message"
      className="relative border-t border-bw-border bg-bw-surface p-3"
    >
      {completions.length > 0 ? (
        <ul
          role="listbox"
          aria-label="Slash command suggestions"
          className="absolute bottom-full left-3 mb-1 w-60 rounded border border-bw-border bg-bw-bg shadow-lg"
        >
          {completions.map((c) => (
            <li
              key={c}
              role="option"
              aria-selected={value === c}
              className="cursor-pointer px-3 py-1.5 text-sm hover:bg-bw-assistant"
              onMouseDown={(e) => {
                e.preventDefault();
                setValue(c + " ");
                textareaRef.current?.focus();
              }}
            >
              {c}
            </li>
          ))}
        </ul>
      ) : null}
      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          disabled={disabled}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          aria-label="Message input"
          aria-multiline="true"
          placeholder={
            disabled
              ? "Reconnecting…"
              : "Message BrainClaw (Enter to send, Shift+Enter for newline)"
          }
          className="max-h-40 flex-1 resize-none rounded border border-bw-border bg-bw-bg px-3 py-2 text-sm outline-none focus:border-bw-accent"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || value.trim().length === 0}
          aria-label="Send message"
          className="rounded bg-bw-accent px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}
