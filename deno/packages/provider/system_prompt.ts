/**
 * `ChatOptions.system` for providers whose wire format carries the system
 * prompt as an ordinary chat message (Ollama `/api/chat`, OpenAI Chat
 * Completions).
 *
 * @module
 */

/**
 * Put `system` at the head of `messages` as a system message.
 *
 * Same precedence as the Anthropic, Gemini and Vertex providers
 * (`options.system ?? <system message>`): when `system` is set it wins over
 * a leading system message the caller already passed, which is replaced
 * rather than sent twice. System messages later in the conversation are
 * kept. An unset or empty `system` returns `messages` unchanged.
 *
 * @param messages Wire-format messages, already converted.
 * @param system `ChatOptions.system`.
 * @param make Builds the provider's wire message for the system prompt.
 */
export function withSystemPrompt<M extends { role: string }>(
  messages: M[],
  system: string | undefined,
  make: (text: string) => M,
): M[] {
  if (!system) return messages;
  const rest = messages[0]?.role === "system" ? messages.slice(1) : messages;
  return [make(system), ...rest];
}
