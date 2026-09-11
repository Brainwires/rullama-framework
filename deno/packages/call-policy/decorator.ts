/**
 * Base class for the `Provider → Provider` decorators in this package: holds
 * the wrapped provider and forwards `name` / `maxOutputTokens()` unchanged.
 *
 * @module
 */

import type {
  ChatOptions,
  ChatResponse,
  Message,
  Provider,
  StreamChunk,
  Tool,
} from "@rullama/core";

/** A provider that wraps another and forwards its identity. */
export abstract class ProviderDecorator implements Provider {
  /** The wrapped provider. */
  readonly inner: Provider;

  /**
   * Store the provider to decorate.
   *
   * @param inner The provider every call is ultimately forwarded to.
   */
  constructor(inner: Provider) {
    this.inner = inner;
  }

  /** The wrapped provider's name, unchanged. */
  get name(): string {
    return this.inner.name;
  }

  /** The wrapped provider's output-token limit, or `undefined` if it doesn't declare one. */
  maxOutputTokens(): number | undefined {
    return this.inner.maxOutputTokens?.();
  }

  /** Subclasses implement the decorated non-streaming call. */
  abstract chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse>;

  /** Subclasses implement the decorated streaming call. */
  abstract streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk>;
}
