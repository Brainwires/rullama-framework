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

  constructor(inner: Provider) {
    this.inner = inner;
  }

  get name(): string {
    return this.inner.name;
  }

  maxOutputTokens(): number | undefined {
    return this.inner.maxOutputTokens?.();
  }

  abstract chat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): Promise<ChatResponse>;

  abstract streamChat(
    messages: Message[],
    tools: Tool[] | undefined,
    options: ChatOptions,
  ): AsyncIterable<StreamChunk>;
}
