/**
 * Shared test utilities: minimal Provider mocks used across decorator tests.
 *
 * Equivalent to Rust's `brainwires_resilience::tests_util` module.
 */

import type {
  ChatOptions,
  ChatResponse,
  Message,
  Provider,
  StreamChunk,
  Tool,
} from "@brainwires/core";
import { Message as MessageClass, createUsage } from "@brainwires/core";

type Mode =
  | { kind: "always_ok" }
  | { kind: "always_err"; msg: string }
  | { kind: "err_then_ok"; msg: string };

/** A trivial provider used for decorator tests. */
export class EchoProvider implements Provider {
  readonly name: string;
  private readonly mode: Mode;
  private remaining_errors: number;
  private _calls = 0;

  private constructor(name: string, mode: Mode, remaining_errors = 0) {
    this.name = name;
    this.mode = mode;
    this.remaining_errors = remaining_errors;
  }

  static ok(name: string): EchoProvider {
    return new EchoProvider(name, { kind: "always_ok" });
  }

  static alwaysErr(name: string, msg: string): EchoProvider {
    return new EchoProvider(name, { kind: "always_err", msg });
  }

  static errThenOk(name: string, errors: number, msg: string): EchoProvider {
    return new EchoProvider(name, { kind: "err_then_ok", msg }, errors);
  }

  calls(): number {
    return this._calls;
  }

  chat(
    _messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): Promise<ChatResponse> {
    this._calls += 1;
    switch (this.mode.kind) {
      case "always_ok":
        return Promise.resolve({
          message: MessageClass.assistant("ok"),
          usage: createUsage(4, 2),
          finish_reason: "stop",
        });
      case "always_err":
        return Promise.reject(new Error(this.mode.msg));
      case "err_then_ok": {
        if (this.remaining_errors > 0) {
          this.remaining_errors -= 1;
          return Promise.reject(new Error(this.mode.msg));
        }
        return Promise.resolve({
          message: MessageClass.assistant("ok"),
          usage: createUsage(4, 2),
          finish_reason: "stop",
        });
      }
    }
  }

  streamChat(
    _messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    return (async function* () {
      yield { type: "text", text: "ok" };
      yield { type: "usage", usage: createUsage(4, 2) };
      yield { type: "done" };
    })();
  }
}

/** A provider whose success/failure behaviour can be flipped at runtime. */
export class ToggleProvider implements Provider {
  readonly name: string;
  private _fail = false;

  constructor(name: string) {
    this.name = name;
  }

  setFail(fail: boolean): void {
    this._fail = fail;
  }

  chat(
    _messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): Promise<ChatResponse> {
    if (this._fail) {
      return Promise.reject(new Error("500 internal server error"));
    }
    return Promise.resolve({
      message: MessageClass.assistant("ok"),
      usage: createUsage(4, 2),
      finish_reason: "stop",
    });
  }

  streamChat(
    _messages: Message[],
    _tools: Tool[] | undefined,
    _options: ChatOptions,
  ): AsyncIterable<StreamChunk> {
    return (async function* () {})();
  }
}
