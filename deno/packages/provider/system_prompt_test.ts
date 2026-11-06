import { assertEquals } from "@std/assert";
import { ChatOptions, Message } from "@rullama/core";
import { OllamaChatProvider } from "./ollama.ts";
import { OpenAiChatProvider } from "./openai.ts";
import { withSystemPrompt } from "./system_prompt.ts";

// ---------------------------------------------------------------------------
// withSystemPrompt
// ---------------------------------------------------------------------------

const sys = (content: string) => ({ role: "system", content });

Deno.test("withSystemPrompt - prepends when there is no system message", () => {
  const out = withSystemPrompt(
    [{ role: "user", content: "hi" }],
    "be terse",
    sys,
  );
  assertEquals(out, [sys("be terse"), { role: "user", content: "hi" }]);
});

Deno.test("withSystemPrompt - unset or empty system leaves messages alone", () => {
  const msgs = [sys("caller"), { role: "user", content: "hi" }];
  assertEquals(withSystemPrompt(msgs, undefined, sys), msgs);
  assertEquals(withSystemPrompt(msgs, "", sys), msgs);
});

Deno.test("withSystemPrompt - replaces a leading system message instead of duplicating it", () => {
  const same = withSystemPrompt(
    [sys("be terse"), { role: "user", content: "hi" }],
    "be terse",
    sys,
  );
  assertEquals(same, [sys("be terse"), { role: "user", content: "hi" }]);
  const other = withSystemPrompt(
    [sys("caller"), { role: "user", content: "hi" }],
    "options",
    sys,
  );
  assertEquals(other, [sys("options"), { role: "user", content: "hi" }]);
});

Deno.test("withSystemPrompt - keeps later system messages", () => {
  const out = withSystemPrompt(
    [{ role: "user", content: "hi" }, sys("mid-conversation")],
    "top",
    sys,
  );
  assertEquals(out, [
    sys("top"),
    { role: "user", content: "hi" },
    sys("mid-conversation"),
  ]);
});

// ---------------------------------------------------------------------------
// Request bodies, through a fake fetch
// ---------------------------------------------------------------------------

/** Run `fn` with `fetch` replaced; returns every JSON request body sent. */
async function captureBodies(
  reply: () => Response,
  fn: () => Promise<void>,
): Promise<Record<string, unknown>[]> {
  const bodies: Record<string, unknown>[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = (_input, init) => {
    bodies.push(JSON.parse(String(init?.body)));
    return Promise.resolve(reply());
  };
  try {
    await fn();
  } finally {
    globalThis.fetch = original;
  }
  return bodies;
}

const ollamaReply = () =>
  Response.json({
    message: { content: "ok" },
    done_reason: "stop",
    prompt_eval_count: 1,
    eval_count: 1,
  });

const ollamaStreamReply = () =>
  new Response(
    JSON.stringify({ message: { content: "ok" }, done: false }) + "\n" +
      JSON.stringify({ done: true, prompt_eval_count: 1, eval_count: 1 }) +
      "\n",
  );

const openaiReply = () =>
  Response.json({
    id: "x",
    choices: [{
      index: 0,
      message: { role: "assistant", content: "ok" },
      finish_reason: "stop",
    }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  });

const openaiStreamReply = () =>
  new Response(
    'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n' +
      "data: [DONE]\n\n",
  );

async function drain(it: AsyncIterable<unknown>): Promise<void> {
  for await (const _ of it) { /* consume */ }
}

interface ProviderCase {
  label: string;
  make: () => OllamaChatProvider | OpenAiChatProvider;
  reply: () => Response;
  streamReply: () => Response;
}

const PROVIDERS: ProviderCase[] = [
  {
    label: "OllamaChatProvider",
    make: () => new OllamaChatProvider("llama3.3", "http://ollama.test"),
    reply: ollamaReply,
    streamReply: ollamaStreamReply,
  },
  {
    label: "OpenAiChatProvider",
    make: () => new OpenAiChatProvider("k", "gpt-4o", "http://openai.test"),
    reply: openaiReply,
    streamReply: openaiStreamReply,
  },
];

/** [title, messages sent, ChatOptions.system, wire messages expected]. */
const SYSTEM_CASES: [string, Message[], string | undefined, unknown[]][] = [
  [
    "sends ChatOptions.system as the first message",
    [Message.user("hi")],
    "be terse",
    [
      { role: "system", content: "be terse" },
      { role: "user", content: "hi" },
    ],
  ],
  [
    "does not duplicate a system message already in messages",
    [Message.system("be terse"), Message.user("hi")],
    "be terse",
    [{ role: "system", content: "be terse" }, { role: "user", content: "hi" }],
  ],
  [
    "ChatOptions.system replaces the caller's leading system message",
    [Message.system("caller"), Message.user("hi")],
    "options",
    [{ role: "system", content: "options" }, { role: "user", content: "hi" }],
  ],
  [
    "without ChatOptions.system the messages pass through",
    [Message.system("caller"), Message.user("hi")],
    undefined,
    [{ role: "system", content: "caller" }, { role: "user", content: "hi" }],
  ],
];

for (const p of PROVIDERS) {
  for (const [title, messages, system, expected] of SYSTEM_CASES) {
    Deno.test(`${p.label}.chat - ${title}`, async () => {
      const provider = p.make();
      const bodies = await captureBodies(p.reply, async () => {
        await provider.chat(messages, undefined, new ChatOptions({ system }));
      });
      assertEquals(bodies[0].messages, expected);
    });
  }

  Deno.test(`${p.label}.streamChat - sends ChatOptions.system as the first message`, async () => {
    const provider = p.make();
    const bodies = await captureBodies(p.streamReply, async () => {
      await drain(
        provider.streamChat(
          [Message.user("hi")],
          undefined,
          new ChatOptions({ system: "be terse" }),
        ),
      );
    });
    assertEquals(bodies[0].messages, [
      { role: "system", content: "be terse" },
      { role: "user", content: "hi" },
    ]);
  });
}
