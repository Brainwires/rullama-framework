import { assert, assertEquals } from "@std/assert";
import { OpenAiFineTune } from "./openai.ts";
import { TogetherFineTune } from "./together.ts";
import { FireworksFineTune } from "./fireworks.ts";

Deno.test("OpenAI parses running + succeeded", () => {
  const running = OpenAiFineTune.parseJobStatus("running", { trained_tokens: 1000 });
  assertEquals(running.status, "running");

  const ok = OpenAiFineTune.parseJobStatus("succeeded", { fine_tuned_model: "ft:gpt-4o-mini:abc" });
  assertEquals(ok.status, "succeeded");
  if (ok.status === "succeeded") assertEquals(ok.model_id, "ft:gpt-4o-mini:abc");
});

Deno.test("OpenAI parses failed", () => {
  const failed = OpenAiFineTune.parseJobStatus("failed", { error: { message: "insufficient tokens" } });
  assertEquals(failed.status, "failed");
  if (failed.status === "failed") assert(failed.error.includes("insufficient"));
});

Deno.test("Together parses completed with output_name", () => {
  const ok = TogetherFineTune.parseJobStatus("completed", { output_name: "org/ft-123" });
  assertEquals(ok.status, "succeeded");
  if (ok.status === "succeeded") assertEquals(ok.model_id, "org/ft-123");
});

Deno.test("Together parses user_cancelled", () => {
  const c = TogetherFineTune.parseJobStatus("user_cancelled", {});
  assertEquals(c.status, "cancelled");
});

Deno.test("Fireworks parses JOB_STATE_COMPLETED", () => {
  const ok = FireworksFineTune.parseJobStatus("JOB_STATE_COMPLETED", { model: "accounts/a/models/b" });
  assertEquals(ok.status, "succeeded");
});

Deno.test("supported base models are non-empty", () => {
  assert(new OpenAiFineTune("k").supportedBaseModels().length > 0);
  assert(new TogetherFineTune("k").supportedBaseModels().length > 0);
  assert(new FireworksFineTune("k").supportedBaseModels().length > 0);
});
