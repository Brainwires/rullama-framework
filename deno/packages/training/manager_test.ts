import { assert, assertEquals, assertRejects } from "@std/assert";
import { TrainingManager } from "./manager.ts";
import { OpenAiFineTune } from "./cloud/openai.ts";
import { TrainingError } from "./error.ts";
import { TrainingJobId } from "./types.ts";

Deno.test("manager registers providers", () => {
  const m = new TrainingManager();
  m.addCloudProvider(new OpenAiFineTune("k"));
  assertEquals(m.cloudProviders(), ["openai"]);
  assert(m.getCloudProvider("openai") !== null);
  assertEquals(m.getCloudProvider("missing"), null);
});

Deno.test("manager errors on unknown provider", async () => {
  const m = new TrainingManager();
  const e = await assertRejects(
    () => m.checkCloudJob("nope", new TrainingJobId("x")),
    TrainingError,
  );
  assertEquals(e.kind, "provider");
});
