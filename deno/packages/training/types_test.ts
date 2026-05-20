import { assert, assertEquals } from "@std/assert";
import {
  completionFraction,
  DatasetId,
  isRunning,
  isSucceeded,
  isTerminal,
  TrainingJobId,
  type TrainingJobStatus,
} from "./types.ts";

Deno.test("job status terminal", () => {
  assert(!isTerminal({ status: "pending" }));
  assert(!isTerminal({ status: "queued" }));
  assert(isTerminal({ status: "succeeded", model_id: "m" }));
  assert(isTerminal({ status: "failed", error: "x" }));
  assert(isTerminal({ status: "cancelled" }));
});

Deno.test("isRunning + isSucceeded tags", () => {
  const running: TrainingJobStatus = {
    status: "running",
    progress: {
      epoch: 1,
      total_epochs: 3,
      step: 5,
      total_steps: 10,
      train_loss: null,
      eval_loss: null,
      learning_rate: null,
      elapsed_secs: 0,
    },
  };
  assert(isRunning(running));
  assert(isSucceeded({ status: "succeeded", model_id: "m" }));
});

Deno.test("progress completion", () => {
  const p = {
    epoch: 0,
    total_epochs: 0,
    step: 50,
    total_steps: 100,
    train_loss: null,
    eval_loss: null,
    learning_rate: null,
    elapsed_secs: 0,
  };
  assertEquals(completionFraction(p), 0.5);
});

Deno.test("job id stringifies", () => {
  const id = new TrainingJobId("ft-abc123");
  assertEquals(id.value, "ft-abc123");
  assertEquals(id.toString(), "ft-abc123");
});

Deno.test("DatasetId from S3 / GCS uris", () => {
  assertEquals(DatasetId.fromS3Uri("s3://bucket/key").value, "s3://bucket/key");
  assertEquals(DatasetId.fromGcsUri("gs://bucket/key").value, "gs://bucket/key");
});
