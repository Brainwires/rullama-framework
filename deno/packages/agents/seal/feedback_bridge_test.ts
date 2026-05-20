import { assert, assertEquals } from "@std/assert";
import { AuditLogger } from "@brainwires/permissions";
import { FeedbackBridge } from "./feedback_bridge.ts";
import { LearningCoordinator } from "./learning.ts";

async function setup(): Promise<[AuditLogger, LearningCoordinator, string]> {
  const dir = await Deno.makeTempDir();
  const logger = AuditLogger.withPath(`${dir}/audit.jsonl`);
  const learning = new LearningCoordinator("test-conv");
  return [logger, learning, dir];
}

Deno.test("process thumbs_up", async () => {
  const [logger, learning, dir] = await setup();
  try {
    logger.submitFeedback("run-1", "thumbs_up", undefined);
    const bridge = new FeedbackBridge(logger, learning);
    const stats = bridge.processFeedbackForRun("run-1");
    assertEquals(stats.processed, 1);
    assertEquals(stats.positive, 1);
    assertEquals(stats.negative, 0);
    assertEquals(stats.corrections_applied, 0);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("process thumbs_down with correction", async () => {
  const [logger, learning, dir] = await setup();
  try {
    logger.submitFeedback("run-2", "thumbs_down", "Use async instead of sync");
    const bridge = new FeedbackBridge(logger, learning);
    const stats = bridge.processFeedbackForRun("run-2");
    assertEquals(stats.processed, 1);
    assertEquals(stats.negative, 1);
    assertEquals(stats.corrections_applied, 1);

    const hints = bridge.learning.global.getPatternHints();
    assertEquals(hints.length, 1);
    assertEquals(hints[0].rule, "Use async instead of sync");
    assertEquals(hints[0].source, "user_feedback");
    assert(Math.abs(hints[0].confidence - 1.0) < Number.EPSILON);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("process multiple feedback for run", async () => {
  const [logger, learning, dir] = await setup();
  try {
    logger.submitFeedback("run-3", "thumbs_up", undefined);
    logger.submitFeedback("run-3", "thumbs_down", "Wrong approach");
    const bridge = new FeedbackBridge(logger, learning);
    const stats = bridge.processFeedbackForRun("run-3");
    assertEquals(stats.processed, 2);
    assertEquals(stats.positive, 1);
    assertEquals(stats.negative, 1);
    assertEquals(stats.corrections_applied, 1);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("process feedback — no results", async () => {
  const [logger, learning, dir] = await setup();
  try {
    const bridge = new FeedbackBridge(logger, learning);
    const stats = bridge.processFeedbackForRun("nonexistent");
    assertEquals(stats.processed, 0);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("feedback isolated between runs", async () => {
  const [logger, learning, dir] = await setup();
  try {
    logger.submitFeedback("run-x", "thumbs_up", undefined);
    logger.submitFeedback("run-y", "thumbs_down", undefined);
    const bridge = new FeedbackBridge(logger, learning);

    const statsX = bridge.processFeedbackForRun("run-x");
    assertEquals(statsX.processed, 1);
    assertEquals(statsX.positive, 1);

    const statsY = bridge.processFeedbackForRun("run-y");
    assertEquals(statsY.processed, 1);
    assertEquals(statsY.negative, 1);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});
