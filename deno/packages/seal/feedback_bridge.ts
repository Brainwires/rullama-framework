/**
 * Feedback Bridge — AuditLogger to SEAL learning loop.
 *
 * Reads user feedback signals (thumbs-up/down + corrections) from the
 * AuditLogger and converts them into SEAL learning signals.
 *
 * Equivalent to Rust's `rullama_agents::seal::feedback_bridge` module.
 */

import {
  type AuditLogger,
  createAuditQuery,
  type FeedbackPolarity,
  type FeedbackSignal,
} from "@rullama/permission";
import type { LearningCoordinator, PatternHint } from "./learning.ts";

/** Statistics from processing a batch of feedback signals. */
export interface FeedbackProcessingStats {
  /** Signals turned into learning outcomes (positive + negative). */
  processed: number;
  /** Signals with `thumbs_up` polarity, each recorded as a successful outcome. */
  positive: number;
  /** Signals with `thumbs_down` polarity, each recorded as a failed outcome. */
  negative: number;
  /** Signals that carried a correction text, each stored as a `PatternHint`. */
  corrections_applied: number;
  /** Audit events skipped because their `polarity` metadata was missing or unrecognised. */
  skipped: number;
}

function emptyStats(): FeedbackProcessingStats {
  return {
    processed: 0,
    positive: 0,
    negative: 0,
    corrections_applied: 0,
    skipped: 0,
  };
}

/**
 * Bridge between the AuditLogger feedback system and the SEAL
 * LearningCoordinator. Uses a pull model: feedback is fetched on demand.
 */
export class FeedbackBridge {
  private audit_logger: AuditLogger;
  /**
   * The coordinator that receives outcomes and pattern hints. Public so tests
   * can inspect learning state (matches Rust field visibility via the
   * `bridge.learning.global.get_pattern_hints()` pattern used there).
   */
  public learning: LearningCoordinator;

  /**
   * Wire an audit logger to a learning coordinator; nothing is read until a
   * `process*` method is called.
   * @param audit_logger Source of feedback signals and `user_feedback` audit events.
   * @param learning Coordinator that outcomes and correction hints are written into.
   */
  constructor(audit_logger: AuditLogger, learning: LearningCoordinator) {
    this.audit_logger = audit_logger;
    this.learning = learning;
  }

  /** Process all feedback signals for a specific run. */
  processFeedbackForRun(run_id: string): FeedbackProcessingStats {
    const signals = this.audit_logger.getFeedbackForRun(run_id);
    const stats = emptyStats();
    for (const signal of signals) this.applySignal(signal, stats);
    return stats;
  }

  /** Process all feedback signals submitted since a given ISO timestamp. */
  processRecentFeedback(since: string): FeedbackProcessingStats {
    const query = createAuditQuery({
      event_type: "user_feedback",
      since,
    });
    const events = this.audit_logger.query(query);
    const stats = emptyStats();

    for (const event of events) {
      const polarityStr = event.metadata["polarity"];
      if (polarityStr === undefined) {
        stats.skipped += 1;
        continue;
      }

      let polarity: FeedbackPolarity;
      if (polarityStr === "thumbs_up") polarity = "thumbs_up";
      else if (polarityStr === "thumbs_down") polarity = "thumbs_down";
      else {
        stats.skipped += 1;
        continue;
      }

      const signal: FeedbackSignal = {
        id: event.metadata["feedback_id"] ?? "",
        run_id: event.metadata["run_id"] ?? "",
        polarity,
        correction: event.metadata["correction"],
        submitted_at: event.timestamp,
      };
      this.applySignal(signal, stats);
    }
    return stats;
  }

  /**
   * Record one signal as a pattern-less query outcome (`thumbs_up` → success
   * with result count 1, otherwise failure with 0), apply any correction as a
   * hint, and bump the matching counters in `stats`.
   */
  private applySignal(
    signal: FeedbackSignal,
    stats: FeedbackProcessingStats,
  ): void {
    const success = signal.polarity === "thumbs_up";
    this.learning.recordOutcome(
      undefined,
      success,
      success ? 1 : 0,
      undefined,
      0,
    );
    if (success) stats.positive += 1;
    else stats.negative += 1;

    if (signal.correction !== undefined) {
      this.applyCorrection(signal.correction, signal.run_id);
      stats.corrections_applied += 1;
    }

    stats.processed += 1;
  }

  /**
   * Store a user correction in global memory as a `PatternHint` with
   * `context_pattern` `run:<run_id>`, confidence `1.0` and source
   * `"user_feedback"`.
   */
  private applyCorrection(correction: string, run_id: string): void {
    const hint: PatternHint = {
      context_pattern: `run:${run_id}`,
      rule: correction,
      confidence: 1.0,
      source: "user_feedback",
    };
    this.learning.global.addPatternHint(hint);
  }
}
