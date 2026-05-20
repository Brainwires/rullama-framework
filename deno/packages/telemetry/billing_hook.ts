/**
 * BillingHook — advisory + enforced paths invoked at every billable action.
 *
 * Equivalent to Rust's `brainwires_telemetry::billing_hook` module.
 */

import type { UsageEvent } from "./usage.ts";

/**
 * Receives billable usage events emitted by the agent run loop.
 *
 * - {@link onUsage} is advisory / fail-open: called after an action has
 *   happened. Errors should be logged but not abort the run.
 * - {@link authorize} is enforced / fail-closed: called before a pending
 *   action is dispatched. Throw {@link BillingError.budgetExhausted} to
 *   reject.
 */
export interface BillingHook {
  /** Record a billable event that has already occurred. */
  onUsage(event: UsageEvent): Promise<void>;

  /** Authorize a pending call before it is dispatched. Default: allow all. */
  authorize?(pending: UsageEvent): Promise<void>;
}
