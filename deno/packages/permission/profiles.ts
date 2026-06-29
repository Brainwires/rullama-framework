/**
 * Capability profile presets.
 *
 * Profile types and factory methods are defined in the types module.
 * This module re-exports them for convenience.
 *
 * Rust equivalent: `rullama-permissions/src/profiles.rs`
 * @module
 */

export type { CapabilityProfile } from "./types.ts";
export { parseCapabilityProfile } from "./types.ts";
