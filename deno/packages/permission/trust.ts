/**
 * Trust Factor System — Dynamic trust scoring for agents.
 *
 * Implements a reputation-based trust system where agents build trust through
 * successful operations and lose trust through violations.
 *
 * Rust equivalent: `rullama-permissions/src/trust.rs`
 * @module
 */

// ── Trust Level ─────────────────────────────────────────────────────

/**
 * Trust level enum representing discrete trust categories.
 *
 * Rust equivalent: `TrustLevel` enum (serde `rename_all = "lowercase"`)
 */
export type TrustLevel = "untrusted" | "low" | "medium" | "high" | "system";

/** Numeric value for a trust level (matches Rust discriminant). */
export function trustLevelToU8(level: TrustLevel): number {
  switch (level) {
    case "untrusted":
      return 0;
    case "low":
      return 1;
    case "medium":
      return 2;
    case "high":
      return 3;
    case "system":
      return 4;
  }
}

/** Convert from numeric level. Rust equivalent: `TrustLevel::from_u8()` */
export function trustLevelFromU8(level: number): TrustLevel {
  switch (level) {
    case 0:
      return "untrusted";
    case 1:
      return "low";
    case 2:
      return "medium";
    case 3:
      return "high";
    case 4:
      return "system";
    default:
      return "low";
  }
}

/** Derive trust level from a score (0.0 to 1.0). Rust equivalent: `TrustLevel::from_score()` */
export function trustLevelFromScore(score: number): TrustLevel {
  if (score >= 0.9) return "high";
  if (score >= 0.7) return "medium";
  if (score >= 0.4) return "low";
  return "untrusted";
}

/** Compare trust levels. Returns negative if a < b, positive if a > b, 0 if equal. */
export function compareTrustLevels(a: TrustLevel, b: TrustLevel): number {
  return trustLevelToU8(a) - trustLevelToU8(b);
}

// ── Violation Severity ──────────────────────────────────────────────

/**
 * Severity of a policy violation.
 *
 * Rust equivalent: `ViolationSeverity` enum (serde `rename_all = "lowercase"`)
 */
export type ViolationSeverity = "minor" | "major" | "critical";

/** Get the penalty for this violation severity. Rust equivalent: `ViolationSeverity::penalty()` */
export function violationPenalty(severity: ViolationSeverity): number {
  switch (severity) {
    case "minor":
      return 0.02;
    case "major":
      return 0.08;
    case "critical":
      return 0.15;
  }
}

/** Get the recent penalty multiplier. Rust equivalent: `ViolationSeverity::recent_penalty()` */
export function violationRecentPenalty(severity: ViolationSeverity): number {
  switch (severity) {
    case "minor":
      return 0.04;
    case "major":
      return 0.15;
    case "critical":
      return 0.30;
  }
}

// ── Violation Counts ────────────────────────────────────────────────

/**
 * Violation counts by severity.
 *
 * Rust equivalent: `ViolationCounts` struct
 */
export interface ViolationCounts {
  minor: number;
  major: number;
  critical: number;
  recent_minor: number;
  recent_major: number;
  recent_critical: number;
}

/** Create default violation counts. */
export function defaultViolationCounts(): ViolationCounts {
  return {
    minor: 0,
    major: 0,
    critical: 0,
    recent_minor: 0,
    recent_major: 0,
    recent_critical: 0,
  };
}

/** Calculate total penalty from violations. Rust equivalent: `ViolationCounts::total_penalty()` */
export function violationsTotalPenalty(counts: ViolationCounts): number {
  const basePenalty = counts.minor * violationPenalty("minor") +
    counts.major * violationPenalty("major") +
    counts.critical * violationPenalty("critical");

  const recentPenalty = counts.recent_minor * violationRecentPenalty("minor") +
    counts.recent_major * violationRecentPenalty("major") +
    counts.recent_critical * violationRecentPenalty("critical");

  return basePenalty + recentPenalty;
}

/** Record a violation. Rust equivalent: `ViolationCounts::record()` */
export function recordViolation(
  counts: ViolationCounts,
  severity: ViolationSeverity,
): void {
  switch (severity) {
    case "minor":
      counts.minor++;
      counts.recent_minor++;
      break;
    case "major":
      counts.major++;
      counts.recent_major++;
      break;
    case "critical":
      counts.critical++;
      counts.recent_critical++;
      break;
  }
}

/** Decay recent violations. Rust equivalent: `ViolationCounts::decay_recent()` */
export function decayRecentViolations(counts: ViolationCounts): void {
  counts.recent_minor = 0;
  counts.recent_major = 0;
  counts.recent_critical = 0;
}

// ── Trust Factor ────────────────────────────────────────────────────

/**
 * Trust factor for an agent.
 *
 * Rust equivalent: `TrustFactor` struct
 */
export interface TrustFactor {
  /** Agent ID. */
  agent_id: string;
  /** Current trust score (0.0 to 1.0). */
  score: number;
  /** Derived trust level. */
  level: TrustLevel;
  /** Violation counts. */
  violations: ViolationCounts;
  /** Number of successful operations. */
  successful_ops: number;
  /** Total number of operations. */
  total_ops: number;
  /** When this factor was last updated (ISO 8601). */
  last_updated: string;
  /** When recent violations should be decayed (ISO 8601). */
  violations_decay_at: string;
  /** Whether this is a system agent (always trusted). */
  is_system: boolean;
}

/** Create a new trust factor for an agent. Rust equivalent: `TrustFactor::new()` */
export function createTrustFactor(agentId: string): TrustFactor {
  const now = new Date().toISOString();
  const decayAt = new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString();
  return {
    agent_id: agentId,
    score: 0.5,
    level: "low",
    violations: defaultViolationCounts(),
    successful_ops: 0,
    total_ops: 0,
    last_updated: now,
    violations_decay_at: decayAt,
    is_system: false,
  };
}

/** Create a system agent trust factor. Rust equivalent: `TrustFactor::system()` */
export function createSystemTrustFactor(agentId: string): TrustFactor {
  const now = new Date().toISOString();
  return {
    agent_id: agentId,
    score: 1.0,
    level: "system",
    violations: defaultViolationCounts(),
    successful_ops: 0,
    total_ops: 0,
    last_updated: now,
    violations_decay_at: now,
    is_system: true,
  };
}

/** Recalculate score and level. Rust equivalent of `TrustFactor::recalculate()` (private in Rust). */
function recalculate(factor: TrustFactor): void {
  if (factor.is_system) return;

  if (new Date().toISOString() > factor.violations_decay_at) {
    decayRecentViolations(factor.violations);
    factor.violations_decay_at = new Date(Date.now() + 24 * 60 * 60 * 1000)
      .toISOString();
  }

  const baseScore = factor.total_ops > 0
    ? factor.successful_ops / factor.total_ops
    : 0.5;

  const penalty = violationsTotalPenalty(factor.violations);
  factor.score = Math.max(0, Math.min(1, baseScore - penalty));
  factor.level = trustLevelFromScore(factor.score);
  factor.last_updated = new Date().toISOString();
}

/** Record a successful operation. Rust equivalent: `TrustFactor::record_success()` */
export function trustFactorRecordSuccess(factor: TrustFactor): void {
  factor.successful_ops++;
  factor.total_ops++;
  recalculate(factor);
}

/** Record a failed operation. Rust equivalent: `TrustFactor::record_failure()` */
export function trustFactorRecordFailure(factor: TrustFactor): void {
  factor.total_ops++;
  recalculate(factor);
}

/** Record a policy violation. Rust equivalent: `TrustFactor::record_violation()` */
export function trustFactorRecordViolation(
  factor: TrustFactor,
  severity: ViolationSeverity,
): void {
  recordViolation(factor.violations, severity);
  factor.total_ops++;
  recalculate(factor);
}

/** Manually set trust level. Rust equivalent: `TrustFactor::set_level()` */
export function trustFactorSetLevel(
  factor: TrustFactor,
  level: TrustLevel,
): void {
  factor.level = level;
  switch (level) {
    case "untrusted":
      factor.score = 0.2;
      break;
    case "low":
      factor.score = 0.5;
      break;
    case "medium":
      factor.score = 0.75;
      break;
    case "high":
      factor.score = 0.95;
      break;
    case "system":
      factor.score = 1.0;
      break;
  }
  factor.last_updated = new Date().toISOString();
}

/** Reset trust factor to defaults. Rust equivalent: `TrustFactor::reset()` */
export function trustFactorReset(factor: TrustFactor): void {
  factor.score = 0.5;
  factor.level = "low";
  factor.violations = defaultViolationCounts();
  factor.successful_ops = 0;
  factor.total_ops = 0;
  factor.last_updated = new Date().toISOString();
}

// ── Trust Statistics ────────────────────────────────────────────────

/**
 * Trust statistics.
 *
 * Rust equivalent: `TrustStatistics` struct
 */
export interface TrustStatistics {
  total_agents: number;
  untrusted: number;
  low_trust: number;
  medium_trust: number;
  high_trust: number;
  system: number;
  total_violations: number;
  total_operations: number;
  average_score: number;
}

// ── Trust Manager ───────────────────────────────────────────────────

/**
 * Trust manager for managing agent trust factors.
 *
 * Rust equivalent: `TrustManager` struct
 */
export class TrustManager {
  #factors: Map<string, TrustFactor> = new Map();
  #storePath: string;
  #persist: boolean;

  private constructor(storePath: string, persist: boolean) {
    this.#storePath = storePath;
    this.#persist = persist;
  }

  /**
   * Create a trust manager with custom path.
   *
   * Rust equivalent: `TrustManager::with_path()`
   */
  static withPath(path: string): TrustManager {
    const manager = new TrustManager(path, true);
    manager.#load();
    return manager;
  }

  /**
   * Create an in-memory only trust manager (no persistence).
   *
   * Rust equivalent: `TrustManager::in_memory()`
   */
  static inMemory(): TrustManager {
    return new TrustManager("", false);
  }

  /**
   * Create a new trust manager with the default path.
   *
   * Rust equivalent: `TrustManager::new()`
   */
  static create(): TrustManager {
    const home = Deno.env.get("HOME") ?? Deno.env.get("USERPROFILE") ?? ".";
    return TrustManager.withPath(`${home}/.rullama/trust_store.json`);
  }

  #load(): void {
    try {
      const content = Deno.readTextFileSync(this.#storePath);
      const store = JSON.parse(content) as {
        factors: Record<string, TrustFactor>;
      };
      for (const [key, factor] of Object.entries(store.factors)) {
        if (new Date().toISOString() > factor.violations_decay_at) {
          decayRecentViolations(factor.violations);
          factor.violations_decay_at = new Date(
            Date.now() + 24 * 60 * 60 * 1000,
          ).toISOString();
        }
        this.#factors.set(key, factor);
      }
    } catch { /* file doesn't exist yet */ }
  }

  /** Save trust data to disk. Rust equivalent: `TrustManager::save()` */
  save(): void {
    if (!this.#persist) return;
    try {
      const parent = this.#storePath.substring(
        0,
        this.#storePath.lastIndexOf("/"),
      );
      if (parent) Deno.mkdirSync(parent, { recursive: true });
      const store = {
        factors: Object.fromEntries(this.#factors),
        last_saved: new Date().toISOString(),
      };
      Deno.writeTextFileSync(this.#storePath, JSON.stringify(store, null, 2));
    } catch { /* ignore */ }
  }

  /** Get or create a trust factor for an agent. Rust equivalent: `TrustManager::get_or_create()` */
  getOrCreate(agentId: string): TrustFactor {
    let factor = this.#factors.get(agentId);
    if (!factor) {
      factor = createTrustFactor(agentId);
      this.#factors.set(agentId, factor);
    }
    return factor;
  }

  /** Get trust factor for an agent (if exists). Rust equivalent: `TrustManager::get()` */
  get(agentId: string): TrustFactor | undefined {
    return this.#factors.get(agentId);
  }

  /** Get trust level for an agent. Rust equivalent: `TrustManager::get_trust_level()` */
  getTrustLevel(agentId: string): TrustLevel {
    return this.#factors.get(agentId)?.level ?? "low";
  }

  /** Record a successful operation. Rust equivalent: `TrustManager::record_success()` */
  recordSuccess(agentId: string): void {
    const factor = this.getOrCreate(agentId);
    trustFactorRecordSuccess(factor);
    this.save();
  }

  /** Record a failed operation. Rust equivalent: `TrustManager::record_failure()` */
  recordFailure(agentId: string): void {
    const factor = this.getOrCreate(agentId);
    trustFactorRecordFailure(factor);
    this.save();
  }

  /** Record a violation. Rust equivalent: `TrustManager::record_violation()` */
  recordViolation(agentId: string, severity: ViolationSeverity): void {
    const factor = this.getOrCreate(agentId);
    trustFactorRecordViolation(factor, severity);
    this.save();
  }

  /** Set trust level for an agent (manual override). Rust equivalent: `TrustManager::set_trust_level()` */
  setTrustLevel(agentId: string, level: TrustLevel): void {
    const factor = this.getOrCreate(agentId);
    trustFactorSetLevel(factor, level);
    this.save();
  }

  /** Reset an agent's trust. Rust equivalent: `TrustManager::reset()` */
  reset(agentId: string): void {
    const factor = this.#factors.get(agentId);
    if (factor) {
      trustFactorReset(factor);
      this.save();
    }
  }

  /** Remove an agent's trust data. Rust equivalent: `TrustManager::remove()` */
  remove(agentId: string): TrustFactor | undefined {
    const factor = this.#factors.get(agentId);
    this.#factors.delete(agentId);
    this.save();
    return factor;
  }

  /** Register a system agent. Rust equivalent: `TrustManager::register_system_agent()` */
  registerSystemAgent(agentId: string): void {
    this.#factors.set(agentId, createSystemTrustFactor(agentId));
    this.save();
  }

  /** Get all agent IDs. Rust equivalent: `TrustManager::agents()` */
  agents(): string[] {
    return [...this.#factors.keys()];
  }

  /** Get statistics. Rust equivalent: `TrustManager::statistics()` */
  statistics(): TrustStatistics {
    const stats: TrustStatistics = {
      total_agents: this.#factors.size,
      untrusted: 0,
      low_trust: 0,
      medium_trust: 0,
      high_trust: 0,
      system: 0,
      total_violations: 0,
      total_operations: 0,
      average_score: 0,
    };

    let totalSuccess = 0;
    for (const factor of this.#factors.values()) {
      switch (factor.level) {
        case "untrusted":
          stats.untrusted++;
          break;
        case "low":
          stats.low_trust++;
          break;
        case "medium":
          stats.medium_trust++;
          break;
        case "high":
          stats.high_trust++;
          break;
        case "system":
          stats.system++;
          break;
      }
      stats.total_violations += factor.violations.minor +
        factor.violations.major + factor.violations.critical;
      stats.total_operations += factor.total_ops;
      totalSuccess += factor.successful_ops;
    }

    if (stats.total_operations > 0) {
      stats.average_score = totalSuccess / stats.total_operations;
    }

    return stats;
  }
}
