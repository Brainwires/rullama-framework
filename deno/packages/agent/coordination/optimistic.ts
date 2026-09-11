/**
 * Optimistic Concurrency with Conflict Resolution.
 *
 * Provides optimistic concurrency control that allows agents to proceed
 * with operations without acquiring locks upfront. Conflicts are detected
 * at commit time and resolved using configured strategies.
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Resource version
// ---------------------------------------------------------------------------

/** Version information for a resource. */
export interface ResourceVersion {
  /** Monotonic version counter; starts at 1 on the first commit. */
  version: number;
  /** Content hash recorded by the commit that produced this version. */
  contentHash: string;
  /** Agent whose commit produced this version. */
  lastModifier: string;
  /** When the version was committed (epoch ms). */
  modifiedAt: number;
}

// ---------------------------------------------------------------------------
// Resolution strategy
// ---------------------------------------------------------------------------

/** Strategies for merging conflicting changes. */
export type MergeStrategy = "text_merge" | "json_merge" | "append" | {
  custom: string;
};

/** Strategy for resolving conflicts. */
export type ResolutionStrategy =
  | { kind: "last_writer_wins" }
  | { kind: "first_writer_wins" }
  | { kind: "merge"; strategy: MergeStrategy }
  | { kind: "escalate" }
  | { kind: "retry"; maxAttempts: number };

// ---------------------------------------------------------------------------
// Conflict types
// ---------------------------------------------------------------------------

/** Describes a conflict between two operations. */
export interface OptimisticConflict {
  /** Resource whose version moved under the committer. */
  resourceId: string;
  /** Agent whose commit was rejected. */
  conflictingAgent: string;
  /** Version the rejected token was based on. */
  expectedVersion: number;
  /** Version actually stored at commit time. */
  actualVersion: number;
  /** Agent that committed the current version (`lastModifier`). */
  holderAgent: string;
  /** When the conflict was detected (epoch ms). */
  detectedAt: number;
}

/** Full conflict information for resolution. */
export interface OptimisticConflictDetails {
  /** Resource in conflict. */
  resourceId: string;
  /** First party to the conflict. */
  agentA: string;
  /** Second party to the conflict. */
  agentB: string;
  /** Version proposed by `agentA`. */
  versionA: ResourceVersion;
  /** Version proposed by `agentB`. */
  versionB: ResourceVersion;
  /** Common ancestor version both parties started from. */
  baseVersion: ResourceVersion;
  /** Content proposed by `agentA`, when available for merging. */
  contentA?: string;
  /** Content proposed by `agentB`, when available for merging. */
  contentB?: string;
}

/** Result of conflict resolution. */
export type Resolution =
  | { kind: "use_version"; agent: string }
  | { kind: "merged"; hash: string }
  | { kind: "abort_both" }
  | { kind: "keep_both"; suffixA: string; suffixB: string }
  | { kind: "retry" }
  | { kind: "escalate"; reason: string };

// ---------------------------------------------------------------------------
// Optimistic token
// ---------------------------------------------------------------------------

/** Token for optimistic operations. */
export interface OptimisticToken {
  /** Resource the operation will commit to. */
  resourceId: string;
  /** Version observed at `beginOptimistic` (0 for an unseen resource). */
  baseVersion: number;
  /** Content hash observed at `beginOptimistic` (empty string for an unseen resource). */
  baseHash: string;
  /** Agent performing the operation. */
  agentId: string;
  /** When the token was issued (epoch ms); compared by `isTokenStale`. */
  createdAt: number;
}

/** Check if a token has expired (stale). */
export function isTokenStale(
  token: OptimisticToken,
  maxAgeMs: number,
): boolean {
  return Date.now() - token.createdAt > maxAgeMs;
}

// ---------------------------------------------------------------------------
// Conflict record
// ---------------------------------------------------------------------------

/** Record of a conflict for history/debugging. */
export interface ConflictRecord {
  /** The conflict that was detected. */
  conflict: OptimisticConflict;
  /** How the controller resolved it. */
  resolution: Resolution;
  /** When the resolution was recorded (epoch ms). */
  resolvedAt: number;
}

// ---------------------------------------------------------------------------
// Commit result
// ---------------------------------------------------------------------------

/** Result of a commit operation. */
export type CommitResult =
  | { kind: "committed"; version: number }
  | { kind: "merged"; version: number; mergedHash: string }
  | { kind: "retry_needed"; currentVersion: number }
  | { kind: "rejected"; reason: string }
  | { kind: "aborted"; reason: string }
  | { kind: "split"; suffixA: string; suffixB: string }
  | { kind: "escalated"; reason: string };

/** Check if the commit result indicates success. */
export function isCommitSuccess(result: CommitResult): boolean {
  return result.kind === "committed" || result.kind === "merged";
}

/** Get the new version from a successful commit. */
export function commitVersion(result: CommitResult): number | undefined {
  if (result.kind === "committed" || result.kind === "merged") {
    return result.version;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/** Statistics about optimistic concurrency. */
export interface OptimisticStats {
  /** Number of resources with at least one committed version. */
  totalResources: number;
  /** Conflicts currently retained in history (bounded by `maxHistory`). */
  totalConflicts: number;
  /** Retained conflicts whose resolution was `"retry"`. */
  resolvedByRetry: number;
  /** Retained conflicts whose resolution was `"escalate"`. */
  escalated: number;
}

// ---------------------------------------------------------------------------
// Optimistic controller
// ---------------------------------------------------------------------------

/** Optimistic concurrency controller. */
export class OptimisticController {
  private versions = new Map<string, ResourceVersion>();
  private strategies = new Map<string, ResolutionStrategy>();
  private defaultStrategy: ResolutionStrategy;
  private conflictHistory: ConflictRecord[] = [];
  private maxHistory: number;

  /**
   * Create a controller with no versions recorded.
   * @param defaultStrategy Used for resources without a registered strategy; defaults to `{ kind: "first_writer_wins" }`.
   * @param maxHistory Maximum conflict records retained (oldest dropped first; default 100).
   */
  constructor(
    defaultStrategy?: ResolutionStrategy,
    maxHistory = 100,
  ) {
    this.defaultStrategy = defaultStrategy ?? { kind: "first_writer_wins" };
    this.maxHistory = maxHistory;
  }

  /** Start an optimistic operation -- returns token with current version. */
  beginOptimistic(agentId: string, resourceId: string): OptimisticToken {
    const current = this.versions.get(resourceId);
    return {
      resourceId,
      baseVersion: current?.version ?? 0,
      baseHash: current?.contentHash ?? "",
      agentId,
      createdAt: Date.now(),
    };
  }

  /** Commit optimistic operation -- throws conflict if version changed. */
  commitOptimistic(
    token: OptimisticToken,
    newContentHash: string,
  ): number {
    const current = this.versions.get(token.resourceId);
    if (current && current.version !== token.baseVersion) {
      const conflict: OptimisticConflict = {
        resourceId: token.resourceId,
        conflictingAgent: token.agentId,
        expectedVersion: token.baseVersion,
        actualVersion: current.version,
        holderAgent: current.lastModifier,
        detectedAt: Date.now(),
      };
      throw conflict;
    }

    const newVersion = token.baseVersion + 1;
    this.versions.set(token.resourceId, {
      version: newVersion,
      contentHash: newContentHash,
      lastModifier: token.agentId,
      modifiedAt: Date.now(),
    });
    return newVersion;
  }

  /** Try to commit, and if conflict occurs, resolve it. */
  commitOrResolve(
    token: OptimisticToken,
    newContentHash: string,
    _newContent?: string,
  ): CommitResult {
    try {
      const version = this.commitOptimistic(token, newContentHash);
      return { kind: "committed", version };
    } catch (e) {
      const conflict = e as OptimisticConflict;
      const resolution = this.resolveConflictAuto(conflict);
      this.recordConflict(conflict, resolution);

      switch (resolution.kind) {
        case "use_version":
          if (resolution.agent === token.agentId) {
            const version = this.forceCommit(
              token.resourceId,
              newContentHash,
              token.agentId,
            );
            return { kind: "committed", version };
          }
          return {
            kind: "rejected",
            reason: `Conflict resolved in favor of ${resolution.agent}`,
          };
        case "merged":
          return {
            kind: "merged",
            version: this.forceCommit(
              token.resourceId,
              resolution.hash,
              token.agentId,
            ),
            mergedHash: resolution.hash,
          };
        case "retry":
          return {
            kind: "retry_needed",
            currentVersion: conflict.actualVersion,
          };
        case "abort_both":
          return {
            kind: "aborted",
            reason: "Both operations aborted due to conflict",
          };
        case "keep_both":
          return {
            kind: "split",
            suffixA: resolution.suffixA,
            suffixB: resolution.suffixB,
          };
        case "escalate":
          return { kind: "escalated", reason: resolution.reason };
      }
    }
  }

  /** Write a new version on top of whatever is stored, bypassing the version check. Returns the new version number. */
  private forceCommit(
    resourceId: string,
    contentHash: string,
    agentId: string,
  ): number {
    const current = this.versions.get(resourceId);
    const newVersion = (current?.version ?? 0) + 1;
    this.versions.set(resourceId, {
      version: newVersion,
      contentHash,
      lastModifier: agentId,
      modifiedAt: Date.now(),
    });
    return newVersion;
  }

  /**
   * Map a conflict to a `Resolution` using the resource's registered strategy
   * (looked up by exact `resourceId`) or the default. `merge` always escalates
   * because no content is available here; `retry` escalates once the version
   * gap reaches `maxAttempts`.
   */
  private resolveConflictAuto(conflict: OptimisticConflict): Resolution {
    const strategy = this.strategies.get(conflict.resourceId) ??
      this.defaultStrategy;

    switch (strategy.kind) {
      case "last_writer_wins":
        return { kind: "use_version", agent: conflict.conflictingAgent };
      case "first_writer_wins":
        return { kind: "use_version", agent: conflict.holderAgent };
      case "retry":
        if (
          conflict.actualVersion - conflict.expectedVersion <
            strategy.maxAttempts
        ) {
          return { kind: "retry" };
        }
        return {
          kind: "escalate",
          reason: `Max retry attempts (${strategy.maxAttempts}) exceeded`,
        };
      case "escalate":
        return {
          kind: "escalate",
          reason: "Configured to escalate all conflicts",
        };
      case "merge":
        return {
          kind: "escalate",
          reason: "Merge requires content, not available",
        };
    }
  }

  /** Append to the conflict history, trimming the oldest entries beyond `maxHistory`. */
  private recordConflict(
    conflict: OptimisticConflict,
    resolution: Resolution,
  ): void {
    this.conflictHistory.push({
      conflict,
      resolution,
      resolvedAt: Date.now(),
    });
    while (this.conflictHistory.length > this.maxHistory) {
      this.conflictHistory.shift();
    }
  }

  /** Register a resolution strategy for a resource pattern. */
  registerStrategy(
    resourcePattern: string,
    strategy: ResolutionStrategy,
  ): void {
    this.strategies.set(resourcePattern, strategy);
  }

  /** Get the current version of a resource. */
  getVersion(resourceId: string): ResourceVersion | undefined {
    return this.versions.get(resourceId);
  }

  /** Check if a resource has been modified since a given version. */
  hasChanged(resourceId: string, sinceVersion: number): boolean {
    const v = this.versions.get(resourceId);
    return v != null && v.version > sinceVersion;
  }

  /** Get conflict history. */
  getConflictHistory(): ConflictRecord[] {
    return [...this.conflictHistory];
  }

  /** Clear conflict history. */
  clearHistory(): void {
    this.conflictHistory = [];
  }

  /** Get statistics. */
  getStats(): OptimisticStats {
    return {
      totalResources: this.versions.size,
      totalConflicts: this.conflictHistory.length,
      resolvedByRetry: this.conflictHistory.filter(
        (r) => r.resolution.kind === "retry",
      ).length,
      escalated: this.conflictHistory.filter(
        (r) => r.resolution.kind === "escalate",
      ).length,
    };
  }
}
