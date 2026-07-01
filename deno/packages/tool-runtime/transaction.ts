/**
 * Two-phase commit transaction manager for file write operations.
 *
 * TransactionManager implements StagingBackend from @rullama/core.
 *
 * ## Protocol
 *
 * 1. **Stage** - calls to stage() write content to a temporary directory with a
 *    key-addressed filename. The target path is not touched.
 *
 * 2. **Commit** - commit() moves each staged file to its target path. Parent
 *    directories are created as needed. On cross-filesystem moves a
 *    copy+delete fallback is used.
 *
 * 3. **Rollback** - rollback() deletes all staged files from the temp dir
 *    without touching any target path.
 *
 * A TransactionManager is single-use per transaction: after commit() or
 * rollback() the queue is empty and new stages can be accepted.
 */

import type {
  CommitResult,
  StagedWrite,
  StagingBackend,
} from "@rullama/core";

interface StagedEntry {
  stagedPath: string;
  targetPath: string;
  content: string;
}

/** Filesystem-backed two-phase commit transaction manager. */
export class TransactionManager implements StagingBackend {
  #stagingDir: string;
  #staged: Map<string, StagedEntry> = new Map();

  private constructor(stagingDir: string) {
    this.#stagingDir = stagingDir;
  }

  /**
   * Create a new manager using a temporary directory.
   * The staging directory is `<tmpdir>/rullama-txn-<random>` and is created
   * on construction.
   */
  static create(stagingDir?: string): TransactionManager {
    const dir = stagingDir ??
      `${
        Deno.env.get("TMPDIR") ?? "/tmp"
      }/rullama-txn-${crypto.randomUUID()}`;
    Deno.mkdirSync(dir, { recursive: true });
    return new TransactionManager(dir);
  }

  /** The temp directory used for staging. */
  get stagingDir(): string {
    return this.#stagingDir;
  }

  /**
   * Stage a write operation.
   * Returns true if the write was newly staged, false if the key was already
   * staged (first write wins; duplicate is a no-op).
   */
  stage(write: StagedWrite): boolean {
    if (this.#staged.has(write.key)) {
      return false;
    }

    const safeName = `${write.key}.staged`;
    const stagedPath = `${this.#stagingDir}/${safeName}`;

    try {
      Deno.writeTextFileSync(stagedPath, write.content);
    } catch (e) {
      console.error(
        `TransactionManager: failed to stage write key=${write.key} path=${stagedPath}: ${e}`,
      );
      return false;
    }

    this.#staged.set(write.key, {
      stagedPath,
      targetPath: write.target_path,
      content: write.content,
    });

    return true;
  }

  /**
   * Commit all staged writes atomically (best-effort).
   * Each staged file is renamed to its target path. On cross-filesystem moves
   * a copy+delete fallback is used. Parent directories are created as needed.
   */
  commit(): CommitResult {
    let committed = 0;
    const paths: string[] = [];

    for (const entry of this.#staged.values()) {
      // Ensure parent directory exists
      const parent = entry.targetPath.substring(
        0,
        entry.targetPath.lastIndexOf("/"),
      );
      if (parent) {
        Deno.mkdirSync(parent, { recursive: true });
      }

      // Attempt atomic rename; fall back to copy+delete
      try {
        Deno.renameSync(entry.stagedPath, entry.targetPath);
      } catch {
        Deno.writeTextFileSync(entry.targetPath, entry.content);
        try {
          Deno.removeSync(entry.stagedPath);
        } catch { /* best-effort cleanup */ }
      }

      committed += 1;
      paths.push(entry.targetPath);
    }

    this.#staged.clear();
    return { committed, paths };
  }

  /** Discard all staged writes without touching any target paths. */
  rollback(): void {
    for (const entry of this.#staged.values()) {
      try {
        Deno.removeSync(entry.stagedPath);
      } catch { /* best-effort cleanup */ }
    }
    this.#staged.clear();
  }

  /** Number of pending staged writes. */
  pendingCount(): number {
    return this.#staged.size;
  }

  /** Clean up the staging directory. Call when done with the manager. */
  dispose(): void {
    this.rollback();
    try {
      Deno.removeSync(this.#stagingDir, { recursive: true });
    } catch { /* best-effort */ }
  }
}
