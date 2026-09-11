/**
 * Two-phase commit for file writes. `TransactionManager` implements
 * `@rullama/core`'s `StagingBackend`: `stage()` writes content to a temporary
 * directory without touching the target, `commit()` moves every staged file into
 * place (creating parent directories, falling back to copy + delete across
 * filesystems) and `rollback()` discards the staged files. Target paths are
 * confined to the working directory.
 * Equivalent to Rust's `rullama_tool_runtime::transaction`.
 *
 * @module
 */

import type { CommitResult, StagedWrite, StagingBackend } from "@rullama/core";
import { confinePathLexical, safeFileName } from "./guards.ts";

interface StagedEntry {
  stagedPath: string;
  targetPath: string;
  content: string;
}

/** Filesystem-backed two-phase commit transaction manager. */
export class TransactionManager implements StagingBackend {
  #stagingDir: string;
  #projectRoot: string | undefined;
  #staged: Map<string, StagedEntry> = new Map();

  private constructor(stagingDir: string, projectRoot?: string) {
    this.#stagingDir = stagingDir;
    this.#projectRoot = projectRoot;
  }

  /**
   * Create a new manager using a temporary directory.
   * The staging directory is `<tmpdir>/rullama-txn-<random>` and is created
   * on construction.
   *
   * @param stagingDir Where staged files live (default: a fresh temp dir).
   * @param projectRoot When given, `stage()` throws a `PathEscapeError` for any
   *   `target_path` that resolves outside this directory.
   */
  static create(stagingDir?: string, projectRoot?: string): TransactionManager {
    const dir = stagingDir ??
      `${Deno.env.get("TMPDIR") ?? "/tmp"}/rullama-txn-${crypto.randomUUID()}`;
    Deno.mkdirSync(dir, { recursive: true });
    return new TransactionManager(dir, projectRoot);
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

    if (this.#projectRoot !== undefined) {
      // Throws PathEscapeError when the target leaves the project root.
      confinePathLexical(this.#projectRoot, write.target_path);
    }
    // Percent-encoded: a key such as `../../x` cannot leave the staging dir.
    const stagedPath = `${this.#stagingDir}/${safeFileName(write.key)}.staged`;

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
