/**
 * Small guards shared by the tool runtime and the built-in tools: path
 * confinement, bounded regular expressions, and filesystem-safe names.
 *
 * @module
 */

import { isAbsolute, relative, resolve } from "@std/path";

/** Longest regular expression a tool accepts from a model (ReDoS budget). */
export const MAX_REGEX_LENGTH = 200;

/** Thrown when a requested path resolves outside its confinement root. */
export class PathEscapeError extends Error {
  /** The root the path had to stay under. */
  readonly root: string;
  /** The offending path, as requested. */
  readonly requested: string;

  /** Create the error for `requested` escaping `root`. */
  constructor(root: string, requested: string) {
    super(`path '${requested}' resolves outside the working directory`);
    this.name = "PathEscapeError";
    this.root = root;
    this.requested = requested;
  }
}

/** True when `candidate` is `root` itself or lies beneath it (both absolute). */
export function isWithin(root: string, candidate: string): boolean {
  const rel = relative(root, candidate);
  return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
}

/**
 * Resolve `requested` against `root` and require the result to stay under
 * `root`. Absolute paths are accepted only when they are already under `root`.
 * Purely lexical — see {@link confinePath} for the symlink-aware check.
 */
export function confinePathLexical(root: string, requested: string): string {
  const base = resolve(root);
  const full = isAbsolute(requested)
    ? resolve(requested)
    : resolve(base, requested);
  if (!isWithin(base, full)) throw new PathEscapeError(base, requested);
  return full;
}

/** Real path of the deepest existing ancestor of `path` (or `path` itself). */
async function realPathOfExisting(path: string): Promise<string> {
  let current = path;
  const missing: string[] = [];
  for (;;) {
    try {
      const real = await Deno.realPath(current);
      return missing.length === 0 ? real : resolve(real, ...missing.reverse());
    } catch (e) {
      if (!(e instanceof Deno.errors.NotFound)) throw e;
      const parent = resolve(current, "..");
      if (parent === current) throw e;
      missing.push(current.slice(parent.length + 1));
      current = parent;
    }
  }
}

/**
 * Like {@link confinePathLexical}, then additionally follows symlinks on both
 * sides so a link inside `root` cannot point at a file outside it. Works for
 * paths that do not exist yet (the deepest existing ancestor is resolved).
 *
 * @returns The lexically resolved absolute path (not the symlink-resolved one),
 *   so callers keep operating on the path the caller named.
 */
export async function confinePath(
  root: string,
  requested: string,
): Promise<string> {
  const full = confinePathLexical(root, requested);
  const realRoot = await Deno.realPath(resolve(root));
  const realFull = await realPathOfExisting(full);
  if (!isWithin(realRoot, realFull)) throw new PathEscapeError(root, requested);
  return full;
}

/**
 * Compile a model-supplied regular expression under a length budget.
 *
 * @throws Error when the pattern is longer than `maxLength` or does not compile.
 */
export function compileBoundedRegex(
  pattern: string,
  flags?: string,
  maxLength = MAX_REGEX_LENGTH,
): RegExp {
  if (pattern.length > maxLength) {
    throw new Error(
      `Regex pattern exceeds maximum length of ${maxLength} characters (got ${pattern.length})`,
    );
  }
  try {
    return new RegExp(pattern, flags);
  } catch (e) {
    throw new Error(
      `Invalid regex pattern '${pattern}': ${(e as Error).message}`,
    );
  }
}

/**
 * A filesystem-safe, collision-free name for an arbitrary key: percent-encodes
 * everything outside `A-Za-z0-9-_.!~*'()`, so it contains no path separators.
 */
export function safeFileName(key: string): string {
  return encodeURIComponent(key);
}
