import fs from "fs";
import path from "path";

/**
 * Root of the brainwires-framework repository.
 * In Docker production: /workspace
 * In dev: two levels up from extras/brainwires-docs/
 *
 * Resolved once at module load. Throws at startup if the path does not
 * exist or is not a directory, making misconfiguration immediately visible
 * rather than producing runtime errors on first request.
 */
const _rawRoot =
  process.env.DOCS_ROOT ?? path.resolve(process.cwd(), "..", "..");

const DOCS_ROOT_RESOLVED: string = (() => {
  const resolved = path.resolve(_rawRoot);
  let stat: fs.Stats;
  try {
    stat = fs.statSync(resolved);
  } catch {
    throw new Error(`DOCS_ROOT does not exist: "${resolved}"`);
  }
  if (!stat.isDirectory()) {
    throw new Error(`DOCS_ROOT is not a directory: "${resolved}"`);
  }
  return resolved;
})();

/** Kept for backward compatibility with api-docs/page.tsx */
export function getDocsRoot(): string {
  return DOCS_ROOT_RESOLVED;
}

/**
 * Read a markdown file at relativePath inside DOCS_ROOT.
 *
 * Security: uses realpathSync to resolve all symlinks before the
 * containment check, so both ".." traversal and symlink escapes are
 * caught in one pass. Internal — callers must validate any user-supplied
 * path segments before calling (see assertSafeName, DOC_SLUG_MAP, etc.).
 */
export function readMarkdownFile(relativePath: string): string | null {
  const root = DOCS_ROOT_RESOLVED;
  const joined = path.join(root, relativePath);

  // Resolve every symlink in the chain so the containment check is
  // operating on the true on-disk path.
  let realPath: string;
  try {
    realPath = fs.realpathSync(joined);
  } catch {
    return null;
  }

  // Containment check: the real path must be inside root.
  // The trailing path.sep guard prevents /workspace matching /workspace-other.
  if (!realPath.startsWith(root + path.sep) && realPath !== root) {
    return null;
  }

  try {
    return fs.readFileSync(realPath, "utf-8");
  } catch {
    return null;
  }
}

/** Slug → relative markdown file path mapping for /docs/[...slug] routes */
export const DOC_SLUG_MAP: Record<string, string> = {
  features: "FEATURES.md",
  contributing: "CONTRIBUTING.md",
  testing: "TESTING.md",
  publishing: "PUBLISHING.md",
  extensibility: "docs/EXTENSIBILITY.md",
  "wishlist/distributed-training": "docs/wishlist-crates/Distributed-Training.md",
};

export function getAllDocSlugs(): { slug: string[] }[] {
  return Object.keys(DOC_SLUG_MAP).map((key) => ({
    slug: key.split("/"),
  }));
}

/** Slug → relative markdown file path mapping for /deno/[...slug] routes */
export const DENO_SLUG_MAP: Record<string, string> = {
  index: "deno/docs/README.md",
  "getting-started": "deno/docs/getting-started.md",
  architecture: "deno/docs/architecture.md",
  agents: "deno/docs/agents.md",
  providers: "deno/docs/providers.md",
  tools: "deno/docs/tools.md",
  storage: "deno/docs/storage.md",
  cognition: "deno/docs/cognition.md",
  networking: "deno/docs/networking.md",
  a2a: "deno/docs/a2a.md",
  permissions: "deno/docs/permissions.md",
  extensibility: "deno/docs/extensibility.md",
};

export function getAllDenoSlugs(): { slug: string[] }[] {
  return Object.keys(DENO_SLUG_MAP)
    .filter((k) => k !== "index")
    .map((key) => ({ slug: key.split("/") }));
}

/**
 * Allowlist guard: only alphanumeric, hyphens, and dots.
 * Covers all real crate and extra names. Rejects path separators,
 * null bytes, and everything else not explicitly permitted.
 * An empty string also fails.
 */
function assertSafeName(name: string): void {
  if (!/^[a-zA-Z0-9.\-]+$/.test(name)) {
    throw new Error(`Unsafe path segment: "${name}"`);
  }
}

export function getCrateReadme(name: string): string | null {
  assertSafeName(name);
  return readMarkdownFile(`crates/${name}/README.md`);
}

export function getExtraReadme(name: string): string | null {
  assertSafeName(name);
  return readMarkdownFile(`extras/${name}/README.md`);
}
