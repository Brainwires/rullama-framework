/**
 * Tests for the markdown file reader in `docs.ts`.
 *
 * `docs.ts` resolves `DOCS_ROOT` at module-load time, so every test sets
 * `process.env.DOCS_ROOT` to its own tempdir and dynamically imports the
 * module after `vi.resetModules()`. This keeps tests independent without
 * needing any refactor of the production code.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

interface DocsModule {
  readMarkdownFile: (relative: string) => string | null;
  getDocsRoot: () => string;
  getCrateReadme: (name: string) => string | null;
  getExtraReadme: (name: string) => string | null;
}

async function loadDocs(root: string): Promise<DocsModule> {
  process.env.DOCS_ROOT = root;
  vi.resetModules();
  return (await import("./docs")) as unknown as DocsModule;
}

describe("docs reader", () => {
  let tempRoot: string;
  const originalDocsRoot = process.env.DOCS_ROOT;

  beforeEach(() => {
    tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "bwdocs-"));
  });

  afterEach(() => {
    if (originalDocsRoot === undefined) {
      delete process.env.DOCS_ROOT;
    } else {
      process.env.DOCS_ROOT = originalDocsRoot;
    }
    fs.rmSync(tempRoot, { recursive: true, force: true });
  });

  it("readDoc_parses_markdown_fixture", async () => {
    const body = "---\ntitle: Hello\n---\n\n# Body text\n\nSome content.\n";
    fs.writeFileSync(path.join(tempRoot, "FEATURES.md"), body, "utf-8");

    const docs = await loadDocs(tempRoot);
    const content = docs.readMarkdownFile("FEATURES.md");
    expect(content).not.toBeNull();
    expect(content).toContain("title: Hello");
    expect(content).toContain("# Body text");
    expect(docs.getDocsRoot()).toBe(fs.realpathSync(tempRoot));
  });

  it("readDoc_rejects_path_traversal", async () => {
    // Create a real file *outside* the docs root to link to.
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), "bwoutside-"));
    const secret = path.join(outside, "secret.md");
    fs.writeFileSync(secret, "TOP SECRET", "utf-8");

    // Drop a symlink inside the docs root pointing outside it.
    const linkPath = path.join(tempRoot, "escape.md");
    try {
      fs.symlinkSync(secret, linkPath);
    } catch (err) {
      // Some filesystems (e.g. Windows without admin, certain sandboxes)
      // do not allow symlinks for unprivileged users. Skip rather than
      // false-fail in that case.
      console.warn("symlink unsupported, skipping traversal test:", err);
      fs.rmSync(outside, { recursive: true, force: true });
      return;
    }

    const docs = await loadDocs(tempRoot);
    const content = docs.readMarkdownFile("escape.md");
    expect(content).toBeNull();

    fs.rmSync(outside, { recursive: true, force: true });
  });

  it("readDoc_rejects_escape_via_dotdot", async () => {
    // Create a file one level above tempRoot; even if the joined path
    // resolves to a real file, the containment check must reject it.
    const parent = path.dirname(tempRoot);
    const escapeTarget = path.join(parent, "bw-escape-target.md");
    fs.writeFileSync(escapeTarget, "should not be readable", "utf-8");

    const docs = await loadDocs(tempRoot);
    const content = docs.readMarkdownFile(
      `../${path.basename(escapeTarget)}`,
    );
    expect(content).toBeNull();

    fs.rmSync(escapeTarget, { force: true });
  });

  it("getCrateReadme_rejects_unsafe_names", async () => {
    const docs = await loadDocs(tempRoot);
    expect(() => docs.getCrateReadme("../../etc/passwd")).toThrowError(
      /Unsafe path segment/,
    );
    expect(() => docs.getCrateReadme("")).toThrowError(/Unsafe path segment/);
    expect(() => docs.getCrateReadme("bad name")).toThrowError(
      /Unsafe path segment/,
    );
  });

  it("getCrateReadme_reads_valid_readme", async () => {
    const cratesDir = path.join(tempRoot, "crates", "my-crate");
    fs.mkdirSync(cratesDir, { recursive: true });
    fs.writeFileSync(
      path.join(cratesDir, "README.md"),
      "# my-crate\n\nHello.\n",
      "utf-8",
    );

    const docs = await loadDocs(tempRoot);
    const content = docs.getCrateReadme("my-crate");
    expect(content).toContain("# my-crate");
  });

  it("getExtraReadme_missing_returns_null", async () => {
    const docs = await loadDocs(tempRoot);
    expect(docs.getExtraReadme("does-not-exist")).toBeNull();
  });
});
