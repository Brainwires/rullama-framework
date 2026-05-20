/**
 * Tests for the build-time search index builder.
 *
 * Like `docs.test.ts`, these tests swap `DOCS_ROOT` via env before
 * dynamically re-importing the modules, so the search builder operates
 * against an isolated tempdir.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import Fuse from "fuse.js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FUSE_OPTIONS, type IndexEntry } from "@/components/docs/search-dialog";

// Matches what `search.ts` exports — declared locally to keep the dynamic
// import fully typed without coupling the test file to the implementation.
interface SearchModule {
  buildSearchIndex: () => Promise<IndexEntry[]>;
  writeSearchIndex: (out: string) => Promise<void>;
}

async function loadSearch(root: string): Promise<SearchModule> {
  process.env.DOCS_ROOT = root;
  vi.resetModules();
  return (await import("./search")) as unknown as SearchModule;
}

/**
 * Populate `tempRoot` with enough content for every href `NAV_TREE` walks
 * to resolve to a real file. We use short stub bodies — the test cares that
 * each entry ends up non-empty, not about the exact bytes.
 */
function seedDocsRoot(tempRoot: string): void {
  const write = (rel: string, body: string): void => {
    const full = path.join(tempRoot, rel);
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, body, "utf-8");
  };

  // Root-level docs referenced by NAV_TREE directly.
  write("README.md", "# brainwires\n\nGetting started body.\n");
  write("CHANGELOG.md", "# Changelog\n\n- v0.1 initial.\n");
  write("FEATURES.md", "# Features\n\nFeature list goes here.\n");
  write("CONTRIBUTING.md", "# Contributing\n\nHow to help.\n");
  write("TESTING.md", "# Testing\n\nRun cargo test.\n");
  write("PUBLISHING.md", "# Publishing\n\nCrates.io steps.\n");
  write("docs/EXTENSIBILITY.md", "# Extension Points\n\nPlug-in surface.\n");
  write(
    "docs/wishlist-crates/Distributed-Training.md",
    "# Distributed Training\n\nWishlist.\n",
  );

  // Deno SDK docs.
  const denoPages = [
    "README",
    "getting-started",
    "architecture",
    "agents",
    "providers",
    "tools",
    "storage",
    "cognition",
    "networking",
    "a2a",
    "permissions",
    "extensibility",
  ];
  for (const page of denoPages) {
    write(`deno/docs/${page}.md`, `# Deno ${page}\n\nContent for ${page}.\n`);
  }

  // Every crate and extra referenced by NAV_TREE needs a README at its
  // canonical location.
  const crates = [
    "brainwires",
    "brainwires-core",
    "brainwires-providers",
    "brainwires-agents",
    "brainwires-cognition",
    "brainwires-training",
    "brainwires-storage",
    "brainwires-mcp",
    "brainwires-mcp-server",
    "brainwires-agent-network",
    "brainwires-tool-system",
    "brainwires-skills",
    "brainwires-hardware",
    "brainwires-datasets",
    "brainwires-autonomy",
    "brainwires-permissions",
    "brainwires-a2a",
    "brainwires-channels",
    "brainwires-code-interpreters",
    "brainwires-analytics",
    "brainwires-wasm",
  ];
  for (const crate of crates) {
    write(
      `crates/${crate}/README.md`,
      `# ${crate}\n\nCrate ${crate} description.\n`,
    );
  }

  const extras = [
    "brainwires-cli",
    "brainwires-proxy",
    "brainwires-brain-server",
    "brainwires-rag-server",
    "brainwires-issues",
    "agent-chat",
    "audio-demo",
    "audio-demo-ffi",
    "reload-daemon",
  ];
  for (const extra of extras) {
    write(
      `extras/${extra}/README.md`,
      `# ${extra}\n\nExtra ${extra} description.\n`,
    );
  }
}

describe("search index builder", () => {
  let tempRoot: string;
  const originalDocsRoot = process.env.DOCS_ROOT;

  beforeEach(() => {
    tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "bwsearch-"));
  });

  afterEach(() => {
    if (originalDocsRoot === undefined) {
      delete process.env.DOCS_ROOT;
    } else {
      process.env.DOCS_ROOT = originalDocsRoot;
    }
    fs.rmSync(tempRoot, { recursive: true, force: true });
  });

  it("buildSearchIndex_covers_all_nav_entries", async () => {
    seedDocsRoot(tempRoot);
    const search = await loadSearch(tempRoot);
    const { NAV_TREE } = await import("./nav");

    const entries = await search.buildSearchIndex();
    const hrefs = new Set(entries.map((e) => e.href));

    // Collect every href NAV_TREE exposes.
    const expectedHrefs: string[] = [];
    const walk = (items: (typeof NAV_TREE)[number][]): void => {
      for (const item of items) {
        if (item.href) expectedHrefs.push(item.href);
        if (item.children) walk(item.children);
      }
    };
    walk(NAV_TREE);

    for (const href of expectedHrefs) {
      expect(hrefs.has(href), `index missing href ${href}`).toBe(true);
    }

    // Every entry has a non-empty title; every entry we backed with a
    // markdown file has a non-empty body. (Collection routes /crates,
    // /extras and the /api-docs stub are allowed to be empty.)
    for (const entry of entries) {
      expect(entry.title, `title missing for ${entry.href}`).toBeTruthy();
      if (!["/crates", "/extras", "/api-docs"].includes(entry.href)) {
        expect(entry.body, `body missing for ${entry.href}`).not.toEqual("");
      }
    }
  });

  it("search_index_content_matches_expected_fixture", async () => {
    // Minimum file set: `docs.ts` resolves `DOCS_ROOT` at import time and
    // also eagerly reads README.md etc. for the root href. We just need the
    // files NAV_TREE visits to exist; missing ones fall through to empty
    // bodies which would fail the assertion below for our target entry.
    seedDocsRoot(tempRoot);

    // Overwrite one file with fixture content we can grep for.
    const fixtureBody =
      "---\ntitle: Fixture\n---\n\n# Fixture heading\n\nUNIQUE-TOKEN-ALPHA.\n";
    fs.writeFileSync(
      path.join(tempRoot, "FEATURES.md"),
      fixtureBody,
      "utf-8",
    );

    const search = await loadSearch(tempRoot);
    const entries = await search.buildSearchIndex();

    const featuresEntry = entries.find((e) => e.href === "/docs/features");
    expect(featuresEntry).toBeDefined();
    expect(featuresEntry!.title).toBe("Features");
    // Frontmatter stripped.
    expect(featuresEntry!.body).not.toContain("title: Fixture");
    // Body preserved.
    expect(featuresEntry!.body).toContain("UNIQUE-TOKEN-ALPHA");
    expect(featuresEntry!.body).toContain("# Fixture heading");
  });

  it("fuse_finds_expected_hit", () => {
    const corpus: IndexEntry[] = [
      {
        href: "/crates/brainwires-cognition",
        title: "brainwires-cognition",
        section: "Framework Crates",
        body: "Unified intelligence layer — knowledge graphs, adaptive prompting, RAG.",
      },
      {
        href: "/crates/brainwires-storage",
        title: "brainwires-storage",
        section: "Framework Crates",
        body: "Backend-agnostic storage and tiered memory.",
      },
      {
        href: "/docs/testing",
        title: "Testing",
        section: "Guides",
        body: "How to run the test suite for contributors.",
      },
    ];

    const fuse = new Fuse(corpus, FUSE_OPTIONS);
    const hits = fuse.search("cognition");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].item.href).toBe("/crates/brainwires-cognition");

    // Body-only match still surfaces the right hit.
    const bodyHits = fuse.search("knowledge graph");
    expect(bodyHits.length).toBeGreaterThan(0);
    expect(bodyHits[0].item.href).toBe("/crates/brainwires-cognition");
  });
});
