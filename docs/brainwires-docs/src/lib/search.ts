/**
 * Build-time search index builder.
 *
 * Walks `NAV_TREE` and resolves every href to a backing markdown file using
 * the existing slug maps in `docs.ts` plus a handful of well-known static
 * routes (README.md, CHANGELOG.md, etc.). Produces a flat array of
 * `IndexEntry` records that the client bundles via `/search-index.json`.
 *
 * Design notes:
 * - Keep this Node-only. It is invoked from `scripts/build-search-index.mjs`
 *   and from `search.test.ts`. The runtime SearchDialog never imports it.
 * - Strip gray-matter frontmatter so the body we index is what a reader
 *   actually sees. Fenced code blocks are kept — developers commonly search
 *   for API names that only appear inside code samples.
 */
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";

import { NAV_TREE, type NavItem } from "./nav.ts";
import {
  DENO_SLUG_MAP,
  DOC_SLUG_MAP,
  getCrateReadme,
  getExtraReadme,
  readMarkdownFile,
} from "./docs.ts";

export interface IndexEntry {
  href: string;
  title: string;
  section: string;
  body: string;
}

/**
 * Collection-level routes that render a grid/index and have no single
 * backing markdown file. We still surface them in search (by title) but
 * leave `body` empty rather than inventing content.
 */
const COLLECTION_ROUTES = new Set(["/crates", "/extras"]);

/**
 * Resolve an href to the markdown body that backs it, or `null` if the
 * route has no meaningful textual content (API reference iframe, collection
 * grids, …).
 */
function resolveMarkdown(href: string): string | null {
  if (href === "/") return readMarkdownFile("README.md");
  if (href === "/changelog") return readMarkdownFile("CHANGELOG.md");
  if (href === "/deno") return readMarkdownFile("deno/docs/README.md");
  if (href === "/api-docs") return null;
  if (COLLECTION_ROUTES.has(href)) return null;

  if (href.startsWith("/docs/")) {
    const slug = href.slice("/docs/".length);
    const file = DOC_SLUG_MAP[slug];
    return file ? readMarkdownFile(file) : null;
  }

  if (href.startsWith("/deno/")) {
    const slug = href.slice("/deno/".length);
    const file = DENO_SLUG_MAP[slug];
    return file ? readMarkdownFile(file) : null;
  }

  if (href.startsWith("/crates/")) {
    const name = href.slice("/crates/".length);
    try {
      return getCrateReadme(name);
    } catch {
      return null;
    }
  }

  if (href.startsWith("/extras/")) {
    const name = href.slice("/extras/".length);
    try {
      return getExtraReadme(name);
    } catch {
      return null;
    }
  }

  return null;
}

/**
 * Flatten `NAV_TREE` into `{ href, title, section }` triples. A section is
 * the nearest ancestor group title ("Guides", "Framework Crates", …); for a
 * top-level item that has no ancestor, the section is the item's own title.
 */
function* walkNav(
  items: NavItem[],
  section: string | null,
): Generator<{ href: string; title: string; section: string }> {
  for (const item of items) {
    const effectiveSection = section ?? item.title;
    if (item.href) {
      yield { href: item.href, title: item.title, section: effectiveSection };
    }
    if (item.children) {
      // If the current item has its own title, it becomes the section for
      // its children (e.g. "Framework Crates" → each crate).
      yield* walkNav(item.children, item.title);
    }
  }
}

/**
 * Strip frontmatter from raw markdown. Returns just the body.
 */
function stripFrontmatter(raw: string): string {
  try {
    return matter(raw).content;
  } catch {
    // gray-matter throws on malformed YAML frontmatter — fall back to raw.
    return raw;
  }
}

/**
 * Build the in-memory search index by walking `NAV_TREE` and reading the
 * backing markdown for every href we can resolve.
 */
export async function buildSearchIndex(): Promise<IndexEntry[]> {
  const entries: IndexEntry[] = [];
  const seen = new Set<string>();

  for (const node of walkNav(NAV_TREE, null)) {
    if (seen.has(node.href)) continue;
    seen.add(node.href);

    const raw = resolveMarkdown(node.href);
    const body = raw ? stripFrontmatter(raw).trim() : "";

    entries.push({
      href: node.href,
      title: node.title,
      section: node.section,
      body,
    });
  }

  return entries;
}

/**
 * Build the index and write it to disk as JSON. Creates parent directories
 * as needed.
 */
export async function writeSearchIndex(outPath: string): Promise<void> {
  const index = await buildSearchIndex();
  const dir = path.dirname(outPath);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(index), "utf-8");
}
