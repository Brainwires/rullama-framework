/**
 * Structural tests for the hand-authored navigation tree.
 *
 * `NAV_TREE` is a static constant rather than a filesystem-driven builder,
 * so the goal here is to catch regressions in shape: every leaf has an
 * href, every section has children, and href conventions match the docs
 * routes referenced by `docs.ts`.
 */

import { describe, expect, it } from "vitest";

import { NAV_TREE, type NavItem } from "./nav";
import { DOC_SLUG_MAP, DENO_SLUG_MAP } from "./docs";

function walk(items: NavItem[], visit: (item: NavItem) => void): void {
  for (const item of items) {
    visit(item);
    if (item.children) walk(item.children, visit);
  }
}

describe("navigation tree", () => {
  it("buildNav_produces_nested_tree_in_order", () => {
    // Top-level ordering is fixed: the reader is expected to see Getting
    // Started first and Changelog / API Reference last.
    const topTitles = NAV_TREE.map((i) => i.title);
    expect(topTitles[0]).toBe("Getting Started");
    expect(topTitles).toContain("Guides");
    expect(topTitles).toContain("Framework Crates");
    expect(topTitles).toContain("Extras");
    expect(topTitles).toContain("Deno SDK");
    expect(topTitles[topTitles.length - 2]).toBe("Changelog");
    expect(topTitles[topTitles.length - 1]).toBe("API Reference");

    // Every section with children must actually have at least one child.
    const sections = NAV_TREE.filter((i) => i.children);
    expect(sections.length).toBeGreaterThan(0);
    for (const section of sections) {
      expect(section.children!.length).toBeGreaterThan(0);
    }
  });

  it("every_node_has_title_and_href_or_children", () => {
    walk(NAV_TREE, (item) => {
      expect(item.title, "title must be non-empty").toBeTruthy();
      const hasHref = typeof item.href === "string" && item.href.length > 0;
      const hasChildren = Array.isArray(item.children) && item.children.length > 0;
      expect(
        hasHref || hasChildren,
        `nav item "${item.title}" must have either href or children`,
      ).toBe(true);
    });
  });

  it("every_href_is_rooted_at_a_known_prefix", () => {
    const allowed = ["/", "/docs", "/crates", "/extras", "/deno", "/changelog", "/api-docs"];
    walk(NAV_TREE, (item) => {
      if (!item.href) return;
      const ok = allowed.some(
        (p) => item.href === p || item.href!.startsWith(p + "/"),
      );
      expect(ok, `href "${item.href}" not under known prefix`).toBe(true);
    });
  });

  it("guide_hrefs_are_backed_by_DOC_SLUG_MAP", () => {
    const guides = NAV_TREE.find((i) => i.title === "Guides");
    expect(guides).toBeDefined();
    expect(guides!.children).toBeDefined();
    for (const guide of guides!.children!) {
      const slug = guide.href!.replace(/^\/docs\//, "");
      expect(
        Object.keys(DOC_SLUG_MAP),
        `guide href "${guide.href}" has no backing slug`,
      ).toContain(slug);
    }
  });

  it("deno_sdk_hrefs_are_backed_by_DENO_SLUG_MAP", () => {
    const deno = NAV_TREE.find((i) => i.title === "Deno SDK");
    expect(deno).toBeDefined();
    for (const item of deno!.children!) {
      const slug = item.href!.replace(/^\/deno\//, "");
      expect(
        Object.keys(DENO_SLUG_MAP),
        `deno href "${item.href}" has no backing slug`,
      ).toContain(slug);
    }
  });
});
