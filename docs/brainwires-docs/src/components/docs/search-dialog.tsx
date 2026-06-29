"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Fuse, { type IFuseOptions } from "fuse.js";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { FileText, Package, Box } from "lucide-react";

/**
 * One row in the build-time search index (see `src/lib/search.ts`).
 * Kept loose here so we don't need to re-import the builder on the client.
 */
export interface IndexEntry {
  href: string;
  title: string;
  section: string;
  body: string;
}

/**
 * Fuse options shared between the runtime dialog and the unit tests, so a
 * test failure maps cleanly to a production regression.
 */
export const FUSE_OPTIONS: IFuseOptions<IndexEntry> = {
  keys: [
    { name: "title", weight: 0.7 },
    { name: "body", weight: 0.3 },
  ],
  threshold: 0.35,
  includeMatches: false,
  minMatchCharLength: 2,
  ignoreLocation: true,
};

type IndexState =
  | { kind: "loading" }
  | { kind: "ready"; fuse: Fuse<IndexEntry>; entries: IndexEntry[] }
  | { kind: "error" };

function iconFor(href: string) {
  if (href.startsWith("/crates/")) return Package;
  if (href.startsWith("/extras/")) return Box;
  return FileText;
}

export function SearchDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const router = useRouter();
  const instanceId = useId();
  const [state, setState] = useState<IndexState>({ kind: "loading" });
  const [query, setQuery] = useState("");
  const fetchedRef = useRef(false);

  // Fetch the build-time index once, the first time the dialog opens.
  useEffect(() => {
    if (!open || fetchedRef.current) return;
    fetchedRef.current = true;

    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/search-index.json", { cache: "force-cache" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const entries = (await res.json()) as IndexEntry[];
        if (cancelled) return;
        const fuse = new Fuse(entries, FUSE_OPTIONS);
        setState({ kind: "ready", fuse, entries });
      } catch {
        if (!cancelled) setState({ kind: "error" });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [open]);

  // Global Cmd/Ctrl+K binding. Preserved from the previous implementation.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        onOpenChange(true);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onOpenChange]);

  // Reset the query whenever the dialog closes so reopening is a fresh
  // search rather than reviving the previous view.
  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);

  const results = useMemo(() => {
    if (state.kind !== "ready") return [];
    const q = query.trim();
    if (q.length < 2) return [];
    return state.fuse.search(q, { limit: 20 }).map((hit) => hit.item);
  }, [state, query]);

  function navigate(href: string) {
    onOpenChange(false);
    router.push(href);
  }

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput
        placeholder="Search documentation…"
        aria-label="Search documentation"
        value={query}
        onValueChange={setQuery}
      />
      <CommandList id={`${instanceId}-list`} role="listbox">
        {state.kind === "loading" && (
          <div className="py-6 text-center text-sm text-muted-foreground">
            Loading…
          </div>
        )}
        {state.kind === "error" && (
          <div className="py-6 text-center text-sm text-muted-foreground">
            Search unavailable
          </div>
        )}
        {state.kind === "ready" && query.trim().length < 2 && (
          <div className="py-6 text-center text-sm text-muted-foreground">
            Start typing to search…
          </div>
        )}
        {state.kind === "ready" && query.trim().length >= 2 && (
          <>
            <CommandEmpty>No results found.</CommandEmpty>
            {results.length > 0 && (
              <CommandGroup heading="Results">
                {results.map((entry) => {
                  const Icon = iconFor(entry.href);
                  const optionId = `${instanceId}-opt-${entry.href}`;
                  return (
                    <CommandItem
                      key={entry.href}
                      id={optionId}
                      role="option"
                      value={`${entry.title} ${entry.section} ${entry.href}`}
                      onSelect={() => navigate(entry.href)}
                      className="gap-2"
                    >
                      <Icon className="size-4 shrink-0 text-muted-foreground" />
                      <span>{entry.title}</span>
                      <span className="ml-auto text-xs text-muted-foreground">
                        {entry.section}
                      </span>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            )}
          </>
        )}
      </CommandList>
    </CommandDialog>
  );
}
