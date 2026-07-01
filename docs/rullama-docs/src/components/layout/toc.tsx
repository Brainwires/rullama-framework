"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface Heading { id: string; text: string; level: number; }

function extractHeadings(markdown: string): Heading[] {
  const headings: Heading[] = [];
  for (const line of markdown.split("\n")) {
    const match = line.match(/^(#{2,3})\s+(.+)/);
    if (match) {
      const text = match[2].replace(/`[^`]+`/g, (m) => m.slice(1, -1));
      const id = text.toLowerCase().replace(/[^\w\s-]/g, "").trim().replace(/\s+/g, "-");
      headings.push({ id, text, level: match[1].length });
    }
  }
  return headings;
}

export function Toc({ markdown }: { markdown: string }) {
  const headings = extractHeadings(markdown);
  const [active, setActive] = useState<string>("");

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => { for (const e of entries) if (e.isIntersecting) setActive(e.target.id); },
      { rootMargin: "0px 0px -70% 0px" }
    );
    document.querySelectorAll("h2[id], h3[id]").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [markdown]);

  if (headings.length === 0) return null;

  return (
    <nav className="hidden xl:block sticky top-14 h-[calc(100vh-3.5rem)] w-56 shrink-0 overflow-y-auto py-6 pr-2 text-sm">
      <p className="text-muted-foreground mb-3 font-medium text-xs uppercase tracking-wider">On this page</p>
      <ul className="space-y-1">
        {headings.map((h) => (
          <li key={h.id} style={{ paddingLeft: h.level === 3 ? "0.75rem" : "0" }}>
            <a
              href={`#${h.id}`}
              className={cn(
                "block truncate rounded py-0.5 transition-colors hover:text-foreground",
                active === h.id ? "text-foreground font-medium" : "text-muted-foreground"
              )}
            >
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
