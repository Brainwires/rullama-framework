import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { CRATE_LIST, TIER_LABELS, TIER_COLORS } from "@/lib/crates";
import { getCrateReadme } from "@/lib/docs";
import { MdxContent } from "@/components/docs/mdx-content";
import { Toc } from "@/components/layout/toc";
import { ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";

export const dynamicParams = false;

interface Props { params: Promise<{ crate: string }>; }

export async function generateStaticParams() { return CRATE_LIST.map((c) => ({ crate: c.name })); }

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { crate } = await params; return { title: crate };
}

export default async function CratePage({ params }: Props) {
  const { crate: crateName } = await params;
  const meta = CRATE_LIST.find((c) => c.name === crateName);
  if (!meta) notFound();
  const content = getCrateReadme(crateName);
  if (!content) notFound();
  return (
    <div className="flex gap-8 px-6 py-8 max-w-[1400px] mx-auto w-full">
      <article className="min-w-0 flex-1">
        <div className="mb-6 flex flex-wrap items-center gap-3">
          <span className={cn("inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium", TIER_COLORS[meta.tier])}>
            {TIER_LABELS[meta.tier]}
          </span>
          <a href={`https://crates.io/crates/${crateName}`} target="_blank" rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
            crates.io <ExternalLink className="size-3" />
          </a>
          <a href={`https://docs.rs/${crateName}`} target="_blank" rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
            docs.rs <ExternalLink className="size-3" />
          </a>
        </div>
        <MdxContent content={content} />
      </article>
      <Toc markdown={content} />
    </div>
  );
}
