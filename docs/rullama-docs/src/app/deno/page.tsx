import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { readMarkdownFile } from "@/lib/docs";
import { MdxContent } from "@/components/docs/mdx-content";
import { Toc } from "@/components/layout/toc";

export const metadata: Metadata = { title: "Deno SDK" };

export default function DenoIndexPage() {
  const content = readMarkdownFile("deno/docs/README.md");
  if (!content) notFound();
  return (
    <div className="flex gap-8 px-6 py-8 max-w-[1400px] mx-auto w-full">
      <article className="min-w-0 flex-1"><MdxContent content={content} /></article>
      <Toc markdown={content} />
    </div>
  );
}
