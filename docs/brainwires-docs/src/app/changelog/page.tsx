import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { readMarkdownFile } from "@/lib/docs";
import { MdxContent } from "@/components/docs/mdx-content";
import { Toc } from "@/components/layout/toc";

export const metadata: Metadata = { title: "Changelog" };

export default function ChangelogPage() {
  const content = readMarkdownFile("CHANGELOG.md");
  if (!content) notFound();
  return (
    <div className="flex gap-8 px-6 py-8 max-w-[1400px] mx-auto w-full">
      <article className="min-w-0 flex-1"><MdxContent content={content} /></article>
      <Toc markdown={content} />
    </div>
  );
}
