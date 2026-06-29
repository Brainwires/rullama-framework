import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { DOC_SLUG_MAP, getAllDocSlugs, readMarkdownFile } from "@/lib/docs";
import { MdxContent } from "@/components/docs/mdx-content";
import { Toc } from "@/components/layout/toc";

export const dynamicParams = false;

interface Props { params: Promise<{ slug: string[] }>; }

export async function generateStaticParams() { return getAllDocSlugs(); }

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const title = slug.join("/").split("/").pop()?.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) ?? "Docs";
  return { title };
}

export default async function DocPage({ params }: Props) {
  const { slug } = await params;
  const filePath = DOC_SLUG_MAP[slug.join("/")];
  if (!filePath) notFound();
  const content = readMarkdownFile(filePath);
  if (!content) notFound();
  return (
    <div className="flex gap-8 px-6 py-8 max-w-[1400px] mx-auto w-full">
      <article className="min-w-0 flex-1"><MdxContent content={content} /></article>
      <Toc markdown={content} />
    </div>
  );
}
