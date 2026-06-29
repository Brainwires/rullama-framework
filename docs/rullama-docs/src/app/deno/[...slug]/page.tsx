import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { DENO_SLUG_MAP, getAllDenoSlugs, readMarkdownFile } from "@/lib/docs";
import { MdxContent } from "@/components/docs/mdx-content";
import { Toc } from "@/components/layout/toc";

export const dynamicParams = false;

interface Props { params: Promise<{ slug: string[] }>; }

export async function generateStaticParams() { return getAllDenoSlugs(); }

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  return { title: slug.join("/").replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) + " — Deno SDK" };
}

export default async function DenoDocPage({ params }: Props) {
  const { slug } = await params;
  const filePath = DENO_SLUG_MAP[slug.join("/")];
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
