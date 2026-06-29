import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { EXTRAS_LIST } from "@/lib/crates";
import { getExtraReadme } from "@/lib/docs";
import { MdxContent } from "@/components/docs/mdx-content";
import { Toc } from "@/components/layout/toc";

export const dynamicParams = false;

interface Props { params: Promise<{ extra: string }>; }

export async function generateStaticParams() {
  return EXTRAS_LIST.filter((e) => e.hasReadme).map((e) => ({ extra: e.name }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { extra } = await params; return { title: extra };
}

export default async function ExtraPage({ params }: Props) {
  const { extra: extraName } = await params;
  const meta = EXTRAS_LIST.find((e) => e.name === extraName);
  if (!meta || !meta.hasReadme) notFound();
  const content = getExtraReadme(extraName);
  if (!content) notFound();
  return (
    <div className="flex gap-8 px-6 py-8 max-w-[1400px] mx-auto w-full">
      <article className="min-w-0 flex-1"><MdxContent content={content} /></article>
      <Toc markdown={content} />
    </div>
  );
}
