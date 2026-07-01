import { MDXRemote } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import rehypePrettyCode from "rehype-pretty-code";
import rehypeSlug from "rehype-slug";
import { ExternalLink } from "lucide-react";
import type { MDXComponents } from "mdx/types";

const components: MDXComponents = {
  a: ({ href, children, ...props }) => {
    const isExternal = href?.startsWith("http");
    return (
      <a
        href={href}
        target={isExternal ? "_blank" : undefined}
        rel={isExternal ? "noopener noreferrer" : undefined}
        {...props}
      >
        {children}
        {isExternal && <ExternalLink className="inline size-3 shrink-0 ml-0.5 -translate-y-px" />}
      </a>
    );
  },
};

export function MdxContent({ content }: { content: string }) {
  return (
    <div className="prose prose-neutral dark:prose-invert max-w-none prose-pre:bg-transparent prose-pre:p-0 prose-code:before:content-none prose-code:after:content-none">
      <MDXRemote
        source={content}
        options={{
          mdxOptions: {
            format: "md",
            remarkPlugins: [remarkGfm],
            rehypePlugins: [
              rehypeSlug,
              [rehypePrettyCode, {
                theme: { dark: "github-dark", light: "github-light" },
                defaultLang: "text",
              }],
            ],
          },
        }}
        components={components}
      />
    </div>
  );
}
