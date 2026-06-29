import type { Metadata } from "next";
import { ExternalLink, BookOpen } from "lucide-react";
import fs from "fs";
import path from "path";

export const metadata: Metadata = { title: "API Reference" };

function hasLocalRustdoc(): boolean {
  try {
    return fs.existsSync(path.join(process.cwd(), "public", "rustdoc", "brainwires", "index.html"));
  } catch { return false; }
}

export default function ApiPage() {
  const hasLocal = hasLocalRustdoc();
  return (
    <div className="px-6 py-8 max-w-[1400px] mx-auto w-full">
      <h1 className="text-3xl font-bold mb-2">API Reference</h1>
      <p className="text-muted-foreground mb-8">Auto-generated Rust API documentation from doc comments.</p>
      {hasLocal ? (
        <>
          <p className="text-sm text-muted-foreground mb-4">
            Serving locally-built rustdoc (generated via{" "}
            <code className="font-mono text-xs bg-muted px-1 rounded">cargo xtask doc</code>).
          </p>
          <iframe src="/rustdoc/brainwires/index.html" className="w-full rounded-lg border"
            style={{ height: "calc(100vh - 16rem)" }} title="Rust API Reference" />
        </>
      ) : (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed py-20 text-center gap-6">
          <BookOpen className="size-12 text-muted-foreground/50" />
          <div>
            <p className="text-lg font-medium mb-1">API docs are on docs.rs</p>
            <p className="text-muted-foreground text-sm max-w-md">
              Full Rustdoc is available on <strong>docs.rs</strong> for all published crates.
              In the Docker production build, rustdoc is served locally via{" "}
              <code className="font-mono text-xs bg-muted px-1 rounded">cargo doc --workspace</code>.
            </p>
          </div>
          <div className="flex flex-wrap justify-center gap-3">
            <a href="https://docs.rs/brainwires" target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/80 transition-colors">
              docs.rs/brainwires <ExternalLink className="size-3.5" />
            </a>
            <a href="https://crates.io/crates/brainwires" target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-1.5 text-sm font-medium hover:bg-muted transition-colors">
              crates.io <ExternalLink className="size-3.5" />
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
