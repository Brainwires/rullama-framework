/**
 * File Context Manager
 *
 * Manages file content for context injection with smart chunking for large
 * files. Prevents re-injection of files already in context and retrieves
 * relevant portions of large files based on a query.
 *
 * Equivalent to Rust's `rullama_core::file_context`.
 */

const MAX_DIRECT_FILE_CHARS = 50_000;
const LARGE_FILE_CHUNK_SIZE = 10_000;
const MAX_FILE_CHUNKS = 5;

/** A chunk of file content with line range + relevance score. */
export interface FileChunk {
  content: string;
  /** 1-indexed starting line. */
  line_start: number;
  /** 1-indexed ending line (inclusive). */
  line_end: number;
  /** Relevance score in [0, 1]. */
  relevance_score: number;
}

/** Content returned from `FileContextManager.getFileContent`. */
export type FileContent =
  | { kind: "full"; content: string }
  | {
    kind: "chunked";
    path: string;
    total_size: number;
    chunks: FileChunk[];
    has_more: boolean;
  }
  | { kind: "already_in_context"; path: string };

async function sha256Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Manages file content for context injection. */
export class FileContextManager {
  #contextFiles: Set<string> = new Set();
  #fileChunks: Map<string, FileChunk[]> = new Map();

  /** SHA256 hex digest of content. */
  static computeHash(content: string): Promise<string> {
    return sha256Hex(content);
  }

  /** Is the file already in the current context? */
  isInContext(path: string): boolean {
    return this.#contextFiles.has(path);
  }

  /** Mark a file as in-context (prevents re-injection). */
  markInContext(path: string): void {
    this.#contextFiles.add(path);
  }

  /** Drop all in-context markers (use at the start of a new turn). */
  clearContext(): void {
    this.#contextFiles.clear();
  }

  /** Count files currently marked in-context. */
  contextFileCount(): number {
    return this.#contextFiles.size;
  }

  /**
   * Fetch file content with smart routing based on size.
   *
   * - `full` for small files (≤ 50 000 chars)
   * - `chunked` for large files (returns up to 5 relevant chunks)
   * - `already_in_context` if previously loaded
   */
  async getFileContent(
    path: string,
    queryContext?: string,
  ): Promise<FileContent> {
    if (this.isInContext(path)) {
      return { kind: "already_in_context", path };
    }

    let content: string;
    try {
      content = await Deno.readTextFile(path);
    } catch (e) {
      throw new Error(`Failed to read file: ${path}: ${(e as Error).message}`);
    }

    if (content.length <= MAX_DIRECT_FILE_CHARS) {
      this.markInContext(path);
      return { kind: "full", content };
    }

    const chunks = this.#getRelevantChunks(path, content, queryContext);
    this.markInContext(path);
    return {
      kind: "chunked",
      path,
      total_size: content.length,
      chunks,
      has_more: content.length > MAX_DIRECT_FILE_CHARS,
    };
  }

  /** Fetch a specific 1-indexed line range. */
  async getFileLines(
    path: string,
    startLine: number,
    endLine: number,
  ): Promise<FileContent> {
    let content: string;
    try {
      content = await Deno.readTextFile(path);
    } catch (e) {
      throw new Error(`Failed to read file: ${path}: ${(e as Error).message}`);
    }

    const lines = content.split("\n");
    const total = lines.length;
    const start = Math.min(Math.max(startLine - 1, 0), total);
    const end = Math.min(endLine, total);

    if (start >= end) return { kind: "full", content: "" };

    const selected = lines.slice(start, end).join("\n");
    this.markInContext(path);

    if (selected.length <= MAX_DIRECT_FILE_CHARS) {
      return { kind: "full", content: selected };
    }

    return {
      kind: "chunked",
      path,
      total_size: content.length,
      chunks: [{
        content: selected,
        line_start: start + 1,
        line_end: end,
        relevance_score: 1.0,
      }],
      has_more: true,
    };
  }

  #getRelevantChunks(
    path: string,
    content: string,
    queryContext: string | undefined,
  ): FileChunk[] {
    const all = this.buildFileChunks(content);
    this.#fileChunks.set(path, all);

    if (queryContext !== undefined && queryContext.length > 0) {
      const relevant = this.findRelevantChunks(all, queryContext);
      if (relevant.length > 0) return relevant;
    }
    return all.slice(0, MAX_FILE_CHUNKS);
  }

  /** Build line-aligned chunks of size ≤ LARGE_FILE_CHUNK_SIZE. */
  buildFileChunks(content: string): FileChunk[] {
    const lines = content.split("\n");
    const chunks: FileChunk[] = [];
    let i = 0;
    while (i < lines.length) {
      const startLine = i + 1;
      let buf = "";
      while (i < lines.length && buf.length < LARGE_FILE_CHUNK_SIZE) {
        if (buf.length > 0) buf += "\n";
        buf += lines[i];
        i += 1;
      }
      if (buf.length > 0) {
        chunks.push({
          content: buf,
          line_start: startLine,
          line_end: i,
          relevance_score: 1.0,
        });
      }
    }
    return chunks;
  }

  /** Rank chunks by simple word-overlap with the query. */
  findRelevantChunks(chunks: FileChunk[], query: string): FileChunk[] {
    const words = query.toLowerCase().split(/\s+/).filter((w) => w.length > 0);
    if (words.length === 0) return [];

    const scored: { chunk: FileChunk; score: number }[] = [];
    for (const chunk of chunks) {
      const lower = chunk.content.toLowerCase();
      const matching = words.filter((w) => lower.includes(w)).length;
      if (matching > 0) {
        const score = matching / words.length;
        scored.push({
          chunk: {
            content: chunk.content,
            line_start: chunk.line_start,
            line_end: chunk.line_end,
            relevance_score: score,
          },
          score,
        });
      }
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, MAX_FILE_CHUNKS).map((s) => s.chunk);
  }

  /** Format `FileContent` for display in an LLM context. */
  static formatContent(file: FileContent): string {
    switch (file.kind) {
      case "full":
        return file.content;
      case "already_in_context":
        return `[File ${file.path} is already shown above]`;
      case "chunked": {
        let out =
          `[File: ${file.path} | Size: ${file.total_size} chars | Showing ${file.chunks.length} relevant sections]\n\n`;
        for (const c of file.chunks) {
          out += `--- Lines ${c.line_start}-${c.line_end} (relevance: ${
            c.relevance_score.toFixed(2)
          }) ---\n${c.content}\n\n`;
        }
        if (file.has_more) {
          out +=
            "[... more content available, ask for specific sections or line numbers ...]\n";
        }
        return out;
      }
    }
  }
}
