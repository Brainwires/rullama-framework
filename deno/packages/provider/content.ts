/**
 * Shared shaping of parsed response content: the core convention is that a
 * response consisting of one text block is a bare string, an empty response
 * is `""`, and anything else is the block list.
 *
 * @module
 */

import type { ContentBlock, MessageContent } from "@rullama/core";

/** Collapse parsed blocks to the core `MessageContent` convention. */
export function collapseBlocks(blocks: ContentBlock[]): MessageContent {
  if (blocks.length === 0) return "";
  if (blocks.length === 1 && blocks[0].type === "text") return blocks[0].text;
  return blocks;
}

/** Map vendor parts to core blocks (dropping `null`s) and collapse the result. */
export function mapBlocks<T>(
  items: T[],
  toBlock: (item: T) => ContentBlock | null,
): MessageContent {
  return collapseBlocks(
    items.map(toBlock).filter((b): b is ContentBlock => b !== null),
  );
}

/** A core text block. */
export function textBlock(text: string): ContentBlock {
  return { type: "text", text };
}

/** A core tool-use block; missing vendor fields default to empty values. */
export function toolUseBlock(
  id: string | undefined,
  name: string | undefined,
  input: unknown,
): ContentBlock {
  return {
    type: "tool_use",
    id: id ?? "",
    name: name ?? "",
    input: input ?? {},
  };
}
