#!/usr/bin/env node
/**
 * Build-time helper: writes public/search-index.json.
 *
 * Runs before `next build` (wired via the "build" script in package.json)
 * and can also be invoked directly via `npm run build:search`.
 *
 * Node 22+ strips TypeScript types natively when the imported file has a
 * `.ts` extension, so we can import `../src/lib/search.ts` directly without
 * a compile step. This relies on the `moduleResolution: "bundler"` tsconfig
 * option permitting `.ts` specifiers at author time.
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

// Suppress the noisy MODULE_TYPELESS_PACKAGE_JSON warning that Node emits on
// every loaded .ts file when the nearest package.json has no `"type"` field.
// Adding `"type": "module"` at the Next.js app level breaks its CJS config
// files, so we swallow the warning instead.
const originalEmit = process.emit;
process.emit = function patchedEmit(event, arg, ...rest) {
  if (
    event === "warning" &&
    arg &&
    arg.code === "MODULE_TYPELESS_PACKAGE_JSON"
  ) {
    return false;
  }
  return originalEmit.call(this, event, arg, ...rest);
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");
const outPath = path.join(root, "public", "search-index.json");

const { writeSearchIndex } = await import(
  path.join(root, "src", "lib", "search.ts")
);

await writeSearchIndex(outPath);
console.log(`[search] wrote ${outPath}`);
