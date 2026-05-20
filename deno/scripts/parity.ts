/**
 * Parity-check script: diff Rust crates/ against Deno packages/.
 *
 * Run from the deno/ directory:
 *   deno run --allow-read scripts/parity.ts
 *
 * Prints a markdown table and exits non-zero if there are unexpected gaps
 * (a Rust crate with no Deno package AND not on the intentional-skip list).
 */

const RUST_CRATES = "../crates";
const DENO_PACKAGES = "./packages";

/** Crates we intentionally leave on the runtime boundary. Keep in sync with docs/parity.md. */
const RUST_ONLY = new Set([
  "brainwires",                 // meta-crate, no deno equivalent
  "brainwires-hardware",
  "brainwires-sandbox",
  "brainwires-sandbox-proxy",
]);

/** Crates that live inside another Deno package (folded rather than 1:1). */
const FOLDED: Record<string, string> = {
  "brainwires-mcp-server": "network",
};

/** Rust crate name → Deno package name. */
function depunctuate(name: string): string {
  return name.replace(/^brainwires-/, "");
}

async function listDirs(path: string): Promise<string[]> {
  const out: string[] = [];
  for await (const e of Deno.readDir(path)) {
    if (e.isDirectory) out.push(e.name);
  }
  out.sort();
  return out;
}

function main() {
  return (async () => {
    const crates = await listDirs(RUST_CRATES);
    const packages = new Set(await listDirs(DENO_PACKAGES));

    const rows: Array<{ crate: string; pkg: string; status: string }> = [];
    let unexpected = 0;

    for (const crate of crates) {
      if (RUST_ONLY.has(crate)) {
        rows.push({ crate, pkg: "—", status: "runtime-boundary (intentional)" });
        continue;
      }
      if (crate in FOLDED) {
        rows.push({ crate, pkg: FOLDED[crate], status: "folded" });
        continue;
      }
      const expected = depunctuate(crate);
      if (packages.has(expected)) {
        rows.push({ crate, pkg: `@brainwires/${expected}`, status: "ok" });
      } else {
        rows.push({ crate, pkg: "(missing)", status: "UNEXPECTED GAP" });
        unexpected += 1;
      }
    }

    // Print table.
    console.log("| Rust crate | Deno package | Status |");
    console.log("|---|---|---|");
    for (const r of rows) {
      console.log(`| \`${r.crate}\` | ${r.pkg === "—" ? "—" : `\`${r.pkg}\``} | ${r.status} |`);
    }
    console.log("");
    console.log(`Total crates: ${rows.length}`);
    console.log(`Unexpected gaps: ${unexpected}`);
    if (unexpected > 0) Deno.exit(1);
  })();
}

await main();
