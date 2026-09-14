// Convert Deno's lcov coverage into an Istanbul `coverage-final.json` for fallow.
//
//   deno task coverage:fallow      # unit suite → lcov → coverage/coverage-final.json
//   deno run -A scripts/coverage-to-istanbul.ts <in.lcov> <out.json> [projectRoot]
//
// Why: fallow's CRAP score (complexity × untested-ness) needs per-function coverage.
// Without a coverage file it ESTIMATES coverage from the import graph — a module that
// tests reach only transitively (the fiber reconciler, driven through `createRoot()`)
// is scored at a pessimistic 40% tier, which makes CRAP fire on any function with
// cyclomatic ≥ 10 regardless of how well-tested it really is. fallow accepts only the
// Istanbul JSON map (not lcov, not raw V8), so this script bridges the two.
//
// `deno coverage --lcov` is already source-mapped to TypeScript lines, so the line
// numbers here line up with the source fallow parses. The Istanbul map is built from
// the lcov records directly: one statement per `DA` line, one function per `FN`
// (spanning to the next function's start), and one branch group per `BRDA` line/block.

type Loc = { line: number; column: number };
type Range = { start: Loc; end: Loc };

/** One file's entry in an Istanbul coverage map. */
interface IstanbulFileCoverage {
  path: string;
  statementMap: Record<string, Range>;
  fnMap: Record<
    string,
    { name: string; decl: Range; loc: Range; line: number }
  >;
  branchMap: Record<
    string,
    { loc: Range; type: string; locations: Range[]; line: number }
  >;
  s: Record<string, number>;
  f: Record<string, number>;
  b: Record<string, number[]>;
}

/** Istanbul coverage map: absolute file path → file coverage. */
type IstanbulCoverageMap = Record<string, IstanbulFileCoverage>;

interface LcovRecord {
  path: string;
  fns: { name: string; line: number; count: number }[];
  lines: [line: number, count: number][];
  branches: Map<string, { line: number; counts: number[] }>;
}

const lineRange = (from: number, to: number): Range => ({
  start: { line: from, column: 0 },
  end: { line: to, column: 10_000 },
});

function newRecord(path: string): LcovRecord {
  return { path, fns: [], lines: [], branches: new Map() };
}

/** Per-prefix lcov line handlers; `rest` is the text after the `PREFIX:`. */
const LCOV_HANDLERS: Record<string, (rec: LcovRecord, rest: string) => void> = {
  "FN:": (rec, rest) => {
    const [l, ...name] = rest.split(",");
    rec.fns.push({ name: name.join(","), line: Number(l), count: 0 });
  },
  "FNDA:": (rec, rest) => {
    const [c, ...name] = rest.split(",");
    const fn = rec.fns.find((f) => f.name === name.join(","));
    if (fn) fn.count = Number(c);
  },
  "DA:": (rec, rest) => {
    const [l, c] = rest.split(",");
    rec.lines.push([Number(l), Number(c)]);
  },
  "BRDA:": (rec, rest) => {
    const [l, block, _branch, c] = rest.split(",");
    const key = `${l}:${block}`;
    const entry = rec.branches.get(key) ?? { line: Number(l), counts: [] };
    entry.counts.push(c === "-" ? 0 : Number(c));
    rec.branches.set(key, entry);
  },
};

/** Feed one lcov line into the record being built; returns true at `end_of_record`. */
function readLcovLine(rec: LcovRecord, line: string): boolean {
  const prefix = Object.keys(LCOV_HANDLERS).find((p) => line.startsWith(p));
  if (prefix) LCOV_HANDLERS[prefix](rec, line.slice(prefix.length));
  return line === "end_of_record";
}

function toIstanbul(rec: LcovRecord): IstanbulFileCoverage {
  const out: IstanbulFileCoverage = {
    path: rec.path,
    statementMap: {},
    fnMap: {},
    branchMap: {},
    s: {},
    f: {},
    b: {},
  };
  const lines = [...rec.lines].sort((a, b) => a[0] - b[0]);
  const lastLine = lines.length > 0 ? lines[lines.length - 1][0] : 1;
  lines.forEach(([line, count], i) => {
    out.statementMap[i] = lineRange(line, line);
    out.s[i] = count;
  });
  const fns = [...rec.fns].sort((a, b) => a.line - b.line);
  fns.forEach((fn, i) => {
    const endLine = i + 1 < fns.length ? fns[i + 1].line - 1 : lastLine;
    out.fnMap[i] = {
      name: fn.name,
      decl: lineRange(fn.line, fn.line),
      loc: lineRange(fn.line, Math.max(fn.line, endLine)),
      line: fn.line,
    };
    out.f[i] = fn.count;
  });
  let bi = 0;
  for (const { line, counts } of rec.branches.values()) {
    out.branchMap[bi] = {
      loc: lineRange(line, line),
      type: "branch",
      locations: counts.map(() => lineRange(line, line)),
      line,
    };
    out.b[bi] = counts;
    bi++;
  }
  return out;
}

/** Resolve an `SF:` path: lcov from `deno coverage` is already absolute. */
function sourcePath(p: string, root: string): string {
  return p.startsWith("/") ? p : `${root.replace(/\/$/, "")}/${p}`;
}

/** Advance the parser by one lcov line; returns the record still being built (if any). */
function feedLine(
  map: IstanbulCoverageMap,
  rec: LcovRecord | null,
  line: string,
  root: string,
): LcovRecord | null {
  if (line.startsWith("SF:")) return newRecord(sourcePath(line.slice(3), root));
  if (rec === null) return null;
  if (!readLcovLine(rec, line)) return rec;
  map[rec.path] = toIstanbul(rec);
  return null;
}

/**
 * Convert lcov text to an Istanbul coverage map. Relative `SF:` paths are resolved
 * against `root`.
 */
function lcovToIstanbul(lcov: string, root: string): IstanbulCoverageMap {
  const map: IstanbulCoverageMap = {};
  let rec: LcovRecord | null = null;
  for (const raw of lcov.split("\n")) {
    rec = feedLine(map, rec, raw.trim(), root);
  }
  return map;
}

if (import.meta.main) {
  const [inPath, outPath, rootArg] = Deno.args;
  if (!inPath || !outPath) {
    console.error(
      "usage: coverage-to-istanbul.ts <in.lcov> <out.json> [projectRoot]",
    );
    Deno.exit(2);
  }
  const map = lcovToIstanbul(
    await Deno.readTextFile(inPath),
    rootArg ?? Deno.cwd(),
  );
  await Deno.writeTextFile(outPath, JSON.stringify(map));
  console.log(
    `coverage-to-istanbul: wrote ${
      Object.keys(map).length
    } files to ${outPath}`,
  );
}
