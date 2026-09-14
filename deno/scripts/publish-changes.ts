// Publish only the packages that changed — by pushing their `<package>-v<version>` tags.
//
//   deno task publish:changes            # report: what would be tagged, what needs a bump
//   deno task publish:changes --push     # create + push the tags (each triggers publish-deno.yml)
//   deno task publish:changes --push --no-wait   # don't wait for each workflow run
//   deno task publish:changes --bump patch      # bump changed-but-unbumped packages, then stop
//
// A package "needs publishing" when the version in its deno.json is not on JSR yet.
// A package "needs a bump" when its files changed since the commit of its last
// `<package>-v*` tag (or since the JSR-published version was tagged) but the version
// is unchanged — that is reported and, with --bump, fixed in deno.json (commit it,
// then run again).
//
// Tags are pushed in dependency order (from each package's `@rullama/*` imports)
// and, unless --no-wait, the script waits for each publish workflow run to succeed
// before pushing the next tag: a dependent published before its dependency lands
// on JSR is rejected by JSR.
//
// --push is resumable. A tag already on origin is not re-created: its last publish
// run is re-run instead. A package that fails to publish does not stop the others;
// only the packages that depend on it are skipped, and the script exits 1 with the
// list, so fixing the cause and running --push again finishes the release.

import { dirname, fromFileUrl, join } from "@std/path";

const ROOT = join(dirname(fromFileUrl(import.meta.url)), "..");
const SCOPE = "rullama";

type Pkg = { name: string; version: string; deps: string[] };

function run(cmd: string, args: string[], cwd = ROOT): string {
  const out = new Deno.Command(cmd, {
    args,
    cwd,
    stdout: "piped",
    stderr: "piped",
  }).outputSync();
  if (!out.success) {
    throw new Error(
      `${cmd} ${args.join(" ")} failed: ${
        new TextDecoder().decode(out.stderr)
      }`,
    );
  }
  return new TextDecoder().decode(out.stdout).trim();
}

/** Non-test TypeScript sources under `dir`. */
function sourceFiles(dir: string): string[] {
  return [...walk(dir)].filter((f) =>
    f.endsWith(".ts") && !f.endsWith("_test.ts")
  );
}

/** Sibling `@rullama/*` package names imported by `text`, excluding `self`. */
function importedSiblings(text: string, self: string): string[] {
  return [...text.matchAll(/from "@rullama\/([a-z-]+)/g)]
    .map((m) => m[1])
    .filter((name) => name !== self);
}

/** Sibling `@rullama/*` packages imported by the non-test sources under `dir`. */
function depsOf(dir: string, self: string): string[] {
  const deps = new Set(
    sourceFiles(dir).flatMap((f) =>
      importedSiblings(Deno.readTextFileSync(f), self)
    ),
  );
  return [...deps].sort();
}

function readPackage(name: string): Pkg {
  const dir = join(ROOT, "packages", name);
  const { version } = JSON.parse(Deno.readTextFileSync(join(dir, "deno.json")));
  return { name, version, deps: depsOf(dir, name) };
}

function loadPackages(): Pkg[] {
  return [...Deno.readDirSync(join(ROOT, "packages"))]
    .filter((e) => e.isDirectory)
    .map((e) => readPackage(e.name))
    .sort((a, b) => a.name.localeCompare(b.name));
}

function* walk(dir: string): Generator<string> {
  for (const e of Deno.readDirSync(dir)) {
    const p = join(dir, e.name);
    if (e.isDirectory) yield* walk(p);
    else yield p;
  }
}

/** Dependencies before dependents (Kahn's algorithm; names sorted for determinism). */
function topoOrder(pkgs: Pkg[]): string[] {
  const byName = new Map(pkgs.map((p) => [p.name, p]));
  const done = new Set<string>();
  const order: string[] = [];
  while (order.length < pkgs.length) {
    const ready = pkgs.filter((p) =>
      !done.has(p.name) && p.deps.every((d) => done.has(d) || !byName.has(d))
    );
    if (ready.length === 0) throw new Error("dependency cycle among packages");
    for (const p of ready) {
      done.add(p.name);
      order.push(p.name);
    }
  }
  return order;
}

async function jsrLatest(name: string): Promise<string | null> {
  const res = await fetch(
    `https://jsr.io/api/scopes/${SCOPE}/packages/${name}`,
    {
      signal: AbortSignal.timeout(15_000),
    },
  );
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`jsr.io ${name}: HTTP ${res.status}`);
  return (await res.json() as { latestVersion: string | null }).latestVersion;
}

async function jsrHasVersion(name: string, version: string): Promise<boolean> {
  const res = await fetch(
    `https://jsr.io/api/scopes/${SCOPE}/packages/${name}/versions/${version}`,
    {
      signal: AbortSignal.timeout(15_000),
    },
  );
  return res.ok;
}

/** Last `<name>-v*` tag, or `deno-v<published>` for packages tagged in lockstep, or null. */
function lastReleaseTag(name: string, published: string | null): string | null {
  const own = run("git", ["tag", "-l", `${name}-v*`, "--sort=-v:refname"])
    .split("\n").filter(Boolean);
  if (own[0]) return own[0];
  if (published && run("git", ["tag", "-l", `deno-v${published}`])) {
    return `deno-v${published}`;
  }
  return null;
}

function changedSince(tag: string, name: string): boolean {
  const out = new Deno.Command("git", {
    args: ["diff", "--quiet", tag, "--", `deno/packages/${name}`],
    cwd: join(ROOT, ".."),
  }).outputSync();
  return out.code === 1;
}

function bump(version: string, kind: "patch" | "minor" | "major"): string {
  const [maj, min, pat] = version.split(".").map(Number);
  return kind === "major"
    ? `${maj + 1}.0.0`
    : kind === "minor"
    ? `${maj}.${min + 1}.0`
    : `${maj}.${min}.${pat + 1}`;
}

type RunInfo = {
  databaseId: number;
  status: string;
  conclusion: string;
  url: string;
};

/** The newest publish-deno.yml run for `tag`, or null while GitHub has none yet. */
function latestRun(tag: string): RunInfo | null {
  const raw = run("gh", [
    "run",
    "list",
    "--workflow",
    "publish-deno.yml",
    "--branch",
    tag,
    "--limit",
    "1",
    "--json",
    "databaseId,status,conclusion,url",
  ]);
  return (JSON.parse(raw || "[]") as RunInfo[])[0] ?? null;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Report a completed run; throws unless it succeeded. */
function settle(tag: string, info: RunInfo): void {
  if (info.conclusion === "success") {
    console.log(`  published ✓ ${info.url}`);
    return;
  }
  throw new Error(
    `publish run for ${tag} ended with ${info.conclusion}: ${info.url}`,
  );
}

async function waitForRun(tag: string): Promise<void> {
  console.log(`  waiting for the publish workflow run of ${tag} …`);
  for (let i = 0; i < 120; i++) {
    await sleep(15_000);
    const info = latestRun(tag);
    if (info?.status === "completed") return settle(tag, info);
  }
  throw new Error(`timed out waiting for the publish run of ${tag}`);
}

const REPO = join(ROOT, "..");

/** True when `tag` is already on origin (an earlier --push got that far). */
function remoteHasTag(tag: string): boolean {
  const ref = `refs/tags/${tag}`;
  return run("git", ["ls-remote", "--tags", "origin", ref], REPO) !== "";
}

/** Re-run the last publish run of a tag that is already on origin. */
function rerunPublish(tag: string): void {
  const last = latestRun(tag);
  if (!last) {
    throw new Error(`${tag} is on origin but has no publish run to re-run`);
  }
  run("gh", ["run", "rerun", String(last.databaseId)]);
  console.log(`re-running ${last.url} (${tag} is already on origin)`);
}

/** Create and push `tag`; when origin already has it, re-run its publish run. */
function release(pkg: Pkg, tag: string): void {
  if (remoteHasTag(tag)) return rerunPublish(tag);
  if (!run("git", ["tag", "-l", tag], REPO)) {
    run("git", [
      "tag",
      "-a",
      tag,
      "-m",
      `@rullama/${pkg.name} ${pkg.version}`,
    ], REPO);
  }
  run("git", ["push", "origin", tag], REPO);
  console.log(`pushed ${tag}`);
}

async function publishOne(pkg: Pkg, wait: boolean): Promise<void> {
  const tag = `${pkg.name}-v${pkg.version}`;
  release(pkg, tag);
  if (wait) await waitForRun(tag);
}

/** Publish one package unless a dependency already failed; false when it did not publish. */
async function tryPublish(
  pkg: Pkg,
  failed: Set<string>,
  wait: boolean,
): Promise<boolean> {
  const blocker = pkg.deps.find((d) => failed.has(d));
  if (blocker) {
    console.log(
      `skipped ${pkg.name}: depends on ${blocker}, which did not publish`,
    );
    return false;
  }
  try {
    await publishOne(pkg, wait);
    return true;
  } catch (e) {
    console.error(`  ✗ ${(e as Error).message}`);
    return false;
  }
}

/**
 * Publish `pkgs` in order. A failure skips only the packages that depend on
 * it; the rest still publish. Returns the names that did not publish.
 */
async function publishAll(pkgs: Pkg[], wait: boolean): Promise<string[]> {
  const failed = new Set<string>();
  for (const pkg of pkgs) {
    if (!(await tryPublish(pkg, failed, wait))) failed.add(pkg.name);
  }
  return [...failed];
}

// ---------------------------------------------------------------------------

const args = new Set(Deno.args);
const push = args.has("--push");
const wait = !args.has("--no-wait");
const bumpKind = (["patch", "minor", "major"] as const).find((k) =>
  args.has("--bump") && Deno.args[Deno.args.indexOf("--bump") + 1] === k
);
if (args.has("--bump") && !bumpKind) {
  throw new Error("--bump needs patch | minor | major");
}

const pkgs = loadPackages();
const order = topoOrder(pkgs);
const toPublish: Pkg[] = [];
const needBump: Pkg[] = [];

for (const name of order) {
  const pkg = pkgs.find((p) => p.name === name)!;
  const published = await jsrLatest(name);
  const onJsr = await jsrHasVersion(name, pkg.version);
  const tag = lastReleaseTag(name, published);
  const changed = tag ? changedSince(tag, name) : true;
  if (!onJsr) toPublish.push(pkg);
  else if (changed) needBump.push(pkg);
  console.log(
    `${name.padEnd(16)} deno.json ${pkg.version.padEnd(8)} jsr ${
      (published ?? "-").padEnd(8)
    } ${
      !onJsr
        ? "→ publish"
        : changed
        ? "→ changed since " + tag + ", NEEDS BUMP"
        : "up to date"
    }`,
  );
}

if (bumpKind) {
  for (const pkg of needBump) {
    const file = join(ROOT, "packages", pkg.name, "deno.json");
    const json = JSON.parse(Deno.readTextFileSync(file));
    json.version = bump(pkg.version, bumpKind);
    Deno.writeTextFileSync(file, JSON.stringify(json, null, 2) + "\n");
    console.log(`bumped ${pkg.name} ${pkg.version} → ${json.version}`);
  }
  console.log(
    needBump.length
      ? "\ncommit the bumps, then run again with --push"
      : "\nnothing to bump",
  );
  Deno.exit(0);
}

if (needBump.length) {
  console.log(
    `\n${needBump.length} package(s) changed without a version bump: ${
      needBump.map((p) => p.name).join(", ")
    }`,
  );
  console.log("run with --bump patch (or minor/major), commit, then --push");
  Deno.exit(1);
}
if (toPublish.length === 0) {
  console.log("\nnothing to publish");
  Deno.exit(0);
}
console.log(
  `\n${toPublish.length} package(s) to publish, in dependency order: ${
    toPublish.map((p) => `${p.name}-v${p.version}`).join(" ")
  }`,
);
if (!push) {
  console.log("dry run — add --push to create and push the tags");
  Deno.exit(0);
}
if (run("git", ["status", "--porcelain"], join(ROOT, "..")) !== "") {
  throw new Error(
    "working tree is dirty — commit first so the tags point at committed code",
  );
}
const notPublished = await publishAll(toPublish, wait);
if (notPublished.length) {
  console.log(
    `\nnot published: ${
      notPublished.join(", ")
    }. Fix the cause, then run --push again.`,
  );
  Deno.exit(1);
}
console.log("\ndone");
