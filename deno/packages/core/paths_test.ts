import { assert } from "@std/assert/assert";
import { assertEquals } from "@std/assert/equals";
import { PlatformPaths } from "./paths.ts";

Deno.test("data/cache/config dirs return non-empty strings", () => {
  assert(PlatformPaths.dataDir().length > 0);
  assert(PlatformPaths.cacheDir().length > 0);
  assert(PlatformPaths.configDir().length > 0);
});

Deno.test("project dirs contain the project folder name", () => {
  assert(PlatformPaths.projectDataDir().includes("rullama-rag"));
  assert(PlatformPaths.projectCacheDir().includes("rullama-rag"));
  assert(PlatformPaths.projectConfigDir().includes("rullama-rag"));
});

Deno.test("LanceDB / hash-cache / git-cache paths end with expected components", () => {
  assert(PlatformPaths.defaultLancedbPath().endsWith("lancedb"));
  assert(PlatformPaths.defaultHashCachePath().endsWith("hash_cache.json"));
  assert(PlatformPaths.defaultGitCachePath().endsWith("git_cache.json"));
  assert(PlatformPaths.defaultConfigPath().endsWith("config.toml"));
});

Deno.test("projectFolderName returns 'rullama-rag'", () => {
  assertEquals(PlatformPaths.projectFolderName(), "rullama-rag");
});

Deno.test("Linux: XDG_DATA_HOME is respected when set", () => {
  if (Deno.build.os !== "linux") return;
  const original = Deno.env.get("XDG_DATA_HOME");
  Deno.env.set("XDG_DATA_HOME", "/custom/data");
  try {
    assertEquals(PlatformPaths.dataDir(), "/custom/data");
  } finally {
    if (original === undefined) Deno.env.delete("XDG_DATA_HOME");
    else Deno.env.set("XDG_DATA_HOME", original);
  }
});

Deno.test("Linux: XDG_CACHE_HOME is respected when set", () => {
  if (Deno.build.os !== "linux") return;
  const original = Deno.env.get("XDG_CACHE_HOME");
  Deno.env.set("XDG_CACHE_HOME", "/custom/cache");
  try {
    assertEquals(PlatformPaths.cacheDir(), "/custom/cache");
  } finally {
    if (original === undefined) Deno.env.delete("XDG_CACHE_HOME");
    else Deno.env.set("XDG_CACHE_HOME", original);
  }
});
