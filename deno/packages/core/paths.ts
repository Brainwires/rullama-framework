/**
 * Centralized platform-specific path computation.
 *
 * Provides consistent path handling across Windows, macOS, and Linux,
 * following the XDG Base Directory specification on Unix-like systems.
 *
 * Equivalent to Rust's `rullama_core::paths::PlatformPaths`.
 */

import { join } from "@std/path";

/** Project folder name used to namespace per-project data/cache/config. */
const PROJECT_FOLDER_NAME = "rullama-rag";

function envOr(name: string, fallback: () => string): string {
  const v = Deno.env.get(name);
  return v && v.length > 0 ? v : fallback();
}

function home(): string {
  return envOr("HOME", () => envOr("USERPROFILE", () => "."));
}

/** Platform-agnostic path utilities. */
// deno-lint-ignore no-namespace
export namespace PlatformPaths {
  /** Detected OS. */
  function isWindows(): boolean {
    return Deno.build.os === "windows";
  }
  function isMacOS(): boolean {
    return Deno.build.os === "darwin";
  }

  /**
   * Platform-appropriate data directory.
   * - Windows: `%LOCALAPPDATA%`
   * - macOS:   `~/Library/Application Support`
   * - Linux:   `$XDG_DATA_HOME` or `~/.local/share`
   */
  export function dataDir(): string {
    if (isWindows()) return envOr("LOCALAPPDATA", () => ".");
    if (isMacOS()) return join(home(), "Library", "Application Support");
    return envOr("XDG_DATA_HOME", () => join(home(), ".local", "share"));
  }

  /**
   * Platform-appropriate cache directory.
   * - Windows: `%LOCALAPPDATA%`
   * - macOS:   `~/Library/Caches`
   * - Linux:   `$XDG_CACHE_HOME` or `~/.cache`
   */
  export function cacheDir(): string {
    if (isWindows()) return envOr("LOCALAPPDATA", () => ".");
    if (isMacOS()) return join(home(), "Library", "Caches");
    return envOr("XDG_CACHE_HOME", () => join(home(), ".cache"));
  }

  /**
   * Platform-appropriate config directory.
   * - Windows: `%APPDATA%`
   * - macOS:   `~/Library/Application Support`
   * - Linux:   `$XDG_CONFIG_HOME` or `~/.config`
   */
  export function configDir(): string {
    if (isWindows()) return envOr("APPDATA", () => ".");
    if (isMacOS()) return join(home(), "Library", "Application Support");
    return envOr("XDG_CONFIG_HOME", () => join(home(), ".config"));
  }

  /** Returns "rullama-rag". */
  export function projectFolderName(): string {
    return PROJECT_FOLDER_NAME;
  }

  /** `{dataDir}/rullama-rag`. */
  export function projectDataDir(): string {
    return join(dataDir(), PROJECT_FOLDER_NAME);
  }

  /** `{cacheDir}/rullama-rag`. */
  export function projectCacheDir(): string {
    return join(cacheDir(), PROJECT_FOLDER_NAME);
  }

  /** `{configDir}/rullama-rag`. */
  export function projectConfigDir(): string {
    return join(configDir(), PROJECT_FOLDER_NAME);
  }

  /** Default LanceDB directory: `{projectDataDir}/lancedb`. */
  export function defaultLancedbPath(): string {
    return join(projectDataDir(), "lancedb");
  }

  /** Default hash-cache file: `{projectCacheDir}/hash_cache.json`. */
  export function defaultHashCachePath(): string {
    return join(projectCacheDir(), "hash_cache.json");
  }

  /** Default git-cache file: `{projectCacheDir}/git_cache.json`. */
  export function defaultGitCachePath(): string {
    return join(projectCacheDir(), "git_cache.json");
  }

  /** Default fastembed model cache: `~/.rullama/cache/fastembed`. */
  export function defaultFastembedCachePath(): string {
    return join(home(), ".rullama", "cache", "fastembed");
  }

  /** Default config file: `{projectConfigDir}/config.toml`. */
  export function defaultConfigPath(): string {
    return join(projectConfigDir(), "config.toml");
  }
}
