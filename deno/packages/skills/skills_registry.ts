/**
 * Skill Registry
 *
 * Central registry for managing Agent Skills with progressive disclosure.
 *
 * ## Progressive Disclosure
 *
 * - At startup: Only metadata (name, description) is loaded
 * - On activation: Full content is loaded on-demand and cached
 *
 * ## Discovery
 *
 * Use {@link SkillRegistry.discoverFrom} with explicit path + source pairs.
 * The CLI adapter provides platform-specific paths (personal + project dirs).
 */

import {
  executionMode,
  type Skill,
  type SkillMetadata,
  type SkillSource,
} from "./skills_metadata.ts";
import { parseSkillFile, parseSkillMetadata } from "./skills_parser.ts";

/**
 * Truncate a description to a maximum length.
 *
 * @param desc - The description to truncate
 * @param maxLen - Maximum length
 * @returns Truncated string
 */
export function truncateDescription(desc: string, maxLen: number): string {
  const firstLine = desc.split("\n")[0] ?? desc;
  if (firstLine.length <= maxLen) return firstLine;
  return firstLine.slice(0, maxLen - 3) + "...";
}

/** A path + source pair for skill discovery. */
export interface DiscoveryPath {
  /** Directory path containing skill files. */
  path: string;
  /** Source classification. */
  source: SkillSource;
}

/**
 * Skill registry managing all available skills.
 *
 * Supports progressive disclosure: metadata is loaded at startup,
 * full content is loaded on-demand and cached.
 */
export class SkillRegistry {
  /** Skills indexed by name (metadata only at startup). */
  private skills: Map<string, SkillMetadata> = new Map();
  /** Cache of fully loaded skills (loaded on-demand). */
  private loadedCache: Map<string, Skill> = new Map();
  /** Paths used for the last discovery (stored for reload). */
  private discoveryPaths: DiscoveryPath[] = [];

  /**
   * Discover skills from explicit path + source pairs.
   *
   * Clears existing skills, then loads metadata from each provided directory.
   * Paths provided later in the array override earlier ones for same-named skills
   * (so project skills can override personal skills when passed last).
   *
   * @param paths - Array of path + source pairs
   */
  discoverFrom(paths: DiscoveryPath[]): void {
    this.skills.clear();
    this.loadedCache.clear();
    this.discoveryPaths = [...paths];

    for (const { path, source } of paths) {
      try {
        const stat = Deno.statSync(path);
        if (stat.isDirectory) {
          this.loadFromDirectory(path, source);
        }
      } catch {
        // Directory doesn't exist, skip
      }
    }
  }

  /**
   * Reload skills using the same paths as the last `discoverFrom` call.
   */
  reload(): void {
    const paths = [...this.discoveryPaths];
    this.discoverFrom(paths);
  }

  /**
   * Load skill metadata from a directory.
   *
   * @param dir - Directory path
   * @param source - Source classification
   */
  private loadFromDirectory(dir: string, source: SkillSource): void {
    for (const entry of Deno.readDirSync(dir)) {
      const fullPath = `${dir}/${entry.name}`;

      if (entry.isDirectory) {
        // Skill in subdirectory: skill-name/SKILL.md
        const skillFile = `${fullPath}/SKILL.md`;
        try {
          Deno.statSync(skillFile);
          this.loadSkillFile(skillFile, source);
        } catch {
          // SKILL.md not found in subdirectory, skip
        }
      } else if (entry.name.endsWith(".md")) {
        // Direct .md file: skill-name.md
        this.loadSkillFile(fullPath, source);
      }
    }
  }

  /**
   * Load a single skill file (metadata only).
   *
   * @param path - Path to skill file
   * @param source - Source classification
   */
  private loadSkillFile(path: string, source: SkillSource): void {
    try {
      const metadata = parseSkillMetadata(path);
      metadata.source = source;
      metadata.sourcePath = path;

      // Project skills override personal skills with same name
      if (source === "project" || !this.skills.has(metadata.name)) {
        this.skills.set(metadata.name, metadata);
      }
    } catch (e) {
      console.warn(
        `Failed to load skill from ${path}: ${
          e instanceof Error ? e.message : String(e)
        }`,
      );
    }
  }

  /**
   * Register a skill directly (for built-in skills).
   *
   * @param metadata - The skill metadata to register
   */
  register(metadata: SkillMetadata): void {
    this.skills.set(metadata.name, metadata);
  }

  /**
   * Get skill metadata by name.
   *
   * @param name - The skill name
   * @returns The metadata if found, or undefined
   */
  getMetadata(name: string): SkillMetadata | undefined {
    return this.skills.get(name);
  }

  /**
   * Lazy load full skill content.
   *
   * Returns cached skill if already loaded, otherwise loads from disk.
   *
   * @param name - The skill name
   * @returns The full Skill
   * @throws Error if the skill is not found or cannot be loaded
   */
  getSkill(name: string): Skill {
    const cached = this.loadedCache.get(name);
    if (cached) return cached;

    const metadata = this.skills.get(name);
    if (!metadata) {
      throw new Error(`Skill not found: ${name}`);
    }

    const skill = parseSkillFile(metadata.sourcePath);
    this.loadedCache.set(name, skill);
    return skill;
  }

  /**
   * Check if a skill exists.
   *
   * @param name - The skill name
   * @returns True if the skill exists in the registry
   */
  contains(name: string): boolean {
    return this.skills.has(name);
  }

  /**
   * List all skill names, sorted alphabetically.
   *
   * @returns Sorted array of skill names
   */
  listSkills(): string[] {
    return [...this.skills.keys()].sort();
  }

  /**
   * Get all metadata for semantic matching.
   *
   * @returns Array of all skill metadata
   */
  allMetadata(): SkillMetadata[] {
    return [...this.skills.values()];
  }

  /**
   * Get metadata for all skills from a specific source.
   *
   * @param source - The source to filter by
   * @returns Array of matching skill metadata
   */
  skillsBySource(source: SkillSource): SkillMetadata[] {
    return [...this.skills.values()].filter((m) => m.source === source);
  }

  /**
   * Get the number of registered skills.
   *
   * @returns The count
   */
  get length(): number {
    return this.skills.size;
  }

  /**
   * Check if registry is empty.
   *
   * @returns True if no skills are registered
   */
  get isEmpty(): boolean {
    return this.skills.size === 0;
  }

  /**
   * Clear the loaded skill cache.
   *
   * Forces skills to be reloaded from disk on next access.
   */
  clearCache(): void {
    this.loadedCache.clear();
  }

  /**
   * Remove a skill from the registry.
   *
   * @param name - The skill name to remove
   * @returns The removed metadata, or undefined
   */
  remove(name: string): SkillMetadata | undefined {
    this.loadedCache.delete(name);
    const meta = this.skills.get(name);
    this.skills.delete(name);
    return meta;
  }

  /**
   * Get skills that match a category.
   *
   * @param category - The category to match
   * @returns Array of matching skill metadata
   */
  skillsByCategory(category: string): SkillMetadata[] {
    return [...this.skills.values()].filter(
      (m) => m.metadata?.["category"] === category,
    );
  }

  /**
   * Format a skill listing for display.
   *
   * @returns Formatted markdown string
   */
  formatSkillList(): string {
    if (this.skills.size === 0) {
      return "No skills available. Add skills to ~/.rullama/skills/ or .rullama/skills/";
    }

    let output = "";
    const personal = this.skillsBySource("personal").sort((a, b) =>
      a.name.localeCompare(b.name)
    );
    const project = this.skillsBySource("project").sort((a, b) =>
      a.name.localeCompare(b.name)
    );

    if (project.length > 0) {
      output += "## Project Skills\n\n";
      for (const skill of project) {
        output += `- **${skill.name}**: ${
          truncateDescription(skill.description, 60)
        }\n`;
      }
      output += "\n";
    }

    if (personal.length > 0) {
      output += "## Personal Skills\n\n";
      for (const skill of personal) {
        // Skip if overridden by project skill
        if (project.some((p) => p.name === skill.name)) continue;
        output += `- **${skill.name}**: ${
          truncateDescription(skill.description, 60)
        }\n`;
      }
    }

    output += "\nUse `/skill <name>` to invoke a skill.\n";

    return output;
  }

  /**
   * Format detailed skill info for display.
   *
   * @param name - The skill name
   * @returns Formatted markdown string
   * @throws Error if the skill is not found
   */
  formatSkillDetail(name: string): string {
    const metadata = this.skills.get(name);
    if (!metadata) {
      throw new Error(`Skill not found: ${name}`);
    }

    let output = "";
    output += `# ${metadata.name}\n\n`;
    output += `**Description**: ${metadata.description}\n\n`;
    output += `**Source**: ${metadata.source}\n`;
    output += `**Execution Mode**: ${executionMode(metadata)}\n`;

    if (metadata["allowed-tools"]) {
      output += `**Allowed Tools**: ${metadata["allowed-tools"].join(", ")}\n`;
    }

    if (metadata.license) {
      output += `**License**: ${metadata.license}\n`;
    }

    if (metadata.model) {
      output += `**Model**: ${metadata.model}\n`;
    }

    if (metadata.metadata && Object.keys(metadata.metadata).length > 0) {
      output += "\n**Metadata**:\n";
      for (const [key, value] of Object.entries(metadata.metadata)) {
        output += `  - ${key}: ${value}\n`;
      }
    }

    output += `\n**File**: ${metadata.sourcePath}\n`;

    return output;
  }
}
