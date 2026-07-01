/**
 * SKILL.md Parser
 *
 * Parses skill files from .rullama/skills/ directories.
 *
 * ## Format
 *
 * ```markdown
 * ---
 * name: skill-name
 * description: What the skill does and when to use it
 * allowed-tools:
 *   - Read
 *   - Grep
 * license: Apache-2.0
 * model: claude-sonnet-4
 * metadata:
 *   category: development
 *   execution: inline
 * ---
 *
 * # Skill Instructions
 *
 * Step-by-step guidance for the agent...
 * ```
 */

import { parse as parseYaml } from "@std/yaml/parse";
import {
  createSkill,
  type Skill,
  type SkillMetadata,
} from "./skills_metadata.ts";

/** Maximum allowed length for a skill description. */
const SKILL_DESCRIPTION_MAX_LENGTH = 1024;

/** Maximum allowed length for a compatibility field. */
const COMPATIBILITY_MAX_LENGTH = 500;

/**
 * Raw YAML frontmatter shape before validation.
 */
interface SkillFrontmatter {
  name: string;
  description: string;
  "allowed-tools"?: string | string[];
  license?: string;
  compatibility?: string;
  model?: string;
  metadata?: Record<string, string>;
  hooks?: string[];
}

/**
 * Normalize allowed-tools from either a YAML list or a space-delimited string.
 *
 * The Agent Skills specification defines allowed-tools as a space-delimited string
 * (e.g., `allowed-tools: Bash(git:*) Read`), but we also accept YAML lists.
 *
 * @param value - The raw value from YAML parsing
 * @returns Normalized array of tool names, or undefined
 */
function normalizeAllowedTools(
  value: string | string[] | undefined | null,
): string[] | undefined {
  if (value == null) return undefined;
  if (Array.isArray(value)) {
    return value.length === 0 ? undefined : value;
  }
  if (typeof value === "string") {
    if (value.trim() === "") return undefined;
    return value.split(/\s+/);
  }
  return undefined;
}

/**
 * Parse only the skill metadata (frontmatter) from a SKILL.md file.
 *
 * This is used for progressive disclosure -- only loading metadata at startup.
 * The full content is loaded lazily when the skill is activated.
 *
 * @param path - Path to the SKILL.md file
 * @returns Parsed SkillMetadata
 * @throws Error if the file cannot be read or parsed
 */
export function parseSkillMetadata(path: string): SkillMetadata {
  const content = Deno.readTextFileSync(path);
  return parseMetadataFromContent(content, path);
}

/**
 * Parse metadata from a content string.
 *
 * @param content - The full file content
 * @param path - Path for error messages
 * @returns Parsed SkillMetadata
 * @throws Error if the content format is invalid
 */
export function parseMetadataFromContent(
  content: string,
  path: string,
): SkillMetadata {
  const parts = content.split("---");

  if (parts.length < 3) {
    throw new Error(
      `Invalid SKILL.md format in ${path}: missing frontmatter (requires --- delimiters)`,
    );
  }

  // parts[0] is content before first ---, parts[1] is frontmatter, parts[2+] is body
  const yamlContent = parts[1].trim();

  let frontmatter: SkillFrontmatter;
  try {
    frontmatter = parseYaml(yamlContent) as SkillFrontmatter;
  } catch (e) {
    throw new Error(
      `Failed to parse skill frontmatter in ${path}: ${
        e instanceof Error ? e.message : String(e)
      }`,
    );
  }

  if (!frontmatter || typeof frontmatter !== "object") {
    throw new Error(
      `Failed to parse skill frontmatter in ${path}: not an object`,
    );
  }

  // Validate constraints
  validateSkillName(frontmatter.name, path);
  validateDescription(frontmatter.description, path);
  if (frontmatter.compatibility != null) {
    validateCompatibility(frontmatter.compatibility, path);
  }

  // Warn if skill name doesn't match parent directory name
  warnNameDirectoryMismatch(frontmatter.name, path);

  const allowedTools = normalizeAllowedTools(frontmatter["allowed-tools"]);

  const metadata: SkillMetadata = {
    name: frontmatter.name,
    description: frontmatter.description,
    source: "personal", // Will be set by caller
    sourcePath: path,
  };

  if (allowedTools != null) metadata["allowed-tools"] = allowedTools;
  if (frontmatter.license != null) metadata.license = frontmatter.license;
  if (frontmatter.compatibility != null) {
    metadata.compatibility = frontmatter.compatibility;
  }
  if (frontmatter.model != null) metadata.model = frontmatter.model;
  if (frontmatter.metadata != null) metadata.metadata = frontmatter.metadata;
  if (frontmatter.hooks != null) metadata.hooks = frontmatter.hooks;

  return metadata;
}

/**
 * Parse a complete skill file (metadata + instructions).
 *
 * Used when a skill is activated and full content is needed.
 *
 * @param path - Path to the SKILL.md file
 * @returns Parsed Skill
 * @throws Error if the file cannot be read or parsed
 */
export function parseSkillFile(path: string): Skill {
  const content = Deno.readTextFileSync(path);
  return parseSkillFromContent(content, path);
}

/**
 * Parse skill from a content string.
 *
 * @param content - The full file content
 * @param path - Path for error messages
 * @returns Parsed Skill
 * @throws Error if the content format is invalid
 */
export function parseSkillFromContent(content: string, path: string): Skill {
  const parts = content.split("---");

  if (parts.length < 3) {
    throw new Error(
      `Invalid SKILL.md format in ${path}: missing frontmatter`,
    );
  }

  const metadata = parseMetadataFromContent(content, path);

  // Extract body (everything after the second ---)
  // Rejoin parts[2..] with "---" in case the body contains "---"
  const instructions = parts.slice(2).join("---").trim();

  return createSkill(metadata, instructions);
}

/**
 * Validate skill name constraints per the Agent Skills specification.
 *
 * - Must be 1-64 characters
 * - Only lowercase letters, digits, and hyphens allowed
 * - Cannot start or end with hyphen
 * - Cannot contain consecutive hyphens
 *
 * @param name - The skill name to validate
 * @param path - Path for error messages
 * @throws Error if the name is invalid
 */
export function validateSkillName(name: string, path?: string): void {
  const ctx = path ? ` in ${path}` : "";

  if (!name || name.length === 0) {
    throw new Error(`Skill name cannot be empty${ctx}`);
  }

  if (name.length > 64) {
    throw new Error(
      `Skill name exceeds 64 characters (got ${name.length}): '${name}'${ctx}`,
    );
  }

  if (name.startsWith("-") || name.endsWith("-")) {
    throw new Error(
      `Skill name cannot start or end with a hyphen: '${name}'${ctx}`,
    );
  }

  if (name.includes("--")) {
    throw new Error(
      `Skill name cannot contain consecutive hyphens: '${name}'${ctx}`,
    );
  }

  for (const c of name) {
    if (
      !(c >= "a" && c <= "z") && !(c >= "0" && c <= "9") && c !== "-"
    ) {
      throw new Error(
        `Skill name must be lowercase with hyphens only, found '${c}' in '${name}'${ctx}`,
      );
    }
  }
}

/**
 * Validate description constraints.
 *
 * - Must not be empty
 * - Max 1024 characters
 *
 * @param desc - The description to validate
 * @param path - Path for error messages
 * @throws Error if the description is invalid
 */
export function validateDescription(desc: string, path?: string): void {
  const ctx = path ? ` in ${path}` : "";

  if (!desc || desc.trim().length === 0) {
    throw new Error(`Skill description cannot be empty${ctx}`);
  }

  if (desc.length > SKILL_DESCRIPTION_MAX_LENGTH) {
    throw new Error(
      `Skill description exceeds 1024 characters (got ${desc.length})${ctx}`,
    );
  }
}

/**
 * Validate compatibility field constraints per the Agent Skills specification.
 *
 * - Must be 1-500 characters if provided
 *
 * @param compat - The compatibility string to validate
 * @param path - Path for error messages
 * @throws Error if the compatibility field is invalid
 */
export function validateCompatibility(compat: string, path?: string): void {
  const ctx = path ? ` in ${path}` : "";

  if (compat.trim().length === 0) {
    throw new Error(
      `Compatibility field cannot be empty when provided${ctx}`,
    );
  }

  if (compat.length > COMPATIBILITY_MAX_LENGTH) {
    throw new Error(
      `Compatibility field exceeds 500 characters (got ${compat.length})${ctx}`,
    );
  }
}

/**
 * Warn if the skill name doesn't match the parent directory name.
 *
 * The Agent Skills specification requires that the name field must match the
 * parent directory name. We emit a warning rather than an error since
 * rullama-skills also supports flat file layout.
 *
 * @param name - The skill name from frontmatter
 * @param path - The file path
 */
function warnNameDirectoryMismatch(name: string, path: string): void {
  const filename = path.split("/").pop() ?? path.split("\\").pop() ?? "";
  if (filename === "SKILL.md") {
    const parts = path.replace(/\\/g, "/").split("/");
    if (parts.length >= 2) {
      const dirName = parts[parts.length - 2];
      if (dirName !== name) {
        console.warn(
          `Skill name '${name}' does not match parent directory '${dirName}' in ${path}. ` +
            `The Agent Skills spec requires these to match.`,
        );
      }
    }
  }
}

/**
 * Render skill template with arguments.
 *
 * Replaces `{{arg_name}}` placeholders with provided values.
 * Supports Handlebars-style conditionals: `{{#if var}}...{{/if}}`
 *
 * @param template - The template string
 * @param args - Key-value arguments for substitution
 * @returns The rendered string
 */
export function renderTemplate(
  template: string,
  args: Record<string, string>,
): string {
  let result = template;

  // Simple template substitution: {{arg_name}}
  for (const [key, value] of Object.entries(args)) {
    const placeholder = `{{${key}}}`;
    result = result.replaceAll(placeholder, value);
  }

  // Handle simple conditionals: {{#if var}}content{{/if}}
  for (const [key, value] of Object.entries(args)) {
    const ifBlock = `{{#if ${key}}}`;
    const endif = "{{/if}}";

    while (true) {
      const start = result.indexOf(ifBlock);
      if (start === -1) break;

      const endOffset = result.indexOf(endif, start);
      if (endOffset === -1) break; // Malformed template

      const blockContent = result.slice(start + ifBlock.length, endOffset);
      const end = endOffset + endif.length;

      // If value is non-empty/truthy, keep the content; otherwise remove block
      const replacement = value.length > 0 && value !== "false" && value !== "0"
        ? blockContent
        : "";

      result = result.slice(0, start) + replacement + result.slice(end);
    }
  }

  // Remove any remaining if blocks for unset variables
  result = result.replace(/\{\{#if \w+\}\}[\s\S]*?\{\{\/if\}\}/g, "");

  return result;
}
