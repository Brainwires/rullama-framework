/**
 * Template Store -- storage for reusable plan templates.
 *
 * Templates are saved plans that can be instantiated for new tasks.
 * Fully ported from Rust's `stores/template_store.rs` including
 * {{variable}} substitution logic.
 * @module
 */

/** Template metadata. */
export interface PlanTemplate {
  /** Unique template ID. */
  templateId: string;
  /** Template name (user-friendly). */
  name: string;
  /** Template description. */
  description: string;
  /** The template content (markdown with placeholders). */
  content: string;
  /** Category for organization (e.g. "feature", "bugfix", "refactor"). */
  category?: string;
  /** Tags for discovery. */
  tags: string[];
  /** Variables/placeholders in the template (e.g. ["component", "feature"]). */
  variables: string[];
  /** Original plan ID this template was derived from (if any). */
  sourcePlanId?: string;
  /** Number of times this template has been used. */
  usageCount: number;
  /** Creation timestamp (Unix seconds). */
  createdAt: number;
  /** Last used timestamp (Unix seconds). */
  lastUsedAt?: number;
}

const VARIABLE_REGEX = /\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}/g;

/** Extract variables from template content. */
export function extractVariables(content: string): string[] {
  const vars = new Set<string>();
  let match: RegExpExecArray | null;
  const re = new RegExp(VARIABLE_REGEX.source, "g");
  while ((match = re.exec(content)) !== null) {
    vars.add(match[1]);
  }
  return [...vars].sort();
}

/** Create a new template. */
export function createTemplate(
  name: string,
  description: string,
  content: string,
): PlanTemplate {
  const now = Math.floor(Date.now() / 1000);
  return {
    templateId: crypto.randomUUID(),
    name,
    description,
    content,
    tags: [],
    variables: extractVariables(content),
    usageCount: 0,
    createdAt: now,
  };
}

/** Create a template from an existing plan. */
export function createTemplateFromPlan(
  name: string,
  description: string,
  planContent: string,
  planId: string,
): PlanTemplate {
  const template = createTemplate(name, description, planContent);
  template.sourcePlanId = planId;
  return template;
}

/** Set category on a template (returns new copy). */
export function withCategory(
  template: PlanTemplate,
  category: string,
): PlanTemplate {
  return { ...template, category };
}

/** Set tags on a template (returns new copy). */
export function withTags(template: PlanTemplate, tags: string[]): PlanTemplate {
  return { ...template, tags };
}

/** Instantiate a template with variable substitutions. */
export function instantiateTemplate(
  template: PlanTemplate,
  substitutions: Map<string, string> | Record<string, string>,
): string {
  let result = template.content;
  const subs = substitutions instanceof Map
    ? substitutions
    : new Map(Object.entries(substitutions));

  for (const [varName, value] of subs) {
    // split/join replaces every occurrence in one pass, so a value that itself
    // contains the placeholder cannot make the substitution loop forever.
    result = result.split(`{{${varName}}}`).join(value);
  }
  return result;
}

/** Mark a template as used (mutates in place). */
export function markUsed(template: PlanTemplate): void {
  template.usageCount += 1;
  template.lastUsedAt = Math.floor(Date.now() / 1000);
}

// -- TemplateStore (in-memory, JSON-serializable) ---------------------------

interface TemplateData {
  templates: PlanTemplate[];
}

/** Template store for persistence. In the Deno port, uses in-memory storage
 *  with optional JSON serialization. */
export class TemplateStore {
  private data: TemplateData = { templates: [] };

  constructor() {}

  /** Load templates from a JSON string. */
  loadFromJson(json: string): void {
    this.data = JSON.parse(json) as TemplateData;
  }

  /** Serialize templates to a JSON string. */
  toJson(): string {
    return JSON.stringify(this.data, null, 2);
  }

  /** Save a template. */
  save(template: PlanTemplate): void {
    this.data.templates = this.data.templates.filter(
      (t) => t.templateId !== template.templateId,
    );
    this.data.templates.push({ ...template });
  }

  /** Get a template by ID. */
  get(templateId: string): PlanTemplate | undefined {
    return this.data.templates.find((t) => t.templateId === templateId);
  }

  /** Get a template by name (case-insensitive partial match). */
  getByName(name: string): PlanTemplate | undefined {
    const nameLower = name.toLowerCase();
    return this.data.templates.find(
      (t) =>
        t.name.toLowerCase().includes(nameLower) ||
        t.templateId.startsWith(name),
    );
  }

  /** List all templates, sorted by usage count (most used first). */
  list(): PlanTemplate[] {
    const templates = [...this.data.templates];
    templates.sort((a, b) => {
      const cmp = b.usageCount - a.usageCount;
      return cmp !== 0 ? cmp : a.name.localeCompare(b.name);
    });
    return templates;
  }

  /** List templates by category. */
  listByCategory(category: string): PlanTemplate[] {
    return this.list().filter((t) => t.category === category);
  }

  /** Search templates by name, description, or tags. */
  search(query: string): PlanTemplate[] {
    const queryLower = query.toLowerCase();
    return this.list().filter(
      (t) =>
        t.name.toLowerCase().includes(queryLower) ||
        t.description.toLowerCase().includes(queryLower) ||
        t.tags.some((tag) => tag.toLowerCase().includes(queryLower)),
    );
  }

  /** Delete a template. Returns true if found and deleted. */
  delete(templateId: string): boolean {
    const originalLen = this.data.templates.length;
    this.data.templates = this.data.templates.filter(
      (t) => t.templateId !== templateId,
    );
    return this.data.templates.length < originalLen;
  }

  /** Update a template (same as save). */
  update(template: PlanTemplate): void {
    this.save(template);
  }

  /** Mark a template as used and save. */
  markUsed(templateId: string): void {
    const template = this.get(templateId);
    if (template) {
      markUsed(template);
      this.save(template);
    }
  }
}
