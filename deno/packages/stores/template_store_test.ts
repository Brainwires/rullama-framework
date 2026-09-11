/**
 * Tests for TemplateStore and template variable substitution.
 */

import { assertEquals, assertNotEquals } from "@std/assert";
import {
  createTemplate,
  createTemplateFromPlan,
  extractVariables,
  instantiateTemplate,
  markUsed,
  TemplateStore,
  withCategory,
  withTags,
} from "./template_store.ts";

Deno.test("extractVariables - extracts unique sorted variables", () => {
  const content = "{{var1}} and {{var2}} and {{var1}} again";
  const vars = extractVariables(content);
  assertEquals(vars.length, 2);
  assertEquals(vars[0], "var1");
  assertEquals(vars[1], "var2");
});

Deno.test("extractVariables - empty content", () => {
  assertEquals(extractVariables("no variables here"), []);
});

Deno.test("extractVariables - ignores invalid variable names", () => {
  assertEquals(extractVariables("{{123invalid}}"), []);
  assertEquals(extractVariables("{{_valid}}").length, 1);
});

Deno.test("createTemplate - generates ID and extracts variables", () => {
  const template = createTemplate(
    "Feature Implementation",
    "Template for implementing new features",
    "1. Create {{component}} component\n2. Add tests for {{feature}}",
  );

  assertNotEquals(template.templateId, "");
  assertEquals(template.name, "Feature Implementation");
  assertEquals(template.variables.length, 2);
  assertEquals(template.variables.includes("component"), true);
  assertEquals(template.variables.includes("feature"), true);
  assertEquals(template.usageCount, 0);
  assertEquals(template.lastUsedAt, undefined);
});

Deno.test("createTemplateFromPlan - sets source plan ID", () => {
  const template = createTemplateFromPlan(
    "My Template",
    "Description",
    "Content",
    "plan-123",
  );
  assertEquals(template.sourcePlanId, "plan-123");
});

Deno.test("instantiateTemplate - substitutes all variables", () => {
  const template = createTemplate(
    "Test",
    "Test template",
    "Implement {{feature}} in {{module}}",
  );

  const result = instantiateTemplate(template, {
    feature: "authentication",
    module: "auth",
  });
  assertEquals(result, "Implement authentication in auth");
});

Deno.test("instantiateTemplate - handles repeated variables", () => {
  const template = createTemplate(
    "Test",
    "Test",
    "{{name}} is great. Use {{name}} everywhere.",
  );

  const result = instantiateTemplate(template, { name: "TypeScript" });
  assertEquals(result, "TypeScript is great. Use TypeScript everywhere.");
});

Deno.test("instantiateTemplate - unsubstituted variables remain", () => {
  const template = createTemplate("Test", "Test", "Hello {{name}}");
  const result = instantiateTemplate(template, {});
  assertEquals(result, "Hello {{name}}");
});

Deno.test("instantiateTemplate - works with Map", () => {
  const template = createTemplate("Test", "Test", "{{x}} + {{y}}");
  const subs = new Map([["x", "1"], ["y", "2"]]);
  assertEquals(instantiateTemplate(template, subs), "1 + 2");
});

Deno.test("markUsed - increments count and sets timestamp", () => {
  const template = createTemplate("Test", "Test", "Content");
  assertEquals(template.usageCount, 0);
  assertEquals(template.lastUsedAt, undefined);

  markUsed(template);
  assertEquals(template.usageCount, 1);
  assertNotEquals(template.lastUsedAt, undefined);
});

Deno.test("withCategory / withTags - set metadata", () => {
  let template = createTemplate("Test", "Test", "Content");
  template = withCategory(template, "feature");
  template = withTags(template, ["rust", "api"]);
  assertEquals(template.category, "feature");
  assertEquals(template.tags.length, 2);
});

Deno.test("TemplateStore - CRUD operations", () => {
  const store = new TemplateStore();

  // Save
  const t1 = createTemplate("Template A", "Description A", "Content A");
  const t2 = createTemplate(
    "Template B",
    "Description B",
    "Content B with {{var}}",
  );
  store.save(t1);
  store.save(t2);

  // Get by ID
  const got = store.get(t1.templateId);
  assertEquals(got?.name, "Template A");

  // Get nonexistent
  assertEquals(store.get("nonexistent"), undefined);

  // List
  const all = store.list();
  assertEquals(all.length, 2);

  // Delete
  const deleted = store.delete(t1.templateId);
  assertEquals(deleted, true);
  assertEquals(store.list().length, 1);

  // Delete nonexistent
  assertEquals(store.delete("nonexistent"), false);
});

Deno.test("TemplateStore - getByName partial match", () => {
  const store = new TemplateStore();
  const t = createTemplate("Feature Implementation", "Desc", "Content");
  store.save(t);

  const found = store.getByName("feature");
  assertEquals(found?.name, "Feature Implementation");

  assertEquals(store.getByName("nonexistent"), undefined);
});

Deno.test("TemplateStore - listByCategory", () => {
  const store = new TemplateStore();
  const t1 = withCategory(createTemplate("A", "a", "a"), "feature");
  const t2 = withCategory(createTemplate("B", "b", "b"), "bugfix");
  const t3 = withCategory(createTemplate("C", "c", "c"), "feature");
  store.save(t1);
  store.save(t2);
  store.save(t3);

  assertEquals(store.listByCategory("feature").length, 2);
  assertEquals(store.listByCategory("bugfix").length, 1);
  assertEquals(store.listByCategory("other").length, 0);
});

Deno.test("TemplateStore - search by name, description, tags", () => {
  const store = new TemplateStore();
  const t1 = withTags(createTemplate("Auth Setup", "setup auth", "content"), [
    "security",
  ]);
  const t2 = createTemplate("Database Migration", "migrate db", "content");
  store.save(t1);
  store.save(t2);

  assertEquals(store.search("auth").length, 1);
  assertEquals(store.search("migrate").length, 1);
  assertEquals(store.search("security").length, 1);
  assertEquals(store.search("nonexistent").length, 0);
});

Deno.test("TemplateStore - markUsed", () => {
  const store = new TemplateStore();
  const t = createTemplate("Test", "Test", "Content");
  store.save(t);

  store.markUsed(t.templateId);
  const updated = store.get(t.templateId);
  assertEquals(updated?.usageCount, 1);
  assertNotEquals(updated?.lastUsedAt, undefined);
});

Deno.test("TemplateStore - update replaces existing", () => {
  const store = new TemplateStore();
  const t = createTemplate("Test", "Original", "Content");
  store.save(t);

  t.description = "Updated";
  store.update(t);

  const got = store.get(t.templateId);
  assertEquals(got?.description, "Updated");
  assertEquals(store.list().length, 1);
});

Deno.test("TemplateStore - JSON round-trip", () => {
  const store = new TemplateStore();
  const t = createTemplate("Test", "Desc", "{{var}} content");
  store.save(t);

  const json = store.toJson();
  const store2 = new TemplateStore();
  store2.loadFromJson(json);

  const got = store2.get(t.templateId);
  assertEquals(got?.name, "Test");
  assertEquals(got?.variables, ["var"]);
});

Deno.test("instantiateTemplate: a value containing its own placeholder does not loop", () => {
  const template = createTemplate("t", "self", "Hello {{name}}!");
  assertEquals(
    instantiateTemplate(template, { name: "{{name}} again" }),
    "Hello {{name}} again!",
  );
});
