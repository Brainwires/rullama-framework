// Example: TemplateStore — reusable plan templates with variable substitution
// Demonstrates creating a TemplateStore, registering templates with {{variable}}
// placeholders, searching/listing, and instantiating with concrete values.
// Run: deno run deno/examples/storage/plan_templates.ts

import {
  createTemplate,
  extractVariables,
  instantiateTemplate,
  markUsed,
  TemplateStore,
  withCategory,
  withTags,
} from "@rullama/storage";

async function main() {
  console.log("=== Plan Templates Example ===\n");

  // 1. Create a template store
  const store = new TemplateStore();
  console.log("TemplateStore created (in-memory)\n");

  // 2. Register a feature-implementation template
  let featureTemplate = createTemplate(
    "Feature Implementation",
    "Step-by-step plan for implementing a new feature",
    `# Feature: {{feature_name}}

## 1. Design
- Define the public API for {{component}}
- Write interface types in \`src/{{module}}/types.ts\`

## 2. Implementation
- Implement {{feature_name}} logic in \`src/{{module}}/mod.ts\`
- Add error handling for {{component}} edge cases

## 3. Testing
- Unit tests for {{component}} in \`tests/{{module}}_test.ts\`
- Integration test covering the {{feature_name}} happy path

## 4. Documentation
- Add JSDoc comments to all public items in {{module}}
`,
  );
  featureTemplate = withCategory(featureTemplate, "feature");
  featureTemplate = withTags(featureTemplate, ["typescript", "implementation"]);

  store.save(featureTemplate);
  console.log(`Saved template: ${featureTemplate.name}`);
  console.log(`  Variables: [${featureTemplate.variables.join(", ")}]`);
  console.log(`  Category:  ${featureTemplate.category}`);
  console.log();

  // 3. Register a bugfix template
  let bugfixTemplate = createTemplate(
    "Bug Fix Workflow",
    "Systematic approach to diagnosing and fixing bugs",
    `# Bug Fix: {{bug_title}}

## 1. Reproduce
- Reproduce {{bug_title}} using the steps from {{issue_tracker}} issue
- Capture failing test output

## 2. Root Cause
- Trace execution path in {{affected_module}}
- Identify the root cause

## 3. Fix
- Apply fix in \`src/{{affected_module}}/\`
- Add regression test for {{bug_title}}

## 4. Verify
- Run full test suite for {{package_name}}
- Confirm the {{issue_tracker}} issue is resolved
`,
  );
  bugfixTemplate = withCategory(bugfixTemplate, "bugfix");
  bugfixTemplate = withTags(bugfixTemplate, ["debugging", "testing"]);

  store.save(bugfixTemplate);
  console.log(`Saved template: ${bugfixTemplate.name}`);
  console.log(`  Variables: [${bugfixTemplate.variables.join(", ")}]`);
  console.log();

  // 4. List all templates (sorted by usage count, then name)
  const all = store.list();
  console.log(`--- All Templates (${all.length}) ---`);
  for (const t of all) {
    console.log(
      `  [${
        t.templateId.slice(0, 8)
      }] ${t.name} -- ${t.description} (used ${t.usageCount} times)`,
    );
  }
  console.log();

  // 5. Search templates by keyword
  const searchResults = store.search("bug");
  console.log(`Search for "bug": ${searchResults.length} result(s)`);
  for (const t of searchResults) {
    console.log(`  ${t.name} -- ${t.description}`);
  }
  console.log();

  // 6. List templates by category
  const features = store.listByCategory("feature");
  console.log(`Category "feature": ${features.length} template(s)`);
  console.log();

  // 7. Instantiate the feature template with concrete values
  console.log("--- Instantiation ---");
  const substitutions: Record<string, string> = {
    feature_name: "message encryption",
    component: "EncryptionService",
    module: "encryption",
  };
  const instantiated = instantiateTemplate(featureTemplate, substitutions);
  console.log("Instantiated plan:");
  console.log(instantiated);

  // 8. Mark a template as used and verify the count updates
  console.log("--- Usage Tracking ---");
  store.markUsed(featureTemplate.templateId);
  store.markUsed(featureTemplate.templateId);
  const updated = store.get(featureTemplate.templateId);
  if (updated) {
    console.log(
      `Template "${updated.name}" usage count: ${updated.usageCount}`,
    );
  }
  console.log();

  // 9. Extract variables from arbitrary content
  console.log("--- Variable Extraction ---");
  const customContent =
    "Deploy {{service}} to {{region}} with {{replicas}} replicas";
  const vars = extractVariables(customContent);
  console.log(`Content: "${customContent}"`);
  console.log(`Variables: [${vars.join(", ")}]`);
  console.log();

  // 10. Serialize to JSON and back
  console.log("--- Serialization ---");
  const json = store.toJson();
  console.log(`Serialized ${json.length} chars of JSON`);

  const store2 = new TemplateStore();
  store2.loadFromJson(json);
  const restored = store2.list();
  console.log(`Restored ${restored.length} templates from JSON`);

  console.log("\nDone.");
}

await main();
