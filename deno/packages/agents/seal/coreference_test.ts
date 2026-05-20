import { assert, assertEquals } from "@std/assert";
import {
  CoreferenceResolver,
  DialogState,
  InMemoryEntityStore,
  salienceTotal,
} from "./coreference.ts";

Deno.test("detect pronouns", () => {
  const resolver = new CoreferenceResolver();
  const refs = resolver.detectReferences("Fix it and run the tests");
  assert(refs.length > 0);
  assert(refs.some((r) => r.text === "it"));
  assertEquals(refs[0].ref_type.kind, "singular_neutral");
});

Deno.test("detect definite NP", () => {
  const resolver = new CoreferenceResolver();
  const refs = resolver.detectReferences("Update the file with the new logic");
  assert(refs.some((r) => r.text === "the file"));
  assert(refs.some((r) =>
    r.ref_type.kind === "definite_np" && r.ref_type.entity_type === "file"
  ));
});

Deno.test("detect demonstrative", () => {
  const resolver = new CoreferenceResolver();
  const refs = resolver.detectReferences("Fix that error in the code");
  assert(refs.some((r) => r.text === "that error"));
  assert(refs.some((r) =>
    r.ref_type.kind === "demonstrative" && r.ref_type.entity_type === "error"
  ));
});

Deno.test("dialog state mention", () => {
  const state = new DialogState();
  state.mentionEntity("main.rs", "file");
  state.nextTurn();
  state.mentionEntity("config.toml", "file");

  assertEquals(state.focus_stack[0], "config.toml");
  assertEquals(state.focus_stack[1], "main.rs");
  assert(state.recencyScore("config.toml") > state.recencyScore("main.rs"));
});

Deno.test("dialog state frequency", () => {
  const state = new DialogState();
  state.mentionEntity("main.rs", "file");
  state.nextTurn();
  state.mentionEntity("main.rs", "file");
  state.nextTurn();
  state.mentionEntity("config.toml", "file");

  assert(state.frequencyScore("main.rs") > state.frequencyScore("config.toml"));
});

Deno.test("resolve pronoun", () => {
  const resolver = new CoreferenceResolver();
  const state = new DialogState();
  const entityStore = new InMemoryEntityStore();

  state.mentionEntity("src/main.rs", "file");
  state.nextTurn();

  const refs = resolver.detectReferences("Fix it");
  const resolved = resolver.resolve(refs, state, entityStore);

  assertEquals(resolved.length, 1);
  assertEquals(resolved[0].antecedent, "src/main.rs");
});

Deno.test("resolve type-constrained", () => {
  const resolver = new CoreferenceResolver();
  const state = new DialogState();
  const entityStore = new InMemoryEntityStore();

  state.mentionEntity("main.rs", "file");
  state.mentionEntity("process_data", "function");
  state.nextTurn();

  const refs = resolver.detectReferences("Update the function");
  const resolved = resolver.resolve(refs, state, entityStore);

  assertEquals(resolved.length, 1);
  assertEquals(resolved[0].antecedent, "process_data");
});

Deno.test("rewrite with resolutions", () => {
  const resolver = new CoreferenceResolver();
  const state = new DialogState();
  const entityStore = new InMemoryEntityStore();

  state.mentionEntity("main.rs", "file");
  state.nextTurn();

  const refs = resolver.detectReferences("Fix it and test");
  const resolved = resolver.resolve(refs, state, entityStore);
  const rewritten = resolver.rewriteWithResolutions("Fix it and test", resolved);

  assertEquals(rewritten, "Fix [main.rs] and test");
});

Deno.test("salience score total", () => {
  const score = {
    recency: 1.0,
    frequency: 0.5,
    graph_centrality: 0.8,
    type_match: 1.0,
    syntactic_prominence: 0.5,
  };
  // 1.0*0.35 + 0.5*0.15 + 0.8*0.20 + 1.0*0.20 + 0.5*0.10 = 0.835
  assert(Math.abs(salienceTotal(score) - 0.835) < 0.001);
});

Deno.test("empty references", () => {
  const resolver = new CoreferenceResolver();
  const refs = resolver.detectReferences("Build the project using cargo");
  // "the project" doesn't match our patterns.
  assert(refs.length === 0 || !refs.some((r) => r.text === "the project"));
});

Deno.test("multiple references", () => {
  const resolver = new CoreferenceResolver();
  const refs = resolver.detectReferences("Fix it and update the file");
  assert(refs.length >= 2);
  const texts = refs.map((r) => r.text);
  assert(texts.includes("it"));
  assert(texts.includes("the file"));
});
