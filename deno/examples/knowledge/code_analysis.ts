// Example: Code Analysis
// Demonstrates symbol extraction, reference finding, call graph construction, and repo map formatting.
// Run: deno run deno/examples/cognition/code_analysis.ts

import {
  buildCallGraph,
  CallGraph,
  createSymbolId,
  definitionToStorageId,
  findReferences,
  RepoMap,
  symbolIdToStorageId,
  symbolKindDisplayName,
} from "@rullama/knowledge";

import type {
  CallEdge,
  CallGraphNode,
  CodeAnalysisDefinition,
  SymbolKind,
} from "@rullama/knowledge";

async function main() {
  console.log("=== rullama Code Analysis Example ===\n");

  // 1. Check supported languages
  console.log("--- Step 1: Language Support ---\n");

  const extensions = RepoMap.supportedExtensions();
  console.log(`Supported extensions: ${extensions.join(", ")}`);
  console.log();

  const testExts = ["ts", "js", "py", "rs", "go", "java"];
  console.log(`${"Extension".padEnd(12)} ${"Supported".padEnd(12)} Language`);
  console.log("-".repeat(44));
  for (const ext of testExts) {
    const supported = RepoMap.supportsExtension(ext);
    const language = RepoMap.languageForExtension(ext) ?? "(none)";
    console.log(
      `${ext.padEnd(12)} ${(supported ? "yes" : "no").padEnd(12)} ${language}`,
    );
  }
  console.log();

  // 2. Extract definitions from a TypeScript source snippet
  console.log("--- Step 2: Extract Definitions ---\n");

  const tsSource = `
import { EventEmitter } from "events";

/** Configuration for the application. */
export interface Config {
  host: string;
  port: number;
  debug: boolean;
}

/** Create a new Config with defaults. */
export function createConfig(): Config {
  return {
    host: "localhost",
    port: 8080,
    debug: false,
  };
}

/** Validate the configuration. */
export function validateConfig(config: Config): boolean {
  if (config.port === 0) {
    throw new Error("port must be non-zero");
  }
  return true;
}

/** Process an incoming request using the given config. */
export function processRequest(config: Config, data: string): string {
  return \`Processed on \${config.host}:\${config.port}: \${data}\`;
}

/** Start the main server loop. */
function main() {
  const config = createConfig();
  validateConfig(config);
  const result = processRequest(config, "hello world");
  console.log(result);
}
`;

  const definitions = RepoMap.extractSymbols({
    filePath: "src/server.ts",
    content: tsSource,
    rootPath: "/demo",
    project: "demo",
  });

  console.log(`Found ${definitions.length} definitions in src/server.ts:\n`);
  console.log(
    `${"Line".padEnd(6)} ${"Name".padEnd(22)} ${"Kind".padEnd(12)} ${
      "Visibility".padEnd(12)
    } Signature`,
  );
  console.log("-".repeat(90));

  for (const def of definitions) {
    const sigPreview = def.signature.length > 40
      ? def.signature.slice(0, 40) + "..."
      : def.signature;

    console.log(
      `${String(def.symbolId.startLine).padEnd(6)} ` +
        `${def.symbolId.name.padEnd(22)} ` +
        `${symbolKindDisplayName(def.symbolId.kind).padEnd(12)} ` +
        `${def.visibility.padEnd(12)} ` +
        `${sigPreview}`,
    );
  }
  console.log();

  // 3. Build a symbol index and find references
  console.log("--- Step 3: Find References ---\n");

  const symbolIndex = new Map<string, CodeAnalysisDefinition[]>();
  for (const def of definitions) {
    const existing = symbolIndex.get(def.symbolId.name) ?? [];
    existing.push(def);
    symbolIndex.set(def.symbolId.name, existing);
  }

  const references = findReferences(
    "src/server.ts",
    tsSource,
    symbolIndex,
    "/demo",
    "demo",
  );

  console.log(`Found ${references.length} references:\n`);
  console.log(
    `${"Line".padEnd(6)} ${"Col".padEnd(8)} ${"Kind".padEnd(18)} Target`,
  );
  console.log("-".repeat(60));

  for (const ref of references) {
    console.log(
      `${String(ref.startLine).padEnd(6)} ` +
        `${String(ref.startCol).padEnd(8)} ` +
        `${ref.referenceKind.padEnd(18)} ` +
        `${ref.targetSymbolId}`,
    );
  }
  console.log();

  // 4. Symbol identification
  console.log("--- Step 4: Symbol Identification ---\n");

  const symConfig = createSymbolId(
    "src/server.ts",
    "Config",
    "interface",
    5,
    0,
  );
  const symCreate = createSymbolId(
    "src/server.ts",
    "createConfig",
    "function",
    12,
    0,
  );
  const symProcess = createSymbolId(
    "src/server.ts",
    "processRequest",
    "function",
    30,
    0,
  );

  console.log("Symbol IDs (for storage/lookup):");
  console.log(`  Config:          ${symbolIdToStorageId(symConfig)}`);
  console.log(`  createConfig:    ${symbolIdToStorageId(symCreate)}`);
  console.log(`  processRequest:  ${symbolIdToStorageId(symProcess)}`);
  console.log();

  if (definitions.length > 0) {
    const first = definitions[0];
    const storageId = definitionToStorageId(first);
    console.log(`First definition storage ID: ${storageId}`);
    console.log(`  Name:      ${first.symbolId.name}`);
    console.log(`  File:      ${first.symbolId.filePath}`);
    console.log(`  Kind:      ${symbolKindDisplayName(first.symbolId.kind)}`);
    console.log(`  Lines:     ${first.symbolId.startLine}-${first.endLine}`);
    console.log(`  Signature: ${first.signature}`);
    if (first.docComment) {
      console.log(`  Docs:      ${first.docComment}`);
    }
  }
  console.log();

  // 5. Build a call graph
  console.log("--- Step 5: Call Graph ---\n");

  const files = new Map<string, string>();
  files.set("src/server.ts", tsSource);

  const callGraph = buildCallGraph(definitions, files);
  console.log(
    `Call graph: ${callGraph.nodes.size} nodes, ${callGraph.edges.length} edges`,
  );
  console.log();

  console.log("Call edges:");
  for (const edge of callGraph.edges) {
    const callerName = edge.callerId.split(":")[1] ?? edge.callerId;
    const calleeName = edge.calleeId.split(":")[1] ?? edge.calleeId;
    console.log(
      `  ${callerName} (line ${edge.callSiteLine}) -> ${calleeName}`,
    );
  }
  console.log();

  // Display call tree from a root node
  for (const [nodeId] of callGraph.nodes) {
    const tree = callGraph.calleeTree(nodeId, 3);
    if (tree && tree.children.length > 0) {
      console.log(`Call tree from ${tree.name}:`);
      printCallTree(tree, 0);
      console.log();
    }
  }

  // 6. Repo map formatting
  console.log("--- Step 6: Repo Map ---\n");

  const repoMap = RepoMap.formatRepoMap(definitions);
  console.log(repoMap);

  console.log("\n=== Done ===");
}

/** Recursively print a call graph tree with indentation. */
function printCallTree(node: CallGraphNode, depth: number): void {
  const indent = "  ".repeat(depth);
  console.log(
    `${indent}${node.name} (${
      symbolKindDisplayName(node.kind)
    }) -- ${node.filePath}:${node.line}`,
  );
  for (const child of node.children) {
    printCallTree(child, depth + 1);
  }
}

await main();
