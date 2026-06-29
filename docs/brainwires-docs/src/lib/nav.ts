export interface NavItem {
  title: string;
  href?: string;
  children?: NavItem[];
}

export const NAV_TREE: NavItem[] = [
  { title: "Getting Started", href: "/" },
  { title: "Features", href: "/docs/features" },
  {
    title: "Guides",
    children: [
      { title: "Extension Points", href: "/docs/extensibility" },
      { title: "Contributing", href: "/docs/contributing" },
      { title: "Testing", href: "/docs/testing" },
      { title: "Publishing", href: "/docs/publishing" },
      { title: "Distributed Training", href: "/docs/wishlist/distributed-training" },
    ],
  },
  {
    title: "Framework Crates", href: "/crates",
    children: [
      { title: "brainwires", href: "/crates/brainwires" },
      { title: "brainwires-core", href: "/crates/brainwires-core" },
      { title: "brainwires-providers", href: "/crates/brainwires-providers" },
      { title: "brainwires-agents", href: "/crates/brainwires-agents" },
      { title: "brainwires-cognition", href: "/crates/brainwires-cognition" },
      { title: "brainwires-training", href: "/crates/brainwires-training" },
      { title: "brainwires-storage", href: "/crates/brainwires-storage" },
      { title: "brainwires-mcp", href: "/crates/brainwires-mcp" },
      { title: "brainwires-mcp-server", href: "/crates/brainwires-mcp-server" },
      { title: "brainwires-agent-network", href: "/crates/brainwires-agent-network" },
      { title: "brainwires-tool-system", href: "/crates/brainwires-tool-system" },
      { title: "brainwires-skills", href: "/crates/brainwires-skills" },
      { title: "brainwires-hardware", href: "/crates/brainwires-hardware" },
      { title: "brainwires-datasets", href: "/crates/brainwires-datasets" },
      { title: "brainwires-autonomy", href: "/crates/brainwires-autonomy" },
      { title: "brainwires-permissions", href: "/crates/brainwires-permissions" },
      { title: "brainwires-a2a", href: "/crates/brainwires-a2a" },
      { title: "brainwires-channels", href: "/crates/brainwires-channels" },
      { title: "brainwires-code-interpreters", href: "/crates/brainwires-code-interpreters" },
      { title: "brainwires-analytics", href: "/crates/brainwires-analytics" },
      { title: "brainwires-wasm", href: "/crates/brainwires-wasm" },
    ],
  },
  {
    title: "Extras", href: "/extras",
    children: [
      { title: "brainwires-cli", href: "/extras/brainwires-cli" },
      { title: "brainwires-proxy", href: "/extras/brainwires-proxy" },
      { title: "brainwires-brain-server", href: "/extras/brainwires-brain-server" },
      { title: "brainwires-rag-server", href: "/extras/brainwires-rag-server" },
      { title: "brainwires-issues", href: "/extras/brainwires-issues" },
      { title: "agent-chat", href: "/extras/agent-chat" },
      { title: "audio-demo", href: "/extras/audio-demo" },
      { title: "audio-demo-ffi", href: "/extras/audio-demo-ffi" },
      { title: "reload-daemon", href: "/extras/reload-daemon" },
    ],
  },
  {
    title: "Deno SDK", href: "/deno",
    children: [
      { title: "Getting Started", href: "/deno/getting-started" },
      { title: "Architecture", href: "/deno/architecture" },
      { title: "Agents", href: "/deno/agents" },
      { title: "Providers", href: "/deno/providers" },
      { title: "Tools", href: "/deno/tools" },
      { title: "Storage", href: "/deno/storage" },
      { title: "Cognition", href: "/deno/cognition" },
      { title: "Networking", href: "/deno/networking" },
      { title: "A2A Protocol", href: "/deno/a2a" },
      { title: "Permissions", href: "/deno/permissions" },
      { title: "Extensibility", href: "/deno/extensibility" },
    ],
  },
  { title: "Changelog", href: "/changelog" },
  { title: "API Reference", href: "/api-docs" },
];
