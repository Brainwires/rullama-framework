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
      { title: "rullama", href: "/crates/rullama" },
      { title: "rullama-core", href: "/crates/rullama-core" },
      { title: "rullama-providers", href: "/crates/rullama-providers" },
      { title: "rullama-agents", href: "/crates/rullama-agents" },
      { title: "rullama-cognition", href: "/crates/rullama-cognition" },
      { title: "rullama-training", href: "/crates/rullama-training" },
      { title: "rullama-storage", href: "/crates/rullama-storage" },
      { title: "rullama-mcp", href: "/crates/rullama-mcp" },
      { title: "rullama-mcp-server", href: "/crates/rullama-mcp-server" },
      { title: "rullama-agent-network", href: "/crates/rullama-agent-network" },
      { title: "rullama-tool-system", href: "/crates/rullama-tool-system" },
      { title: "rullama-skills", href: "/crates/rullama-skills" },
      { title: "rullama-hardware", href: "/crates/rullama-hardware" },
      { title: "rullama-datasets", href: "/crates/rullama-datasets" },
      { title: "rullama-autonomy", href: "/crates/rullama-autonomy" },
      { title: "rullama-permissions", href: "/crates/rullama-permissions" },
      { title: "rullama-a2a", href: "/crates/rullama-a2a" },
      { title: "rullama-channels", href: "/crates/rullama-channels" },
      { title: "rullama-code-interpreters", href: "/crates/rullama-code-interpreters" },
      { title: "rullama-analytics", href: "/crates/rullama-analytics" },
      { title: "rullama-wasm", href: "/crates/rullama-wasm" },
    ],
  },
  {
    title: "Extras", href: "/extras",
    children: [
      { title: "rullama-cli", href: "/extras/rullama-cli" },
      { title: "rullama-proxy", href: "/extras/rullama-proxy" },
      { title: "rullama-brain-server", href: "/extras/rullama-brain-server" },
      { title: "rullama-rag-server", href: "/extras/rullama-rag-server" },
      { title: "rullama-issues", href: "/extras/rullama-issues" },
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
