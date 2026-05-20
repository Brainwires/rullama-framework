//! Example: MCP Client configuration and API overview
//!
//! Demonstrates how to configure `McpClient` and `McpServerConfig`,
//! and shows the full API surface for connecting to MCP servers, listing
//! tools/resources/prompts, and calling tools. Since we cannot connect to
//! a real MCP server in this example, we demonstrate the configuration and
//! API patterns using mock data.
//!
//! Run: cargo run -p brainwires-mcp --example mcp_client --features native

use brainwires_mcp_client::{McpClient, McpServerConfig};

#[tokio::main]
async fn main() {
    println!("=== MCP Client Example ===\n");

    // ── 1. Configure MCP servers ────────────────────────────────────────
    println!("--- Step 1: Server Configuration ---");

    // Define MCP server configurations
    let filesystem_server = McpServerConfig {
        name: "filesystem".to_string(),
        command: "npx".to_string(),
        args: vec![
            "-y".to_string(),
            "@modelcontextprotocol/server-filesystem".to_string(),
            "/tmp".to_string(),
        ],
        env: None,
    };
    println!("  Server: {}", filesystem_server.name);
    println!(
        "  Command: {} {}",
        filesystem_server.command,
        filesystem_server.args.join(" ")
    );
    println!();

    // Server with environment variables
    let mut env_vars = std::collections::HashMap::new();
    env_vars.insert("GITHUB_TOKEN".to_string(), "ghp_demo_token".to_string());

    let github_server = McpServerConfig {
        name: "github".to_string(),
        command: "npx".to_string(),
        args: vec![
            "-y".to_string(),
            "@modelcontextprotocol/server-github".to_string(),
        ],
        env: Some(env_vars),
    };
    println!("  Server: {}", github_server.name);
    println!(
        "  Command: {} {}",
        github_server.command,
        github_server.args.join(" ")
    );
    println!(
        "  Env vars: {:?}",
        github_server
            .env
            .as_ref()
            .map(|e| e.keys().collect::<Vec<_>>())
    );
    println!();

    // Configs serialize to JSON for persistence
    let json = serde_json::to_string_pretty(&filesystem_server).unwrap();
    println!("  Serialized config:");
    println!("  {}", json);
    println!();

    // ── 2. Create the MCP client ────────────────────────────────────────
    println!("--- Step 2: Create McpClient ---");

    let client = McpClient::new("brainwires-example", "0.1.0");
    println!("  Created McpClient");
    println!("  Connected servers: {:?}", client.list_connected().await);
    println!();

    // ── 3. Demonstrate the connection API (without real servers) ─────────
    println!("--- Step 3: Connection API Overview ---");
    println!("  The McpClient provides these async methods:\n");

    println!("  // Connect to a server (spawns process, performs MCP handshake)");
    println!("  client.connect(&server_config).await?;\n");

    println!("  // Check connection status");
    println!("  client.is_connected(\"filesystem\").await;\n");

    println!("  // List all connected servers");
    println!("  client.list_connected().await;\n");

    // ── 4. Demonstrate the tool API ─────────────────────────────────────
    println!("--- Step 4: Tool API ---");
    println!("  // List available tools from a connected server");
    println!("  let tools = client.list_tools(\"filesystem\").await?;");
    println!("  for tool in &tools {{");
    println!("      println!(\"Tool: {{:?}}\", tool);");
    println!("  }}\n");

    println!("  // Call a tool with JSON arguments");
    println!("  let result = client.call_tool(");
    println!("      \"filesystem\",");
    println!("      \"read_file\",");
    println!("      Some(serde_json::json!({{ \"path\": \"/tmp/test.txt\" }})),");
    println!("  ).await?;\n");

    // Show what a tool call looks like with mock arguments
    let mock_args = serde_json::json!({
        "path": "/tmp/example.txt",
        "encoding": "utf-8"
    });
    println!("  Mock tool arguments:");
    println!("  {}\n", serde_json::to_string_pretty(&mock_args).unwrap());

    // ── 5. Demonstrate resources and prompts API ────────────────────────
    println!("--- Step 5: Resources & Prompts API ---");
    println!("  // List resources exposed by a server");
    println!("  let resources = client.list_resources(\"filesystem\").await?;\n");

    println!("  // Read a specific resource by URI");
    println!(
        "  let content = client.read_resource(\"filesystem\", \"file:///tmp/data.json\").await?;\n"
    );

    println!("  // List prompt templates");
    println!("  let prompts = client.list_prompts(\"github\").await?;\n");

    println!("  // Get a prompt with arguments");
    println!("  let prompt = client.get_prompt(");
    println!("      \"github\",");
    println!("      \"review_pr\",");
    println!("      Some(serde_json::json!({{ \"pr_number\": 42 }})),");
    println!("  ).await?;\n");

    // ── 6. Server info and capabilities ─────────────────────────────────
    println!("--- Step 6: Server Info & Capabilities ---");
    println!("  // After connecting, query server metadata");
    println!("  let info = client.get_server_info(\"filesystem\").await?;");
    println!("  println!(\"Server: {{}} v{{}}\", info.name, info.version);\n");

    println!("  let caps = client.get_capabilities(\"filesystem\").await?;");
    println!("  // Capabilities indicate which features the server supports:");
    println!("  // - tools: server exposes callable tools");
    println!("  // - resources: server exposes readable resources");
    println!("  // - prompts: server exposes prompt templates\n");

    // ── 7. Disconnection and cancellation ───────────────────────────────
    println!("--- Step 7: Cleanup ---");
    println!("  // Disconnect from a specific server");
    println!("  client.disconnect(\"filesystem\").await?;\n");

    println!("  // Cancel a pending request (JSON-RPC $/cancelRequest)");
    println!("  client.cancel_request(\"github\", request_id).await?;\n");

    // Verify no connections remain
    println!(
        "  Final connected servers: {:?}",
        client.list_connected().await
    );

    println!("\nDone! To connect to real MCP servers, ensure the server");
    println!(
        "command is installed (e.g., `npm install -g @modelcontextprotocol/server-filesystem`)"
    );
    println!("and call client.connect(&config).await to establish a connection.");
}
