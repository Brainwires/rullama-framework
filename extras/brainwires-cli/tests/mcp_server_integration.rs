//! Integration tests for MCP server functionality

use anyhow::Result;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// Test that MCP server starts and responds to initialize
#[test]
#[ignore] // Requires MCP server infrastructure - run manually
fn test_mcp_server_initialize() -> Result<()> {
    let mut child = Command::new("cargo")
        .args(["run", "--", "chat", "--mcp-server"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdin = child.stdin.as_mut().expect("Failed to open stdin");
    let stdout = child.stdout.take().expect("Failed to open stdout");

    // Send initialize request
    let init_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0"
            }
        }
    });

    writeln!(stdin, "{}", init_request)?;
    stdin.flush()?;

    // Read response
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();

    // Skip log lines and find JSON response
    let mut response_line = None;
    for line in lines.by_ref() {
        let line = line?;
        if line.starts_with('{') && line.contains("\"jsonrpc\"") {
            response_line = Some(line);
            break;
        }
    }

    let response_line = response_line.expect("No JSON-RPC response found");
    let response: serde_json::Value = serde_json::from_str(&response_line)?;

    // Verify response structure
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], 1);
    assert!(response["result"].is_object());
    assert_eq!(response["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(response["result"]["serverInfo"]["name"], "brainwires-cli");
    assert!(response["result"]["capabilities"]["tools"].is_object());

    // Clean up
    let _ = stdin;
    let _ = child.wait();

    Ok(())
}

/// Test that MCP server lists tools
#[test]
#[ignore] // Requires MCP server infrastructure - run manually
fn test_mcp_server_list_tools() -> Result<()> {
    let mut child = Command::new("cargo")
        .args(["run", "--", "chat", "--mcp-server"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdin = child.stdin.as_mut().expect("Failed to open stdin");
    let stdout = child.stdout.take().expect("Failed to open stdout");

    // Send initialize first
    let init_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    });
    writeln!(stdin, "{}", init_request)?;

    // Send tools/list request
    let list_request = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    });
    writeln!(stdin, "{}", list_request)?;
    stdin.flush()?;

    // Read responses
    let reader = BufReader::new(stdout);
    let mut json_responses = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('{')
            && line.contains("\"jsonrpc\"")
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(&line)
        {
            json_responses.push(value);
            if json_responses.len() >= 2 {
                break;
            }
        }
    }

    // Find the tools/list response (id: 2)
    let tools_response = json_responses
        .iter()
        .find(|r| r["id"] == 2)
        .expect("No tools/list response found");

    // Verify response
    assert_eq!(tools_response["jsonrpc"], "2.0");
    assert!(tools_response["result"]["tools"].is_array());

    let tools = tools_response["result"]["tools"].as_array().unwrap();
    assert!(!tools.is_empty(), "Should have at least some tools");

    // Check for agent management tools
    let tool_names: Vec<String> = tools
        .iter()
        .filter_map(|t| t["name"].as_str().map(String::from))
        .collect();

    assert!(
        tool_names.contains(&"agent_spawn".to_string()),
        "Should include agent_spawn tool"
    );
    assert!(
        tool_names.contains(&"agent_list".to_string()),
        "Should include agent_list tool"
    );
    assert!(
        tool_names.contains(&"agent_status".to_string()),
        "Should include agent_status tool"
    );
    assert!(
        tool_names.contains(&"agent_stop".to_string()),
        "Should include agent_stop tool"
    );

    // Clean up
    let _ = stdin;
    let _ = child.wait();

    Ok(())
}

/// Test error handling for invalid method
#[test]
#[ignore] // Requires MCP server infrastructure - run manually
fn test_mcp_server_invalid_method() -> Result<()> {
    let mut child = Command::new("cargo")
        .args(["run", "--", "chat", "--mcp-server"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdin = child.stdin.as_mut().expect("Failed to open stdin");
    let stdout = child.stdout.take().expect("Failed to open stdout");

    // Send request with invalid method
    let invalid_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "invalid/method",
        "params": {}
    });
    writeln!(stdin, "{}", invalid_request)?;
    stdin.flush()?;

    // Read response
    let reader = BufReader::new(stdout);
    let mut response_line = None;

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('{') && line.contains("\"jsonrpc\"") {
            response_line = Some(line);
            break;
        }
    }

    let response_line = response_line.expect("No response found");
    let response: serde_json::Value = serde_json::from_str(&response_line)?;

    // Verify error response
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], 1);
    assert!(response["error"].is_object());
    assert_eq!(response["error"]["code"], -32601); // Method not found

    // Clean up
    let _ = stdin;
    let _ = child.wait();

    Ok(())
}

/// Test that agent tools have correct schemas
#[test]
#[ignore] // Requires MCP server infrastructure - run manually
fn test_agent_tool_schemas() -> Result<()> {
    let mut child = Command::new("cargo")
        .args(["run", "--", "chat", "--mcp-server"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdin = child.stdin.as_mut().expect("Failed to open stdin");
    let stdout = child.stdout.take().expect("Failed to open stdout");

    // Initialize and list tools
    let init = json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}});
    writeln!(stdin, "{}", init)?;

    let list = json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}});
    writeln!(stdin, "{}", list)?;
    stdin.flush()?;

    // Read responses
    let reader = BufReader::new(stdout);
    let mut json_responses = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('{')
            && line.contains("\"jsonrpc\"")
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(&line)
        {
            json_responses.push(value);
            if json_responses.len() >= 2 {
                break;
            }
        }
    }

    let tools_response = json_responses
        .iter()
        .find(|r| r["id"] == 2)
        .expect("No tools response");

    let tools = tools_response["result"]["tools"].as_array().unwrap();

    // Find agent_spawn and verify its schema
    let agent_spawn = tools
        .iter()
        .find(|t| t["name"] == "agent_spawn")
        .expect("agent_spawn not found");

    assert!(
        agent_spawn["description"]
            .as_str()
            .unwrap()
            .contains("autonomous"),
        "Should describe autonomous behavior"
    );
    assert!(agent_spawn["inputSchema"].is_object());
    assert_eq!(agent_spawn["inputSchema"]["type"], "object");
    assert!(agent_spawn["inputSchema"]["properties"]["description"].is_object());
    assert_eq!(
        agent_spawn["inputSchema"]["properties"]["description"]["type"],
        "string"
    );
    assert!(
        agent_spawn["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .contains(&json!("description"))
    );

    // Find agent_status and verify its schema
    let agent_status = tools
        .iter()
        .find(|t| t["name"] == "agent_status")
        .expect("agent_status not found");

    assert!(agent_status["inputSchema"]["properties"]["agent_id"].is_object());
    assert!(
        agent_status["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .contains(&json!("agent_id"))
    );

    // Clean up
    let _ = stdin;
    let _ = child.wait();

    Ok(())
}
