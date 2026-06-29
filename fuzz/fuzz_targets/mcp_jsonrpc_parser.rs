#![no_main]
//! Fuzz target for `rullama_mcp_client::JsonRpcRequest` deserialisation.
//!
//! The MCP server's request handler ultimately invokes `serde_json::from_slice`
//! on attacker-controlled bytes. This target throws raw bytes at the
//! deserialiser and asserts only that we don't panic, OOM, or stack-overflow.
//! Discovered crashes get archived under `fuzz/corpus/mcp_jsonrpc_parser/`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The framework currently parses MCP requests by deserialising into
    // `JsonRpcRequest`. Any panic, OOM, or stack-overflow here is a bug.
    let _ = serde_json::from_slice::<rullama_mcp_client::JsonRpcRequest>(data);
});
