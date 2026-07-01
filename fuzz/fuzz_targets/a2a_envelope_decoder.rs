#![no_main]
//! Fuzz target for `rullama_a2a::agent_card::AgentCard` deserialisation.
//!
//! AgentCards are exchanged between federated agents via the A2A protocol;
//! a malformed peer-supplied card must not crash the parser. The actual
//! signature verification lives in consumer code; this target attacks only
//! the serde envelope layer.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = serde_json::from_slice::<rullama_a2a::agent_card::AgentCard>(data);
});
