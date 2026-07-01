//! Agent card discovery from well-known URL.

use crate::agent_card::AgentCard;
use crate::error::A2aError;

/// Fetch an agent card from the well-known discovery endpoint.
///
/// Fetches `{base_url}/.well-known/agent-card.json`.
#[cfg(feature = "client")]
pub async fn discover_agent_card(base_url: &str) -> Result<AgentCard, A2aError> {
    let url = format!(
        "{}/.well-known/agent-card.json",
        base_url.trim_end_matches('/')
    );

    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| A2aError::internal(format!("Discovery request failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(A2aError::internal(format!(
            "Discovery failed with status: {}",
            resp.status()
        )));
    }

    let card: AgentCard = resp
        .json()
        .await
        .map_err(|e| A2aError::internal(format!("Failed to parse agent card: {e}")))?;

    Ok(card)
}
