use crate::utils::cost_tracker::CostTracker;
use anyhow::Result;

pub async fn handle_cost(period: Option<String>, reset: bool) -> Result<()> {
    if reset {
        // Drop any persisted usage by saving a fresh tracker over it.
        let fresh = CostTracker::new();
        fresh.save().await?;
        println!("Cost tracker reset.");
        return Ok(());
    }

    let tracker = CostTracker::load().await?;
    // Default to today so a quick `brainwires cost` after a prompt shows data.
    let period = period.unwrap_or_else(|| "today".to_string());
    let summary = tracker.get_usage_summary(&period);

    if summary == "No usage data" {
        // Be honest about *why* the user might not see data. Writing is wired up
        // in `cli/chat/streaming.rs`, but only fires when the provider emits a
        // `StreamChunk::Usage` event. Some transports (notably the brainwires
        // SaaS relay over HTTP) may not yet surface token counts per request.
        println!(
            "\nNo usage data for period '{}'.\n\n\
             Usage is recorded when the provider emits token counts on the \
             response stream. If you've run chats recently and still see this:\n  \
             - The brainwires SaaS provider does not always forward per-request \
               token counts from upstream; try a direct provider \
               (`--provider anthropic` or `--provider openai`) to verify.\n  \
             - Try another period: `brainwires cost --period week|month|30days`.\n",
            period
        );
    } else {
        println!("\n{}", summary);
    }
    Ok(())
}
