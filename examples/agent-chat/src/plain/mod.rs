use anyhow::Result;
use std::io::{self, BufRead, Write};

use crate::chat_session::{ApprovalResponse, ChatSession, StreamEvent};

pub async fn run(mut session: ChatSession) -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    // Set up approval callback for plain mode: prompt on stderr
    session.set_approval_callback(Box::new(|tool_name, input| {
        let summary = serde_json::to_string_pretty(input).unwrap_or_default();
        let truncated = if summary.len() > 200 {
            format!("{}...", &summary[..200])
        } else {
            summary
        };
        eprintln!("\n--- Tool approval ---");
        eprintln!("Tool: {tool_name}");
        eprintln!("Args: {truncated}");
        eprint!("[Y]es / [N]o / [A]lways: ");
        let _ = io::stderr().flush();

        let mut line = String::new();
        let _ = io::stdin().lock().read_line(&mut line);
        match line.trim().to_lowercase().as_str() {
            "y" | "yes" | "" => ApprovalResponse::Yes,
            "a" | "always" => ApprovalResponse::Always,
            _ => ApprovalResponse::No,
        }
    }));

    println!(
        "agent-chat ({}) - Type /help for commands, Ctrl+D to exit",
        session.provider_name()
    );
    println!();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            // EOF (Ctrl+D)
            println!();
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/exit" | "/quit" => break,
            "/help" => {
                println!("Commands:");
                println!("  /help   - Show this help");
                println!("  /clear  - Clear conversation history");
                println!("  /exit   - Exit");
                println!();
                continue;
            }
            "/clear" => {
                session.clear();
                println!("Conversation cleared.");
                continue;
            }
            _ => {}
        }

        match session.send_message(input).await {
            Ok(events) => {
                let mut printed_newline = false;
                for event in events {
                    match event {
                        StreamEvent::Text(t) => {
                            print!("{t}");
                            stdout.flush()?;
                            printed_newline = false;
                        }
                        StreamEvent::ToolCall { name, .. } => {
                            if !printed_newline {
                                println!();
                            }
                            eprintln!("[calling tool: {name}]");
                            printed_newline = true;
                        }
                        StreamEvent::ToolResult {
                            name,
                            is_error,
                            content,
                            ..
                        } => {
                            let status = if is_error { "ERROR" } else { "ok" };
                            let preview = if content.len() > 100 {
                                format!("{}...", &content[..100])
                            } else {
                                content.clone()
                            };
                            eprintln!("[{name} {status}: {preview}]");
                            printed_newline = true;
                        }
                        StreamEvent::Usage {
                            prompt_tokens,
                            completion_tokens,
                        } => {
                            eprintln!("\n[tokens: {prompt_tokens} in / {completion_tokens} out]");
                            printed_newline = true;
                        }
                    }
                }
                if !printed_newline {
                    println!();
                }
                println!();
            }
            Err(e) => {
                eprintln!("Error: {e}");
                println!();
            }
        }
    }

    Ok(())
}
