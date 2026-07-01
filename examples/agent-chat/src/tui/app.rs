use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::prelude::*;
use std::time::Duration;
use tokio::sync::mpsc;

use super::render;
use crate::chat_session::{ApprovalResponse, ChatSession, StreamEvent};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    Normal,
    Waiting,
    HelpDialog,
    ExitDialog,
    ApprovalDialog,
    ConsoleView,
    ConversationFullscreen,
    InputFullscreen,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
    Tool,
    Error,
}

pub struct PendingApproval {
    pub tool_name: String,
    pub input: serde_json::Value,
    pub response_tx: tokio::sync::oneshot::Sender<ApprovalResponse>,
}

pub struct App {
    pub session: ChatSession,
    pub mode: AppMode,
    pub input: String,
    pub cursor_pos: usize,
    pub display_messages: Vec<ChatMessage>,
    pub scroll_offset: u16,
    pub status_text: String,
    pub console_log: Vec<String>,
    pub pending_approval: Option<PendingApproval>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    event_rx: Option<mpsc::UnboundedReceiver<AppEvent>>,
}

pub enum AppEvent {
    StreamEvent(StreamEvent),
    StreamDone,
    StreamError(String),
    ApprovalRequest {
        tool_name: String,
        input: serde_json::Value,
        response_tx: tokio::sync::oneshot::Sender<ApprovalResponse>,
    },
}

impl App {
    pub fn new(session: ChatSession) -> Self {
        let status_text = session.provider_name().to_string();
        Self {
            session,
            mode: AppMode::Normal,
            input: String::new(),
            cursor_pos: 0,
            display_messages: Vec::new(),
            scroll_offset: 0,
            status_text,
            console_log: Vec::new(),
            pending_approval: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            event_rx: None,
        }
    }

    pub async fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<()> {
        // Set up approval callback that sends through channel
        let (approval_tx, mut approval_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        self.event_rx = Some(event_rx);

        // We'll wire up the approval callback on the session
        // The callback sends to approval_tx which we forward to event_tx
        let event_tx_clone = event_tx.clone();
        tokio::spawn(async move {
            while let Some((tool_name, input, resp_tx)) = approval_rx.recv().await {
                let _ = event_tx_clone.send(AppEvent::ApprovalRequest {
                    tool_name,
                    input,
                    response_tx: resp_tx,
                });
            }
        });

        loop {
            terminal.draw(|f| render::draw(f, self))?;

            // Poll for crossterm events with a short timeout
            if event::poll(Duration::from_millis(50))?
                && let Event::Key(key) = event::read()?
                && self.handle_key(key, &event_tx, &approval_tx).await?
            {
                break;
            }

            // Process any pending app events
            let mut pending_events = Vec::new();
            if let Some(ref mut rx) = self.event_rx {
                while let Ok(ev) = rx.try_recv() {
                    pending_events.push(ev);
                }
            }
            for ev in pending_events {
                self.handle_app_event(ev);
            }
        }

        Ok(())
    }

    async fn handle_key(
        &mut self,
        key: event::KeyEvent,
        event_tx: &mpsc::UnboundedSender<AppEvent>,
        _approval_tx: &mpsc::UnboundedSender<(
            String,
            serde_json::Value,
            tokio::sync::oneshot::Sender<ApprovalResponse>,
        )>,
    ) -> Result<bool> {
        match self.mode {
            AppMode::ExitDialog => match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => return Ok(true),
                _ => self.mode = AppMode::Normal,
            },
            AppMode::HelpDialog => {
                self.mode = AppMode::Normal;
            }
            AppMode::ApprovalDialog => match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    if let Some(pending) = self.pending_approval.take() {
                        let _ = pending.response_tx.send(ApprovalResponse::Yes);
                    }
                    self.mode = AppMode::Waiting;
                }
                KeyCode::Char('n') | KeyCode::Char('N') => {
                    if let Some(pending) = self.pending_approval.take() {
                        let _ = pending.response_tx.send(ApprovalResponse::No);
                    }
                    self.mode = AppMode::Waiting;
                }
                KeyCode::Char('a') | KeyCode::Char('A') => {
                    if let Some(pending) = self.pending_approval.take() {
                        let _ = pending.response_tx.send(ApprovalResponse::Always);
                    }
                    self.mode = AppMode::Waiting;
                }
                _ => {}
            },
            AppMode::ConsoleView => match key.code {
                KeyCode::Esc | KeyCode::Char('q') => self.mode = AppMode::Normal,
                _ => {}
            },
            AppMode::ConversationFullscreen => match key.code {
                KeyCode::Esc => self.mode = AppMode::Normal,
                KeyCode::Up => {
                    self.scroll_offset = self.scroll_offset.saturating_add(1);
                }
                KeyCode::Down => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(1);
                }
                _ => {}
            },
            AppMode::InputFullscreen => match key.code {
                KeyCode::Esc => self.mode = AppMode::Normal,
                _ => self.handle_input_key(key),
            },
            AppMode::Waiting => {
                // While waiting for response, only Ctrl+C works
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    self.mode = AppMode::ExitDialog;
                }
            }
            AppMode::Normal => match key.code {
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.mode = AppMode::ExitDialog;
                }
                KeyCode::F(1) => {
                    self.mode = AppMode::HelpDialog;
                }
                KeyCode::F(2) => {
                    self.mode = AppMode::ConsoleView;
                }
                KeyCode::F(3) => {
                    self.mode = AppMode::ConversationFullscreen;
                }
                KeyCode::F(4) => {
                    self.mode = AppMode::InputFullscreen;
                }
                KeyCode::Enter => {
                    if !self.input.trim().is_empty() {
                        self.submit_message(event_tx).await;
                    }
                }
                KeyCode::Up => {
                    self.scroll_offset = self.scroll_offset.saturating_add(1);
                }
                KeyCode::Down => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(1);
                }
                _ => self.handle_input_key(key),
            },
        }
        Ok(false)
    }

    fn handle_input_key(&mut self, key: event::KeyEvent) {
        match key.code {
            KeyCode::Char(c) => {
                self.input.insert(self.cursor_pos, c);
                self.cursor_pos += c.len_utf8();
            }
            KeyCode::Backspace => {
                if self.cursor_pos > 0 {
                    let prev = self.input[..self.cursor_pos]
                        .chars()
                        .last()
                        .map(|c| c.len_utf8())
                        .unwrap_or(0);
                    self.cursor_pos -= prev;
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Delete => {
                if self.cursor_pos < self.input.len() {
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Left => {
                if self.cursor_pos > 0 {
                    let prev = self.input[..self.cursor_pos]
                        .chars()
                        .last()
                        .map(|c| c.len_utf8())
                        .unwrap_or(0);
                    self.cursor_pos -= prev;
                }
            }
            KeyCode::Right => {
                if self.cursor_pos < self.input.len() {
                    let next = self.input[self.cursor_pos..]
                        .chars()
                        .next()
                        .map(|c| c.len_utf8())
                        .unwrap_or(0);
                    self.cursor_pos += next;
                }
            }
            KeyCode::Home => self.cursor_pos = 0,
            KeyCode::End => self.cursor_pos = self.input.len(),
            _ => {}
        }
    }

    async fn submit_message(&mut self, event_tx: &mpsc::UnboundedSender<AppEvent>) {
        let input = self.input.clone();
        self.input.clear();
        self.cursor_pos = 0;
        self.scroll_offset = 0;

        // Handle slash commands
        match input.trim() {
            "/clear" => {
                self.session.clear();
                self.display_messages.clear();
                return;
            }
            "/help" => {
                self.mode = AppMode::HelpDialog;
                return;
            }
            "/exit" | "/quit" => {
                self.mode = AppMode::ExitDialog;
                return;
            }
            _ => {}
        }

        self.display_messages.push(ChatMessage {
            role: MessageRole::User,
            content: input.clone(),
        });

        self.mode = AppMode::Waiting;

        // Spawn the provider call
        let event_tx = event_tx.clone();
        // We need to get a mutable ref to session, but we're borrowing self.
        // Use a channel-based approach: collect events from send_message
        let result = self.session.send_message(&input).await;

        match result {
            Ok(events) => {
                for ev in events {
                    let _ = event_tx.send(AppEvent::StreamEvent(ev));
                }
                let _ = event_tx.send(AppEvent::StreamDone);
            }
            Err(e) => {
                let _ = event_tx.send(AppEvent::StreamError(e.to_string()));
            }
        }
    }

    fn handle_app_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::StreamEvent(se) => match se {
                StreamEvent::Text(t) => {
                    if let Some(last) = self.display_messages.last_mut()
                        && last.role == MessageRole::Assistant
                    {
                        last.content.push_str(&t);
                        return;
                    }
                    self.display_messages.push(ChatMessage {
                        role: MessageRole::Assistant,
                        content: t,
                    });
                }
                StreamEvent::ToolCall { name, input } => {
                    let summary = serde_json::to_string(&input).unwrap_or_default();
                    let truncated = if summary.len() > 100 {
                        format!("{}...", &summary[..100])
                    } else {
                        summary
                    };
                    self.display_messages.push(ChatMessage {
                        role: MessageRole::Tool,
                        content: format!("[calling {name}: {truncated}]"),
                    });
                    self.console_log.push(format!("Tool call: {name}"));
                }
                StreamEvent::ToolResult {
                    name,
                    content,
                    is_error,
                } => {
                    let status = if is_error { "ERROR" } else { "ok" };
                    let preview = if content.len() > 200 {
                        format!("{}...", &content[..200])
                    } else {
                        content.clone()
                    };
                    self.display_messages.push(ChatMessage {
                        role: MessageRole::Tool,
                        content: format!("[{name} {status}: {preview}]"),
                    });
                }
                StreamEvent::Usage {
                    prompt_tokens,
                    completion_tokens,
                } => {
                    self.prompt_tokens = prompt_tokens;
                    self.completion_tokens = completion_tokens;
                }
            },
            AppEvent::StreamDone => {
                self.mode = AppMode::Normal;
            }
            AppEvent::StreamError(e) => {
                self.display_messages.push(ChatMessage {
                    role: MessageRole::Error,
                    content: format!("Error: {e}"),
                });
                self.mode = AppMode::Normal;
            }
            AppEvent::ApprovalRequest {
                tool_name,
                input,
                response_tx,
            } => {
                self.pending_approval = Some(PendingApproval {
                    tool_name,
                    input,
                    response_tx,
                });
                self.mode = AppMode::ApprovalDialog;
            }
        }
    }
}
