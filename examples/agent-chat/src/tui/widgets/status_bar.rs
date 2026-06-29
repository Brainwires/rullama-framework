use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

use crate::tui::app::{App, AppMode};

pub fn draw(f: &mut Frame, area: Rect, app: &App) {
    let mode_str = match app.mode {
        AppMode::Normal => "NORMAL",
        AppMode::Waiting => "WAITING",
        AppMode::HelpDialog => "HELP",
        AppMode::ExitDialog => "EXIT?",
        AppMode::ApprovalDialog => "APPROVE?",
        AppMode::ConsoleView => "CONSOLE",
        AppMode::ConversationFullscreen => "CHAT",
        AppMode::InputFullscreen => "INPUT",
    };

    let tokens_str = if app.prompt_tokens > 0 || app.completion_tokens > 0 {
        format!(" | ~{}+{} tokens", app.prompt_tokens, app.completion_tokens)
    } else {
        String::new()
    };

    let text = format!(
        " {} | {}{} | F1:Help",
        app.status_text, mode_str, tokens_str
    );

    let bar = Paragraph::new(text).style(Style::default().bg(Color::DarkGray).fg(Color::White));

    f.render_widget(bar, area);
}
