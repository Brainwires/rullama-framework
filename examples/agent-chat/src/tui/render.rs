use ratatui::prelude::*;

use super::app::{App, AppMode};
use super::widgets;

pub fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),    // Chat history
            Constraint::Length(3), // Input
            Constraint::Length(1), // Status bar
        ])
        .split(f.area());

    match app.mode {
        AppMode::ConversationFullscreen => {
            widgets::chat_view::draw(f, f.area(), app);
        }
        AppMode::InputFullscreen => {
            widgets::input_area::draw_fullscreen(f, f.area(), app);
        }
        AppMode::ConsoleView => {
            widgets::console_view::draw(f, f.area(), app);
        }
        _ => {
            widgets::chat_view::draw(f, chunks[0], app);
            widgets::input_area::draw(f, chunks[1], app);
            widgets::status_bar::draw(f, chunks[2], app);
        }
    }

    // Overlays
    match app.mode {
        AppMode::HelpDialog => widgets::help_dialog::draw(f),
        AppMode::ExitDialog => widgets::exit_dialog::draw(f),
        AppMode::ApprovalDialog => widgets::approval_dialog::draw(f, app),
        _ => {}
    }
}
