use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::tui::app::{App, AppMode};

pub fn draw(f: &mut Frame, area: Rect, app: &App) {
    let title = match app.mode {
        AppMode::Waiting => " Input (waiting...) ",
        _ => " Input (Enter to send) ",
    };

    let input_text = if app.input.is_empty() && app.mode == AppMode::Normal {
        "Type your message..."
    } else {
        &app.input
    };

    let style = if app.input.is_empty() && app.mode == AppMode::Normal {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    let paragraph = Paragraph::new(Span::styled(input_text, style))
        .block(Block::default().borders(Borders::ALL).title(title));

    f.render_widget(paragraph, area);

    // Set cursor position
    if app.mode == AppMode::Normal || app.mode == AppMode::InputFullscreen {
        let cursor_x = area.x + 1 + app.cursor_pos as u16;
        let cursor_y = area.y + 1;
        if cursor_x < area.x + area.width - 1 {
            f.set_cursor_position(Position::new(cursor_x, cursor_y));
        }
    }
}

pub fn draw_fullscreen(f: &mut Frame, area: Rect, app: &App) {
    let paragraph = Paragraph::new(app.input.as_str())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Input (Esc to return, Enter to send) "),
        )
        .style(Style::default().fg(Color::White));

    f.render_widget(paragraph, area);

    let cursor_x = area.x + 1 + (app.cursor_pos as u16 % (area.width - 2));
    let cursor_y = area.y + 1 + (app.cursor_pos as u16 / (area.width - 2));
    f.set_cursor_position(Position::new(cursor_x, cursor_y));
}
