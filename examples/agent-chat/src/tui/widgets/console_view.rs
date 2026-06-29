use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::app::App;

pub fn draw(f: &mut Frame, area: Rect, app: &App) {
    let lines: Vec<Line> = if app.console_log.is_empty() {
        vec![Line::from(Span::styled(
            "No log entries yet.",
            Style::default().fg(Color::DarkGray),
        ))]
    } else {
        app.console_log
            .iter()
            .map(|entry| {
                Line::from(Span::styled(
                    entry.as_str(),
                    Style::default().fg(Color::Gray),
                ))
            })
            .collect()
    };

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Console (Esc/q to return) "),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(paragraph, area);
}
