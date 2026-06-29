use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::app::{App, AppMode, MessageRole};

pub fn draw(f: &mut Frame, area: Rect, app: &App) {
    let mut lines: Vec<Line> = Vec::new();

    if app.display_messages.is_empty() {
        lines.push(Line::from(Span::styled(
            "Start typing to chat...",
            Style::default().fg(Color::DarkGray),
        )));
    }

    for msg in &app.display_messages {
        let (prefix, style) = match msg.role {
            MessageRole::User => (
                "You: ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            MessageRole::Assistant => ("AI: ", Style::default().fg(Color::White)),
            MessageRole::Tool => (
                "",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::DIM),
            ),
            MessageRole::Error => (
                "Error: ",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
        };

        // Split content into lines for wrapping
        for (i, text_line) in msg.content.lines().enumerate() {
            let mut spans = Vec::new();
            if i == 0 && !prefix.is_empty() {
                spans.push(Span::styled(prefix, style.add_modifier(Modifier::BOLD)));
            }

            // Basic markdown: **bold** and `code`
            let parts = parse_basic_markdown(text_line, style);
            spans.extend(parts);

            lines.push(Line::from(spans));
        }
        lines.push(Line::from(""));
    }

    if app.mode == AppMode::Waiting {
        lines.push(Line::from(Span::styled(
            "Thinking...",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::DIM),
        )));
    }

    // Apply scroll offset
    let total_lines = lines.len() as u16;
    let visible_height = area.height.saturating_sub(2); // account for borders
    let max_scroll = total_lines.saturating_sub(visible_height);
    let scroll = max_scroll.saturating_sub(app.scroll_offset);

    let title = if app.mode == AppMode::ConversationFullscreen {
        " Chat (Esc to return) "
    } else {
        " Chat "
    };

    let paragraph = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));

    f.render_widget(paragraph, area);
}

fn parse_basic_markdown<'a>(text: &'a str, base_style: Style) -> Vec<Span<'a>> {
    let mut spans = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Check for **bold**
        if let Some(start) = remaining.find("**") {
            if start > 0 {
                spans.push(Span::styled(&remaining[..start], base_style));
            }
            let after_start = &remaining[start + 2..];
            if let Some(end) = after_start.find("**") {
                spans.push(Span::styled(
                    &after_start[..end],
                    base_style.add_modifier(Modifier::BOLD),
                ));
                remaining = &after_start[end + 2..];
                continue;
            } else {
                spans.push(Span::styled(&remaining[start..], base_style));
                return spans;
            }
        }

        // Check for `code`
        if let Some(start) = remaining.find('`') {
            if start > 0 {
                spans.push(Span::styled(&remaining[..start], base_style));
            }
            let after_start = &remaining[start + 1..];
            if let Some(end) = after_start.find('`') {
                spans.push(Span::styled(
                    &after_start[..end],
                    Style::default().fg(Color::Green),
                ));
                remaining = &after_start[end + 1..];
                continue;
            } else {
                spans.push(Span::styled(&remaining[start..], base_style));
                return spans;
            }
        }

        // No more markdown
        spans.push(Span::styled(remaining, base_style));
        break;
    }

    spans
}
