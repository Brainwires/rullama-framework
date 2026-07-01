use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use crate::tui::app::App;

pub fn draw(f: &mut Frame, app: &App) {
    let area = centered_rect(60, 40, f.area());

    let (tool_name, input_summary) = if let Some(ref pending) = app.pending_approval {
        let summary = serde_json::to_string_pretty(&pending.input).unwrap_or_default();
        let truncated = if summary.len() > 300 {
            format!("{}...", &summary[..300])
        } else {
            summary
        };
        (pending.tool_name.as_str(), truncated)
    } else {
        ("unknown", String::new())
    };

    let text = format!(
        "Tool: {tool_name}\n\n\
         Args:\n{input_summary}\n\n\
         [Y]es  [N]o  [A]lways"
    );

    f.render_widget(Clear, area);
    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Tool Approval ")
                .border_style(Style::default().fg(Color::Yellow)),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(paragraph, area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
