use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

pub fn draw(f: &mut Frame) {
    let area = centered_rect(30, 5, f.area());

    f.render_widget(Clear, area);
    let paragraph = Paragraph::new("  Exit? (y/n)")
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red)),
        )
        .alignment(Alignment::Center);

    f.render_widget(paragraph, area);
}

fn centered_rect(percent_x: u16, lines: u16, r: Rect) -> Rect {
    let height = lines.min(r.height);
    let y = (r.height.saturating_sub(height)) / 2;
    let width = r.width * percent_x / 100;
    let x = (r.width.saturating_sub(width)) / 2;
    Rect::new(r.x + x, r.y + y, width, height)
}
