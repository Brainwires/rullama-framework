use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

pub fn draw(f: &mut Frame) {
    let area = centered_rect(50, 60, f.area());

    let text = "\
Keybindings:

  Enter       Send message
  Ctrl+C      Exit (confirm)
  F1          This help
  F2          Console/debug log
  F3          Fullscreen chat
  F4          Fullscreen input
  Up/Down     Scroll chat history

Slash commands:
  /help       Show help
  /clear      Clear conversation
  /exit       Exit

Press any key to close.";

    f.render_widget(Clear, area);
    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Help ")
                .border_style(Style::default().fg(Color::Cyan)),
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
