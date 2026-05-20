# Brainwires CLI Project Instructions

## Project Overview
This is a Rust-based CLI tool for interacting with AI assistants. It features a TUI interface, slash commands, and conversation management.

## Code Style
- Use Rust idioms and best practices
- Prefer explicit error handling with `Result` and `?`
- Document public APIs with rustdoc comments
- Keep functions focused and small

## Architecture Principles
- Modular design with clear separation of concerns
- TUI in `src/tui/`
- Commands in `src/commands/`
- Storage in `src/storage/`
- Providers in `src/providers/`

## Testing
- Write tests for new functionality
- Use `#[cfg(test)]` modules
- Run `cargo test` before committing

## Import Example
You can import other files using @filename.md syntax.
