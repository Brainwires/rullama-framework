//! Document chunking for natural language documents
//!
//! Provides paragraph and sentence-aware chunking strategies optimized for
//! documents (as opposed to AST-based chunking for code). Respects natural
//! boundaries like paragraphs, sentences, and markdown section headers.

use super::types::{ChunkerConfig, DocumentChunk};

/// Document chunker with configurable strategies
pub struct DocumentChunker {
    config: ChunkerConfig,
}

impl DocumentChunker {
    /// Create a new chunker with default config
    pub fn new() -> Self {
        Self {
            config: ChunkerConfig::default(),
        }
    }

    /// Create a new chunker with custom config
    pub fn with_config(config: ChunkerConfig) -> Self {
        Self { config }
    }

    /// Chunk a document into pieces
    pub fn chunk(&self, document_id: &str, content: &str) -> Vec<DocumentChunk> {
        if content.is_empty() {
            return Vec::new();
        }

        // Detect if this is markdown
        let is_markdown = Self::looks_like_markdown(content);

        if is_markdown && self.config.respect_headers {
            self.chunk_markdown(document_id, content)
        } else {
            self.chunk_plain_text(document_id, content)
        }
    }

    /// Check if content appears to be markdown
    fn looks_like_markdown(content: &str) -> bool {
        let sample = &content[..content.len().min(2000)];

        // Look for markdown indicators
        sample.contains("\n# ")
            || sample.contains("\n## ")
            || sample.contains("\n### ")
            || sample.starts_with("# ")
            || sample.contains("```")
            || sample.contains("**")
            || sample.contains("[](")
    }

    /// Chunk markdown content respecting section headers
    fn chunk_markdown(&self, document_id: &str, content: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();
        let mut current_section: Option<String> = None;
        let mut section_content = String::new();
        let mut section_start = 0;

        for line in content.lines() {
            // Check for header
            if let Some(header) = Self::extract_markdown_header(line) {
                // Flush previous section if it has content
                if !section_content.trim().is_empty() {
                    self.add_section_chunks(
                        document_id,
                        &section_content,
                        section_start,
                        current_section.as_deref(),
                        &mut chunks,
                    );
                }

                // Start new section
                current_section = Some(header);
                section_start = section_content.len();
                section_content.clear();
            }

            section_content.push_str(line);
            section_content.push('\n');
        }

        // Flush remaining content
        if !section_content.trim().is_empty() {
            self.add_section_chunks(
                document_id,
                &section_content,
                section_start,
                current_section.as_deref(),
                &mut chunks,
            );
        }

        // Update total_chunks for all chunks
        let total = chunks.len() as u32;
        for (i, chunk) in chunks.iter_mut().enumerate() {
            chunk.chunk_index = i as u32;
            chunk.total_chunks = total;
            chunk.chunk_id = format!("{}:{}", document_id, i);
        }

        chunks
    }

    /// Extract header text from a markdown header line
    fn extract_markdown_header(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            // Count # symbols
            let hash_count = trimmed.chars().take_while(|c| *c == '#').count();
            if hash_count <= 6 {
                let header_text = trimmed[hash_count..].trim();
                if !header_text.is_empty() {
                    return Some(header_text.to_string());
                }
            }
        }
        None
    }

    /// Add chunks from a section, respecting paragraph boundaries
    fn add_section_chunks(
        &self,
        document_id: &str,
        content: &str,
        base_offset: usize,
        section: Option<&str>,
        chunks: &mut Vec<DocumentChunk>,
    ) {
        if self.config.respect_paragraphs {
            self.chunk_by_paragraphs(document_id, content, base_offset, section, chunks);
        } else {
            self.chunk_by_size(document_id, content, base_offset, section, chunks);
        }
    }

    /// Chunk plain text content
    fn chunk_plain_text(&self, document_id: &str, content: &str) -> Vec<DocumentChunk> {
        let mut chunks = Vec::new();

        if self.config.respect_paragraphs {
            self.chunk_by_paragraphs(document_id, content, 0, None, &mut chunks);
        } else {
            self.chunk_by_size(document_id, content, 0, None, &mut chunks);
        }

        // Update total_chunks for all chunks
        let total = chunks.len() as u32;
        for (i, chunk) in chunks.iter_mut().enumerate() {
            chunk.chunk_index = i as u32;
            chunk.total_chunks = total;
            chunk.chunk_id = format!("{}:{}", document_id, i);
        }

        chunks
    }

    /// Chunk by paragraph boundaries
    fn chunk_by_paragraphs(
        &self,
        document_id: &str,
        content: &str,
        base_offset: usize,
        section: Option<&str>,
        chunks: &mut Vec<DocumentChunk>,
    ) {
        // Split on double newlines (paragraph boundaries)
        let paragraphs: Vec<&str> = content.split("\n\n").collect();

        let mut current_chunk = String::new();
        let mut chunk_start = 0;
        let mut current_offset = 0;

        for (i, para) in paragraphs.iter().enumerate() {
            let para_trimmed = para.trim();
            if para_trimmed.is_empty() {
                current_offset += para.len() + 2; // +2 for "\n\n"
                continue;
            }

            // Check if adding this paragraph would exceed target size
            let would_exceed =
                current_chunk.len() + para_trimmed.len() > self.config.target_chunk_size;
            let _is_last = i == paragraphs.len() - 1;

            if would_exceed && !current_chunk.is_empty() {
                // Flush current chunk
                let chunk_content = current_chunk.trim().to_string();
                if chunk_content.len() >= self.config.min_chunk_size {
                    let mut chunk = DocumentChunk::new(
                        document_id.to_string(),
                        chunk_content,
                        base_offset + chunk_start,
                        base_offset + current_offset,
                        0, // Will be updated later
                        0, // Will be updated later
                    );
                    if let Some(s) = section {
                        chunk = chunk.with_section(s.to_string());
                    }
                    chunks.push(chunk);
                }

                // Start new chunk with overlap
                current_chunk = self.create_overlap(&current_chunk);
                chunk_start = current_offset.saturating_sub(self.config.overlap_size);
            }

            // Add paragraph to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(para_trimmed);
            current_offset += para.len() + 2;

            // Handle very long paragraphs
            if current_chunk.len() > self.config.max_chunk_size {
                self.split_long_chunk(
                    document_id,
                    &current_chunk,
                    base_offset + chunk_start,
                    section,
                    chunks,
                );
                current_chunk.clear();
                chunk_start = current_offset;
            }
        }

        // Flush remaining content
        // Always add remaining content if: a) we have nothing yet, or b) it meets min size
        let chunk_content = current_chunk.trim().to_string();
        if !chunk_content.is_empty()
            && (chunks.is_empty() || chunk_content.len() >= self.config.min_chunk_size)
        {
            let mut chunk = DocumentChunk::new(
                document_id.to_string(),
                chunk_content,
                base_offset + chunk_start,
                base_offset + current_offset,
                0,
                0,
            );
            if let Some(s) = section {
                chunk = chunk.with_section(s.to_string());
            }
            chunks.push(chunk);
        }
    }

    /// Create overlap content from the end of a chunk
    fn create_overlap(&self, content: &str) -> String {
        if content.len() <= self.config.overlap_size {
            return content.to_string();
        }

        // Try to find a sentence boundary for cleaner overlap
        let overlap_region = &content[content.len() - self.config.overlap_size..];

        if let Some(pos) = overlap_region.find(". ") {
            return overlap_region[pos + 2..].to_string();
        }

        // Fall back to word boundary
        if let Some(pos) = overlap_region.find(' ') {
            return overlap_region[pos + 1..].to_string();
        }

        overlap_region.to_string()
    }

    /// Split a chunk that exceeds max size by sentences or words
    fn split_long_chunk(
        &self,
        document_id: &str,
        content: &str,
        base_offset: usize,
        section: Option<&str>,
        chunks: &mut Vec<DocumentChunk>,
    ) {
        let sentences = Self::split_sentences(content);

        // If sentences don't help (content is one giant block), split by words
        if sentences.len() == 1 && content.len() > self.config.target_chunk_size {
            self.split_by_words(document_id, content, base_offset, section, chunks);
            return;
        }

        let mut current_chunk = String::new();
        let mut chunk_start = 0;
        let mut current_offset = 0;

        for sentence in sentences {
            if current_chunk.len() + sentence.len() > self.config.target_chunk_size
                && !current_chunk.is_empty()
            {
                // Flush current chunk
                let mut chunk = DocumentChunk::new(
                    document_id.to_string(),
                    current_chunk.trim().to_string(),
                    base_offset + chunk_start,
                    base_offset + current_offset,
                    0,
                    0,
                );
                if let Some(s) = section {
                    chunk = chunk.with_section(s.to_string());
                }
                chunks.push(chunk);

                current_chunk = self.create_overlap(&current_chunk);
                chunk_start = current_offset.saturating_sub(self.config.overlap_size);
            }

            current_chunk.push_str(&sentence);
            current_chunk.push(' ');
            current_offset += sentence.len() + 1;
        }

        // Flush remaining
        if !current_chunk.trim().is_empty() {
            let mut chunk = DocumentChunk::new(
                document_id.to_string(),
                current_chunk.trim().to_string(),
                base_offset + chunk_start,
                base_offset + current_offset,
                0,
                0,
            );
            if let Some(s) = section {
                chunk = chunk.with_section(s.to_string());
            }
            chunks.push(chunk);
        }
    }

    /// Split content by words when sentence splitting doesn't help
    fn split_by_words(
        &self,
        document_id: &str,
        content: &str,
        base_offset: usize,
        section: Option<&str>,
        chunks: &mut Vec<DocumentChunk>,
    ) {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut current_chunk = String::new();
        let mut chunk_start = 0;
        let mut current_offset = 0;

        for word in words {
            if current_chunk.len() + word.len() + 1 > self.config.target_chunk_size
                && !current_chunk.is_empty()
            {
                // Flush current chunk
                let mut chunk = DocumentChunk::new(
                    document_id.to_string(),
                    current_chunk.trim().to_string(),
                    base_offset + chunk_start,
                    base_offset + current_offset,
                    0,
                    0,
                );
                if let Some(s) = section {
                    chunk = chunk.with_section(s.to_string());
                }
                chunks.push(chunk);

                current_chunk = self.create_overlap(&current_chunk);
                chunk_start = current_offset.saturating_sub(self.config.overlap_size);
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(word);
            current_offset += word.len() + 1;
        }

        // Flush remaining
        if !current_chunk.trim().is_empty() {
            let mut chunk = DocumentChunk::new(
                document_id.to_string(),
                current_chunk.trim().to_string(),
                base_offset + chunk_start,
                base_offset + current_offset,
                0,
                0,
            );
            if let Some(s) = section {
                chunk = chunk.with_section(s.to_string());
            }
            chunks.push(chunk);
        }
    }

    /// Split text into sentences (simple heuristic)
    fn split_sentences(text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            current.push(c);

            // End of sentence markers
            if (c == '.' || c == '!' || c == '?') && current.len() > 1 {
                // Check for common abbreviations
                let lower = current.to_lowercase();
                let is_abbreviation = lower.ends_with("mr.")
                    || lower.ends_with("mrs.")
                    || lower.ends_with("dr.")
                    || lower.ends_with("vs.")
                    || lower.ends_with("etc.")
                    || lower.ends_with("e.g.")
                    || lower.ends_with("i.e.");

                if !is_abbreviation {
                    sentences.push(current.trim().to_string());
                    current = String::new();
                }
            }
        }

        if !current.trim().is_empty() {
            sentences.push(current.trim().to_string());
        }

        sentences
    }

    /// Chunk by fixed size (fallback when not respecting boundaries)
    fn chunk_by_size(
        &self,
        document_id: &str,
        content: &str,
        base_offset: usize,
        section: Option<&str>,
        chunks: &mut Vec<DocumentChunk>,
    ) {
        let mut start = 0;

        while start < content.len() {
            let end = (start + self.config.target_chunk_size).min(content.len());

            // Try to find a word boundary
            let actual_end = if end < content.len() {
                content[start..end]
                    .rfind(' ')
                    .map(|pos| start + pos + 1)
                    .unwrap_or(end)
            } else {
                end
            };

            let chunk_content = content[start..actual_end].trim().to_string();
            if chunk_content.len() >= self.config.min_chunk_size {
                let mut chunk = DocumentChunk::new(
                    document_id.to_string(),
                    chunk_content,
                    base_offset + start,
                    base_offset + actual_end,
                    0,
                    0,
                );
                if let Some(s) = section {
                    chunk = chunk.with_section(s.to_string());
                }
                chunks.push(chunk);
            }

            // Move start forward, accounting for overlap
            start = actual_end.saturating_sub(self.config.overlap_size);
            if start
                <= chunks
                    .last()
                    .map(|c| c.start_offset - base_offset)
                    .unwrap_or(0)
            {
                start = actual_end;
            }
        }
    }
}

impl Default for DocumentChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_content() {
        let chunker = DocumentChunker::new();
        let chunks = chunker.chunk("doc1", "");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_small_content_single_chunk() {
        let chunker = DocumentChunker::new();
        let content = "This is a small document that fits in one chunk.";
        let chunks = chunker.chunk("doc1", content);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, content);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].total_chunks, 1);
    }

    #[test]
    fn test_paragraph_chunking() {
        let chunker = DocumentChunker::with_config(ChunkerConfig {
            target_chunk_size: 100,
            max_chunk_size: 200,
            min_chunk_size: 10,
            overlap_size: 20,
            respect_paragraphs: true,
            respect_headers: false,
        });

        let content = "First paragraph with some content here.\n\nSecond paragraph with different content.\n\nThird paragraph to test chunking.";
        let chunks = chunker.chunk("doc1", content);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_markdown_header_detection() {
        assert!(DocumentChunker::looks_like_markdown("# Title\n\nContent"));
        assert!(DocumentChunker::looks_like_markdown(
            "Some text\n## Subtitle\n"
        ));
        assert!(DocumentChunker::looks_like_markdown("```rust\ncode\n```"));
        assert!(!DocumentChunker::looks_like_markdown(
            "Plain text without any markdown."
        ));
    }

    #[test]
    fn test_markdown_header_extraction() {
        assert_eq!(
            DocumentChunker::extract_markdown_header("# Title"),
            Some("Title".to_string())
        );
        assert_eq!(
            DocumentChunker::extract_markdown_header("## Subtitle"),
            Some("Subtitle".to_string())
        );
        assert_eq!(
            DocumentChunker::extract_markdown_header("### Nested"),
            Some("Nested".to_string())
        );
        assert_eq!(
            DocumentChunker::extract_markdown_header("Regular text"),
            None
        );
    }

    #[test]
    fn test_sentence_splitting() {
        let sentences = DocumentChunker::split_sentences(
            "First sentence. Second sentence! Third sentence? Fourth.",
        );
        assert_eq!(sentences.len(), 4);
        assert_eq!(sentences[0], "First sentence.");
        assert_eq!(sentences[1], "Second sentence!");
        assert_eq!(sentences[2], "Third sentence?");
    }

    #[test]
    fn test_abbreviations_not_split() {
        let sentences =
            DocumentChunker::split_sentences("Dr. Smith went to the store. He bought milk.");
        // Should be 2 sentences, not split on "Dr."
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn test_config_presets() {
        let small = ChunkerConfig::small();
        let large = ChunkerConfig::large();

        assert!(small.target_chunk_size < large.target_chunk_size);
        assert!(small.max_chunk_size < large.max_chunk_size);
    }
}
