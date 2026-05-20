//! Document text extraction for various file formats
//!
//! Extracts text content from PDF, Markdown, plain text, and DOCX files.
//! Uses format-specific libraries for accurate extraction.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

use super::types::{DocumentType, ExtractedDocument};

/// Document processor for text extraction
pub struct DocumentProcessor;

impl DocumentProcessor {
    /// Extract text content from a file
    pub fn extract_text(file_path: &Path) -> Result<ExtractedDocument> {
        let file_type = DocumentType::from_path(file_path);
        let bytes = fs::read(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;

        Self::extract_from_bytes(&bytes, file_type)
    }

    /// Extract text content from bytes
    pub fn extract_from_bytes(bytes: &[u8], file_type: DocumentType) -> Result<ExtractedDocument> {
        match file_type {
            DocumentType::Pdf => Self::extract_pdf(bytes),
            DocumentType::Markdown => Self::extract_markdown(bytes),
            DocumentType::PlainText => Self::extract_plain_text(bytes),
            DocumentType::Docx => Self::extract_docx(bytes),
            DocumentType::Unknown => Self::extract_plain_text(bytes),
        }
    }

    /// Compute SHA256 hash of file content
    pub fn compute_hash(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    /// Compute SHA256 hash of a file
    pub fn compute_file_hash(file_path: &Path) -> Result<String> {
        let bytes = fs::read(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;
        Ok(Self::compute_hash(&bytes))
    }

    /// Extract text from PDF bytes
    #[cfg(feature = "pdf-extract-feature")]
    fn extract_pdf(bytes: &[u8]) -> Result<ExtractedDocument> {
        use pdf_extract::extract_text_from_mem;

        let text = extract_text_from_mem(bytes).context("Failed to extract text from PDF")?;

        // Try to detect page count (approximate from text)
        let page_breaks = text.matches('\x0c').count(); // Form feed is often used
        let page_count = if page_breaks > 0 {
            Some(page_breaks + 1)
        } else {
            None
        };

        let mut doc = ExtractedDocument::new(text, DocumentType::Pdf);

        if let Some(count) = page_count {
            doc = doc.with_page_count(count);
        }

        // Try to extract title from first line if it looks like a title
        let title = doc.content.lines().next().map(|l| l.trim().to_string());
        if let Some(trimmed) = title.filter(|t| !t.is_empty() && t.len() < 200) {
            doc = doc.with_title(trimmed);
        }

        Ok(doc)
    }

    /// Fallback PDF extraction when pdf-extract feature is disabled
    #[cfg(not(feature = "pdf-extract-feature"))]
    fn extract_pdf(bytes: &[u8]) -> Result<ExtractedDocument> {
        // Basic fallback: try to extract visible text patterns
        let content = Self::extract_pdf_text_fallback(bytes);

        Ok(ExtractedDocument::new(content, DocumentType::Pdf)
            .with_warning("PDF extraction requires pdf-extract feature".to_string()))
    }

    /// Basic PDF text extraction fallback (extracts visible ASCII strings)
    #[cfg(not(feature = "pdf-extract-feature"))]
    fn extract_pdf_text_fallback(bytes: &[u8]) -> String {
        // Look for text between parentheses (PDF string literals)
        // and between BT/ET text blocks
        let mut result = String::new();
        let mut in_string = false;
        let mut current_string = String::new();

        for byte in bytes {
            let c = *byte as char;

            if c == '(' && !in_string {
                in_string = true;
                current_string.clear();
            } else if c == ')' && in_string {
                in_string = false;
                if current_string
                    .chars()
                    .all(|c| c.is_ascii_graphic() || c.is_whitespace())
                    && !current_string.is_empty()
                {
                    result.push_str(&current_string);
                    result.push(' ');
                }
            } else if in_string && c.is_ascii() {
                current_string.push(c);
            }
        }

        // Clean up the result
        result.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Extract text from Markdown bytes
    fn extract_markdown(bytes: &[u8]) -> Result<ExtractedDocument> {
        let content = String::from_utf8_lossy(bytes).to_string();

        let mut doc = ExtractedDocument::new(content.clone(), DocumentType::Markdown);

        // Extract title from first header
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some(title) = trimmed.strip_prefix("# ") {
                doc = doc.with_title(title.trim().to_string());
                break;
            }
        }

        Ok(doc)
    }

    /// Extract text from plain text bytes
    fn extract_plain_text(bytes: &[u8]) -> Result<ExtractedDocument> {
        let content = String::from_utf8_lossy(bytes).to_string();
        Ok(ExtractedDocument::new(content, DocumentType::PlainText))
    }

    /// Extract text from DOCX bytes
    fn extract_docx(bytes: &[u8]) -> Result<ExtractedDocument> {
        use std::io::Cursor;
        use zip::ZipArchive;

        let reader = Cursor::new(bytes);
        let mut archive = ZipArchive::new(reader).context("Failed to open DOCX as ZIP archive")?;

        // DOCX stores content in word/document.xml
        let mut content = String::new();
        let mut title = None;

        if let Ok(mut document_xml) = archive.by_name("word/document.xml") {
            use std::io::Read;
            let mut xml_content = String::new();
            document_xml
                .read_to_string(&mut xml_content)
                .context("Failed to read document.xml")?;

            // Extract text from XML (simple approach - strip tags)
            content = Self::extract_text_from_xml(&xml_content);
        }

        // Try to get title from core properties
        if let Ok(mut core_xml) = archive.by_name("docProps/core.xml") {
            use std::io::Read;
            let mut xml_content = String::new();
            if core_xml.read_to_string(&mut xml_content).is_ok() {
                title = Self::extract_title_from_core_xml(&xml_content);
            }
        }

        let mut doc = ExtractedDocument::new(content, DocumentType::Docx);

        if let Some(t) = title {
            doc = doc.with_title(t);
        }

        Ok(doc)
    }

    /// Extract text content from DOCX XML (simple tag stripping)
    fn extract_text_from_xml(xml: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;
        let in_text = false;
        let mut current_text = String::new();

        // Simple state machine to extract text between <w:t> tags
        for c in xml.chars() {
            if c == '<' {
                in_tag = true;
                if in_text && !current_text.is_empty() {
                    result.push_str(&current_text);
                    current_text.clear();
                }
            } else if c == '>' {
                in_tag = false;
            } else if in_tag {
                // Check for text element start/end
                // We're looking for patterns like <w:t> or </w:t>
            } else {
                // Outside of tags, collect text
                current_text.push(c);
            }
        }

        // More sophisticated approach: use regex to find text nodes
        use std::sync::LazyLock;
        static RE_TEXT: LazyLock<regex::Regex> =
            LazyLock::new(|| regex::Regex::new(r"<w:t[^>]*>([^<]*)</w:t>").expect("valid regex"));
        static RE_PARA: LazyLock<regex::Regex> =
            LazyLock::new(|| regex::Regex::new(r"</w:p>").expect("valid regex"));
        result.clear();

        for cap in RE_TEXT.captures_iter(xml) {
            if let Some(text) = cap.get(1) {
                result.push_str(text.as_str());
            }
        }

        // Add paragraph breaks
        let with_breaks = RE_PARA.replace_all(&result, "\n\n");

        with_breaks.to_string()
    }

    /// Extract title from DOCX core.xml
    fn extract_title_from_core_xml(xml: &str) -> Option<String> {
        let re = regex::Regex::new(r"<dc:title>([^<]+)</dc:title>").ok()?;
        re.captures(xml)
            .and_then(|cap| cap.get(1))
            .map(|m| m.as_str().to_string())
    }

    /// Detect the document type from magic bytes
    pub fn detect_type_from_bytes(bytes: &[u8]) -> DocumentType {
        if bytes.len() < 4 {
            return DocumentType::Unknown;
        }

        // PDF: starts with %PDF
        if bytes.starts_with(b"%PDF") {
            return DocumentType::Pdf;
        }

        // DOCX/ZIP: starts with PK (ZIP magic)
        if bytes.starts_with(b"PK\x03\x04") {
            // Could be DOCX - check for word/ directory
            let reader = std::io::Cursor::new(bytes);
            if let Ok(mut archive) = zip::ZipArchive::new(reader)
                && archive.by_name("word/document.xml").is_ok()
            {
                return DocumentType::Docx;
            }
        }

        // Check for text-based formats
        if let Ok(text) = std::str::from_utf8(&bytes[..bytes.len().min(1000)]) {
            // Markdown detection
            if text.contains("# ")
                || text.contains("## ")
                || text.contains("```")
                || text.contains("[](")
            {
                return DocumentType::Markdown;
            }

            // If it's valid UTF-8, assume plain text
            return DocumentType::PlainText;
        }

        DocumentType::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hash() {
        let hash = DocumentProcessor::compute_hash(b"hello world");
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_extract_plain_text() {
        let content = b"Hello, world!\nThis is a test.";
        let doc = DocumentProcessor::extract_from_bytes(content, DocumentType::PlainText).unwrap();

        assert_eq!(doc.file_type, DocumentType::PlainText);
        assert!(doc.content.contains("Hello, world!"));
        assert!(doc.content.contains("This is a test."));
    }

    #[test]
    fn test_extract_markdown() {
        let content = b"# Title\n\nSome content here.\n\n## Section\n\nMore content.";
        let doc = DocumentProcessor::extract_from_bytes(content, DocumentType::Markdown).unwrap();

        assert_eq!(doc.file_type, DocumentType::Markdown);
        assert_eq!(doc.title, Some("Title".to_string()));
        assert!(doc.content.contains("Some content here."));
    }

    #[test]
    fn test_detect_type_from_bytes_pdf() {
        let bytes = b"%PDF-1.4 some content";
        assert_eq!(
            DocumentProcessor::detect_type_from_bytes(bytes),
            DocumentType::Pdf
        );
    }

    #[test]
    fn test_detect_type_from_bytes_markdown() {
        let bytes = b"# Title\n\nContent with **bold** text.";
        assert_eq!(
            DocumentProcessor::detect_type_from_bytes(bytes),
            DocumentType::Markdown
        );
    }

    #[test]
    fn test_detect_type_from_bytes_plain_text() {
        let bytes = b"Just some plain text without any special formatting.";
        assert_eq!(
            DocumentProcessor::detect_type_from_bytes(bytes),
            DocumentType::PlainText
        );
    }

    #[test]
    fn test_extract_text_from_xml() {
        let xml = r#"<w:p><w:t>Hello</w:t><w:t> </w:t><w:t>World</w:t></w:p>"#;
        let text = DocumentProcessor::extract_text_from_xml(xml);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn test_extract_title_from_core_xml() {
        let xml = r#"<cp:coreProperties><dc:title>My Document</dc:title></cp:coreProperties>"#;
        let title = DocumentProcessor::extract_title_from_core_xml(xml);
        assert_eq!(title, Some("My Document".to_string()));
    }

    #[test]
    fn test_extracted_document_empty_check() {
        let doc = ExtractedDocument::new("   ".to_string(), DocumentType::PlainText);
        assert!(doc.is_empty());

        let doc = ExtractedDocument::new("content".to_string(), DocumentType::PlainText);
        assert!(!doc.is_empty());
    }
}
