//! Image Analysis Types
//!
//! Core types for storing and searching analyzed images with embeddings.
//! Images are stored with their LLM-generated analysis for semantic search.

use serde::{Deserialize, Serialize};

/// Supported image formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageFormat {
    /// PNG format.
    Png,
    /// JPEG format.
    Jpeg,
    /// GIF format.
    Gif,
    /// WebP format.
    Webp,
    /// SVG format.
    Svg,
    /// Unknown format.
    Unknown,
}

impl ImageFormat {
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "png" => Self::Png,
            "jpg" | "jpeg" => Self::Jpeg,
            "gif" => Self::Gif,
            "webp" => Self::Webp,
            "svg" => Self::Svg,
            _ => Self::Unknown,
        }
    }

    /// Detect format from path
    pub fn from_path(path: &std::path::Path) -> Self {
        path.extension()
            .and_then(|e| e.to_str())
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }

    /// Detect format from MIME type
    pub fn from_mime(mime: &str) -> Self {
        match mime {
            "image/png" => Self::Png,
            "image/jpeg" => Self::Jpeg,
            "image/gif" => Self::Gif,
            "image/webp" => Self::Webp,
            "image/svg+xml" => Self::Svg,
            _ => Self::Unknown,
        }
    }

    /// Get MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
            Self::Svg => "image/svg+xml",
            Self::Unknown => "application/octet-stream",
        }
    }

    /// Check if format is supported for analysis
    pub fn is_supported(&self) -> bool {
        !matches!(self, Self::Unknown)
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Gif => "gif",
            Self::Webp => "webp",
            Self::Svg => "svg",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for ImageFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for ImageFormat {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::from_extension(s))
    }
}

/// Metadata for an analyzed image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMetadata {
    /// Unique image identifier
    pub image_id: String,
    /// Message this image belongs to (if from a message)
    pub message_id: Option<String>,
    /// Conversation this image belongs to
    pub conversation_id: String,
    /// Original file name (if known)
    pub file_name: Option<String>,
    /// Image format
    pub format: ImageFormat,
    /// MIME type
    pub mime_type: String,
    /// Image width in pixels
    pub width: Option<u32>,
    /// Image height in pixels
    pub height: Option<u32>,
    /// File size in bytes
    pub file_size_bytes: u64,
    /// SHA256 hash for deduplication
    pub file_hash: String,
    /// LLM-generated analysis/description
    pub analysis: String,
    /// OCR-extracted text (if applicable)
    pub extracted_text: Option<String>,
    /// Auto-generated tags
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: i64,
}

impl ImageMetadata {
    /// Create new image metadata
    pub fn new(
        image_id: String,
        conversation_id: String,
        format: ImageFormat,
        file_size_bytes: u64,
        file_hash: String,
        analysis: String,
    ) -> Self {
        Self {
            image_id,
            message_id: None,
            conversation_id,
            file_name: None,
            format,
            mime_type: format.mime_type().to_string(),
            width: None,
            height: None,
            file_size_bytes,
            file_hash,
            analysis,
            extracted_text: None,
            tags: Vec::new(),
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    /// Builder: set message ID
    pub fn with_message_id(mut self, message_id: String) -> Self {
        self.message_id = Some(message_id);
        self
    }

    /// Builder: set file name
    pub fn with_file_name(mut self, file_name: String) -> Self {
        self.file_name = Some(file_name);
        self
    }

    /// Builder: set dimensions
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Builder: set extracted text
    pub fn with_extracted_text(mut self, text: String) -> Self {
        self.extracted_text = Some(text);
        self
    }

    /// Builder: set tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Get combined searchable text (analysis + extracted text)
    pub fn searchable_text(&self) -> String {
        let mut text = self.analysis.clone();
        if let Some(ref extracted) = self.extracted_text {
            text.push_str("\n\n");
            text.push_str(extracted);
        }
        if !self.tags.is_empty() {
            text.push_str("\n\nTags: ");
            text.push_str(&self.tags.join(", "));
        }
        text
    }
}

/// Request to search images
#[derive(Debug, Clone)]
pub struct ImageSearchRequest {
    /// Search query
    pub query: String,
    /// Filter by conversation
    pub conversation_id: Option<String>,
    /// Maximum results
    pub limit: usize,
    /// Minimum similarity score (0.0-1.0)
    pub min_score: f32,
    /// Filter by format
    pub format: Option<ImageFormat>,
    /// Include images with extracted text matching query
    pub include_ocr: bool,
}

impl ImageSearchRequest {
    /// Create a new search request
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            conversation_id: None,
            limit: 10,
            min_score: 0.5,
            format: None,
            include_ocr: true,
        }
    }

    /// Builder: filter by conversation
    pub fn with_conversation(mut self, conversation_id: String) -> Self {
        self.conversation_id = Some(conversation_id);
        self
    }

    /// Builder: set limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Builder: set minimum score
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Builder: filter by format
    pub fn with_format(mut self, format: ImageFormat) -> Self {
        self.format = Some(format);
        self
    }
}

/// Result from image search
#[derive(Debug, Clone)]
pub struct ImageSearchResult {
    /// Image ID
    pub image_id: String,
    /// Conversation ID
    pub conversation_id: String,
    /// File name (if known)
    pub file_name: Option<String>,
    /// Image format
    pub format: ImageFormat,
    /// LLM analysis
    pub analysis: String,
    /// Extracted text (if any)
    pub extracted_text: Option<String>,
    /// Tags
    pub tags: Vec<String>,
    /// Similarity score
    pub score: f32,
    /// Image width in pixels.
    pub width: Option<u32>,
    /// Image height in pixels.
    pub height: Option<u32>,
    /// Creation timestamp
    pub created_at: i64,
}

impl ImageSearchResult {
    /// Create from metadata with score
    pub fn from_metadata(meta: ImageMetadata, score: f32) -> Self {
        Self {
            image_id: meta.image_id,
            conversation_id: meta.conversation_id,
            file_name: meta.file_name,
            format: meta.format,
            analysis: meta.analysis,
            extracted_text: meta.extracted_text,
            tags: meta.tags,
            score,
            width: meta.width,
            height: meta.height,
            created_at: meta.created_at,
        }
    }
}

/// Image data storage options
#[derive(Debug, Clone)]
pub enum ImageStorage {
    /// Store as base64 in database
    Base64(String),
    /// Store as file path reference
    FilePath(String),
    /// Store as URL reference
    Url(String),
}

impl ImageStorage {
    /// Create base64 storage from bytes
    #[cfg(feature = "native")]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        use base64::{Engine, engine::general_purpose::STANDARD};
        Self::Base64(STANDARD.encode(bytes))
    }

    /// Get the storage type as string
    pub fn storage_type(&self) -> &'static str {
        match self {
            Self::Base64(_) => "base64",
            Self::FilePath(_) => "file",
            Self::Url(_) => "url",
        }
    }

    /// Get the stored value
    pub fn value(&self) -> &str {
        match self {
            Self::Base64(v) | Self::FilePath(v) | Self::Url(v) => v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_format_from_extension() {
        assert_eq!(ImageFormat::from_extension("png"), ImageFormat::Png);
        assert_eq!(ImageFormat::from_extension("jpg"), ImageFormat::Jpeg);
        assert_eq!(ImageFormat::from_extension("jpeg"), ImageFormat::Jpeg);
        assert_eq!(ImageFormat::from_extension("gif"), ImageFormat::Gif);
        assert_eq!(ImageFormat::from_extension("webp"), ImageFormat::Webp);
        assert_eq!(ImageFormat::from_extension("svg"), ImageFormat::Svg);
        assert_eq!(ImageFormat::from_extension("bmp"), ImageFormat::Unknown);
    }

    #[test]
    fn test_image_format_mime_type() {
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
    }

    #[test]
    fn test_image_format_from_mime() {
        assert_eq!(ImageFormat::from_mime("image/png"), ImageFormat::Png);
        assert_eq!(ImageFormat::from_mime("image/jpeg"), ImageFormat::Jpeg);
    }

    #[test]
    fn test_image_metadata_builder() {
        let meta = ImageMetadata::new(
            "img-123".to_string(),
            "conv-456".to_string(),
            ImageFormat::Png,
            1024,
            "hash123".to_string(),
            "A screenshot of code".to_string(),
        )
        .with_message_id("msg-789".to_string())
        .with_file_name("screenshot.png".to_string())
        .with_dimensions(1920, 1080)
        .with_tags(vec!["code".to_string(), "screenshot".to_string()]);

        assert_eq!(meta.image_id, "img-123");
        assert_eq!(meta.message_id, Some("msg-789".to_string()));
        assert_eq!(meta.width, Some(1920));
        assert_eq!(meta.height, Some(1080));
        assert_eq!(meta.tags.len(), 2);
    }

    #[test]
    fn test_searchable_text() {
        let meta = ImageMetadata::new(
            "img-123".to_string(),
            "conv-456".to_string(),
            ImageFormat::Png,
            1024,
            "hash123".to_string(),
            "A diagram showing architecture".to_string(),
        )
        .with_extracted_text("Component A -> Component B".to_string())
        .with_tags(vec!["diagram".to_string(), "architecture".to_string()]);

        let text = meta.searchable_text();
        assert!(text.contains("diagram showing architecture"));
        assert!(text.contains("Component A"));
        assert!(text.contains("Tags: diagram, architecture"));
    }

    #[test]
    fn test_image_search_request_builder() {
        let request = ImageSearchRequest::new("architecture diagram")
            .with_conversation("conv-123".to_string())
            .with_limit(5)
            .with_min_score(0.7)
            .with_format(ImageFormat::Png);

        assert_eq!(request.query, "architecture diagram");
        assert_eq!(request.conversation_id, Some("conv-123".to_string()));
        assert_eq!(request.limit, 5);
        assert_eq!(request.min_score, 0.7);
        assert_eq!(request.format, Some(ImageFormat::Png));
    }

    #[test]
    fn test_image_storage() {
        let storage = ImageStorage::from_bytes(b"test image data");
        assert_eq!(storage.storage_type(), "base64");
        assert!(!storage.value().is_empty());

        let file_storage = ImageStorage::FilePath("/path/to/image.png".to_string());
        assert_eq!(file_storage.storage_type(), "file");
        assert_eq!(file_storage.value(), "/path/to/image.png");
    }
}
