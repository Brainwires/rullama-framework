use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use super::super::error::{DatasetError, DatasetResult};
use super::super::types::{PreferencePair, TrainingExample};

/// Streaming JSONL reader — memory-efficient, reads one line at a time.
pub struct JsonlReader<R: Read> {
    reader: BufReader<R>,
    line_number: usize,
}

impl JsonlReader<std::fs::File> {
    /// Open a JSONL file for reading.
    pub fn open(path: impl AsRef<Path>) -> DatasetResult<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        Ok(Self::new(file))
    }
}

impl<R: Read> JsonlReader<R> {
    /// Create a new JSONL reader wrapping the given reader.
    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
            line_number: 0,
        }
    }

    /// Read the next example from the JSONL stream.
    pub fn next_example(&mut self) -> DatasetResult<Option<TrainingExample>> {
        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = self.reader.read_line(&mut line)?;
            self.line_number += 1;

            if bytes_read == 0 {
                return Ok(None);
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let example: TrainingExample =
                serde_json::from_str(trimmed).map_err(|e| DatasetError::Validation {
                    message: format!("line {}: {}", self.line_number, e),
                })?;
            return Ok(Some(example));
        }
    }

    /// Read all examples into a Vec.
    pub fn read_all(&mut self) -> DatasetResult<Vec<TrainingExample>> {
        let mut examples = Vec::new();
        while let Some(example) = self.next_example()? {
            examples.push(example);
        }
        tracing::debug!("Read {} examples from JSONL", examples.len());
        Ok(examples)
    }

    /// Read the next preference pair from the JSONL stream.
    pub fn next_preference(&mut self) -> DatasetResult<Option<PreferencePair>> {
        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = self.reader.read_line(&mut line)?;
            self.line_number += 1;

            if bytes_read == 0 {
                return Ok(None);
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let pair: PreferencePair =
                serde_json::from_str(trimmed).map_err(|e| DatasetError::Validation {
                    message: format!("line {}: {}", self.line_number, e),
                })?;
            return Ok(Some(pair));
        }
    }

    /// Read all preference pairs into a Vec.
    pub fn read_all_preferences(&mut self) -> DatasetResult<Vec<PreferencePair>> {
        let mut pairs = Vec::new();
        while let Some(pair) = self.next_preference()? {
            pairs.push(pair);
        }
        tracing::debug!("Read {} preference pairs from JSONL", pairs.len());
        Ok(pairs)
    }

    /// Current line number (1-based).
    pub fn line_number(&self) -> usize {
        self.line_number
    }
}

/// Convenience: read all examples from a JSONL file path.
pub fn read_jsonl(path: impl AsRef<Path>) -> DatasetResult<Vec<TrainingExample>> {
    let mut reader = JsonlReader::open(path)?;
    reader.read_all()
}

/// Convenience: read all preference pairs from a JSONL file path.
pub fn read_jsonl_preferences(path: impl AsRef<Path>) -> DatasetResult<Vec<PreferencePair>> {
    let mut reader = JsonlReader::open(path)?;
    reader.read_all_preferences()
}

/// Iterator adapter over JsonlReader.
impl<R: Read> Iterator for JsonlReader<R> {
    type Item = DatasetResult<TrainingExample>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_example() {
            Ok(Some(example)) => Some(Ok(example)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_jsonl() -> &'static str {
        r#"{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"}]}
{"messages":[{"role":"system","content":"Be helpful"},{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]}
"#
    }

    #[test]
    fn test_read_jsonl_from_cursor() {
        let cursor = Cursor::new(sample_jsonl());
        let mut reader = JsonlReader::new(cursor);
        let examples = reader.read_all().unwrap();
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].messages.len(), 2);
        assert_eq!(examples[1].messages.len(), 3);
    }

    #[test]
    fn test_reader_iterator() {
        let cursor = Cursor::new(sample_jsonl());
        let reader = JsonlReader::new(cursor);
        let examples: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(examples.len(), 2);
    }

    #[test]
    fn test_reader_skips_blank_lines() {
        let data = r#"{"messages":[{"role":"user","content":"A"},{"role":"assistant","content":"B"}]}

{"messages":[{"role":"user","content":"C"},{"role":"assistant","content":"D"}]}
"#;
        let cursor = Cursor::new(data);
        let mut reader = JsonlReader::new(cursor);
        let examples = reader.read_all().unwrap();
        assert_eq!(examples.len(), 2);
    }

    #[test]
    fn test_reader_error_on_invalid_json() {
        let data = "not valid json\n";
        let cursor = Cursor::new(data);
        let mut reader = JsonlReader::new(cursor);
        let result = reader.next_example();
        assert!(result.is_err());
    }

    #[test]
    fn test_read_preference_pairs() {
        let data = r#"{"prompt":[{"role":"user","content":"Q1"}],"chosen":[{"role":"assistant","content":"Good"}],"rejected":[{"role":"assistant","content":"Bad"}]}
{"prompt":[{"role":"user","content":"Q2"}],"chosen":[{"role":"assistant","content":"Yes"}],"rejected":[{"role":"assistant","content":"No"}]}
"#;
        let cursor = Cursor::new(data);
        let mut reader = JsonlReader::new(cursor);
        let pairs = reader.read_all_preferences().unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].prompt[0].content, "Q1");
    }
}
