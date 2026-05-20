use std::io::{BufWriter, Write};
use std::path::Path;

use super::super::error::DatasetResult;
use super::super::types::{PreferencePair, TrainingExample};

/// Buffered JSONL writer.
pub struct JsonlWriter<W: Write> {
    writer: BufWriter<W>,
    count: usize,
}

impl JsonlWriter<std::fs::File> {
    /// Create a new JSONL file for writing.
    pub fn create(path: impl AsRef<Path>) -> DatasetResult<Self> {
        let file = std::fs::File::create(path.as_ref())?;
        Ok(Self::new(file))
    }
}

impl<W: Write> JsonlWriter<W> {
    /// Create a new JSONL writer wrapping the given writer.
    pub fn new(writer: W) -> Self {
        Self {
            writer: BufWriter::new(writer),
            count: 0,
        }
    }

    /// Write a single training example as a JSONL line.
    pub fn write_example(&mut self, example: &TrainingExample) -> DatasetResult<()> {
        serde_json::to_writer(&mut self.writer, example)?;
        self.writer.write_all(b"\n")?;
        self.count += 1;
        Ok(())
    }

    /// Write multiple examples.
    pub fn write_all(&mut self, examples: &[TrainingExample]) -> DatasetResult<()> {
        for example in examples {
            self.write_example(example)?;
        }
        Ok(())
    }

    /// Write a raw serializable value as a JSONL line (for format converters).
    pub fn write_raw<T: serde::Serialize>(&mut self, value: &T) -> DatasetResult<()> {
        serde_json::to_writer(&mut self.writer, value)?;
        self.writer.write_all(b"\n")?;
        self.count += 1;
        Ok(())
    }

    /// Write a single preference pair as a JSONL line.
    pub fn write_preference(&mut self, pair: &PreferencePair) -> DatasetResult<()> {
        serde_json::to_writer(&mut self.writer, pair)?;
        self.writer.write_all(b"\n")?;
        self.count += 1;
        Ok(())
    }

    /// Write multiple preference pairs.
    pub fn write_all_preferences(&mut self, pairs: &[PreferencePair]) -> DatasetResult<()> {
        for pair in pairs {
            self.write_preference(pair)?;
        }
        Ok(())
    }

    /// Flush the underlying buffer.
    pub fn flush(&mut self) -> DatasetResult<()> {
        self.writer.flush()?;
        Ok(())
    }

    /// Number of examples written.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Convenience: write examples to a JSONL file.
pub fn write_jsonl(path: impl AsRef<Path>, examples: &[TrainingExample]) -> DatasetResult<usize> {
    let mut writer = JsonlWriter::create(path)?;
    writer.write_all(examples)?;
    writer.flush()?;
    Ok(writer.count())
}

/// Convenience: write preference pairs to a JSONL file.
pub fn write_jsonl_preferences(
    path: impl AsRef<Path>,
    pairs: &[PreferencePair],
) -> DatasetResult<usize> {
    let mut writer = JsonlWriter::create(path)?;
    writer.write_all_preferences(pairs)?;
    writer.flush()?;
    Ok(writer.count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::types::TrainingMessage;
    use std::io::Cursor;

    #[test]
    fn test_write_and_roundtrip() {
        let examples = vec![
            TrainingExample::with_id(
                "ex1",
                vec![
                    TrainingMessage::user("Hello"),
                    TrainingMessage::assistant("Hi!"),
                ],
            ),
            TrainingExample::with_id(
                "ex2",
                vec![
                    TrainingMessage::system("Be helpful"),
                    TrainingMessage::user("Q"),
                    TrainingMessage::assistant("A"),
                ],
            ),
        ];

        let mut buf = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            let mut writer = JsonlWriter::new(cursor);
            writer.write_all(&examples).unwrap();
            writer.flush().unwrap();
            assert_eq!(writer.count(), 2);
        }

        // Read back
        let cursor = Cursor::new(&buf);
        let mut reader = crate::datasets::jsonl::reader::JsonlReader::new(cursor);
        let read_back = reader.read_all().unwrap();
        assert_eq!(read_back.len(), 2);
        assert_eq!(read_back[0].messages.len(), 2);
        assert_eq!(read_back[1].messages.len(), 3);
    }

    #[test]
    fn test_write_and_read_preferences() {
        use crate::datasets::types::TrainingMessage;
        let pairs = vec![PreferencePair::new(
            vec![TrainingMessage::user("Q")],
            vec![TrainingMessage::assistant("Good")],
            vec![TrainingMessage::assistant("Bad")],
        )];

        let mut buf = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            let mut writer = JsonlWriter::new(cursor);
            writer.write_all_preferences(&pairs).unwrap();
            writer.flush().unwrap();
            assert_eq!(writer.count(), 1);
        }

        let cursor = Cursor::new(&buf);
        let mut reader = crate::datasets::jsonl::reader::JsonlReader::new(cursor);
        let read_back = reader.read_all_preferences().unwrap();
        assert_eq!(read_back.len(), 1);
        assert_eq!(read_back[0].prompt[0].content, "Q");
    }
}
