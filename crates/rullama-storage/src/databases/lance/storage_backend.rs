//! [`StorageBackend`] implementation for [`LanceDatabase`].

use anyhow::{Context, Result};
use arrow_array::{Float32Array, RecordBatch, RecordBatchIterator, RecordBatchReader};
use futures::stream::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;

use crate::databases::traits::StorageBackend;
use crate::databases::types::{FieldDef, Filter, Record, ScoredRecord};

use super::arrow_convert::{
    batch_to_records, extract_field_value, field_defs_to_schema, filter_to_sql, records_to_batch,
};
use super::database::LanceDatabase;

#[async_trait::async_trait]
impl StorageBackend for LanceDatabase {
    async fn ensure_table(&self, table_name: &str, schema: &[FieldDef]) -> Result<()> {
        let table_names = self.connection.table_names().execute().await?;
        if table_names.contains(&table_name.to_string()) {
            return Ok(());
        }

        let arrow_schema = Arc::new(field_defs_to_schema(schema));
        let batches: Box<dyn RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(vec![], arrow_schema));
        self.connection
            .create_table(table_name, batches)
            .execute()
            .await
            .with_context(|| format!("Failed to create table '{table_name}'"))?;
        Ok(())
    }

    async fn insert(&self, table_name: &str, records: Vec<Record>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let table = self
            .connection
            .open_table(table_name)
            .execute()
            .await
            .with_context(|| format!("Failed to open table '{table_name}'"))?;

        let batch = records_to_batch(&records)?;
        let schema = batch.schema();
        let batches: Box<dyn RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
        table
            .add(batches)
            .execute()
            .await
            .with_context(|| format!("Failed to insert into '{table_name}'"))?;
        Ok(())
    }

    async fn query(
        &self,
        table_name: &str,
        filter: Option<&Filter>,
        limit: Option<usize>,
    ) -> Result<Vec<Record>> {
        let table = self
            .connection
            .open_table(table_name)
            .execute()
            .await
            .with_context(|| format!("Failed to open table '{table_name}'"))?;

        let mut q = table.query();
        if let Some(f) = filter {
            q = q.only_if(filter_to_sql(f));
        }
        if let Some(n) = limit {
            q = q.limit(n);
        }

        let batches: Vec<RecordBatch> = q
            .execute()
            .await
            .with_context(|| format!("Failed to query '{table_name}'"))?
            .try_collect()
            .await?;

        let mut results = Vec::new();
        for batch in &batches {
            batch_to_records(batch, &mut results)?;
        }
        Ok(results)
    }

    async fn delete(&self, table_name: &str, filter: &Filter) -> Result<()> {
        let table = self
            .connection
            .open_table(table_name)
            .execute()
            .await
            .with_context(|| format!("Failed to open table '{table_name}'"))?;

        table
            .delete(&filter_to_sql(filter))
            .await
            .with_context(|| format!("Failed to delete from '{table_name}'"))?;
        Ok(())
    }

    async fn count(&self, table_name: &str, filter: Option<&Filter>) -> Result<usize> {
        let table = self
            .connection
            .open_table(table_name)
            .execute()
            .await
            .with_context(|| format!("Failed to open table '{table_name}'"))?;

        let mut q = table.query();
        if let Some(f) = filter {
            q = q.only_if(filter_to_sql(f));
        }
        let batches: Vec<RecordBatch> = q.execute().await?.try_collect().await?;
        Ok(batches.iter().map(|b| b.num_rows()).sum())
    }

    async fn vector_search(
        &self,
        table_name: &str,
        vector_column: &str,
        vector: Vec<f32>,
        limit: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<ScoredRecord>> {
        let table = self
            .connection
            .open_table(table_name)
            .execute()
            .await
            .with_context(|| format!("Failed to open table '{table_name}'"))?;

        let mut q = table.vector_search(vector)?.column(vector_column);
        q = q.limit(limit);
        if let Some(f) = filter {
            q = q.only_if(filter_to_sql(f));
        }

        let batches: Vec<RecordBatch> = q.execute().await?.try_collect().await?;

        let mut results = Vec::new();
        for batch in &batches {
            let distance_col = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for row in 0..batch.num_rows() {
                let mut record = Vec::new();
                for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                    if field.name() == "_distance" {
                        continue;
                    }
                    let val = extract_field_value(batch, col_idx, row, field)?;
                    record.push((field.name().clone(), val));
                }

                let distance = distance_col.map_or(0.0, |c| c.value(row));
                let score = 1.0 / (1.0 + distance);

                results.push(ScoredRecord { record, score });
            }
        }
        Ok(results)
    }
}
