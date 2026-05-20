//! Persistence Layer
//!
//! This module provides SQLite storage for task clusters and technique performance.

use super::clustering::TaskCluster;
use super::techniques::PromptingTechnique;
use super::temperature::TemperaturePerformance;
use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use serde_json;
use std::path::Path;

/// Manages persistent storage of task clusters and performance data
pub struct ClusterStorage {
    conn: Connection,
}

impl ClusterStorage {
    /// Create a new cluster storage at the specified database path
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let conn = Connection::open(db_path)?;

        // Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON", [])?;

        // Create tables if they don't exist
        Self::create_tables(&conn)?;

        Ok(Self { conn })
    }

    /// Create all required tables
    fn create_tables(conn: &Connection) -> Result<()> {
        // Clusters table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS clusters (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                embedding BLOB NOT NULL,
                techniques TEXT NOT NULL,
                example_tasks TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )",
            [],
        )?;

        // Technique performance table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS technique_performance (
                cluster_id TEXT NOT NULL,
                technique TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                avg_iterations REAL NOT NULL DEFAULT 0.0,
                avg_quality REAL NOT NULL DEFAULT 0.0,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (cluster_id, technique),
                FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
            )",
            [],
        )?;

        // Temperature performance table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS temperature_performance (
                cluster_id TEXT NOT NULL,
                temperature_key INTEGER NOT NULL,
                success_rate REAL NOT NULL DEFAULT 0.5,
                avg_quality REAL NOT NULL DEFAULT 0.5,
                sample_count INTEGER NOT NULL DEFAULT 0,
                last_updated INTEGER NOT NULL,
                PRIMARY KEY (cluster_id, temperature_key),
                FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
            )",
            [],
        )?;

        // Create indexes for common queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clusters_updated
             ON clusters(updated_at DESC)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_technique_perf_cluster
             ON technique_performance(cluster_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_temp_perf_cluster
             ON temperature_performance(cluster_id)",
            [],
        )?;

        Ok(())
    }

    /// Save a task cluster to the database
    pub fn save_cluster(&mut self, cluster: &TaskCluster) -> Result<()> {
        let embedding_bytes =
            bincode::serde::encode_to_vec(&cluster.embedding, bincode::config::standard())
                .context("Failed to serialize embedding")?;

        let techniques_json =
            serde_json::to_string(&cluster.techniques).context("Failed to serialize techniques")?;

        let tasks_json = serde_json::to_string(&cluster.example_tasks)
            .context("Failed to serialize example tasks")?;

        let timestamp = chrono::Utc::now().timestamp();

        self.conn.execute(
            "INSERT OR REPLACE INTO clusters
             (id, description, embedding, techniques, example_tasks, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5,
                     COALESCE((SELECT created_at FROM clusters WHERE id = ?1), ?6),
                     ?6)",
            params![
                cluster.id,
                cluster.description,
                embedding_bytes,
                techniques_json,
                tasks_json,
                timestamp,
            ],
        )?;

        Ok(())
    }

    /// Load all clusters from the database
    pub fn load_clusters(&self) -> Result<Vec<TaskCluster>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, description, embedding, techniques, example_tasks
             FROM clusters
             ORDER BY updated_at DESC",
        )?;

        let clusters = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let description: String = row.get(1)?;
                let embedding_bytes: Vec<u8> = row.get(2)?;
                let techniques_json: String = row.get(3)?;
                let tasks_json: String = row.get(4)?;

                // Deserialize embedding
                let (embedding, _): (Vec<f32>, _) = bincode::serde::decode_from_slice(
                    &embedding_bytes,
                    bincode::config::standard(),
                )
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                // Deserialize techniques
                let techniques: Vec<PromptingTechnique> = serde_json::from_str(&techniques_json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                // Deserialize example tasks
                let example_tasks: Vec<String> = serde_json::from_str(&tasks_json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                Ok(TaskCluster {
                    id,
                    description,
                    embedding,
                    techniques,
                    example_tasks,
                    seal_query_cores: Vec::new(), // Not stored currently
                    avg_seal_quality: 0.5,        // Not stored currently
                    recommended_complexity: super::techniques::ComplexityLevel::Moderate,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(clusters)
    }

    /// Load a specific cluster by ID
    pub fn load_cluster(&self, cluster_id: &str) -> Result<Option<TaskCluster>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, description, embedding, techniques, example_tasks
             FROM clusters
             WHERE id = ?1",
        )?;

        let mut rows = stmt.query([cluster_id])?;

        if let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let description: String = row.get(1)?;
            let embedding_bytes: Vec<u8> = row.get(2)?;
            let techniques_json: String = row.get(3)?;
            let tasks_json: String = row.get(4)?;

            let (embedding, _): (Vec<f32>, _) =
                bincode::serde::decode_from_slice(&embedding_bytes, bincode::config::standard())?;
            let techniques = serde_json::from_str(&techniques_json)?;
            let example_tasks = serde_json::from_str(&tasks_json)?;

            Ok(Some(TaskCluster {
                id,
                description,
                embedding,
                techniques,
                example_tasks,
                seal_query_cores: Vec::new(),
                avg_seal_quality: 0.5,
                recommended_complexity: super::techniques::ComplexityLevel::Moderate,
            }))
        } else {
            Ok(None)
        }
    }

    /// Delete a cluster and all its associated performance data
    pub fn delete_cluster(&mut self, cluster_id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM clusters WHERE id = ?1", [cluster_id])?;
        Ok(())
    }

    /// Save temperature performance for a cluster
    pub fn save_temperature_performance(
        &mut self,
        cluster_id: &str,
        temperature_key: i32,
        perf: &TemperaturePerformance,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO temperature_performance
             (cluster_id, temperature_key, success_rate, avg_quality, sample_count, last_updated)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                cluster_id,
                temperature_key,
                perf.success_rate,
                perf.avg_quality,
                perf.sample_count,
                perf.last_updated,
            ],
        )?;
        Ok(())
    }

    /// Load temperature performance for a cluster
    pub fn load_temperature_performance(
        &self,
        cluster_id: &str,
    ) -> Result<Vec<(i32, TemperaturePerformance)>> {
        let mut stmt = self.conn.prepare(
            "SELECT temperature_key, success_rate, avg_quality, sample_count, last_updated
             FROM temperature_performance
             WHERE cluster_id = ?1",
        )?;

        let perfs = stmt
            .query_map([cluster_id], |row| {
                let temp_key: i32 = row.get(0)?;
                let perf = TemperaturePerformance {
                    success_rate: row.get(1)?,
                    avg_quality: row.get(2)?,
                    sample_count: row.get(3)?,
                    last_updated: row.get(4)?,
                };
                Ok((temp_key, perf))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(perfs)
    }

    /// Get statistics about stored data
    pub fn get_stats(&self) -> Result<StorageStats> {
        let cluster_count: u32 =
            self.conn
                .query_row("SELECT COUNT(*) FROM clusters", [], |row| row.get(0))?;

        let technique_perf_count: u32 =
            self.conn
                .query_row("SELECT COUNT(*) FROM technique_performance", [], |row| {
                    row.get(0)
                })?;

        let temp_perf_count: u32 =
            self.conn
                .query_row("SELECT COUNT(*) FROM temperature_performance", [], |row| {
                    row.get(0)
                })?;

        let db_size_bytes = std::fs::metadata(self.conn.path().unwrap_or_default())
            .map(|m| m.len())
            .unwrap_or(0);

        Ok(StorageStats {
            cluster_count,
            technique_perf_count,
            temp_perf_count,
            db_size_bytes,
        })
    }

    /// Vacuum the database to reclaim space
    pub fn vacuum(&mut self) -> Result<()> {
        self.conn.execute("VACUUM", [])?;
        Ok(())
    }
}

/// Statistics about stored data
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Number of stored task clusters.
    pub cluster_count: u32,
    /// Number of technique performance records.
    pub technique_perf_count: u32,
    /// Number of temperature performance records.
    pub temp_perf_count: u32,
    /// Total database size in bytes.
    pub db_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::techniques::ComplexityLevel;

    #[test]
    fn test_create_and_load_cluster() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let mut storage = ClusterStorage::new(&db_path).unwrap();

        // Create test cluster
        let cluster = TaskCluster {
            id: "test_cluster".to_string(),
            description: "Test cluster description".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            techniques: vec![
                PromptingTechnique::ChainOfThought,
                PromptingTechnique::RolePlaying,
            ],
            example_tasks: vec!["Task 1".to_string(), "Task 2".to_string()],
            seal_query_cores: vec![],
            avg_seal_quality: 0.8,
            recommended_complexity: ComplexityLevel::Moderate,
        };

        // Save cluster
        storage.save_cluster(&cluster).unwrap();

        // Load cluster
        let loaded = storage.load_cluster("test_cluster").unwrap().unwrap();
        assert_eq!(loaded.id, "test_cluster");
        assert_eq!(loaded.description, "Test cluster description");
        assert_eq!(loaded.embedding, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(loaded.techniques.len(), 2);
        assert_eq!(loaded.example_tasks.len(), 2);
    }

    #[test]
    fn test_load_all_clusters() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let mut storage = ClusterStorage::new(&db_path).unwrap();

        // Create multiple clusters
        for i in 0..3 {
            let cluster = TaskCluster {
                id: format!("cluster_{}", i),
                description: format!("Cluster {}", i),
                embedding: vec![i as f32; 4],
                techniques: vec![PromptingTechnique::ChainOfThought],
                example_tasks: vec![format!("Task {}", i)],
                seal_query_cores: vec![],
                avg_seal_quality: 0.5,
                recommended_complexity: ComplexityLevel::Simple,
            };
            storage.save_cluster(&cluster).unwrap();
        }

        // Load all
        let clusters = storage.load_clusters().unwrap();
        assert_eq!(clusters.len(), 3);
    }

    #[test]
    fn test_delete_cluster() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let mut storage = ClusterStorage::new(&db_path).unwrap();

        let cluster = TaskCluster::new(
            "test".to_string(),
            "Test".to_string(),
            vec![0.5; 4],
            vec![PromptingTechnique::RolePlaying],
            vec!["Example".to_string()],
        );

        storage.save_cluster(&cluster).unwrap();
        assert!(storage.load_cluster("test").unwrap().is_some());

        storage.delete_cluster("test").unwrap();
        assert!(storage.load_cluster("test").unwrap().is_none());
    }

    #[test]
    fn test_temperature_performance_storage() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let mut storage = ClusterStorage::new(&db_path).unwrap();

        // Create cluster first
        let cluster = TaskCluster::new(
            "test".to_string(),
            "Test".to_string(),
            vec![0.5; 4],
            vec![],
            vec![],
        );
        storage.save_cluster(&cluster).unwrap();

        // Save temperature performance
        let perf = TemperaturePerformance {
            success_rate: 0.85,
            avg_quality: 0.9,
            sample_count: 10,
            last_updated: 12345,
        };

        storage
            .save_temperature_performance("test", 0, &perf)
            .unwrap();

        // Load temperature performance
        let loaded = storage.load_temperature_performance("test").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, 0); // temperature_key
        assert_eq!(loaded[0].1.sample_count, 10);
        assert!((loaded[0].1.success_rate - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_storage_stats() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let mut storage = ClusterStorage::new(&db_path).unwrap();

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.cluster_count, 0);

        // Add a cluster
        let cluster = TaskCluster::new(
            "test".to_string(),
            "Test".to_string(),
            vec![0.5; 4],
            vec![],
            vec![],
        );
        storage.save_cluster(&cluster).unwrap();

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.cluster_count, 1);
        assert!(stats.db_size_bytes > 0);
    }
}
