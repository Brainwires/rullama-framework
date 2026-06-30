//! Shared SQL generation layer for SQL-based database backends.
//!
//! This module provides a [`SqlDialect`](crate::databases::sql::SqlDialect)
//! trait and a set of builder functions that generate parameterized SQL
//! statements from the backend-agnostic
//! [`Filter`](crate::databases::types::Filter),
//! [`FieldDef`](crate::databases::types::FieldDef), and
//! [`FieldValue`](crate::databases::types::FieldValue) types.
//!
//! Each SQL backend provides its own dialect implementation that handles
//! differences in placeholder syntax, identifier quoting, and type mapping:
//!
//! - **PostgreSQL** (`PostgresDialect`) — `$N` placeholders, `"double-quote"` identifiers, pgvector types
//! - **MySQL** (`MySqlDialect`) — `?` placeholders, `` `backtick` `` identifiers, JSON for vectors
//! - **SurrealDB** (`SurrealDialect`) — `$N` placeholders, `"double-quote"` identifiers, native array types
//!
//! ## Design goals
//!
//! - **Zero raw SQL in domain stores** — stores call these builders instead
//!   of hand-writing SQL strings.
//! - **Parameterized queries only** — values are always passed as bind
//!   parameters, never interpolated into the query string.
//! - **Dialect-agnostic filter tree** — the recursive
//!   [`filter_to_sql`](crate::databases::sql::filter_to_sql) function walks
//!   a [`Filter`](crate::databases::types::Filter) tree and emits the
//!   correct SQL for any dialect.

pub mod mysql;
pub mod postgres;
pub mod surrealdb;

use crate::databases::types::{FieldDef, FieldType, FieldValue, Filter};

// ── SqlDialect trait ──────────────────────────────────────────────────────

/// Abstraction over SQL dialect differences.
///
/// Implementors translate the generic rullama types into dialect-specific
/// SQL fragments.  The builder functions in this module accept a `&dyn SqlDialect`
/// so they can generate correct SQL for any supported database.
pub trait SqlDialect {
    /// Map a [`FieldType`] to the dialect's column type string.
    ///
    /// For example, `FieldType::Utf8` might become `"TEXT"` on PostgreSQL and
    /// MySQL, while `FieldType::Vector(384)` becomes `"vector(384)"` on
    /// PostgreSQL (pgvector) but `"JSON"` on MySQL.
    fn map_type(&self, field_type: &FieldType) -> String;

    /// Return the bind-parameter placeholder for the *n*-th parameter (1-based).
    ///
    /// PostgreSQL uses `$1`, `$2`, ...; MySQL uses `?` regardless of position.
    fn placeholder(&self, n: usize) -> String;

    /// Quote an identifier (table name, column name) for the dialect.
    ///
    /// PostgreSQL and SurrealDB use double-quotes (`"id"`), MySQL uses
    /// backticks (`` `id` ``).
    fn quote_ident(&self, ident: &str) -> String;
}

// ── Filter → SQL ──────────────────────────────────────────────────────────

/// Convert a [`Filter`] tree into a parameterized SQL `WHERE` clause.
///
/// Returns `(sql_fragment, bind_values)`.  The `param_offset` argument
/// indicates the starting parameter index (1-based) so that callers can
/// combine filter SQL with other parameters in the same statement.
///
/// # Examples
///
/// ```ignore
/// let filter = Filter::And(vec![
///     Filter::Eq("name".into(), FieldValue::Utf8(Some("Alice".into()))),
///     Filter::Gt("age".into(), FieldValue::Int32(Some(21))),
/// ]);
/// let (sql, vals) = filter_to_sql(&filter, &PostgresDialect, 1);
/// assert_eq!(sql, r#"("name" = $1 AND "age" > $2)"#);
/// ```
pub fn filter_to_sql(
    filter: &Filter,
    dialect: &dyn SqlDialect,
    param_offset: usize,
) -> (String, Vec<FieldValue>) {
    match filter {
        Filter::Eq(col, val) => {
            let sql = format!(
                "{} = {}",
                dialect.quote_ident(col),
                dialect.placeholder(param_offset)
            );
            (sql, vec![val.clone()])
        }
        Filter::Ne(col, val) => {
            let sql = format!(
                "{} != {}",
                dialect.quote_ident(col),
                dialect.placeholder(param_offset)
            );
            (sql, vec![val.clone()])
        }
        Filter::Lt(col, val) => {
            let sql = format!(
                "{} < {}",
                dialect.quote_ident(col),
                dialect.placeholder(param_offset)
            );
            (sql, vec![val.clone()])
        }
        Filter::Lte(col, val) => {
            let sql = format!(
                "{} <= {}",
                dialect.quote_ident(col),
                dialect.placeholder(param_offset)
            );
            (sql, vec![val.clone()])
        }
        Filter::Gt(col, val) => {
            let sql = format!(
                "{} > {}",
                dialect.quote_ident(col),
                dialect.placeholder(param_offset)
            );
            (sql, vec![val.clone()])
        }
        Filter::Gte(col, val) => {
            let sql = format!(
                "{} >= {}",
                dialect.quote_ident(col),
                dialect.placeholder(param_offset)
            );
            (sql, vec![val.clone()])
        }
        Filter::NotNull(col) => {
            let sql = format!("{} IS NOT NULL", dialect.quote_ident(col));
            (sql, vec![])
        }
        Filter::IsNull(col) => {
            let sql = format!("{} IS NULL", dialect.quote_ident(col));
            (sql, vec![])
        }
        Filter::In(col, values) => {
            if values.is_empty() {
                // Empty IN list is always false.
                return ("1 = 0".to_string(), vec![]);
            }
            let placeholders: Vec<String> = (0..values.len())
                .map(|i| dialect.placeholder(param_offset + i))
                .collect();
            let sql = format!(
                "{} IN ({})",
                dialect.quote_ident(col),
                placeholders.join(", ")
            );
            (sql, values.clone())
        }
        Filter::And(filters) => {
            if filters.is_empty() {
                return ("1 = 1".to_string(), vec![]);
            }
            let mut parts = Vec::new();
            let mut all_vals = Vec::new();
            let mut offset = param_offset;
            for f in filters {
                let (sql, vals) = filter_to_sql(f, dialect, offset);
                offset += vals.len();
                parts.push(sql);
                all_vals.extend(vals);
            }
            let sql = format!("({})", parts.join(" AND "));
            (sql, all_vals)
        }
        Filter::Or(filters) => {
            if filters.is_empty() {
                return ("1 = 0".to_string(), vec![]);
            }
            let mut parts = Vec::new();
            let mut all_vals = Vec::new();
            let mut offset = param_offset;
            for f in filters {
                let (sql, vals) = filter_to_sql(f, dialect, offset);
                offset += vals.len();
                parts.push(sql);
                all_vals.extend(vals);
            }
            let sql = format!("({})", parts.join(" OR "));
            (sql, all_vals)
        }
        Filter::Raw(raw) => (raw.clone(), vec![]),
    }
}

// ── Builder functions ─────────────────────────────────────────────────────

/// Build a `CREATE TABLE IF NOT EXISTS` statement.
///
/// The first field in `schema` is used as the primary key.
pub fn build_create_table(table: &str, schema: &[FieldDef], dialect: &dyn SqlDialect) -> String {
    let columns: Vec<String> = schema
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let col_type = dialect.map_type(&f.field_type);
            let nullable = if f.nullable { "" } else { " NOT NULL" };
            let pk = if i == 0 { " PRIMARY KEY" } else { "" };
            format!(
                "{} {}{}{}",
                dialect.quote_ident(&f.name),
                col_type,
                nullable,
                pk
            )
        })
        .collect();

    format!(
        "CREATE TABLE IF NOT EXISTS {} ({})",
        dialect.quote_ident(table),
        columns.join(", ")
    )
}

/// Build a parameterized `INSERT INTO` statement.
///
/// Returns `(sql, flattened_values)` where `flattened_values` contains the
/// bind parameters for **all** records in insertion order.
pub fn build_insert(
    table: &str,
    column_names: &[&str],
    records: &[Vec<FieldValue>],
    dialect: &dyn SqlDialect,
) -> (String, Vec<FieldValue>) {
    let quoted_cols: Vec<String> = column_names
        .iter()
        .map(|c| dialect.quote_ident(c))
        .collect();

    let mut all_vals = Vec::new();
    let mut row_groups = Vec::new();
    let mut idx = 1usize;

    for row in records {
        let placeholders: Vec<String> = row
            .iter()
            .map(|v| {
                let p = dialect.placeholder(idx);
                idx += 1;
                all_vals.push(v.clone());
                p
            })
            .collect();
        row_groups.push(format!("({})", placeholders.join(", ")));
    }

    let sql = format!(
        "INSERT INTO {} ({}) VALUES {}",
        dialect.quote_ident(table),
        quoted_cols.join(", "),
        row_groups.join(", ")
    );

    (sql, all_vals)
}

/// Build a parameterized `SELECT *` statement with optional filter and limit.
///
/// Returns `(sql, bind_values)`.
pub fn build_select(
    table: &str,
    filter: Option<&Filter>,
    limit: Option<usize>,
    dialect: &dyn SqlDialect,
) -> (String, Vec<FieldValue>) {
    let mut sql = format!("SELECT * FROM {}", dialect.quote_ident(table));
    let mut vals = Vec::new();

    if let Some(f) = filter {
        let (where_sql, where_vals) = filter_to_sql(f, dialect, 1);
        sql.push_str(&format!(" WHERE {}", where_sql));
        vals = where_vals;
    }

    if let Some(n) = limit {
        sql.push_str(&format!(" LIMIT {}", n));
    }

    (sql, vals)
}

/// Build a parameterized `DELETE FROM` statement.
///
/// Returns `(sql, bind_values)`.
pub fn build_delete(
    table: &str,
    filter: &Filter,
    dialect: &dyn SqlDialect,
) -> (String, Vec<FieldValue>) {
    let (where_sql, vals) = filter_to_sql(filter, dialect, 1);
    let sql = format!(
        "DELETE FROM {} WHERE {}",
        dialect.quote_ident(table),
        where_sql
    );
    (sql, vals)
}

/// Build a parameterized `SELECT COUNT(*)` statement with optional filter.
///
/// Returns `(sql, bind_values)`.
pub fn build_count(
    table: &str,
    filter: Option<&Filter>,
    dialect: &dyn SqlDialect,
) -> (String, Vec<FieldValue>) {
    let mut sql = format!("SELECT COUNT(*) FROM {}", dialect.quote_ident(table));
    let mut vals = Vec::new();

    if let Some(f) = filter {
        let (where_sql, where_vals) = filter_to_sql(f, dialect, 1);
        sql.push_str(&format!(" WHERE {}", where_sql));
        vals = where_vals;
    }

    (sql, vals)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::mysql::MySqlDialect;
    use super::postgres::PostgresDialect;
    use super::*;

    // ── filter_to_sql tests ───────────────────────────────────────────

    #[test]
    fn test_filter_eq_postgres() {
        let filter = Filter::Eq("name".into(), FieldValue::Utf8(Some("Alice".into())));
        let (sql, vals) = filter_to_sql(&filter, &PostgresDialect, 1);
        assert_eq!(sql, r#""name" = $1"#);
        assert_eq!(vals.len(), 1);
    }

    #[test]
    fn test_filter_eq_mysql() {
        let filter = Filter::Eq("name".into(), FieldValue::Utf8(Some("Alice".into())));
        let (sql, vals) = filter_to_sql(&filter, &MySqlDialect, 1);
        assert_eq!(sql, "`name` = ?");
        assert_eq!(vals.len(), 1);
    }

    #[test]
    fn test_filter_and_compound() {
        let filter = Filter::And(vec![
            Filter::Eq("name".into(), FieldValue::Utf8(Some("Alice".into()))),
            Filter::Gt("age".into(), FieldValue::Int32(Some(21))),
        ]);
        let (sql, vals) = filter_to_sql(&filter, &PostgresDialect, 1);
        assert_eq!(sql, r#"("name" = $1 AND "age" > $2)"#);
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_filter_or() {
        let filter = Filter::Or(vec![
            Filter::Eq("status".into(), FieldValue::Utf8(Some("active".into()))),
            Filter::Eq("status".into(), FieldValue::Utf8(Some("pending".into()))),
        ]);
        let (sql, vals) = filter_to_sql(&filter, &PostgresDialect, 1);
        assert_eq!(sql, r#"("status" = $1 OR "status" = $2)"#);
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_filter_in() {
        let filter = Filter::In(
            "id".into(),
            vec![
                FieldValue::Int64(Some(1)),
                FieldValue::Int64(Some(2)),
                FieldValue::Int64(Some(3)),
            ],
        );
        let (sql, vals) = filter_to_sql(&filter, &PostgresDialect, 1);
        assert_eq!(sql, r#""id" IN ($1, $2, $3)"#);
        assert_eq!(vals.len(), 3);
    }

    #[test]
    fn test_filter_in_mysql() {
        let filter = Filter::In(
            "id".into(),
            vec![FieldValue::Int64(Some(1)), FieldValue::Int64(Some(2))],
        );
        let (sql, vals) = filter_to_sql(&filter, &MySqlDialect, 1);
        assert_eq!(sql, "`id` IN (?, ?)");
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_filter_null_checks() {
        let (sql, vals) = filter_to_sql(&Filter::IsNull("email".into()), &PostgresDialect, 1);
        assert_eq!(sql, r#""email" IS NULL"#);
        assert!(vals.is_empty());

        let (sql, vals) = filter_to_sql(&Filter::NotNull("email".into()), &PostgresDialect, 1);
        assert_eq!(sql, r#""email" IS NOT NULL"#);
        assert!(vals.is_empty());
    }

    #[test]
    fn test_filter_empty_and_or() {
        let (sql, vals) = filter_to_sql(&Filter::And(vec![]), &PostgresDialect, 1);
        assert_eq!(sql, "1 = 1");
        assert!(vals.is_empty());

        let (sql, vals) = filter_to_sql(&Filter::Or(vec![]), &PostgresDialect, 1);
        assert_eq!(sql, "1 = 0");
        assert!(vals.is_empty());
    }

    #[test]
    fn test_filter_empty_in() {
        let (sql, vals) = filter_to_sql(&Filter::In("x".into(), vec![]), &PostgresDialect, 1);
        assert_eq!(sql, "1 = 0");
        assert!(vals.is_empty());
    }

    #[test]
    fn test_filter_raw() {
        let (sql, vals) = filter_to_sql(
            &Filter::Raw("custom_fn(col) > 0".into()),
            &PostgresDialect,
            1,
        );
        assert_eq!(sql, "custom_fn(col) > 0");
        assert!(vals.is_empty());
    }

    #[test]
    fn test_filter_nested_and_or() {
        let filter = Filter::And(vec![
            Filter::Eq("a".into(), FieldValue::Int32(Some(1))),
            Filter::Or(vec![
                Filter::Eq("b".into(), FieldValue::Int32(Some(2))),
                Filter::Eq("c".into(), FieldValue::Int32(Some(3))),
            ]),
        ]);
        let (sql, vals) = filter_to_sql(&filter, &PostgresDialect, 1);
        assert_eq!(sql, r#"("a" = $1 AND ("b" = $2 OR "c" = $3))"#);
        assert_eq!(vals.len(), 3);
    }

    #[test]
    fn test_filter_comparison_ops() {
        let (sql, _) = filter_to_sql(
            &Filter::Lt("x".into(), FieldValue::Int32(Some(5))),
            &PostgresDialect,
            1,
        );
        assert_eq!(sql, r#""x" < $1"#);

        let (sql, _) = filter_to_sql(
            &Filter::Lte("x".into(), FieldValue::Int32(Some(5))),
            &PostgresDialect,
            1,
        );
        assert_eq!(sql, r#""x" <= $1"#);

        let (sql, _) = filter_to_sql(
            &Filter::Ne("x".into(), FieldValue::Int32(Some(5))),
            &PostgresDialect,
            1,
        );
        assert_eq!(sql, r#""x" != $1"#);

        let (sql, _) = filter_to_sql(
            &Filter::Gte("x".into(), FieldValue::Int32(Some(5))),
            &PostgresDialect,
            1,
        );
        assert_eq!(sql, r#""x" >= $1"#);
    }

    // ── Builder tests ─────────────────────────────────────────────────

    #[test]
    fn test_build_create_table_postgres() {
        let schema = vec![
            FieldDef::required("id", FieldType::Utf8),
            FieldDef::required("count", FieldType::Int64),
            FieldDef::optional("embedding", FieldType::Vector(384)),
        ];
        let sql = build_create_table("my_table", &schema, &PostgresDialect);
        assert_eq!(
            sql,
            r#"CREATE TABLE IF NOT EXISTS "my_table" ("id" TEXT NOT NULL PRIMARY KEY, "count" BIGINT NOT NULL, "embedding" vector(384))"#
        );
    }

    #[test]
    fn test_build_create_table_mysql() {
        let schema = vec![
            FieldDef::required("id", FieldType::Utf8),
            FieldDef::optional("active", FieldType::Boolean),
        ];
        let sql = build_create_table("users", &schema, &MySqlDialect);
        assert_eq!(
            sql,
            "CREATE TABLE IF NOT EXISTS `users` (`id` TEXT NOT NULL PRIMARY KEY, `active` BOOLEAN)"
        );
    }

    #[test]
    fn test_build_insert() {
        let cols = ["id", "name"];
        let records = vec![
            vec![
                FieldValue::Utf8(Some("1".into())),
                FieldValue::Utf8(Some("Alice".into())),
            ],
            vec![
                FieldValue::Utf8(Some("2".into())),
                FieldValue::Utf8(Some("Bob".into())),
            ],
        ];
        let (sql, vals) = build_insert("users", &cols, &records, &PostgresDialect);
        assert_eq!(
            sql,
            r#"INSERT INTO "users" ("id", "name") VALUES ($1, $2), ($3, $4)"#
        );
        assert_eq!(vals.len(), 4);
    }

    #[test]
    fn test_build_insert_mysql() {
        let cols = ["id", "name"];
        let records = vec![vec![
            FieldValue::Utf8(Some("1".into())),
            FieldValue::Utf8(Some("Alice".into())),
        ]];
        let (sql, vals) = build_insert("users", &cols, &records, &MySqlDialect);
        assert_eq!(sql, "INSERT INTO `users` (`id`, `name`) VALUES (?, ?)");
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_build_select_no_filter() {
        let (sql, vals) = build_select("messages", None, Some(10), &PostgresDialect);
        assert_eq!(sql, r#"SELECT * FROM "messages" LIMIT 10"#);
        assert!(vals.is_empty());
    }

    #[test]
    fn test_build_select_with_filter() {
        let filter = Filter::Eq("user_id".into(), FieldValue::Utf8(Some("u1".into())));
        let (sql, vals) = build_select("messages", Some(&filter), None, &PostgresDialect);
        assert_eq!(sql, r#"SELECT * FROM "messages" WHERE "user_id" = $1"#);
        assert_eq!(vals.len(), 1);
    }

    #[test]
    fn test_build_delete() {
        let filter = Filter::Eq("id".into(), FieldValue::Utf8(Some("123".into())));
        let (sql, vals) = build_delete("tasks", &filter, &PostgresDialect);
        assert_eq!(sql, r#"DELETE FROM "tasks" WHERE "id" = $1"#);
        assert_eq!(vals.len(), 1);
    }

    #[test]
    fn test_build_count_no_filter() {
        let (sql, vals) = build_count("events", None, &PostgresDialect);
        assert_eq!(sql, r#"SELECT COUNT(*) FROM "events""#);
        assert!(vals.is_empty());
    }

    #[test]
    fn test_build_count_with_filter() {
        let filter = Filter::Gt("score".into(), FieldValue::Float64(Some(0.5)));
        let (sql, vals) = build_count("events", Some(&filter), &PostgresDialect);
        assert_eq!(sql, r#"SELECT COUNT(*) FROM "events" WHERE "score" > $1"#);
        assert_eq!(vals.len(), 1);
    }
}
