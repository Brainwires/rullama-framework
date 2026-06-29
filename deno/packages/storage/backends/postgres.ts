/**
 * PostgreSQL + pgvector backend for StorageBackend and VectorDatabase.
 *
 * Port of the Rust `rullama-storage/src/databases/postgres/mod.rs`.
 *
 * Uses the `pg` npm package for connection pooling and pgvector's `<=>`
 * cosine distance operator for vector similarity search.
 * @module
 */

import pg from "pg";
import type {
  ChunkMetadata,
  DatabaseStats,
  SearchResult,
} from "@rullama/core";
import type { StorageBackend, VectorDatabase } from "../traits.ts";
import type {
  FieldDef,
  FieldValue,
  Filter,
  Record as BwRecord,
  ScoredRecord,
} from "../types.ts";

const DEFAULT_TABLE = "code_embeddings";
const DEFAULT_URL = "postgresql://localhost:5432/rullama";

// ---------------------------------------------------------------------------
// SQL helpers
// ---------------------------------------------------------------------------

/** Map a FieldType kind to a PostgreSQL column type string. */
function mapFieldType(ft: FieldDef["fieldType"]): string {
  switch (ft.kind) {
    case "Utf8":
      return "TEXT";
    case "Int32":
      return "INTEGER";
    case "Int64":
      return "BIGINT";
    case "UInt32":
      return "INTEGER";
    case "UInt64":
      return "BIGINT";
    case "Float32":
      return "REAL";
    case "Float64":
      return "DOUBLE PRECISION";
    case "Boolean":
      return "BOOLEAN";
    case "Vector":
      return `vector(${ft.dimension})`;
  }
}

/** Extract the raw JS value from a FieldValue for use as a pg bind parameter. */
export function fieldValueToParam(fv: FieldValue): unknown {
  switch (fv.kind) {
    case "Utf8":
    case "Int32":
    case "Int64":
    case "Float32":
    case "Float64":
    case "Boolean":
      return fv.value;
    case "UInt32":
      return fv.value;
    case "UInt64":
      return fv.value;
    case "Vector":
      // pgvector expects the text representation `[1,2,3]`.
      return `[${fv.value.join(",")}]`;
  }
}

/**
 * Convert a Filter tree into a parameterised SQL WHERE fragment.
 *
 * Returns `[sql, values]` where `values` are the bind parameters.
 * `paramOffset` is the 1-based starting parameter index.
 */
export function filterToSql(
  filter: Filter,
  paramOffset: number,
): [string, FieldValue[]] {
  switch (filter.kind) {
    case "Eq":
      return [`"${filter.field}" = $${paramOffset}`, [filter.value]];
    case "Ne":
      return [`"${filter.field}" != $${paramOffset}`, [filter.value]];
    case "Lt":
      return [`"${filter.field}" < $${paramOffset}`, [filter.value]];
    case "Lte":
      return [`"${filter.field}" <= $${paramOffset}`, [filter.value]];
    case "Gt":
      return [`"${filter.field}" > $${paramOffset}`, [filter.value]];
    case "Gte":
      return [`"${filter.field}" >= $${paramOffset}`, [filter.value]];
    case "NotNull":
      return [`"${filter.field}" IS NOT NULL`, []];
    case "IsNull":
      return [`"${filter.field}" IS NULL`, []];
    case "In": {
      if (filter.values.length === 0) return ["1 = 0", []];
      const placeholders = filter.values.map((_, i) => `$${paramOffset + i}`);
      return [
        `"${filter.field}" IN (${placeholders.join(", ")})`,
        [...filter.values],
      ];
    }
    case "And": {
      if (filter.filters.length === 0) return ["1 = 1", []];
      const parts: string[] = [];
      const allVals: FieldValue[] = [];
      let offset = paramOffset;
      for (const f of filter.filters) {
        const [sql, vals] = filterToSql(f, offset);
        offset += vals.length;
        parts.push(sql);
        allVals.push(...vals);
      }
      return [`(${parts.join(" AND ")})`, allVals];
    }
    case "Or": {
      if (filter.filters.length === 0) return ["1 = 0", []];
      const parts: string[] = [];
      const allVals: FieldValue[] = [];
      let offset = paramOffset;
      for (const f of filter.filters) {
        const [sql, vals] = filterToSql(f, offset);
        offset += vals.length;
        parts.push(sql);
        allVals.push(...vals);
      }
      return [`(${parts.join(" OR ")})`, allVals];
    }
    case "Raw":
      return [filter.expression, []];
  }
}

/** Build a CREATE TABLE IF NOT EXISTS DDL statement. */
export function buildCreateTable(
  tableName: string,
  schema: FieldDef[],
): string {
  const cols = schema.map((f, i) => {
    const pgType = mapFieldType(f.fieldType);
    const nullable = f.nullable ? "" : " NOT NULL";
    const pk = i === 0 ? " PRIMARY KEY" : "";
    return `"${f.name}" ${pgType}${nullable}${pk}`;
  });
  return `CREATE TABLE IF NOT EXISTS "${tableName}" (${cols.join(", ")})`;
}

/** Build an INSERT INTO statement with $N placeholders. Returns [sql, params]. */
export function buildInsert(
  tableName: string,
  records: BwRecord[],
): [string, unknown[]] {
  if (records.length === 0) return ["", []];
  const colNames = records[0].map(([name]) => name);
  const quotedCols = colNames.map((c) => `"${c}"`);
  const allParams: unknown[] = [];
  const rowGroups: string[] = [];
  let idx = 1;
  for (const rec of records) {
    const placeholders: string[] = [];
    for (const [, fv] of rec) {
      placeholders.push(`$${idx}`);
      allParams.push(fieldValueToParam(fv));
      idx++;
    }
    rowGroups.push(`(${placeholders.join(", ")})`);
  }
  const sql = `INSERT INTO "${tableName}" (${quotedCols.join(", ")}) VALUES ${
    rowGroups.join(", ")
  }`;
  return [sql, allParams];
}

/** Build a SELECT * with optional WHERE / LIMIT. Returns [sql, params]. */
export function buildSelect(
  tableName: string,
  filter?: Filter,
  limit?: number,
): [string, unknown[]] {
  let sql = `SELECT * FROM "${tableName}"`;
  const params: unknown[] = [];
  if (filter) {
    const [whereSql, vals] = filterToSql(filter, 1);
    sql += ` WHERE ${whereSql}`;
    params.push(...vals.map(fieldValueToParam));
  }
  if (limit !== undefined) sql += ` LIMIT ${limit}`;
  return [sql, params];
}

/** Build a DELETE FROM with WHERE. Returns [sql, params]. */
export function buildDelete(
  tableName: string,
  filter: Filter,
): [string, unknown[]] {
  const [whereSql, vals] = filterToSql(filter, 1);
  return [
    `DELETE FROM "${tableName}" WHERE ${whereSql}`,
    vals.map(fieldValueToParam),
  ];
}

/** Build a SELECT COUNT(*) with optional WHERE. Returns [sql, params]. */
export function buildCount(
  tableName: string,
  filter?: Filter,
): [string, unknown[]] {
  let sql = `SELECT COUNT(*) FROM "${tableName}"`;
  const params: unknown[] = [];
  if (filter) {
    const [whereSql, vals] = filterToSql(filter, 1);
    sql += ` WHERE ${whereSql}`;
    params.push(...vals.map(fieldValueToParam));
  }
  return [sql, params];
}

/** Parse a pg Row into a Record using column metadata. */
function rowToRecord(
  row: { [key: string]: unknown },
  fields: pg.FieldDef[],
): BwRecord {
  const record: BwRecord = [];
  for (const field of fields) {
    const name = field.name;
    const val = row[name];
    let fv: FieldValue;
    // pg returns OID-typed data; we map by the pg dataTypeID.
    // Common OIDs: 25=TEXT, 1043=VARCHAR, 23=INT4, 20=INT8,
    //   700=FLOAT4, 701=FLOAT8, 16=BOOL
    if (val === null || val === undefined) {
      fv = { kind: "Utf8", value: null };
    } else if (typeof val === "string") {
      // Could be a pgvector string like "[1,2,3]"
      if (val.startsWith("[") && val.endsWith("]")) {
        try {
          const arr = JSON.parse(val);
          if (
            Array.isArray(arr) &&
            arr.every((v: unknown) => typeof v === "number")
          ) {
            fv = { kind: "Vector", value: arr as number[] };
          } else {
            fv = { kind: "Utf8", value: val };
          }
        } catch {
          fv = { kind: "Utf8", value: val };
        }
      } else {
        fv = { kind: "Utf8", value: val };
      }
    } else if (typeof val === "number") {
      if (Number.isInteger(val)) {
        fv = { kind: "Int64", value: val };
      } else {
        fv = { kind: "Float64", value: val };
      }
    } else if (typeof val === "boolean") {
      fv = { kind: "Boolean", value: val };
    } else {
      fv = { kind: "Utf8", value: String(val) };
    }
    record.push([name, fv]);
  }
  return record;
}

// ---------------------------------------------------------------------------
// PostgresDatabase
// ---------------------------------------------------------------------------

/** Configuration for PostgresDatabase. */
export interface PostgresConfig {
  /** Full connection string, e.g. "postgresql://user:pass@host:5432/db". */
  connectionString?: string;
  /** pg.PoolConfig for fine-grained control. */
  poolConfig?: pg.PoolConfig;
  /** Name of the embeddings table (default: "code_embeddings"). */
  tableName?: string;
}

/**
 * PostgreSQL + pgvector backed database implementing both StorageBackend
 * and VectorDatabase.
 */
export class PostgresDatabase implements StorageBackend, VectorDatabase {
  readonly pool: pg.Pool;
  readonly tableName: string;

  constructor(config?: PostgresConfig) {
    const connString = config?.connectionString ?? DEFAULT_URL;
    this.pool = config?.poolConfig
      ? new pg.Pool(config.poolConfig)
      : new pg.Pool({ connectionString: connString });
    this.tableName = config?.tableName ?? DEFAULT_TABLE;
  }

  /** Return the default connection URL. */
  static defaultUrl(): string {
    return DEFAULT_URL;
  }

  // ── StorageBackend ─────────────────────────────────────────────────

  async ensureTable(tableName: string, schema: FieldDef[]): Promise<void> {
    const hasVector = schema.some((f) => f.fieldType.kind === "Vector");
    const client = await this.pool.connect();
    try {
      if (hasVector) {
        await client.query("CREATE EXTENSION IF NOT EXISTS vector");
      }
      const ddl = buildCreateTable(tableName, schema);
      await client.query(ddl);
    } finally {
      client.release();
    }
  }

  async insert(tableName: string, records: BwRecord[]): Promise<void> {
    if (records.length === 0) return;
    const [sql, params] = buildInsert(tableName, records);
    await this.pool.query(sql, params);
  }

  async query(
    tableName: string,
    filter?: Filter,
    limit?: number,
  ): Promise<BwRecord[]> {
    const [sql, params] = buildSelect(tableName, filter, limit);
    const result = await this.pool.query(sql, params);
    return result.rows.map((row: { [key: string]: unknown }) =>
      rowToRecord(row, result.fields)
    );
  }

  async delete(tableName: string, filter: Filter): Promise<void> {
    const [sql, params] = buildDelete(tableName, filter);
    await this.pool.query(sql, params);
  }

  async count(tableName: string, filter?: Filter): Promise<number> {
    const [sql, params] = buildCount(tableName, filter);
    const result = await this.pool.query(sql, params);
    return parseInt(result.rows[0].count, 10);
  }

  async vectorSearch(
    tableName: string,
    vectorColumn: string,
    vector: number[],
    limit: number,
    filter?: Filter,
  ): Promise<ScoredRecord[]> {
    const vecStr = `[${vector.join(",")}]`;
    let whereClause = "";
    const filterParams: unknown[] = [];

    if (filter) {
      const [sql, vals] = filterToSql(filter, 2); // $1 = vector
      whereClause = `WHERE ${sql}`;
      filterParams.push(...vals.map(fieldValueToParam));
    }

    const limitIdx = 2 + filterParams.length;
    const sql =
      `SELECT *, 1.0 - ("${vectorColumn}" <=> $1::vector) AS __score ` +
      `FROM "${tableName}" ${whereClause} ` +
      `ORDER BY "${vectorColumn}" <=> $1::vector ` +
      `LIMIT $${limitIdx}`;

    const allParams: unknown[] = [vecStr, ...filterParams, limit];
    const result = await this.pool.query(sql, allParams);

    return result.rows.map((row: { [key: string]: unknown }) => {
      const score = Number(row.__score ?? 0);
      const rec: BwRecord = [];
      for (const field of result.fields) {
        if (field.name === "__score") continue;
        const val = row[field.name];
        let fv: FieldValue;
        if (val === null || val === undefined) {
          fv = { kind: "Utf8", value: null };
        } else if (typeof val === "string") {
          fv = { kind: "Utf8", value: val };
        } else if (typeof val === "number") {
          fv = Number.isInteger(val)
            ? { kind: "Int64", value: val }
            : { kind: "Float64", value: val };
        } else if (typeof val === "boolean") {
          fv = { kind: "Boolean", value: val };
        } else {
          fv = { kind: "Utf8", value: String(val) };
        }
        rec.push([field.name, fv]);
      }
      return { record: rec, score };
    });
  }

  // ── VectorDatabase ─────────────────────────────────────────────────

  async initialize(dimension: number): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query("CREATE EXTENSION IF NOT EXISTS vector");
      await client.query(`
        CREATE TABLE IF NOT EXISTS ${this.tableName} (
          id          BIGSERIAL PRIMARY KEY,
          embedding   vector(${dimension}),
          file_path   TEXT    NOT NULL,
          root_path   TEXT,
          project     TEXT,
          start_line  INTEGER NOT NULL,
          end_line    INTEGER NOT NULL,
          language    TEXT,
          extension   TEXT,
          file_hash   TEXT    NOT NULL,
          indexed_at  BIGINT  NOT NULL,
          content     TEXT    NOT NULL
        )
      `);
      await client.query(
        `CREATE INDEX IF NOT EXISTS idx_${this.tableName}_file_path ON ${this.tableName} (file_path)`,
      );
      await client.query(
        `CREATE INDEX IF NOT EXISTS idx_${this.tableName}_root_path ON ${this.tableName} (root_path)`,
      );
      await client.query(
        `CREATE INDEX IF NOT EXISTS idx_${this.tableName}_project ON ${this.tableName} (project)`,
      );
      await client.query(
        `CREATE INDEX IF NOT EXISTS idx_${this.tableName}_embedding ON ${this.tableName} USING hnsw (embedding vector_cosine_ops)`,
      );
    } finally {
      client.release();
    }
  }

  async storeEmbeddings(
    embeddings: number[][],
    metadata: ChunkMetadata[],
    contents: string[],
    _rootPath: string,
  ): Promise<number> {
    if (embeddings.length === 0) return 0;
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const insertSql = `
        INSERT INTO ${this.tableName}
          (embedding, file_path, root_path, project,
           start_line, end_line, language, extension,
           file_hash, indexed_at, content)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
      `;
      for (let i = 0; i < embeddings.length; i++) {
        const meta = metadata[i];
        const vecStr = `[${embeddings[i].join(",")}]`;
        await client.query(insertSql, [
          vecStr,
          meta.file_path,
          meta.root_path ?? null,
          meta.project ?? null,
          meta.start_line,
          meta.end_line,
          meta.language ?? null,
          meta.extension ?? null,
          meta.file_hash,
          meta.indexed_at,
          contents[i],
        ]);
      }
      await client.query("COMMIT");
    } catch (err) {
      await client.query("ROLLBACK");
      throw err;
    } finally {
      client.release();
    }
    return embeddings.length;
  }

  // deno-lint-ignore require-await
  async search(
    queryVector: number[],
    queryText: string,
    limit: number,
    minScore: number,
    project?: string,
    rootPath?: string,
    _hybrid?: boolean,
  ): Promise<SearchResult[]> {
    return this.searchFiltered(
      queryVector,
      queryText,
      limit,
      minScore,
      project,
      rootPath,
      _hybrid,
    );
  }

  async searchFiltered(
    queryVector: number[],
    _queryText: string,
    limit: number,
    minScore: number,
    project?: string,
    rootPath?: string,
    _hybrid?: boolean,
    fileExtensions?: string[],
    languages?: string[],
    pathPatterns?: string[],
  ): Promise<SearchResult[]> {
    const vecStr = `[${queryVector.join(",")}]`;
    const sql = `
      SELECT
        file_path, root_path, project, start_line, end_line,
        language, extension, indexed_at, content,
        1.0 - (embedding <=> $1::vector) AS vector_score
      FROM ${this.tableName}
      WHERE 1=1
        AND ($2::text IS NULL OR project = $2)
        AND ($3::text IS NULL OR root_path = $3)
        AND (cardinality($4::text[]) = 0 OR extension = ANY($4))
        AND (cardinality($5::text[]) = 0 OR language = ANY($5))
      ORDER BY embedding <=> $1::vector
      LIMIT $6
    `;
    const result = await this.pool.query(sql, [
      vecStr,
      project ?? null,
      rootPath ?? null,
      fileExtensions ?? [],
      languages ?? [],
      limit,
    ]);

    let results: SearchResult[] = result.rows
      .filter((r: { [key: string]: unknown }) =>
        Number(r.vector_score) >= minScore
      )
      .map((r: { [key: string]: unknown }) => ({
        file_path: String(r.file_path),
        root_path: r.root_path != null ? String(r.root_path) : undefined,
        content: String(r.content ?? ""),
        score: Number(r.vector_score),
        vector_score: Number(r.vector_score),
        keyword_score: undefined,
        start_line: Number(r.start_line),
        end_line: Number(r.end_line),
        language: String(r.language ?? "Unknown"),
        project: r.project != null ? String(r.project) : undefined,
        indexed_at: Number(r.indexed_at),
      }));

    // Post-filter by path patterns (simple substring match fallback).
    if (pathPatterns && pathPatterns.length > 0) {
      results = results.filter((r) =>
        pathPatterns.some((p) => r.file_path.includes(p))
      );
    }

    return results;
  }

  async deleteByFile(filePath: string): Promise<number> {
    const result = await this.pool.query(
      `DELETE FROM ${this.tableName} WHERE file_path = $1`,
      [filePath],
    );
    return result.rowCount ?? 0;
  }

  async clear(): Promise<void> {
    await this.pool.query(`TRUNCATE ${this.tableName}`);
  }

  async getStatistics(): Promise<DatabaseStats> {
    const countResult = await this.pool.query(
      `SELECT COUNT(*) AS total FROM ${this.tableName}`,
    );
    const total = parseInt(countResult.rows[0].total, 10);

    const langResult = await this.pool.query(
      `SELECT language, COUNT(*) AS lang_count FROM ${this.tableName} GROUP BY language`,
    );
    const language_breakdown: [string, number][] = langResult.rows.map(
      (r: { [key: string]: unknown }) => [
        String(r.language ?? "Unknown"),
        parseInt(String(r.lang_count), 10),
      ],
    );

    return {
      total_points: total,
      total_vectors: total,
      language_breakdown,
    };
  }

  async flush(): Promise<void> {
    // PostgreSQL persists transactionally; no explicit flush needed.
  }

  async countByRootPath(rootPath: string): Promise<number> {
    const result = await this.pool.query(
      `SELECT COUNT(*) AS cnt FROM ${this.tableName} WHERE root_path = $1`,
      [rootPath],
    );
    return parseInt(result.rows[0].cnt, 10);
  }

  async getIndexedFiles(rootPath: string): Promise<string[]> {
    const result = await this.pool.query(
      `SELECT DISTINCT file_path FROM ${this.tableName} WHERE root_path = $1`,
      [rootPath],
    );
    return result.rows.map((r: { [key: string]: unknown }) =>
      String(r.file_path)
    );
  }

  async searchWithEmbeddings(
    queryVector: number[],
    queryText: string,
    limit: number,
    minScore: number,
    project?: string,
    rootPath?: string,
    hybrid?: boolean,
  ): Promise<[SearchResult[], number[][]]> {
    const results = await this.search(
      queryVector,
      queryText,
      limit,
      minScore,
      project,
      rootPath,
      hybrid,
    );
    const emptyEmbeddings = results.map(() => [] as number[]);
    return [results, emptyEmbeddings];
  }
}
