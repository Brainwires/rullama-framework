/**
 * @module @rullama/seal
 *
 * SEAL (Self-Evolving Agentic Learning) for the rullama. Implements
 * techniques from the SEAL paper to enhance conversational question answering,
 * semantic parsing, and self-evolving agent capabilities. `SealProcessor` runs
 * a query through coreference resolution, query-core extraction, pattern
 * learning and structural reflection in-process; the individual stages are
 * exported for standalone use.
 *
 * Components:
 * - Coreference Resolution — pronouns / definite NPs → concrete entities
 * - Query Core Extraction — natural language → structured S-expressions
 * - Learning Coordinator — local + global pattern memory
 * - Reflection — post-execution error detection / correction
 * - FeedbackBridge — AuditLogger user feedback → SEAL learning
 * - SealKnowledgeCoordinator — BKS/PKS wiring (concrete BKS/PKS work lives
 *   Rust-side; the Deno `@rullama/knowledge` package is interface-only)
 *
 * Equivalent to Rust's `rullama_agents::seal` module.
 */

import { CoreferenceResolver, type DialogState } from "./coreference.ts";
import { LearningCoordinator } from "./learning.ts";
import {
  type QueryCore,
  QueryCoreExtractor,
  type QueryResult,
} from "./query_core.ts";
import {
  defaultReflectionConfig,
  ReflectionModule,
  type ReflectionReport,
} from "./reflection.ts";
import type {
  EntityStoreT,
  RelationshipGraphT,
  SealProcessingResult,
} from "./types.ts";

// ─── Re-exports ─────────────────────────────────────────────────────────────

export {
  compatibleTypes,
  CoreferenceResolver,
  DialogState,
  InMemoryEntityStore,
  type ReferenceType,
  type ResolvedReference,
  type SalienceScore,
  salienceTotal,
  type UnresolvedReference,
} from "./coreference.ts";

export {
  ConfidenceStats,
  type CoreferenceRecord,
  GlobalMemory,
  isHighConfidence,
  isLowConfidence,
  LearningCoordinator,
  type LearningStats,
  LocalMemory,
  type PatternHint,
  QueryPattern,
  type QueryRecord,
  type ResolutionPattern,
  type ResponseConfidence,
  ToolErrorPattern,
  ToolStats,
  TrackedEntity,
} from "./learning.ts";

export {
  asVariable,
  type CompareOp,
  type FilterPredicate,
  isVariable,
  newQueryCore,
  queryConstant,
  type QueryCore,
  QueryCoreExtractor,
  queryCoreToSexp,
  queryCount,
  QueryExecutor,
  type QueryExpr,
  queryJoin,
  type QueryOp,
  type QueryResult,
  queryResultEmpty,
  queryResultError,
  type QueryResultValue,
  queryResultWithValues,
  queryVar,
  type QuestionType,
  relationInverse,
  relationName,
  relationToEdgeType,
  type RelationType,
  type SuperlativeDir,
} from "./query_core.ts";

export {
  type CorrectionRecord,
  defaultReflectionConfig,
  type ErrorType,
  errorTypeDescription,
  Issue,
  type ReflectionConfig,
  ReflectionModule,
  ReflectionReport,
  type Severity,
  severityAtLeast,
  severityCompare,
  type SuggestedFix,
  suggestedFixDescription,
} from "./reflection.ts";

export {
  FeedbackBridge,
  type FeedbackProcessingStats,
} from "./feedback_bridge.ts";

export {
  type BehavioralKnowledgeCache,
  type BehavioralTruth,
  DEFAULT_PATTERN_PROMOTION_THRESHOLD,
  defaultIntegrationConfig,
  type EntityResolutionStrategy,
  type IntegrationConfig,
  integrationConfigDisabled,
  integrationConfigSealToKnowledgeOnly,
  newBehavioralTruth,
  type PersonalFact,
  type PersonalKnowledgeCache,
  type ScoredTruth,
  SealKnowledgeCoordinator,
  type TruthCategory,
  type TruthSource,
  validateIntegrationConfig,
} from "./knowledge_integration.ts";

export {
  type EdgeType,
  type EntityStoreT,
  type EntityType,
  newSealProcessingResult,
  type RelationshipGraphT,
  type SealProcessingResult,
  type ToolErrorCategory,
  type ToolOutcome,
} from "./types.ts";

// ─── SealConfig / SealProcessor ─────────────────────────────────────────────

/** Configuration for the SEAL processor. */
export interface SealConfig {
  /** Run coreference detection/resolution (step 1 of `process`). */
  enable_coreference: boolean;
  /** Extract a `QueryCore` from the resolved query (step 2). */
  enable_query_cores: boolean;
  /** Consult and update the learning coordinator (step 3) and honour `recordOutcome`. */
  enable_learning: boolean;
  /** Validate the extracted core structurally and score quality (step 4). */
  enable_reflection: boolean;
  /** Reflection retry budget; stored for parity with Rust, not consulted by `process`. */
  max_reflection_retries: number;
  /** Resolutions with confidence below this are discarded and not rewritten into the query. */
  min_coreference_confidence: number;
  /** Reliability bar for pattern use; stored for parity with Rust, not consulted by `process`. */
  min_pattern_reliability: number;
}

/**
 * Default SEAL config — all stages enabled, `max_reflection_retries` 2,
 * `min_coreference_confidence` 0.5, `min_pattern_reliability` 0.7.
 */
export function defaultSealConfig(): SealConfig {
  return {
    enable_coreference: true,
    enable_query_cores: true,
    enable_learning: true,
    enable_reflection: true,
    max_reflection_retries: 2,
    min_coreference_confidence: 0.5,
    min_pattern_reliability: 0.7,
  };
}

/** Main SEAL processor that orchestrates all components. */
export class SealProcessor {
  /** The configuration the processor was built with. */
  readonly config: SealConfig;
  private coreferenceResolver: CoreferenceResolver;
  private queryExtractor: QueryCoreExtractor;
  /**
   * Learning state; replaced by {@link initConversation}. Public so tests
   * (mirroring Rust) can inspect `processor.learning_coordinator.local.conversation_id`.
   */
  learning_coordinator: LearningCoordinator;
  private reflection_module: ReflectionModule;

  /**
   * Build a processor with fresh stage instances; the learning coordinator
   * starts with an empty conversation id until {@link initConversation}.
   */
  constructor(config: SealConfig) {
    this.config = config;
    this.coreferenceResolver = new CoreferenceResolver();
    this.queryExtractor = new QueryCoreExtractor();
    this.learning_coordinator = new LearningCoordinator("");
    this.reflection_module = new ReflectionModule(defaultReflectionConfig());
  }

  /** Create a processor with default config. */
  static withDefaults(): SealProcessor {
    return new SealProcessor(defaultSealConfig());
  }

  /** Initialise the learning coordinator for a new conversation. */
  initConversation(conversation_id: string): void {
    this.learning_coordinator = new LearningCoordinator(conversation_id);
  }

  /** Process a user query through the SEAL pipeline. */
  process(
    query: string,
    dialog_state: DialogState,
    entity_store: EntityStoreT,
    graph?: RelationshipGraphT,
  ): SealProcessingResult {
    const result: SealProcessingResult = {
      original_query: query,
      resolved_query: query,
      query_core: undefined,
      matched_pattern: undefined,
      resolutions: [],
      quality_score: 1.0,
      issues: [],
    };

    // Step 1 — Coreference Resolution.
    if (this.config.enable_coreference) {
      const references = this.coreferenceResolver.detectReferences(query);
      if (references.length > 0) {
        const resolutions = this.coreferenceResolver.resolve(
          references,
          dialog_state,
          entity_store,
          graph,
        );
        const confident = resolutions.filter((r) =>
          r.confidence >= this.config.min_coreference_confidence
        );
        if (confident.length > 0) {
          result.resolved_query = this.coreferenceResolver
            .rewriteWithResolutions(query, confident);
          result.resolutions = confident;
        }
      }
    }

    // Step 2 — Query Core Extraction.
    if (this.config.enable_query_cores) {
      const entities = entity_store.topEntityInfo(50);
      const core = this.queryExtractor.extract(result.resolved_query, entities);
      if (core !== undefined) {
        if (result.resolved_query !== query) {
          core.resolved = result.resolved_query;
        }
        result.query_core = core;
      }
    }

    // Step 3 — Learning Coordinator.
    if (this.config.enable_learning) {
      const pattern = this.learning_coordinator.processQuery(
        query,
        result.resolved_query,
        result.query_core,
        dialog_state.current_turn,
      );
      if (pattern !== undefined) {
        result.matched_pattern = pattern.id;
      }
    }

    // Step 4 — Reflection structural validation.
    if (this.config.enable_reflection && result.query_core !== undefined) {
      result.issues = this.reflection_module.validateQueryCore(
        result.query_core,
      );
      result.quality_score = result.issues.length === 0
        ? 1.0
        : 0.8 - Math.min(result.issues.length * 0.1, 0.5);
    }

    return result;
  }

  /** Record the outcome of a query execution for learning. */
  recordOutcome(
    pattern_id: string | undefined,
    success: boolean,
    result_count: number,
    query_core: QueryCore | undefined,
    execution_time_ms: number,
  ): void {
    if (this.config.enable_learning) {
      this.learning_coordinator.recordOutcome(
        pattern_id,
        success,
        result_count,
        query_core,
        execution_time_ms,
      );
    }
  }

  /** Analyze execution result with the reflection module. */
  reflect(
    query_core: QueryCore,
    result: QueryResult,
    graph: RelationshipGraphT,
  ): ReflectionReport {
    return this.reflection_module.analyze(query_core, result, graph);
  }

  /** Learning-context block for prompt injection. */
  getLearningContext(): string {
    return this.learning_coordinator.getContextForPrompt();
  }

  /** The coreference resolver used by step 1. */
  coreference(): CoreferenceResolver {
    return this.coreferenceResolver;
  }
  /** The query-core extractor used by step 2. */
  queryExtractorAccess(): QueryCoreExtractor {
    return this.queryExtractor;
  }
  /** The current learning coordinator (same object as `learning_coordinator`). */
  learningMut(): LearningCoordinator {
    return this.learning_coordinator;
  }
  /** The reflection module used by step 4 and {@link reflect}. */
  reflection(): ReflectionModule {
    return this.reflection_module;
  }
}
