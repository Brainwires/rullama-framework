/**
 * SEAL (Self-Evolving Agentic Learning) Integration Module.
 *
 * Implements techniques from the SEAL paper to enhance conversational question
 * answering, semantic parsing, and self-evolving agent capabilities.
 *
 * Components:
 * - Coreference Resolution — pronouns / definite NPs → concrete entities
 * - Query Core Extraction — natural language → structured S-expressions
 * - Learning Coordinator — local + global pattern memory
 * - Reflection — post-execution error detection / correction
 * - FeedbackBridge — AuditLogger user feedback → SEAL learning
 * - SealKnowledgeCoordinator — BKS/PKS wiring (concrete BKS/PKS work lives
 *   Rust-side; the Deno `@brainwires/knowledge` package is interface-only)
 *
 * Equivalent to Rust's `brainwires_agents::seal` module.
 */

import type { EntityStoreT, RelationshipGraphT } from "@brainwires/core";
import {
  CoreferenceResolver,
  type DialogState,
  type ResolvedReference,
} from "./coreference.ts";
import { LearningCoordinator } from "./learning.ts";
import {
  type QueryCore,
  QueryCoreExtractor,
  type QueryResult,
} from "./query_core.ts";
import {
  defaultReflectionConfig,
  type Issue,
  ReflectionModule,
  type ReflectionReport,
} from "./reflection.ts";
import {
  newSealProcessingResult,
  type SealProcessingResult,
} from "./types.ts";

// ─── Re-exports ─────────────────────────────────────────────────────────────

export {
  CoreferenceResolver,
  DialogState,
  InMemoryEntityStore,
  type ReferenceType,
  type ResolvedReference,
  type SalienceScore,
  salienceTotal,
  type UnresolvedReference,
  compatibleTypes,
} from "./coreference.ts";

export {
  type CoreferenceRecord,
  ConfidenceStats,
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
  type QueryCore,
  QueryCoreExtractor,
  queryCoreToSexp,
  queryConstant,
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
  type QuestionType,
  type RelationType,
  queryVar,
  relationInverse,
  relationName,
  relationToEdgeType,
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
  severityAtLeast,
  severityCompare,
  type Severity,
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
  newSealProcessingResult,
  type SealProcessingResult,
};

// ─── SealConfig / SealProcessor ─────────────────────────────────────────────

/** Configuration for the SEAL processor. */
export interface SealConfig {
  enable_coreference: boolean;
  enable_query_cores: boolean;
  enable_learning: boolean;
  enable_reflection: boolean;
  max_reflection_retries: number;
  min_coreference_confidence: number;
  min_pattern_reliability: number;
}

/** Default SEAL config — all stages enabled. */
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
  readonly config: SealConfig;
  private coreferenceResolver: CoreferenceResolver;
  private queryExtractor: QueryCoreExtractor;
  // Public so tests (mirroring Rust) can inspect `processor.learning_coordinator.local.conversation_id`.
  learning_coordinator: LearningCoordinator;
  private reflection_module: ReflectionModule;

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
      result.issues = this.reflection_module.validateQueryCore(result.query_core);
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

  coreference(): CoreferenceResolver {
    return this.coreferenceResolver;
  }
  queryExtractorAccess(): QueryCoreExtractor {
    return this.queryExtractor;
  }
  learningMut(): LearningCoordinator {
    return this.learning_coordinator;
  }
  reflection(): ReflectionModule {
    return this.reflection_module;
  }
}
