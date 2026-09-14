/**
 * Hyperparameters, LoRA, and alignment configuration.
 *
 * Equivalent to Rust's `rullama_training::config` module.
 */

/** Learning-rate schedule shape applied over the course of training. */
export type LrScheduler =
  | "constant"
  | "linear"
  | "cosine"
  | "cosine_warm_restarts";

/**
 * Provider-neutral training hyperparameters.
 *
 * The cloud providers forward only the subset their API accepts (all three
 * send `epochs`, `batch_size`, `learning_rate`); the rest are carried for
 * parity with the Rust local-training path.
 */
export interface TrainingHyperparams {
  /** Number of passes over the training set. */
  epochs: number;
  /** Examples per optimizer step. */
  batch_size: number;
  /** Absolute peak learning rate (OpenAI converts it to a multiplier over its `2e-5` base). */
  learning_rate: number;
  /** Steps of linear LR warm-up before the scheduler takes over. */
  warmup_steps: number;
  /** L2 weight-decay coefficient. */
  weight_decay: number;
  /** Learning-rate schedule. */
  lr_scheduler: LrScheduler;
  /** RNG seed for data shuffling / initialization. */
  seed: number;
  /** Maximum sequence length in tokens; longer examples are truncated. */
  max_seq_len: number;
  /** Micro-batches accumulated before each optimizer step. */
  gradient_accumulation_steps: number;
  /** Gradient-clipping threshold (global norm). */
  max_grad_norm: number;
}

/**
 * Default hyperparameters: 3 epochs, batch 4, LR `2e-5`, 100 warm-up steps,
 * weight decay 0.01, cosine schedule, seed 42, 2048 max tokens, 4×
 * gradient accumulation, grad-norm clip 1.0.
 */
export function defaultHyperparams(): TrainingHyperparams {
  return {
    epochs: 3,
    batch_size: 4,
    learning_rate: 2e-5,
    warmup_steps: 100,
    weight_decay: 0.01,
    lr_scheduler: "cosine",
    seed: 42,
    max_seq_len: 2048,
    gradient_accumulation_steps: 4,
    max_grad_norm: 1.0,
  };
}

/** Adapter method for parameter-efficient fine-tuning. */
export type AdapterMethod =
  | { kind: "lora" }
  | { kind: "qlora"; bits: number }
  | { kind: "dora" }
  | { kind: "qdora"; bits: number };

/** `true` for the quantized variants (`qlora`, `qdora`). */
export function isQuantized(m: AdapterMethod): boolean {
  return m.kind === "qlora" || m.kind === "qdora";
}

/** Base-weight quantization width for `qlora`/`qdora`, or `null` for the unquantized methods. */
export function quantizationBits(m: AdapterMethod): number | null {
  return m.kind === "qlora" || m.kind === "qdora" ? m.bits : null;
}

/** LoRA adapter configuration. */
export interface LoraConfig {
  /** Rank of the low-rank update matrices (`r`). */
  rank: number;
  /** LoRA scaling factor; the update is scaled by `alpha / rank`. */
  alpha: number;
  /** Dropout probability applied to the adapter input. */
  dropout: number;
  /** Names of the weight matrices to adapt (e.g. `"q_proj"`). */
  target_modules: string[];
  /** Which adapter family / quantization to use. */
  method: AdapterMethod;
}

/** Default LoRA: rank 16, alpha 32, dropout 0.05, on `q/k/v/o_proj`, plain (unquantized) `lora`. */
export function defaultLoraConfig(): LoraConfig {
  return {
    rank: 16,
    alpha: 32,
    dropout: 0.05,
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"],
    method: { kind: "lora" },
  };
}

/** Alignment training method. */
export type AlignmentMethod =
  | { kind: "none" }
  | { kind: "dpo"; beta: number }
  | { kind: "orpo"; lambda: number };

/** No alignment stage — plain supervised fine-tuning. */
export function defaultAlignment(): AlignmentMethod {
  return { kind: "none" };
}

/**
 * Direct Preference Optimization.
 *
 * @param beta KL-penalty strength against the reference model (default `0.1`).
 */
export function dpoAlignment(beta: number = 0.1): AlignmentMethod {
  return { kind: "dpo", beta };
}

/**
 * Odds-Ratio Preference Optimization. None of the shipped cloud providers
 * accept it — `createJob` throws a `validation` error.
 *
 * @param lambda Weight of the odds-ratio loss term (default `0.5`).
 */
export function orpoAlignment(lambda: number = 0.5): AlignmentMethod {
  return { kind: "orpo", lambda };
}
