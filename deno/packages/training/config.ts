/**
 * Hyperparameters, LoRA, and alignment configuration.
 *
 * Equivalent to Rust's `brainwires_training::config` module.
 */

export type LrScheduler =
  | "constant"
  | "linear"
  | "cosine"
  | "cosine_warm_restarts";

export interface TrainingHyperparams {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  warmup_steps: number;
  weight_decay: number;
  lr_scheduler: LrScheduler;
  seed: number;
  max_seq_len: number;
  gradient_accumulation_steps: number;
  max_grad_norm: number;
}

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

export function isQuantized(m: AdapterMethod): boolean {
  return m.kind === "qlora" || m.kind === "qdora";
}

export function quantizationBits(m: AdapterMethod): number | null {
  return m.kind === "qlora" || m.kind === "qdora" ? m.bits : null;
}

/** LoRA adapter configuration. */
export interface LoraConfig {
  rank: number;
  alpha: number;
  dropout: number;
  target_modules: string[];
  method: AdapterMethod;
}

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

export function defaultAlignment(): AlignmentMethod {
  return { kind: "none" };
}

export function dpoAlignment(beta: number = 0.1): AlignmentMethod {
  return { kind: "dpo", beta };
}

export function orpoAlignment(lambda: number = 0.5): AlignmentMethod {
  return { kind: "orpo", lambda };
}
