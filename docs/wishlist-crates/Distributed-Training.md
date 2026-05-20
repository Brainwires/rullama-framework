# Distributed Training for `rullama-finetune` / `rullama-training`

> **Note (v0.11):** `brainwires-training` and `brainwires-finetune-local` moved
> to the sibling `rullama` workspace as `rullama-training` and `rullama-finetune`.
> This wishlist is preserved here for historical context; if you pick it up,
> implement it against the rullama crates and `brainwires-network::mesh`
> (which remains in this workspace) for the transport layer.

## Context

The `rullama-finetune` crate (formerly `brainwires-finetune-local` in this workspace) supports local training (single-machine Burn framework with LoRA/QLoRA/DoRA adapters). Cloud fine-tuning lives separately in this workspace's `brainwires-finetune`. There is no support for distributing local training across multiple machines or GPUs on different nodes.

The goal is to add a feature-gated `distributed` module that enables **data-parallel distributed training** across a mesh of nodes, leveraging the existing `brainwires-network::mesh` (topology, routing, discovery abstractions) and the rest of `brainwires-network` (transport layers) for communication.

## Decision: Coordinator Layer in `brainwires-training`, Not a New Crate

**Why not a new crate?** Distributed training is tightly coupled to training internals (gradient extraction, optimizer state, checkpoint format, dataset sharding). Separating it would require exposing a large internal API surface. It belongs in `brainwires-training` behind a feature flag.

**Why not self-contained?** `brainwires-network::mesh` already provides exactly the right abstractions (topology, routing, discovery) and the rest of `brainwires-network` provides the transport layer. Reimplementing networking would be wasteful.

**Architecture:** A coordinator layer that wraps the existing `BurnBackend`. Each node runs Burn locally; the coordinator orchestrates data sharding, gradient sync, and checkpointing via mesh message routing.

## Key Design Decisions

1. **Don't modify `TrainingBackend` trait** — it's sync and that's fine for single-node. The distributed coordinator works directly with Burn types, bypassing the trait.
2. **Extract step-level training helpers** from `burn_backend.rs` into shared functions so both `BurnBackend::train()` and the distributed worker can reuse them without duplication.
3. **Data parallelism first** — each node has the full model, trains on a dataset shard, syncs gradients. Model parallelism is a stub for future work.
4. **Messages use `MessageRouter::route_message()`** with serialized `TrainingMessage` enums as `&[u8]` payloads.

## File Structure

All new files under `src/distributed/`, gated by `#[cfg(feature = "distributed")]`:

```
src/distributed/
  mod.rs              — module root, re-exports
  config.rs           — DistributedTrainingConfig, NodeRole, GradientSyncStrategy, ParallelismMode
  messages.rs         — TrainingMessage enum, GradientPayload (coordinator<->worker protocol)
  coordinator.rs      — DistributedTrainingCoordinator (orchestrates training across mesh)
  worker.rs           — WorkerNode (runs local step-level training, sends/receives gradients)
  gradient_sync.rs    — GradientSynchronizer trait + AllReduceSynchronizer, RingAllReduceSynchronizer
  data_shard.rs       — compute_shards(), ShardedDataset wrapper
  fault_tolerance.rs  — FaultDetector (heartbeat-based), redistribute_shards()
  checkpoint.rs       — DistributedCheckpointManager (extends local CheckpointManager)
```

Modified files:
- `Cargo.toml` — add `distributed` feature, `brainwires-network` dep (with `mesh` feature)
- `src/lib.rs` — add `#[cfg(feature = "distributed")] pub mod distributed;` + re-exports
- `src/error.rs` — add distributed error variants (GradientSync, WorkerFailed, InsufficientWorkers, Mesh)
- `src/manager.rs` — add `#[cfg(feature = "distributed")] pub async fn train_distributed()`
- `src/local/burn_backend.rs` — extract step-level helpers into reusable functions (no behavior change)

## Key Types

### `DistributedTrainingConfig` (`distributed/config.rs`)
```rust
pub struct DistributedTrainingConfig {
    pub local_config: LocalTrainingConfig,  // shared across all nodes
    pub role: NodeRole,                     // Coordinator or Worker
    pub gradient_sync: GradientSyncStrategy,// AllReduce, RingAllReduce, AsyncSgd
    pub parallelism: ParallelismMode,       // DataParallel (default), ModelParallel (stub)
    pub world_size: usize,                  // expected worker count
    pub rank: usize,                        // this node's rank (0 = coordinator)
    pub sync_every_steps: u64,              // gradient sync frequency
    pub elastic: bool,                      // allow join/leave
    pub min_workers: usize,                 // minimum to continue
    pub sync_timeout_secs: u64,             // timeout before declaring failure
}
```

### `TrainingMessage` (`distributed/messages.rs`)
```rust
pub enum TrainingMessage {
    // Coordinator -> Workers
    AssignShard { worker_rank, start_idx, end_idx, epoch },
    BroadcastGradients { step, gradients: GradientPayload },
    SaveCheckpoint { step },
    TrainingComplete { metrics },
    Stop,

    // Workers -> Coordinator
    ReportGradients { worker_rank, step, gradients: GradientPayload },
    StepComplete { worker_rank, step, loss },
    CheckpointSaved { worker_rank, step },
    Heartbeat { worker_rank, step },
    WorkerFailed { worker_rank, error },

    // Ring AllReduce (peer-to-peer)
    RingPartial { from_rank, chunk_id, step, data: Vec<f32> },

    // Elastic
    JoinRequest { node_id, capabilities },
    JoinAccepted { assigned_rank, shard_start, shard_end },
    LeaveNotice { worker_rank },
}

pub struct GradientPayload {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,  // name -> (data, shape)
}
```

### `GradientSynchronizer` (`distributed/gradient_sync.rs`)
```rust
#[async_trait]
pub trait GradientSynchronizer: Send + Sync {
    async fn synchronize(
        &self,
        local_gradients: GradientPayload,
        step: u64,
    ) -> Result<GradientPayload, TrainingError>;
}
```
Implementations: `AllReduceSynchronizer` (star topology, coordinator aggregates), `RingAllReduceSynchronizer` (ring topology, bandwidth-optimal).

### `DistributedTrainingCoordinator` (`distributed/coordinator.rs`)
```rust
pub struct DistributedTrainingCoordinator {
    config: DistributedTrainingConfig,
    topology: Box<dyn MeshTopology>,
    router: Box<dyn MessageRouter>,
    discovery: Box<dyn PeerDiscovery>,
    workers: HashMap<usize, WorkerState>,
    checkpoint_manager: CheckpointManager,
}

impl DistributedTrainingCoordinator {
    pub async fn train(
        &mut self,
        progress_callback: Box<dyn Fn(TrainingProgress) + Send>,
    ) -> Result<TrainedModelArtifact, TrainingError>;
}
```

### `WorkerNode` (`distributed/worker.rs`)
```rust
pub struct WorkerNode {
    config: DistributedTrainingConfig,
    router: Box<dyn MessageRouter>,
    coordinator_id: Uuid,
}

impl WorkerNode {
    pub async fn run(&mut self) -> Result<(), TrainingError>;
}
```
Uses step-level Burn helpers directly (not `TrainingBackend` trait).

## Cargo.toml Changes

```toml
# New optional dependencies
brainwires-network = { workspace = true, optional = true, features = ["mesh"] }

[features]
distributed = ["local", "dep:brainwires-network", "dep:tokio"]
full = ["cloud", "local", "bedrock", "vertex", "distributed"]
```

## Error Variants to Add (`src/error.rs`)

```rust
#[error("Gradient sync error: {0}")]
GradientSync(String),

#[error("Worker {rank} failed: {reason}")]
WorkerFailed { rank: usize, reason: String },

#[error("Insufficient workers: have {available}, need {required}")]
InsufficientWorkers { available: usize, required: usize },

#[cfg(feature = "distributed")]
#[error("Mesh error: {0}")]
Mesh(#[from] brainwires_mesh::MeshError),
```

## Integration with Existing Code

### Burn Backend Refactor (non-breaking)
Extract from `burn_backend.rs` into shared helpers:
- `train_single_step()` — forward pass, loss, backward, return gradients
- `apply_gradients()` — apply gradient payload to model parameters
- `serialize_gradients()` / `deserialize_gradients()` — convert Burn `GradientsParams` to/from `GradientPayload`

The existing `BurnBackend::train()` continues to call these internally with no behavior change. The distributed `WorkerNode` also calls them in its step loop.

### TrainingManager Addition
```rust
#[cfg(feature = "distributed")]
pub async fn train_distributed(
    &self,
    config: DistributedTrainingConfig,
    topology: Box<dyn MeshTopology>,
    router: Box<dyn MessageRouter>,
    discovery: Box<dyn PeerDiscovery>,
    callback: Box<dyn Fn(TrainingProgress) + Send>,
) -> Result<TrainedModelArtifact, TrainingError>
```

## Implementation Phases

### Phase 1: Foundation
1. Add feature flag to `Cargo.toml`
2. Create `distributed/mod.rs`, `config.rs`, `messages.rs`
3. Add error variants to `error.rs`
4. Add conditional module + re-exports to `lib.rs`

### Phase 2: Data Sharding
5. Create `distributed/data_shard.rs` — `compute_shards()`, `ShardedDataset`

### Phase 3: Burn Step-Level Helpers
6. Extract `train_single_step()`, `apply_gradients()`, `serialize_gradients()` from `burn_backend.rs`

### Phase 4: Gradient Sync
7. Create `distributed/gradient_sync.rs` — `GradientSynchronizer` trait
8. Implement `AllReduceSynchronizer` (star topology via `MessageRouter`)
9. Implement `RingAllReduceSynchronizer` (ring topology)

### Phase 5: Worker
10. Create `distributed/worker.rs` — `WorkerNode` with step-level training loop

### Phase 6: Coordinator
11. Create `distributed/coordinator.rs` — discovery, shard assignment, gradient aggregation, progress

### Phase 7: Fault Tolerance & Checkpointing
12. Create `distributed/fault_tolerance.rs` — `FaultDetector`, `redistribute_shards()`
13. Create `distributed/checkpoint.rs` — `DistributedCheckpointManager`

### Phase 8: Integration
14. Add `train_distributed()` to `TrainingManager`
15. Final re-exports in `lib.rs`

## Known Challenges

1. **Gradient serialization from Burn**: Must convert `GradientsParams` to `Vec<f32>`. The `checkpointing.rs` `save_weights` pattern (HashMap<String, (Vec<f32>, Vec<usize>)>) provides precedent.
2. **Bandwidth**: Full gradient payloads can be large. LoRA adapters are small by design so this is acceptable initially. Gradient compression (top-k, fp16) is a future optimization.
3. **Model parallelism**: Complex pipeline parallelism requires deep Burn integration. Implement as a stub initially; full `DataParallel` first.
4. **Sync/async bridge**: Workers run Burn (sync) in `tokio::task::spawn_blocking`, then async mesh communication for gradient exchange.

## Verification

1. **Unit tests**: Shard computation, gradient averaging, message serialization roundtrips
2. **Integration tests**: Mock `MessageRouter` + `MeshTopology`, run coordinator + 2 workers in-process
3. **Build check**: `cargo build --features distributed` compiles cleanly
4. **Existing tests pass**: `cargo test` (default features) unchanged
5. **Feature isolation**: `cargo build` (default = cloud only) must not pull in mesh/relay deps
