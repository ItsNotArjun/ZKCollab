# Code Structure (src/)

For each Rust file under src/, this document summarizes:
- **Input**: Main data/types it consumes.
- **Output**: Main data/types it produces.
- **Role**: What it does in the system.

This is intentionally high-level; see the source for full details.

---

## Top-Level Crate Files

### src/lib.rs
- **Input**: `config.yaml` (or CLI-provided config path), environment variables, filesystem for layer-setup directory.
- **Output**: Global `CONFIG` (parsed YAML config), `LAYER_SETUP_DIR` path, module exports (`basic_block`, `graph`, `layer`, `onnx`, `ptau`, `util`, `training*`).
- **Role**: Crate root. Loads configuration once, exposes all core modules, and defines the per-config layer-setup cache directory.

### src/main.rs
- **Input**: `CONFIG` (model path, input path, ptau path, prover/verifier paths), ONNX model file, optional JSON inputs, SRS/ptau file.
- **Output**: Generated proving keys and models on disk, encoded inputs/outputs, a Plonk/KZG proof file, and a verification run (plus timing and file-size logs).
- **Role**: CLI entrypoint for ONNX inference proofs. Loads ONNX → Graph, runs witness generation, performs setup, generates and verifies a proof, and measures artifact sizes.

### src/graph.rs
- **Input**: A list of `BasicBlock`s, model tensors (per block), input tensors, and an SRS; later, encoded `Data`/`DataEnc` plus RNG.
- **Output**: Computed node outputs (forward run), encoded outputs (`Data`), per-block setups (commitments & polynomials), per-node proofs, and verification pairing checks.
- **Role**: Core graph/circuit orchestration. Represents a computation as a DAG of basic blocks and wires, runs it, and wraps setup/prove/verify for each node, with support for skipping “precomputable” blocks.

### src/onnx.rs
- **Input**: ONNX model graph (via `tract-onnx`), ONNX attributes and initializers, `CONFIG` scale-factor settings.
- **Output**: A `Graph` built from per-op layer graphs, model tensors (constants/weights) in field representation, and type/shape metadata.
- **Role**: ONNX front-end. Parses an ONNX file, quantizes/encodes constants, maps each ONNX op to an internal `Layer`, and stitches local graphs into a global `Graph` compatible with proving.

### src/basic_block.rs
- **Input**: Field tensors and commitment SRS; for each block, model parameters and input tensors.
- **Output**: `Data` and `DataEnc` commitment objects, `BasicBlock` trait API, and concrete block types (re-exported) plus SRS and caching types.
- **Role**: Defines the `BasicBlock` abstraction and shared data types for vector commitments and Plonk blocks, and re-exports all primitive gadgets (add/mul/matmul, copy-constraints, CQ/CQ2, etc.).

### src/layer.rs
- **Input**: Per-op shapes and data types, constants and ONNX attributes.
- **Output**: For each ONNX op, a `(Graph, output_shapes, output_types)` pair implementing that op as a graph of basic blocks.
- **Role**: High-level ONNX layer interface. Each `Layer::graph` builds a local proof-capable graph for one ONNX operation (Conv, MatMul, Range, Tile, etc.).

### src/ptau.rs
- **Input**: Path to a ptau/SRS file, desired size parameters from `CONFIG.ptau`.
- **Output**: An `SRS` instance populated with G1/G2 powers and related data, suitable for all basic blocks.
- **Role**: Trusted-setup loader. Reads ptau files and translates them into the SRS structure used throughout proving and verifying.

### src/tests.rs
- **Input**: Test harness (cargo test), internal modules.
- **Output**: Unit/integration tests for core components (depends on what is implemented inside; typically sanity checks for gadgets and graph behavior).
- **Role**: Central place for crate-level tests (beyond per-file `#[cfg(test)]` modules).

### src/training_constraints.rs
- **Input**: Field tensors for x, W, upstream gradients, SGD parameters and outputs; transcript bytes; scale configuration.
- **Output**: Constraint checks (success or descriptive error) for linear VJP, ReLU VJP, SGD update, and randomness binding challenge.
- **Role**: Low-level training gadgets. Implements the math for one-layer backprop (`enforce_linear_vjp`, `enforce_relu_vjp`), fixed-point SGD update (`enforce_sgd_update`), and Fiat–Shamir randomness binding (`derive_challenge`, `enforce_randomness_binding`).

### src/training.rs
- **Input**: SRS, three `Graph`s (forward, backward, optimizer), corresponding models/inputs; and high-level IR objects (`TrainingStepIR`).
- **Output**: `TrainingProofBundle` (Plonk proofs for forward/backward/optimizer plus a joint randomness commitment) and `TrainingStateCommitment` for IR-level checks.
- **Role**: Training-step machinery. On the graph side, runs and proves the three training subgraphs, binding them with a hash. On the IR side, defines a semantic single-step training relation (forward + backward + SGD update + randomness binding) and checks it.

### src/training_compile.rs
- **Input**: A `TrainingSpec` (sequence of `ForwardOp`s: Linear/ReLU), scalar parameters and inputs, SRS in tests.
- **Output**: A `Graph` plus model tensors implementing a simplified linear+ReLU training step circuit, and helpers to fabricate example IO.
- **Role**: Compiler from a simple linear+ReLU architecture into a training `Graph`. Wires together `LinearForwardBlock`, `LinearBackwardBlock`, `ReLUBackwardBlock`, `SGDUpdateBlock`, and `ChallengeBlock` to realize a full training-step proof, with tests that run setup/prove/verify.

### src/util.rs
- **Input**: Re-exports from util submodules.
- **Output**: Public `util::*` API used across the crate (config, FFT, MSM, ONNX helpers, prover/verifier helpers, random, serialization, shape utils, etc.).
- **Role**: Facade for all utility modules.

---

## Basic Blocks (src/basic_block)

Each basic-block file defines one or more `BasicBlock` implementations used inside `Graph`. Inputs are typically model tensors and input tensors; outputs are field tensors and their commitment encoding.

### src/basic_block/add.rs
- **Input**: Two or more tensors to be added.
- **Output**: Elementwise sum tensor.
- **Role**: Addition gadget; enforces `z = x + y` (possibly batched) at the commitment level.

### src/basic_block/bool_check.rs
- **Input**: Tensors expected to be boolean (0/1) or satisfying simple boolean constraints.
- **Output**: Same tensors, with constraints applied.
- **Role**: Booleanity checker; enforces that certain wires carry bits.

### src/basic_block/clip.rs
- **Input**: Tensor and clip bounds.
- **Output**: Clipped tensor within the allowed range.
- **Role**: Range-clipping gadget used by Clip layers and quantization logic.

### src/basic_block/concat.rs
- **Input**: Multiple tensors along a concatenation axis.
- **Output**: A single concatenated tensor.
- **Role**: Structural gadget to join tensors; used by ONNX Concat.

### src/basic_block/constant.rs
- **Input**: None at runtime (constants are baked into the model tensor).
- **Output**: Constant tensors.
- **Role**: Constant/basic constant-of-shape gadgets (`ConstBasicBlock`, `Const2BasicBlock`, `ConstOfShapeBasicBlock`).

### src/basic_block/copy_constraint.rs
- **Input**: Indices and values representing copy relations across wires, plus padding policy.
- **Output**: Copy-permutation partitions and permutation commitments.
- **Role**: Copy/permutation-argument gadget. Encodes equality-of-wires constraints via Plonk copy permutations (including padding handling).

### src/basic_block/cq.rs and src/basic_block/cq2.rs
- **Input**: CQ parameters and input tensors.
- **Output**: CQ/CQ2 tables and commitments.
- **Role**: Commitment-quotient gadgets used for efficient range/lookup-style constraints and CQ-friendly layouts.

### src/basic_block/cqlin.rs
- **Input**: Inputs for linear CQ-style constraints.
- **Output**: CQ-linear constraint outputs.
- **Role**: Linearized variant of CQ constraints (CQLin), used to reduce proving cost for certain gadgets.

### src/basic_block/div.rs
- **Input**: Numerators, denominators, or modulus constants.
- **Output**: Quotients/remainders or scaled divisions.
- **Role**: Division/modulo gadgets (`DivConstBasicBlock`, `DivScalarBasicBlock`, `ModConstBasicBlock`).

### src/basic_block/eq.rs
- **Input**: Two (or more) tensors to compare.
- **Output**: Equality indicators or constrained outputs.
- **Role**: Equality gadgets; enforce elementwise equality or provide equality proofs.

### src/basic_block/id.rs
- **Input**: Any tensor.
- **Output**: Same tensor.
- **Role**: Identity/basic routing gadget; often used where an explicit node is needed but no computation is required.

### src/basic_block/less.rs
- **Input**: Pairs of tensors.
- **Output**: Boolean tensor for `x < y` comparisons.
- **Role**: Comparison gadget to implement ONNX Less and related logic.

### src/basic_block/matmul.rs
- **Input**: Two rank-2 (or compatible) tensors.
- **Output**: Matrix product tensor.
- **Role**: Core MatMul gadget used in many layers (Gemm, Linear, Conv im2col-style computations).

### src/basic_block/max.rs
- **Input**: One or more tensors.
- **Output**: Elementwise max, and optionally proof-related outputs.
- **Role**: Max gadget and associated proof helper (`MaxBasicBlock`, `MaxProofBasicBlock`).

### src/basic_block/mul.rs
- **Input**: Two (or more) tensors or tensor + scalar/constant.
- **Output**: Elementwise product, or scalar-multiplied tensor.
- **Role**: Multiplication gadget family (`MulBasicBlock`, `MulConstBasicBlock`, `MulScalarBasicBlock`).

### src/basic_block/one_to_one.rs
- **Input**: Single tensor.
- **Output**: Single tensor.
- **Role**: Generic elementwise gadget for simple unary operations (passthrough with constraints).

### src/basic_block/ops.rs
- **Input**: Various tensors, depending on op.
- **Output**: Tensors for specialized ops.
- **Role**: Shared helper ops implemented as basic blocks (small composite gadgets).

### src/basic_block/ordered.rs
- **Input**: Tensor expected to be sorted/ordered.
- **Output**: Same tensor, with ordering constraints.
- **Role**: Ordering gadget, e.g. for top-k or sorting proofs.

### src/basic_block/permute.rs
- **Input**: Tensor and permutation indices.
- **Output**: Permuted tensor.
- **Role**: Index-permutation gadget used by many structural layers.

### src/basic_block/range.rs
- **Input**: Scalar or tensor values.
- **Output**: Same values with range constraints applied.
- **Role**: Range-constraint gadget (often backed by CQ/CQ2 tables).

### src/basic_block/repeater.rs
- **Input**: Tensor and repeat count.
- **Output**: Repeated tensor along some dimension.
- **Role**: Structural repetition gadget (tile-like at basic-block level).

### src/basic_block/reshape.rs
- **Input**: Tensor and target shape.
- **Output**: Reshaped tensor.
- **Role**: Shape-only gadget; reinterprets data layout without changing values.

### src/basic_block/rope.rs
- **Input**: Tensor and positional parameters.
- **Output**: Tensor with rotary positional embeddings applied.
- **Role**: RoPE gadget used for transformer-style models.

### src/basic_block/sort.rs
- **Input**: Unsorted tensor.
- **Output**: Sorted tensor and/or permutation info.
- **Role**: Sorting gadget, often paired with `OrderedBasicBlock` for correctness.

### src/basic_block/split.rs
- **Input**: Single tensor.
- **Output**: Multiple tensors (split along some axis).
- **Role**: Structural splitter gadget.

### src/basic_block/sub.rs
- **Input**: Two tensors.
- **Output**: Elementwise difference.
- **Role**: Subtraction gadget.

### src/basic_block/sum.rs
- **Input**: Tensor (or list of tensors).
- **Output**: Reduced sum.
- **Role**: Summation/reduction gadget.

### src/basic_block/transpose.rs
- **Input**: Tensor and permutation of axes.
- **Output**: Transposed tensor.
- **Role**: Axis-reordering gadget.

### src/basic_block/training.rs
- **Input**: Training-related tensors (inputs, weights, upstream grads, learning rate), and SRS.
- **Output**: Specialized training basic blocks: `LinearForwardBlock`, `LinearBackwardBlock`, `ReLUBackwardBlock`, `SGDUpdateBlock`, `ChallengeBlock`.
- **Role**: Provides basic blocks tailored to training: a matmul-based forward, scalar VJP blocks, SGD updates, and a challenge block for binding activations/gradients/updates.

---

## Layers (src/layer)

Each file defines a `Layer` implementation for a specific ONNX op, mapping ONNX shapes/types/constants to a local `Graph` built from basic blocks.

- **src/layer/and.rs** – Input: boolean tensors; Output: elementwise AND; Role: ONNX And.
- **src/layer/arithmetic.rs** – Input: numeric tensors; Output: Add/Sub results; Role: arithmetic layers shared by Add/Sub.
- **src/layer/cast.rs** – Input: tensor + target type; Output: casted tensor; Role: ONNX Cast/Identity.
- **src/layer/clip.rs** – Input: tensor + clip bounds; Output: clipped tensor; Role: ONNX Clip.
- **src/layer/concat.rs** – Input: list of tensors; Output: concatenated tensor; Role: ONNX Concat.
- **src/layer/constantofshape.rs** – Input: shape tensor + value; Output: filled tensor; Role: ConstantOfShape.
- **src/layer/conv.rs** – Input: activations, weights, (optionally biases); Output: convolution result; Role: Conv/ConvTranspose via splat/im2col + matmul.
- **src/layer/div.rs** – Input: tensors; Output: elementwise division; Role: ONNX Div.
- **src/layer/einsum.rs** – Input: tensors + einsum equation; Output: contracted tensor; Role: Einsum.
- **src/layer/equal.rs** – Input: tensors; Output: equality mask; Role: ONNX Equal.
- **src/layer/expand.rs** – Input: tensor + target shape; Output: broadcasted tensor; Role: ONNX Expand.
- **src/layer/flatten.rs** – Input: tensor; Output: flattened tensor; Role: Flatten.
- **src/layer/gather.rs** – Input: data + indices; Output: gathered tensor; Role: Gather.
- **src/layer/gathernd.rs** – Input: data + ND indices; Output: gathered tensor; Role: GatherND.
- **src/layer/gemm.rs** – Input: matrices + bias; Output: affine matmul; Role: Gemm.
- **src/layer/less.rs** – Input: tensors; Output: `<` mask; Role: Less.
- **src/layer/lstm.rs** – Input: sequence + weights; Output: LSTM outputs; Role: LSTM.
- **src/layer/matmul.rs** – Input: matrices; Output: matrix product; Role: MatMul.
- **src/layer/max.rs** – Input: tensors; Output: elementwise max/min; Role: Max/Min.
- **src/layer/mul.rs** – Input: tensors; Output: elementwise product; Role: Mul.
- **src/layer/neg.rs** – Input: tensor; Output: negated tensor; Role: Neg.
- **src/layer/nonlinear.rs** – Input: tensor; Output: activations (e.g. Tanh, Sigmoid, Erf); Role: non-linear activations.
- **src/layer/norm.rs** – Input: tensor + norm params; Output: normalized tensor; Role: InstanceNorm/BatchNorm.
- **src/layer/not.rs** – Input: boolean tensor; Output: NOT; Role: Not.
- **src/layer/pool.rs** – Input: tensor; Output: pooled tensor; Role: MaxPool and related pooling.
- **src/layer/pow.rs** – Input: base/exponent; Output: power; Role: Pow.
- **src/layer/range.rs** – Input: start/limit/delta; Output: 1D range; Role: Range.
- **src/layer/reducemean.rs** – Input: tensor; Output: reduced mean; Role: ReduceMean.
- **src/layer/reshape.rs** – Input: tensor + shape; Output: reshaped tensor; Role: Reshape.
- **src/layer/resize.rs** – Input: tensor + scales/sizes; Output: resized tensor; Role: Resize.
- **src/layer/scatternd.rs** – Input: data + indices + updates; Output: scattered tensor; Role: ScatterND.
- **src/layer/shape.rs** – Input: tensor; Output: its shape tensor; Role: Shape.
- **src/layer/slice.rs** – Input: tensor + start/ends/steps; Output: sliced tensor; Role: Slice.
- **src/layer/softmax.rs** – Input: logits; Output: normalized probabilities; Role: Softmax.
- **src/layer/split.rs** – Input: tensor; Output: list of tensors; Role: Split.
- **src/layer/squeeze.rs** – Input: tensor + axes; Output: squeezed/unsqueezed tensor; Role: Squeeze/Unsqueeze.
- **src/layer/tile.rs** – Input: tensor + repeats; Output: tiled tensor; Role: Tile.
- **src/layer/topk.rs** – Input: tensor + k; Output: top-k values/indices; Role: TopK/ArgMax.
- **src/layer/transpose.rs** – Input: tensor + perm; Output: transposed tensor; Role: Transpose.
- **src/layer/where.rs** – Input: condition + X + Y; Output: selected values; Role: Where.
- **src/layer/xor.rs** – Input: boolean tensors; Output: XOR; Role: Xor.

---

## Utilities (src/util)

### src/util/config.rs
- **Input**: YAML configuration file.
- **Output**: `Config` struct (and nested `OnnxConfig`, `PtauConfig`, `ScaleFactorConfig`, `ProverConfig`, `VerifierConfig`).
- **Role**: Defines configuration schema used by `CONFIG` in lib.rs.

### src/util/arithmetic.rs
- **Input**: Field elements/tensors.
- **Output**: Helper arithmetic routines.
- **Role**: Utility math helpers used across gadgets and layers (e.g., scaling, safe divisions, etc.).

### src/util/copy_constraint.rs
- **Input**: Indices/partitions describing copy relations.
- **Output**: Data structures used by `CopyConstraintBasicBlock`.
- **Role**: Utility logic for partitioning and padding copy-constraint tables.

### src/util/fft.rs
- **Input**: Field vectors/tensors.
- **Output**: FFT/NTT results and inverse transforms.
- **Role**: Polynomial/FFT helpers for commitments and CQ tables.

### src/util/iter.rs
- **Input**: Iterators or collections.
- **Output**: Convenience iterator utilities.
- **Role**: Small iteration helpers (e.g., `vec_iter`) used to avoid cloning.

### src/util/msm.rs
- **Input**: Lists of group elements and scalars.
- **Output**: Multi-scalar multiplication result.
- **Role**: Core MSM implementation used for commitments and pairing aggregation.

### src/util/onnx.rs
- **Input**: ONNX tensors, datatypes, shapes.
- **Output**: Translated shapes/types, fake tensors, quantized weights, etc.
- **Role**: ONNX-specific utilities (shape extraction, datatype mapping, fake input/weight generation).

### src/util/poly.rs
- **Input**: Polynomial coefficients or evaluations.
- **Output**: Polynomial transformations.
- **Role**: Polynomial helper routines for commitments and CQ.

### src/util/prover.rs
- **Input**: Basic blocks, SRS, CQ ranges.
- **Output**: CQ tables and `Data` encodings.
- **Role**: Prover utilities for generating CQ lookup tables and converting raw field tensors into committed `Data`.

### src/util/random.rs
- **Input**: RNG seeds/config.
- **Output**: Random field elements/vectors.
- **Role**: Randomness helpers for tests and protocol steps.

### src/util/serialization.rs
- **Input**: In-memory structures.
- **Output**: Serialized bytes and deserialized structures.
- **Role**: (De)serialization helpers (bincode, Arkworks) for models, setups, proofs.

### src/util/shape.rs
- **Input**: Shape vectors and axes.
- **Output**: Derived shapes.
- **Role**: Shape-calculation helpers used in many layers.

### src/util/verifier.rs
- **Input**: `PairingCheck` sets from basic blocks.
- **Output**: Aggregated G1/G2 pairs for a smaller number of pairing checks.
- **Role**: Verifier-side optimization: combines many pairing equations into fewer pairings using random linear combinations and MSM.

### src/util.rs
- **Input**: All util submodules.
- **Output**: Re-exported util functions/types.
- **Role**: Convenience wrapper so callers can do `use crate::util::*`.

---

## Binaries (src/bin)

### src/bin/witness_gen.rs
- **Input**: Same configuration/model/inputs as main.rs, but focused on witness generation.
- **Output**: Witness data (model, inputs, outputs) stored to disk, without necessarily running full proving.
- **Role**: Standalone binary for generating witnesses for a given ONNX model and config (useful for benchmarking and debugging without full proof generation).



### sample.onnx graph - structure

- $\boxed{y = \text{ReLU} (W_2 \cdot \text{ReLU}(W_1x + b_1) + b_2)}$