# Plan for Secure Cross-Silo Federated Learning

This document describes the roadmap for building a secure cross-silo
federated learning system using zero-knowledge proofs and
blockchain-based binding, coordination, and verification.

---

## Phase 0 — Baseline (Completed)

### 0.1 Inference Proof Support
**Status:** ✅ Completed

- ONNX → internal graph conversion
- Layer abstraction (Layer, Graph, BasicBlock)
- Plonk-based proof system (BN254, KZG)
- End-to-end proof + verification for inference

---

### 0.2 Constraint & Gadget Library
**Status:** ✅ Completed

- Arithmetic gadgets (add, mul, etc.)
- Boolean constraints
- Copy / permutation constraints
- Padding and FFT-friendly layouts

---

## Phase 1 — Single-Step Training Proof (Completed)

### 1.1 Forward Pass in Training Context
**Status:** ✅ Completed

- Forward pass reused from inference
- Loss computation encoded in-circuit (via training-specific constraints)
- Activations wired for gradient flow

---

### 1.2 Backward Pass (Differentiation)
**Status:** ✅ Completed

- Symbolic/analytic differentiation for supported ops
- Gradient tensors represented in the constraint graph
- Gradient correctness enforced by constraints (e.g., linear and ReLU VJPs)

---

### 1.3 Optimizer Update (SGD-like)
**Status:** 🟡 Partially Completed

- Parameter update logic encoded as arithmetic constraints (SGD-style update)
- Fixed-point learning-rate-based updates supported

**Open questions:**
- Formal optimizer specification beyond simple SGD variants
- Fixed-point / numeric semantics for general models
- Explicit optimizer parameter binding as public inputs / commitments

---

### 1.4 End-to-End Single Training Step Proof
**Status:** ✅ Completed

- Forward + backward + update compiled into one circuit / proof system
- Proof attests to correct state transition:

$$
(W_t, B_t) \;\longrightarrow\; W_{t+1}
$$

with the intended semantics:

$$
W_{t+1} = W_t - \eta \cdot \nabla \mathcal{L}(W_t; B_t)
$$

This single-step proof is the **atomic unit** for all future work.

---

## Phase 2 — On-Chain Binding & Verifiable Training (Planned)

**Goal:** Cryptographically bind training inputs, randomness, policy, and
outputs using smart contracts, so proofs correspond to *real* training
and cannot be replayed or cherry-picked.

---

### 2.1 On-Chain Dataset Commitment
**Status:** ⏳ Planned

Client submits to a smart contract a commitment to its local dataset:

$$
C_D = \mathrm{Commit}(D)
$$

For each batch element $(x, y) \in B_t$, the training circuit verifies
membership in the committed dataset:

$$
(x, y) \in D
$$

via a Merkle inclusion proof:

$$
\mathrm{VerifyMerkle}(x, y, C_D) = \text{true}
$$

This prevents clients from training and proving on different datasets.

---

### 2.2 On-Chain Public Randomness & Batch Selection
**Status:** ⏳ Planned

The smart contract publishes a per-round public randomness seed.

Batch indices are derived deterministically as:

$$
B_t = \mathrm{PRF}(\text{seed}, \text{round\_id}, \text{client\_id}, t)
$$

- Public seed provided per round on-chain
- Prevents cherry-picking or adaptive batch selection
- Circuit verifies that the correct batch corresponding to the seed was used

---

### 2.3 On-Chain Training Policy Binding
**Status:** ⏳ Planned

The smart contract commits to the training policy, including:

- Optimizer type
- Learning rate $\eta$
- Batch size
- Loss function $\mathcal{L}$
- Number of local steps $K$

The policy (or a hash of it) becomes a public input / commitment, and
the ZK circuit enforces compliance. This prevents provers from silently
changing training behavior between proof instances.

---

### 2.4 On-Chain Update Commitment
**Status:** ⏳ Planned

After $K$ local steps, the client computes the local update:

$$
\Delta W = W_K - W_0
$$

and commits to it on-chain:

$$
C_{\Delta W} = \mathrm{Commit}(\Delta W)
$$

Only this committed update is allowed to participate in aggregation.

The bound single-step (or K-step) statement becomes:

$$
	ext{Given } C_D, \text{policy}, \text{seed} :\quad (W_t, B_t) \;\longrightarrow\; W_{t+1}
$$

---

### 2.5 On-Chain Proof Submission & Verification
**Status:** ⏳ Planned

Client submits to the smart contract (directly or via an off-chain
verifier service):

- Accumulation of it's model (single-step or recursively folded multi-step)
- References to $C_D$ and $C_{\Delta W}$

<!-- The smart contract:
- Verifies the accumulation (directly, or checks a succinct proof from an
	off-chain verifier)
- Ensures the accumulation is bound to the registered dataset and update
- Marks the client update as valid / eligible for aggregation -->

---

### 2.6 Non-Triviality Constraints
**Status:** ⏳ Planned

To avoid degenerate or adversarial updates, the protocol and circuits
should enforce *non-triviality* conditions, e.g.:

- Bounded gradient norm or minimum loss improvement
- Reject zero, replayed, or fabricated updates
- Optional on-chain policy that encodes these thresholds and is checked
	in-circuit

These constraints ensure that only meaningful contributions enter the
aggregation phase.

---

## Phase 3 — Recursive Proof of Training (Planned)

**Goal:** Compress many local training steps into one succinct **recursive
accumulation** while preserving all binding guarantees from
Phase 2.

Local training evolution:

$$
W_0 \;\longrightarrow\; W_1 \;\longrightarrow\; \dots \;\longrightarrow\; W_K
$$

Recursive accumulation attests:

$$
W_0 \;\longrightarrow\; W_K
$$

Formally:

$$
\exists \; W_1, \dots, W_{K-1}
\;\; \text{s.t.} \;\;
W_{i+1} = \mathrm{Step}(W_i, B_i)
\quad \forall i \in [0, K-1]
$$

Each recursive step applies one more
training accumulation.

---

### 3.1 Recursive Folding of Bound Training Steps
**Status:** ⏳ Planned

- Each recursive step:
	<!-- - Verifies the previous (folded) proof -->
	- Applies one additional **bound** training step (respecting dataset,
		randomness, and policy commitments from Phase 2)
- Final recursive accumulation attests to the full local training trajectory
	from $W_0$ to $W_K$ under the bound inputs and policy

---

### 3.2 State Commitment Propagation
**Status:** ⏳ Planned

- Each recursive step takes a commitment to the previous model state
	and outputs a commitment to the new model state
- The recursion enforces continuity of the model sequence, preventing
	state forking or inconsistent trajectories
- Enables succinct audit trails: a verifier only needs the initial and
	final commitments plus the recursive proof

---

## Phase 4 — Multi-Round & Scalability (Future Work)

### 4.1 Recursive Folding Across FL Rounds
**Status:** 🔮 Future Work

Extend recursion and aggregation across *federated learning rounds* to
get end-to-end auditability of long-running training.

Across rounds, model evolution is:

$$
W_0 \;\longrightarrow\; W_1 \;\longrightarrow\; W_2 \;\longrightarrow\; \dots \;\longrightarrow\; W_K
$$

Conceptually:

- Within each round: recursively fold local client steps (Phase 3)
- Across rounds: fold global updates, producing a single long-horizon
  Proof of Training

---

## Phase 5 — Proof of Aggregation (Planned)

**Goal:** Prove that the aggregator correctly combined valid client
updates, *and* that those updates are themselves bound to on-chain
Proofs of Training (potentially already recursively folded across
rounds).

---

### 5.1 Aggregation with On-Chain Binding
**Status:** ⏳ Planned

Only updates whose dataset commitments, policies, and training proofs
are registered on-chain are eligible for aggregation.

Aggregation rule (for example, weighted averaging):

$$
\Delta W_{\text{global}} = \sum_{i=1}^{N} \alpha_i \cdot \Delta W_i
$$

- Each $\Delta W_i$ must match a committed $C_{\Delta W_i}$ associated
  with a valid Proof of Training
- Weights $\alpha_i$ are determined by the protocol (e.g. data size,
  reputation, or uniform)

---

### 5.2 Aggregation Circuit
**Status:** ⏳ Planned

Design a ZK circuit (or set of circuits) that:

- Verifies correct application of the aggregation rule (e.g., weighted
  average)
- Takes as *inputs* the committed client updates and aggregation
  coefficients
- Outputs a commitment to the aggregated global update $\Delta W_{\text{global}}$

This circuit can be composed with recursive PoT proofs.

---

### 5.3 Linking PoT to PoA
**Status:** ⏳ Planned

- Ensure that every aggregation input $\Delta W_i$ corresponds to a
  **valid** node in the Proof of Training bound on-chain
- Prevent any unproven, fake, or off-policy updates from entering the
  aggregation circuit
- Optionally encode this linkage directly in the aggregation circuit via
  commitments / hashes to the underlying PoT statements

---

### 5.4 Aggregation Proof & Finalization
**Status:** ⏳ Planned

Aggregator submits:

- 1 unified proof for all the models under it.

- Aggregation proof (for the chosen rule, e.g. weighted average)
- Commitment to the resulting global update
- References to all client PoT commitments / proofs used

Smart contract:

- Verifies aggregation correctness (directly or via a succinct outer
  proof)
- Checks that all included client updates are on-chain and valid
- Finalizes the round and optionally releases rewards / enforces slashing

---

### Final Proof of Training Statement

The final end-to-end Proof of Training and Aggregation establishes:

$$
\exists \; W_1, \dots, W_{K-1}
\;\; \text{s.t.} \;\;
W_0 \;\longrightarrow\; W_K
$$

subject to the constraints:

$$
\forall t :\;
B_t = \mathrm{PRF}(\text{seed}, \dots)
\;\land\;
B_t \subseteq D
\;\land\;
	ext{training policy respected}
$$

and that:

- All local updates are committed on-chain and backed by valid PoTs
- Aggregation is performed correctly over these committed updates
- The final global model commitment corresponds to the claimed training
  history

