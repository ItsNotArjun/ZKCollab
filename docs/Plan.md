# Plan for Secure Cross-Silo Federated Learning

This document describes the roadmap for building a secure cross-silo
federated learning system using zero-knowledge proofs, **recursive folding**,
and blockchain-based binding, coordination, and verification.

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

## Phase 2 — On-Chain Binding & Accumulator Instances (Planned)

**Goal:** Cryptographically bind training inputs, randomness, policy, and
outputs using smart contracts. Clients will generate **Accumulator Instances**
instead of full SNARKs for efficiency.

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

### 2.4 On-Chain Update Commitment & Instance Binding
**Status:** ⏳ Planned

After $K$ local steps, the client computes the local update:

$$
\Delta W = W_K - W_0
$$

The client commits to it on-chain:

$$
C_{\Delta W} = \mathrm{Commit}(\Delta W)
$$

**Crucially**, this $C_{\Delta W}$ is embedded as a public input in the
client's ZK Accumulator Instance. This binds the cryptographic verification
trace to the actual weights submitted.

---

### 2.5 Accumulator Instance Submission
**Status:** ⏳ Planned

Client submits to the Aggregator (Off-Chain):

- The Model Update $\Delta W$
- The **Accumulator Instance** $(U_{client})$
- References to on-chain $C_D$ and $C_{\Delta W}$

*Note: Verification does not happen here. The Smart Contract only records the binding; the Aggregator performs the verification via folding in Phase 5.*

---

## Phase 3 — Recursive Client-Side Folding (Planned)

**Goal:** Compress many local training steps into one succinct **Client
Instance** using intra-client recursion.

Local training evolution:

$$
W_0 \;\longrightarrow\; W_1 \;\longrightarrow\; \dots \;\longrightarrow\; W_K
$$

Recursive folding asserts:

$$
\exists \; W_1, \dots, W_{K-1}
\;\; \text{s.t.} \;\;
W_{i+1} = \mathrm{Step}(W_i, B_i)
\quad \forall i \in [0, K-1]
$$

---

### 3.1 Recursive Folding of Bound Training Steps
**Status:** ⏳ Planned

- The client uses Mira/Nova folding to compress $K$ steps.
- Each recursive step:
    - Folds the previous step's instance
    - Applies one additional **bound** training step
- Result: A single Client Instance $U_{client}$ representing the full local training trajectory.

---

## Phase 4 — Multi-Round & Scalability (Future Work)

### 4.1 Recursive Folding Across FL Rounds
**Status:** 🔮 Future Work

Extend recursion and aggregation across *federated learning rounds* to
get end-to-end auditability of long-running training.

---

## Phase 5 — Parallel Aggregation & Verification (Planned)

**Goal:** Prove that the aggregator correctly combined valid client
updates using **Parallel Folding** (for validity) and **Homomorphic
Commitments** (for aggregation logic).

---

### 5.1 Parallel Validity Folding (The Tree)
**Status:** ⏳ Planned

The Aggregator constructs a Binary Folding Tree to compress $N$ client instances:

$$
U_{root} = \text{Fold}(\dots \text{Fold}(U_1, U_2) \dots )
$$

- **Optimistic Execution:** Aggregator folds all instances.
- **Bisect & Ban:** If the root check fails, the tree is bisected to identify and remove malicious client instances.
- **Result:** A single Root Instance $U_{root}$ that attests "All folded clients trained correctly."

---

### 5.2 Homomorphic Aggregation (The Math)
**Status:** ⏳ Planned

The Aggregator computes the global model using the additive properties of commitments:

$$
C_{total} = \sum_{i=1}^{N} C_{\Delta W_i}
$$

$$
\Delta W_{\text{global}} = \frac{1}{N} \sum \Delta W_i
$$

The commitment $C_{total}$ serves as the cryptographic truth of the sum of all *valid* updates.

---

### 5.3 Final Proof Generation
**Status:** ⏳ Planned

The Aggregator runs the full SNARK Prover **only once** on the final $U_{root}$.

- Input: $U_{root}$
- Output: One succinct Proof $\Pi_{final}$

---

### 5.4 Dual-Check Verification (On-Chain)
**Status:** ⏳ Planned

The Smart Contract performs the final verification to close the round.

**Check 1: Validity (The Proof)**
- Verifies $\Pi_{final}$.
- *Ensures:* Every client in the fold respected the training policy.

**Check 2: Integrity (The Homomorphic Check)**
- Verifies that the global model matches the sum of the client commitments:
  $$
  \mathrm{Commit}(\Delta W_{\text{global}}) \stackrel{?}{=} \frac{1}{N} \cdot C_{total}
  $$
- *Ensures:* The Aggregator did not fake the global weights.

---

### Final Statement

The end-to-end system establishes:

$$
\exists \; \{W_{i,0} \to W_{i,K}\}_{i=1}^N
$$

subject to:
1.  **Correct Training:** All $N$ clients followed the gradient descent path with valid data.
2.  **Correct Aggregation:** The Global Model is the exact average of these $N$ clients.