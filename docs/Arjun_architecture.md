# Orchestration Flow: Training, Proving, Verification, and Aggregation

This document explains how training, proof generation, verification, and aggregation
are orchestrated in a secure cross-silo federated learning system.

The system combines:
- **Off-chain computation** for efficiency (training and proof generation)
- **Zero-knowledge proofs** for correctness guarantees
- **Blockchain smart contracts** for binding, coordination, and enforcement

---

## System Actors

### 1. Client (Data Silo)
- Holds private local dataset $D$
- Performs local training
- Generates model accumulations

### 2. Aggregator
- Coordinates aggregation of client updates via accumulations
- Creates 1 singular proof for all the clients under it
- Generates a proof of correct aggregation

### 3. Verifier
- Verifies zero-knowledge proofs
- On Chain

### 4. Blockchain (Smart Contracts)
- Provides global coordination and public randomness
- Stores commitments and verification outcomes
- Enforces incentives and penalties

---

## High-Level Design Principle

- **Training happens off-chain**
- **Proof generation happens off-chain**
- **Binding and verification are anchored on-chain**
- **Aggregation is verifiable and enforced on-chain**

This separation ensures scalability without sacrificing security.

---

## Step-by-Step Orchestration Flow

### Step 0 — Round Initialization (On-Chain)

A smart contract initializes a federated learning round by publishing:

- Round identifier $\text{round\_id}$
- Training policy commitment (optimizer, learning rate, loss, number of steps)
- Public randomness (seed) $\text{seed}$
- Participation rules (deadlines, stake, rewards)

The public randomness is used to deterministically select training batches.

---

### Step 1 — Local Training (Off-Chain)

- input binding step --- see what input binding is in step 2.

Each client performs standard local training using its private dataset $D$.

Training consists of $K$ local steps:

$$
W_0 \;\longrightarrow\; W_1 \;\longrightarrow\; \dots \;\longrightarrow\; W_K
$$

The client computes its model update:

$$
\Delta W = W_K - W_0
$$

This training is fast and uses conventional ML frameworks (e.g. PyTorch, GPUs),
but is not trusted by itself.

---

### Step 2 — Input and Output Binding (On-Chain)

Before generating a accumulation, the client binds its inputs and outputs using a smart contract.

- Input binding happens before training
- Output binding happens after training



#### Dataset Commitment

The client commits to its full local dataset:

$$
C_D = \mathrm{Commit}(D)
$$

This prevents the client from changing or cherry-picking data later.

#### Update Commitment

After training, the client commits to its update:

$$
C_{\Delta W} = \mathrm{Commit}(\Delta W)
$$

Once submitted on-chain, these commitments are immutable.

---

### Step 3 — Deterministic Batch Selection 

Using the on-chain public seed, batch indices for each step are derived as:

$$
B_t = \mathrm{PRF}(\text{seed}, \text{round\_id}, \text{client\_id}, t)
$$

This ensures:
- No adaptive or adversarial batch selection
- Verifiable batch usage inside the proof

---

### Step 4 — Accumulation Generation (Off-Chain)

The client generates a accumulation of its training.

The accumulation attests that:

$$
(W_t, B_t) \;\longrightarrow\; W_{t+1}
$$

for each step, with intended semantics:

$$
W_{t+1} = W_t - \eta \cdot \nabla \mathcal{L}(W_t; B_t)
$$

Using recursion, multiple steps are compressed into a single accumulation asserting:

$$
W_0 \;\longrightarrow\; W_K
$$

Formally, the accumulation establishes:

$$
\exists \; W_1, \dots, W_{K-1}
\;\; \text{s.t.} \;\;
W_{i+1} = \mathrm{Step}(W_i, B_i)
\quad \forall i \in [0, K-1]
$$

subject to:
- Dataset commitment $C_D$
- Training policy commitment
- Batch selection derived from $\text{seed}$
- Output commitment $C_{\Delta W}$
---
The client then submits its accumulation in the form of a tuple: $$ Pi​=\{{Wnew​,Ui​}\} $$
where $Wnew$ are the new weights and $Ui$ is the accumulation. 

---

### Step 5 — Proof Submission and Verification (On-Chain)

The client submits to the smart contract:

- Accumulation of Training (or its hash)
- References to $C_D$ and $C_{\Delta W}$

Verification is done at aggregation
---

## Aggregation Phase

### Step 6 — Aggregation of Client Updates (Off-Chain)
The aggregator collects all client accumulation and recursively combines them in
the form of a binary tree.

The aggregator then generates a singular proof for the entire tree. If the proof is valid,
all clients under the aggregator are valid.
If the proof is not valid, the aggregator splits the tree to find the individual bad apples.
Once found, the aggregator removes the bad apples and penalises them.
The proof is them recomputed.


The aggregator collects all client updates $\Delta W_i$
whose commitments and proofs were accepted.

The global update is computed as:

$$
\Delta W_{\text{global}} = \sum_{i=1}^{N} \alpha_i \cdot \Delta W_i
$$

where $\alpha_i$ are aggregation weights.

---

### Step 7 — Proof of Aggregation (Off-Chain)

The aggregator generates a Proof of Aggregation (PoA) attesting that:

- All inputs correspond to valid Proofs of Training
- The aggregation rule was applied correctly

---

### Step 8 — Aggregation Verification and Finalization (On-Chain)

The aggregator submits:
- Aggregation proof
- Commitment to the global update

The smart contract:
- Verifies the Proof of Aggregation
- Finalizes the federated round
- Releases rewards or applies penalties

The global model update is now the input for the next round.

---

