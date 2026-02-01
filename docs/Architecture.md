# Orchestration Flow: Training, Folding, Verification, and Aggregation

This document explains how training, **parallel proof folding**, verification, and aggregation are orchestrated in a secure cross-silo federated learning system.

The system combines:
- **Off-chain computation** for efficiency (training and instance generation)
- **Zero-knowledge Folding Schemes (Mira)** for scalable aggregation
- **Blockchain smart contracts** for binding, coordination, and final enforcement

---

## System Actors

### 1. Client (Data Silo)
- Holds private local dataset $D$
- Performs local training ($W \to W'$)
- **Generates an Accumulator Instance $(U, W)$**: A lightweight cryptographic claim of correctness (not a full SNARK).

### 2. Aggregator
- **Orchestrator of Folding**: Receives instances from all clients and folds them into a single "Root Instance" using a binary tree structure.
- **Homomorphic Aggregator**: Sums the commitments of all client weights to compute the verifiable global average.
- **Final Prover**: Generates the *single* final succinct proof for the entire batch.

### 3. Verifier (Smart Contract)
- **The Final Judge**: Verifies the one final proof submitted by the Aggregator.
- **Consistency Checker**: Verifies that the global model is mathematically consistent with the sum of the client commitments (Homomorphic Check).

### 4. Blockchain (Smart Contracts)
- Provides global coordination and public randomness
- Stores commitments ($C_D, C_{\Delta W}$) to prevent equivocation
- Enforces incentives and penalties

---

## High-Level Design Principle

- **Training happens off-chain**
- **Client Proofs are "Deferred"**: Clients generate lightweight instances, not heavy proofs.
- **Aggregation is "Folding"**: The Aggregator compresses $N$ instances into 1.
- **Binding is anchored on-chain**: Clients commit to their outputs *before* the aggregator reveals the result.
- **Verification is O(1)**: The chain verifies only ONE proof regardless of the number of clients.

---

## Step-by-Step Orchestration Flow

### Step 0 — Round Initialization (On-Chain)

A smart contract initializes a federated learning round by publishing:

- Round identifier $\text{round\_id}$
- Training policy commitment (optimizer, learning rate, loss, number of steps)
- Public randomness (seed) $\text{seed}$
- Participation rules (deadlines, stake, rewards)

---

### Step 1 — Local Training (Off-Chain)

Each client performs standard local training using its private dataset $D$.

Training consists of $K$ local steps:
$$
W_0 \;\longrightarrow\; W_1 \;\longrightarrow\; \dots \;\longrightarrow\; W_K
$$

The client computes its model update:
$$
\Delta W = W_K - W_0
$$

---

### Step 2 — Input and Output Binding (On-Chain)

Before submitting the result to the aggregator, the client binds its inputs and outputs using a smart contract to prevent "changing their mind" later.

#### Dataset Commitment (Input)
The client commits to its full local dataset:
$$
C_D = \mathrm{Commit}(D)
$$

#### Update Commitment (Output)
The client calculates the cryptographic commitment to their new weights. **Crucially, this value is embedded in their ZK Instance.**
$$
C_{\Delta W} = \mathrm{Commit}(\Delta W)
$$

The client posts $C_{\Delta W}$ (or a hash of their Instance) to the chain. Once submitted, they cannot change their update.

---

### Step 3 — Deterministic Batch Selection 

Using the on-chain public seed, batch indices for each step are derived as:

$$
B_t = \mathrm{PRF}(\text{seed}, \text{round\_id}, \text{client\_id}, t)
$$

This ensures verifiable batch usage inside the proof without revealing the data itself.

---

### Step 4 — Accumulator Instance Generation (Off-Chain)

*Replaced "Proof of Training" with "Instance Generation"*

Instead of generating a heavy SNARK, the client generates a **Mira Accumulator Instance** $(U)$.

This Instance asserts that:
$$
\exists \; \text{Execution Trace} \;\; \text{s.t.} \;\; W_{new} = \text{Train}(W_{old}, D, B_t)
$$

Properties of this Instance:
- **Lightweight:** Generation is fast compared to a full SNARK.
- **Contains Commitment:** The instance $U$ public inputs include $C_{\Delta W}$.
- **Foldable:** It is mathematically compatible with other clients' instances.

---

### Step 5 — Submission to Aggregator (Off-Chain)

The client sends the following packet to the Aggregator:
1.  The Model Update $\Delta W$
2.  The Accumulator Instance $U$

*Note: The Smart Contract does NOT verify proofs here. It simply records that the Client has committed to a specific result in Step 2.*

---

## Aggregation Phase

### Step 6 — Parallel Folding & Homomorphic Averaging (Off-Chain)

The Aggregator performs two parallel operations on the received packets:

#### Path A: Validity Folding (The Tree)
The Aggregator verifies the instances "optimistically" and folds them using a Binary Tree structure:
$$
U_{final} = \text{Fold}(\dots \text{Fold}(U_1, U_2), \dots \text{Fold}(U_{N-1}, U_N))
$$
If any fold fails (indicating a malicious client), the Aggregator bisects the tree to find and discard the bad actor.

#### Path B: Homomorphic Aggregation (The Math)
The Aggregator sums the commitments provided by the valid clients:
$$
C_{total} = \sum_{i=1}^{N} C_{\Delta W_i}
$$
It then calculates the actual global average:
$$
\Delta W_{\text{global}} = \frac{1}{N} \sum \Delta W_i
$$

---

### Step 7 — Final Proof Generation (Off-Chain)

The Aggregator generates **One Final Proof** ($\Pi_{final}$) that attests to the validity of the **Root Instance** $U_{final}$.

By proving $U_{final}$, the Aggregator mathematically proves that *all* folded clients trained correctly.

---

### Step 8 — Aggregation Verification and Finalization (On-Chain)

The Aggregator submits to the Smart Contract:
1.  The Final Proof $\Pi_{final}$
2.  The Global Update $\Delta W_{\text{global}}$
3.  The Sum of Commitments $C_{total}$

The Smart Contract performs **Two Checks**:

#### Check A: Validity Check (Did they train?)
It verifies $\Pi_{final}$.
> *Pass implication:* Every client included in the fold trained their model correctly according to the rules.

#### Check B: Homomorphic Check (Did the Aggregator sum honestly?)
It calculates the commitment of the submitted global update and compares it to the sum of client commitments:
$$
\mathrm{Commit}(\Delta W_{\text{global}}) \stackrel{?}{=} \frac{1}{N} \cdot C_{total}
$$
> *Pass implication:* The Aggregator actually used the weights from the valid clients to compute the global model.

**If both pass:** The round is finalized, rewards are released, and $\Delta W_{\text{global}}$ becomes the start of the next round.