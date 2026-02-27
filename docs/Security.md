# Security in Secure Cross-Silo Federated Learning

This document details the security mechanisms and guarantees in our federated learning system, focusing on data binding, commitments, and the enforcement of protocol integrity through zero-knowledge proofs and blockchain anchoring.

---

## 1. Data Binding and Commitments

### Dataset Commitment
- **Before training begins**, each client computes a cryptographic commitment to its full local dataset $D$:
	$$
	C_D = \mathrm{Commit}(D)
	$$
- This commitment is submitted on-chain and is immutable, preventing clients from changing or cherry-picking data after the fact.
- During proof generation, the circuit can require Merkle inclusion proofs to ensure that each training batch element $(x, y)$ is a member of the committed dataset $D$.

### Update Commitment
- **After local training**, the client computes the model update $\Delta W = W_K - W_0$ and commits to it:
	$$
	C_{\Delta W} = \mathrm{Commit}(\Delta W)
	$$
- This commitment is also submitted on-chain, binding the client to the specific update produced by its training run.
- Only updates with valid commitments are eligible for aggregation.

### Training Policy and Public Randomness Binding
- The smart contract publishes a commitment to the training policy (optimizer, learning rate, loss, number of steps, etc.) and a public randomness seed for each round.
- Batch selection for each step is derived deterministically from the public seed, round ID, client ID, and step index:
	$$
	B_t = \mathrm{PRF}(\text{seed}, \text{round\_id}, \text{client\_id}, t)
	$$
- The circuit enforces that the correct batch, as determined by the public randomness, is used for each training step, preventing adaptive or adversarial batch selection.

---

## 2. Zero-Knowledge Proofs of Training

- Each client generates a zero-knowledge Proof of Training (PoT) attesting that, for each step:
	$$
	(W_t, B_t) \longrightarrow W_{t+1}
	$$
	with the intended update rule (e.g., SGD):
	$$
	W_{t+1} = W_t - \eta \cdot \nabla \mathcal{L}(W_t; B_t)
	$$
- The proof shows that the client followed the committed policy, used the correct batches, and produced the committed update, all without revealing private data or model parameters.
- The circuit can be extended to enforce Merkle inclusion for batch elements, policy compliance, and non-triviality constraints (e.g., minimum loss improvement, bounded gradient norm).

---

## 3. On-Chain Verification and Enforcement

- Proofs and commitments are submitted to a smart contract, which verifies (or attests to off-chain verification of) the PoT.
- Only updates with valid proofs and matching commitments are accepted for aggregation.
- The smart contract enforces that all protocol rules (dataset, policy, randomness, update) are respected, and can apply incentives or penalties accordingly.

---

## 4. Aggregation and Proof of Aggregation (PoA)

- The aggregator collects all valid, committed client updates and computes the global update (e.g., weighted average):
	$$
	\Delta W_{\text{global}} = \sum_{i=1}^{N} \alpha_i \cdot \Delta W_i
	$$
- A Proof of Aggregation (PoA) is generated to attest that only valid, on-chain-committed updates were included and that the aggregation rule was applied correctly.
- The final global model commitment is published on-chain, ensuring end-to-end auditability.

---

## 5. Recursive Proofs and State Continuity

- The protocol supports recursive folding of training steps and rounds, allowing many steps or rounds to be compressed into a single succinct proof.
- Each recursive step verifies the previous state commitment and applies one more step, ensuring continuity and preventing state forking.
- This enables scalable, long-horizon auditability of the entire federated learning process.

---

## 6. Security Guarantees

- **Data immutability:** Dataset and update commitments prevent post-hoc changes or cherry-picking.
- **Policy and randomness binding:** Public commitments and deterministic batch selection prevent adaptive or adversarial training.
- **Proof soundness:** Zero-knowledge proofs ensure that only valid training and aggregation steps are accepted, without leaking private data.
- **On-chain enforcement:** Smart contracts anchor all commitments and proofs, enabling transparent, tamper-resistant coordination and verification.
- **Non-triviality:** Optional constraints can enforce that only meaningful updates are accepted, preventing degenerate or adversarial behavior.

---

## 7. Future Extensions

- Stronger non-triviality checks (e.g., minimum accuracy improvement)
- Support for more complex optimizers and policies
- Enhanced privacy via more advanced ZK techniques
- End-to-end recursive proofs across multiple federated rounds

---

This security architecture ensures that all critical steps in federated learning are cryptographically bound, verifiable, and auditable, providing strong guarantees against manipulation, data leakage, and protocol deviation.
