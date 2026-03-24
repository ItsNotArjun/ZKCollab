# ZKCollab Data Binding — Presenter's Cheat Sheet

## Pre-Demo Setup (do these *before* the presentation)

```bash
# From Windows PowerShell, verify everything is built:
cd C:\Users\Abhishek\ZK_fork\ZKCollab
cargo build --release --bin poseidon_field --bin generate_sample_proof
cargo test --test data_binding_integration --no-run
```

---

## Live Demo Commands

### 1. Run the Full Visual Pipeline

```bash
# In WSL terminal:
cd /mnt/c/Users/Abhishek/ZK_fork/ZKCollab
bash scripts/demo_run.sh
```

> **What to narrate**: This script computes the Poseidon Merkle root of the
> training data, commits it to a smart contract, injects it into the witness,
> and generates a ZK-SNARK proof that data binding holds.

---

### 2. Run the "Happy Path" Test (Honest Prover Succeeds)

```powershell
# In Windows PowerShell:
cd C:\Users\Abhishek\ZK_fork\ZKCollab
cargo test --test data_binding_integration test_data_binding_succeeds -- --nocapture
```

> **What to narrate**: This loads the real witness and verifies that
> `Poseidon_Merkle(raw_data) == committed_root`. The honest prover passes.

---

### 3. Run the "Sabotage" Tests (Tampered Data is Rejected)

```powershell
# Tampered root (attacker changes the committed root):
cargo test --test data_binding_integration test_data_binding_fails_on_tampered_root -- --nocapture

# Poisoned data (attacker modifies training data):
cargo test --test data_binding_integration test_data_binding_fails_on_tampered_data -- --nocapture

# Completely fabricated witness:
cargo test --test data_binding_integration test_data_binding_fails_on_completely_fake_witness -- --nocapture
```

> **What to narrate**: Even changing ONE element of the training data
> produces a completely different Merkle root (avalanche effect),
> causing the data-binding constraint to fail. The proof is rejected.

---

### 4. Run ALL Tests at Once

```powershell
cargo test --test data_binding_integration -- --nocapture
```

> This runs all 8 tests: happy path, sabotage (3 variants),
> Poseidon sanity checks, cross-engine consistency, and the full SNARK proof.

---

### 5. Run the Full SNARK Proof Test (Showstopper)

```powershell
cargo test --test data_binding_integration test_full_snark -- --nocapture
```

> **What to narrate**: This is the real deal — it runs the complete
> SNARK proving pipeline (algebraic checks + data binding + graph-based
> constraint system) and generates a 2880-byte cryptographic proof.

---

## Key Talking Points

- **Data Binding** = Poseidon Merkle root of training data committed on-chain
- **Security**: Even a 1-bit change in training data → completely different root → proof fails
- **Pipeline**: PyTorch → Witness → Poseidon Root → Smart Contract → ZK Proof
- **Zero Knowledge**: The verifier learns nothing about the training data, only that it matches the committed root
- **Dimension-Agnostic**: Works for any tensor shape (4-dim, 1024-dim, etc.)

---

## Quick Troubleshooting

| Problem | Fix |
|---------|-----|
| `step_witness_v1.json` missing | `python generate_witness.py` |
| `step_witness_v2.json` stale | `python scripts/inject_root.py` |
| Hardhat deploy fails | `npm install` then retry |
| Tests fail on root mismatch | Re-run `demo_run.sh` to regenerate all artifacts |
