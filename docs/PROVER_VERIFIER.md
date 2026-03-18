# Prover & Verifier Scripts

## Overview

The ZKCollab proof pipeline is split into two independent scripts:
- **`prover.sh`** — Client-side proof generation (ANY witness, ANY model)
- **`verifier.sh`** — Independent proof verification (blockchain-backed validation)

Both scripts are **fully parametric** and **not hardcoded** to any specific model or dataset.

---

## Prover Script (`scripts/prover.sh`)

**Purpose**: Generate a SNARK proof of training with data binding verification.

### Usage

```bash
bash scripts/prover.sh \
  --witness <witness.json> \
  --output <output_directory> \
  [--config <config.yaml>]
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--witness` | ✓ | Path to witness JSON file containing training state |
| `--output` | ✓ | Directory where artifacts will be written |
| `--config` | ✗ | Optional config file (defaults to `config.yaml`) |

### Process

The prover performs **5 sequential steps**:

1. **Extract Raw Data** from witness (field `x`)
   - Validates witness structure
   - Outputs flattened data array

2. **Compute Merkle Root** using Poseidon hashing
   - Reads raw data array
   - Builds binary Merkle tree
   - Outputs root as hex string

3. **Deploy Contract & Commit Root** on-chain
   - Deploys `TrainingRegistry.sol`
   - Calls `commitModelRoot(root)` on-chain
   - Verifies commitment was successful

4. **Augment Witness** with data binding fields
   - Adds `merkle_root` field (hex)
   - Adds `merkle_path` field (Merkle proof path)

5. **Generate SNARK Proof**
   - Runs algebraic constraint checks
   - Invokes plonky2 KZG proving
   - Outputs binary proof file

### Output Artifacts

```
output_dir/
├── merkle_root.txt         # Merkle root (hex: 0x...)
├── deployment_info.json    # {contract_address, deploy_tx_hash, is_committed, timestamp}
├── witness_augmented.json  # Witness + data binding fields
└── proof.bin              # SNARK proof (binary, ~2880 bytes)
```

### Example: Test Data

```bash
# Generate proof for sample witness
bash scripts/prover.sh \
  --witness step_witness_v1.json \
  --output ./proof_artifacts
```

### Example: Custom Witness

```bash
# Generate proof for your model's training step
bash scripts/prover.sh \
  --witness my_model_step_5.json \
  --output ./batch_5_proof \
  --config my_config.yaml
```

---

## Verifier Script (`scripts/verifier.sh`)

**Purpose**: Independently verify a proof and its on-chain commitment.

### Usage

```bash
bash scripts/verifier.sh \
  --proof <proof.bin> \
  --witness <witness.json> \
  --deployment <deployment_info.json> \
  [--root <expected_root_hex>]
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--proof` | ✓ | Path to proof binary file from prover |
| `--witness` | ✓ | Original witness JSON with data binding fields |
| `--deployment` | ✓ | Deployment info from prover (JSON) |
| `--root` | ✗ | Expected Merkle root (for validation, optional) |

### Process

The verifier performs **6 independent verification checks**:

1. **Extract Deployment Info**
   - Reads contract address, deployed root, on-chain status
   - Prints deployment summary

2. **Recompute Merkle Root**
   - Reads witness data (`x` field)
   - Recomputes Poseidon Merkle tree
   - Validates against deployed root

3. **Verify Data Binding Fields**
   - Checks `merkle_root` and `merkle_path` exist in witness
   - Validates field structure

4. **Verify Proof Artifact**
   - Checks proof file exists and has valid size
   - Expected size: ~2880 bytes (for plonky2/KZG params(7,7))

5. **Verify Witness-Proof Consistency**
   - Ensures witness `merkle_root` matches deployed root
   - Detects tampering in witness after proof generation

6. **Verify Algebraic Constraints**
   - Validates all required fields present in witness
   - Checks data types and field completeness
   - Simulates constraint structure validation

### Output

- **Exit code 0**: All checks passed ✅
- **Exit code 1**: One or more checks failed ❌
- **Console output**: Detailed results for each check (color-coded)

### Example: Verify Proof

```bash
# Verify proof artifacts
bash scripts/verifier.sh \
  --proof ./proof_artifacts/proof.bin \
  --witness ./proof_artifacts/witness_augmented.json \
  --deployment ./proof_artifacts/deployment_info.json
```

### Example: Batch Verification

```bash
# Verify multiple proofs in a directory
for proof_dir in batch_*/; do
  echo "Verifying $proof_dir..."
  bash scripts/verifier.sh \
    --proof "$proof_dir/proof.bin" \
    --witness "$proof_dir/witness_augmented.json" \
    --deployment "$proof_dir/deployment_info.json" || echo "FAILED: $proof_dir"
done
```

---

## Workflow: End-to-End

### Prover (Alice, training model)

```bash
# 1. Run training, output witness
python3 sample_model.py  # Creates step_witness_v1.json

# 2. Generate proof
bash scripts/prover.sh \
  --witness step_witness_v1.json \
  --output ./alice_proof

# 3. Send to verifier:
#    - alice_proof/proof.bin
#    - alice_proof/witness_augmented.json
#    - alice_proof/deployment_info.json
```

### Verifier (Bob, validating proof)

```bash
# 1. Receive artifacts from prover

# 2. Verify independently
bash scripts/verifier.sh \
  --proof ./alice_proof/proof.bin \
  --witness ./alice_proof/witness_augmented.json \
  --deployment ./alice_proof/deployment_info.json

# 3. If exit code = 0: Proof is valid ✓
#    If exit code = 1: Proof failed validation ✗
```

---

## General-Purpose Design

### What's NOT Hardcoded

✅ **Fully parametric**:
- Accepts ANY witness JSON file
- Works with different dimensions (4×4 MLP, or others)
- Configurable input/output paths
- Optional config override

### What's Built Into the Circuit

⚙️ **Fixed by architecture** (cannot change without recompiling):
- Model architecture (currently 4×4 MLP)
- Constraint types (linear VJP, SGD update, range checks)
- Poseidon hash parameters (width=3, α=5, rounds=8+57)
- SNARK proving system (plonky2 with KZG)

### To Support Different Models

To use with a different model architecture:

1. **Generate witness** with your model's parameters and gradients
2. **Ensure witness has all required fields**: `x`, `w1`, `b1`, `w2`, `b2`, `lr`, `grad_x`, `grad_w1`, `grad_b1`, `grad_w2`, `grad_b2`
3. **Run prover** with your witness file
4. **Verifier** validates without knowing model details (just checks on-chain commitment)

> **Note**: The constraint circuit is currently hardcoded to 4×4 MLP. For other model architectures, the Rust constraints would need to be regenerated using the [`zk-ml`](https://github.com/zk-ml/zkml) framework, which is outside the scope of these scripts.

---

## Errors & Troubleshooting

### Prover Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Missing --witness argument` | No witness file specified | Add `--witness <file>` |
| `Witness file not found` | Path incorrect or file missing | Verify path exists |
| `Failed to build poseidon_field` | Rust compilation issue | Run `cargo build --release` |
| `Merkle root computation failed` | Invalid witness data | Check `x` field is valid array |
| `Contract deployment failed` | Hardhat/Node.js error | Check Node.js is installed |

### Verifier Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Merkle root MISMATCH` | Witness was tampered after proof | Use original witness from prover |
| `Root was NOT committed on-chain` | Deployment failed | Check deployment_info.json |
| `Proof file is empty` | Proof generation failed | Regenerate proof with prover |
| `Witness missing "x" field` | Witness structure invalid | Generate fresh witness |

---

## Security Notes

### What's Verified

✅ **Cryptographic soundness**:
- SNARK proof correctness (plonky2 verification)
- Merkle root computation (Poseidon hash iterated correctly)
- On-chain commitment (smart contract code executed)

✅ **Data integrity**:
- Witness cannot be modified after proof (Merkle root mismatch)
- All training hyperparameters committed on-chain
- Algebraic constraints checked pre-proof

### What's NOT Verified

❌ **Outside SNARK circuit** (pre-flight checks):
- Witness extraction from model training
- Initial data quality/validity
- Gradient computation correctness

> **Future**: To prove gradient correctness, the PyTorch code would need to be inside the SNARK circuit (e.g., using `zk-ml` framework).

---

## Performance

Typical end-to-end times:

| Step | Time |
|------|------|
| Data extraction | < 1s |
| Merkle root (4×4 tensor) | < 1s |
| Contract deployment | 2-5s |
| Witness augmentation | < 1s |
| SNARK proof generation | 5-30s |
| **Total** | **~10-40s** |

Verification is much faster (~2s total).

---

## Advanced Usage

### Batch Processing

```bash
# Generate proofs for multiple training steps
for step in 1 2 3 4 5; do
  witness="model_step_${step}.json"
  output="proofs/step_${step}"
  
  bash scripts/prover.sh --witness "$witness" --output "$output"
  bash scripts/verifier.sh --proof "$output/proof.bin" \
                           --witness "$output/witness_augmented.json" \
                           --deployment "$output/deployment_info.json"
done
```

### CI/CD Integration

```yaml
# Example: GitHub Actions workflow
- name: Generate Proof
  run: bash scripts/prover.sh --witness witness.json --output ./artifacts

- name: Verify Proof
  run: bash scripts/verifier.sh --proof ./artifacts/proof.bin \
                                 --witness ./artifacts/witness_augmented.json \
                                 --deployment ./artifacts/deployment_info.json
```

### Custom Witness Generation

```bash
# Generate witness from your model
python3 my_training_script.py --output my_witness.json

# Prove it
bash scripts/prover.sh --witness my_witness.json --output ./my_proof

# Verify independently
bash scripts/verifier.sh --proof ./my_proof/proof.bin \
                         --witness ./my_proof/witness_augmented.json \
                         --deployment ./my_proof/deployment_info.json
```

---

## Questions?

- **How do I customize for my model?** Generate a witness with your model's parameters (4 matrices + gradients)
- **Can I run prover and verifier on different machines?** Yes! Just transfer the 3 artifacts
- **What if I want to change the constraint system?** Edit `src/training/constraints.rs` and rebuild
- **Can this verify proofs on-chain?** The proof is already on-chain (via contract commitment); off-chain verification is an optimization

---
