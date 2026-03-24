//! Data Binding circuit module for Proof-of-Training.
//!
//! Enforces that the raw training data used in a training step
//! cryptographically matches a dataset root committed on-chain.
//!
//! The circuit verifies:
//!   Poseidon_Merkle(raw_data, merkle_path) == merkle_root
//!
//! All data structures are dimension-agnostic: compile-time const
//! generics (`PATH_DEPTH`) parameterise the Merkle depth, while
//! the raw data length is fully dynamic at runtime.

use ark_bn254::Fr;
use ark_ff::{Field, PrimeField};
use ndarray::{ArrayD, IxDyn};
use sha3::{Digest, Keccak256};

// -----------------------------------------------------------------------
// Poseidon constants (BN254 scalar field, width=3, α=5)
// -----------------------------------------------------------------------
const POSEIDON_WIDTH: usize = 3;
const POSEIDON_FULL_ROUNDS: usize = 8;
const POSEIDON_PARTIAL_ROUNDS: usize = 57;

/// Deterministic round-constant generation (must match the C++ engine).
fn generate_round_constants() -> Vec<Fr> {
    let total = POSEIDON_WIDTH * (POSEIDON_FULL_ROUNDS + POSEIDON_PARTIAL_ROUNDS);
    // Seed identical to C++ engine.
    // The C++ code sets U256 limbs directly (little-endian 64-bit limbs):
    //   limbs[0] = 0x5a4b436f6c6c6162  ("ZKCollab" chars, big-endian in limb)
    //   limbs[1] = 0x506f736569646f6e  ("Poseidon" chars, big-endian in limb)
    //   limbs[2] = 0x000000524300000000 → 0x524300000000  ("RC" + padding)
    //   limbs[3] = 0
    // U256 is stored as little-endian limbs with each limb in native
    // byte order, so the 32-byte LE representation is:
    let seed_bytes: [u8; 32] = {
        let mut buf = [0u8; 32];
        let l0: u64 = 0x5a4b436f6c6c6162;
        let l1: u64 = 0x506f736569646f6e;
        let l2: u64 = 0x524300000000;
        // limbs[3] = 0  (already zero in buf)
        buf[0..8].copy_from_slice(&l0.to_le_bytes());
        buf[8..16].copy_from_slice(&l1.to_le_bytes());
        buf[16..24].copy_from_slice(&l2.to_le_bytes());
        buf
    };
    let mut state = Fr::from_le_bytes_mod_order(&seed_bytes);
    let mut rcs = Vec::with_capacity(total);
    for i in 0..total {
        state = state * state + Fr::from((i + 1) as u64);
        rcs.push(state);
    }
    rcs
}

/// S-box: x ↦ x^5.
#[inline]
fn sbox(x: Fr) -> Fr {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x
}

/// MDS mixing for width-3 state: M = [[2,1,1],[1,2,1],[1,1,2]].
#[inline]
fn mds_mix(state: &mut [Fr; POSEIDON_WIDTH]) {
    let (a, b, c) = (state[0], state[1], state[2]);
    state[0] = a + a + b + c;
    state[1] = a + b + b + c;
    state[2] = a + b + c + c;
}

/// Full Poseidon permutation (matches the C++ implementation).
fn poseidon_permutation(state: &mut [Fr; POSEIDON_WIDTH]) {
    let rcs = generate_round_constants();
    let mut rc_idx = 0;
    let half_full = POSEIDON_FULL_ROUNDS / 2;

    // First half full rounds.
    for _ in 0..half_full {
        for i in 0..POSEIDON_WIDTH {
            state[i] += rcs[rc_idx];
            rc_idx += 1;
        }
        for i in 0..POSEIDON_WIDTH {
            state[i] = sbox(state[i]);
        }
        mds_mix(state);
    }
    // Partial rounds.
    for _ in 0..POSEIDON_PARTIAL_ROUNDS {
        for i in 0..POSEIDON_WIDTH {
            state[i] += rcs[rc_idx];
            rc_idx += 1;
        }
        state[0] = sbox(state[0]);
        mds_mix(state);
    }
    // Second half full rounds.
    for _ in 0..half_full {
        for i in 0..POSEIDON_WIDTH {
            state[i] += rcs[rc_idx];
            rc_idx += 1;
        }
        for i in 0..POSEIDON_WIDTH {
            state[i] = sbox(state[i]);
        }
        mds_mix(state);
    }
}

/// Poseidon sponge hash over an arbitrary-length slice of field elements.
/// Rate = WIDTH − 1 = 2.  Identical to the C++ engine.
pub fn poseidon_hash(inputs: &[Fr]) -> Fr {
    let mut state = [Fr::from(0u64); POSEIDON_WIDTH];
    let rate = POSEIDON_WIDTH - 1;
    let mut idx = 0;
    while idx < inputs.len() {
        for i in 0..rate {
            if idx < inputs.len() {
                state[i + 1] += inputs[idx];
                idx += 1;
            }
        }
        poseidon_permutation(&mut state);
    }
    // Domain separation: absorb input length.
    state[1] += Fr::from(inputs.len() as u64);
    poseidon_permutation(&mut state);
    state[0]
}

// -----------------------------------------------------------------------
// Merkle tree helpers
// -----------------------------------------------------------------------

/// Compute the Poseidon Merkle root of a set of field-element leaves.
/// Pads to next power of two with Poseidon({}) as the zero leaf.
pub fn compute_merkle_root(leaves: &[Fr]) -> Fr {
    if leaves.is_empty() {
        return poseidon_hash(&[]);
    }
    let zero_hash = poseidon_hash(&[]);
    let mut layer: Vec<Fr> = leaves.iter().map(|l| poseidon_hash(&[*l])).collect();
    // Pad to next power of two.
    let mut n = 1;
    while n < layer.len() {
        n <<= 1;
    }
    layer.resize(n, zero_hash);

    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len() / 2);
        for pair in layer.chunks(2) {
            next.push(poseidon_hash(&[pair[0], pair[1]]));
        }
        layer = next;
    }
    layer[0]
}

/// Verify a Merkle proof for a given leaf at `index` against `root`.
pub fn verify_merkle_proof(leaf: Fr, index: usize, path: &[Fr], root: Fr) -> bool {
    let mut current = poseidon_hash(&[leaf]);
    let mut idx = index;
    for sibling in path {
        current = if idx & 1 == 0 {
            poseidon_hash(&[current, *sibling])
        } else {
            poseidon_hash(&[*sibling, current])
        };
        idx >>= 1;
    }
    current == root
}

// -----------------------------------------------------------------------
// Data Binding constraint enforcement
// -----------------------------------------------------------------------

/// Parameters for the data-binding circuit.
/// `PATH_DEPTH` is a compile-time constant that sets the maximum Merkle
/// depth — the *actual* depth is derived at runtime from the data length,
/// making the system fully dimension-agnostic.
pub struct DataBindingParams<const PATH_DEPTH: usize>;

/// The full data-binding witness.
#[derive(Clone, Debug)]
pub struct DataBindingWitness {
    /// **Public Input**: The Merkle root committed on-chain.
    pub merkle_root: Fr,
    /// **Private Witness**: The raw flattened training tensor elements.
    pub raw_data: Vec<Fr>,
    /// **Private Witness**: Merkle sibling path (one hash per tree level).
    pub merkle_path: Vec<Fr>,
}

/// Enforce the data-binding constraint:
///
///   `Poseidon_Merkle(raw_data) == merkle_root`
///
/// The Merkle root is recomputed from `raw_data` and compared against
/// the public input.  This function returns `Ok(())` if the constraint
/// holds, or an `Err` describing the mismatch.
///
/// The function is fully dimension-agnostic — `raw_data` can be any
/// length (it represents an arbitrarily shaped flattened tensor).
pub fn enforce_data_binding<const PATH_DEPTH: usize>(
    witness: &DataBindingWitness,
) -> Result<(), String> {
    if witness.raw_data.is_empty() {
        return Err("data_binding: raw_data is empty".into());
    }

    // Recompute the Merkle root from the raw data.
    let computed_root = compute_merkle_root(&witness.raw_data);

    if computed_root != witness.merkle_root {
        return Err(format!(
            "data_binding: Merkle root mismatch.\n  expected (public input): {:?}\n  computed from raw_data:  {:?}",
            witness.merkle_root, computed_root
        ));
    }

    // If a Merkle path is provided, verify it for the first leaf as a
    // secondary check (useful when the verifier has only a single leaf
    // and a path rather than the full dataset).
    if !witness.merkle_path.is_empty() {
        let first_leaf = witness.raw_data[0];
        if !verify_merkle_proof(first_leaf, 0, &witness.merkle_path, witness.merkle_root) {
            return Err("data_binding: Merkle path verification failed for leaf 0".into());
        }
    }

    Ok(())
}

/// Parse a 0x-prefixed hex string into an `Fr` element.
pub fn fr_from_hex(hex: &str) -> Result<Fr, String> {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16))
        .collect::<Result<Vec<u8>, _>>()
        .map_err(|e| format!("hex parse error: {e}"))?;
    // Fr::from_le_bytes_mod_order expects little-endian; our hex is big-endian.
    let mut le_bytes = bytes;
    le_bytes.reverse();
    Ok(Fr::from_le_bytes_mod_order(&le_bytes))
}

// -----------------------------------------------------------------------
// Integration helper: build a DataBindingWitness from the augmented JSON
// -----------------------------------------------------------------------

/// Convert an i64 witness value into a field element (matching proof.rs).
fn i64_to_fr(v: i64) -> Fr {
    if v >= 0 {
        Fr::from(v as u64)
    } else {
        -Fr::from((-v) as u64)
    }
}

/// Construct a `DataBindingWitness` from the raw fields present in an
/// augmented witness JSON (as produced by `inject_root.py`).
///
/// * `merkle_root_hex` – "0x…" hex string.
/// * `raw_data`        – flat Vec<i64> of the training tensor.
/// * `merkle_path_hex` – Vec of "0x…" hex strings (may be empty/dummy).
pub fn build_data_binding_witness(
    merkle_root_hex: &str,
    raw_data: &[i64],
    merkle_path_hex: &[String],
) -> Result<DataBindingWitness, String> {
    let merkle_root = fr_from_hex(merkle_root_hex)?;
    let raw_data_fr: Vec<Fr> = raw_data.iter().map(|&v| i64_to_fr(v)).collect();
    let merkle_path: Vec<Fr> = merkle_path_hex
        .iter()
        .map(|h| fr_from_hex(h))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DataBindingWitness {
        merkle_root,
        raw_data: raw_data_fr,
        merkle_path,
    })
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: compute root from data, then verify the binding succeeds.
    #[test]
    fn test_data_binding_roundtrip() {
        let data: Vec<Fr> = (1..=17).map(|i| Fr::from(i as u64)).collect();
        let root = compute_merkle_root(&data);
        let witness = DataBindingWitness {
            merkle_root: root,
            raw_data: data,
            merkle_path: vec![],
        };
        assert!(enforce_data_binding::<8>(&witness).is_ok());
    }

    /// Tampered data must fail.
    #[test]
    fn test_data_binding_tampered() {
        let data: Vec<Fr> = (1..=8).map(|i| Fr::from(i as u64)).collect();
        let root = compute_merkle_root(&data);
        let mut bad_data = data.clone();
        bad_data[3] = Fr::from(9999u64);
        let witness = DataBindingWitness {
            merkle_root: root,
            raw_data: bad_data,
            merkle_path: vec![],
        };
        assert!(enforce_data_binding::<4>(&witness).is_err());
    }

    /// Verify Merkle proof for a single leaf.
    #[test]
    fn test_merkle_proof_single_leaf() {
        let data: Vec<Fr> = vec![Fr::from(42u64), Fr::from(7u64), Fr::from(13u64)];
        // Build the tree manually to extract a proof.
        let zero_hash = poseidon_hash(&[]);
        let leaves: Vec<Fr> = data.iter().map(|l| poseidon_hash(&[*l])).collect();
        // Pad to 4
        let padded = vec![leaves[0], leaves[1], leaves[2], zero_hash];

        // Internal nodes
        let n01 = poseidon_hash(&[padded[0], padded[1]]);
        let n23 = poseidon_hash(&[padded[2], padded[3]]);
        let root = poseidon_hash(&[n01, n23]);

        // Proof for leaf 0: sibling=leaves[1], then sibling=n23
        let path = vec![padded[1], n23];
        assert!(verify_merkle_proof(data[0], 0, &path, root));

        // Proof for leaf 1: sibling=leaves[0], then sibling=n23
        let path1 = vec![padded[0], n23];
        assert!(verify_merkle_proof(data[1], 1, &path1, root));
    }

    /// Large tensor (dimension-agnostic stress test).
    #[test]
    fn test_data_binding_large_tensor() {
        let n = 1024;
        let data: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64)).collect();
        let root = compute_merkle_root(&data);
        let witness = DataBindingWitness {
            merkle_root: root,
            raw_data: data,
            merkle_path: vec![],
        };
        assert!(enforce_data_binding::<16>(&witness).is_ok());
    }

    /// Empty data must be rejected.
    #[test]
    fn test_data_binding_empty() {
        let witness = DataBindingWitness {
            merkle_root: Fr::from(0u64),
            raw_data: vec![],
            merkle_path: vec![],
        };
        assert!(enforce_data_binding::<0>(&witness).is_err());
    }
}
