//! Integration tests for the Data Binding pipeline.
//!
//! These tests prove that the ZKCollab Data Binding mechanism
//! correctly enforces the connection between the training data
//! and the on-chain committed Merkle root.

use ark_bn254::Fr;
use std::fs::File;
use std::io::BufReader;

use zk_torch::training::data_binding::{
    build_data_binding_witness, compute_merkle_root, enforce_data_binding, fr_from_hex,
    poseidon_hash,
};
use zk_torch::training::witness::SampleWitnessV1;

// ---------------------------------------------------------------------------
// Helper: load the real witness from disk
// ---------------------------------------------------------------------------
fn load_witness_v2() -> SampleWitnessV1 {
    let path = format!("{}/step_witness_v2.json", env!("CARGO_MANIFEST_DIR"));
    let file = File::open(&path).expect("step_witness_v2.json not found — run the pipeline first");
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).expect("failed to parse step_witness_v2.json")
}

fn i64_to_fr(v: i64) -> Fr {
    if v >= 0 {
        Fr::from(v as u64)
    } else {
        -Fr::from((-v) as u64)
    }
}

// ===================================================================
//  TEST 1 — Happy Path: Data Binding Succeeds
// ===================================================================

/// End-to-end test that loads the real augmented witness
/// (step_witness_v2.json) and verifies that the data-binding
/// constraint holds: Poseidon_Merkle(raw_data) == merkle_root.
///
/// This proves that an honest prover who uses the correct data
/// will always pass the binding check.
#[test]
fn test_data_binding_succeeds() {
    let witness = load_witness_v2();

    // The augmented witness must carry the data-binding fields.
    let root_hex = witness
        .merkle_root
        .as_ref()
        .expect("witness missing merkle_root — was inject_root.py run?");
    let raw_data = witness
        .raw_data
        .as_ref()
        .expect("witness missing raw_data");
    let merkle_path = witness.merkle_path.clone().unwrap_or_default();

    // Build the circuit witness and enforce the constraint.
    let db_witness =
        build_data_binding_witness(root_hex, raw_data, &merkle_path)
            .expect("failed to build DataBindingWitness");

    let result = enforce_data_binding::<32>(&db_witness);

    assert!(
        result.is_ok(),
        "Data binding should PASS for honest data! Error: {:?}",
        result.err()
    );
    println!("[PASS] test_data_binding_succeeds — constraint satisfied ✓");
}

/// Verify that the Rust-side Poseidon Merkle root computation
/// matches the root stored in merkle_root.txt (produced by the
/// C++ engine), proving cross-language consistency.
#[test]
fn test_merkle_root_matches_committed_value() {
    let witness = load_witness_v2();
    let root_hex = witness.merkle_root.as_ref().unwrap();
    let raw_data = witness.raw_data.as_ref().unwrap();

    // Recompute the root in pure Rust.
    let leaves: Vec<Fr> = raw_data.iter().map(|&v| i64_to_fr(v)).collect();
    let computed_root = compute_merkle_root(&leaves);

    // Parse the committed root from the hex string.
    let committed_root = fr_from_hex(root_hex).expect("invalid merkle_root hex");

    assert_eq!(
        computed_root, committed_root,
        "Rust Poseidon root must match the committed C++ root"
    );
    println!(
        "[PASS] test_merkle_root_matches_committed_value — Rust root == C++ root ✓"
    );
}

// ===================================================================
//  TEST 2 — Sabotage: Tampered Merkle Root Must Fail
// ===================================================================

/// A malicious prover tries to submit a *fake* Merkle root while
/// keeping the real training data.  The data-binding constraint
/// must reject this, preventing the prover from claiming a proof
/// over a different dataset than the one actually used.
#[test]
fn test_data_binding_fails_on_tampered_root() {
    let witness = load_witness_v2();
    let raw_data = witness.raw_data.as_ref().unwrap();
    let merkle_path = witness.merkle_path.clone().unwrap_or_default();

    // Construct a FAKE root (just some random-looking hex).
    let fake_root = "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";

    let db_witness =
        build_data_binding_witness(fake_root, raw_data, &merkle_path)
            .expect("failed to build witness with fake root");

    let result = enforce_data_binding::<32>(&db_witness);

    assert!(
        result.is_err(),
        "Data binding MUST FAIL when the Merkle root is tampered!"
    );

    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("Merkle root mismatch"),
        "Error should mention root mismatch, got: {}",
        err_msg
    );
    println!(
        "[PASS] test_data_binding_fails_on_tampered_root — tampered root rejected ✓"
    );
}

/// A malicious prover keeps the correct Merkle root but swaps
/// one element of the training data (data poisoning attack).
/// The binding must detect this and reject the proof.
#[test]
fn test_data_binding_fails_on_tampered_data() {
    let witness = load_witness_v2();
    let root_hex = witness.merkle_root.as_ref().unwrap();
    let raw_data = witness.raw_data.as_ref().unwrap();
    let merkle_path = witness.merkle_path.clone().unwrap_or_default();

    // Poison the data: flip the first element.
    let mut poisoned_data = raw_data.clone();
    poisoned_data[0] = poisoned_data[0].wrapping_add(9999);

    let db_witness =
        build_data_binding_witness(root_hex, &poisoned_data, &merkle_path)
            .expect("failed to build witness with poisoned data");

    let result = enforce_data_binding::<32>(&db_witness);

    assert!(
        result.is_err(),
        "Data binding MUST FAIL when training data is poisoned!"
    );

    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("Merkle root mismatch"),
        "Error should mention root mismatch, got: {}",
        err_msg
    );
    println!(
        "[PASS] test_data_binding_fails_on_tampered_data — poisoned data rejected ✓"
    );
}

/// Edge case: the prover submits completely different data
/// (totally unrelated values) with a mismatched root.
/// Both root and data are wrong — double sabotage must fail.
#[test]
fn test_data_binding_fails_on_completely_fake_witness() {
    // Fabricate entirely synthetic data.
    let fake_data: Vec<i64> = vec![111, 222, 333, 444];
    let fake_root = "0x0000000000000000000000000000000000000000000000000000000000000001";

    let db_witness =
        build_data_binding_witness(fake_root, &fake_data, &[])
            .expect("failed to build completely fake witness");

    let result = enforce_data_binding::<32>(&db_witness);

    assert!(
        result.is_err(),
        "Data binding MUST FAIL for a completely fabricated witness!"
    );
    println!(
        "[PASS] test_data_binding_fails_on_completely_fake_witness — fabricated witness rejected ✓"
    );
}

// ===================================================================
//  TEST 3 — Unit-level Poseidon & Merkle Sanity
// ===================================================================

/// Verify Poseidon hash is deterministic.
#[test]
fn test_poseidon_deterministic() {
    let input = vec![Fr::from(42u64), Fr::from(7u64)];
    let h1 = poseidon_hash(&input);
    let h2 = poseidon_hash(&input);
    assert_eq!(h1, h2, "Poseidon must be deterministic");
}

/// Verify that changing even one bit of input changes the root.
#[test]
fn test_merkle_root_avalanche() {
    let data_a: Vec<Fr> = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64)];
    let data_b: Vec<Fr> = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(5u64)];
    let root_a = compute_merkle_root(&data_a);
    let root_b = compute_merkle_root(&data_b);
    assert_ne!(
        root_a, root_b,
        "Changing one leaf must produce a different root (avalanche effect)"
    );
}

// ===================================================================
//  TEST 4 — Full SNARK Proof with Data Binding (Happy Path)
// ===================================================================

/// Loads the augmented witness and runs the full SNARK prover
/// (algebraic checks + data binding + graph-based proving).
/// This is the most comprehensive test — it proves that a valid
/// training step with correct data binding generates a proof.
#[test]
fn test_full_snark_proof_with_data_binding() {
    let witness = load_witness_v2();

    // Load the SRS (Powers of Tau).
    let srs = zk_torch::ptau::load_file("challenge", 7, 7);

    // Run the full SNARK prover.
    let result = zk_torch::training::proof::prove_sample_step_snark(&srs, &witness);

    assert!(
        result.is_ok(),
        "Full SNARK proof generation should succeed with correct data binding! Error: {:?}",
        result.err()
    );

    let proof = result.unwrap();
    assert!(
        !proof.proof_bytes.is_empty(),
        "Proof bytes must not be empty"
    );
    println!(
        "[PASS] test_full_snark_proof_with_data_binding — proof generated ({} bytes) ✓",
        proof.proof_bytes.len()
    );
}
