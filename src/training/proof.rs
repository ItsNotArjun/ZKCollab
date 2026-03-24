use crate::basic_block::{Data, SRS};
use crate::training::compile::{build_training_graph, ForwardOp, TrainingSpec};
use crate::training::constraints::{FIXED_SCALE, enforce_linear_vjp, enforce_relu_vjp, enforce_sgd_update};
use crate::training::data_binding::{build_data_binding_witness, enforce_data_binding};
use crate::util::convert_to_data;
use ark_bn254::{Fr, G1Affine, G2Affine};
use ark_poly::univariate::DensePolynomial;
use ark_serialize::CanonicalSerialize;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use plonky2::util::timing::TimingTree;
use rand::{rngs::StdRng, SeedableRng};

use super::witness::SampleWitnessV1;

/// Full zk proof artifact (to be backed by a SNARK prover).
/// For now, we focus on enforcing the algebraic relations between
/// forward pass, gradients, and SGD update using the richer witness.
#[derive(Debug, Clone)]
pub struct SampleProofV1 {
  /// Placeholder for serialized SNARK proof bytes.
  pub proof_bytes: Vec<u8>,
}

fn i64_to_fr(v: i64) -> Fr {
  if v >= 0 {
    Fr::from(v as u64)
  } else {
    -Fr::from((-v) as u64)
  }
}

fn vec_to_array1(v: &[i64]) -> ArrayD<Fr> {
  let data: Vec<Fr> = v.iter().map(|&x| i64_to_fr(x)).collect();
  Array1::from(data).into_dyn()
}

fn mat_to_array2(m: &[Vec<i64>]) -> ArrayD<Fr> {
  let rows = m.len();
  let cols = m.first().map(|r| r.len()).unwrap_or(0);
  let mut flat: Vec<Fr> = Vec::with_capacity(rows * cols);
  for r in m {
    for &x in r {
      flat.push(i64_to_fr(x));
    }
  }
  Array2::from_shape_vec((rows, cols), flat).unwrap().into_dyn()
}

/// Enforce, at the algebraic level, that the witness-provided
/// gradients and SGD updates for the 2-layer 4x4 MLP are
/// consistent with the forward activations and parameters.
fn enforce_sample_step_algebra(w: &SampleWitnessV1) -> Result<(), String> {
  if w.fixed_scale_k != 16 {
    return Err("unexpected fixed_scale_k; expected 16".to_string());
  }

  // Inputs and targets
  let x = vec_to_array1(&w.x);
  let _y_target = vec_to_array1(&w.y_target);

  // Forward activations
  let z1 = vec_to_array1(&w.z1);
  let a1 = vec_to_array1(&w.a1);
  let z2 = vec_to_array1(&w.z2);
  let y_pred = vec_to_array1(&w.y_pred);

  // Gradients wrt activations and input
  let grad_y = vec_to_array1(&w.grad_y);
  let grad_z2 = vec_to_array1(&w.grad_z2);
  let grad_a1 = vec_to_array1(&w.grad_a1);
  let grad_z1 = vec_to_array1(&w.grad_z1);
  let grad_x = vec_to_array1(&w.grad_x);

  // Parameters before/after and their gradients
  let w1_before = mat_to_array2(&w.w1_before);
  let b1_before = vec_to_array1(&w.b1_before);
  let w2_before = mat_to_array2(&w.w2_before);
  let b2_before = vec_to_array1(&w.b2_before);
  let w1_after = mat_to_array2(&w.w1_after);
  let b1_after = vec_to_array1(&w.b1_after);
  let w2_after = mat_to_array2(&w.w2_after);
  let b2_after = vec_to_array1(&w.b2_after);

  let grad_w1 = mat_to_array2(&w.grad_w1);
  let grad_b1 = vec_to_array1(&w.grad_b1);
  let grad_w2 = mat_to_array2(&w.grad_w2);
  let grad_b2 = vec_to_array1(&w.grad_b2);

  // Enforce VJP for final linear layer: a1 -> z2
  enforce_linear_vjp(&a1, &w2_before, &grad_z2, &grad_a1, &grad_w2)?;

  // Enforce VJP for final ReLU: z2 -> y_pred
  enforce_relu_vjp(&z2, &grad_y, &grad_z2)?;

  // Enforce VJP for first linear layer: x -> z1
  enforce_linear_vjp(&x, &w1_before, &grad_z1, &grad_x, &grad_w1)?;

  // Enforce VJP for first ReLU: z1 -> a1
  enforce_relu_vjp(&z1, &grad_a1, &grad_z1)?;

  // SGD updates for weights and biases
  let lr_scaled = w.lr_scaled as i64;
  enforce_sgd_update(&w1_before, &grad_w1, lr_scaled, &w1_after)?;
  enforce_sgd_update(&b1_before, &grad_b1, lr_scaled, &b1_after)?;
  enforce_sgd_update(&w2_before, &grad_w2, lr_scaled, &w2_after)?;
  enforce_sgd_update(&b2_before, &grad_b2, lr_scaled, &b2_after)?;

  // ---- Data Binding constraint ----
  // If the witness carries an injected merkle_root, enforce that
  // Poseidon_Merkle(raw_data) == merkle_root.
  if let (Some(ref root_hex), Some(ref raw_data)) = (&w.merkle_root, &w.raw_data) {
    let path_hex: Vec<String> = w.merkle_path.clone().unwrap_or_default();
    let db_witness = build_data_binding_witness(root_hex, raw_data, &path_hex)?;
    // PATH_DEPTH=32 is a generous upper bound; the runtime data
    // length determines the actual tree depth dynamically.
    enforce_data_binding::<32>(&db_witness)?;
  }

  Ok(())
}

/// Given an SRS and a rich SampleWitnessV1, attempt to enforce the
/// forward, gradient, and SGD relations for the 2-layer MLP.
///
/// NOTE: The witness is produced via floating-point PyTorch and then
/// quantized to fixed-point integers. Due to rounding, the exact
/// fixed-field equalities we would like to check may not always
/// hold. For now we *log* inconsistencies instead of panicking, so
/// that the sample proof pipeline can run end-to-end.
pub fn prove_sample_step(_srs: &SRS, witness: &SampleWitnessV1) -> SampleProofV1 {
  if let Err(e) = enforce_sample_step_algebra(witness) {
    panic!("sample training step algebraic/data-binding check failed: {}", e);
  }

  // Placeholder: real implementation will call into Graph-based
  // SNARK prover and populate `proof_bytes`.
  SampleProofV1 { proof_bytes: Vec::new() }
}

fn serialize_proofs(
  proofs: &Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>)>,
) -> Result<Vec<u8>, ark_serialize::SerializationError> {
  let mut buf = Vec::new();
  for (g1s, g2s, frs) in proofs {
    for g in g1s {
      g.serialize_uncompressed(&mut buf)?;
    }
    for g in g2s {
      g.serialize_uncompressed(&mut buf)?;
    }
    for f in frs {
      f.serialize_uncompressed(&mut buf)?;
    }
  }
  Ok(buf)
}

/// End-to-end SNARK-backed proof for the SampleModel training step.
/// Accepts the full 4×4 MLP witness produced by generate_witness.py.
pub fn prove_sample_step_snark(srs: &SRS, witness: &SampleWitnessV1) -> Result<SampleProofV1, String> {
  // Enforce algebraic and data-binding checks as a hard gate.
  // If this fails, we do not allow proof generation to proceed.
  enforce_sample_step_algebra(witness)?;

  let spec = TrainingSpec {
    ops: vec![ForwardOp::Linear, ForwardOp::ReLU, ForwardOp::Linear, ForwardOp::ReLU],
  };
  let (mut graph, models, num_weights) = build_training_graph(&spec);
  if num_weights != 2 {
    return Err(format!("unexpected number of weight inputs in training graph: {}", num_weights));
  }

  // Convert full witness vectors/matrices to field-element tensors.
  let lr_fr = i64_to_fr(witness.lr_scaled) / Fr::from(FIXED_SCALE);

  // Inputs layout expected by build_training_graph:
  //  0: forward input activation  (vector, dim n)
  //  1: upstream gradient seed     (vector, dim m)
  //  2: learning rate              (scalar)
  //  3: randomness challenge target for the Challenge/Eq blocks (scalar)
  //  4..: per-layer weights        (matrices, m x n)
  let x_tensor = vec_to_array1(&witness.x);
  let upstream_tensor = vec_to_array1(&witness.grad_y);
  let lr_tensor = ArrayD::from_shape_vec(IxDyn(&[1]), vec![lr_fr]).unwrap();
  let dummy_chal_tensor = ArrayD::from_shape_vec(IxDyn(&[1]), vec![Fr::from(0u64)]).unwrap();
  let w1_tensor = mat_to_array2(&witness.w1_before);
  let w2_tensor = mat_to_array2(&witness.w2_before);

  let mut input_tensors = vec![x_tensor, upstream_tensor, lr_tensor, dummy_chal_tensor, w1_tensor, w2_tensor];

  let model_refs: Vec<&ArrayD<Fr>> = models.iter().collect();

  // First run: compute the training challenge produced inside the
  // graph so we can supply the same value as an explicit input.
  let inputs_ref_first: Vec<&ArrayD<Fr>> = input_tensors.iter().collect();
  let outputs_first = graph.run(&inputs_ref_first, &model_refs);

  let chal_bb_idx = graph
    .basic_blocks
    .iter()
    .position(|b| format!("{:?}", b).contains("ChallengeBlock"))
    .ok_or("ChallengeBlock not found in training graph")?;
  let chal_node_idx = graph
    .nodes
    .iter()
    .position(|n| n.basic_block == chal_bb_idx)
    .ok_or("Challenge node not found in training graph")?;

  let chal_tensor = outputs_first
    .get(chal_node_idx)
    .and_then(|v| v.get(0))
    .ok_or("Challenge node produced no outputs")?
    .clone();

  // Overwrite the randomness-input slot with the derived challenge
  // and re-run so all downstream values are consistent.
  input_tensors[3] = chal_tensor.clone();
  let inputs_ref: Vec<&ArrayD<Fr>> = input_tensors.iter().collect();
  let outputs = graph.run(&inputs_ref, &model_refs);

  // SNARK proving pipeline: setup -> encodeOutputs -> prove.
  let models_data: Vec<ArrayD<Data>> = models.iter().map(|m| convert_to_data(srs, m)).collect();
  let models_data_ref: Vec<&ArrayD<Data>> = models_data.iter().collect();

  let setups_proj = graph.setup(srs, &models_data_ref);
  let setups_affine: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<DensePolynomial<Fr>>)> = setups_proj
    .iter()
    .map(|(g1p, g2p, polys)| {
      (
        g1p.iter().map(|x| (*x).into()).collect(),
        g2p.iter().map(|x| (*x).into()).collect(),
        polys.clone(),
      )
    })
    .collect();
  let setups_ref: Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>)> =
    setups_affine.iter().map(|s| (&s.0, &s.1, &s.2)).collect();

  let inputs_data: Vec<ArrayD<Data>> = input_tensors.iter().map(|t| convert_to_data(srs, t)).collect();
  let inputs_data_ref: Vec<&ArrayD<Data>> = inputs_data.iter().collect();

  let outputs_ref_fr: Vec<Vec<&ArrayD<Fr>>> = outputs.iter().map(|o| o.iter().collect()).collect();
  let outputs_ref_fr: Vec<&Vec<&ArrayD<Fr>>> = outputs_ref_fr.iter().collect();

  let mut timing = TimingTree::default();
  let encoded_outputs = graph.encodeOutputs(srs, &models_data_ref, &inputs_data_ref, &outputs_ref_fr, &mut timing);
  let encoded_outputs_ref_tmp: Vec<Vec<&ArrayD<Data>>> = encoded_outputs.iter().map(|v| v.iter().collect()).collect();
  let encoded_outputs_ref: Vec<&Vec<&ArrayD<Data>>> = encoded_outputs_ref_tmp.iter().collect();

  let mut rng = StdRng::from_entropy();
  let proofs = graph.prove(srs, &setups_ref, &models_data_ref, &inputs_data_ref, &encoded_outputs_ref, &mut rng, &mut timing);

  let proof_bytes = serialize_proofs(&proofs).map_err(|e| e.to_string())?;
  Ok(SampleProofV1 { proof_bytes })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::training::compile::{build_training_graph, ForwardOp, TrainingSpec};

  /// Verify the graph `run()` operates on full 4×4 tensors, not scalars.
  /// Traces shapes at every node to confirm nothing is silently 1×1.
  #[test]
  fn test_graph_run_uses_full_4x4_tensors() {
    let spec = TrainingSpec {
      ops: vec![ForwardOp::Linear, ForwardOp::ReLU, ForwardOp::Linear, ForwardOp::ReLU],
    };
    let (graph, models, num_weights) = build_training_graph(&spec);
    assert_eq!(num_weights, 2);

    // Build 4-element vectors and 4×4 weight matrices
    let x: Vec<Fr> = (1..=4).map(|v| Fr::from(v as u64)).collect();
    let x_tensor = ArrayD::from_shape_vec(IxDyn(&[4]), x).unwrap();

    let grad_y: Vec<Fr> = (5..=8).map(|v| Fr::from(v as u64)).collect();
    let upstream_tensor = ArrayD::from_shape_vec(IxDyn(&[4]), grad_y).unwrap();

    let lr_tensor = ArrayD::from_shape_vec(IxDyn(&[1]), vec![Fr::from(1u64)]).unwrap();
    let chal_tensor = ArrayD::from_shape_vec(IxDyn(&[1]), vec![Fr::from(0u64)]).unwrap();

    let w1_data: Vec<Fr> = (1..=16).map(|v| Fr::from(v as u64)).collect();
    let w1_tensor = ArrayD::from_shape_vec(IxDyn(&[4, 4]), w1_data).unwrap();

    let w2_data: Vec<Fr> = (17..=32).map(|v| Fr::from(v as u64)).collect();
    let w2_tensor = ArrayD::from_shape_vec(IxDyn(&[4, 4]), w2_data).unwrap();

    let input_tensors = vec![x_tensor, upstream_tensor, lr_tensor, chal_tensor, w1_tensor, w2_tensor];
    let inputs_ref: Vec<&ArrayD<Fr>> = input_tensors.iter().collect();
    let model_refs: Vec<&ArrayD<Fr>> = models.iter().collect();

    let outputs = graph.run(&inputs_ref, &model_refs);

    // Node 0: LinearForwardBlock  x[4] * w1[4,4] → [4]
    assert_eq!(outputs[0].len(), 1, "LinearForward should produce 1 output");
    assert_eq!(outputs[0][0].shape(), &[4], "LinearForward output must be [4], not [1]");

    // Node 1: IdBasicBlock (ReLU stand-in in forward) → [4]
    assert_eq!(outputs[1][0].shape(), &[4], "Id/ReLU forward output must be [4]");

    // Node 2: LinearForwardBlock  a1[4] * w2[4,4] → [4]
    assert_eq!(outputs[2][0].shape(), &[4], "Second LinearForward output must be [4]");

    // Node 3: IdBasicBlock → [4]
    assert_eq!(outputs[3][0].shape(), &[4], "Second Id output must be [4]");

    // Node 4: ReLUBackwardBlock → [4]
    assert_eq!(outputs[4][0].shape(), &[4], "ReLU backward output must be [4]");

    // Node 5: LinearBackwardBlock → grad_x [4], grad_w [4,4]
    assert_eq!(outputs[5].len(), 2, "LinearBackward should produce 2 outputs");
    assert_eq!(outputs[5][0].shape(), &[4], "LinearBackward grad_x must be [4]");
    assert_eq!(outputs[5][1].shape(), &[4, 4], "LinearBackward grad_w must be [4,4]");

    // Node 6: ReLUBackwardBlock → [4]
    assert_eq!(outputs[6][0].shape(), &[4], "Second ReLU backward output must be [4]");

    // Node 7: LinearBackwardBlock → grad_x [4], grad_w [4,4]
    assert_eq!(outputs[7][0].shape(), &[4], "Second LinearBackward grad_x must be [4]");
    assert_eq!(outputs[7][1].shape(), &[4, 4], "Second LinearBackward grad_w must be [4,4]");

    // Node 8: SGDUpdateBlock w1 → [4,4]
    assert_eq!(outputs[8][0].shape(), &[4, 4], "SGD update for w1 must be [4,4]");

    // Node 9: SGDUpdateBlock w2 → [4,4]
    assert_eq!(outputs[9][0].shape(), &[4, 4], "SGD update for w2 must be [4,4]");

    // Node 10-11: AddBasicBlock (flattened sums for challenge)
    // Node 12: ChallengeBlock → [1]
    assert_eq!(outputs[12][0].shape(), &[1], "ChallengeBlock output must be [1]");

    // Node 13: EqBasicBlock (no run output, just verification)
    println!("All 4×4 tensor shape assertions passed!");
  }
}
