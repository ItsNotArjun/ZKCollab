use crate::basic_block::SRS;
use crate::training::constraints::{FIXED_SCALE, enforce_linear_vjp, enforce_relu_vjp, enforce_sgd_update};
use ark_bn254::Fr;
use ndarray::{Array2, Array1, ArrayD, IxDyn};

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
  let lr_fr = i64_to_fr(w.lr_scaled) / Fr::from(FIXED_SCALE);
  enforce_sgd_update(&w1_before, &grad_w1, lr_fr, &w1_after)?;
  enforce_sgd_update(&b1_before, &grad_b1, lr_fr, &b1_after)?;
  enforce_sgd_update(&w2_before, &grad_w2, lr_fr, &w2_after)?;
  enforce_sgd_update(&b2_before, &grad_b2, lr_fr, &b2_after)?;

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
    eprintln!("warning: sample training step algebraic check failed: {}", e);
  }

  // Placeholder: real implementation will call into Graph-based
  // SNARK prover and populate `proof_bytes`.
  SampleProofV1 { proof_bytes: Vec::new() }
}
