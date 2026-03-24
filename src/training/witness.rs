use serde::Deserialize;

/// Witness format for a single SGD step of the 2-layer 4x4 SampleModel.
///
/// This matches the JSON produced by generate_witness.py.
#[derive(Debug, Clone, Deserialize)]
pub struct SampleWitnessV1 {
  pub fixed_scale_k: u32,
  pub x: Vec<i64>,
  pub y_target: Vec<i64>,
  pub lr_scaled: i64,
  // Parameters before/after the SGD step
  pub w1_before: Vec<Vec<i64>>,
  pub b1_before: Vec<i64>,
  pub w2_before: Vec<Vec<i64>>,
  pub b2_before: Vec<i64>,
  pub w1_after: Vec<Vec<i64>>,
  pub b1_after: Vec<i64>,
  pub w2_after: Vec<Vec<i64>>,
  pub b2_after: Vec<i64>,
  // Forward activations
  pub z1: Vec<i64>,
  pub a1: Vec<i64>,
  pub z2: Vec<i64>,
  pub y_pred: Vec<i64>,
  // Gradients wrt activations and input
  pub grad_y: Vec<i64>,
  pub grad_z2: Vec<i64>,
  pub grad_a1: Vec<i64>,
  pub grad_z1: Vec<i64>,
  pub grad_x: Vec<i64>,
  // Per-parameter gradients
  pub grad_w1: Vec<Vec<i64>>,
  pub grad_b1: Vec<i64>,
  pub grad_w2: Vec<Vec<i64>>,
  pub grad_b2: Vec<i64>,
  // Data Binding fields (injected by scripts/inject_root.py)
  #[serde(default)]
  pub merkle_root: Option<String>,
  #[serde(default)]
  pub raw_data: Option<Vec<i64>>,
  #[serde(default)]
  pub merkle_path: Option<Vec<String>>,
  #[serde(default)]
  pub raw_data_length: Option<usize>,
}
