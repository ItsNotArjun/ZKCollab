use crate::util::fr_to_int;
use ark_bn254::Fr;
use ark_ff::PrimeField;
use ndarray::{ArrayD, Ix1, Ix2, IxDyn};
use sha3::{Digest, Keccak256};

pub const FIXED_SCALE_K: u32 = 16;
pub const FIXED_SCALE: u64 = 1u64 << FIXED_SCALE_K;

pub fn derive_challenge(transcript: &[u8]) -> Fr {
  let mut hasher = Keccak256::new();
  hasher.update(transcript);
  let digest = hasher.finalize();
  Fr::from_le_bytes_mod_order(&digest)
}

fn same_shape(a: &ArrayD<Fr>, b: &ArrayD<Fr>) -> bool {
  a.shape() == b.shape()
}

pub fn enforce_randomness_binding(challenge: Fr, transcript: &[u8]) -> Result<(), String> {
  let derived = derive_challenge(transcript);
  if challenge != derived {
    return Err("randomness binding failed: challenge mismatch".to_string());
  }
  Ok(())
}

/// Convert a field element back to i64. Assumes the encoded integer
/// fits within ±2^62.  Small positives sit in limb 0; small negatives
/// are stored as p − |v| and detected via the upper limbs.
fn fr_to_i64(x: Fr) -> i64 {
  let b = x.into_bigint();
  if b.0[1] == 0 && b.0[2] == 0 && b.0[3] == 0 {
    b.0[0] as i64
  } else {
    let neg = (-x).into_bigint();
    -(neg.0[0] as i64)
  }
}

/// Round a Q32 product back to Q16 using banker's rounding (round-half-to-even).
///
/// This divides `prod` by `FIXED_SCALE` with deterministic tie-breaking:
/// when the remainder is exactly half the scale, the result is rounded to
/// the nearest even integer.  Both the Python witness generator and this
/// Rust verifier use the identical algorithm, so the comparison can be
/// exact (no tolerance required).
pub(crate) fn round_fixed_point(prod: i64) -> i64 {
  let scale = FIXED_SCALE as i64;
  let quotient = prod / scale; // truncation toward zero (Rust semantics)
  let remainder = prod - quotient * scale; // same sign as prod
  let abs_rem = remainder.abs();
  let half = scale / 2; // 32768 for FIXED_SCALE = 2^16
  if abs_rem < half {
    quotient
  } else if abs_rem > half {
    if remainder > 0 { quotient + 1 } else { quotient - 1 }
  } else {
    // Tie: round to nearest even
    if quotient % 2 == 0 {
      quotient
    } else if remainder > 0 {
      quotient + 1
    } else {
      quotient - 1
    }
  }
}

pub fn enforce_linear_vjp(
  x: &ArrayD<Fr>,
  w: &ArrayD<Fr>,
  upstream: &ArrayD<Fr>,
  grad_x: &ArrayD<Fr>,
  grad_w: &ArrayD<Fr>,
) -> Result<(), String> {
  if x.ndim() != 1 || upstream.ndim() != 1 || w.ndim() != 2 {
    return Err("linear VJP shape mismatch".to_string());
  }
  let w2 = w.view().into_dimensionality::<Ix2>().map_err(|_| "W not 2D")?;
  let x1 = x.view().into_dimensionality::<Ix1>().map_err(|_| "x not 1D")?;
  let up = upstream.view().into_dimensionality::<Ix1>().map_err(|_| "upstream not 1D")?;
  let m = w2.shape()[0];
  let n = w2.shape()[1];
  if x1.len() != n || up.len() != m {
    return Err("linear VJP dimension mismatch".to_string());
  }

  // grad_x = W^T @ upstream, rescaled from Q32 to Q16 using banker's rounding.
  // The witness must use the identical round_fixed_point algorithm so this
  // check is an exact equality (no tolerance).
  if !same_shape(&ArrayD::zeros(IxDyn(x1.shape())), grad_x) {
    return Err("linear VJP grad_x shape mismatch".to_string());
  }
  let gx1 = grad_x.view().into_dimensionality::<Ix1>().map_err(|_| "grad_x not 1D")?;
  for j in 0..n {
    let mut acc: i64 = 0;
    for i in 0..m {
      acc += fr_to_i64(w2[(i, j)]) * fr_to_i64(up[i]);
    }
    let expected = round_fixed_point(acc);
    let actual = fr_to_i64(gx1[j]);
    if expected != actual {
      return Err(format!(
        "linear VJP grad_x incorrect at index {}: expected {} got {}",
        j, expected, actual
      ));
    }
  }

  // grad_w = upstream ⊗ x, rescaled from Q32 to Q16 using banker's rounding.
  if !same_shape(&ArrayD::zeros(IxDyn(w2.shape())), grad_w) {
    return Err("linear VJP grad_w shape mismatch".to_string());
  }
  let gw2 = grad_w.view().into_dimensionality::<Ix2>().map_err(|_| "grad_w not 2D")?;
  for i in 0..m {
    for j in 0..n {
      let prod = fr_to_i64(up[i]) * fr_to_i64(x1[j]);
      let expected = round_fixed_point(prod);
      let actual = fr_to_i64(gw2[(i, j)]);
      if expected != actual {
        return Err(format!(
          "linear VJP grad_w incorrect at [{},{}]: expected {} got {}",
          i, j, expected, actual
        ));
      }
    }
  }

  Ok(())
}

pub fn enforce_relu_vjp(x: &ArrayD<Fr>, upstream: &ArrayD<Fr>, grad_x: &ArrayD<Fr>) -> Result<(), String> {
  if x.shape() != upstream.shape() || x.shape() != grad_x.shape() {
    return Err("ReLU VJP shape mismatch".to_string());
  }
  for ((&xi, &u), &gx) in x.iter().zip(upstream.iter()).zip(grad_x.iter()) {
    let active = fr_to_int(xi) > 0;
    let expected = if active { u } else { Fr::from(0u64) };
    if expected != gx {
      return Err("ReLU VJP grad_x incorrect".to_string());
    }
  }
  Ok(())
}

pub fn enforce_sgd_update(w_t: &ArrayD<Fr>, grad: &ArrayD<Fr>, lr_scaled: i64, w_next: &ArrayD<Fr>) -> Result<(), String> {
  if w_t.shape() != grad.shape() || w_t.shape() != w_next.shape() {
    return Err("SGD shape mismatch".to_string());
  }
  for ((&wt, &g), &wn) in w_t.iter().zip(grad.iter()).zip(w_next.iter()) {
    let wt_i = fr_to_i64(wt);
    let g_i = fr_to_i64(g);
    let wn_i = fr_to_i64(wn);
    let update = round_fixed_point(lr_scaled * g_i);
    let expected = wt_i - update;
    if expected != wn_i {
      return Err(format!(
        "SGD update constraint violated: expected {} got {}",
        expected, wn_i
      ));
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::{arr1, arr2};

  #[test]
  fn test_linear_vjp_correct() {
    // Values are Q16-scaled: real values 3, 2, 5 → 3*2^16, 2*2^16, 5*2^16
    let s = FIXED_SCALE;
    let x = arr1(&[Fr::from(3 * s)]).into_dyn();
    let w = arr2(&[[Fr::from(2 * s)]]).into_dyn();
    let upstream = arr1(&[Fr::from(5 * s)]).into_dyn();
    // grad_x = W^T @ upstream / FIXED_SCALE = (2*s)*(5*s)/s = 10*s
    let grad_x = arr1(&[Fr::from(10 * s)]).into_dyn();
    // grad_w = upstream * x / FIXED_SCALE = (5*s)*(3*s)/s = 15*s
    let grad_w = arr2(&[[Fr::from(15 * s)]]).into_dyn();
    assert!(enforce_linear_vjp(&x, &w, &upstream, &grad_x, &grad_w).is_ok());
  }

  #[test]
  fn test_linear_vjp_bad_grad() {
    let s = FIXED_SCALE;
    let x = arr1(&[Fr::from(3 * s)]).into_dyn();
    let w = arr2(&[[Fr::from(2 * s)]]).into_dyn();
    let upstream = arr1(&[Fr::from(5 * s)]).into_dyn();
    let grad_x = arr1(&[Fr::from(10 * s)]).into_dyn();
    // Wrong by one Q16 unit (s): should be 15*s, here 16*s
    let grad_w = arr2(&[[Fr::from(16 * s)]]).into_dyn();
    assert!(enforce_linear_vjp(&x, &w, &upstream, &grad_x, &grad_w).is_err());
  }

  #[test]
  fn test_relu_vjp() {
    let x = arr1(&[Fr::from(3u64), Fr::from(0u64), Fr::from(7u64), Fr::from(1u64)]).into_dyn();
    let up = arr1(&[Fr::from(2u64), Fr::from(3u64), Fr::from(5u64), Fr::from(11u64)]).into_dyn();
    let grad_x = arr1(&[Fr::from(2u64), Fr::from(0u64), Fr::from(5u64), Fr::from(11u64)]).into_dyn();
    assert!(enforce_relu_vjp(&x, &up, &grad_x).is_ok());
  }

  #[test]
  fn test_relu_vjp_bad() {
    let x = arr1(&[Fr::from(3u64)]).into_dyn();
    let up = arr1(&[Fr::from(2u64)]).into_dyn();
    let grad_x = arr1(&[Fr::from(0u64)]).into_dyn();
    assert!(enforce_relu_vjp(&x, &up, &grad_x).is_err());
  }

  #[test]
  fn test_sgd_update_correct() {
    // w=100*S, grad=1*S, lr_scaled=S (lr=1.0) → update = S*S/S = S
    // w_next = 100*S - S = 99*S
    let s = FIXED_SCALE;
    let w = arr1(&[Fr::from(100 * s)]).into_dyn();
    let grad = arr1(&[Fr::from(1 * s)]).into_dyn();
    let lr_scaled = s as i64;
    let w_next = arr1(&[Fr::from(99 * s)]).into_dyn();
    assert!(enforce_sgd_update(&w, &grad, lr_scaled, &w_next).is_ok());
  }

  #[test]
  fn test_sgd_update_bad() {
    let s = FIXED_SCALE;
    let w = arr1(&[Fr::from(100 * s)]).into_dyn();
    let grad = arr1(&[Fr::from(1 * s)]).into_dyn();
    let lr_scaled = s as i64;
    let w_next = arr1(&[Fr::from(100 * s)]).into_dyn(); // unchanged = wrong
    assert!(enforce_sgd_update(&w, &grad, lr_scaled, &w_next).is_err());
  }

  #[test]
  fn test_randomness_binding() {
    let transcript = b"models|inputs|outputs|step0|backward";
    let c = derive_challenge(transcript);
    assert!(enforce_randomness_binding(c, transcript).is_ok());
    assert!(enforce_randomness_binding(c, b"different").is_err());
  }

  #[test]
  fn test_round_fixed_point_basic() {
    let s = FIXED_SCALE as i64;
    // Exact multiples: no rounding needed
    assert_eq!(round_fixed_point(2 * s * s), 2 * s);
    assert_eq!(round_fixed_point(-3 * s * s), -3 * s);
    // Below half: round toward zero
    assert_eq!(round_fixed_point(s + s / 2 - 1), 1);
    assert_eq!(round_fixed_point(-(s + s / 2 - 1)), -1);
    // Above half: round away from zero
    assert_eq!(round_fixed_point(s + s / 2 + 1), 2);
    assert_eq!(round_fixed_point(-(s + s / 2 + 1)), -2);
  }

  #[test]
  fn test_round_fixed_point_ties_to_even() {
    let s = FIXED_SCALE as i64;
    // Tie at 0.5: quotient=0 (even) → stay at 0
    assert_eq!(round_fixed_point(s / 2), 0);
    // Tie at 1.5: quotient=1 (odd) → round up to 2
    assert_eq!(round_fixed_point(s + s / 2), 2);
    // Tie at 2.5: quotient=2 (even) → stay at 2
    assert_eq!(round_fixed_point(2 * s + s / 2), 2);
    // Tie at -0.5: quotient=0 (even) → stay at 0
    assert_eq!(round_fixed_point(-(s / 2)), 0);
    // Tie at -1.5: quotient=-1 (odd) → round to -2
    assert_eq!(round_fixed_point(-(s + s / 2)), -2);
  }
}
