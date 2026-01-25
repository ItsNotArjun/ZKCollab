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

pub fn enforce_linear_vjp(
  x: &ArrayD<Fr>,
  w: &ArrayD<Fr>,
  upstream: &ArrayD<Fr>,
  grad_x: &ArrayD<Fr>,
  grad_w: &ArrayD<Fr>,
) -> Result<(), String> {
  // Supports y = W x with W shape (m, n), x shape (n,), upstream shape (m,)
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

  // grad_x = W^T * upstream
  let mut expected_grad_x = ArrayD::zeros(IxDyn(x1.shape()));
  {
    let mut exp = expected_grad_x.view_mut().into_dimensionality::<Ix1>().unwrap();
    for j in 0..n {
      let mut acc = Fr::from(0u64);
      for i in 0..m {
        acc += w2[(i, j)] * up[i];
      }
      exp[j] = acc;
    }
  }

  // grad_w = upstream * x^T
  let mut expected_grad_w = ArrayD::zeros(IxDyn(w2.shape()));
  {
    let mut exp = expected_grad_w.view_mut().into_dimensionality::<Ix2>().unwrap();
    for i in 0..m {
      for j in 0..n {
        exp[(i, j)] = up[i] * x1[j];
      }
    }
  }

  if !same_shape(&expected_grad_x, grad_x) || !same_shape(&expected_grad_w, grad_w) {
    return Err("linear VJP output shape mismatch".to_string());
  }
  if expected_grad_x != *grad_x {
    return Err("linear VJP grad_x incorrect".to_string());
  }
  if expected_grad_w != *grad_w {
    return Err("linear VJP grad_w incorrect".to_string());
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

pub fn enforce_sgd_update(w_t: &ArrayD<Fr>, grad: &ArrayD<Fr>, lr: Fr, w_next: &ArrayD<Fr>) -> Result<(), String> {
  if w_t.shape() != grad.shape() || w_t.shape() != w_next.shape() {
    return Err("SGD shape mismatch".to_string());
  }
  let scale = Fr::from(FIXED_SCALE);
  for ((&wt, &g), &wn) in w_t.iter().zip(grad.iter()).zip(w_next.iter()) {
    let update = lr * g / scale;
    if wt - update != wn {
      return Err("SGD update constraint violated".to_string());
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
    // y = W x, W=[2], x=[3], upstream=[5] => grad_x = [10], grad_w = [[15]]
    let x = arr1(&[Fr::from(3u64)]).into_dyn();
    let w = arr2(&[[Fr::from(2u64)]]).into_dyn();
    let upstream = arr1(&[Fr::from(5u64)]).into_dyn();
    let grad_x = arr1(&[Fr::from(10u64)]).into_dyn();
    let grad_w = arr2(&[[Fr::from(15u64)]]).into_dyn();
    assert!(enforce_linear_vjp(&x, &w, &upstream, &grad_x, &grad_w).is_ok());
  }

  #[test]
  fn test_linear_vjp_bad_grad() {
    let x = arr1(&[Fr::from(3u64)]).into_dyn();
    let w = arr2(&[[Fr::from(2u64)]]).into_dyn();
    let upstream = arr1(&[Fr::from(5u64)]).into_dyn();
    let grad_x = arr1(&[Fr::from(10u64)]).into_dyn();
    let grad_w = arr2(&[[Fr::from(16u64)]]).into_dyn(); // incorrect
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
    let grad_x = arr1(&[Fr::from(0u64)]).into_dyn(); // should be 2
    assert!(enforce_relu_vjp(&x, &up, &grad_x).is_err());
  }

  #[test]
  fn test_sgd_update_correct() {
    // w_next = w - lr*grad/2^16 with lr=2^16 (i.e., lr=1.0 in fixed-point)
    let w = arr1(&[Fr::from(100u64)]).into_dyn();
    let grad = arr1(&[Fr::from(1u64)]).into_dyn();
    let lr = Fr::from(FIXED_SCALE);
    let w_next = arr1(&[Fr::from(99u64)]).into_dyn();
    assert!(enforce_sgd_update(&w, &grad, lr, &w_next).is_ok());
  }

  #[test]
  fn test_sgd_update_bad() {
    let w = arr1(&[Fr::from(100u64)]).into_dyn();
    let grad = arr1(&[Fr::from(1u64)]).into_dyn();
    let lr = Fr::from(FIXED_SCALE);
    let w_next = arr1(&[Fr::from(100u64)]).into_dyn(); // no update
    assert!(enforce_sgd_update(&w, &grad, lr, &w_next).is_err());
  }

  #[test]
  fn test_randomness_binding() {
    let transcript = b"models|inputs|outputs|step0|backward";
    let c = derive_challenge(transcript);
    assert!(enforce_randomness_binding(c, transcript).is_ok());
    assert!(enforce_randomness_binding(c, b"different").is_err());
  }
}
