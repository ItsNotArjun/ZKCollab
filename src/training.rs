#![allow(dead_code)]
use crate::basic_block::{Data, DataEnc, SRS};
use crate::graph::Graph;
use crate::util::convert_to_data;
use crate::util::fr_to_int;
use crate::training_constraints::{derive_challenge, enforce_linear_vjp, enforce_randomness_binding, enforce_relu_vjp, enforce_sgd_update, FIXED_SCALE};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_poly::univariate::DensePolynomial;
use ark_serialize::{CanonicalSerialize, SerializationError};
use ndarray::{ArrayD, Ix1, Ix2, IxDyn};
use plonky2::util::timing::TimingTree;
use rand::{rngs::StdRng, SeedableRng};
use sha3::{Digest, Keccak256};

pub type ProofTuple = (Vec<G1Affine>, Vec<G2Affine>, Vec<Fr>);

#[derive(Clone, Debug)]
pub struct TrainingProofBundle {
  pub forward: Vec<ProofTuple>,
  pub backward: Vec<ProofTuple>,
  pub optimizer: Vec<ProofTuple>,
  pub randomness_commitment: [u8; 32],
  pub composed_proof: Vec<u8>,
}

fn setup_graph<'a>(
  srs: &SRS,
  graph: &Graph,
  models: &'a Vec<&ArrayD<Fr>>,
  _timing: &mut TimingTree,
) -> (Vec<ArrayD<Data>>, Vec<(Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>)>) {
  let models_data: Vec<ArrayD<Data>> = models.iter().map(|m| convert_to_data(srs, m)).collect();
  let models_ref: Vec<&ArrayD<Data>> = models_data.iter().collect();
  let setups = graph.setup(srs, &models_ref);
  (models_data, setups)
}

fn encode_and_prove(
  srs: &SRS,
  graph: &mut Graph,
  setups: &Vec<(Vec<G1Projective>, Vec<G2Projective>, Vec<DensePolynomial<Fr>>)>,
  models: &Vec<ArrayD<Data>>,
  inputs: &Vec<&ArrayD<Fr>>,
  outputs: &Vec<Vec<ArrayD<Fr>>>,
  rng: &mut StdRng,
  timing: &mut TimingTree,
) -> Vec<ProofTuple> {
  let setups_affine: Vec<(Vec<G1Affine>, Vec<G2Affine>, Vec<DensePolynomial<Fr>>)> = setups
    .iter()
    .map(|s| {
      (
        s.0.iter().map(|x| (*x).into()).collect(),
        s.1.iter().map(|x| (*x).into()).collect(),
        s.2.clone(),
      )
    })
    .collect();
  let setups_ref: Vec<(&Vec<G1Affine>, &Vec<G2Affine>, &Vec<DensePolynomial<Fr>>)> =
    setups_affine.iter().map(|s| (&s.0, &s.1, &s.2)).collect();

  let models_ref: Vec<&ArrayD<Data>> = models.iter().collect();
  let inputs_data: Vec<ArrayD<Data>> = inputs.iter().map(|i| convert_to_data(srs, i)).collect();
  let inputs_ref: Vec<&ArrayD<Data>> = inputs_data.iter().collect();
  let outputs_ref_fr: Vec<Vec<&ArrayD<Fr>>> = outputs.iter().map(|o| o.iter().collect()).collect();
  let outputs_ref_fr: Vec<&Vec<&ArrayD<Fr>>> = outputs_ref_fr.iter().collect();
  let encoded_outputs = graph.encodeOutputs(srs, &models_ref, &inputs_ref, &outputs_ref_fr, timing);
  let encoded_outputs_ref: Vec<Vec<&ArrayD<Data>>> = encoded_outputs.iter().map(|o| o.iter().collect()).collect();
  let encoded_outputs_ref: Vec<&Vec<&ArrayD<Data>>> = encoded_outputs_ref.iter().collect();
  graph.prove(srs, &setups_ref, &models_ref, &inputs_ref, &encoded_outputs_ref, rng, timing)
}

fn commit_randomness(forward: &Vec<ProofTuple>, backward: &Vec<ProofTuple>, optimizer: &Vec<ProofTuple>) -> Result<[u8; 32], SerializationError> {
  let mut hasher = Keccak256::new();
  hasher.update(serialize_proofs(forward)?);
  hasher.update(serialize_proofs(backward)?);
  hasher.update(serialize_proofs(optimizer)?);
  let mut out = [0u8; 32];
  hasher.finalize_into((&mut out).into());
  Ok(out)
}

fn serialize_proofs(proofs: &Vec<ProofTuple>) -> Result<Vec<u8>, SerializationError> {
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

pub fn prove_training_step(
  srs: &SRS,
  forward_graph: &mut Graph,
  backward_graph: &mut Graph,
  optimizer_graph: &mut Graph,
  forward_models: &Vec<&ArrayD<Fr>>,
  backward_models: &Vec<&ArrayD<Fr>>,
  optimizer_models: &Vec<&ArrayD<Fr>>,
  forward_inputs: &Vec<&ArrayD<Fr>>,
  backward_inputs: &Vec<&ArrayD<Fr>>,
  optimizer_inputs: &Vec<&ArrayD<Fr>>,
  rng: &mut StdRng,
) -> Result<TrainingProofBundle, SerializationError> {
  let mut timing = TimingTree::default();

  let forward_outputs = forward_graph.run(forward_inputs, forward_models);
  let backward_outputs = backward_graph.run(backward_inputs, backward_models);
  let optimizer_outputs = optimizer_graph.run(optimizer_inputs, optimizer_models);

  let (forward_models_data, forward_setups) = setup_graph(srs, forward_graph, forward_models, &mut timing);
  let (backward_models_data, backward_setups) = setup_graph(srs, backward_graph, backward_models, &mut timing);
  let (optimizer_models_data, optimizer_setups) = setup_graph(srs, optimizer_graph, optimizer_models, &mut timing);

  let forward_proofs = encode_and_prove(
    srs,
    forward_graph,
    &forward_setups,
    &forward_models_data,
    forward_inputs,
    &forward_outputs,
    rng,
    &mut timing,
  );
  let backward_proofs = encode_and_prove(
    srs,
    backward_graph,
    &backward_setups,
    &backward_models_data,
    backward_inputs,
    &backward_outputs,
    rng,
    &mut timing,
  );
  let optimizer_proofs = encode_and_prove(
    srs,
    optimizer_graph,
    &optimizer_setups,
    &optimizer_models_data,
    optimizer_inputs,
    &optimizer_outputs,
    rng,
    &mut timing,
  );

  let randomness_commitment = commit_randomness(&forward_proofs, &backward_proofs, &optimizer_proofs)?;
  let mut composed_proof = Vec::new();
  composed_proof.extend_from_slice(&serialize_proofs(&forward_proofs)?);
  composed_proof.extend_from_slice(&serialize_proofs(&backward_proofs)?);
  composed_proof.extend_from_slice(&serialize_proofs(&optimizer_proofs)?);
  composed_proof.extend_from_slice(&randomness_commitment);

  Ok(TrainingProofBundle {
    forward: forward_proofs,
    backward: backward_proofs,
    optimizer: optimizer_proofs,
    randomness_commitment,
    composed_proof,
  })
}

#[derive(Clone, Debug)]
pub struct ForwardIR {
  pub input: ArrayD<Fr>,
  pub weights: ArrayD<Fr>,
  pub activation: ArrayD<Fr>,
  pub saved_for_backward: ArrayD<Fr>,
  pub saved_commitment: [u8; 32],
  pub with_relu: bool,
}

#[derive(Clone, Debug)]
pub enum VJPKindIR {
  Linear {
    upstream: ArrayD<Fr>,
    grad_x: ArrayD<Fr>,
    grad_w: ArrayD<Fr>,
  },
  ReLU {
    upstream: ArrayD<Fr>,
    grad_x: ArrayD<Fr>,
  },
}

#[derive(Clone, Debug)]
pub struct VJPBlockIR {
  pub activation: ArrayD<Fr>,
  pub saved_commitment: [u8; 32],
  pub weights: Option<ArrayD<Fr>>,
  pub kind: VJPKindIR,
}

#[derive(Clone, Debug)]
pub struct BackwardIR {
  pub vjp_blocks: Vec<VJPBlockIR>,
}

#[derive(Clone, Debug)]
pub struct SGDUpdateIR {
  pub w_t: ArrayD<Fr>,
  pub grad: ArrayD<Fr>,
  pub lr: Fr,
  pub w_next: ArrayD<Fr>,
}

#[derive(Clone, Debug)]
pub struct OptimizerIR {
  pub updates: Vec<SGDUpdateIR>,
}

#[derive(Clone, Debug)]
pub struct TrainingStepIR {
  pub forward: ForwardIR,
  pub backward: BackwardIR,
  pub optimizer: OptimizerIR,
  pub randomness_challenge: Fr,
}

#[derive(Clone, Debug)]
pub struct TrainingStateCommitment {
  pub params_before: ArrayD<Fr>,
  pub params_after: ArrayD<Fr>,
  pub digest: [u8; 32],
  pub randomness_challenge: Fr,
}

fn tensor_to_bytes(t: &ArrayD<Fr>) -> Result<Vec<u8>, SerializationError> {
  let mut buf = Vec::new();
  for v in t.iter() {
    v.serialize_uncompressed(&mut buf)?;
  }
  Ok(buf)
}

fn matmul_vec(w: &ArrayD<Fr>, x: &ArrayD<Fr>) -> Result<ArrayD<Fr>, String> {
  if w.ndim() != 2 || x.ndim() != 1 {
    return Err("forward matmul shape mismatch".to_string());
  }
  let w2 = w.view().into_dimensionality::<Ix2>().map_err(|_| "W not 2D".to_string())?;
  let x1 = x.view().into_dimensionality::<Ix1>().map_err(|_| "x not 1D".to_string())?;
  if w2.shape()[1] != x1.len() {
    return Err("forward matmul dimension mismatch".to_string());
  }
  let m = w2.shape()[0];
  let n = w2.shape()[1];
  let mut out = ArrayD::zeros(IxDyn(&[m]));
  {
    let mut out1 = out.view_mut().into_dimensionality::<Ix1>().unwrap();
    for i in 0..m {
      let mut acc = Fr::from(0u64);
      for j in 0..n {
        acc += w2[(i, j)] * x1[j];
      }
      out1[i] = acc;
    }
  }
  Ok(out)
}

fn relu_inplace(t: &mut ArrayD<Fr>) {
  t.iter_mut().for_each(|x| {
    if fr_to_int(*x) <= 0 {
      *x = Fr::from(0u64);
    }
  });
}

fn commit_tensor(t: &ArrayD<Fr>) -> Result<[u8; 32], SerializationError> {
  let mut hasher = Keccak256::new();
  hasher.update(tensor_to_bytes(t)?);
  let mut out = [0u8; 32];
  hasher.finalize_into((&mut out).into());
  Ok(out)
}

fn enforce_forward_step(forward: &ForwardIR) -> Result<[u8; 32], String> {
  let linear = matmul_vec(&forward.weights, &forward.input)?;
  let mut activation = linear.clone();
  if forward.with_relu {
    relu_inplace(&mut activation);
  }
  if activation != forward.activation {
    return Err("forward activation incorrect".to_string());
  }
  let commitment = commit_tensor(&forward.saved_for_backward).map_err(|e| e.to_string())?;
  if commitment != forward.saved_commitment {
    return Err("forward activation commitment mismatch".to_string());
  }
  Ok(commitment)
}

fn enforce_vjp_block(block: &VJPBlockIR, forward_weights: &ArrayD<Fr>, expected_commitment: &[u8; 32]) -> Result<Option<ArrayD<Fr>>, String> {
  if &block.saved_commitment != expected_commitment {
    return Err("backward activation not tied to forward output".to_string());
  }
  match &block.kind {
    VJPKindIR::Linear { upstream, grad_x, grad_w } => {
      let w = block.weights.as_ref().ok_or("linear VJP missing weights")?;
      if w != forward_weights {
        return Err("backward weights are inconsistent with forward parameters".to_string());
      }
      enforce_linear_vjp(&block.activation, w, upstream, grad_x, grad_w)?;
      Ok(Some(grad_w.clone()))
    }
    VJPKindIR::ReLU { upstream, grad_x } => {
      enforce_relu_vjp(&block.activation, upstream, grad_x)?;
      Ok(None)
    }
  }
}

fn enforce_optimizer_update(update: &SGDUpdateIR) -> Result<(), String> {
  enforce_sgd_update(&update.w_t, &update.grad, update.lr, &update.w_next)
}

pub fn build_training_transcript(step: &TrainingStepIR) -> Result<Vec<u8>, SerializationError> {
  let mut buf = Vec::new();
  buf.extend_from_slice(&tensor_to_bytes(&step.forward.input)?);
  buf.extend_from_slice(&tensor_to_bytes(&step.forward.weights)?);
  buf.extend_from_slice(&tensor_to_bytes(&step.forward.activation)?);
  buf.extend_from_slice(&tensor_to_bytes(&step.forward.saved_for_backward)?);
  buf.extend_from_slice(&step.forward.saved_commitment);
  for block in &step.backward.vjp_blocks {
    buf.extend_from_slice(&tensor_to_bytes(&block.activation)?);
    buf.extend_from_slice(&block.saved_commitment);
    if let Some(w) = &block.weights {
      buf.extend_from_slice(&tensor_to_bytes(w)?);
    }
    match &block.kind {
      VJPKindIR::Linear { upstream, grad_x, grad_w } => {
        buf.extend_from_slice(&tensor_to_bytes(upstream)?);
        buf.extend_from_slice(&tensor_to_bytes(grad_x)?);
        buf.extend_from_slice(&tensor_to_bytes(grad_w)?);
      }
      VJPKindIR::ReLU { upstream, grad_x } => {
        buf.extend_from_slice(&tensor_to_bytes(upstream)?);
        buf.extend_from_slice(&tensor_to_bytes(grad_x)?);
      }
    }
  }
  for upd in &step.optimizer.updates {
    buf.extend_from_slice(&tensor_to_bytes(&upd.w_t)?);
    buf.extend_from_slice(&tensor_to_bytes(&upd.grad)?);
    buf.extend_from_slice(&tensor_to_bytes(&upd.w_next)?);
  }
  Ok(buf)
}

fn commit_state(params_before: &ArrayD<Fr>, params_after: &ArrayD<Fr>, randomness_challenge: Fr) -> Result<[u8; 32], SerializationError> {
  let mut hasher = Keccak256::new();
  hasher.update(tensor_to_bytes(params_before)?);
  hasher.update(tensor_to_bytes(params_after)?);
  let mut challenge_bytes = Vec::new();
  randomness_challenge.serialize_uncompressed(&mut challenge_bytes)?;
  hasher.update(challenge_bytes);
  let mut out = [0u8; 32];
  hasher.finalize_into((&mut out).into());
  Ok(out)
}

fn apply_sgd(w_t: &ArrayD<Fr>, grad: &ArrayD<Fr>, lr: Fr) -> ArrayD<Fr> {
  let scale = Fr::from(FIXED_SCALE);
  let mut next = w_t.clone();
  for ((next_cell, wt), g) in next.iter_mut().zip(w_t.iter()).zip(grad.iter()) {
    *next_cell = *wt - lr * *g / scale;
  }
  next
}

pub fn prove_training_step_ir(step: &TrainingStepIR) -> Result<TrainingStateCommitment, String> {
  if step.backward.vjp_blocks.is_empty() {
    return Err("training step missing backward blocks".to_string());
  }
  let optimizer_update = step.optimizer.updates.first().ok_or("training step missing optimizer updates".to_string())?;
  if step.forward.weights != optimizer_update.w_t {
    return Err("forward parameters and optimizer input disagree".to_string());
  }

  let forward_commitment = enforce_forward_step(&step.forward)?;
  let mut last_grad_w: Option<ArrayD<Fr>> = None;
  for block in &step.backward.vjp_blocks {
    if let Some(gw) = enforce_vjp_block(block, &step.forward.weights, &forward_commitment)? {
      last_grad_w = Some(gw);
    }
  }

  if let Some(gw) = &last_grad_w {
    if gw != &optimizer_update.grad {
      return Err("optimizer gradient does not match backward gradient".to_string());
    }
  } else {
    return Err("no gradient produced for optimizer update".to_string());
  }

  enforce_optimizer_update(optimizer_update)?;

  let transcript = build_training_transcript(step).map_err(|e| e.to_string())?;
  enforce_randomness_binding(step.randomness_challenge, &transcript)?;
  let digest = commit_state(&optimizer_update.w_t, &optimizer_update.w_next, step.randomness_challenge).map_err(|e| e.to_string())?;

  Ok(TrainingStateCommitment {
    params_before: optimizer_update.w_t.clone(),
    params_after: optimizer_update.w_next.clone(),
    digest,
    randomness_challenge: step.randomness_challenge,
  })
}

pub fn inference_as_training(input: ArrayD<Fr>, weights: ArrayD<Fr>, with_relu: bool) -> Result<TrainingStepIR, String> {
  let mut activation = matmul_vec(&weights, &input)?;
  if with_relu {
    relu_inplace(&mut activation);
  }
  let commitment = commit_tensor(&input).map_err(|e| e.to_string())?;
  let zero_grad_output = activation.map(|_| Fr::from(0u64));
  let zero_grad_input = input.map(|_| Fr::from(0u64));
  let zero_grad_w = weights.map(|_| Fr::from(0u64));
  let w_next = apply_sgd(&weights, &zero_grad_w, Fr::from(0u64));
  let update = SGDUpdateIR { w_t: weights.clone(), grad: zero_grad_w.clone(), lr: Fr::from(0u64), w_next };
  let saved_for_backward = input.clone();
  let step = TrainingStepIR {
    forward: ForwardIR {
      input: input.clone(),
      weights: weights.clone(),
      activation,
      saved_for_backward: saved_for_backward.clone(),
      saved_commitment: commitment,
      with_relu,
    },
    backward: BackwardIR {
      vjp_blocks: vec![VJPBlockIR {
        activation: saved_for_backward.clone(),
        saved_commitment: commitment,
        weights: Some(weights.clone()),
        kind: VJPKindIR::Linear {
          upstream: zero_grad_output.clone(),
          grad_x: zero_grad_input.clone(),
          grad_w: zero_grad_w.clone(),
        },
      }],
    },
    optimizer: OptimizerIR { updates: vec![update] },
    randomness_challenge: derive_challenge(&[]),
  };
  Ok(step)
}

#[cfg(test)]
mod training_ir_tests {
  use super::*;
  use ndarray::{arr1, arr2};

  fn build_linear_relu_step(w_val: u64, input_val: u64, upstream_val: u64, lr_scale: u64) -> TrainingStepIR {
    let input = arr1(&[Fr::from(input_val)]).into_dyn();
    let weights = arr2(&[[Fr::from(w_val)]]).into_dyn();
    let activation = arr1(&[Fr::from(w_val) * Fr::from(input_val)]).into_dyn();
    let saved_commitment = commit_tensor(&input).unwrap();
    let upstream = arr1(&[Fr::from(upstream_val)]).into_dyn();
    let grad_x = arr1(&[Fr::from(w_val) * Fr::from(upstream_val)]).into_dyn();
    let grad_w = arr2(&[[Fr::from(input_val) * Fr::from(upstream_val)]]).into_dyn();
    let saved_for_backward = input.clone();
    let update = SGDUpdateIR { w_t: weights.clone(), grad: grad_w.clone(), lr: Fr::from(lr_scale), w_next: apply_sgd(&weights, &grad_w, Fr::from(lr_scale)) };
    let mut step = TrainingStepIR {
      forward: ForwardIR {
        input: input.clone(),
        weights: weights.clone(),
        activation,
        saved_for_backward: saved_for_backward.clone(),
        saved_commitment,
        with_relu: true,
      },
      backward: BackwardIR {
        vjp_blocks: vec![VJPBlockIR {
          activation: saved_for_backward.clone(),
          saved_commitment,
          weights: Some(weights.clone()),
          kind: VJPKindIR::Linear {
            upstream: upstream.clone(),
            grad_x,
            grad_w,
          },
        }],
      },
      optimizer: OptimizerIR { updates: vec![update] },
      randomness_challenge: Fr::from(0u64),
    };
    let transcript = build_training_transcript(&step).unwrap();
    step.randomness_challenge = derive_challenge(&transcript);
    step
  }

  #[test]
  fn training_step_succeeds() {
    let step = build_linear_relu_step(1, 1, 1, FIXED_SCALE);
    let result = prove_training_step_ir(&step).unwrap();
    assert_eq!(result.params_before, step.optimizer.updates[0].w_t);
    assert_eq!(result.params_after, step.optimizer.updates[0].w_next);
  }

  #[test]
  fn training_step_fails_on_bad_gradients() {
    let mut step = build_linear_relu_step(1, 1, 1, FIXED_SCALE);
    if let VJPKindIR::Linear { ref mut grad_w, .. } = step.backward.vjp_blocks[0].kind {
      grad_w[[0, 0]] = Fr::from(2u64);
    }
    let transcript = build_training_transcript(&step).unwrap();
    step.randomness_challenge = derive_challenge(&transcript);
    assert!(prove_training_step_ir(&step).is_err());
  }

  #[test]
  fn training_step_fails_on_bad_optimizer_math() {
    let mut step = build_linear_relu_step(1, 1, 1, FIXED_SCALE);
    step.optimizer.updates[0].w_next = step.optimizer.updates[0].w_t.clone(); // no update applied
    let transcript = build_training_transcript(&step).unwrap();
    step.randomness_challenge = derive_challenge(&transcript);
    assert!(prove_training_step_ir(&step).is_err());
  }

  #[test]
  fn training_step_rejects_randomness_replay() {
    let mut step = build_linear_relu_step(1, 1, 1, FIXED_SCALE);
    step.randomness_challenge = Fr::from(123u64);
    assert!(prove_training_step_ir(&step).is_err());
  }
}
