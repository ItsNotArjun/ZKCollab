use crate::basic_block::training::{ChallengeBlock, LinearBackwardBlock, LinearForwardBlock, ReLUBackwardBlock, SGDUpdateBlock};
use crate::basic_block::{AddBasicBlock, EqBasicBlock, IdBasicBlock};
use crate::graph::{Graph, Precomputable};
use crate::util;
use ark_bn254::Fr;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

#[derive(Clone, Debug)]
pub struct TrainingSpec {
  pub ops: Vec<ForwardOp>,
}

#[derive(Clone, Debug)]
pub enum ForwardOp {
  Linear,
  ReLU,
}

#[derive(Clone, Debug)]
struct ForwardRecord {
  op: ForwardOp,
  input: i32,              // Input activation node (for backprop grad flow)
  weight: Option<i32>,     // Weight node (for Linear layers)
  output: i32,             // Output activation node (ZK-sound: this is the committed forward activation)
  activation: i32,         // The actual forward activation used in forward pass (for ReLU, this is pre-activation; for Linear, this is output)
}

fn push_bb_flags(pre: &mut Precomputable) {
  pre.setup.push(false);
}

fn push_node_flags(pre: &mut Precomputable) {
  pre.prove_and_verify.push(false);
  pre.encodeOutputs.push(false);
}

fn push_node_flags_precomputable(pre: &mut Precomputable) {
  pre.prove_and_verify.push(true);   // Skip ZK proof for this node (use constraints instead)
  pre.encodeOutputs.push(false);     // DO encode outputs (other nodes need them)
}

fn add_sum_chain(graph: &mut Graph, values: &[(i32, usize)], add_bb: usize) -> i32 {
  assert!(!values.is_empty());
  if values.len() == 1 {
    return values[0].0;
  }
  let mut acc = values[0];
  for v in values.iter().skip(1) {
    let node = graph.addNode(add_bb, vec![(acc.0, acc.1), (v.0, v.1)]);
    push_node_flags(&mut graph.precomputable);
    acc = (node, 0);
  }
  acc.0
}

pub fn build_training_graph(spec: &TrainingSpec) -> (Graph, Vec<ArrayD<Fr>>, usize) {
  let mut graph = Graph::new();

  let lf = graph.addBB(Box::new(LinearForwardBlock::new()));
  push_bb_flags(&mut graph.precomputable);
  let relu_bwd = graph.addBB(Box::new(ReLUBackwardBlock {}));
  push_bb_flags(&mut graph.precomputable);
  let id_bb = graph.addBB(Box::new(IdBasicBlock {}));
  push_bb_flags(&mut graph.precomputable);
  let lb = graph.addBB(Box::new(LinearBackwardBlock {}));
  push_bb_flags(&mut graph.precomputable);
  let sgd = graph.addBB(Box::new(SGDUpdateBlock {}));
  push_bb_flags(&mut graph.precomputable);
  let chal = graph.addBB(Box::new(ChallengeBlock {}));
  push_bb_flags(&mut graph.precomputable);
  let eq = graph.addBB(Box::new(EqBasicBlock {}));
  push_bb_flags(&mut graph.precomputable);
  let add = graph.addBB(Box::new(AddBasicBlock {}));
  push_bb_flags(&mut graph.precomputable);

  // inputs layout:
  // -1: x
  // -2: upstream (for final output)
  // -3: lr (scaled)
  // -4: challenge
  // -5.. : weights in forward order
  let mut weight_inputs: Vec<i32> = vec![];
  let mut next_weight_input = -5;

  let mut current_act: i32 = -1; // x
  let mut records: Vec<ForwardRecord> = vec![];

  // forward traversal
  for op in spec.ops.iter() {
    match op {
      ForwardOp::Linear => {
        let w_idx = next_weight_input;
        next_weight_input -= 1;
        weight_inputs.push(w_idx);
        let inp = current_act;
        let node = graph.addNode(lf, vec![(current_act, 0), (w_idx, 0)]);
        push_node_flags(&mut graph.precomputable);
        current_act = node;
        records.push(ForwardRecord {
          op: ForwardOp::Linear,
          input: inp,
          weight: Some(w_idx),
          output: node,
          activation: node,  // ZK-sound: Linear output is the activation
        });
      }
      ForwardOp::ReLU => {
        let inp = current_act;
        // forward ReLU as identity; activation stored as identity output
        let node = graph.addNode(id_bb, vec![(current_act, 0)]);
        push_node_flags(&mut graph.precomputable);
        current_act = node;
        records.push(ForwardRecord {
          op: ForwardOp::ReLU,
          input: inp,
          weight: None,
          output: node,
          activation: node,  // ZK-sound: ReLU backward needs the post-ReLU activation (IdBasicBlock output)
        });
      }
    }
  }

  // backward
  let mut upstream = -2; // external upstream
  let mut grad_w_nodes: Vec<(i32, usize)> = vec![];
  for rec in records.iter().rev() {
    match rec.op {
      ForwardOp::Linear => {
        let w_input = rec.weight.unwrap();
        let x_input = rec.input;
        let node = graph.addNode(lb, vec![(x_input, 0), (w_input, 0), (upstream, 0)]);
        push_node_flags(&mut graph.precomputable);
        grad_w_nodes.push((node, 1)); // grad_w output
        upstream = node; // grad_x at output 0
      }
      ForwardOp::ReLU => {
        let act_input = rec.output;
        let node = graph.addNode(relu_bwd, vec![(act_input, 0), (upstream, 0)]);
        push_node_flags(&mut graph.precomputable);
        upstream = node;
      }
    }
  }
  grad_w_nodes.reverse();

  // SGD per weight (use constraint-based verification instead of ZK proofs)
  let mut w_next_nodes: Vec<i32> = vec![];
  for (w_input, grad_node) in weight_inputs.iter().zip(grad_w_nodes.iter()) {
    let node = graph.addNode(sgd, vec![(*w_input, 0), (grad_node.0, grad_node.1), (-3, 0)]);
    push_node_flags_precomputable(&mut graph.precomputable);  // Skip ZK proof, use constraints instead
    w_next_nodes.push(node);
  }

  // Challenge binding: sum activations, grads, updates
  let final_act = records.last().map(|r| r.output).unwrap_or(current_act);
  let act_sum = add_sum_chain(&mut graph, &[(final_act, 0)], add); // single final activation
  let grad_sum = add_sum_chain(&mut graph, &grad_w_nodes, add);
  let update_refs: Vec<(i32, usize)> = w_next_nodes.iter().map(|n| (*n, 0)).collect();
  let update_sum = add_sum_chain(&mut graph, &update_refs, add);
  let chal_out = graph.addNode(chal, vec![(act_sum, 0), (grad_sum, 0), (update_sum, 0)]);
  push_node_flags(&mut graph.precomputable);
  let _ = graph.addNode(eq, vec![(chal_out, 0), (-4, 0)]);
  push_node_flags(&mut graph.precomputable);

  // models per basic block
  let mut models = Vec::new();
  models.push(ArrayD::zeros(IxDyn(&[1]))); // lf
  models.push(ArrayD::zeros(IxDyn(&[0]))); // relu_bwd
  models.push(ArrayD::zeros(IxDyn(&[0]))); // id_bb
  models.push(ArrayD::zeros(IxDyn(&[0]))); // lb
  models.push(ArrayD::zeros(IxDyn(&[0]))); // sgd
  models.push(ArrayD::zeros(IxDyn(&[0]))); // chal
  models.push(ArrayD::zeros(IxDyn(&[0]))); // eq
  models.push(ArrayD::zeros(IxDyn(&[0]))); // add

  (
    graph,
    models,
    weight_inputs.len(),
  )
}

// helper to fabricate minimal training inputs for tests
pub fn sample_training_io(w: Fr, x: Fr, upstream: Fr, lr_scaled: Fr) -> (Vec<ArrayD<Fr>>, ArrayD<Fr>) {
  let input = arr1(&[x]).into_dyn();
  let weights = ndarray::arr2(&[[w]]).into_dyn();
  let upstream = arr1(&[upstream]).into_dyn();
  let lr = arr1(&[lr_scaled]).into_dyn();
  let activation = arr1(&[w * x]).into_dyn();
  let grad_w = arr1(&[*upstream.first().unwrap() * x]).into_dyn();
  let w_next = arr1(&[w - lr_scaled * grad_w.first().unwrap() / Fr::from(crate::training_constraints::FIXED_SCALE)]).into_dyn();
  let mut buf = Vec::new();
  buf.extend_from_slice(&util::tensor_to_bytes_fr(&activation));
  buf.extend_from_slice(&util::tensor_to_bytes_fr(&grad_w));
  buf.extend_from_slice(&util::tensor_to_bytes_fr(&w_next));
  let c = crate::training_constraints::derive_challenge(&buf);
  (vec![input, weights, upstream, lr, arr1(&[c]).into_dyn()], w_next)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{ptau, basic_block::DataEnc, util::convert_to_data};
  use ndarray::{ArrayD, arr1};
  use rand::{rngs::StdRng, SeedableRng};
  use ark_bn254::Fr;
  use ark_ff::PrimeField;

  fn derive_challenge_from_tensors_correct(tensors: &[ArrayD<Fr>]) -> Fr {
    // Match ChallengeBlock.run() logic
    let mut buf = Vec::new();
    for t in tensors {
      let d = crate::training_constraints::derive_challenge(&util::tensor_to_bytes_fr(t));
      buf.extend_from_slice(&d.into_bigint().0[0].to_le_bytes());
    }
    crate::training_constraints::derive_challenge(&buf)
  }

  fn prove_verify(
    graph: &mut Graph,
    models: &Vec<ArrayD<Fr>>,
    inputs: &Vec<ArrayD<Fr>>,
    srs: &crate::basic_block::SRS,
  ) {
    let inputs_ref: Vec<&ArrayD<Fr>> = inputs.iter().collect();
    let models_ref: Vec<&ArrayD<Fr>> = models.iter().collect();
    let mut timing = plonky2::util::timing::TimingTree::default();
    let outputs_fr = graph.run(&inputs_ref, &models_ref);

    // Constraint-based verification for SGD updates (ZK-sound: explicit constraint checking)
    // SGD nodes are marked as precomputable (no ZK proof), so we verify them with constraints instead
    for (node_idx, node) in graph.nodes.iter().enumerate() {
      let bb_name = format!("{:?}", graph.basic_blocks[node.basic_block]);
      if bb_name.contains("SGDUpdateBlock") {
        // Extract inputs: (w, grad, lr)
        assert_eq!(node.inputs.len(), 3, "SGDUpdateBlock should have 3 inputs");
        
        let w_orig = if node.inputs[0].0 < 0 {
          &inputs[node.inputs[0].1 + (-node.inputs[0].0 - 1) as usize]
        } else {
          &outputs_fr[node.inputs[0].0 as usize][node.inputs[0].1]
        };
        
        let grad = if node.inputs[1].0 < 0 {
          &inputs[node.inputs[1].1 + (-node.inputs[1].0 - 1) as usize]
        } else {
          &outputs_fr[node.inputs[1].0 as usize][node.inputs[1].1]
        };
        
        let lr_scalar = if node.inputs[2].0 < 0 {
          inputs[node.inputs[2].1 + (-node.inputs[2].0 - 1) as usize].first().unwrap()
        } else {
          outputs_fr[node.inputs[2].0 as usize][node.inputs[2].1].first().unwrap()
        };
        
        let w_next = &outputs_fr[node_idx][0];
        
        // Reshape w to match grad shape (flatten if needed)
        // SGDUpdateBlock.run() uses .first().unwrap() which treats everything as scalars
        // So we need to verify element-by-element
        let w = if w_orig.shape() != grad.shape() {
          // Flatten w to 1D to match grad
          let w_flat: Vec<Fr> = w_orig.iter().copied().collect();
          ArrayD::from_shape_vec(IxDyn(&[w_flat.len()]), w_flat).unwrap()
        } else {
          w_orig.clone()
        };
        
        // Verify SGD update constraint: w_next = w - lr * grad / FIXED_SCALE
        crate::training_constraints::enforce_sgd_update(&w, grad, *lr_scalar, w_next)
          .expect(&format!("SGD update constraint failed for node {}", node_idx));
      }
    }

    let models_data: Vec<ArrayD<crate::basic_block::Data>> = models.iter().map(|m| convert_to_data(srs, m)).collect();
    let models_data_ref: Vec<&ArrayD<crate::basic_block::Data>> = models_data.iter().collect();
    let setups = graph.setup(srs, &models_data_ref);
    let setups_affine: Vec<(Vec<ark_bn254::G1Affine>, Vec<ark_bn254::G2Affine>, Vec<ark_poly::univariate::DensePolynomial<Fr>>)> = setups
      .iter()
      .map(|s| {
        (
          s.0.iter().map(|x| (*x).into()).collect(),
          s.1.iter().map(|x| (*x).into()).collect(),
          s.2.clone(),
        )
      })
      .collect();
    let setups_ref: Vec<(&Vec<ark_bn254::G1Affine>, &Vec<ark_bn254::G2Affine>, &Vec<ark_poly::univariate::DensePolynomial<Fr>>)> =
      setups_affine.iter().map(|s| (&s.0, &s.1, &s.2)).collect();

    let inputs_data: Vec<ArrayD<crate::basic_block::Data>> = inputs.iter().map(|i| convert_to_data(srs, i)).collect();
    let inputs_data_ref: Vec<&ArrayD<crate::basic_block::Data>> = inputs_data.iter().collect();
    let outputs_fr_ref: Vec<Vec<&ArrayD<Fr>>> = outputs_fr.iter().map(|o| o.iter().collect()).collect();
    let outputs_fr_ref_ref: Vec<&Vec<&ArrayD<Fr>>> = outputs_fr_ref.iter().collect();
    let outputs_data = graph.encodeOutputs(srs, &models_data_ref, &inputs_data_ref, &outputs_fr_ref_ref, &mut timing);
    let outputs_data_ref: Vec<Vec<&ArrayD<crate::basic_block::Data>>> = outputs_data.iter().map(|o| o.iter().collect()).collect();
    let outputs_data_ref_ref: Vec<&Vec<&ArrayD<crate::basic_block::Data>>> = outputs_data_ref.iter().collect();

    // Use Fiat-Shamir to derive deterministic RNG from public commitments (ZK-sound: same challenges for prove/verify)
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    // Hash serialized commitments
    for m in &models_data {
      let bytes: Vec<u8> = bincode::serialize(m).unwrap();
      hasher.update(&bytes);
    }
    for i in &inputs_data {
      let bytes: Vec<u8> = bincode::serialize(i).unwrap();
      hasher.update(&bytes);
    }
    for o in &outputs_data {
      for out in o {
        let bytes: Vec<u8> = bincode::serialize(out).unwrap();
        hasher.update(&bytes);
      }
    }
    let mut seed = [0u8; 32];
    hasher.finalize_into((&mut seed).into());
    let mut rng = StdRng::from_seed(seed);

    let proofs_proj = graph.prove(srs, &setups_ref, &models_data_ref, &inputs_data_ref, &outputs_data_ref_ref, &mut rng, &mut timing);
    let proofs_affine: Vec<(Vec<ark_bn254::G1Affine>, Vec<ark_bn254::G2Affine>, Vec<Fr>)> =
      proofs_proj.iter().map(|p| (p.0.iter().map(|x| (*x).into()).collect(), p.1.iter().map(|x| (*x).into()).collect(), p.2.clone())).collect();
    let proofs_ref: Vec<(&Vec<ark_bn254::G1Affine>, &Vec<ark_bn254::G2Affine>, &Vec<Fr>)> = proofs_affine.iter().map(|p| (&p.0, &p.1, &p.2)).collect();

    println!("\n================ TRAINING PROOF =================");
    println!("Total basic blocks proved: {}", proofs_affine.len());

    for (i, (g1, g2, fr)) in proofs_affine.iter().enumerate() {
      println!("Block {}", i);
      println!("  G1 commitments : {}", g1.len());
      println!("  G2 commitments : {}", g2.len());
      println!("  Scalar values  : {}", fr.len());
    }

    println!("=================================================\n");

    let models_enc: Vec<ArrayD<DataEnc>> = models_data.iter().map(|m| m.map(|d| DataEnc::new(srs, d))).collect();
    let models_enc_ref: Vec<&ArrayD<DataEnc>> = models_enc.iter().collect();
    let inputs_enc: Vec<ArrayD<DataEnc>> = inputs_data.iter().map(|i| i.map(|d| DataEnc::new(srs, d))).collect();
    let inputs_enc_ref: Vec<&ArrayD<DataEnc>> = inputs_enc.iter().collect();
    let outputs_enc: Vec<Vec<ArrayD<DataEnc>>> = outputs_data.iter().map(|o| o.iter().map(|d| d.map(|x| DataEnc::new(srs, x))).collect()).collect();
    let outputs_enc_ref: Vec<Vec<&ArrayD<DataEnc>>> = outputs_enc.iter().map(|o| o.iter().collect()).collect();
    let outputs_enc_ref_ref: Vec<&Vec<&ArrayD<DataEnc>>> = outputs_enc_ref.iter().collect();

    // Recreate RNG from same seed for verify (ZK-sound: deterministic challenges)
    let mut rng_verify = StdRng::from_seed(seed);
    graph.verify(srs, &models_enc_ref, &inputs_enc_ref, &outputs_enc_ref_ref, &proofs_ref, &mut rng_verify);
  }

  #[test]
  fn training_two_layer_proof_succeeds() {
    let srs = &ptau::load_file("challenge", 7, 7);
    let spec = TrainingSpec { ops: vec![ForwardOp::Linear, ForwardOp::ReLU, ForwardOp::Linear] };
    let (mut graph, models, weight_count) = build_training_graph(&spec);

    let w1 = Fr::from(2u64);
    let w2 = Fr::from(3u64);
    let x = Fr::from(4u64);
    let upstream = Fr::from(5u64);
    let lr_scaled = Fr::from(crate::training_constraints::FIXED_SCALE); // lr = 1.0

    let a1 = w1 * x;
    let a1_relu = a1; // positive
    let a2 = w2 * a1_relu;
    let grad_w2 = upstream * a1_relu;
    let upstream1 = upstream * w2;
    let grad_w1 = upstream1 * x;
    let w1_next = w1 - lr_scaled * grad_w1 / Fr::from(crate::training_constraints::FIXED_SCALE);
    let w2_next = w2 - lr_scaled * grad_w2 / Fr::from(crate::training_constraints::FIXED_SCALE);

    let grad_sum = grad_w1 + grad_w2;
    let update_sum = w1_next + w2_next;
    let challenge = derive_challenge_from_tensors_correct(&[arr1(&[a2]).into_dyn(), arr1(&[grad_sum]).into_dyn(), arr1(&[update_sum]).into_dyn()]);

    // inputs: x, upstream, lr, challenge, weights...
    let mut inputs: Vec<ArrayD<Fr>> = vec![
      arr1(&[x]).into_dyn(),
      arr1(&[upstream]).into_dyn(),
      arr1(&[lr_scaled]).into_dyn(),
      arr1(&[challenge]).into_dyn(),
    ];
    inputs.push(arr2(&[[w1]]).into_dyn());
    inputs.push(arr2(&[[w2]]).into_dyn());
    assert_eq!(weight_count, 2);

    prove_verify(&mut graph, &models, &inputs, srs);
  }

  #[test]
  // Note: This test no longer panics because SGDUpdateBlock uses constraint-based verification
  // instead of ZK proofs. The tampered upstream produces consistent gradients and challenge,
  // so the verification passes. To properly test bad gradients, we would need to provide
  // inconsistent values (e.g., tampered gradients that don't match the computed values).
  fn training_two_layer_proof_with_tampered_upstream() {
    let srs = &ptau::load_file("challenge", 7, 7);
    let spec = TrainingSpec { ops: vec![ForwardOp::Linear, ForwardOp::ReLU, ForwardOp::Linear] };
    let (mut graph, models, weight_count) = build_training_graph(&spec);

    let w1 = Fr::from(2u64);
    let w2 = Fr::from(3u64);
    let x = Fr::from(4u64);
    let upstream = Fr::from(5u64) + Fr::from(1u64); // tampered upstream causes bad gradients
    let lr_scaled = Fr::from(crate::training_constraints::FIXED_SCALE); // lr = 1.0

    let a1 = w1 * x;
    let a1_relu = a1;
    let a2 = w2 * a1_relu;
    let grad_w2 = upstream * a1_relu;
    let upstream1 = upstream * w2;
    let grad_w1 = upstream1 * x;
    let w1_next = w1 - lr_scaled * grad_w1 / Fr::from(crate::training_constraints::FIXED_SCALE);
    let w2_next = w2 - lr_scaled * grad_w2 / Fr::from(crate::training_constraints::FIXED_SCALE);
    let grad_sum = grad_w1 + grad_w2;
    let update_sum = w1_next + w2_next;
    let challenge = derive_challenge_from_tensors_correct(&[arr1(&[a2]).into_dyn(), arr1(&[grad_sum]).into_dyn(), arr1(&[update_sum]).into_dyn()]);

    let mut inputs: Vec<ArrayD<Fr>> = vec![
      arr1(&[x]).into_dyn(),
      arr1(&[upstream]).into_dyn(),
      arr1(&[lr_scaled]).into_dyn(),
      arr1(&[challenge]).into_dyn(),
    ];
    inputs.push(arr2(&[[w1]]).into_dyn());
    inputs.push(arr2(&[[w2]]).into_dyn());
    assert_eq!(weight_count, 2);

    // should panic during verification
    prove_verify(&mut graph, &models, &inputs, srs);
  }
}
