use crate::basic_block::training::{ChallengeBlock, LinearBackwardBlock, LinearForwardBlock, ReLUBackwardBlock, SGDUpdateBlock};
use crate::basic_block::{AddBasicBlock, EqBasicBlock, IdBasicBlock};
use crate::graph::{Graph, Precomputable};
use ark_bn254::Fr;
use ndarray::{ArrayD, IxDyn};

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
  input: i32,
  weight: Option<i32>,
  output: i32,
  activation: i32,
}

fn push_bb_flags(pre: &mut Precomputable) {
  pre.setup.push(false);
}

fn push_node_flags(pre: &mut Precomputable) {
  pre.prove_and_verify.push(false);
  pre.encodeOutputs.push(false);
}

fn push_node_flags_precomputable(pre: &mut Precomputable) {
  pre.prove_and_verify.push(true);
  pre.encodeOutputs.push(false);
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

  let mut weight_inputs: Vec<i32> = vec![];
  let mut next_weight_input = -5;

  let mut current_act: i32 = -1;
  let mut records: Vec<ForwardRecord> = vec![];

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
          activation: node,
        });
      }
      ForwardOp::ReLU => {
        let inp = current_act;
        let node = graph.addNode(id_bb, vec![(current_act, 0)]);
        push_node_flags(&mut graph.precomputable);
        current_act = node;
        records.push(ForwardRecord {
          op: ForwardOp::ReLU,
          input: inp,
          weight: None,
          output: node,
          activation: node,
        });
      }
    }
  }

  let mut upstream = -2;
  let mut grad_w_nodes: Vec<(i32, usize)> = vec![];
  for rec in records.iter().rev() {
    match rec.op {
      ForwardOp::Linear => {
        let w_input = rec.weight.unwrap();
        let x_input = rec.input;
        let node = graph.addNode(lb, vec![(x_input, 0), (w_input, 0), (upstream, 0)]);
        push_node_flags(&mut graph.precomputable);
        grad_w_nodes.push((node, 1));
        upstream = node;
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

  let mut w_next_nodes: Vec<i32> = vec![];
  for (w_input, grad_node) in weight_inputs.iter().zip(grad_w_nodes.iter()) {
    let node = graph.addNode(sgd, vec![(*w_input, 0), (grad_node.0, grad_node.1), (-3, 0)]);
    push_node_flags_precomputable(&mut graph.precomputable);
    w_next_nodes.push(node);
  }

  let final_act = records.last().map(|r| r.output).unwrap_or(current_act);
  let act_sum = add_sum_chain(&mut graph, &[(final_act, 0)], add);
  let grad_sum = add_sum_chain(&mut graph, &grad_w_nodes, add);
  let update_refs: Vec<(i32, usize)> = w_next_nodes.iter().map(|n| (*n, 0)).collect();
  let update_sum = add_sum_chain(&mut graph, &update_refs, add);
  let chal_out = graph.addNode(chal, vec![(act_sum, 0), (grad_sum, 0), (update_sum, 0)]);
  push_node_flags(&mut graph.precomputable);
  let _ = graph.addNode(eq, vec![(chal_out, 0), (-4, 0)]);
  push_node_flags(&mut graph.precomputable);

  let mut models = Vec::new();
  models.push(ArrayD::zeros(IxDyn(&[1])));
  models.push(ArrayD::zeros(IxDyn(&[0])));
  models.push(ArrayD::zeros(IxDyn(&[0])));
  models.push(ArrayD::zeros(IxDyn(&[0])));
  models.push(ArrayD::zeros(IxDyn(&[0])));
  models.push(ArrayD::zeros(IxDyn(&[0])));
  models.push(ArrayD::zeros(IxDyn(&[0])));
  models.push(ArrayD::zeros(IxDyn(&[0])));

  (
    graph,
    models,
    weight_inputs.len(),
  )
}
// NOTE: This module now only provides the generic training graph
// builder. Sample- or model-specific training circuits should be
// constructed by callers (e.g. for the 4x4 MLP) using
// `build_training_graph` with an appropriate `TrainingSpec`.

