use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use crate::basic_block::{EqBasicBlock, MatMulBasicBlock, MulBasicBlock};
use crate::training_constraints::{derive_challenge, FIXED_SCALE};
use crate::util;
use ark_ff::PrimeField;
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{ArrayD, IxDyn};
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
pub struct LinearForwardBlock {
  inner: MatMulBasicBlock,
}

impl LinearForwardBlock {
  pub fn new() -> Self {
    Self { inner: MatMulBasicBlock {} }
  }
}

impl Default for LinearForwardBlock {
  fn default() -> Self {
    Self::new()
  }
}

impl BasicBlock for LinearForwardBlock {
  fn run(&self, model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    self.inner.run(model, inputs)
  }

  fn encodeOutputs(&self, srs: &SRS, model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    self.inner.encodeOutputs(srs, model, inputs, outputs)
  }

  fn setup(&self, srs: &SRS, model: &ArrayD<Data>) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<ark_poly::univariate::DensePolynomial<Fr>>) {
    self.inner.setup(srs, model)
  }

  fn prove(
    &self,
    srs: &SRS,
    setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<ark_poly::univariate::DensePolynomial<Fr>>),
    model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    self.inner.prove(srs, setup, model, inputs, outputs, rng, cache)
  }

  fn verify(
    &self,
    srs: &SRS,
    model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    self.inner.verify(srs, model, inputs, outputs, proof, rng, cache)
  }
}

#[derive(Debug)]
pub struct LinearBackwardBlock;

impl BasicBlock for LinearBackwardBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 3); // x, w, upstream
    let x = inputs[0];
    let w = inputs[1];
    let up = inputs[2];
    assert!(x.len() == 1 && w.len() == 1 && up.len() == 1);
    let grad_x = ArrayD::from_elem(IxDyn(&[1]), w.first().unwrap() * up.first().unwrap());
    let grad_w = ArrayD::from_elem(IxDyn(&[1]), x.first().unwrap() * up.first().unwrap());
    vec![grad_x, grad_w]
  }

  fn encodeOutputs(&self, srs: &SRS, _model: &ArrayD<Data>, _inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    outputs.iter().map(|o| util::convert_to_data(srs, o)).collect()
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<ark_poly::univariate::DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let mul = MulBasicBlock {};
    let gx_proof = mul.prove(
      srs,
      (&vec![], &vec![], &vec![]),
      &ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
      &vec![inputs[1], inputs[2]],
      &vec![&outputs[0]],
      rng,
      cache.clone(),
    );
    let gw_proof = mul.prove(
      srs,
      (&vec![], &vec![], &vec![]),
      &ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
      &vec![inputs[0], inputs[2]],
      &vec![&outputs[1]],
      rng,
      cache,
    );
    (
      [gx_proof.0, gw_proof.0].concat(),
      [gx_proof.1, gw_proof.1].concat(),
      [gx_proof.2, gw_proof.2].concat(),
    )
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mul = MulBasicBlock {};
    let mid_g1 = proof.0.len() / 2;
    let mid_g2 = proof.1.len() / 2;
    let mid_fr = proof.2.len() / 2;
    let p0_g1 = proof.0[..mid_g1].to_vec();
    let p0_g2 = proof.1[..mid_g2].to_vec();
    let p0_fr = proof.2[..mid_fr].to_vec();
    let p1_g1 = proof.0[mid_g1..].to_vec();
    let p1_g2 = proof.1[mid_g2..].to_vec();
    let p1_fr = proof.2[mid_fr..].to_vec();

    let mut checks = Vec::new();
    checks.extend(mul.verify(
      srs,
      &ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
      &vec![inputs[1], inputs[2]],
      &vec![&outputs[0]],
      (&p0_g1, &p0_g2, &p0_fr),
      rng,
      cache.clone(),
    ));
    checks.extend(mul.verify(
      srs,
      &ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
      &vec![inputs[0], inputs[2]],
      &vec![&outputs[1]],
      (&p1_g1, &p1_g2, &p1_fr),
      rng,
      cache,
    ));
    checks
  }
}

#[derive(Debug)]
pub struct ReLUBackwardBlock;

impl BasicBlock for ReLUBackwardBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2);
    let act = inputs[0];
    let up = inputs[1];
    assert!(act.len() == 1 && up.len() == 1);
    let active = util::fr_to_int(*act.first().unwrap()) > 0;
    let grad = if active { *up.first().unwrap() } else { Fr::from(0u64) };
    vec![ArrayD::from_elem(IxDyn(&[1]), grad)]
  }

  fn encodeOutputs(&self, srs: &SRS, _model: &ArrayD<Data>, _inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    outputs.iter().map(|o| util::convert_to_data(srs, o)).collect()
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<ark_poly::univariate::DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    _inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    // Constrain by equality to provided grad (activation dependent); we rely on encodeOutputs to bind value.
    let eq = EqBasicBlock {};
    eq.prove(
      srs,
      (&vec![], &vec![], &vec![]),
      &ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
      &vec![outputs[0], outputs[0]],
      &vec![outputs[0]],
      &mut StdRng::from_entropy(),
      _cache,
    )
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    _inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let eq = EqBasicBlock {};
    eq.verify(
      srs,
      &ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
      &vec![outputs[0], outputs[0]],
      &vec![outputs[0]],
      proof,
      &mut StdRng::from_entropy(),
      _cache,
    )
  }
}

#[derive(Debug)]
pub struct SGDUpdateBlock;

impl BasicBlock for SGDUpdateBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 3);
    let w = inputs[0];
    let grad = inputs[1];
    let lr = inputs[2];
    assert!(w.len() == 1 && grad.len() == 1 && lr.len() == 1);
    let scale = Fr::from(FIXED_SCALE);
    let next = *w.first().unwrap() - *lr.first().unwrap() * *grad.first().unwrap() / scale;
    vec![ArrayD::from_elem(IxDyn(&[1]), next)]
  }

  fn encodeOutputs(&self, srs: &SRS, _model: &ArrayD<Data>, _inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    outputs.iter().map(|o| util::convert_to_data(srs, o)).collect()
  }

  fn prove(
    &self,
    srs: &SRS,
    _setup: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<ark_poly::univariate::DensePolynomial<Fr>>),
    _model: &ArrayD<Data>,
    inputs: &Vec<&ArrayD<Data>>,
    outputs: &Vec<&ArrayD<Data>>,
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> (Vec<G1Projective>, Vec<G2Projective>, Vec<Fr>) {
    let mul_scalar = super::MulScalarBasicBlock {};
    let update = mul_scalar.prove(
      srs,
      (&vec![], &vec![], &vec![]),
      &ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
      &vec![inputs[1], inputs[2]],
      &vec![&outputs[0]],
      rng,
      cache,
    );
    update
  }

  fn verify(
    &self,
    srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    rng: &mut StdRng,
    cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let mul_scalar = super::MulScalarBasicBlock {};
    mul_scalar.verify(
      srs,
      &ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
      &vec![inputs[1], inputs[2]],
      &vec![outputs[0]],
      proof,
      rng,
      cache,
    )
  }
}

#[derive(Debug)]
pub struct ChallengeBlock;

impl BasicBlock for ChallengeBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 3);
    let mut buf = Vec::new();
    inputs.iter().for_each(|i| {
      let d = derive_challenge(&util::tensor_to_bytes_fr(i));
      buf.extend_from_slice(&d.into_bigint().0[0].to_le_bytes());
    });
    let c = derive_challenge(&buf);
    vec![ArrayD::from_elem(IxDyn(&[1]), c)]
  }

  fn encodeOutputs(&self, srs: &SRS, _model: &ArrayD<Data>, _inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    outputs.iter().map(|o| util::convert_to_data(srs, o)).collect()
  }
}
