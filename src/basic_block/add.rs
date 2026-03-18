use super::{BasicBlock, Data, DataEnc, PairingCheck, ProveVerifyCache, SRS};
use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ndarray::{arr0, azip, ArrayD, IxDyn};
use rand::rngs::StdRng;

#[derive(Debug)]
pub struct AddBasicBlock;
impl BasicBlock for AddBasicBlock {
  fn run(&self, _model: &ArrayD<Fr>, inputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Fr>> {
    assert!(inputs.len() == 2);
    // Flatten inputs to 1D for element-wise addition (supports any ndim).
    let flat0: Vec<Fr> = inputs[0].iter().cloned().collect();
    let flat1: Vec<Fr> = inputs[1].iter().cloned().collect();
    let len = std::cmp::max(flat0.len(), flat1.len());
    let mut r = ArrayD::zeros(IxDyn(&[len]));
    if flat0.len() == 1 {
      azip!((r in &mut r, &x in &ArrayD::from_shape_vec(IxDyn(&[flat1.len()]), flat1).unwrap()) *r = x + flat0[0]);
    } else if flat1.len() == 1 {
      azip!((r in &mut r, &x in &ArrayD::from_shape_vec(IxDyn(&[flat0.len()]), flat0).unwrap()) *r = x + flat1[0]);
    } else {
      let a = ArrayD::from_shape_vec(IxDyn(&[flat0.len()]), flat0).unwrap();
      let b = ArrayD::from_shape_vec(IxDyn(&[flat1.len()]), flat1).unwrap();
      azip!((r in &mut r, &x in &a, &y in &b) *r = x + y);
    }
    vec![r]
  }

  fn encodeOutputs(&self, _srs: &SRS, _model: &ArrayD<Data>, inputs: &Vec<&ArrayD<Data>>, outputs: &Vec<&ArrayD<Fr>>) -> Vec<ArrayD<Data>> {
    let a = &inputs[0].first().unwrap();
    let b = &inputs[1].first().unwrap();
    vec![arr0(Data {
      raw: outputs[0].clone().into_raw_vec(),
      poly: (&a.poly) + (&b.poly),
      g1: a.g1 + b.g1,
      r: a.r + b.r,
    })
    .into_dyn()]
  }

  fn verify(
    &self,
    _srs: &SRS,
    _model: &ArrayD<DataEnc>,
    inputs: &Vec<&ArrayD<DataEnc>>,
    outputs: &Vec<&ArrayD<DataEnc>>,
    _proof: (&Vec<G1Affine>, &Vec<G2Affine>, &Vec<Fr>),
    _rng: &mut StdRng,
    _cache: ProveVerifyCache,
  ) -> Vec<PairingCheck> {
    let a = inputs[0].first().unwrap();
    let b = inputs[1].first().unwrap();
    let c = outputs[0].first().unwrap();
    // Verify f(x)+g(x)=h(x)
    assert!(a.g1 + b.g1 == c.g1);
    vec![]
  }
}
