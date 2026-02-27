use std::fs::File;
use std::io::BufReader;

use zk_torch::basic_block::SRS;
use zk_torch::ptau;
use zk_torch::training::proof::{prove_sample_step, SampleProofV1};
use zk_torch::training::witness::SampleWitnessV1;

fn main() {
  // In a full implementation, these parameters would come from
  // config. For now we assume the repo's bundled "challenge" ptau
  // file, which is sized for (n = 7, m = 7) as used elsewhere in
  // the codebase.
  let _srs: SRS = ptau::load_file("challenge", 7, 7);

  let path = std::env::args().nth(1).unwrap_or_else(|| "step_witness_v1.json".to_string());
  let file = File::open(&path).expect("failed to open witness JSON");
  let reader = BufReader::new(file);
  let witness: SampleWitnessV1 = serde_json::from_reader(reader).expect("failed to parse SampleWitnessV1 JSON");

  let proof: SampleProofV1 = prove_sample_step(&_srs, &witness);
  println!(
    "sample training step checks passed; proof_bytes len = {}",
    proof.proof_bytes.len()
  );
}
