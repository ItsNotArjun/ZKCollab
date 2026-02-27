use std::fs::File;
use std::io::BufReader;

use zk_torch::basic_block::SRS;
use zk_torch::ptau;
use zk_torch::training::proof::{prove_sample_step_snark, SampleProofV1};
use zk_torch::training::witness::SampleWitnessV1;
use zk_torch::CONFIG;

fn to_hex(bytes: &[u8]) -> String {
  let mut s = String::with_capacity(bytes.len() * 2);
  for b in bytes {
    use std::fmt::Write as _;
    let _ = write!(&mut s, "{:02x}", b);
  }
  s
}

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

  let proof: SampleProofV1 = prove_sample_step_snark(&_srs, &witness).expect("SNARK proof generation failed");
  let proof_len = proof.proof_bytes.len();

  // Write proof bytes to a file, following the zk-torch style of
  // using config-driven paths. We suffix the standard proof_path
  // so as not to collide with inference proofs.
  let proof_file = format!("{}_training", CONFIG.prover.proof_path);
  std::fs::write(&proof_file, &proof.proof_bytes).expect("failed to write training proof file");

  println!(
    "sample training step checks passed; proof_bytes len = {} (written to {})",
    proof_len,
    proof_file
  );
  println!("proof_bytes_hex = {}", to_hex(&proof.proof_bytes));
}
