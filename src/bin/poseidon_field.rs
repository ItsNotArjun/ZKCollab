//! Drop-in replacement for the C++ `poseidon_field` binary.
//!
//! Reads a flat JSON array of integers, computes the Poseidon Merkle
//! root over the BN254 scalar field, and prints the hex root.
//!
//! Usage:
//!   poseidon_field <input.json> [output.txt]

use ark_bn254::Fr;
use ark_ff::PrimeField;
use std::env;
use std::fs;
use std::io::Write;

use zk_torch::training::data_binding::compute_merkle_root;

fn i64_to_fr(v: i64) -> Fr {
    if v >= 0 {
        Fr::from(v as u64)
    } else {
        -Fr::from((-v) as u64)
    }
}

fn fr_to_hex(val: Fr) -> String {
    let bigint = val.into_bigint();
    format!(
        "0x{:016x}{:016x}{:016x}{:016x}",
        bigint.0[3], bigint.0[2], bigint.0[1], bigint.0[0]
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.json> [output.txt]", args[0]);
        std::process::exit(1);
    }

    let contents = fs::read_to_string(&args[1]).unwrap_or_else(|e| {
        eprintln!("Error: cannot open {}: {}", args[1], e);
        std::process::exit(1);
    });

    let values: Vec<i64> = serde_json::from_str(&contents).unwrap_or_else(|e| {
        eprintln!("Error: failed to parse JSON array: {}", e);
        std::process::exit(1);
    });

    if values.is_empty() {
        eprintln!("Error: no integers found in input JSON");
        std::process::exit(1);
    }

    eprintln!("Read {} field elements from {}", values.len(), args[1]);

    let field_elements: Vec<Fr> = values.iter().map(|&v| i64_to_fr(v)).collect();
    let root = compute_merkle_root(&field_elements);
    let hex_root = fr_to_hex(root);

    println!("{}", hex_root);

    if args.len() >= 3 {
        let mut f = fs::File::create(&args[2]).unwrap_or_else(|e| {
            eprintln!("Error: cannot write {}: {}", args[2], e);
            std::process::exit(1);
        });
        writeln!(f, "{}", hex_root).unwrap();
        eprintln!("Root written to {}", args[2]);
    }
}
