#!/usr/bin/env python3
"""
inject_root.py — Dimension-agnostic witness injector for the Data Binding layer.

Reads:
  1. A Merkle root hex string (from a file or CLI arg).
  2. The existing step_witness_v1.json.

Injects into the witness JSON:
  - "merkle_root":  the committed root (hex string).
  - "raw_data":     the fully flattened "x" array (works for any shape).
  - "merkle_path":  a dummy Merkle path (list of hex strings, length = ceil(log2(N))).

Writes the augmented witness to step_witness_v2.json (or a user-specified path).

Usage:
    python scripts/inject_root.py [--root ROOT_HEX | --root-file FILE]
                                  [--witness FILE] [--output FILE]
"""

import argparse
import json
import math
import os
import sys


def flatten(obj):
    """Recursively flatten an arbitrarily nested list/array to a 1-D list."""
    if isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            result.extend(flatten(item))
        return result
    return [obj]


def dummy_merkle_path(num_leaves: int) -> list:
    """
    Generate a dummy Merkle sibling path.
    Length = ceil(log2(num_leaves))  (0 if num_leaves <= 1).
    Each entry is a zero hash placeholder (0x00...00, 64 hex chars).
    """
    if num_leaves <= 1:
        return []
    depth = math.ceil(math.log2(num_leaves))
    zero = "0x" + "0" * 64
    return [zero] * depth


def main():
    parser = argparse.ArgumentParser(description="Inject Merkle root into witness JSON")
    parser.add_argument("--root", type=str, default=None,
                        help="Merkle root hex string (0x-prefixed)")
    parser.add_argument("--root-file", type=str, default=None,
                        help="File containing the Merkle root hex string")
    parser.add_argument("--witness", type=str, default="step_witness_v2.json",
                        help="Input witness JSON path")
    parser.add_argument("--output", type=str, default="step_witness_v2.json",
                        help="Output augmented witness JSON path")
    args = parser.parse_args()

    # ---------- Resolve Merkle root ----------
    root_hex = args.root
    if root_hex is None:
        root_file = args.root_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "merkle_root.txt"
        )
        if not os.path.isfile(root_file):
            print(f"Error: Neither --root nor a readable --root-file ({root_file}) provided.",
                  file=sys.stderr)
            sys.exit(1)
        with open(root_file, "r") as f:
            root_hex = f.read().strip()

    if not root_hex.startswith("0x"):
        root_hex = "0x" + root_hex
    # Normalize to lowercase, 0x + 64 hex chars.
    root_hex = "0x" + root_hex[2:].lower().zfill(64)

    # ---------- Load witness ----------
    witness_path = args.witness
    if not os.path.isfile(witness_path):
        print(f"Error: witness file not found: {witness_path}", file=sys.stderr)
        sys.exit(1)
    with open(witness_path, "r") as f:
        witness = json.load(f)

    # ---------- Flatten "x" (dimension-agnostic) ----------
    if "x" not in witness:
        print("Error: witness JSON has no 'x' field.", file=sys.stderr)
        sys.exit(1)

    raw_data = flatten(witness["x"])
    num_elements = len(raw_data)
    print(f"Flattened 'x' into {num_elements} element(s).")

    # ---------- Inject ----------
    witness["merkle_root"] = root_hex
    witness["raw_data"] = raw_data
    witness["merkle_path"] = []  # Empty: the prover verifies via full root recomputation
    witness["raw_data_length"] = num_elements

    # ---------- Write ----------
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(witness, f, indent=2)
        f.write("\n")
    print(f"Augmented witness written to {output_path}")


if __name__ == "__main__":
    main()
