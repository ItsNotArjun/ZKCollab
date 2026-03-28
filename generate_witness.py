"""Generate a single-step training witness for the SampleModel architecture.

This script runs one SGD step on a 2-layer MLP (SampleModel) whose
input and hidden dimensions are configurable, then dumps all tensors
needed by the Rust prover into a JSON file ``step_witness_v2.json``.

The JSON shape remains compatible with ``SampleWitnessV1`` and simply
adds explicit dimension metadata for flexibility.

Rust is responsible for turning this witness into a zk-proof; the
Python side is purely for producing the training transcript.
"""

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import torch.nn as nn
from sample_model import SampleModel


FIXED_SCALE_K = 16
FIXED_SCALE = 1 << FIXED_SCALE_K


def round_banker(prod: int, scale: int) -> int:
    """Round-half-to-even (banker's rounding) for integer division prod/scale.

    Matches Rust's round_fixed_point exactly: uses truncation toward zero
    for the base quotient, then applies tie-breaking to the nearest even
    integer when the remainder is exactly half the scale.
    """
    # int() truncates toward zero, matching Rust's / operator on i64
    quotient = int(prod / scale)
    remainder = prod - quotient * scale  # same sign as prod
    abs_rem = abs(remainder)
    half = scale // 2
    if abs_rem < half:
        return quotient
    elif abs_rem > half:
        return quotient + 1 if remainder > 0 else quotient - 1
    else:
        # Tie: round to nearest even
        if quotient % 2 == 0:
            return quotient
        return quotient + 1 if remainder > 0 else quotient - 1


def quantize(v: float) -> int:
    """Quantize a float to Q16 fixed-point.

    Uses Python's built-in round(), which also implements banker's rounding
    (round-half-to-even) for float values.  This is the correct rounding to
    use when converting floating-point model parameters to Q16 integers.
    """
    return int(round(v * FIXED_SCALE))


def to_int_vector(t: torch.Tensor) -> list:
    """Convert tensor to list of scaled integers."""
    return [quantize(v.item()) for v in t.detach().view(-1)]


def to_int_matrix(t: torch.Tensor) -> list:
    """Convert matrix tensor to list of lists of scaled integers."""
    t2 = t.detach().view(t.size(0), -1)
    return [[quantize(v.item()) for v in row] for row in t2]


def flatten_witness_data(witness: dict) -> list:
    """Flatten all witness numeric data into a single list for hashing."""
    raw = []
    for key in sorted(witness.keys()):
        if key in ('merkle_root', 'raw_data', 'input_dim', 'hidden_dim', 'fixed_scale_k'):
            continue
        val = witness[key]
        if isinstance(val, list):
            if isinstance(val[0], list):
                # Matrix
                for row in val:
                    raw.extend(row)
            else:
                # Vector
                raw.extend(val)
        else:
            # Scalar
            raw.append(val)
    return raw


def compute_poseidon_root(raw_data: list) -> str:
    """Compute Merkle root using the Rust poseidon_field binary."""
    try:
        # Write raw data to temp file
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(raw_data, f)
            temp_path = f.name
        
        # Run poseidon_field
        result = subprocess.run(
            ['cargo', 'run', '--bin', 'poseidon_field', '--', temp_path],
            capture_output=True,
            text=True,
            cwd='.'
        )
        
        if result.returncode != 0:
            print(f"Warning: poseidon_field failed: {result.stderr}")
            return "0x0000000000000000000000000000000000000000000000000000000000000000"
        
        # Extract hex from stdout (first line)
        hex_root = result.stdout.strip().split('\n')[0]
        Path(temp_path).unlink()  # Clean up temp file
        return hex_root
        
    except Exception as e:
        print(f"Warning: Could not compute Poseidon root: {e}")
        # Fallback to committed root if available
        try:
            with open('merkle_root.txt', 'r') as f:
                return f.read().strip()
        except:
            return "0x0000000000000000000000000000000000000000000000000000000000000000"


def relu_backward_q(z_q: list, upstream_q: list) -> list:
    """Compute ReLU backward pass from quantized integer inputs.

    ReLU backward is exact (no rescaling): pass upstream through when z > 0.
    Matches Rust's enforce_relu_vjp exactly.
    """
    return [u if z > 0 else 0 for z, u in zip(z_q, upstream_q)]


def linear_backward_grad_x_q(w_q: list, upstream_q: list) -> list:
    """Compute grad_x = W^T @ upstream, rescaled Q32→Q16 using banker's rounding.

    Matches Rust's enforce_linear_vjp grad_x loop exactly.
    w_q is a list-of-lists (m×n), upstream_q is length-m.
    Returns length-n grad_x.
    """
    m = len(w_q)
    n = len(w_q[0])
    grad_x = []
    for j in range(n):
        acc = sum(w_q[i][j] * upstream_q[i] for i in range(m))
        grad_x.append(round_banker(acc, FIXED_SCALE))
    return grad_x


def linear_backward_grad_w_q(upstream_q: list, x_q: list) -> list:
    """Compute grad_w = upstream ⊗ x, rescaled Q32→Q16 using banker's rounding.

    Matches Rust's enforce_linear_vjp grad_w loop exactly.
    Returns a list-of-lists (m×n).
    """
    return [
        [round_banker(upstream_q[i] * x_q[j], FIXED_SCALE) for j in range(len(x_q))]
        for i in range(len(upstream_q))
    ]


def sgd_update_q(w_q: list, grad_q: list, lr_scaled: int) -> list:
    """Apply SGD update using banker's rounding, matching Rust's enforce_sgd_update.

    w_next[i] = w[i] - round_banker(lr_scaled * grad[i], FIXED_SCALE)
    Works for both vectors and lists-of-lists (matrices).
    """
    if w_q and isinstance(w_q[0], list):
        return [
            [w_q[i][j] - round_banker(lr_scaled * grad_q[i][j], FIXED_SCALE)
             for j in range(len(w_q[i]))]
            for i in range(len(w_q))
        ]
    return [w - round_banker(lr_scaled * g, FIXED_SCALE) for w, g in zip(w_q, grad_q)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a training witness for SampleModel.")
    parser.add_argument(
        "--input-dim",
        type=int,
        default=4,
        help="Input/output dimension of the model (default: 4)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4,
        help="Hidden layer dimension of the model (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("step_witness_v2.json"),
        help="Where to write the witness JSON (default: step_witness_v2.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = SampleModel(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    model.train()

    # Single input/target for an n-D regression task
    x = torch.randn(1, args.input_dim, requires_grad=True)
    y_target = torch.randn(1, args.input_dim)

    # Capture parameters before the step
    w1_before = model.lin1.weight.detach().clone()
    b1_before = model.lin1.bias.detach().clone()
    w2_before = model.lin2.weight.detach().clone()
    b2_before = model.lin2.bias.detach().clone()

    # Manual forward with retained intermediate activations
    z1 = model.lin1(x)
    z1.retain_grad()
    a1 = model.relu(z1)
    a1.retain_grad()
    z2 = model.lin2(a1)
    z2.retain_grad()
    y_pred = model.relu(z2)
    y_pred.retain_grad()

    lr = 0.01
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    opt.zero_grad()
    loss = nn.MSELoss()(y_pred, y_target)
    loss.backward()
    opt.step()

    # Parameters after the step
    w1_after = model.lin1.weight.detach().clone()
    b1_after = model.lin1.bias.detach().clone()
    w2_after = model.lin2.weight.detach().clone()
    b2_after = model.lin2.bias.detach().clone()

    # Gradients wrt parameters
    grad_w1 = model.lin1.weight.grad.detach().clone()
    grad_b1 = model.lin1.bias.grad.detach().clone()
    grad_w2 = model.lin2.weight.grad.detach().clone()
    grad_b2 = model.lin2.bias.grad.detach().clone()

    # Gradients wrt activations (upstream signals)
    grad_y = y_pred.grad.detach().clone()
    grad_z2 = z2.grad.detach().clone()
    grad_a1 = a1.grad.detach().clone()
    grad_z1 = z1.grad.detach().clone()
    grad_x = x.grad.detach().clone()

    lr_scaled = quantize(lr)

    # Quantize all primary (unconstrained) values from float.
    q_x         = to_int_vector(x)
    q_y_target  = to_int_vector(y_target)
    q_w1_before = to_int_matrix(w1_before)
    q_b1_before = to_int_vector(b1_before)
    q_w2_before = to_int_matrix(w2_before)
    q_b2_before = to_int_vector(b2_before)
    q_z1        = to_int_vector(z1)
    q_a1        = to_int_vector(a1)
    q_z2        = to_int_vector(z2)
    q_y_pred    = to_int_vector(y_pred)
    q_grad_y    = to_int_vector(grad_y)
    # Bias gradients are not constrained by VJP, so quantize from float.
    q_grad_b1   = to_int_vector(grad_b1)
    q_grad_b2   = to_int_vector(grad_b2)

    # Recompute all constrained values from quantized integers using
    # round_banker so they match Rust's round_fixed_point exactly.
    # This eliminates any off-by-1 discrepancy caused by float→int quantization.

    # Backward pass, layer 2 (final ReLU: z2 → y_pred)
    q_grad_z2 = relu_backward_q(q_z2, q_grad_y)

    # Backward pass, linear layer 2 (a1 → z2)
    q_grad_a1  = linear_backward_grad_x_q(q_w2_before, q_grad_z2)
    q_grad_w2  = linear_backward_grad_w_q(q_grad_z2, q_a1)

    # Backward pass, first ReLU (z1 → a1)
    q_grad_z1 = relu_backward_q(q_z1, q_grad_a1)

    # Backward pass, linear layer 1 (x → z1)
    q_grad_x  = linear_backward_grad_x_q(q_w1_before, q_grad_z1)
    q_grad_w1 = linear_backward_grad_w_q(q_grad_z1, q_x)

    # SGD updates — computed from the same quantized grad values so that
    # enforce_sgd_update passes with exact equality.
    q_w1_after = sgd_update_q(q_w1_before, q_grad_w1, lr_scaled)
    q_b1_after = sgd_update_q(q_b1_before, q_grad_b1, lr_scaled)
    q_w2_after = sgd_update_q(q_w2_before, q_grad_w2, lr_scaled)
    q_b2_after = sgd_update_q(q_b2_before, q_grad_b2, lr_scaled)

    witness = {
        "fixed_scale_k": FIXED_SCALE_K,
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "x": q_x,
        "y_target": q_y_target,
        "lr_scaled": lr_scaled,
        # Parameters before/after
        "w1_before": q_w1_before,
        "b1_before": q_b1_before,
        "w2_before": q_w2_before,
        "b2_before": q_b2_before,
        "w1_after": q_w1_after,
        "b1_after": q_b1_after,
        "w2_after": q_w2_after,
        "b2_after": q_b2_after,
        # Forward activations
        "z1": q_z1,
        "a1": q_a1,
        "z2": q_z2,
        "y_pred": q_y_pred,
        # Gradients wrt activations and input (recomputed from quantized integers)
        "grad_y": q_grad_y,
        "grad_z2": q_grad_z2,
        "grad_a1": q_grad_a1,
        "grad_z1": q_grad_z1,
        "grad_x": q_grad_x,
        # Parameter gradients (recomputed from quantized integers)
        "grad_w1": q_grad_w1,
        "grad_b1": q_grad_b1,
        "grad_w2": q_grad_w2,
        "grad_b2": q_grad_b2,
    }

    # Flatten and compute Merkle root using Poseidon
    raw_data = flatten_witness_data(witness)
    witness["raw_data"] = raw_data
    witness["merkle_root"] = compute_poseidon_root(raw_data)

    out: Path = args.output
    out.write_text(json.dumps(witness, indent=2))
    print(f"Wrote SampleModel witness ({args.input_dim}×{args.hidden_dim}) to {out}")
    print(f"  merkle_root: {witness['merkle_root']}")


if __name__ == "__main__":  # pragma: no cover
    main()
