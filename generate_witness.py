import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from sample_model import SampleModel


"""Generate a single-step training witness for the SampleModel architecture.

This script runs one SGD step on a configurable 2-layer MLP (SampleModel)
and dumps all the tensors needed by the Rust prover into a JSON
file `step_witness_v2.json`.

Dimensions are controlled via --input-dim and --hidden-dim (both default to 4
to preserve backward compatibility with the original 4×4 architecture).

Rust is responsible for turning this witness into a zk-proof; the
Python side is purely for producing the training transcript.
"""


FIXED_SCALE_K = 16
FIXED_SCALE = 1 << FIXED_SCALE_K


def to_int_matrix(t: torch.Tensor):
    t2 = t.detach().view(t.size(0), -1)
    return [[int(round(v.item() * FIXED_SCALE)) for v in row] for row in t2]


def to_int_vector(t: torch.Tensor):
    return [int(round(v.item() * FIXED_SCALE)) for v in t.detach().view(-1)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a single-step SGD training witness for SampleModel."
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=4,
        help="Input dimension of the MLP (default: 4)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4,
        help="Hidden (and output) dimension of the MLP (default: 4)",
    )
    args = parser.parse_args()

    input_dim: int = args.input_dim
    hidden_dim: int = args.hidden_dim

    model = SampleModel(input_dim=input_dim, hidden_dim=hidden_dim)
    model.train()

    # Single input/target for an n-dimensional regression task
    x = torch.randn(1, input_dim, requires_grad=True)
    y_target = torch.randn(1, hidden_dim)

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
    grad_y = y_pred.grad.detach().clone()          # upstream into final ReLU output
    grad_z2 = z2.grad.detach().clone()             # grad wrt pre-activation of final ReLU
    grad_a1 = a1.grad.detach().clone()             # upstream into first ReLU output
    grad_z1 = z1.grad.detach().clone()             # grad wrt pre-activation of first ReLU
    grad_x = x.grad.detach().clone()               # grad wrt input

    lr_scaled = int(round(lr * FIXED_SCALE))

    witness = {
        "fixed_scale_k": FIXED_SCALE_K,
        "x": to_int_vector(x),
        "y_target": to_int_vector(y_target),
        "lr_scaled": lr_scaled,
        # Parameters before/after
        "w1_before": to_int_matrix(w1_before),
        "b1_before": to_int_vector(b1_before),
        "w2_before": to_int_matrix(w2_before),
        "b2_before": to_int_vector(b2_before),
        "w1_after": to_int_matrix(w1_after),
        "b1_after": to_int_vector(b1_after),
        "w2_after": to_int_matrix(w2_after),
        "b2_after": to_int_vector(b2_after),
        # Forward activations
        "z1": to_int_vector(z1),
        "a1": to_int_vector(a1),
        "z2": to_int_vector(z2),
        "y_pred": to_int_vector(y_pred),
        # Gradients wrt activations and input
        "grad_y": to_int_vector(grad_y),
        "grad_z2": to_int_vector(grad_z2),
        "grad_a1": to_int_vector(grad_a1),
        "grad_z1": to_int_vector(grad_z1),
        "grad_x": to_int_vector(grad_x),
        # Parameter gradients
        "grad_w1": to_int_matrix(grad_w1),
        "grad_b1": to_int_vector(grad_b1),
        "grad_w2": to_int_matrix(grad_w2),
        "grad_b2": to_int_vector(grad_b2),
    }

    out = Path("step_witness_v2.json")
    out.write_text(json.dumps(witness, indent=2))
    print(f"Wrote SampleModel witness ({input_dim}×{hidden_dim}) to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
