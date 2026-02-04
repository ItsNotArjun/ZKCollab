# import onnx
# import netron
#
#
# def main() -> None:
#     model_path = "sample.onnx"
#
#     # Basic ONNX sanity checks / textual inspection
#     model = onnx.load(model_path)
#     onnx.checker.check_model(model)
#
#     print("Graph:")
#     print(model.graph)
#
#     print("\nNodes (op_type, inputs, outputs):")
#     for node in model.graph.node:
#         print(node.op_type, list(node.input), list(node.output))
#
#     print("\nValue infos (name, shape):")
#     for value in model.graph.value_info:
#         ttype = value.type.tensor_type
#         shape = [d.dim_value for d in ttype.shape.dim]
#         print(value.name, shape)
#
#     # Start Netron viewer (uses its default address/port) and keep process alive
#     print("\nStarting Netron (check the console for the URL) ...")
#     netron.start(model_path)
#     input("\nNetron running. Press Enter to exit... ")
#
#
# if __name__ == "__main__":
#     main()

import os
from pathlib import Path


def list_basic_blocks() -> None:
    """Print all basic block names from src/basic_block/*.rs."""

    repo_root = Path(__file__).resolve().parent
    basic_block_dir = repo_root / "src" / "basic_block"

    if not basic_block_dir.is_dir():
        print(f"basic_block directory not found at: {basic_block_dir}")
        return

    rs_files = sorted(basic_block_dir.glob("*.rs"))

    print("Basic blocks found (from src/basic_block):")
    for path in rs_files:
        name = path.stem
        # Skip potential module glue files like mod.rs if present
        if name == "mod":
            continue
        print(f"- {name}")


if __name__ == "__main__":
    list_basic_blocks()
