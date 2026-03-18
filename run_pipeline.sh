#!/usr/bin/env bash
# run_pipeline.sh — End-to-end Data Binding pipeline orchestrator.
#
# Steps:
#   1. Compile the C++ Poseidon commitment engine.
#   2. Extract the flat training tensor from the witness JSON and hash it.
#   3. Inject the Merkle root into the witness via the Python injector.
#   4. Deploy TrainingRegistry to a local Hardhat network and commit the root.
#
# Prerequisites:
#   - g++ (C++17)
#   - Python 3
#   - Node.js + npm (Hardhat & deps installed via npm install)
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh

set -euo pipefail

# Prevent Hardhat from blocking on first-run telemetry prompt.
export HARDHAT_DISABLE_TELEMETRY_PROMPT=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_DIR/build"
WITNESS_V1="$PROJECT_DIR/step_witness_v1.json"
WITNESS_V2="$PROJECT_DIR/step_witness_v2.json"
ROOT_FILE="$PROJECT_DIR/merkle_root.txt"
FLAT_DATA="$BUILD_DIR/flat_data.json"

echo "============================================"
echo "  ZKCollab Data Binding Pipeline"
echo "============================================"
echo ""

# -----------------------------------------------------------------
# Step 0: Prepare build directory
# -----------------------------------------------------------------
mkdir -p "$BUILD_DIR"

# -----------------------------------------------------------------
# Step 1: Compile the C++ Poseidon commitment engine
# -----------------------------------------------------------------
echo "[1/4] Compiling C++ Poseidon engine..."
g++ -std=c++17 -O2 -o "$BUILD_DIR/poseidon_field" "$PROJECT_DIR/src/poseidon_field.cpp"
echo "      Binary: $BUILD_DIR/poseidon_field"

# -----------------------------------------------------------------
# Step 2: Extract flat tensor from witness and compute Merkle root
# -----------------------------------------------------------------
echo "[2/4] Computing Merkle root of training data..."

# Use Python to extract and flatten the "x" field from the witness,
# regardless of its dimensionality.
python3 -c "
import json, sys
with open('$WITNESS_V1') as f:
    w = json.load(f)
def flatten(obj):
    if isinstance(obj, (list, tuple)):
        r = []
        for i in obj:
            r.extend(flatten(i))
        return r
    return [obj]
flat = flatten(w['x'])
with open('$FLAT_DATA', 'w') as f:
    json.dump(flat, f)
print(f'Extracted {len(flat)} elements from witness x field')
"

"$BUILD_DIR/poseidon_field" "$FLAT_DATA" "$ROOT_FILE"
MERKLE_ROOT=$(cat "$ROOT_FILE")
echo "      Merkle root: $MERKLE_ROOT"

# -----------------------------------------------------------------
# Step 3: Inject root into witness JSON via Python
# -----------------------------------------------------------------
echo "[3/4] Injecting Merkle root into witness..."
python3 "$PROJECT_DIR/scripts/inject_root.py" \
    --root "$MERKLE_ROOT" \
    --witness "$WITNESS_V1" \
    --output "$WITNESS_V2"
echo "      Augmented witness: $WITNESS_V2"

# -----------------------------------------------------------------
# Step 4: Deploy TrainingRegistry and commit root on-chain
# -----------------------------------------------------------------
echo "[4/4] Deploying TrainingRegistry and committing Merkle root..."

# Install npm dependencies if needed.
if [ ! -d "$PROJECT_DIR/node_modules" ]; then
    echo "      Installing npm dependencies..."
    (cd "$PROJECT_DIR" && npm install --save-dev "hardhat@^2.22.0" "@nomicfoundation/hardhat-toolbox@hh2")
fi

# Use Hardhat's built-in in-process network (no separate node required).
# This avoids telemetry prompts and background-process issues entirely.
export MERKLE_ROOT="$MERKLE_ROOT"
(cd "$PROJECT_DIR" && npx hardhat run scripts/deploy.js)

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  - Merkle root:      $ROOT_FILE"
echo "  - Augmented witness: $WITNESS_V2"
echo "  - Deployment info:   $PROJECT_DIR/deployment_info.json"
echo "============================================"
