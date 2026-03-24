#!/usr/bin/env bash
# ===================================================================
#  ZKCollab Data Binding Demo — Visual Execution Script
# ===================================================================
#
#  Runs the full end-to-end pipeline with colored, narrated output.
#
#  Environment:
#    - WSL (Ubuntu): g++ for Poseidon C++ engine, Hardhat via nvm
#    - Windows interop: cargo.exe for Rust, python.exe for PyTorch
#
#  Usage (from WSL):
#    chmod +x scripts/demo_run.sh
#    cd /mnt/c/Users/Abhishek/ZK_fork/ZKCollab
#    bash scripts/demo_run.sh
#
# ===================================================================

set -euo pipefail

# ---- Color helpers ----
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'          # No Color

info()    { echo -e "${CYAN}[INFO]${NC}    $*"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}    $*"; }
fail()    { echo -e "${RED}[ERROR]${NC}   $*"; exit 1; }
header()  { echo -e "\n${BOLD}═══════════════════════════════════════════════════${NC}"; echo -e "${BOLD}  $*${NC}"; echo -e "${BOLD}═══════════════════════════════════════════════════${NC}\n"; }

# ---- Paths ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
WITNESS_V1="$PROJECT_DIR/step_witness_v1.json"
WITNESS_V2="$PROJECT_DIR/step_witness_v2.json"
ROOT_FILE="$PROJECT_DIR/merkle_root.txt"
FLAT_DATA="$BUILD_DIR/flat_data.json"

# Resolve Windows-native tools accessible from WSL
CARGO_EXE="$(command -v cargo 2>/dev/null || echo /mnt/c/Users/Abhishek/.cargo/bin/cargo.exe)"

# Python: WSL python3 for JSON ops, Windows Python only for PyTorch
PYTHON3="python3"
PYTHON_TORCH=""
if /mnt/c/Python313/python.exe -c "import torch" 2>/dev/null; then
  PYTHON_TORCH="/mnt/c/Python313/python.exe"
fi

# Convert a WSL /mnt/c/... path to Windows C:\... for Windows Python
wsl_to_win() {
  echo "$1" | sed 's|^/mnt/\(.\)|\U\1:|; s|/|\\|g'
}

# Hardhat needs Node.js from nvm
export HARDHAT_DISABLE_TELEMETRY_PROMPT=true
if [ -s "$HOME/.nvm/nvm.sh" ]; then
  . "$HOME/.nvm/nvm.sh"
  nvm use 22 >/dev/null 2>&1 || true
fi

cd "$PROJECT_DIR"
mkdir -p "$BUILD_DIR"

# ===================================================================
header "ZKCollab  —  Data Binding Demo Pipeline"
echo -e "  ${CYAN}Architecture${NC}: 2-layer 4×4 MLP (Linear→ReLU→Linear→ReLU)"
echo -e "  ${CYAN}Pipeline${NC}    : Witness → Poseidon Root → On-Chain Commit → ZK Proof"
echo ""

# ============================ STEP 0 ================================
header "Step 0: Generate Training Witness (PyTorch)"
info "Running one SGD step on SampleModel (4×4 MLP)..."
if [ ! -f "$WITNESS_V1" ]; then
  if [ -n "$PYTHON_TORCH" ]; then
    $PYTHON_TORCH "$(wsl_to_win "$PROJECT_DIR/generate_witness.py")" || fail "Witness generation failed"
  else
    fail "PyTorch not found. Install torch on Windows Python or generate step_witness_v1.json manually."
  fi
  success "Witness generated → step_witness_v1.json"
else
  info "step_witness_v1.json already exists — skipping generation."
fi
ELEMENT_COUNT=$($PYTHON3 -c "
import json
def flatten(o):
    if isinstance(o,(list,tuple)):
        r=[]
        for i in o: r.extend(flatten(i))
        return r
    return [o]
w=json.load(open('$WITNESS_V1'))
print(len(flatten(w['x'])))
")
success "Witness contains ${BOLD}${ELEMENT_COUNT}${NC} input elements (x field)"

# ============================ STEP 1 ================================
header "Step 1: Compute Poseidon Merkle Root (Rust Engine)"
info "Building Rust Poseidon engine (BN254, width=3, α=5)..."
if ! $CARGO_EXE build --release --bin poseidon_field 2>/dev/null; then
  fail "cargo build failed for poseidon_field"
fi
success "Poseidon engine built."

info "Extracting flat tensor from witness 'x' field..."
$PYTHON3 -c "
import json
def flatten(o):
    if isinstance(o,(list,tuple)):
        r=[]
        for i in o: r.extend(flatten(i))
        return r
    return [o]
w=json.load(open('$WITNESS_V1'))
flat=flatten(w['x'])
json.dump(flat, open('$FLAT_DATA','w'))
print(f'Extracted {len(flat)} elements')
"

info "Computing Poseidon Merkle root..."
POSEIDON_BIN="$PROJECT_DIR/target/release/poseidon_field"
# Handle Windows-style .exe from WSL
if [ ! -f "$POSEIDON_BIN" ] && [ -f "${POSEIDON_BIN}.exe" ]; then
  POSEIDON_BIN="${POSEIDON_BIN}.exe"
fi
# Use relative paths since poseidon_field.exe (Windows) doesn't understand /mnt/c/...
"$POSEIDON_BIN" "build/flat_data.json" "merkle_root.txt" 2>/dev/null
MERKLE_ROOT=$(cat "$ROOT_FILE" | tr -d '[:space:]')
echo ""
echo -e "    ${BOLD}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "    ${BOLD}║${NC}  ${GREEN}Merkle Root${NC}: ${YELLOW}${MERKLE_ROOT}${NC}  ${BOLD}║${NC}"
echo -e "    ${BOLD}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
success "Root written to merkle_root.txt"

# ============================ STEP 2 ================================
header "Step 2: Deploy Smart Contract & Commit Root On-Chain"
info "Deploying TrainingRegistry to Hardhat in-process network..."

# Install npm deps if missing
if [ ! -d "$PROJECT_DIR/node_modules" ]; then
  info "Installing npm dependencies..."
  (cd "$PROJECT_DIR" && npm install --save-dev "hardhat@^2.22.0" "@nomicfoundation/hardhat-toolbox@hh2") \
    || fail "npm install failed"
fi

export MERKLE_ROOT="$MERKLE_ROOT"
DEPLOY_OUTPUT=$(cd "$PROJECT_DIR" && npx hardhat run scripts/deploy.js 2>&1) \
  || fail "Contract deployment failed:\n$DEPLOY_OUTPUT"

CONTRACT_ADDR=$(echo "$DEPLOY_OUTPUT" | grep -oP 'deployed at: \K0x[a-fA-F0-9]+' || echo "N/A")
TX_HASH=$(echo "$DEPLOY_OUTPUT" | grep -oP 'committed in tx: \K0x[a-fA-F0-9]+' || echo "N/A")
IS_COMMITTED=$(echo "$DEPLOY_OUTPUT" | grep -oP 'isCommitted = \K\w+' || echo "N/A")

echo ""
echo -e "    ${GREEN}Contract Address${NC} : ${YELLOW}${CONTRACT_ADDR}${NC}"
echo -e "    ${GREEN}Commit Tx Hash${NC}   : ${YELLOW}${TX_HASH}${NC}"
echo -e "    ${GREEN}On-Chain Verified${NC} : ${YELLOW}${IS_COMMITTED}${NC}"
echo ""
success "Root committed on-chain!"

# ============================ STEP 3 ================================
header "Step 3: Inject Merkle Root into Witness JSON (Python)"
info "Running inject_root.py to create step_witness_v2.json..."
$PYTHON3 "$PROJECT_DIR/scripts/inject_root.py" \
  --root "$MERKLE_ROOT" \
  --witness "$WITNESS_V1" \
  --output "$WITNESS_V2" \
  || fail "Root injection failed"
echo ""
echo -e "    ${GREEN}Injected fields${NC}:"
echo -e "      • merkle_root   : ${YELLOW}${MERKLE_ROOT:0:20}...${NC}"
echo -e "      • raw_data      : ${YELLOW}${ELEMENT_COUNT} elements${NC}"
echo -e "      • merkle_path   : ${YELLOW}[] (full root recomputation used)${NC}"
echo ""
success "Augmented witness → step_witness_v2.json"

# ============================ STEP 4 ================================
header "Step 4: Compile & Generate ZK Proof of Training"
info "Building Rust release binary (generate_sample_proof)..."
# Build may emit warnings on stderr; check exit code explicitly
if ! $CARGO_EXE build --release --bin generate_sample_proof 2>/dev/null; then
  fail "cargo build failed for generate_sample_proof"
fi
success "Binary compiled."

info "Generating SNARK proof from augmented witness..."
echo -e "    ${CYAN}(This runs: algebraic checks + data-binding verification + SNARK proving)${NC}"
echo ""

# Run the proof generator binary directly (already built above).
# Use the same relative-path approach as poseidon_field for Windows interop.
PROOF_BIN="$PROJECT_DIR/target/release/generate_sample_proof"
if [ ! -f "$PROOF_BIN" ] && [ -f "${PROOF_BIN}.exe" ]; then
  PROOF_BIN="${PROOF_BIN}.exe"
fi

PROOF_OUTPUT=$("$PROOF_BIN" "step_witness_v2.json" 2>&1) \
  || fail "Proof generation failed:\n$PROOF_OUTPUT"

echo "$PROOF_OUTPUT" | while IFS= read -r line; do
  if echo "$line" | grep -qi "warning"; then
    echo -e "    ${YELLOW}⚠ ${line}${NC}"
  elif echo "$line" | grep -qi "passed\|success\|proof_bytes"; then
    echo -e "    ${GREEN}✓ ${line}${NC}"
  else
    echo -e "    ${CYAN}  ${line}${NC}"
  fi
done

PROOF_LEN=$(echo "$PROOF_OUTPUT" | grep -oP 'proof_bytes len = \K[0-9]+' || echo "?")
echo ""

# ============================ FINAL ================================
header "Pipeline Complete!"
echo -e "  ${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "  ${GREEN}║                                                               ║${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ Poseidon Merkle Root computed from training data${NC}          ${GREEN}║${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ Root committed to TrainingRegistry smart contract${NC}         ${GREEN}║${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ Witness augmented with data-binding fields${NC}                ${GREEN}║${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ ZK-SNARK proof generated (${PROOF_LEN} bytes)${NC}                      ${GREEN}║${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ Data Binding constraint satisfied!${NC}                        ${GREEN}║${NC}"
echo -e "  ${GREEN}║                                                               ║${NC}"
echo -e "  ${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}Artifacts${NC}:"
echo -e "    • Merkle root    : merkle_root.txt"
echo -e "    • Witness (v1)   : step_witness_v1.json"
echo -e "    • Witness (v2)   : step_witness_v2.json"
echo -e "    • Deployment     : deployment_info.json"
echo -e "    • Proof          : proofs_training"
echo ""
success "Data Binding Verified — Proof of Training is sound! 🎓"
