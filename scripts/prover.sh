#!/usr/bin/env bash
# ===================================================================
#  ZKCollab Prover Pipeline — Visual Execution Script
# ===================================================================

set -euo pipefail

WITNESS_PATH=""
OUTPUT_DIR=""
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}    $*"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
fail()    { echo -e "${RED}[ERROR]${NC}   $*"; exit 1; }
header()  { echo -e "\n${BOLD}═══════════════════════════════════════════════════${NC}"; echo -e "${BOLD}  $*${NC}"; echo -e "${BOLD}═══════════════════════════════════════════════════${NC}\n"; }

usage() {
  echo "Usage: bash scripts/prover.sh --witness <witness.json> --output <output_dir>"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --witness) WITNESS_PATH="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done

[[ -n "$WITNESS_PATH" && -n "$OUTPUT_DIR" ]] || fail "Missing required arguments"
[[ -f "$WITNESS_PATH" ]] || fail "Witness file not found: $WITNESS_PATH"

mkdir -p "$OUTPUT_DIR"
WITNESS_ABS="$(cd "$(dirname "$WITNESS_PATH")" && pwd)/$(basename "$WITNESS_PATH")"
OUTPUT_ABS="$(cd "$OUTPUT_DIR" && pwd)"
FLAT_DATA="$OUTPUT_ABS/flat_data.json"
MERKLE_ROOT_FILE="$OUTPUT_ABS/merkle_root.txt"
AUGMENTED_WITNESS="$OUTPUT_ABS/witness_augmented.json"
DEPLOYMENT_FILE="$OUTPUT_ABS/deployment_info.json"
PROOF_BIN_FILE="$OUTPUT_ABS/proof.bin"
PROOF_HEX_FILE="$OUTPUT_ABS/proof_hex.txt"
PROVER_REPORT_FILE="$OUTPUT_ABS/prover_report.txt"

CARGO_EXE="$(command -v cargo 2>/dev/null || echo /mnt/c/Users/Abhishek/.cargo/bin/cargo.exe)"
POSEIDON_BIN="$PROJECT_DIR/target/release/poseidon_field"
PROOF_BIN="$PROJECT_DIR/target/release/generate_sample_proof"

if [[ ! -f "$POSEIDON_BIN" && -f "${POSEIDON_BIN}.exe" ]]; then
  POSEIDON_BIN="${POSEIDON_BIN}.exe"
fi
if [[ ! -f "$PROOF_BIN" && -f "${PROOF_BIN}.exe" ]]; then
  PROOF_BIN="${PROOF_BIN}.exe"
fi

relpath_from_project() {
  python3 - <<'PY' "$PROJECT_DIR" "$1"
import os, sys
print(os.path.relpath(sys.argv[2], sys.argv[1]))
PY
}

cd "$PROJECT_DIR"

header "ZKCollab  —  Prover Pipeline"
echo -e "  ${CYAN}Role${NC}        : Training Client (Prover)"
echo -e "  ${CYAN}Architecture${NC}: 2-layer 4×4 MLP (Linear→ReLU→Linear→ReLU)"
echo -e "  ${CYAN}Pipeline${NC}    : Witness → Poseidon Root → On-Chain Commit → ZK Proof"
echo ""

header "Step 1: Compute Poseidon Merkle Root (Rust Engine)"
info "Building Rust Poseidon engine (BN254, width=3, α=5)..."
"$CARGO_EXE" build --release --bin poseidon_field 2>/dev/null || fail "cargo build failed for poseidon_field"
success "Poseidon engine built."

info "Extracting flat tensor from witness 'x' field..."
python3 - <<'PY' "$WITNESS_ABS" "$FLAT_DATA"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    w = json.load(f)
if 'x' not in w:
    raise SystemExit("witness missing x field")
json.dump(w['x'], open(sys.argv[2], 'w', encoding='utf-8'))
print(f"Extracted {len(w['x'])} elements")
PY

info "Computing Poseidon Merkle root..."
REL_FLAT_DATA="$(relpath_from_project "$FLAT_DATA")"
REL_ROOT_FILE="$(relpath_from_project "$MERKLE_ROOT_FILE")"
"$POSEIDON_BIN" "$REL_FLAT_DATA" "$REL_ROOT_FILE" 2>/dev/null || fail "Poseidon root computation failed"
MERKLE_ROOT="$(tr -d '[:space:]' < "$MERKLE_ROOT_FILE")"
echo ""
echo -e "    ${BOLD}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "    ${BOLD}║${NC}  ${GREEN}Merkle Root${NC}: ${YELLOW}${MERKLE_ROOT}${NC}  ${BOLD}║${NC}"
echo -e "    ${BOLD}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
success "Root written to $(basename "$MERKLE_ROOT_FILE")"

header "Step 2: Deploy Smart Contract & Commit Root On-Chain"
info "Deploying TrainingRegistry to Hardhat in-process network..."
export MERKLE_ROOT
export HARDHAT_DISABLE_TELEMETRY_PROMPT=true
DEPLOY_OUTPUT="$(npx hardhat run scripts/deploy.js --network hardhat 2>&1)" || fail "Contract deployment failed"

CONTRACT_ADDR="$(echo "$DEPLOY_OUTPUT" | grep -oP 'deployed at: \K0x[a-fA-F0-9]+' | head -1 || true)"
TX_HASH="$(echo "$DEPLOY_OUTPUT" | grep -oP 'committed in tx: \K0x[a-fA-F0-9]+' | head -1 || true)"
IS_COMMITTED="$(echo "$DEPLOY_OUTPUT" | grep -oP 'isCommitted = \K\w+' | head -1 || true)"
[[ -n "$CONTRACT_ADDR" && -n "$TX_HASH" && -n "$IS_COMMITTED" ]] || fail "Could not parse deploy output"

cat > "$DEPLOYMENT_FILE" << EOF
{
  "contract_address": "$CONTRACT_ADDR",
  "deploy_tx_hash": "$TX_HASH",
  "merkle_root": "$MERKLE_ROOT",
  "is_committed": $IS_COMMITTED
}
EOF

echo ""
echo -e "    ${GREEN}Contract Address${NC}  : ${YELLOW}${CONTRACT_ADDR}${NC}"
echo -e "    ${GREEN}Commit Tx Hash${NC}    : ${YELLOW}${TX_HASH}${NC}"
echo -e "    ${GREEN}On-Chain Verified${NC} : ${YELLOW}${IS_COMMITTED}${NC}"
echo ""
success "Root committed on-chain!"

header "Step 3: Inject Merkle Root into Witness JSON (Python)"
info "Running inject_root.py to create witness_augmented.json..."
python3 scripts/inject_root.py --root "$MERKLE_ROOT" --witness "$WITNESS_ABS" --output "$AUGMENTED_WITNESS" || fail "Root injection failed"
success "Augmented witness → $(basename "$AUGMENTED_WITNESS")"

header "Step 4: Compile & Generate ZK Proof of Training"
info "Building Rust release binary (generate_sample_proof)..."
"$CARGO_EXE" build --release --bin generate_sample_proof 2>/dev/null || fail "cargo build failed for generate_sample_proof"
success "Binary compiled."

info "Generating SNARK proof from augmented witness..."
echo -e "    ${CYAN}(This runs: algebraic checks + data-binding verification + SNARK proving)${NC}"
echo ""
REL_AUGMENTED="$(relpath_from_project "$AUGMENTED_WITNESS")"
PROOF_OUTPUT="$($PROOF_BIN "$REL_AUGMENTED" 2>&1)" || fail "Proof generation failed"

echo "$PROOF_OUTPUT" | while IFS= read -r line; do
  if echo "$line" | grep -qi "passed\|proof_bytes"; then
    echo -e "    ${GREEN}✓ ${line}${NC}"
  else
    echo -e "    ${CYAN}  ${line}${NC}"
  fi
done

[[ -f "$PROJECT_DIR/proofs_training" ]] || fail "Expected proof file proofs_training not found"
cp "$PROJECT_DIR/proofs_training" "$PROOF_BIN_FILE"
xxd -p -c 120 "$PROOF_BIN_FILE" > "$PROOF_HEX_FILE"
PROOF_LEN="$(python3 - <<'PY' "$PROOF_BIN_FILE"
import os, sys
print(os.path.getsize(sys.argv[1]))
PY
)"

cat > "$PROVER_REPORT_FILE" << EOF
ZKCollab Prover Confirmation
============================
merkle_root: $MERKLE_ROOT
contract_address: $CONTRACT_ADDR
commit_tx_hash: $TX_HASH
is_committed: $IS_COMMITTED
proof_file: $PROOF_BIN_FILE
proof_hex_file: $PROOF_HEX_FILE
proof_size_bytes: $PROOF_LEN
witness_input: $WITNESS_ABS
witness_augmented: $AUGMENTED_WITNESS
EOF

header "Prover Complete!"
echo -e "  ${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ Poseidon Merkle Root computed from training data${NC}          ${GREEN}║${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ Root committed to TrainingRegistry smart contract${NC}         ${GREEN}║${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ Witness augmented with data-binding fields${NC}                ${GREEN}║${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ ZK-SNARK proof generated (${PROOF_LEN} bytes)${NC}                      ${GREEN}║${NC}"
echo -e "  ${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}Artifacts${NC}:"
echo -e "    • Merkle root      : $MERKLE_ROOT_FILE"
echo -e "    • Deployment       : $DEPLOYMENT_FILE"
echo -e "    • Witness (aug)    : $AUGMENTED_WITNESS"
echo -e "    • Proof (binary)   : $PROOF_BIN_FILE"
echo -e "    • Proof (hex)      : $PROOF_HEX_FILE"
echo -e "    • Prover report    : $PROVER_REPORT_FILE"
echo ""
success "Proof generated and exported for verifier client."
