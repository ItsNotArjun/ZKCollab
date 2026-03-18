#!/usr/bin/env bash
# ===================================================================
#  ZKCollab Verifier Pipeline — Visual Execution Script
# ===================================================================

set -euo pipefail

PROOF_PATH=""
WITNESS_PATH=""
DEPLOYMENT_PATH=""
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
  echo "Usage: bash scripts/verifier.sh --proof <proof.bin> --witness <witness_augmented.json> --deployment <deployment_info.json> [--output <output_dir>]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --proof) PROOF_PATH="$2"; shift 2 ;;
    --witness) WITNESS_PATH="$2"; shift 2 ;;
    --deployment) DEPLOYMENT_PATH="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done

[[ -n "$PROOF_PATH" && -n "$WITNESS_PATH" && -n "$DEPLOYMENT_PATH" ]] || fail "Missing required arguments"
[[ -f "$PROOF_PATH" ]] || fail "Proof file not found: $PROOF_PATH"
[[ -f "$WITNESS_PATH" ]] || fail "Witness file not found: $WITNESS_PATH"
[[ -f "$DEPLOYMENT_PATH" ]] || fail "Deployment file not found: $DEPLOYMENT_PATH"

PROOF_ABS="$(cd "$(dirname "$PROOF_PATH")" && pwd)/$(basename "$PROOF_PATH")"
WITNESS_ABS="$(cd "$(dirname "$WITNESS_PATH")" && pwd)/$(basename "$WITNESS_PATH")"
DEPLOYMENT_ABS="$(cd "$(dirname "$DEPLOYMENT_PATH")" && pwd)/$(basename "$DEPLOYMENT_PATH")"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(dirname "$PROOF_ABS")"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_ABS="$(cd "$OUTPUT_DIR" && pwd)"
VERIFICATION_REPORT_FILE="$OUTPUT_ABS/verification_report.txt"
VERIFICATION_CONFIRMATION_FILE="$OUTPUT_ABS/verification_confirmation.json"

POSEIDON_BIN="$PROJECT_DIR/target/release/poseidon_field"
if [[ ! -f "$POSEIDON_BIN" && -f "${POSEIDON_BIN}.exe" ]]; then
  POSEIDON_BIN="${POSEIDON_BIN}.exe"
fi
[[ -x "$POSEIDON_BIN" ]] || fail "poseidon binary not executable: $POSEIDON_BIN"

relpath_from_project() {
  python3 - <<'PY' "$PROJECT_DIR" "$1"
import os, sys
print(os.path.relpath(sys.argv[2], sys.argv[1]))
PY
}

TMP_DIR="$PROJECT_DIR/.verify_tmp_$$"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

VERIFICATION_PASSED=true

cd "$PROJECT_DIR"

header "ZKCollab  —  Verifier Pipeline"
echo -e "  ${CYAN}Role${NC}        : Verification Client (Verifier)"
echo -e "  ${CYAN}Pipeline${NC}    : Proof + Witness + Deployment → Recompute + Validate"
echo ""

header "Step 1: Read Deployment & Commitment Info"
DEPLOYED_ROOT="$(python3 - <<'PY' "$DEPLOYMENT_ABS"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    d = json.load(f)
print(d.get('merkle_root', ''))
PY
)"
IS_COMMITTED="$(python3 - <<'PY' "$DEPLOYMENT_ABS"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    d = json.load(f)
print(str(d.get('is_committed', '')).lower())
PY
)"
[[ -n "$DEPLOYED_ROOT" ]] || fail "deployment merkle_root missing"
[[ "$IS_COMMITTED" == "true" ]] || fail "deployment is_committed is not true"
success "On-chain commitment metadata is valid"

header "Step 2: Recompute Poseidon Merkle Root from Witness"
FLAT_DATA="$TMP_DIR/flat_data.json"
COMPUTED_ROOT_FILE="$TMP_DIR/computed_root.txt"
python3 - <<'PY' "$WITNESS_ABS" "$FLAT_DATA"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    w = json.load(f)
if 'x' not in w:
    raise SystemExit('witness missing x field')
json.dump(w['x'], open(sys.argv[2], 'w', encoding='utf-8'))
PY

REL_FLAT="$(relpath_from_project "$FLAT_DATA")"
REL_COMPUTED="$(relpath_from_project "$COMPUTED_ROOT_FILE")"
"$POSEIDON_BIN" "$REL_FLAT" "$REL_COMPUTED" 2>/dev/null || fail "Poseidon root recomputation failed"
COMPUTED_ROOT="$(tr -d '[:space:]' < "$COMPUTED_ROOT_FILE")"
if [[ "$COMPUTED_ROOT" != "$DEPLOYED_ROOT" ]]; then
  VERIFICATION_PASSED=false
fi

echo ""
echo -e "    ${GREEN}Deployed Root${NC} : ${YELLOW}${DEPLOYED_ROOT}${NC}"
echo -e "    ${GREEN}Computed Root${NC} : ${YELLOW}${COMPUTED_ROOT}${NC}"
echo ""

header "Step 3: Validate Witness Binding Fields"
WITNESS_ROOT="$(python3 - <<'PY' "$WITNESS_ABS"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    w = json.load(f)
print(w.get('merkle_root', ''))
PY
)"
python3 - <<'PY' "$WITNESS_ABS"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    w = json.load(f)
required = ['merkle_root', 'merkle_path', 'x', 'grad_x', 'grad_w1', 'grad_w2']
missing = [k for k in required if k not in w]
if missing:
    raise SystemExit('missing fields: ' + ','.join(missing))
PY

[[ "$WITNESS_ROOT" == "$DEPLOYED_ROOT" ]] || VERIFICATION_PASSED=false
success "Witness fields are present"

header "Step 4: Validate Proof Artifact"
PROOF_SIZE="$(python3 - <<'PY' "$PROOF_ABS"
import os, sys
print(os.path.getsize(sys.argv[1]))
PY
)"
[[ "$PROOF_SIZE" -gt 0 ]] || fail "Proof file is empty"

if [[ "$COMPUTED_ROOT" != "$DEPLOYED_ROOT" ]]; then
  fail "Computed root does not match deployed root"
fi
if [[ "$WITNESS_ROOT" != "$DEPLOYED_ROOT" ]]; then
  fail "Witness merkle_root does not match deployed root"
fi

header "Verifier Complete!"
echo -e "  ${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ Deployment commitment is present${NC}                           ${GREEN}║${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ Poseidon root recomputation matches committed root${NC}         ${GREEN}║${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ Witness data-binding fields are valid${NC}                    ${GREEN}║${NC}"
echo -e "  ${GREEN}║${NC}   ${BOLD}✓ Proof artifact exists and is non-empty${NC}                  ${GREEN}║${NC}"
echo -e "  ${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

cat > "$VERIFICATION_REPORT_FILE" << EOF
ZKCollab Verifier Confirmation
==============================
deployed_root: $DEPLOYED_ROOT
computed_root: $COMPUTED_ROOT
witness_root: $WITNESS_ROOT
proof_file: $PROOF_ABS
proof_size_bytes: $PROOF_SIZE
result: PASSED
EOF

cat > "$VERIFICATION_CONFIRMATION_FILE" << EOF
{
  "result": "PASSED",
  "deployed_root": "$DEPLOYED_ROOT",
  "computed_root": "$COMPUTED_ROOT",
  "witness_root": "$WITNESS_ROOT",
  "proof_file": "$PROOF_ABS",
  "proof_size_bytes": $PROOF_SIZE
}
EOF

echo -e "  ${CYAN}Artifacts${NC}:"
echo -e "    • Verification report      : $VERIFICATION_REPORT_FILE"
echo -e "    • Verification confirmation: $VERIFICATION_CONFIRMATION_FILE"
echo ""
success "Verification confirmation exported."
