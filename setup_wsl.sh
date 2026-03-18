#!/usr/bin/env bash
# =============================================================================
#  setup_wsl.sh — One-shot WSL/Ubuntu bootstrap for the ZKCollab PoT pipeline.
#
#  Installs:
#    1. System tools   (build-essential, pkg-config, libssl-dev, curl, git)
#    2. Rust nightly   (via rustup)
#    3. Python 3       (python3, pip, venv)
#    4. Node.js 20 LTS (via NVM) + Hardhat & Ethers v6 npm packages
#
#  Usage:
#    chmod +x setup_wsl.sh
#    ./setup_wsl.sh
# =============================================================================
set -euo pipefail

# --------------------------------------------------
# Helper
# --------------------------------------------------
section() { echo -e "\n\033[1;36m[$1/$TOTAL] $2\033[0m"; }
TOTAL=4

echo "============================================"
echo "  ZKCollab — WSL Environment Setup"
echo "============================================"

# --------------------------------------------------
# 1. System packages
# --------------------------------------------------
section 1 "Updating apt and installing system dependencies"

sudo apt-get update -y
sudo apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    git \
    clang \
    cmake

echo "    System packages installed."

# --------------------------------------------------
# 2. Rust (via rustup)
# --------------------------------------------------
section 2 "Installing Rust toolchain (nightly)"

if command -v rustup &>/dev/null; then
    echo "    rustup already installed — updating..."
    rustup update nightly
    rustup default nightly
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
    # Make cargo available in the current shell session.
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

echo "    Rust version: $(rustc --version)"
echo "    Cargo version: $(cargo --version)"

# --------------------------------------------------
# 3. Python 3
# --------------------------------------------------
section 3 "Installing Python 3 environment"

sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv

# inject_root.py uses only stdlib (json, math, os, sys, argparse).
# generate_witness.py / sample_model.py may need torch — install only
# the minimal set so the data-binding pipeline works out of the box.
# If PyTorch is needed later:  pip3 install torch --index-url https://download.pytorch.org/whl/cpu
echo "    Python version: $(python3 --version)"

# --------------------------------------------------
# 4. Node.js (via NVM) + Hardhat
# --------------------------------------------------
section 4 "Installing Node.js via NVM and Hardhat dependencies"

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"

if [ ! -d "$NVM_DIR" ]; then
    echo "    Installing NVM..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
fi

# Load NVM into the current shell.
# shellcheck disable=SC1091
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

# Install Node.js 22 LTS (required by Hardhat).
if ! command -v node &>/dev/null || ! node -v | grep -q "^v22"; then
    echo "    Installing Node.js 22 LTS..."
    nvm install 22
    nvm use 22
    nvm alias default 22
fi

echo "    Node version: $(node --version)"
echo "    npm version:  $(npm --version)"

# Install Hardhat + toolbox in the project directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

if [ ! -f "$PROJECT_DIR/package.json" ]; then
    echo "    Initializing npm project..."
    (cd "$PROJECT_DIR" && npm init -y --silent)
fi

# Remove ESM type if present (Hardhat 2 uses CommonJS).
(cd "$PROJECT_DIR" && npm pkg delete type 2>/dev/null || true)

echo "    Installing Hardhat 2 and @nomicfoundation/hardhat-toolbox..."
(cd "$PROJECT_DIR" && npm install --save-dev hardhat@^2.22.0 "@nomicfoundation/hardhat-toolbox@hh2")

# --------------------------------------------------
# Done
# --------------------------------------------------
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Quick verification commands:"
echo "    rustc --version"
echo "    cargo --version"
echo "    python3 --version"
echo "    node --version"
echo "    npx hardhat --version"
echo ""
echo "  Next steps:"
echo "    1. Open a new terminal (or run: source ~/.bashrc)"
echo "    2. Start a local Hardhat node:  npx hardhat node"
echo "    3. Run the pipeline:            ./run_pipeline.sh"
echo ""
