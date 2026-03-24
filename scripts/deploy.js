// scripts/deploy.js — Deploy TrainingRegistry to localhost and commit a root.
//
// Usage:
//   npx hardhat run scripts/deploy.js --network localhost
//
// Expects MERKLE_ROOT env var (hex string) to commit after deployment.

const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
    // Read Merkle root from env or from a file produced by the C++ engine.
    let rootHex = process.env.MERKLE_ROOT;
    if (!rootHex) {
        const rootFile = path.resolve(__dirname, "..", "merkle_root.txt");
        if (fs.existsSync(rootFile)) {
            rootHex = fs.readFileSync(rootFile, "utf-8").trim();
        }
    }
    if (!rootHex) {
        throw new Error(
            "No MERKLE_ROOT env variable set and no merkle_root.txt found."
        );
    }

    // Ensure proper bytes32 padding (left-pad to 66 chars = 0x + 64 hex).
    if (!rootHex.startsWith("0x")) rootHex = "0x" + rootHex;
    rootHex = rootHex.toLowerCase();
    // Pad to 32 bytes if shorter (shouldn't happen but defensive).
    const hexBody = rootHex.slice(2).padStart(64, "0");
    const root = "0x" + hexBody;

    console.log(`Deploying TrainingRegistry...`);
    const Factory = await ethers.getContractFactory("TrainingRegistry");
    const registry = await Factory.deploy();
    await registry.waitForDeployment();
    const addr = await registry.getAddress();
    console.log(`TrainingRegistry deployed at: ${addr}`);

    console.log(`Committing Merkle root: ${root}`);
    const tx = await registry.commitRoot(root);
    await tx.wait();
    console.log(`Root committed in tx: ${tx.hash}`);

    // Verify on-chain.
    const [signer] = await ethers.getSigners();
    const ok = await registry.isCommitted(signer.address, root);
    console.log(`On-chain verification: isCommitted = ${ok}`);

    // Write deployment info for downstream consumption.
    const info = {
        contractAddress: addr,
        merkleRoot: root,
        deployer: signer.address,
        network: "localhost",
    };
    const outPath = path.resolve(__dirname, "..", "deployment_info.json");
    fs.writeFileSync(outPath, JSON.stringify(info, null, 2) + "\n");
    console.log(`Deployment info written to ${outPath}`);
}

main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
});
