// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title TrainingRegistry
/// @notice Stores Poseidon Merkle roots of training datasets, mapped to client
///         addresses.  A root must be committed *before* a proof of training
///         can reference that dataset.
contract TrainingRegistry {
    /// client => list of committed dataset roots
    mapping(address => bytes32[]) private _roots;

    /// Quick existence check: (client, root) => committed?
    mapping(address => mapping(bytes32 => bool)) private _committed;

    event DatasetCommitted(address indexed client, bytes32 indexed root, uint256 timestamp);

    /// @notice Commit a dataset Merkle root for msg.sender.
    /// @param root The 32-byte Poseidon Merkle root produced off-chain.
    function commitRoot(bytes32 root) external {
        require(root != bytes32(0), "TrainingRegistry: zero root");
        require(!_committed[msg.sender][root], "TrainingRegistry: already committed");

        _committed[msg.sender][root] = true;
        _roots[msg.sender].push(root);

        emit DatasetCommitted(msg.sender, root, block.timestamp);
    }

    /// @notice Check whether a root has been committed by a given client.
    function isCommitted(address client, bytes32 root) external view returns (bool) {
        return _committed[client][root];
    }

    /// @notice Return all roots committed by a client.
    function getRoots(address client) external view returns (bytes32[] memory) {
        return _roots[client];
    }
}
