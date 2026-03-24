// poseidon_field.cpp — Dimension-agnostic Poseidon commitment engine.
//
// Reads a JSON file containing a flat array of integers (a flattened tensor
// of *any* length N), hashes the elements using a Poseidon-like sponge
// construction over a 256-bit prime field, and prints the resulting 32-byte
// Merkle root as a hex string.
//
// Build:  g++ -std=c++17 -O2 -o poseidon_field src/poseidon_field.cpp
// Usage:  ./poseidon_field <input.json> [output.txt]
//
// The JSON must be a flat array of integers, e.g. [1, -3, 42, 7, ...].

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Lightweight 256-bit unsigned integer (little-endian limbs)
// ---------------------------------------------------------------------------
struct U256 {
    uint64_t limbs[4] = {0, 0, 0, 0};
};

static bool ge(const U256 &a, const U256 &b) {
    for (int i = 3; i >= 0; --i) {
        if (a.limbs[i] != b.limbs[i])
            return a.limbs[i] > b.limbs[i];
    }
    return true; // equal
}

static U256 add256(const U256 &a, const U256 &b) {
    U256 r;
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t s = (__uint128_t)a.limbs[i] + b.limbs[i] + carry;
        r.limbs[i] = (uint64_t)s;
        carry = (uint64_t)(s >> 64);
    }
    return r;
}

static U256 sub256(const U256 &a, const U256 &b) {
    U256 r;
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t s = (__uint128_t)a.limbs[i] - b.limbs[i] - borrow;
        r.limbs[i] = (uint64_t)s;
        borrow = (s >> 127) ? 1 : 0;
    }
    return r;
}

// BN254 scalar field prime:
// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
static U256 BN254_P() {
    U256 p;
    p.limbs[0] = 0x43e1f593f0000001ULL;
    p.limbs[1] = 0x2833e84879b97091ULL;
    p.limbs[2] = 0xb85045b68181585dULL;
    p.limbs[3] = 0x30644e72e131a029ULL;
    return p;
}

static U256 mod_reduce(U256 v) {
    U256 p = BN254_P();
    while (ge(v, p))
        v = sub256(v, p);
    return v;
}

static U256 mod_add(const U256 &a, const U256 &b) {
    return mod_reduce(add256(a, b));
}

static U256 mod_mul(const U256 &a, const U256 &b) {
    // Schoolbook 256x256 -> 512, then Barrett-like reduction via repeated
    // subtraction (sufficient for our hash use-case; not constant-time but
    // this is *not* a secret-key operation).
    __uint128_t partials[8] = {};
    for (int i = 0; i < 4; ++i) {
        __uint128_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            __uint128_t prod = (__uint128_t)a.limbs[i] * b.limbs[j] + partials[i + j] + carry;
            partials[i + j] = (uint64_t)prod;
            carry = prod >> 64;
        }
        partials[i + 4] += carry;
    }
    // Collect into 8 limbs, then reduce modulo p by repeated subtraction.
    // We cap iterations to avoid infinite loops on any edge case.
    U256 hi, lo;
    lo.limbs[0] = (uint64_t)partials[0]; lo.limbs[1] = (uint64_t)partials[1];
    lo.limbs[2] = (uint64_t)partials[2]; lo.limbs[3] = (uint64_t)partials[3];
    hi.limbs[0] = (uint64_t)partials[4]; hi.limbs[1] = (uint64_t)partials[5];
    hi.limbs[2] = (uint64_t)partials[6]; hi.limbs[3] = (uint64_t)partials[7];

    // Reduce: result = (hi * 2^256 + lo) mod p.
    // 2^256 mod p is a known constant.  We compute hi * R mod p + lo mod p.
    // R = 2^256 mod p = p + 1 - p = ... We just do repeated subtraction
    // which is simple and correct.
    U256 p = BN254_P();

    // r = hi * (2^256 mod p)
    // 2^256 mod p can be pre-computed, but for simplicity we build it:
    // Since p < 2^254, 2^256 mod p fits in 256 bits.
    // 2^256 mod p = 2^256 - p  (since 2^256 = 1*p + r where r < p)
    // but 2^256 is > 4*p so we need the actual value.  Let's just compute
    // a full 512-bit mod via simple long-division-style reduction.

    // ------ Simple approach: treat 512-bit number and subtract p ------
    // Store as 8x64 and repeatedly subtract p<<(64*i).
    uint64_t full[8];
    for (int i = 0; i < 4; ++i) full[i] = lo.limbs[i];
    for (int i = 0; i < 4; ++i) full[i + 4] = hi.limbs[i];

    // Reduce from the top.  p fits in ~254 bits (limbs[3] < 2^62).
    // We subtract p * q for appropriate q at each 64-bit position from
    // the top down.
    for (int shift = 4; shift >= 0; --shift) {
        // While full[shift..shift+4] >= p, subtract p << (64*shift)
        for (int iter = 0; iter < 100; ++iter) {
            // Check if portion >= p
            bool can = false;
            for (int k = 3; k >= 0; --k) {
                uint64_t fv = (shift + k < 8) ? full[shift + k] : 0;
                if (fv > p.limbs[k]) { can = true; break; }
                if (fv < p.limbs[k]) { can = false; break; }
                if (k == 0) can = true; // equal
            }
            if (!can) break;
            uint64_t borrow = 0;
            for (int k = 0; k < 4 && shift + k < 8; ++k) {
                __uint128_t s = (__uint128_t)full[shift + k] - p.limbs[k] - borrow;
                full[shift + k] = (uint64_t)s;
                borrow = (s >> 127) ? 1 : 0;
            }
        }
    }
    U256 result;
    for (int i = 0; i < 4; ++i) result.limbs[i] = full[i];
    return mod_reduce(result);
}

static U256 mod_pow(U256 base, uint64_t exp) {
    U256 result;
    result.limbs[0] = 1;
    base = mod_reduce(base);
    while (exp > 0) {
        if (exp & 1)
            result = mod_mul(result, base);
        base = mod_mul(base, base);
        exp >>= 1;
    }
    return result;
}

static U256 from_int64(int64_t v) {
    U256 r;
    if (v >= 0) {
        r.limbs[0] = (uint64_t)v;
    } else {
        // Represent as p - |v|
        U256 p = BN254_P();
        U256 abs_v;
        abs_v.limbs[0] = (uint64_t)(-v);
        r = sub256(p, abs_v);
    }
    return r;
}

static std::string to_hex32(const U256 &v) {
    // Print as 64-char hex (big-endian bytes, i.e., 0x prefix style).
    std::ostringstream oss;
    oss << "0x";
    for (int i = 3; i >= 0; --i) {
        oss << std::hex << std::setfill('0') << std::setw(16)
            << v.limbs[i];
    }
    return oss.str();
}

// ---------------------------------------------------------------------------
// Poseidon-like sponge (width = 3, alpha = 5, simplified round constants)
// ---------------------------------------------------------------------------
// NOTE: This uses representative round constants. For production, use the
// canonical Poseidon constants generated by the official script for BN254.

static const int FULL_ROUNDS = 8;
static const int PARTIAL_ROUNDS = 57;
static const int WIDTH = 3;

static U256 round_constants[WIDTH * (FULL_ROUNDS + PARTIAL_ROUNDS)];
static bool rc_initialized = false;

static void init_round_constants() {
    if (rc_initialized) return;
    // Deterministic pseudo-random generation of round constants.
    // seed: "ZKCollab_Poseidon_RC"
    U256 state;
    state.limbs[0] = 0x5a4b436f6c6c6162ULL; // "ZKCollab"
    state.limbs[1] = 0x506f736569646f6eULL; // "Poseidon"
    state.limbs[2] = 0x524300000000ULL;       // "RC"
    state.limbs[3] = 0;
    int total = WIDTH * (FULL_ROUNDS + PARTIAL_ROUNDS);
    for (int i = 0; i < total; ++i) {
        state = mod_add(mod_mul(state, state), from_int64(i + 1));
        round_constants[i] = mod_reduce(state);
    }
    rc_initialized = true;
}

static void sbox(U256 &v) {
    // x^5
    U256 v2 = mod_mul(v, v);
    U256 v4 = mod_mul(v2, v2);
    v = mod_mul(v4, v);
}

static void poseidon_permutation(U256 state[WIDTH]) {
    init_round_constants();
    int rc_idx = 0;
    int half_full = FULL_ROUNDS / 2;
    // First half full rounds
    for (int r = 0; r < half_full; ++r) {
        for (int i = 0; i < WIDTH; ++i)
            state[i] = mod_add(state[i], round_constants[rc_idx++]);
        for (int i = 0; i < WIDTH; ++i)
            sbox(state[i]);
        // MDS mix (simple: state[i] = sum(state[j] * MDS[i][j]))
        // Using a Cauchy MDS for width=3:
        // [[2,1,1],[1,2,1],[1,1,2]]
        U256 tmp[WIDTH];
        for (int i = 0; i < WIDTH; ++i) {
            tmp[i] = mod_add(state[i], state[i]); // 2*self
            for (int j = 0; j < WIDTH; ++j) {
                if (j != i)
                    tmp[i] = mod_add(tmp[i], state[j]);
            }
        }
        for (int i = 0; i < WIDTH; ++i) state[i] = tmp[i];
    }
    // Partial rounds
    for (int r = 0; r < PARTIAL_ROUNDS; ++r) {
        for (int i = 0; i < WIDTH; ++i)
            state[i] = mod_add(state[i], round_constants[rc_idx++]);
        sbox(state[0]); // only first element
        U256 tmp[WIDTH];
        for (int i = 0; i < WIDTH; ++i) {
            tmp[i] = mod_add(state[i], state[i]);
            for (int j = 0; j < WIDTH; ++j) {
                if (j != i)
                    tmp[i] = mod_add(tmp[i], state[j]);
            }
        }
        for (int i = 0; i < WIDTH; ++i) state[i] = tmp[i];
    }
    // Second half full rounds
    for (int r = 0; r < half_full; ++r) {
        for (int i = 0; i < WIDTH; ++i)
            state[i] = mod_add(state[i], round_constants[rc_idx++]);
        for (int i = 0; i < WIDTH; ++i)
            sbox(state[i]);
        U256 tmp[WIDTH];
        for (int i = 0; i < WIDTH; ++i) {
            tmp[i] = mod_add(state[i], state[i]);
            for (int j = 0; j < WIDTH; ++j) {
                if (j != i)
                    tmp[i] = mod_add(tmp[i], state[j]);
            }
        }
        for (int i = 0; i < WIDTH; ++i) state[i] = tmp[i];
    }
}

// Poseidon sponge absorb-squeeze over arbitrary-length input.
// Absorb rate = WIDTH - 1 = 2 elements at a time.
static U256 poseidon_hash(const std::vector<U256> &inputs) {
    U256 state[WIDTH] = {};
    const int rate = WIDTH - 1; // 2
    size_t idx = 0;
    while (idx < inputs.size()) {
        for (int i = 0; i < rate && idx < inputs.size(); ++i, ++idx)
            state[i + 1] = mod_add(state[i + 1], inputs[idx]);
        poseidon_permutation(state);
    }
    // Domain separation: absorb input length
    state[1] = mod_add(state[1], from_int64((int64_t)inputs.size()));
    poseidon_permutation(state);
    return state[0]; // squeeze
}

// Compute a Merkle tree over the absorbed chunks and return root.
// Leaves = Poseidon(chunk), then pair-wise hashing up.
static U256 merkle_root(const std::vector<U256> &field_elements) {
    if (field_elements.empty()) {
        return poseidon_hash({});
    }
    // Build leaves: each leaf is a single field element hashed.
    std::vector<U256> leaves;
    for (auto &fe : field_elements) {
        leaves.push_back(poseidon_hash({fe}));
    }
    // Pad to next power of 2.
    size_t n = 1;
    while (n < leaves.size()) n <<= 1;
    U256 zero_hash = poseidon_hash({});
    while (leaves.size() < n)
        leaves.push_back(zero_hash);

    // Build tree bottom-up.
    while (leaves.size() > 1) {
        std::vector<U256> next;
        for (size_t i = 0; i + 1 < leaves.size(); i += 2) {
            next.push_back(poseidon_hash({leaves[i], leaves[i + 1]}));
        }
        leaves = next;
    }
    return leaves[0];
}

// ---------------------------------------------------------------------------
// JSON parser (minimal, handles flat arrays of integers)
// ---------------------------------------------------------------------------
static std::vector<int64_t> parse_json_array(const std::string &json) {
    std::vector<int64_t> result;
    bool in_array = false;
    std::string num_buf;
    for (char c : json) {
        if (c == '[') { in_array = true; continue; }
        if (c == ']') { in_array = false;
            if (!num_buf.empty()) {
                result.push_back(std::stoll(num_buf));
                num_buf.clear();
            }
            break;
        }
        if (!in_array) continue;
        if (c == ',' || c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            if (!num_buf.empty()) {
                result.push_back(std::stoll(num_buf));
                num_buf.clear();
            }
        } else {
            num_buf += c;
        }
    }
    return result;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.json> [output.txt]\n";
        return 1;
    }

    // Read input JSON
    std::ifstream ifs(argv[1]);
    if (!ifs) {
        std::cerr << "Error: cannot open " << argv[1] << "\n";
        return 1;
    }
    std::string contents((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());
    ifs.close();

    auto values = parse_json_array(contents);
    if (values.empty()) {
        std::cerr << "Error: no integers found in input JSON\n";
        return 1;
    }

    std::cerr << "Read " << values.size() << " field elements from "
              << argv[1] << "\n";

    // Convert to field elements
    std::vector<U256> field_elements;
    field_elements.reserve(values.size());
    for (auto v : values)
        field_elements.push_back(from_int64(v));

    // Compute Merkle root
    U256 root = merkle_root(field_elements);
    std::string hex_root = to_hex32(root);

    // Output
    std::cout << hex_root << std::endl;

    if (argc >= 3) {
        std::ofstream ofs(argv[2]);
        if (ofs) {
            ofs << hex_root << std::endl;
            std::cerr << "Root written to " << argv[2] << "\n";
        }
    }

    return 0;
}
