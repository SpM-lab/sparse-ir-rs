//! Xorshift64 pseudo-random number generator
//!
//! Simple, fast, and portable PRNG that works identically in Rust and C++.
//! Generates uniformly distributed uint64_t values, which can be converted to [0, 1).
//!
//! This is a deterministic PRNG suitable for reproducible test data.
//! The same seed will always produce the same sequence.
//!
//! # Example
//! ```cpp
//! Xorshift64 rng(12345);
//! double x = rng.next_f64(); // [0, 1)
//! ```

#ifndef XORSHIFT64_HPP
#define XORSHIFT64_HPP

#include <cstdint>

class Xorshift64 {
public:
    /// Create a new Xorshift64 generator with the given seed
    ///
    /// @param seed Initial seed (0 is allowed, but 0 will always produce 0)
    explicit Xorshift64(uint64_t seed) {
        // If seed is 0, use a non-zero default to avoid all-zero state
        state_ = (seed == 0) ? 0x853c49e6748fea9bULL : seed;
    }

    /// Generate next uint64_t value
    uint64_t next_u64() {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        return state_;
    }

    /// Generate next double in range [0, 1)
    ///
    /// Uses the upper 53 bits of the uint64_t (IEEE 754 double precision mantissa)
    /// to ensure good distribution in [0, 1).
    double next_f64() {
        // Use upper 53 bits (52 mantissa + 1 implicit bit) for [0, 1)
        // This gives 2^53 distinct values in [0, 1)
        uint64_t bits = next_u64();
        // Extract upper 53 bits and normalize to [0, 1)
        return static_cast<double>(bits >> 11) / static_cast<double>(1ULL << 53);
    }

    /// Generate next double in range [0, 1] (inclusive upper bound)
    ///
    /// Note: This includes 1.0, which may be useful for some applications.
    double next_f64_inclusive() {
        // Use all 64 bits, but normalize to [0, 1] including 1.0
        uint64_t bits = next_u64();
        return static_cast<double>(bits) / static_cast<double>(UINT64_MAX);
    }

private:
    uint64_t state_;
};

#endif // XORSHIFT64_HPP

