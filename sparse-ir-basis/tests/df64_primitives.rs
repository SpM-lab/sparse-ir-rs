//! Df64 primitive micro-benchmarks (experiment for SpM-lab/sparse-ir-rs#330).
//!
//! ```text
//! cargo test -p sparse-ir-basis --release --test df64_primitives -- --ignored --nocapture
//! # and, in a *clean* target dir so that xprec itself is rebuilt:
//! CARGO_TARGET_DIR=/tmp/fma-target RUSTFLAGS="-C target-feature=+fma" \
//!   cargo test -p sparse-ir-basis --release --test df64_primitives -- --ignored --nocapture
//! ```
//!
//! Latency and throughput are measured separately: the QR/matvec inner loops
//! are reductions with independent elements, so throughput is what matters,
//! while a single accumulator chain measures latency.

use sparse_ir_basis::{CustomNumeric, Df64};
use std::time::Instant;

const N: usize = 2_000_000;
const CHAINS: usize = 8;

fn ns(t: f64, n: usize) -> f64 {
    t / n as f64 * 1e9
}

#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn df64_primitives() {
    let xs: Vec<Df64> = (0..N)
        .map(|i| Df64::from((i as f64) * 1e-7 - 0.5))
        .collect();
    let ys: Vec<Df64> = (0..N)
        .map(|i| Df64::from((i as f64) * 1e-9 + 0.25))
        .collect();
    let x64: Vec<f64> = xs.iter().map(|x| x.to_f64()).collect();
    let y64: Vec<f64> = ys.iter().map(|y| y.to_f64()).collect();

    // --- f64 floor ---
    let t = Instant::now();
    let mut acc = 0.0f64;
    for i in 0..N {
        acc += x64[i] * y64[i];
    }
    let f64_madd_lat = ns(t.elapsed().as_secs_f64(), N);
    let t = Instant::now();
    let mut a = [0.0f64; CHAINS];
    for i in (0..N).step_by(CHAINS) {
        for k in 0..CHAINS {
            a[k] += x64[i + k] * y64[i + k];
        }
    }
    let f64_madd_thr = ns(t.elapsed().as_secs_f64(), N);
    std::hint::black_box((acc, a));

    // --- Df64 latency (one chain) ---
    let t = Instant::now();
    let mut accd = Df64::from(0.0);
    for i in 0..N {
        accd += xs[i] * ys[i];
    }
    let dd_madd_lat = ns(t.elapsed().as_secs_f64(), N);
    let t = Instant::now();
    let mut acca = Df64::from(0.0);
    for i in 0..N {
        acca += xs[i];
    }
    let dd_add_lat = ns(t.elapsed().as_secs_f64(), N);
    let t = Instant::now();
    let mut accm = Df64::from(1.0);
    for i in 0..N {
        accm *= ys[i];
    }
    let dd_mul_lat = ns(t.elapsed().as_secs_f64(), N);

    // --- Df64 throughput (independent chains) ---
    let t = Instant::now();
    let mut a = [Df64::from(0.0); CHAINS];
    for i in (0..N).step_by(CHAINS) {
        for k in 0..CHAINS {
            a[k] += xs[i + k] * ys[i + k];
        }
    }
    let dd_madd_thr = ns(t.elapsed().as_secs_f64(), N);
    let t = Instant::now();
    let mut b = [Df64::from(0.0); CHAINS];
    for i in (0..N).step_by(CHAINS) {
        for k in 0..CHAINS {
            b[k] += xs[i + k];
        }
    }
    let dd_add_thr = ns(t.elapsed().as_secs_f64(), N);
    std::hint::black_box((accd, acca, accm, a, b));

    println!("\nper-op nanoseconds ({N} iterations, {CHAINS} chains)");
    println!("  f64   mul+add : latency {f64_madd_lat:5.2}   throughput {f64_madd_thr:5.2}");
    println!("  Df64  add     : latency {dd_add_lat:5.2}   throughput {dd_add_thr:5.2}");
    println!("  Df64  mul     : latency {dd_mul_lat:5.2}");
    println!("  Df64  mul+add : latency {dd_madd_lat:5.2}   throughput {dd_madd_thr:5.2}");
    println!(
        "  Df64/f64 throughput ratio for mul+add = {:.1}x  (a double-double mul+add needs ~7-11 f64 ops)",
        dd_madd_thr / f64_madd_thr
    );
}

/// End-to-end SVE cost with the same binary, so the effect of `+fma` on the
/// real workload (Df64 QR + SVD) can be read off directly.
#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn sve_time() {
    use sparse_ir_basis::{LogisticKernel, TworkType, compute_sve};
    use std::time::Instant;
    for lambda in [1e5_f64, 1e6] {
        let t = Instant::now();
        let sve = compute_sve(
            LogisticKernel::new(lambda).unwrap(),
            Some(1e-10),
            None,
            None,
            TworkType::Auto,
        )
        .unwrap();
        println!(
            "compute_sve Lambda={lambda}: {:6.2} s  ({} singular values)",
            t.elapsed().as_secs_f64(),
            sve.s().len()
        );
    }
}
