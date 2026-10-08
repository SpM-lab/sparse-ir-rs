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

// --- A portable (FMA-free) double-double multiply-add, Dekker/Veltkamp style ---

const SPLIT: f64 = 134_217_729.0; // 2^27 + 1

#[inline]
fn split(a: f64) -> (f64, f64) {
    let t = SPLIT * a;
    let hi = t - (t - a);
    (hi, a - hi)
}

#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

#[inline]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

#[inline]
fn dd_muladd(x: (f64, f64), y: (f64, f64), c: (f64, f64)) -> (f64, f64) {
    let (p1, p2) = two_prod(x.0, y.0);
    let p2 = p2 + x.0 * y.1 + x.1 * y.0;
    let (s1, s2) = two_sum(c.0, p1);
    let s2 = s2 + c.1 + p2;
    quick_two_sum(s1, s2)
}

/// `acc = acc + x*y` for Df64 vs a portable FMA-free double-double.
#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn dekker_vs_fma() {
    let xs: Vec<Df64> = (0..N)
        .map(|i| Df64::from((i as f64) * 1e-7 - 0.5))
        .collect();
    let ys: Vec<Df64> = (0..N)
        .map(|i| Df64::from((i as f64) * 1e-9 + 0.25))
        .collect();
    let xf: Vec<(f64, f64)> = xs.iter().map(|x| (x.hi(), x.lo())).collect();
    let yf: Vec<(f64, f64)> = ys.iter().map(|y| (y.hi(), y.lo())).collect();

    let t = Instant::now();
    let mut a = [Df64::from(0.0); CHAINS];
    for i in (0..N).step_by(CHAINS) {
        for k in 0..CHAINS {
            a[k] += xs[i + k] * ys[i + k];
        }
    }
    let df64 = ns(t.elapsed().as_secs_f64(), N);

    let t = Instant::now();
    let mut c = [(0.0f64, 0.0f64); CHAINS];
    for i in (0..N).step_by(CHAINS) {
        for k in 0..CHAINS {
            c[k] = dd_muladd(xf[i + k], yf[i + k], c[k]);
        }
    }
    let dekker = ns(t.elapsed().as_secs_f64(), N);
    std::hint::black_box((a, c));
    println!("\nDf64 acc += x*y: xprec ops {df64:.2} ns   portable Dekker {dekker:.2} ns");
}

/// Correctness of the portable path: the Dekker multiply-add must agree with
/// xprec's `c + x*y` to double-double rounding.
#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn dekker_equivalence() {
    let mut seed = 12345u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let mut worst: f64 = 0.0;
    let mut worst_abs: f64 = 0.0;
    for _ in 0..200_000 {
        let x = Df64::from(rnd() * 10.0f64.powi((rnd() * 10.0) as i32));
        let y = Df64::from(rnd() * 10.0f64.powi((rnd() * 10.0) as i32));
        let c = Df64::from(rnd());
        let got = dd_muladd((x.hi(), x.lo()), (y.hi(), y.lo()), (c.hi(), c.lo()));
        let want = c + x * y;
        let d = ((got.0 + got.1) - want.to_f64()).abs();
        let scale = (want.to_f64()).abs().max(1e-30);
        worst = worst.max(d / scale);
        worst_abs = worst_abs.max(d);
    }
    println!(
        "\nDekker dd_muladd vs xprec c + x*y over 200k random triples: worst abs {worst_abs:.3e}, worst rel {worst:.3e} (Df64 eps = 2.5e-32)"
    );
}
