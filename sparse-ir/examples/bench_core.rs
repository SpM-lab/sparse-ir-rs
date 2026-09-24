//! Baseline performance benchmark for basis generation and fitting.
//!
//! Used to guard against regressions when swapping the linear-algebra layer
//! (see issue #259). Reports the median wall time over repeated runs.
//!
//! Run with: `cargo run --release --example bench_core [-- --quick]`
//! Pin BLAS/rayon threads externally (e.g. `RAYON_NUM_THREADS=1`) for stable numbers.

use mdarray::{DynRank, Tensor};
use num_complex::Complex;
use sparse_ir::{
    DiscreteLehmannRepresentation, Fermionic, FiniteTempBasis, LogisticKernel, MatsubaraSampling,
    TauSampling,
    sve::{TworkType, compute_sve},
};
use std::hint::black_box;
use std::time::Instant;

fn median_ms<F: FnMut()>(nrun: usize, mut f: F) -> f64 {
    f(); // warm-up
    let mut t: Vec<f64> = (0..nrun)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[nrun / 2]
}

fn report(name: &str, ms: f64) {
    println!("{name:<55} {ms:>12.4} ms");
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let nrun_heavy = if quick { 3 } else { 7 };
    let nrun_light = if quick { 11 } else { 51 };
    let beta = 1.0;

    // --- Basis generation (SVE + basis) ---
    for &lambda in &[1e1, 1e3, 1e5] {
        for &(eps, twork) in &[
            (1e-6, TworkType::Float64),
            (1e-10, TworkType::Float64X2),
            (1e-15, TworkType::Float64X2),
        ] {
            let ms = median_ms(nrun_heavy, || {
                black_box(compute_sve(
                    LogisticKernel::new(lambda),
                    eps,
                    None,
                    None,
                    twork,
                ));
            });
            report(
                &format!("sve       Λ={lambda:.0e} ε={eps:.0e} {twork:?}"),
                ms,
            );
        }
    }

    // --- Sampling construction and fit/evaluate ---
    for &lambda in &[1e3, 1e5] {
        let eps = 1e-10;
        let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(
            LogisticKernel::new(lambda),
            beta,
            Some(eps),
            None,
        );
        let l = basis.size();
        let tag = format!("Λ={lambda:.0e} L={l}");

        report(
            &format!("basis     {tag} (incl. SVE)"),
            median_ms(nrun_heavy, || {
                black_box(FiniteTempBasis::<LogisticKernel, Fermionic>::new(
                    LogisticKernel::new(lambda),
                    beta,
                    Some(eps),
                    None,
                ));
            }),
        );
        report(
            &format!("tau_samp  new {tag}"),
            median_ms(nrun_light, || {
                black_box(TauSampling::<Fermionic>::new(&basis));
            }),
        );
        report(
            &format!("matsu     new {tag}"),
            median_ms(nrun_light, || {
                black_box(MatsubaraSampling::<Fermionic>::new(&basis));
            }),
        );
        report(
            &format!("dlr       new {tag}"),
            median_ms(nrun_light, || {
                black_box(DiscreteLehmannRepresentation::<Fermionic>::new(&basis).unwrap());
            }),
        );

        let tau = TauSampling::<Fermionic>::new(&basis);
        let mats = MatsubaraSampling::<Fermionic>::new(&basis);
        let dlr = DiscreteLehmannRepresentation::<Fermionic>::new(&basis).unwrap();

        for &extra in &[1usize, 100, 10_000] {
            for &dim in &[0usize, 1] {
                let shape = |n: usize| {
                    if dim == 0 {
                        vec![n, extra]
                    } else {
                        vec![extra, n]
                    }
                };
                let coeffs = Tensor::<f64, DynRank>::from_fn(&shape(l)[..], |i| {
                    ((i[0] * 7 + i[1] * 3) % 11) as f64 * 0.1 - 0.5
                });
                let gtau = tau.evaluate_nd(None, &coeffs, dim);
                let zcoeffs = Tensor::<Complex<f64>, DynRank>::from_fn(&shape(l)[..], |i| {
                    Complex::new(coeffs[[i[0], i[1]]], 0.0)
                });
                let giw = mats.evaluate_nd(None, &zcoeffs, dim);
                let n = if extra >= 10_000 {
                    nrun_heavy
                } else {
                    nrun_light
                };
                let t = format!("{tag} extra={extra} dim={dim}");
                report(
                    &format!("tau  eval {t}"),
                    median_ms(n, || {
                        black_box(tau.evaluate_nd(None, &coeffs, dim));
                    }),
                );
                report(
                    &format!("tau  fit  {t}"),
                    median_ms(n, || {
                        black_box(tau.fit_nd(None, &gtau, dim));
                    }),
                );
                report(
                    &format!("matsu eval {t}"),
                    median_ms(n, || {
                        black_box(mats.evaluate_nd(None, &zcoeffs, dim));
                    }),
                );
                report(
                    &format!("matsu fit  {t}"),
                    median_ms(n, || {
                        black_box(mats.fit_nd(None, &giw, dim));
                    }),
                );
                report(
                    &format!("ir2dlr     {t}"),
                    median_ms(n, || {
                        black_box(dlr.from_ir_nd(None, &coeffs, dim));
                    }),
                );
            }
        }
    }
}
