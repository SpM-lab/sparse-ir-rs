//! Experiments for SpM-lab/sparse-ir-rs#330, not a test suite.
//!
//! Every test is `#[ignore]`d; run them explicitly with
//!
//! ```text
//! cargo test -p sparse-ir-basis --release --test tsvd_rtol_probe -- --ignored --nocapture
//! ```
//!
//! What they measure (numbers in the issue):
//! * `tsvd_rtol_probe`        - loosening the TSVD tolerance degrades the leading
//!                              left singular subspace (~1e8 x rtol): rtol=1e-12
//!                              gives 5.7e-5 at Lambda=1e6
//! * `synthetic_reproduction` - the same hazard on a small synthetic matrix
//!                              (~0.2 s per call), i.e. it can be a cheap
//!                              regression test
//! * `svd_shape_scaling`      - cost of the SVD phase alone
//! * `matvec_throughput`      - Df64 1.7e8 flop/s vs f64 1.3e9 (same loop, 7.7x)
//! * `modulus_cost`           - Df64 `modulus()` is 11.8 ns; the QR pivot search
//!                              calls it per trailing element (4.9 s at Lambda=1e6)
//! * `pivot_search_cost`      - `tsvd_df64` vs `tsvd_f64` on the same matrix: 71x

use nalgebra::DMatrix;
use sparse_ir_basis::{
    CentrosymmSVE, CustomNumeric, Df64, LogisticKernel, SVEStrategy, svd_decompose, tsvd_df64,
    tsvd_f64,
};
use std::time::Instant;

/// || A_k A_k^T - B_k B_k^T ||_F / sqrt(k): subspace distance of the leading k
/// left singular vectors (sign and ordering independent).
fn projector_diff(a: &DMatrix<Df64>, b: &DMatrix<Df64>, k: usize) -> f64 {
    let mut acc: f64 = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.nrows() {
            let mut d = Df64::from(0.0);
            for l in 0..k {
                d += a[(i, l)] * a[(j, l)] - b[(i, l)] * b[(j, l)];
            }
            acc += d.to_f64() * d.to_f64();
        }
    }
    acc.sqrt() / (k as f64).sqrt()
}

fn probe(lambda: f64, eps: f64) -> Result<(), Box<dyn std::error::Error>> {
    let kernel = LogisticKernel::new(lambda)?;
    let sve = CentrosymmSVE::<Df64, LogisticKernel>::new(kernel, eps)?;
    let mats = sve.matrices();
    let m = &mats[0]; // even block
    let (nrows, ncols) = *m.shape();
    let mut dm = DMatrix::<Df64>::zeros(nrows, ncols);
    for j in 0..ncols {
        for i in 0..nrows {
            dm[(i, j)] = m[[i, j]];
        }
    }
    println!("\n=== Lambda = {lambda} (eps = {eps}), matrix {nrows} x {ncols} (even block) ===");

    let t = Instant::now();
    let truth = svd_decompose(&dm, f64::EPSILON);
    println!(
        "  plain SVD (ground truth): rank {:4}  {:6.1} s",
        truth.rank,
        t.elapsed().as_secs_f64()
    );

    let k = (0..truth.s.len())
        .take_while(|&l| (truth.s[l] / truth.s[0]).to_f64() > 1e-10)
        .count();
    println!("  singular values above 1e-10 * s0: {k}");

    for rtol in [0.0, 1e-14, 1e-12, 1e-10, 1e-8] {
        let rt = if rtol == 0.0 {
            Df64::from(2.0) * Df64::epsilon()
        } else {
            Df64::from(rtol)
        };
        let t = Instant::now();
        let got = tsvd_df64(&dm, rt).unwrap();
        let dt = t.elapsed().as_secs_f64();
        let kk = k.min(got.s.len()).min(truth.s.len());
        let ds = (0..kk)
            .map(|l| ((truth.s[l] - got.s[l]) / truth.s[l]).to_f64().abs())
            .fold(0.0f64, f64::max);
        let pu = projector_diff(&truth.u, &got.u, kk);
        let pv = projector_diff(&truth.v, &got.v, kk);
        println!(
            "  tsvd rtol={:>8.1e}: rank {:4}  {:6.1} s   max rel ds (first {kk}) = {ds:.2e}   \
             subspace dist U = {pu:.2e}  V = {pv:.2e}",
            rtol, got.rank, dt
        );
    }
    Ok(())
}

#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn tsvd_rtol_probe() {
    for lambda in [1e4, 1e5, 1e6] {
        probe(lambda, 1e-10).unwrap();
    }
}

/// Cost of the SVD phase alone, on the shape the QR leaves behind.
#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn svd_shape_scaling() {
    use nalgebra::DMatrix;
    use std::time::Instant;
    let mut seed = 12345u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    // shapes: (rows, cols) of the truncated R, plus the full matrix for reference
    for (m, n) in [(165usize, 1920usize), (200, 1920), (165, 165)] {
        let a = DMatrix::<Df64>::from_fn(m, n, |_, _| Df64::from(rnd()));
        let t = Instant::now();
        let r = svd_decompose(&a, f64::EPSILON);
        println!(
            "svd_decompose {m:5} x {n:5}: {:8.2} s  (rank {})",
            t.elapsed().as_secs_f64(),
            r.rank
        );
    }
}

/// Does the hazard reproduce on a small synthetic matrix (for a cheap
/// regression test)?  A = U diag(sigma) V^T with a geometric spectrum.
#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn synthetic_reproduction() {
    use nalgebra::DMatrix;
    use std::time::Instant;
    let (m, n) = (200usize, 250usize);
    let k = m; // full rank, long smooth tail like the real kernel matrix
    let mut seed = 7u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    // Orthonormal U (m x k) and V (n x k) from the SVD of random matrices.
    let qu = svd_decompose(
        &DMatrix::<Df64>::from_fn(m, k, |_, _| Df64::from(rnd())),
        1e-16,
    )
    .u;
    let qv = svd_decompose(
        &DMatrix::<Df64>::from_fn(n, k, |_, _| Df64::from(rnd())),
        1e-16,
    )
    .u;
    let sigma: Vec<Df64> = (0..k)
        .map(|i| Df64::from(10f64.powf(-(i as f64) / 5.0)))
        .collect();
    let mut a = DMatrix::<Df64>::zeros(m, n);
    for l in 0..k {
        for i in 0..m {
            let uil = qu[(i, l)];
            for j in 0..n {
                a[(i, j)] += uil * sigma[l] * qv[(j, l)];
            }
        }
    }
    let truth = svd_decompose(&a, 1e-16);
    let need = (0..truth.s.len())
        .take_while(|&l| (truth.s[l] / truth.s[0]).to_f64() > 1e-10)
        .count();
    println!("\n=== synthetic {m} x {n}, {k} decaying modes, {need} above 1e-10 ===");
    for rtol in [0.0, 1e-14, 1e-12, 1e-10, 1e-8] {
        let rt = if rtol == 0.0 {
            Df64::from(2.0) * Df64::epsilon()
        } else {
            Df64::from(rtol)
        };
        let t = Instant::now();
        let got = tsvd_df64(&a, rt).unwrap();
        let kk = need.min(got.s.len()).min(truth.s.len());
        println!(
            "  tsvd rtol={rtol:>8.1e}: rank {:4}  {:5.2} s   subspace dist U = {:.2e}  V = {:.2e}",
            got.rank,
            t.elapsed().as_secs_f64(),
            projector_diff(&truth.u, &got.u, kk),
            projector_diff(&truth.v, &got.v, kk)
        );
    }
}

/// Prerequisite for an all-Df64 matvec-based partial SVD: how fast is a Df64
/// matvec / block-matvec compared with f64?
#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn matvec_throughput() {
    use std::time::Instant;
    let (m, n, kb) = (1440usize, 1920usize, 64usize);
    let mut seed = 99u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let a64: Vec<f64> = (0..m * n).map(|_| rnd()).collect();
    let add: Vec<Df64> = a64.iter().map(|&v| Df64::from(v)).collect();
    let x64: Vec<f64> = (0..n * kb).map(|_| rnd()).collect();
    let xdd: Vec<Df64> = x64.iter().map(|&v| Df64::from(v)).collect();

    // Df64, single column
    let mut ydd = vec![Df64::from(0.0); m];
    let t = Instant::now();
    for i in 0..m {
        let mut acc = Df64::from(0.0);
        for j in 0..n {
            acc += add[i + m * j] * xdd[j];
        }
        ydd[i] = acc;
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "\nDf64 matvec {m}x{n}: {dt:.3} s = {:.2e} Df64-flop/s (2*m*n = {:.2e} flop)",
        2.0 * (m * n) as f64 / dt,
        2.0 * (m * n) as f64
    );

    // Df64, 64 columns
    let mut yb = vec![Df64::from(0.0); m * kb];
    let t = Instant::now();
    for q in 0..kb {
        for i in 0..m {
            let mut acc = Df64::from(0.0);
            for j in 0..n {
                acc += add[i + m * j] * xdd[j + n * q];
            }
            yb[i + m * q] = acc;
        }
    }
    let dtb = t.elapsed().as_secs_f64();
    println!(
        "Df64 block matvec {m}x{n} x {kb}: {dtb:.3} s = {:.2e} Df64-flop/s",
        2.0 * (m * n * kb) as f64 / dtb
    );

    // f64, single column, same loop order
    let mut y64 = vec![0.0f64; m];
    let t = Instant::now();
    for i in 0..m {
        let mut acc = 0.0f64;
        for j in 0..n {
            acc += a64[i + m * j] * x64[j];
        }
        y64[i] = acc;
    }
    let dtf = t.elapsed().as_secs_f64();
    println!(
        "f64  matvec {m}x{n}: {dtf:.3} s = {:.2e} f64-flop/s",
        2.0 * (m * n) as f64 / dtf
    );
    println!("  ratio Df64/f64 (same loop) = {:.1}x", dt / dtf);
    std::hint::black_box((&ydd, &yb, &y64));
}

/// Where does the QR time go?  Compare the Df64 path with the f64 path on the
/// very same matrix: the arithmetic ratio is ~7-9x, so a much larger ratio
/// means something else (e.g. the Df64 pivot search) dominates.
#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn pivot_search_cost() {
    use nalgebra::DMatrix;
    use std::time::Instant;
    let kernel = LogisticKernel::new(1e6).unwrap();
    let sve = CentrosymmSVE::<Df64, LogisticKernel>::new(kernel, 1e-10).unwrap();
    let mats = sve.matrices();
    let m = &mats[0];
    let (nr, nc) = *m.shape();
    let mut add = DMatrix::<Df64>::zeros(nr, nc);
    for j in 0..nc {
        for i in 0..nr {
            add[(i, j)] = m[[i, j]];
        }
    }
    let mut a64 = DMatrix::<f64>::zeros(nr, nc);
    let mut a64_from_dd = DMatrix::<f64>::zeros(nr, nc);
    for j in 0..nc {
        for i in 0..nr {
            a64[(i, j)] = m[[i, j]].to_f64(); // rounded from Df64
            a64_from_dd[(i, j)] = add[(i, j)].to_f64();
        }
    }
    println!("\n=== tsvd on the same {nr} x {nc} kernel matrix, eps=1e-10 ===");
    let t = Instant::now();
    let r1 = tsvd_df64(&add, Df64::from(2.0) * Df64::epsilon()).unwrap();
    let tdd = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let r2 = tsvd_f64(&a64_from_dd, 2.0 * f64::EPSILON).unwrap();
    let tf = t.elapsed().as_secs_f64();
    println!("  tsvd_df64: rank {:4}  {tdd:7.2} s", r1.rank);
    println!("  tsvd_f64 : rank {:4}  {tf:7.2} s", r2.rank);
    println!(
        "  ratio Df64/f64 = {:.1}x   (arithmetic ratio is ~7-9x)",
        tdd / tf
    );
    let _ = a64;
}

/// Hypothesis for the 70x: the column-pivoted QR's max-norm search calls
/// `ComplexField::modulus` on every trailing element at every step, which for
/// a real `Df64` is a double-double sqrt of a perfect square.
#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn modulus_cost() {
    use nalgebra::ComplexField;
    use std::time::Instant;
    let n = 2_000_000usize;
    let xs64: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-7 - 0.5).collect();
    let xs: Vec<Df64> = xs64.iter().map(|&v| Df64::from(v)).collect();

    let t = Instant::now();
    let mut acc = Df64::from(0.0);
    for &x in &xs {
        acc += x.modulus();
    }
    let dtd = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut accn = Df64::from(0.0);
    for &x in &xs {
        accn += x.norm1();
    }
    let dtn = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut accs = Df64::from(0.0);
    for &x in &xs {
        accs += x * x;
    }
    let dts = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut accm = Df64::from(0.0);
    for &x in &xs {
        accm += Df64::from(x.to_f64().abs());
    }
    let dtm = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut acc64 = 0.0f64;
    for &x in &xs64 {
        acc64 += x.abs();
    }
    let dtf = t.elapsed().as_secs_f64();
    let per = dtd / n as f64;
    // loop + read + convert overhead, to subtract from the numbers above
    let t = Instant::now();
    let mut accf = 0.0f64;
    for &x in &xs {
        accf += x.to_f64();
    }
    let dto = t.elapsed().as_secs_f64();
    let ns0 = |d: f64| d / n as f64 * 1e9;
    println!(
        "loop+read+convert+f64-add overhead = {:.2} ns/iter",
        ns0(dto)
    );
    let ns = |d: f64| d / n as f64 * 1e9;
    println!(
        "
per-element ns: norm1()={:.2} (icamax_full)  x*x={:.2}  to_f64().abs()={:.2}  f64 abs={:.2}  [modulus()={:.2}]",
        ns(dtn), ns(dts), ns(dtm), ns(dtf), ns(dtd)
    );
    // trailing-search work of the QR: sum_i (m-i)(n-i) for m=1440, n=1920, k=165
    let (m, n_, k) = (1440f64, 1920f64, 165f64);
    let s: f64 = (0..k as usize)
        .map(|i| (m - i as f64) * (n_ - i as f64))
        .sum();
    println!("  QR pivot-search elements for {m}x{n_} x {k} steps = {s:.3e}");
    println!("  -> implied cost of modulus() alone = {:.1} s", s * per);
    std::hint::black_box((acc, accn, accs, accm, acc64));
}
