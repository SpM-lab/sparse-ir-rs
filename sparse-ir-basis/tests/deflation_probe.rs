//! Experiments for SpM-lab/sparse-ir-rs#330, not a test suite.
//!
//! Every test is `#[ignore]`d; run them explicitly with
//!
//! ```text
//! cargo test -p sparse-ir-basis --release --test deflation_probe -- --ignored --nocapture
//! ```
//!
//! What they measure (numbers in the issue):
//! * `deflation_experiment` - f64 head + Df64 residual tail: 5-12x faster, but the
//!                            f64 leakage (eps_f64 * s_0, amplified by 1/s_l) caps
//!                            the left subspace at ~1e-4 .. 1e-6
//! * `all_df64_experiment`  - all-Df64 random-sketched partial SVD: 3-5x faster,
//!                            singular values and V good, U only 1e-5 .. 1e-7
//! * `ideal_q_experiment`   - with an exact Q the machinery is exact (3e-25), so
//!                            the error is in the sketch, not the post-processing
//! * `svdq_experiment`      - building Q by SVD instead of MGS does not help

use nalgebra::DMatrix;
use sparse_ir_basis::{
    CentrosymmSVE, CustomNumeric, Df64, LogisticKernel, SVEStrategy, svd_decompose, tsvd_df64,
    tsvd_f64,
};
use std::time::Instant;

/// || A_k A_k^T - B_k B_k^T ||_F / sqrt(k)
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

fn dot(a: &[Df64], b: &[Df64]) -> Df64 {
    let mut acc = Df64::from(0.0);
    for (x, y) in a.iter().zip(b) {
        acc += *x * *y;
    }
    acc
}

/// y -= alpha * x
fn axpy(y: &mut [Df64], alpha: Df64, x: &[Df64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi -= alpha * *xi;
    }
}

/// Modified Gram-Schmidt: orthonormal basis of the columns of `y` (m x l)
/// with one reorthogonalization pass.
fn orthonormalize(y: &mut DMatrix<Df64>, m: usize, l: usize) {
    for c in 0..l {
        for _pass in 0..2 {
            for d in 0..c {
                let qd: Vec<Df64> = (0..m).map(|i| y[(i, d)]).collect();
                let cur: Vec<Df64> = (0..m).map(|i| y[(i, c)]).collect();
                let h = dot(&qd, &cur);
                for i in 0..m {
                    let yd = y[(i, d)];
                    y[(i, c)] -= h * yd;
                }
            }
        }
        let nrm = dot(
            &(0..m).map(|i| y[(i, c)]).collect::<Vec<_>>(),
            &(0..m).map(|i| y[(i, c)]).collect::<Vec<_>>(),
        )
        .to_f64()
        .max(0.0)
        .sqrt();
        for i in 0..m {
            y[(i, c)] = y[(i, c)] * Df64::from(1.0 / nrm);
        }
    }
}

/// Matrix times vector, column-major column loop.
fn mv(a: &DMatrix<Df64>, x: &[Df64], y: &mut [Df64]) {
    let (m, n) = (a.nrows(), a.ncols());
    let s = a.as_slice();
    for v in y.iter_mut() {
        *v = Df64::from(0.0);
    }
    for j in 0..n {
        let xj = x[j];
        let col = &s[j * m..(j + 1) * m];
        for i in 0..m {
            y[i] += col[i] * xj;
        }
    }
}

/// max relative error of the first `k` entries of `a` against `b`
fn rel_err(a: &[Df64], b: &[Df64], k: usize) -> f64 {
    (0..k.min(a.len()).min(b.len()))
        .map(|l| ((a[l] - b[l]) / b[l]).to_f64().abs())
        .fold(0.0f64, f64::max)
}

fn deflation_probe(lambda: f64, eps: f64, p: usize) -> Result<(), Box<dyn std::error::Error>> {
    let kernel = LogisticKernel::new(lambda)?;
    let sve = CentrosymmSVE::<Df64, LogisticKernel>::new(kernel, eps)?;
    let mats = sve.matrices();
    let m0 = &mats[0];
    let (m, n) = *m0.shape();
    let mut a = DMatrix::<Df64>::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            a[(i, j)] = m0[[i, j]];
        }
    }
    let a64 = DMatrix::<f64>::from_fn(m, n, |i, j| a[(i, j)].to_f64());

    println!("\n=== Lambda={lambda}, eps={eps}, matrix {m}x{n} (even block) ===");

    // reference
    let t = Instant::now();
    let truth = tsvd_df64(&a, Df64::from(2.0) * Df64::epsilon())?;
    let t_ref = t.elapsed().as_secs_f64();
    let need = (0..truth.s.len())
        .take_while(|&l| (truth.s[l] / truth.s[0]).to_f64() > eps)
        .count();
    println!(
        "  reference (dense Df64): rank {:4}  {t_ref:6.1} s   needed (s/s0 > eps): {need}",
        truth.rank
    );

    // f64 head: spectrum only, plus the factors
    let t = Instant::now();
    let head = tsvd_f64(&a64, 2.0 * f64::EPSILON)?;
    let t_head = t.elapsed().as_secs_f64();
    let s64: Vec<f64> = head.s.iter().cloned().collect();
    let s0 = s64[0];
    let k = s64
        .iter()
        .take_while(|&&s| s / s0 > 10.0 * f64::EPSILON / eps)
        .count();
    let n_tail_above = s64.iter().take_while(|&&s| s / s0 >= eps / 10.0).count();
    let n_tail = n_tail_above.saturating_sub(k);
    let l_tail = n_tail + p;
    println!(
        "  f64 head: rank {:4}  {t_head:6.2} s   -> k={k}, n_tail={n_tail}, l_tail={l_tail} (p={p})",
        head.rank
    );
    if l_tail > n.min(m) || k + l_tail == 0 {
        println!("  (sizes out of range, skipping)");
        return Ok(());
    }

    // head factors in Df64 (first k columns)
    let uh = DMatrix::<Df64>::from_fn(m, k, |i, j| Df64::from(head.u[(i, j)]));
    let sh: Vec<Df64> = (0..k).map(|j| Df64::from(head.s[j])).collect();
    let vh = DMatrix::<Df64>::from_fn(n, k, |i, j| Df64::from(head.v[(i, j)]));

    // Df64 tail on the residual R = A - Uh Sh Vh^T
    let t = Instant::now();
    let mut seed = 2024u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Df64::from(((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5)
    };
    let mut y = DMatrix::<Df64>::zeros(m, l_tail);
    let mut x = vec![Df64::from(0.0); n];
    let mut ax = vec![Df64::from(0.0); m];
    for c in 0..l_tail {
        for xj in x.iter_mut() {
            *xj = rnd();
        }
        mv(&a, &x, &mut ax); // A x
        // subtract Uh (Sh (Vh^T x))
        let mut t1 = vec![Df64::from(0.0); k];
        for j in 0..k {
            t1[j] = dot(&(0..n).map(|i| vh[(i, j)]).collect::<Vec<_>>(), &x) * sh[j];
        }
        for i in 0..m {
            let mut s = Df64::from(0.0);
            for j in 0..k {
                s += uh[(i, j)] * t1[j];
            }
            y[(i, c)] = ax[i] - s;
        }
    }
    orthonormalize(&mut y, m, l_tail);
    let q = &y;

    // B = Q^T R = Q^T A - (Q^T Uh) Sh Vh^T
    let mut bmat = DMatrix::<Df64>::zeros(l_tail, n);
    {
        let asl = a.as_slice();
        for j in 0..n {
            let col = &asl[j * m..(j + 1) * m];
            for r in 0..l_tail {
                let mut acc = Df64::from(0.0);
                for i in 0..m {
                    acc += q[(i, r)] * col[i];
                }
                bmat[(r, j)] = acc;
            }
        }
    }
    let mut qu = DMatrix::<Df64>::zeros(l_tail, k);
    for r in 0..l_tail {
        for j in 0..k {
            let mut acc = Df64::from(0.0);
            for i in 0..m {
                acc += q[(i, r)] * uh[(i, j)];
            }
            qu[(r, j)] = acc * sh[j];
        }
    }
    for j in 0..n {
        for r in 0..l_tail {
            let mut acc = Df64::from(0.0);
            for jj in 0..k {
                acc += qu[(r, jj)] * vh[(j, jj)];
            }
            bmat[(r, j)] -= acc;
        }
    }
    let tail = svd_decompose(&bmat, 1e-30);
    let t_tail = t.elapsed().as_secs_f64();

    // combined: head ++ tail
    let nt = tail.rank;
    let mut u = DMatrix::<Df64>::zeros(m, k + nt);
    let mut v = DMatrix::<Df64>::zeros(n, k + nt);
    let mut s = vec![Df64::from(0.0); k + nt];
    for j in 0..k {
        for i in 0..m {
            u[(i, j)] = uh[(i, j)];
        }
        for i in 0..n {
            v[(i, j)] = vh[(i, j)];
        }
        s[j] = sh[j];
    }
    for j in 0..nt {
        for i in 0..m {
            let mut acc = Df64::from(0.0);
            for r in 0..l_tail {
                acc += q[(i, r)] * tail.u[(r, j)];
            }
            u[(i, k + j)] = acc;
        }
        for i in 0..n {
            v[(i, k + j)] = tail.v[(i, j)];
        }
        s[k + j] = tail.s[j];
    }

    let kk = need.min(s.len()).min(truth.s.len());
    println!(
        "  Df64 tail: l_tail={l_tail}  {t_tail:6.2} s   total {:.2} s",
        t_head + t_tail
    );
    println!(
        "  vs reference over first {kk}:  max rel ds = {:.2e}   subspace dist U = {:.2e}  V = {:.2e}",
        rel_err(
            &s,
            &(0..truth.s.len()).map(|l| truth.s[l]).collect::<Vec<_>>(),
            kk
        ),
        projector_diff(&truth.u, &u, kk),
        projector_diff(&truth.v, &v, kk)
    );
    Ok(())
}

#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn deflation_experiment() {
    for lambda in [1e4, 1e5, 1e6] {
        for eps in [1e-8, 1e-10, 1e-12] {
            if let Err(e) = deflation_probe(lambda, eps, 16) {
                println!("Lambda={lambda} eps={eps}: ERROR {e}");
            }
        }
    }
}

/// All-Df64 matvec-based partial SVD (no f64 head, so no f64 leakage).
/// Sizing still comes from a cheap f64 spectrum run.
fn all_df64_probe(lambda: f64, eps: f64, p: usize) -> Result<(), Box<dyn std::error::Error>> {
    let kernel = LogisticKernel::new(lambda)?;
    let sve = CentrosymmSVE::<Df64, LogisticKernel>::new(kernel, eps)?;
    let mats = sve.matrices();
    let m0 = &mats[0];
    let (m, n) = *m0.shape();
    let mut a = DMatrix::<Df64>::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            a[(i, j)] = m0[[i, j]];
        }
    }
    let a64 = DMatrix::<f64>::from_fn(m, n, |i, j| a[(i, j)].to_f64());

    println!("\n=== all-Df64: Lambda={lambda}, eps={eps}, matrix {m}x{n} (even block) ===");
    let t = Instant::now();
    let truth = tsvd_df64(&a, Df64::from(2.0) * Df64::epsilon())?;
    let t_ref = t.elapsed().as_secs_f64();
    let need = (0..truth.s.len())
        .take_while(|&l| (truth.s[l] / truth.s[0]).to_f64() > eps)
        .count();

    // sizing only: f64 spectrum (cheap)
    let t = Instant::now();
    let spec = tsvd_f64(&a64, 2.0 * f64::EPSILON)?;
    let t_spec = t.elapsed().as_secs_f64();
    let s0 = spec.s[0];
    let n_above = spec.s.iter().take_while(|&&s| s / s0 >= eps / 10.0).count();
    let l = (n_above + p).min(m.min(n));
    let (rank_ref, tr) = (truth.rank, t_ref);
    println!(
        "  reference: rank {rank_ref}  {tr:.2} s  needed {need}  |  sizing f64: {t_spec:.2} s -> l={l} (n_above={n_above}, p={p})"
    );

    let t = Instant::now();
    let mut seed = 4242u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Df64::from(((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5)
    };
    let mut y = DMatrix::<Df64>::zeros(m, l);
    let mut x = vec![Df64::from(0.0); n];
    let mut ax = vec![Df64::from(0.0); m];
    for c in 0..l {
        for xj in x.iter_mut() {
            *xj = rnd();
        }
        mv(&a, &x, &mut ax);
        for i in 0..m {
            y[(i, c)] = ax[i];
        }
    }
    orthonormalize(&mut y, m, l);
    let q = y;
    let mut bmat = DMatrix::<Df64>::zeros(l, n);
    {
        let asl = a.as_slice();
        for j in 0..n {
            let col = &asl[j * m..(j + 1) * m];
            for r in 0..l {
                let mut acc = Df64::from(0.0);
                for i in 0..m {
                    acc += q[(i, r)] * col[i];
                }
                bmat[(r, j)] = acc;
            }
        }
    }
    let tail = svd_decompose(&bmat, 1e-30);
    let u = &q * &tail.u;
    let t_tot = t.elapsed().as_secs_f64();
    let kk = need.min(tail.s.len()).min(truth.s.len());
    println!(
        "  partial: {t_tot:6.2} s (total {:.2} s)   vs reference over {kk}:   max rel ds = {:.2e}   U = {:.2e}  V = {:.2e}",
        t_spec + t_tot,
        rel_err(
            tail.s.as_slice(),
            &(0..truth.s.len()).map(|l| truth.s[l]).collect::<Vec<_>>(),
            kk
        ),
        projector_diff(&truth.u, &u, kk),
        projector_diff(&truth.v, &tail.v, kk)
    );
    let _ = t_ref;
    Ok(())
}

#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn all_df64_experiment() {
    for lambda in [1e4, 1e5, 1e6] {
        for eps in [1e-8, 1e-10, 1e-12] {
            if let Err(e) = all_df64_probe(lambda, eps, 16) {
                println!("Lambda={lambda} eps={eps}: ERROR {e}");
            }
        }
    }
}

/// Diagnostic: give the method an EXACT leading left subspace (the reference's
/// own U), so the intrinsic accuracy of "Q -> B = Q^T A -> SVD" is separated
/// from the quality of Q produced by the sketch.
fn ideal_q_probe(lambda: f64, eps: f64, l: usize) -> Result<(), Box<dyn std::error::Error>> {
    let kernel = LogisticKernel::new(lambda)?;
    let sve = CentrosymmSVE::<Df64, LogisticKernel>::new(kernel, eps)?;
    let mats = sve.matrices();
    let m0 = &mats[0];
    let (m, n) = *m0.shape();
    let mut a = DMatrix::<Df64>::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            a[(i, j)] = m0[[i, j]];
        }
    }
    let truth = tsvd_df64(&a, Df64::from(2.0) * Df64::epsilon())?;
    let need = (0..truth.s.len())
        .take_while(|&lv| (truth.s[lv] / truth.s[0]).to_f64() > eps)
        .count();
    let t = Instant::now();
    let q = DMatrix::<Df64>::from_fn(m, l, |i, j| truth.u[(i, j)]);
    let mut bmat = DMatrix::<Df64>::zeros(l, n);
    {
        let asl = a.as_slice();
        for j in 0..n {
            let col = &asl[j * m..(j + 1) * m];
            for r in 0..l {
                let mut acc = Df64::from(0.0);
                for i in 0..m {
                    acc += q[(i, r)] * col[i];
                }
                bmat[(r, j)] = acc;
            }
        }
    }
    let part = svd_decompose(&bmat, 1e-30);
    let u = &q * &part.u;
    let dt = t.elapsed().as_secs_f64();
    let kk = need.min(part.s.len()).min(truth.s.len());
    println!(
        "  ideal Q (l={l}): {dt:.2} s  rel ds = {:.2e}   U = {:.2e}  V = {:.2e}",
        rel_err(
            part.s.as_slice(),
            &(0..truth.s.len()).map(|lv| truth.s[lv]).collect::<Vec<_>>(),
            kk
        ),
        projector_diff(&truth.u, &u, kk),
        projector_diff(&truth.v, &part.v, kk)
    );
    Ok(())
}

#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn ideal_q_experiment() {
    for lambda in [1e5, 1e6] {
        for eps in [1e-10] {
            println!("\n=== ideal-Q diagnostic: Lambda={lambda}, eps={eps} ===");
            for l in [80usize, 120] {
                ideal_q_probe(lambda, eps, l).unwrap();
            }
        }
    }
}

/// Same as all_df64_probe but builds Q from the SVD of the sketch instead of
/// MGS, to see whether the Q construction is the limiting factor.
fn all_df64_svdq_probe(lambda: f64, eps: f64, p: usize) -> Result<(), Box<dyn std::error::Error>> {
    let kernel = LogisticKernel::new(lambda)?;
    let sve = CentrosymmSVE::<Df64, LogisticKernel>::new(kernel, eps)?;
    let mats = sve.matrices();
    let m0 = &mats[0];
    let (m, n) = *m0.shape();
    let mut a = DMatrix::<Df64>::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            a[(i, j)] = m0[[i, j]];
        }
    }
    let a64 = DMatrix::<f64>::from_fn(m, n, |i, j| a[(i, j)].to_f64());
    let truth = tsvd_df64(&a, Df64::from(2.0) * Df64::epsilon())?;
    let need = (0..truth.s.len())
        .take_while(|&lv| (truth.s[lv] / truth.s[0]).to_f64() > eps)
        .count();
    let spec = tsvd_f64(&a64, 2.0 * f64::EPSILON)?;
    let s0 = spec.s[0];
    let l0 = (spec.s.iter().take_while(|&&s| s / s0 >= eps / 10.0).count() + p).min(m.min(n));

    let t = Instant::now();
    let mut seed = 31337u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Df64::from(((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5)
    };
    let mut y = DMatrix::<Df64>::zeros(m, l0);
    let mut x = vec![Df64::from(0.0); n];
    let mut ax = vec![Df64::from(0.0); m];
    for c in 0..l0 {
        for xj in x.iter_mut() {
            *xj = rnd();
        }
        mv(&a, &x, &mut ax);
        for i in 0..m {
            y[(i, c)] = ax[i];
        }
    }
    let q = svd_decompose(&y, 1e-30).u;
    let l = q.ncols();
    let mut bmat = DMatrix::<Df64>::zeros(l, n);
    {
        let asl = a.as_slice();
        for j in 0..n {
            let col = &asl[j * m..(j + 1) * m];
            for r in 0..l {
                let mut acc = Df64::from(0.0);
                for i in 0..m {
                    acc += q[(i, r)] * col[i];
                }
                bmat[(r, j)] = acc;
            }
        }
    }
    let part = svd_decompose(&bmat, 1e-30);
    let u = &q * &part.u;
    let dt = t.elapsed().as_secs_f64();
    let kk = need.min(part.s.len()).min(truth.s.len());
    println!(
        "  SVD-Q Lambda={lambda} eps={eps}: l={l0}->{l}  {dt:.2} s  rel ds = {:.2e}   U = {:.2e}  V = {:.2e}",
        rel_err(
            part.s.as_slice(),
            &(0..truth.s.len()).map(|lv| truth.s[lv]).collect::<Vec<_>>(),
            kk
        ),
        projector_diff(&truth.u, &u, kk),
        projector_diff(&truth.v, &part.v, kk)
    );
    Ok(())
}

#[test]
#[ignore = "experiment: run with --ignored --nocapture"]
fn svdq_experiment() {
    for (lambda, eps) in [(1e5, 1e-10), (1e6, 1e-10), (1e6, 1e-12)] {
        all_df64_svdq_probe(lambda, eps, 16).unwrap();
    }
}
