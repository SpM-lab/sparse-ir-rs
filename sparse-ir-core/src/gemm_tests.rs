use super::*;

fn val(i: usize) -> f64 {
    ((i * 37 + 11) % 23) as f64 / 7.0 - 1.5
}

trait TestScalar:
    GemmScalar + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::fmt::Debug
{
    fn make(i: usize) -> Self;
    fn conj_(self) -> Self;
    fn abs_(self) -> f64;
}

impl TestScalar for f64 {
    fn make(i: usize) -> Self {
        val(i)
    }
    fn conj_(self) -> Self {
        self
    }
    fn abs_(self) -> f64 {
        self.abs()
    }
}

impl TestScalar for Complex<f64> {
    fn make(i: usize) -> Self {
        Complex::new(val(i), val(i + 101))
    }
    fn conj_(self) -> Self {
        self.conj()
    }
    fn abs_(self) -> f64 {
        self.norm()
    }
}

/// Element (i, j) of op(X) for stored X with leading dimension ld.
fn op_at<T: TestScalar>(x: &[T], op: Transpose, ld: usize, i: usize, j: usize) -> T {
    match op {
        Transpose::N => x[i + ld * j],
        Transpose::T => x[j + ld * i],
        Transpose::C => x[j + ld * i].conj_(),
    }
}

fn check_gemm<T: TestScalar>(backend: Option<&GemmBackendHandle>) {
    let ops = [Transpose::N, Transpose::T, Transpose::C];
    let (m, n, k) = (5, 4, 3);
    let (alpha, beta) = (T::make(3), T::make(8));
    for &ta in &ops {
        for &tb in &ops {
            let (ar, ac) = if ta == Transpose::N { (m, k) } else { (k, m) };
            let (br, bc) = if tb == Transpose::N { (k, n) } else { (n, k) };
            // Padded leading dimensions exercise strided addressing.
            let (lda, ldb, ldc) = (ar + 2, br + 1, m + 3);
            let a: Vec<T> = (0..lda * ac).map(T::make).collect();
            let b: Vec<T> = (0..ldb * bc).map(|i| T::make(i + 50)).collect();
            let c0: Vec<T> = (0..ldc * n).map(|i| T::make(i + 90)).collect();
            let mut c = c0.clone();
            gemm(
                backend, ta, tb, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc,
            )
            .unwrap();
            for j in 0..n {
                for i in 0..m {
                    let mut acc = T::zero();
                    for l in 0..k {
                        acc = acc + op_at(&a, ta, lda, i, l) * op_at(&b, tb, ldb, l, j);
                    }
                    let expected = alpha * acc + beta * c0[i + ldc * j];
                    let err = (c[i + ldc * j] - expected).abs_();
                    assert!(err < 1e-12, "{ta:?}{tb:?} ({i},{j}) err={err}");
                }
                // Padding rows of C are untouched.
                for i in m..ldc {
                    assert_eq!(c[i + ldc * j], c0[i + ldc * j]);
                }
            }
        }
    }
}

#[test]
fn faer_gemm_all_ops_f64() {
    check_gemm::<f64>(Some(&GemmBackendHandle::default()));
}

#[test]
fn faer_gemm_all_ops_c64() {
    check_gemm::<Complex<f64>>(Some(&GemmBackendHandle::default()));
}

#[test]
fn beta_zero_ignores_nan_in_c() {
    let a = [1.0, 2.0];
    let b = [3.0];
    let mut c = [f64::NAN, f64::NAN];
    gemm(
        Some(&GemmBackendHandle::default()),
        Transpose::N,
        Transpose::N,
        2,
        1,
        1,
        1.0,
        &a,
        2,
        &b,
        1,
        0.0,
        &mut c,
        2,
    )
    .unwrap();
    assert_eq!(c, [3.0, 6.0]);
}

#[test]
fn short_buffer_is_rejected() {
    let a = [1.0; 3];
    let b = [1.0; 4];
    let mut c = [0.0; 4];
    let err = gemm(
        None,
        Transpose::N,
        Transpose::N,
        2,
        2,
        2,
        1.0,
        &a,
        2,
        &b,
        2,
        0.0,
        &mut c,
        2,
    )
    .unwrap_err();
    assert!(matches!(err, GemmError::InvalidArgument(_)));
}

#[test]
fn lp64_overflow_is_reported() {
    let big = i32::MAX as usize + 1;
    let err = blas_ints::<libc::c_int>("LP64", 1, big, 1, 1, 1, 1).unwrap_err();
    assert_eq!(
        err,
        GemmError::DimensionOverflow {
            name: "n",
            value: big,
            abi: "LP64"
        }
    );
    assert!(blas_ints::<i64>("ILP64", 1, big, 1, 1, 1, 1).is_ok());
}

fn check_apply<T: TestScalar>(m: usize, n: usize, pre: usize, post: usize) {
    let a: Vec<T> = (0..m * n).map(|i| T::make(i % 13)).collect();
    let x: Vec<T> = (0..pre * n * post).map(|i| T::make(i + 17)).collect();
    let mut y = vec![T::zero(); pre * m * post];
    apply_along_axis(None, &a, m, n, &x, pre, post, &mut y).unwrap();
    for q in 0..post {
        for i in 0..m {
            for p in 0..pre {
                let mut acc = T::zero();
                for j in 0..n {
                    acc = acc + a[i + m * j] * x[p + pre * (j + n * q)];
                }
                let err = (y[p + pre * (i + m * q)] - acc).abs_();
                let tol = 1e-12 * (1.0 + acc.abs_());
                assert!(err < tol, "m={m} n={n} pre={pre} post={post} err={err}");
            }
        }
    }
}

#[test]
fn apply_along_axis_all_paths() {
    // Small matvec, pre == 1 GEMM, slice loop (pre >= 32), and packed paths.
    for &(pre, post) in &[(1, 1), (1, 4), (1, 7), (5, 1), (40, 3), (3, 6), (2, 0)] {
        check_apply::<f64>(4, 3, pre, post);
        check_apply::<Complex<f64>>(4, 3, pre, post);
    }
    // Matrix-vector products above the small-matvec threshold use the backend.
    check_apply::<f64>(130, 129, 1, 1);
    check_apply::<Complex<f64>>(130, 129, 1, 2);
    // Degenerate contraction extent.
    check_apply::<f64>(4, 0, 1, 1);
}

#[test]
fn matmul_typed() {
    let a = crate::Matrix::from_vec_col_major([2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let b = crate::Matrix::from_vec_col_major([3, 1], vec![1.0, 1.0, 1.0]).unwrap();
    let c = matmul(None, &a, &b).unwrap();
    assert_eq!(c.host_data().unwrap(), &[6.0, 15.0]);
    assert!(matmul(None, &a, &a).is_err());
}

#[test]
fn clear_backend_restores_faer() {
    clear_blas_backend();
    let (name, external, ilp64) = get_backend_info();
    assert!(name.contains("Faer"));
    assert!(!external && !ilp64);
}
