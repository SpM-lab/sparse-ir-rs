//! Main SVE computation functions

use crate::error::{Error, require_accuracy, require_nonzero_size};
use crate::fpu_check::FpuGuard;
use crate::kernel::{AbstractKernel, KernelProperties, SVEHints};
use crate::matrix::Mat;
use crate::numeric::CustomNumeric;

use super::result::SVEResult;
use super::strategy::{CentrosymmSVE, NonCentrosymmSVE, SVEStrategy};
use super::types::{TworkType, safe_epsilon};

/// Default cutoff relative to the required accuracy ε
///
/// Singular values far below ε are never used by a basis built for ε (it
/// keeps those ≥ ε·s₀, plus one more for its accuracy estimate), but
/// resolving them dominated the SVE cost: down to 2·ε_work, the even/odd
/// SVDs had about twice the rank needed.
///
/// The cutoff is also the stopping tolerance of the ACA in the block SVDs,
/// whose stopping test can underestimate the residual: with 10⁻⁶·ε, the
/// singular values a basis uses deviated from the full spectrum by up to
/// 2·10⁻⁴ relative (logistic, Λ = 10⁶, ε = 10⁻¹²). At 10⁻¹⁰·ε they agree to
/// ≲ 4·10⁻¹¹ relative for Λ ≤ 10⁶, ε = 10⁻¹⁰…10⁻¹⁵. For ε ≤ 4.4·10⁻⁶
/// (computed in f64), 10⁻¹⁰·ε lies below 2·ε_work, so the full spectrum is
/// resolved.
pub const DEFAULT_CUTOFF_FACTOR: f64 = 1e-10;

/// Default relative cutoff for singular value truncation:
/// `max(2 * T::epsilon(), DEFAULT_CUTOFF_FACTOR * epsilon)`
///
/// `2 * T::epsilon()` is `2^-51` (about 4.44e-16) for `f64` and `2^-104`
/// (about 4.93e-32) for `Df64`. Both SVE paths use the default when `cutoff`
/// is `None`. For `epsilon` ≤ 4.4e-6 computed in `f64` the default is
/// `2 * T::epsilon()`, convention-matched with libsparseir, whose
/// `pre_postprocess` in `backend/cxx/include/sparseir/impl/sve_impl.ipp`
/// (commit 4bc58ea) uses `T(2) * std::numeric_limits<T>::epsilon()` when the
/// cutoff is NaN; for larger `epsilon`, and in `Df64`, it is
/// `DEFAULT_CUTOFF_FACTOR * epsilon`, larger than libsparseir's.
fn default_cutoff<T: CustomNumeric>(epsilon: f64) -> T {
    CustomNumeric::max(
        T::from_f64_unchecked(2.0) * T::epsilon(),
        T::from_f64_unchecked(DEFAULT_CUTOFF_FACTOR * epsilon),
    )
}

/// Release unused memory back to the OS.
///
/// SVE computation allocates large temporary buffers for SVD.
/// After computation, these are freed but the allocator may retain them.
/// This function asks the allocator to return unused memory to the OS.
///
/// Supported platforms:
/// - macOS: uses `malloc_zone_pressure_relief`
/// - Linux (glibc): uses `malloc_trim`
/// - Other platforms: no-op (memory is still freed, just not returned to OS immediately)
#[inline]
fn release_unused_memory() {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
    }

    // Only use malloc_trim on Linux with glibc (not musl)
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        unsafe { malloc_trim(0) };
    }
}

/// Main SVE computation function
///
/// Chooses the SVE strategy from the kernel's symmetry and the working
/// precision from epsilon. A centrosymmetric kernel
/// ([`AbstractKernel::is_centrosymmetric`]) is block-diagonalized into its
/// even and odd reduced kernels ([`CentrosymmSVE`]), whose singular
/// functions have definite parity; any other kernel is expanded over its full
/// domain `[xmin, xmax] × [ymin, ymax]` ([`NonCentrosymmSVE`]).
///
/// # Arguments
///
/// * `kernel` - The kernel to expand
/// * `epsilon` - Required accuracy, in (0, 1). `None` selects the best
///   accuracy of the working precision (about 1.6e-16 in Float64X2).
/// * `cutoff` - Relative tolerance for singular value truncation: singular
///   values smaller than `cutoff` times the largest singular value are
///   discarded. `None` selects `max(2 * machine epsilon, f * epsilon)` with
///   `f` = [`DEFAULT_CUTOFF_FACTOR`] and the machine epsilon of the working
///   precision: `2 * machine epsilon` (4.44e-16, the libsparseir default) for
///   Float64 with `epsilon` ≤ 4.4e-6, otherwise `1e-10 * epsilon`; pass
///   `2 * machine epsilon` for the full numerically resolvable spectrum. The SVD of each block resolves
///   down to the smaller of `cutoff` and its default (never below
///   `2 * machine epsilon`), so a larger `cutoff` only truncates the result.
///   Must be in [0, 1].
/// * `max_num_svals` - Maximum number of singular values to keep
/// * `twork` - Working precision type (Auto for automatic selection)
///
/// # Returns
///
/// SVEResult containing singular functions and values
///
/// # Errors
///
/// * [`Error::InvalidParameter`] if `epsilon` is not in (0, 1), `cutoff` is
///   not in [0, 1], or `max_num_svals` is `Some(0)`; checked before any work
/// * [`Error::NonFiniteInput`] if the discretized kernel has a NaN or
///   infinite entry (e.g. a cutoff Λ so small that 1/Λ overflows)
/// * [`Error::DecompositionFailed`] if an SVD fails
///
/// # FPU State Warning
///
/// This function checks for dangerous FPU settings (Flush-to-Zero and Denormals-Are-Zero)
/// that can cause incorrect results. If detected, it temporarily corrects the FPU state
/// and prints a warning. If you see this warning, add `-fp-model precise` flag when
/// compiling with Intel Fortran.
pub fn compute_sve<K>(
    kernel: K,
    epsilon: Option<f64>,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
    twork: TworkType,
) -> Result<SVEResult, Error>
where
    K: AbstractKernel + KernelProperties + Clone + 'static,
{
    let even_odd = kernel.is_centrosymmetric();
    compute_sve_impl(kernel, epsilon, cutoff, max_num_svals, twork, even_odd)
}

/// SVE of any kernel over its full domain, without exploiting symmetry
///
/// Discretizes the kernel on its full domain `[xmin, xmax] × [ymin, ymax]`
/// with [`NonCentrosymmSVE`] even when it is centrosymmetric. For a
/// non-centrosymmetric kernel this is what [`compute_sve`] does.
///
/// A centrosymmetric kernel is expanded correctly as well: its half-domain
/// [`SVEHints`] segments are mirrored onto the full domain, so the singular
/// values agree with [`compute_sve`] up to rounding. The even/odd block
/// structure is not exploited, however: the SVD is taken of one matrix with
/// twice as many rows and columns as each block used by [`compute_sve`]
/// (asymptotically about four times the work), the singular functions carry no
/// parity tag, and the singular functions of (nearly) degenerate singular
/// values may mix the even and odd sectors. Prefer [`compute_sve`] unless the
/// full-domain expansion itself is wanted.
///
/// # Arguments
///
/// * `kernel` - The kernel to expand (can be centrosymmetric or non-centrosymmetric)
/// * `epsilon` - Required accuracy, in (0, 1). `None` selects the best
///   accuracy of the working precision (about 1.6e-16 in Float64X2).
/// * `cutoff` - Relative tolerance for singular value truncation, as in
///   [`compute_sve`]. Must be in [0, 1].
/// * `max_num_svals` - Maximum number of singular values to keep
/// * `twork` - Working precision type (Auto for automatic selection)
///
/// # Returns
///
/// SVEResult containing singular functions and values
///
/// # Errors
///
/// * [`Error::InvalidParameter`] if `epsilon` is not in (0, 1), `cutoff` is
///   not in [0, 1], or `max_num_svals` is `Some(0)`; checked before any work
/// * [`Error::NonFiniteInput`] if the discretized kernel has a NaN or
///   infinite entry (e.g. a cutoff Λ so small that 1/Λ overflows)
/// * [`Error::DecompositionFailed`] if an SVD fails
///
/// # FPU State Warning
///
/// This function checks for dangerous FPU settings (Flush-to-Zero and Denormals-Are-Zero)
/// that can cause incorrect results. If detected, it temporarily corrects the FPU state
/// and prints a warning. If you see this warning, add `-fp-model precise` flag when
/// compiling with Intel Fortran.
pub fn compute_sve_general<K>(
    kernel: K,
    epsilon: Option<f64>,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
    twork: TworkType,
) -> Result<SVEResult, Error>
where
    K: AbstractKernel + KernelProperties + Clone + 'static,
{
    compute_sve_impl(kernel, epsilon, cutoff, max_num_svals, twork, false)
}

/// Check the parameters shared by [`compute_sve`] and [`compute_sve_general`]
fn check_sve_parameters(
    epsilon: Option<f64>,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
) -> Result<(), Error> {
    require_accuracy("epsilon", epsilon)?;
    if let Some(c) = cutoff {
        if !(0.0..=1.0).contains(&c) {
            return Err(Error::InvalidParameter {
                name: "cutoff",
                value: format!("{c:?}"),
                reason: "must be in [0, 1]".to_string(),
            });
        }
    }
    require_nonzero_size("max_num_svals", max_num_svals)
}

/// [`compute_sve`] by the even/odd blocks of a centrosymmetric kernel
/// ([`CentrosymmSVE`]) if `even_odd`, otherwise over the full domain
/// ([`NonCentrosymmSVE`])
fn compute_sve_impl<K>(
    kernel: K,
    epsilon: Option<f64>,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
    twork: TworkType,
    even_odd: bool,
) -> Result<SVEResult, Error>
where
    K: AbstractKernel + KernelProperties + Clone + 'static,
{
    check_sve_parameters(epsilon, cutoff, max_num_svals)?;

    // Protect computation from dangerous FPU settings (FZ/DAZ)
    // This temporarily disables FZ/DAZ and restores them after computation
    let _fpu_guard = FpuGuard::new_protect_computation();

    // Determine safe epsilon and working precision
    let (safe_epsilon, twork_actual) = safe_epsilon(epsilon, twork);

    // Dispatch based on working precision
    let result = match twork_actual {
        TworkType::Float64 => compute_sve_with_precision::<f64, K>(
            kernel,
            safe_epsilon,
            cutoff,
            max_num_svals,
            even_odd,
        ),
        TworkType::Float64X2 => compute_sve_with_precision::<crate::Df64, K>(
            kernel,
            safe_epsilon,
            cutoff.map(crate::Df64::from),
            max_num_svals,
            even_odd,
        ),
        _ => panic!("Invalid TworkType: {:?}", twork_actual),
    };

    // Release temporary memory back to OS after SVE computation
    release_unused_memory();

    result
}

/// [`compute_sve_impl`] in the working precision `T`
fn compute_sve_with_precision<T, K>(
    kernel: K,
    epsilon: f64,
    cutoff: Option<T>,
    max_num_svals: Option<usize>,
    even_odd: bool,
) -> Result<SVEResult, Error>
where
    T: CustomNumeric + Send + Sync + Clone + 'static,
    K: AbstractKernel + KernelProperties + Clone + 'static,
    K::SVEHintsType<T>: SVEHints<T> + Clone,
{
    // 1. SVE strategy
    let sve: Box<dyn SVEStrategy<T>> = if even_odd {
        Box::new(CentrosymmSVE::new(kernel, epsilon)?)
    } else {
        Box::new(NonCentrosymmSVE::new(kernel, epsilon)?)
    };

    // 2. Discretized kernel blocks, evaluated lazily: the SVD below only
    // touches their pivot rows and columns
    let blocks = sve.blocks();

    // Singular values below the cutoff are discarded in step 4, so the block
    // SVDs need not resolve them. The cutoff never goes below 2·ε of the
    // working type (f64 or Df64).
    let default_cutoff = default_cutoff::<T>(epsilon);
    let rtol_t = cutoff.unwrap_or(default_cutoff);
    // The block SVDs resolve down to the cutoff, but never stop above the
    // default one: their singular values close to the tolerance are not
    // accurate (see `tsvd_aca`). A larger cutoff only truncates the result.
    let svd_rtol = CustomNumeric::min(rtol_t, default_cutoff).to_f64();

    // 3. Compute the SVD of each block by adaptive cross approximation
    let mut u_list = Vec::new();
    let mut s_list = Vec::new();
    let mut v_list = Vec::new();

    for block in &blocks {
        let (u, s, v) = crate::tsvd::compute_svd_aca(
            block.nrows(),
            block.ncols(),
            &|i, j| block.entry(i, j),
            svd_rtol,
        )?;
        u_list.push(u);
        s_list.push(s);
        v_list.push(v);
    }

    // 4. Truncate based on cutoff (see default_cutoff)
    let (u_trunc, s_trunc, v_trunc) = truncate(u_list, s_list, v_list, rtol_t, max_num_svals);

    // 5. Post-process to create SVEResult
    sve.postprocess(u_trunc, s_trunc, v_trunc)
}

/// Truncate SVD results based on cutoff and maximum size
///
/// # Arguments
///
/// * `u_list` - List of U matrices
/// * `s_list` - List of singular value vectors
/// * `v_list` - List of V matrices
/// * `rtol` - Relative tolerance for truncation
/// * `max_num_svals` - Maximum number of singular values to keep
///
/// # Returns
///
/// Tuple of (truncated_u_list, truncated_s_list, truncated_v_list)
pub(crate) fn truncate<T: CustomNumeric>(
    u_list: Vec<Mat<T>>,
    s_list: Vec<Vec<T>>,
    v_list: Vec<Mat<T>>,
    rtol: T,
    max_num_svals: Option<usize>,
) -> (Vec<Mat<T>>, Vec<Vec<T>>, Vec<Mat<T>>) {
    let zero = T::zero();

    // Validate
    if let Some(max) = max_num_svals {
        if max == 0 {
            panic!("max_num_svals must be positive");
        }
    }
    if rtol < zero || rtol > T::from_f64_unchecked(1.0) {
        panic!("rtol must be in [0, 1]");
    }

    // Find global maximum singular value
    let mut all_svals = Vec::new();
    for s in &s_list {
        all_svals.extend(s.iter().copied());
    }

    let max_sval = all_svals
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(zero);

    // Determine cutoff
    let cutoff = if let Some(max_count) = max_num_svals {
        if max_count < all_svals.len() {
            let mut sorted = all_svals.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let nth = sorted[max_count - 1];
            if rtol * max_sval > nth {
                rtol * max_sval
            } else {
                nth
            }
        } else {
            rtol * max_sval
        }
    } else {
        rtol * max_sval
    };

    // Truncate each result
    let mut u_trunc = Vec::new();
    let mut s_trunc = Vec::new();
    let mut v_trunc = Vec::new();

    for i in 0..s_list.len() {
        let s = &s_list[i];
        let u = &u_list[i];
        let v = &v_list[i];

        // Count singular values above cutoff
        let mut n_keep = 0;
        for &val in s.iter() {
            if val >= cutoff {
                n_keep += 1;
            }
        }

        // Slice U: keep first n_keep columns. Preserve one output block per
        // input block even when all singular values in that block are removed.
        let u_shape = *u.shape();
        let u_sliced = Mat::<T>::from_fn([u_shape.0, n_keep], |idx| u[[idx[0], idx[1]]]);
        u_trunc.push(u_sliced);

        s_trunc.push(s[..n_keep].to_vec());

        // Slice V: keep first n_keep columns
        let v_shape = *v.shape();
        let v_sliced = Mat::<T>::from_fn([v_shape.0, n_keep], |idx| v[[idx[0], idx[1]]]);
        v_trunc.push(v_sliced);
    }

    (u_trunc, s_trunc, v_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default cutoff is `2 * machine epsilon` of each working precision
    /// (issue #249) when `DEFAULT_CUTOFF_FACTOR * epsilon` is smaller, as for
    /// f64 with epsilon <= 4.4e-6; `Df64` epsilon is
    /// `f64::EPSILON^2 / 2 = 2^-105`. Otherwise it is
    /// `DEFAULT_CUTOFF_FACTOR * epsilon`.
    #[test]
    fn test_default_cutoff() {
        for epsilon in [1e-8, 1e-6, 4.4e-6] {
            assert_eq!(default_cutoff::<f64>(epsilon), 2.0 * f64::EPSILON);
            assert_eq!(default_cutoff::<f64>(epsilon), 2f64.powi(-51));
        }
        assert_eq!(default_cutoff::<f64>(1e-4 / DEFAULT_CUTOFF_FACTOR), 1e-4);

        let df64 = default_cutoff::<crate::Df64>(1e-30);
        assert_eq!(df64, crate::Df64::from(2f64.powi(-104)));
        assert_eq!((df64.hi(), df64.lo()), (2f64.powi(-104), 0.0));
        let df64 = default_cutoff::<crate::Df64>(1e-10);
        assert_eq!(df64, crate::Df64::from(DEFAULT_CUTOFF_FACTOR * 1e-10));
    }

    #[test]
    fn test_truncate_by_rtol() {
        let u = vec![Mat::<f64>::from_elem([3, 3], 1.0)];
        let s = vec![vec![10.0, 5.0, 0.1]];
        let v = vec![Mat::<f64>::from_elem([3, 3], 1.0)];

        // rtol = 0.1, max_sval = 10.0, cutoff = 1.0
        // Keep values >= 1.0: [10.0, 5.0]
        let (_, s_trunc, _) = truncate(u, s, v, 0.1, None);

        assert_eq!(s_trunc[0].len(), 2);
        assert_eq!(s_trunc[0][0], 10.0);
        assert_eq!(s_trunc[0][1], 5.0);
    }

    #[test]
    fn test_truncate_by_max_size() {
        let u = vec![Mat::<f64>::from_elem([3, 3], 1.0)];
        let s = vec![vec![10.0, 5.0, 2.0]];
        let v = vec![Mat::<f64>::from_elem([3, 3], 1.0)];

        // max_num_svals = 2
        let (_, s_trunc, _) = truncate(u, s, v, 0.0, Some(2));

        assert_eq!(s_trunc[0].len(), 2);
    }

    #[test]
    #[should_panic(expected = "max_num_svals must be positive")]
    fn test_truncate_rejects_zero_max_size() {
        let u = vec![Mat::<f64>::from_elem([1, 1], 1.0)];
        let s = vec![vec![1.0]];
        let v = vec![Mat::<f64>::from_elem([1, 1], 1.0)];

        truncate(u, s, v, 0.0, Some(0));
    }

    #[test]
    fn test_truncate_preserves_empty_blocks() {
        let u = vec![
            Mat::<f64>::from_elem([2, 1], 1.0),
            Mat::<f64>::from_elem([2, 1], 2.0),
        ];
        let s = vec![vec![10.0], vec![1.0]];
        let v = vec![
            Mat::<f64>::from_elem([2, 1], 1.0),
            Mat::<f64>::from_elem([2, 1], 2.0),
        ];

        let (u_trunc, s_trunc, v_trunc) = truncate(u, s, v, 0.5, None);

        assert_eq!(s_trunc.len(), 2);
        assert_eq!(s_trunc[0], vec![10.0]);
        assert!(s_trunc[1].is_empty());
        assert_eq!(*u_trunc[1].shape(), (2, 0));
        assert_eq!(*v_trunc[1].shape(), (2, 0));
    }

    #[test]
    #[should_panic(expected = "rtol must be in [0, 1]")]
    fn test_truncate_invalid_rtol() {
        let u = vec![Mat::<f64>::from_elem([1, 1], 1.0)];
        let s = vec![vec![1.0]];
        let v = vec![Mat::<f64>::from_elem([1, 1], 1.0)];

        truncate(u, s, v, 1.5, None);
    }
}
