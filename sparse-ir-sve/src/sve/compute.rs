//! Main SVE computation functions

use crate::checks::{require_accuracy, require_nonzero_size};
use crate::error::Error;
use crate::fpu_check::FpuGuard;
use crate::kernel::{AbstractKernel, ExpandableKernel, SVEKernel};
use crate::matrix::Mat;
use crate::numeric::CustomNumeric;

use super::result::SVEResult;
use super::strategy::{NonCentrosymmSVE, SVEStrategy};
use super::types::{TworkType, safe_epsilon};

/// Default relative cutoff for singular value truncation: `2 * T::epsilon()`
///
/// This is `2^-51` (about 4.44e-16) for `f64` and `2^-104` (about 4.93e-32)
/// for `Df64`. Both SVE paths use it when `cutoff` is `None`.
/// Convention-matched with libsparseir, whose `pre_postprocess` in
/// `backend/cxx/include/sparseir/impl/sve_impl.ipp` (commit 4bc58ea) uses
/// `T(2) * std::numeric_limits<T>::epsilon()` when the cutoff is NaN.
fn default_cutoff<T: CustomNumeric>() -> T {
    T::from_f64_unchecked(2.0) * T::epsilon()
}

/// Cutoff of [`SVEOptions::cutoff_from_epsilon`] relative to the required
/// accuracy ε
///
/// A basis of singular functions accurate to ε keeps the singular values
/// ≥ ε·s₀, plus one more to estimate its error; resolving the spectrum
/// further down to 2·ε_work dominates the cost of the SVE in Df64. The cutoff
/// is also the stopping tolerance of the ACA in the block SVDs, whose
/// stopping test can underestimate the residual, so the factor leaves a
/// margin: with 10⁻⁶·ε, the singular values such a basis uses deviated from
/// the full spectrum by up to 2·10⁻⁴ relative, measured on the kernel
/// e^{-Λy(x+1)/2} / (1 + e^{-Λy}) on [-1, 1]² with Λ = 10⁶, ε = 10⁻¹²; at
/// 10⁻¹⁰·ε they agree to ≲ 4·10⁻¹¹ relative for Λ ≤ 10⁶,
/// ε = 10⁻¹⁰…10⁻¹⁵.
pub const ACCURACY_CUTOFF_FACTOR: f64 = 1e-10;

/// Cutoff of [`SVEOptions::cutoff_from_epsilon`]:
/// `max(2 * T::epsilon(), ACCURACY_CUTOFF_FACTOR * epsilon)`
///
/// For `epsilon` ≤ 4.4e-6 computed in `f64` this is [`default_cutoff`].
fn basis_cutoff<T: CustomNumeric>(epsilon: f64) -> T {
    CustomNumeric::max(
        default_cutoff::<T>(),
        T::from_f64_unchecked(ACCURACY_CUTOFF_FACTOR * epsilon),
    )
}

/// How far down the block SVDs resolve the singular values
#[derive(Clone, Copy, Debug)]
enum Resolve {
    /// To the working precision, `2 * T::epsilon()`
    Full,
    /// To [`basis_cutoff`], which is also the truncation cutoff
    Basis,
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

/// The truncation of the spectrum ([`SVEOptions`])
#[derive(Debug, Clone, Copy, PartialEq)]
enum Cutoff {
    /// `2 * machine epsilon` of the working precision
    Default,
    /// The given relative cutoff
    Relative(f64),
    /// [`basis_cutoff`] of the accuracy, also where the block SVDs stop
    FromEpsilon,
}

/// Options of [`compute_sve_with`]
///
/// [`SVEOptions::new`] gives the defaults: the best accuracy of the working
/// precision, which is chosen from the accuracy ([`TworkType::Auto`]), every
/// singular value above `2 * machine epsilon` of it, and the even/odd
/// expansion of a centrosymmetric kernel. Each method sets one option:
///
/// ```
/// use sparse_ir_sve::{SVEOptions, TworkType};
///
/// let options = SVEOptions::new()
///     .epsilon(1e-10)
///     .max_num_svals(50)
///     .twork(TworkType::Float64X2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SVEOptions {
    epsilon: Option<f64>,
    cutoff: Cutoff,
    max_num_svals: Option<usize>,
    twork: TworkType,
    full_domain: bool,
}

impl Default for SVEOptions {
    fn default() -> Self {
        Self {
            epsilon: None,
            cutoff: Cutoff::Default,
            max_num_svals: None,
            twork: TworkType::Auto,
            full_domain: false,
        }
    }
}

impl SVEOptions {
    /// The default options
    pub fn new() -> Self {
        Self::default()
    }

    /// Required accuracy, in (0, 1); by default the best accuracy of the
    /// working precision (about 1.6e-16 in Float64X2)
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = Some(epsilon);
        self
    }

    /// Relative cutoff, in [0, 1]: the singular values below `cutoff` times
    /// the largest one are discarded
    ///
    /// By default `2 * machine epsilon` of the working precision, about
    /// 4.44e-16 in Float64 and 4.93e-32 in Float64X2 (the libsparseir
    /// default). The SVD of each block already discards the singular values
    /// below `2 * machine epsilon` times its largest one, so a smaller
    /// cutoff has little effect. Replaces [`Self::cutoff_from_epsilon`].
    pub fn cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = Cutoff::Relative(cutoff);
        self
    }

    /// Keep only the singular values that an expansion accurate to
    /// `epsilon` uses
    ///
    /// Truncates the spectrum at `max(2 * machine epsilon, f * epsilon)`
    /// times the largest singular value, with `f` =
    /// [`ACCURACY_CUTOFF_FACTOR`]. In Float64X2 (`epsilon` below about 1e-8)
    /// the block SVDs stop there, which is about 1.5 to 2.3 times faster
    /// than the default, with about 20% fewer singular values. In Float64
    /// the result is that of the default truncated at the cutoff, which for
    /// `epsilon` ≤ 4.4e-6 is the same result.
    ///
    /// The singular values ≥ `epsilon` times the largest one, plus one more,
    /// agree with those of the default to about 1e-10 relative, not to the
    /// working precision. Replaces [`Self::cutoff`].
    pub fn cutoff_from_epsilon(mut self) -> Self {
        self.cutoff = Cutoff::FromEpsilon;
        self
    }

    /// Maximum number of singular values to keep, positive
    pub fn max_num_svals(mut self, max_num_svals: usize) -> Self {
        self.max_num_svals = Some(max_num_svals);
        self
    }

    /// Working precision; by default chosen from the accuracy
    /// ([`TworkType::Auto`])
    pub fn twork(mut self, twork: TworkType) -> Self {
        self.twork = twork;
        self
    }

    /// Whether to expand a centrosymmetric kernel over its full domain,
    /// without exploiting its symmetry; false by default
    ///
    /// The singular values agree with the even/odd expansion up to rounding,
    /// but the SVD is taken of one matrix with twice as many rows and columns
    /// as each even/odd block (about four times the work), the singular
    /// functions have no parity, and those of (nearly) degenerate singular
    /// values may mix the even and odd sectors. Other kernels are always
    /// expanded over their full domain.
    pub fn full_domain(mut self, full_domain: bool) -> Self {
        self.full_domain = full_domain;
        self
    }
}

/// The SVE of `kernel` with `options`
///
/// A centrosymmetric kernel ([`ReducedKernel`]) is block-diagonalized into
/// its even and odd reduced kernels, whose singular functions have definite
/// parity, unless [`SVEOptions::full_domain`] is set; any other kernel is
/// expanded over its full domain `[xmin, xmax] × [ymin, ymax]`. The working
/// precision is chosen from the accuracy unless [`SVEOptions::twork`] sets
/// it. [`compute_sve`] takes the most common options as arguments.
///
/// [`ReducedKernel`]: crate::kernel::ReducedKernel
///
/// # Errors
///
/// * [`Error::InvalidParameter`] if the accuracy is not in (0, 1), the
///   cutoff is not in [0, 1], or the maximum number of singular values is
///   0; checked before any work
/// * [`Error::NonFiniteInput`] if the discretized kernel has a NaN or
///   infinite entry (e.g. a kernel value that overflows)
/// * [`Error::DecompositionFailed`] if an SVD fails
///
/// # FPU State Warning
///
/// This function checks for dangerous FPU settings (Flush-to-Zero and
/// Denormals-Are-Zero) that can cause incorrect results. If detected, it
/// temporarily corrects the FPU state and prints a warning. If you see this
/// warning, add `-fp-model precise` flag when compiling with Intel Fortran.
pub fn compute_sve_with<K: ExpandableKernel>(
    kernel: K,
    options: &SVEOptions,
) -> Result<SVEResult, Error> {
    let (cutoff, resolve) = match options.cutoff {
        Cutoff::Default => (None, Resolve::Full),
        Cutoff::Relative(cutoff) => (Some(cutoff), Resolve::Full),
        Cutoff::FromEpsilon => (None, Resolve::Basis),
    };
    let (epsilon, max_num_svals, twork) = (options.epsilon, options.max_num_svals, options.twork);
    if options.full_domain {
        let aca = !kernel.implements_reduced_kernel();
        compute_sve_impl(
            FullDomain { kernel, aca },
            epsilon,
            cutoff,
            max_num_svals,
            twork,
            resolve,
        )
    } else {
        compute_sve_impl(
            KernelDomain(kernel),
            epsilon,
            cutoff,
            max_num_svals,
            twork,
            resolve,
        )
    }
}

/// Main SVE computation function
///
/// Chooses the SVE strategy from the kernel ([`ExpandableKernel`]) and the
/// working precision from epsilon. A centrosymmetric kernel
/// ([`ReducedKernel`]) is block-diagonalized into its even and odd reduced
/// kernels ([`CentrosymmSVE`]), whose singular functions have definite
/// parity; any other kernel is expanded over its full domain
/// `[xmin, xmax] × [ymin, ymax]` ([`NonCentrosymmSVE`]). This is
/// [`compute_sve_with`] with these options; the others are those of
/// [`SVEOptions::new`].
///
/// [`ReducedKernel`]: crate::kernel::ReducedKernel
/// [`CentrosymmSVE`]: crate::unstable::sve::CentrosymmSVE
///
/// # Arguments
///
/// * `kernel` - The kernel to expand
/// * `epsilon` - Required accuracy, in (0, 1). `None` selects the best
///   accuracy of the working precision (about 1.6e-16 in Float64X2).
/// * `cutoff` - Relative tolerance for singular value truncation: singular
///   values smaller than `cutoff` times the largest singular value are
///   discarded. `None` selects `2 * machine epsilon` of the working precision,
///   about 4.44e-16 for Float64 and 4.93e-32 for Float64X2 (the libsparseir
///   default). The SVD of each block already discards singular values below
///   `2 * machine epsilon` times that block's largest singular value, so a
///   smaller `cutoff` has little effect. Must be in [0, 1].
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
///   infinite entry (e.g. a kernel value that overflows)
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
    K: ExpandableKernel,
{
    compute_sve_impl(
        KernelDomain(kernel),
        epsilon,
        cutoff,
        max_num_svals,
        twork,
        Resolve::Full,
    )
}

/// [`compute_sve_with`] with [`SVEOptions::full_domain`], for any kernel
///
/// It takes a kernel of any type, so it knows a centrosymmetric kernel, whose
/// Df64 block it decomposes without the ACA, only by
/// [`AbstractKernel::is_centrosymmetric`]; `compute_sve_with` knows it by
/// [`ReducedKernel`](crate::kernel::ReducedKernel).
///
/// # Errors
///
/// The errors of [`compute_sve`]
#[deprecated(note = "use compute_sve_with and SVEOptions::full_domain")]
#[doc(hidden)]
pub fn compute_sve_general<K>(
    kernel: K,
    epsilon: Option<f64>,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
    twork: TworkType,
) -> Result<SVEResult, Error>
where
    K: AbstractKernel + SVEKernel + Clone + 'static,
{
    // A kernel of any type tells its symmetry only by is_centrosymmetric.
    #[allow(deprecated)]
    let aca = !kernel.is_centrosymmetric();
    compute_sve_impl(
        FullDomain { kernel, aca },
        epsilon,
        cutoff,
        max_num_svals,
        twork,
        Resolve::Full,
    )
}

/// Check the parameters of [`compute_sve_with`]
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

/// The SVE strategy of a kernel, in the working precision chosen later
trait StrategyOf {
    fn strategy<T>(self, epsilon: f64) -> Result<Box<dyn SVEStrategy<T>>, Error>
    where
        T: CustomNumeric + Send + Sync + Clone + 'static;

    /// Whether the blocks are decomposed by ACA in Df64 (see
    /// [`compute_sve_with_precision`])
    fn aca(&self) -> bool;
}

/// The strategy of the kernel ([`ExpandableKernel::sve_strategy`]): the
/// even/odd blocks of a centrosymmetric kernel, otherwise the full domain
struct KernelDomain<K>(K);

impl<K: ExpandableKernel> StrategyOf for KernelDomain<K> {
    fn strategy<T>(self, epsilon: f64) -> Result<Box<dyn SVEStrategy<T>>, Error>
    where
        T: CustomNumeric + Send + Sync + Clone + 'static,
    {
        self.0.sve_strategy(epsilon)
    }

    fn aca(&self) -> bool {
        true
    }
}

/// The full domain ([`NonCentrosymmSVE`]) for any kernel
struct FullDomain<K> {
    kernel: K,
    /// Whether to take the Df64 block SVD by ACA: not for a centrosymmetric
    /// kernel, whose full-domain block holds the even and odd spectra
    /// interleaved in nearly degenerate pairs, where the ACA missed up to 5
    /// singular values near the cutoff (logistic, Λ = 1e5 and 1e6,
    /// ε = 1e-15)
    aca: bool,
}

impl<K> StrategyOf for FullDomain<K>
where
    K: AbstractKernel + SVEKernel + Clone + 'static,
{
    fn strategy<T>(self, epsilon: f64) -> Result<Box<dyn SVEStrategy<T>>, Error>
    where
        T: CustomNumeric + Send + Sync + Clone + 'static,
    {
        Ok(Box::new(NonCentrosymmSVE::new(self.kernel, epsilon)?))
    }

    fn aca(&self) -> bool {
        self.aca
    }
}

/// The SVE of [`compute_sve`] and [`compute_sve_with`] by the strategy
/// `sve`
fn compute_sve_impl<S: StrategyOf>(
    sve: S,
    epsilon: Option<f64>,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
    twork: TworkType,
    resolve: Resolve,
) -> Result<SVEResult, Error> {
    check_sve_parameters(epsilon, cutoff, max_num_svals)?;

    // Protect computation from dangerous FPU settings (FZ/DAZ)
    // This temporarily disables FZ/DAZ and restores them after computation
    let _fpu_guard = FpuGuard::new_protect_computation();

    // Determine safe epsilon and working precision
    let (safe_epsilon, twork_actual) = safe_epsilon(epsilon, twork);

    // Dispatch based on working precision
    let aca = sve.aca();
    let result = match twork_actual {
        TworkType::Float64 => compute_sve_with_precision::<f64>(
            sve.strategy(safe_epsilon)?,
            safe_epsilon,
            cutoff,
            max_num_svals,
            resolve,
            aca,
        ),
        TworkType::Float64X2 => compute_sve_with_precision::<crate::Df64>(
            sve.strategy(safe_epsilon)?,
            safe_epsilon,
            cutoff.map(crate::Df64::from),
            max_num_svals,
            resolve,
            aca,
        ),
        _ => panic!("Invalid TworkType: {:?}", twork_actual),
    };

    // Release temporary memory back to OS after SVE computation
    release_unused_memory();

    result
}

/// [`compute_sve_impl`] in the working precision `T`
fn compute_sve_with_precision<T>(
    sve: Box<dyn SVEStrategy<T>>,
    epsilon: f64,
    cutoff: Option<T>,
    max_num_svals: Option<usize>,
    resolve: Resolve,
    aca: bool,
) -> Result<SVEResult, Error>
where
    T: CustomNumeric + Send + Sync + Clone + 'static,
{
    // 1. SVE strategy: `sve`

    // The block SVDs resolve the spectrum to the working precision, which an
    // explicit cutoff then only truncates in step 4 (the singular values of
    // an SVD close to its tolerance are not accurate, see `tsvd_aca`), or,
    // for `SVEOptions::cutoff_from_epsilon` in Df64, only down to its cutoff.
    let (rtol_t, svd_rtol) = match resolve {
        Resolve::Full => (cutoff.unwrap_or_else(default_cutoff::<T>), 0.0),
        Resolve::Basis => {
            let c = basis_cutoff::<T>(epsilon);
            (c, c.to_f64())
        }
    };

    // 2.-3. The SVD of each block, the blocks concurrently (they are
    // independent). The work is counted as forming the blocks in full.
    let svds = if !aca || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // In f64, by the column-pivoted QR of the formed blocks, as before
        // the ACA: the f64 SVE is that of main bitwise. Taken by ACA, its
        // singular values agree to the working precision, but the number of
        // those at the 2·ε_work noise floor differed in about one case out
        // of eight, by up to 8; and the f64 SVE is cheap either way. Also
        // in Df64 when `aca` is false (see `FullDomain::aca`).
        let matrices = sve.matrices();
        let work = matrices
            .iter()
            .map(|m| m.shape().0 * m.shape().1 * crate::cost::KERNEL_ENTRY)
            .sum();
        crate::parallel::map_on_pool(matrices.len(), work, |b| {
            crate::tsvd::compute_svd_dtensor(&matrices[b])
        })
    } else {
        // In Df64, by adaptive cross approximation of the blocks, evaluated
        // lazily: the SVD only touches their pivot rows and columns (ACA
        // reads fewer entries but does more per entry read).
        let blocks = sve.blocks();
        let work = blocks
            .iter()
            .map(|block| block.nrows() * block.ncols() * crate::cost::KERNEL_ENTRY)
            .sum();
        crate::parallel::map_on_pool(blocks.len(), work, |b| {
            let block = &blocks[b];
            crate::tsvd::compute_svd_aca(
                block.nrows(),
                block.ncols(),
                &|i, j| block.entry(i, j),
                svd_rtol,
            )
        })
    };

    let mut u_list = Vec::new();
    let mut s_list = Vec::new();
    let mut v_list = Vec::new();
    for svd in svds {
        let (u, s, v) = svd?;
        u_list.push(u);
        s_list.push(s);
        v_list.push(v);
    }

    // 4. Truncate based on cutoff (default: 2 * T::epsilon(), see default_cutoff)
    let (u_trunc, s_trunc, v_trunc) = truncate(u_list, s_list, v_list, rtol_t, max_num_svals);

    // 5. Post-process to create SVEResult
    sve.postprocess(u_trunc, s_trunc, v_trunc)
}

/// [`compute_sve`] by the dense CPQR-based SVD of the fully formed blocks
/// ([`SVEStrategy::matrices`] and [`compute_svd_dtensor`]), as before the ACA
/// (and as `compute_sve` still does in f64): the reference the ACA path is
/// tested against
///
/// It forms every block and decomposes it in full, which in Float64X2 takes
/// far longer than [`compute_sve`] (about 50 s, on 8 threads, for the kernel
/// e^{-Λy(x+1)/2} / (1 + e^{-Λy}) with Λ = 10⁶), so it is meant for checking
/// the result of [`compute_sve`] on a new kernel. The cutoff is the default one of
/// [`compute_sve`], `2 * machine epsilon`.
///
/// With `full_domain`, the strategy of [`SVEOptions::full_domain`]
/// ([`NonCentrosymmSVE`]) instead of that of the kernel.
///
/// # Errors
///
/// The errors of [`compute_sve`]
///
/// [`compute_svd_dtensor`]: crate::tsvd::compute_svd_dtensor
pub fn compute_sve_dense<K>(
    kernel: K,
    epsilon: Option<f64>,
    twork: TworkType,
    full_domain: bool,
) -> Result<SVEResult, Error>
where
    K: ExpandableKernel,
{
    fn run<T, K>(kernel: K, epsilon: f64, full_domain: bool) -> Result<SVEResult, Error>
    where
        T: CustomNumeric + Send + Sync + Clone + 'static,
        K: ExpandableKernel,
    {
        let sve: Box<dyn SVEStrategy<T>> = if full_domain {
            Box::new(NonCentrosymmSVE::new(kernel, epsilon)?)
        } else {
            kernel.sve_strategy::<T>(epsilon)?
        };
        let (mut u_list, mut s_list, mut v_list) = (Vec::new(), Vec::new(), Vec::new());
        for matrix in sve.matrices() {
            let (u, s, v) = crate::tsvd::compute_svd_dtensor(&matrix)?;
            u_list.push(u);
            s_list.push(s);
            v_list.push(v);
        }
        let (u, s, v) = truncate(u_list, s_list, v_list, default_cutoff::<T>(), None);
        sve.postprocess(u, s, v)
    }
    check_sve_parameters(epsilon, None, None)?;
    let _fpu_guard = FpuGuard::new_protect_computation();
    let (safe_epsilon, twork) = safe_epsilon(epsilon, twork);
    match twork {
        TworkType::Float64 => run::<f64, K>(kernel, safe_epsilon, full_domain),
        TworkType::Float64X2 => run::<crate::Df64, K>(kernel, safe_epsilon, full_domain),
        TworkType::Auto => unreachable!("safe_epsilon resolves TworkType::Auto"),
    }
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

    /// The default cutoff is exactly `2 * machine epsilon` of each working
    /// precision (issue #249); `Df64` epsilon is `f64::EPSILON^2 / 2 = 2^-105`.
    #[test]
    fn test_default_cutoff_is_two_machine_epsilon() {
        assert_eq!(default_cutoff::<f64>(), 2.0 * f64::EPSILON);
        assert_eq!(default_cutoff::<f64>(), 2f64.powi(-51));

        let df64 = default_cutoff::<crate::Df64>();
        assert_eq!(df64, crate::Df64::from(2f64.powi(-104)));
        assert_eq!((df64.hi(), df64.lo()), (2f64.powi(-104), 0.0));
    }

    /// The cutoff of `SVEOptions::cutoff_from_epsilon` is `ACCURACY_CUTOFF_FACTOR *
    /// epsilon`, floored at the default `2 * machine epsilon`, which it is for
    /// f64 with epsilon <= 4.4e-6
    #[test]
    fn test_basis_cutoff() {
        for epsilon in [1e-8, 1e-6, 4.4e-6] {
            assert_eq!(basis_cutoff::<f64>(epsilon), default_cutoff::<f64>());
        }
        assert_eq!(basis_cutoff::<f64>(1e-4 / ACCURACY_CUTOFF_FACTOR), 1e-4);
        assert_eq!(
            basis_cutoff::<crate::Df64>(1e-30),
            default_cutoff::<crate::Df64>()
        );
        assert_eq!(
            basis_cutoff::<crate::Df64>(1e-10),
            crate::Df64::from(ACCURACY_CUTOFF_FACTOR * 1e-10)
        );
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
