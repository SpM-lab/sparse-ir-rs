//! Discrete Lehmann Representation (DLR)
//!
//! This module provides the Discrete Lehmann Representation (DLR) basis,
//! which represents Green's functions as a linear combination of poles on the
//! real-frequency axis.

use crate::fitters::FitScalar;
use crate::fitters::RealMatrixFitter;
use crate::freq::MatsubaraFreq;
use crate::gemm::GemmBackendHandle;
use crate::matrix::Mat;
use crate::traits::{Statistics, StatisticsType};
use num_complex::Complex;
use std::marker::PhantomData;
use std::sync::OnceLock;
use tenferro_tensor::TypedTensor;

/// Errors returned when constructing a [`DiscreteLehmannRepresentation`] or
/// an [`IrDlrTransform`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DlrError {
    /// The number of default poles is less than the basis size. This can
    /// happen with certain kernel types (e.g., `RegularizedBoseKernel`) due
    /// to numerical precision limitations in root finding.
    InsufficientDefaultPoles {
        /// Basis size.
        basis_size: usize,
        /// Number of poles actually found.
        n_poles: usize,
    },
    /// The kernel does not support the requested statistics (e.g.
    /// `RegularizedBoseKernel` with fermionic statistics).
    KernelStatisticsMismatch,
    /// A construction parameter (β, ωmax, accuracy, poles) is invalid.
    InvalidParameter(String),
    /// The IR basis and the DLR do not describe the same domain.
    IncompatibleIrBasis(String),
}

impl std::fmt::Display for DlrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DlrError::InsufficientDefaultPoles {
                basis_size,
                n_poles,
            } => write!(
                f,
                "only {n_poles} default poles were found for an IR basis of size {basis_size}"
            ),
            DlrError::KernelStatisticsMismatch => {
                write!(f, "the kernel does not support the requested statistics")
            }
            DlrError::InvalidParameter(msg) => write!(f, "invalid DLR parameter: {msg}"),
            DlrError::IncompatibleIrBasis(msg) => write!(f, "incompatible IR basis: {msg}"),
        }
    }
}

impl std::error::Error for DlrError {}

/// Generic single-pole Green's function at imaginary time τ
///
/// Computes G(τ) for either fermionic or bosonic statistics based on the type parameter S.
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `tau` - Imaginary time (can be outside [0, β))
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Example
/// ```ignore
/// use sparse_ir::traits::Fermionic;
/// let g_f = gtau_single_pole::<Fermionic>(0.5, 5.0, 1.0);
///
/// use sparse_ir::traits::Bosonic;
/// let g_b = gtau_single_pole::<Bosonic>(0.5, 5.0, 1.0);
/// ```
pub fn gtau_single_pole<S: StatisticsType>(tau: f64, omega: f64, beta: f64) -> f64 {
    match S::STATISTICS {
        Statistics::Fermionic => fermionic_single_pole(tau, omega, beta),
        Statistics::Bosonic => bosonic_single_pole(tau, omega, beta),
    }
}

/// Compute fermionic single-pole Green's function at imaginary time τ
///
/// Evaluates G(τ) = -exp(-ω×τ) / (1 + exp(-β×ω)) for a single pole at frequency ω.
///
/// Supports extended τ ranges with anti-periodic boundary conditions:
/// - G(τ + β) = -G(τ) (fermionic anti-periodicity)
/// - Valid for τ ∈ (-β, 2β)
///
/// # Arguments
/// * `tau` - Imaginary time (can be outside [0, β))
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Example
/// ```ignore
/// let beta = 1.0;
/// let omega = 5.0;
/// let tau = 0.5 * beta;
/// let g = fermionic_single_pole(tau, omega, beta);
/// ```
pub fn fermionic_single_pole(tau: f64, omega: f64, beta: f64) -> f64 {
    use crate::taufuncs::normalize_tau;
    use crate::traits::Fermionic;

    // Normalize τ to [0, β] and track sign from anti-periodicity
    // G(τ + β) = -G(τ) for fermions
    let (tau_normalized, sign) = normalize_tau::<Fermionic>(tau, beta);

    // Avoid overflow for large negative ω by factoring out exp(βω).
    // Both branches keep the exponent non-positive.
    let value = if omega >= 0.0 {
        -(-omega * tau_normalized).exp() / (1.0 + (-beta * omega).exp())
    } else {
        -(omega * (beta - tau_normalized)).exp() / (1.0 + (beta * omega).exp())
    };

    sign * value
}

/// Compute bosonic single-pole Green's function at imaginary time τ
///
/// Evaluates G(τ) = exp(-ω×τ) / (1 - exp(-β×ω)) for a single pole at frequency ω.
///
/// Supports extended τ ranges with periodic boundary conditions:
/// - G(τ + β) = G(τ) (bosonic periodicity)
/// - Valid for τ ∈ (-β, 2β)
///
/// # Arguments
/// * `tau` - Imaginary time (can be outside [0, β))
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Example
/// ```ignore
/// let beta = 1.0;
/// let omega = 5.0;
/// let tau = 0.5 * beta;
/// let g = bosonic_single_pole(tau, omega, beta);
/// ```
pub fn bosonic_single_pole(tau: f64, omega: f64, beta: f64) -> f64 {
    use crate::taufuncs::normalize_tau;
    use crate::traits::Bosonic;

    // Normalize τ to [0, β] using periodicity
    // G(τ + β) = G(τ) for bosons
    let tau_normalized = normalize_tau::<Bosonic>(tau, beta).0;

    if omega >= 0.0 {
        (-omega * tau_normalized).exp() / (1.0 - (-beta * omega).exp())
    } else {
        -(omega * (beta - tau_normalized)).exp() / (1.0 - (beta * omega).exp())
    }
}

/// Generic single-pole Green's function at Matsubara frequency
///
/// Computes G(iωn) = 1/(iωn - ω) for a single pole at frequency ω.
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `matsubara_freq` - Matsubara frequency
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Complex-valued Green's function G(iωn)
pub fn giwn_single_pole<S: StatisticsType>(
    matsubara_freq: &MatsubaraFreq<S>,
    omega: f64,
    beta: f64,
) -> Complex<f64> {
    // G(iωn) = 1/(iωn - ω)
    let wn = matsubara_freq.value(beta);
    let denominator = Complex::new(0.0, 1.0) * wn - Complex::new(omega, 0.0);
    Complex::new(1.0, 0.0) / denominator
}

// ============================================================================
// Discrete Lehmann Representation
// ============================================================================

/// Discrete Lehmann Representation (DLR)
///
/// Represents Green's functions as a linear combination of poles on the
/// real-frequency axis:
///
/// ```text
/// G(iν) = Σ_i a[i] * reg[i] / (iν - ω[i])
/// ```
///
/// where:
/// - `ω[i]` are pole positions on the real axis
/// - `a[i]` are expansion coefficients
/// - `reg[i]` are kernel-dependent pole weights on the physical ω grid
///
/// Two constructions are available:
/// - **independent (default)**: [`DiscreteLehmannRepresentation::new`] /
///   [`DlrBuilder`] select the poles by an interpolative decomposition of the
///   discretized logistic kernel (Kaye, Chen, Parcollet, PRB 105, 235115),
///   without building an IR basis;
/// - **IR-derived**: [`DiscreteLehmannRepresentation::from_ir`] uses the
///   default real-frequency sampling points of an IR basis as poles.
///
/// Conversions to and from an IR basis are provided by [`IrDlrTransform`];
/// an IR-derived DLR carries one for its source basis.
///
/// Imaginary-time and Matsubara nodes for square interpolation are selected
/// by a row interpolative decomposition and are exposed through
/// [`Basis::default_tau_sampling_points`](crate::basis_trait::Basis) and
/// [`Basis::default_matsubara_sampling_points`](crate::basis_trait::Basis),
/// so [`TauSampling::new`](crate::TauSampling) and
/// [`MatsubaraSampling::new`](crate::MatsubaraSampling) work on a DLR.
///
/// The public `regularizers` field stores the raw kernel regularizer
/// `w(β, ω_i)`. Internally, DLR evaluations use `pole_weights`, which include
/// the ω-domain normalization carried by `FiniteTempBasis`.
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
pub struct DiscreteLehmannRepresentation<S>
where
    S: StatisticsType,
{
    /// Pole positions on the real-frequency axis ω ∈ [-ωmax, ωmax]
    pub poles: Vec<f64>,

    /// Inverse temperature β
    pub beta: f64,

    /// Maximum frequency ωmax
    pub wmax: f64,

    /// LogisticKernel reference basis used for Basis trait compatibility
    kernel: crate::kernel::LogisticKernel,

    /// Power with which the source kernel scales the spectral variable.
    kernel_ypower: i32,

    /// Accuracy of the representation
    pub accuracy: f64,

    /// Regularizers for each pole: regularizer[i] = w(β, ω_i)
    pub regularizers: Vec<f64>,

    /// Pole weights used in tau and Matsubara evaluations.
    ///
    /// `FiniteTempBasis` rescales the ω-domain singular values by `wmax^-ypower`.
    /// Combined with the dimensionless kernel regularizer `y^ypower =
    /// (ω / wmax)^ypower`, the physical pole basis carries an additional
    /// factor `wmax^(-2 * ypower)`.
    pole_weights: Vec<f64>,

    /// IR <-> DLR transform for the source basis of an IR-derived DLR.
    ir: Option<IrDlrTransform>,

    /// Lazily selected interpolation nodes.
    tau_nodes: OnceLock<Vec<f64>>,
    matsubara_nodes: OnceLock<Vec<i64>>,
    matsubara_nodes_positive: OnceLock<Vec<i64>>,

    /// Marker for statistics type
    _phantom: PhantomData<S>,
}

/// Builder for the independent (interpolative-decomposition) DLR.
///
/// ```
/// use sparse_ir::{DlrBuilder, Fermionic};
/// let dlr = DlrBuilder::<Fermionic>::new(10.0, 10.0)
///     .accuracy(1e-10)
///     .build()
///     .unwrap();
/// assert!(dlr.poles.len() > 0);
/// ```
#[derive(Debug, Clone)]
pub struct DlrBuilder<S: StatisticsType> {
    beta: f64,
    wmax: f64,
    accuracy: f64,
    max_size: Option<usize>,
    _phantom: PhantomData<S>,
}

impl<S: StatisticsType + 'static> DlrBuilder<S> {
    /// Default target accuracy.
    pub const DEFAULT_ACCURACY: f64 = 1e-15;

    /// Start a DLR for inverse temperature `beta` and frequency cutoff `wmax`.
    pub fn new(beta: f64, wmax: f64) -> Self {
        Self {
            beta,
            wmax,
            accuracy: Self::DEFAULT_ACCURACY,
            max_size: None,
            _phantom: PhantomData,
        }
    }

    /// Target relative accuracy of the kernel interpolation.
    pub fn accuracy(mut self, accuracy: f64) -> Self {
        self.accuracy = accuracy;
        self
    }

    /// Upper bound on the number of poles.
    pub fn max_size(mut self, max_size: usize) -> Self {
        self.max_size = Some(max_size);
        self
    }

    /// Select the poles and build the representation.
    ///
    /// # Errors
    /// Returns [`DlrError::InvalidParameter`] when β or ωmax is not positive
    /// and finite, or the accuracy is not in `(0, 1)`.
    pub fn build(self) -> Result<DiscreteLehmannRepresentation<S>, DlrError> {
        let valid = |x: f64| x.is_finite() && x > 0.0;
        if !valid(self.beta) || !valid(self.wmax) {
            return Err(DlrError::InvalidParameter(format!(
                "beta = {} and wmax = {} must be positive and finite",
                self.beta, self.wmax
            )));
        }
        if !(self.accuracy > 0.0 && self.accuracy < 1.0) {
            return Err(DlrError::InvalidParameter(format!(
                "accuracy = {} must lie in (0, 1)",
                self.accuracy
            )));
        }
        if self.max_size == Some(0) {
            return Err(DlrError::InvalidParameter(
                "max_size must be positive".into(),
            ));
        }
        let lambda = self.beta * self.wmax;
        let poles: Vec<f64> = crate::dlr_id::select_poles(lambda, self.accuracy, self.max_size)
            .into_iter()
            .map(|w| w / self.beta)
            .collect();
        let kernel = crate::kernel::LogisticKernel::new(lambda);
        Ok(DiscreteLehmannRepresentation::from_parts(
            self.beta,
            self.wmax,
            self.accuracy,
            poles,
            &kernel,
            None,
        ))
    }
}

impl<S> DiscreteLehmannRepresentation<S>
where
    S: StatisticsType,
{
    pub fn kernel_ypower(&self) -> i32 {
        self.kernel_ypower
    }

    pub fn pole_weights(&self) -> &[f64] {
        &self.pole_weights
    }

    /// Build a DLR independently of any IR basis (default construction).
    ///
    /// Equivalent to `DlrBuilder::new(beta, wmax).accuracy(accuracy).build()`.
    ///
    /// # Errors
    /// See [`DlrBuilder::build`].
    pub fn new(beta: f64, wmax: f64, accuracy: f64) -> Result<Self, DlrError>
    where
        S: 'static,
    {
        DlrBuilder::new(beta, wmax).accuracy(accuracy).build()
    }

    fn from_parts<K>(
        beta: f64,
        wmax: f64,
        accuracy: f64,
        poles: Vec<f64>,
        kernel: &K,
        ir: Option<IrDlrTransform>,
    ) -> Self
    where
        S: 'static,
        K: crate::kernel::KernelProperties,
    {
        let kernel_ypower = kernel.ypower();
        let regularizers: Vec<f64> = poles
            .iter()
            .map(|&pole| kernel.regularizer::<S>(beta, pole))
            .collect();
        let pole_weights = pole_weights_for::<S, K>(kernel, beta, wmax, &poles);
        Self {
            poles,
            beta,
            wmax,
            kernel: crate::kernel::LogisticKernel::new(beta * wmax),
            kernel_ypower,
            accuracy,
            regularizers,
            pole_weights,
            ir,
            tau_nodes: OnceLock::new(),
            matsubara_nodes: OnceLock::new(),
            matsubara_nodes_positive: OnceLock::new(),
            _phantom: PhantomData,
        }
    }

    /// Create a DLR from an IR basis with custom poles.
    ///
    /// The pole weights follow the kernel of `basis`, and the returned DLR
    /// carries an [`IrDlrTransform`] for `basis`.
    ///
    /// # Errors
    /// Returns [`DlrError::KernelStatisticsMismatch`] if the kernel does not
    /// support the requested statistics (e.g. `RegularizedBoseKernel` with
    /// fermionic statistics).
    pub fn from_ir_with_poles<K>(
        basis: &impl crate::basis_trait::Basis<S, Kernel = K>,
        poles: Vec<f64>,
    ) -> Result<Self, DlrError>
    where
        S: 'static,
        K: crate::kernel::KernelProperties + Clone,
    {
        check_kernel_statistics::<S, K>(basis.kernel())?;
        let (beta, wmax) = (basis.beta(), basis.wmax());
        let weights = pole_weights_for::<S, K>(basis.kernel(), beta, wmax, &poles);
        let ir = IrDlrTransform::with_weights(basis, &poles, &weights)?;
        Ok(Self::from_parts(
            beta,
            wmax,
            basis.accuracy(),
            poles,
            basis.kernel(),
            Some(ir),
        ))
    }

    /// Create a DLR from an IR basis, using its default real-frequency
    /// sampling points as poles.
    ///
    /// # Errors
    /// Returns [`DlrError::InsufficientDefaultPoles`] if the number of default
    /// poles is less than the basis size. This can happen with certain kernel
    /// types (e.g., `RegularizedBoseKernel`) due to numerical precision
    /// limitations in root finding.
    pub fn from_ir<K>(
        basis: &impl crate::basis_trait::Basis<S, Kernel = K>,
    ) -> Result<Self, DlrError>
    where
        S: 'static,
        K: crate::kernel::KernelProperties + Clone,
    {
        check_kernel_statistics::<S, K>(basis.kernel())?;
        let poles = basis.default_omega_sampling_points();
        let basis_size = basis.size();
        if basis_size > poles.len() {
            return Err(DlrError::InsufficientDefaultPoles {
                basis_size,
                n_poles: poles.len(),
            });
        }
        Self::from_ir_with_poles(basis, poles)
    }

    fn zero_pole_tau_limit(&self) -> f64 {
        match self.kernel_ypower {
            0 => -0.5,
            1 => -1.0 / (self.beta * self.wmax * self.wmax),
            _ => panic!(
                "DLR tau evaluation does not support kernel ypower = {}",
                self.kernel_ypower
            ),
        }
    }

    fn zero_pole_matsubara_limit(&self) -> f64 {
        match self.kernel_ypower {
            0 => -0.5 * self.beta,
            1 => -1.0 / (self.wmax * self.wmax),
            _ => panic!(
                "DLR Matsubara evaluation does not support kernel ypower = {}",
                self.kernel_ypower
            ),
        }
    }

    // ========================================================================
    // Public API (generic, user-friendly)
    // ========================================================================

    /// IR <-> DLR transform of the source basis (IR-derived DLR only).
    pub fn ir_transform(&self) -> Option<&IrDlrTransform> {
        self.ir.as_ref()
    }

    fn require_ir(&self) -> crate::Result<&IrDlrTransform> {
        self.ir.as_ref().ok_or(crate::Error::Unsupported(
            "this DLR was not built from an IR basis; use IrDlrTransform::new",
        ))
    }

    /// Convert IR coefficients of the source basis to DLR along axis `dim`
    /// (least squares).
    ///
    /// `T` is `f64` or `Complex<f64>`.
    ///
    /// # Errors
    /// Returns [`crate::Error::Unsupported`] for a DLR that was not built from
    /// an IR basis, or an error if `gl.shape()[dim]` differs from the IR basis
    /// size.
    pub fn from_ir_nd<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        gl: &TypedTensor<T>,
        dim: usize,
    ) -> crate::Result<TypedTensor<T>> {
        self.require_ir()?.ir_to_dlr_nd(backend, gl, dim)
    }

    /// Convert DLR coefficients to IR coefficients of the source basis along
    /// axis `dim`.
    ///
    /// `T` is `f64` or `Complex<f64>`.
    ///
    /// # Errors
    /// Returns [`crate::Error::Unsupported`] for a DLR that was not built from
    /// an IR basis, or an error if `g_dlr.shape()[dim]` differs from the
    /// number of poles.
    pub fn to_ir_nd<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        g_dlr: &TypedTensor<T>,
        dim: usize,
    ) -> crate::Result<TypedTensor<T>> {
        self.require_ir()?.dlr_to_ir_nd(backend, g_dlr, dim)
    }

    /// IR basis size of the source basis (IR-derived DLR only).
    pub fn ir_basis_size(&self) -> Option<usize> {
        self.ir.as_ref().map(IrDlrTransform::ir_size)
    }
}

impl<S> DiscreteLehmannRepresentation<S>
where
    S: StatisticsType + 'static,
{
    /// Imaginary-time interpolation nodes (sorted), one per pole.
    pub fn tau_nodes(&self) -> &[f64] {
        use crate::basis_trait::Basis;
        self.tau_nodes.get_or_init(|| {
            let taus: Vec<f64> = crate::dlr_id::tau_candidates(self.beta * self.wmax)
                .iter()
                .map(|t| self.beta * t.value())
                .collect();
            let mat = self.evaluate_tau(&taus);
            let rows = crate::dlr_id::select_rows(
                mat.host_data().expect("host matrix"),
                taus.len(),
                self.poles.len(),
                self.poles.len(),
            );
            rows.into_iter().map(|i| taus[i]).collect()
        })
    }

    /// Matsubara interpolation nodes as indices `n` (`ν = nπ/β`, sorted).
    ///
    /// With `positive_only`, nodes are chosen among `n >= 0` so that the
    /// stacked real and imaginary parts determine the coefficients of a
    /// Green's function with `G(-iν) = conj(G(iν))`.
    pub fn matsubara_nodes(&self, positive_only: bool) -> &[i64] {
        let cell = if positive_only {
            &self.matsubara_nodes_positive
        } else {
            &self.matsubara_nodes
        };
        cell.get_or_init(|| self.select_matsubara_nodes(positive_only))
    }

    fn select_matsubara_nodes(&self, positive_only: bool) -> Vec<i64> {
        use crate::basis_trait::Basis;
        let zeta = match S::STATISTICS {
            Statistics::Fermionic => 1,
            Statistics::Bosonic => 0,
        };
        let ns = crate::dlr_id::matsubara_candidates(self.beta * self.wmax, zeta, positive_only);
        let freqs: Vec<MatsubaraFreq<S>> = ns
            .iter()
            .map(|&n| MatsubaraFreq::new(n).expect("candidate parity matches statistics"))
            .collect();
        let mat = self.evaluate_matsubara(&freqs);
        let data = mat.host_data().expect("host matrix");
        let (nf, r) = (freqs.len(), self.poles.len());
        if !positive_only {
            let rows = crate::dlr_id::select_rows(data, nf, r, r);
            return rows.into_iter().map(|i| ns[i]).collect();
        }
        // Stack [Re; Im] so each frequency contributes two real rows.
        let mut stacked = vec![0.0; 2 * nf * r];
        for j in 0..r {
            for i in 0..nf {
                let z = data[i + nf * j];
                stacked[i + 2 * nf * j] = z.re;
                stacked[nf + i + 2 * nf * j] = z.im;
            }
        }
        let mut out: Vec<i64> = crate::dlr_id::select_rows(&stacked, 2 * nf, r, r)
            .into_iter()
            .map(|i| ns[i % nf])
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

fn check_kernel_statistics<S, K>(kernel: &K) -> Result<(), DlrError>
where
    S: StatisticsType,
    K: crate::kernel::KernelProperties,
{
    // RegularizedBoseKernel (ypower == 1) is meaningful only for bosonic
    // statistics; its regularizer panics for fermionic input.
    if S::STATISTICS == Statistics::Fermionic && kernel.ypower() == 1 {
        return Err(DlrError::KernelStatisticsMismatch);
    }
    Ok(())
}

/// Pole weights `w(β, ω_i) / wmax^(2 ypower)` of `kernel`.
fn pole_weights_for<S, K>(kernel: &K, beta: f64, wmax: f64, poles: &[f64]) -> Vec<f64>
where
    S: StatisticsType + 'static,
    K: crate::kernel::KernelProperties,
{
    let scale = wmax.powi(2 * kernel.ypower());
    poles
        .iter()
        .map(|&pole| kernel.regularizer::<S>(beta, pole) / scale)
        .collect()
}

// ============================================================================
// IR <-> DLR transform
// ============================================================================

/// Linear map between the coefficients of an IR basis and of a DLR on the
/// same domain.
///
/// The DLR basis function of pole `ω_i` is expanded in the IR basis as
/// `u_i = Σ_l T[l, i] U_l` with `T[l, i] = -s_l V_l(ω_i) (w_i / w^{IR}_i)`,
/// where `w_i` and `w^{IR}_i` are the pole weights of the DLR and of the IR
/// kernel. DLR -> IR applies `T`; IR -> DLR is its least-squares inverse.
pub struct IrDlrTransform {
    fitter: RealMatrixFitter,
}

impl IrDlrTransform {
    /// Build the transform between `basis` and `dlr`.
    ///
    /// # Errors
    /// Returns [`DlrError::IncompatibleIrBasis`] if β differs, a pole lies
    /// outside `[-ωmax, ωmax]` of the basis, or the basis kernel weight
    /// vanishes at a pole; [`DlrError::KernelStatisticsMismatch`] if the basis
    /// kernel does not support the statistics.
    pub fn new<S, K>(
        basis: &impl crate::basis_trait::Basis<S, Kernel = K>,
        dlr: &DiscreteLehmannRepresentation<S>,
    ) -> Result<Self, DlrError>
    where
        S: StatisticsType + 'static,
        K: crate::kernel::KernelProperties + Clone,
    {
        check_kernel_statistics::<S, K>(basis.kernel())?;
        let rel = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
        if !rel(basis.beta(), dlr.beta) {
            return Err(DlrError::IncompatibleIrBasis(format!(
                "beta differs: IR {} vs DLR {}",
                basis.beta(),
                dlr.beta
            )));
        }
        Self::with_weights(basis, &dlr.poles, &dlr.pole_weights)
    }

    fn with_weights<S, K>(
        basis: &impl crate::basis_trait::Basis<S, Kernel = K>,
        poles: &[f64],
        weights: &[f64],
    ) -> Result<Self, DlrError>
    where
        S: StatisticsType + 'static,
        K: crate::kernel::KernelProperties + Clone,
    {
        let (beta, wmax) = (basis.beta(), basis.wmax());
        if let Some(&p) = poles.iter().find(|p| p.abs() > wmax * (1.0 + 1e-12)) {
            return Err(DlrError::IncompatibleIrBasis(format!(
                "pole {p} lies outside [-{wmax}, {wmax}]"
            )));
        }
        let ir_weights = pole_weights_for::<S, K>(basis.kernel(), beta, wmax, poles);
        let mut ratio = Vec::with_capacity(poles.len());
        for (i, (&w, &wir)) in weights.iter().zip(&ir_weights).enumerate() {
            if w == wir {
                ratio.push(1.0);
            } else if wir == 0.0 {
                return Err(DlrError::IncompatibleIrBasis(format!(
                    "IR kernel weight vanishes at pole {}",
                    poles[i]
                )));
            } else {
                ratio.push(w / wir);
            }
        }
        let v_at_poles = crate::sampling::mat_from_matrix(&basis.evaluate_omega(poles))
            .expect("Basis::evaluate_omega returns a host matrix"); // [n_poles, basis_size]
        let s = basis.svals();
        let fitmat = Mat::<f64>::from_fn([basis.size(), poles.len()], |idx| {
            let (l, i) = (idx[0], idx[1]);
            -s[l] * v_at_poles[[i, l]] * ratio[i]
        });
        Ok(Self {
            fitter: RealMatrixFitter::new(fitmat),
        })
    }

    /// IR basis size.
    pub fn ir_size(&self) -> usize {
        self.fitter.n_points()
    }

    /// Number of DLR poles.
    pub fn dlr_size(&self) -> usize {
        self.fitter.basis_size()
    }

    /// The `ir_size x dlr_size` matrix `T` mapping DLR to IR coefficients.
    pub fn matrix(&self) -> &crate::Matrix<f64> {
        self.fitter.matrix()
    }

    /// IR -> DLR along axis `dim` (least squares).
    ///
    /// # Errors
    /// Returns an error if `gl.shape()[dim]` differs from the IR basis size.
    pub fn ir_to_dlr_nd<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        gl: &TypedTensor<T>,
        dim: usize,
    ) -> crate::Result<TypedTensor<T>> {
        self.fitter.fit_nd(backend, gl, dim)
    }

    /// DLR -> IR along axis `dim`.
    ///
    /// # Errors
    /// Returns an error if `g_dlr.shape()[dim]` differs from the number of
    /// poles.
    pub fn dlr_to_ir_nd<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        g_dlr: &TypedTensor<T>,
        dim: usize,
    ) -> crate::Result<TypedTensor<T>> {
        self.fitter.evaluate_nd(backend, g_dlr, dim)
    }
}

// ============================================================================
// Basis trait implementation for DLR
// ============================================================================

impl<S> crate::basis_trait::Basis<S> for DiscreteLehmannRepresentation<S>
where
    S: StatisticsType + 'static,
{
    type Kernel = crate::kernel::LogisticKernel;

    fn kernel(&self) -> &Self::Kernel {
        // DLR always uses LogisticKernel for weight computations
        &self.kernel
    }

    fn beta(&self) -> f64 {
        self.beta
    }

    fn wmax(&self) -> f64 {
        self.wmax
    }

    fn lambda(&self) -> f64 {
        self.beta * self.wmax
    }

    fn size(&self) -> usize {
        self.poles.len()
    }

    fn accuracy(&self) -> f64 {
        self.accuracy
    }

    fn significance(&self) -> Vec<f64> {
        // All poles are equally significant in DLR
        vec![1.0; self.poles.len()]
    }

    fn svals(&self) -> Vec<f64> {
        // All poles are equally significant in DLR (no singular value concept)
        vec![1.0; self.poles.len()]
    }

    fn default_tau_sampling_points(&self) -> Vec<f64> {
        self.tau_nodes().to_vec()
    }

    fn default_matsubara_sampling_points(
        &self,
        positive_only: bool,
    ) -> Vec<crate::freq::MatsubaraFreq<S>> {
        self.matsubara_nodes(positive_only)
            .iter()
            .map(|&n| crate::freq::MatsubaraFreq::new(n).expect("node parity matches statistics"))
            .collect()
    }

    fn evaluate_tau(&self, tau: &[f64]) -> crate::Matrix<f64> {
        use crate::taufuncs::normalize_tau;

        let n_points = tau.len();
        let n_poles = self.poles.len();
        Mat::<f64>::from_fn([n_points, n_poles], |idx| {
            let tau_val = tau[idx[0]];
            let pole = self.poles[idx[1]];
            let pole_weight = self.pole_weights[idx[1]];
            match S::STATISTICS {
                Statistics::Fermionic => {
                    gtau_single_pole::<S>(tau_val, pole, self.beta) * pole_weight
                }
                Statistics::Bosonic => {
                    if pole == 0.0 {
                        self.zero_pole_tau_limit()
                    } else if pole > 0.0 {
                        let tau_norm = normalize_tau::<S>(tau_val, self.beta).0;
                        let denominator = -(-self.beta * pole).exp_m1();
                        -(-tau_norm * pole).exp() * pole_weight / denominator
                    } else {
                        let tau_norm = normalize_tau::<S>(tau_val, self.beta).0;
                        let denominator = -(self.beta * pole).exp_m1();
                        (pole * (self.beta - tau_norm)).exp() * pole_weight / denominator
                    }
                }
            }
        })
        .into_typed()
    }

    fn evaluate_matsubara(
        &self,
        freqs: &[crate::freq::MatsubaraFreq<S>],
    ) -> crate::Matrix<num_complex::Complex<f64>> {
        use num_complex::Complex;

        let n_points = freqs.len();
        let n_poles = self.poles.len();

        // Evaluate MatsubaraPoles basis functions
        Mat::<Complex<f64>>::from_fn([n_points, n_poles], |idx| {
            let freq = &freqs[idx[0]];
            let pole = self.poles[idx[1]];
            let pole_weight = self.pole_weights[idx[1]];

            // iν = i * π * (2n + ζ) / β
            let iv = freq.value_imaginary(self.beta);

            // u_i(iν) = pole_weight / (iν - pole_i), where `pole_weight`
            // matches the ω-domain normalization of the source IR basis.
            if S::STATISTICS == Statistics::Bosonic && pole == 0.0 {
                if crate::freq::is_zero(freq) {
                    Complex::new(self.zero_pole_matsubara_limit(), 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                }
            } else {
                Complex::new(pole_weight, 0.0) / (iv - Complex::new(pole, 0.0))
            }
        })
        .into_typed()
    }

    fn evaluate_omega(&self, _omega: &[f64]) -> crate::Matrix<f64> {
        // TODO(#205): For the IR basis, evaluate_omega returns V_l(omega).
        // For DLR, the "basis functions" in omega-space are single-pole
        // functions (conceptually delta functions at the pole positions),
        // which do not have a well-defined continuous representation on the
        // real-frequency axis analogous to V_l(omega).  A proper
        // implementation would require either:
        //   (a) returning the IR basis's V_l(omega) (but DLR does not store
        //       the IR basis), or
        //   (b) defining an appropriate discretized representation for the
        //       pole basis in omega-space.
        // Until the semantics are clarified, this remains unimplemented.
        unimplemented!(
            "evaluate_omega is not well-defined for DLR; \
             use the underlying IR basis for real-frequency evaluation"
        )
    }

    fn default_omega_sampling_points(&self) -> Vec<f64> {
        // DLR poles ARE the omega sampling points
        self.poles.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Bosonic, Fermionic};

    /// Generic test for periodicity/anti-periodicity
    fn test_periodicity_generic<S: StatisticsType>(expected_sign: f64, stat_name: &str) {
        let beta = 1.0;
        let omega = 5.0;

        // Test periodicity by comparing G(τ) with G(τ-β)
        // Since normalize_tau is restricted to [-β, β], we test:
        // For τ ∈ (0, β]: compare G(τ) with G(τ-β)
        // For fermions: G(τ) should equal -G(τ-β)
        // For bosons: G(τ) should equal G(τ-β)
        for tau in [0.1, 0.3, 0.7] {
            let g_tau = gtau_single_pole::<S>(tau, omega, beta);
            let g_tau_minus_beta = gtau_single_pole::<S>(tau - beta, omega, beta);

            // For fermions: G(τ) = -G(τ-β) → G(τ-β) = -G(τ)
            // For bosons: G(τ) = G(τ-β)
            let expected = expected_sign * g_tau;

            assert!(
                (expected - g_tau_minus_beta).abs() < 1e-14,
                "{} periodicity violated at τ={}: G(τ)={}, G(τ-β)={}, expected={}",
                stat_name,
                tau,
                g_tau,
                g_tau_minus_beta,
                expected
            );
        }
    }

    #[test]
    fn test_fermionic_antiperiodicity() {
        // Fermions: G(τ+β) = -G(τ)
        test_periodicity_generic::<Fermionic>(-1.0, "Fermionic");
    }

    #[test]
    fn test_bosonic_periodicity() {
        // Bosons: G(τ+β) = G(τ)
        test_periodicity_generic::<Bosonic>(1.0, "Bosonic");
    }

    #[test]
    fn test_generic_function_matches_specific() {
        let beta = 1.0;
        let omega = 5.0;
        let tau = 0.5;

        // Test that generic function matches specific functions
        let g_f_specific = fermionic_single_pole(tau, omega, beta);
        let g_f_generic = gtau_single_pole::<Fermionic>(tau, omega, beta);

        let g_b_specific = bosonic_single_pole(tau, omega, beta);
        let g_b_generic = gtau_single_pole::<Bosonic>(tau, omega, beta);

        assert!(
            (g_f_specific - g_f_generic).abs() < 1e-14,
            "Fermionic: specific={}, generic={}",
            g_f_specific,
            g_f_generic
        );
        assert!(
            (g_b_specific - g_b_generic).abs() < 1e-14,
            "Bosonic: specific={}, generic={}",
            g_b_specific,
            g_b_generic
        );
    }
}

#[cfg(test)]
#[path = "dlr_tests.rs"]
mod dlr_tests;

#[cfg(test)]
#[path = "dlr_independent_tests.rs"]
mod dlr_independent_tests;
