//! Discrete Lehmann Representation (DLR)
//!
//! This module provides the Discrete Lehmann Representation (DLR) basis,
//! which represents Green's functions as a linear combination of poles on the
//! real-frequency axis.

use crate::error::{Error, require_finite, require_positive_finite};
use crate::fitters::FitScalar;
use crate::fitters::RealMatrixFitter;
use crate::freq::MatsubaraFreq;
use crate::gemm::GemmBackendHandle;
use crate::matrix::Mat;
use crate::taufuncs::normalize_tau;
use crate::traits::{Statistics, StatisticsType};
use num_complex::Complex;
use std::marker::PhantomData;
use std::sync::OnceLock;
use tenferro_tensor::TypedTensor;

/// Generic single-pole Green's function at imaginary time τ
///
/// Computes G(τ) for either fermionic or bosonic statistics based on the type parameter S.
/// Both use G(τ) = -exp(-ω×τ) / (1 ± exp(-β×ω)) (+ for fermions, - for bosons),
/// the imaginary-time counterpart of [`giwn_single_pole`]'s G(iν) = 1/(iν - ω).
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `tau` - Imaginary time τ ∈ [-β, β]; τ < 0 is mapped to [0, β] by
///   (anti-)periodicity (see [`fermionic_single_pole`] and [`bosonic_single_pole`])
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Errors
///
/// * [`Error::InvalidParameter`] if `beta` is not positive and finite, or
///   `omega` is not finite
/// * [`Error::OutOfDomain`] if `tau` is outside [-β, β] or NaN
///
/// # Example
/// ```
/// use sparse_ir::traits::{Bosonic, Fermionic};
/// use sparse_ir::{bosonic_single_pole, fermionic_single_pole, gtau_single_pole};
///
/// let g_f = gtau_single_pole::<Fermionic>(0.5, 5.0, 1.0).unwrap();
/// assert_eq!(g_f, fermionic_single_pole(0.5, 5.0, 1.0).unwrap());
///
/// let g_b = gtau_single_pole::<Bosonic>(0.5, 5.0, 1.0).unwrap();
/// assert_eq!(g_b, bosonic_single_pole(0.5, 5.0, 1.0).unwrap());
/// ```
pub fn gtau_single_pole<S: StatisticsType>(tau: f64, omega: f64, beta: f64) -> Result<f64, Error> {
    match S::STATISTICS {
        Statistics::Fermionic => fermionic_single_pole(tau, omega, beta),
        Statistics::Bosonic => bosonic_single_pole(tau, omega, beta),
    }
}

/// Compute fermionic single-pole Green's function at imaginary time τ
///
/// Evaluates G(τ) = -exp(-ω×τ) / (1 + exp(-β×ω)) for a single pole at frequency ω.
///
/// Supports negative τ with anti-periodic boundary conditions:
/// - G(τ + β) = -G(τ) (fermionic anti-periodicity)
/// - Valid for τ ∈ [-β, β]; τ is normalized with
///   [`normalize_tau`]
///
/// # Arguments
/// * `tau` - Imaginary time τ ∈ [-β, β]
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Errors
///
/// * [`Error::InvalidParameter`] if `beta` is not positive and finite, or
///   `omega` is not finite
/// * [`Error::OutOfDomain`] if `tau` is outside [-β, β] or NaN
///
/// # Example
/// ```
/// use sparse_ir::fermionic_single_pole;
///
/// let beta = 1.0;
/// let omega = 5.0;
/// let tau = 0.5 * beta;
/// let g = fermionic_single_pole(tau, omega, beta).unwrap();
///
/// let expected = -(-omega * tau).exp() / (1.0 + (-beta * omega).exp());
/// assert!((g - expected).abs() < 1e-15);
///
/// // Anti-periodicity: G(τ - β) = -G(τ)
/// assert!((fermionic_single_pole(tau - beta, omega, beta).unwrap() + g).abs() < 1e-15);
/// ```
pub fn fermionic_single_pole(tau: f64, omega: f64, beta: f64) -> Result<f64, Error> {
    use crate::traits::Fermionic;

    // Normalize τ to [0, β] and track sign from anti-periodicity
    // G(τ + β) = -G(τ) for fermions
    let (tau_normalized, sign) = normalize_tau::<Fermionic>(tau, beta)?;
    require_finite("omega", omega)?;
    Ok(sign * fermionic_single_pole_unchecked(tau_normalized, omega, beta))
}

/// Fermionic single-pole G(τ) for τ already in [0, β], without the sign of
/// the antiperiodic continuation
///
/// Checked callers only: β positive and finite, ω finite. The DLR checks its
/// poles in `from_ir_with_poles` and they cannot be changed afterwards.
pub(crate) fn fermionic_single_pole_unchecked(tau_normalized: f64, omega: f64, beta: f64) -> f64 {
    // Avoid overflow for large negative ω by factoring out exp(βω).
    // Both branches keep the exponent non-positive.
    if omega >= 0.0 {
        -(-omega * tau_normalized).exp() / (1.0 + (-beta * omega).exp())
    } else {
        -(omega * (beta - tau_normalized)).exp() / (1.0 + (beta * omega).exp())
    }
}

/// Compute bosonic single-pole Green's function at imaginary time τ
///
/// Evaluates G(τ) = -exp(-ω×τ) / (1 - exp(-β×ω)) for a single pole at frequency ω.
///
/// This is the imaginary-time counterpart of [`giwn_single_pole`]:
/// G(iνn) = ∫₀^β dτ exp(iνn×τ) G(τ) = 1/(iνn - ω). G(τ) is negative for ω > 0
/// and positive for ω < 0, the same sign convention as [`fermionic_single_pole`]
/// and the bosonic τ functions of [`DiscreteLehmannRepresentation`].
///
/// Supports negative τ with periodic boundary conditions:
/// - G(τ + β) = G(τ) (bosonic periodicity)
/// - Valid for τ ∈ [-β, β]; τ is normalized with
///   [`normalize_tau`]
///
/// ω = 0 is a genuine pole of the Bose factor, so the result is infinite there:
/// `-inf` for `omega = +0.0` (the ω → 0⁺ limit) and `+inf` for `omega = -0.0`.
/// [`DiscreteLehmannRepresentation`] evaluates a zero pole through its finite,
/// regularized limit instead.
///
/// # Arguments
/// * `tau` - Imaginary time τ ∈ [-β, β]
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Errors
///
/// * [`Error::InvalidParameter`] if `beta` is not positive and finite, or
///   `omega` is not finite
/// * [`Error::OutOfDomain`] if `tau` is outside [-β, β] or NaN
///
/// # Example
/// ```
/// use sparse_ir::bosonic_single_pole;
///
/// let beta = 1.0;
/// let omega = 5.0;
/// let tau = 0.5 * beta;
/// let g = bosonic_single_pole(tau, omega, beta).unwrap();
///
/// let expected = -(-omega * tau).exp() / (1.0 - (-beta * omega).exp());
/// assert!((g - expected).abs() <= 1e-14 * expected.abs());
/// assert!(g < 0.0);
///
/// // Periodicity: G(τ - β) = G(τ)
/// assert!((bosonic_single_pole(tau - beta, omega, beta).unwrap() - g).abs() < 1e-15);
/// ```
pub fn bosonic_single_pole(tau: f64, omega: f64, beta: f64) -> Result<f64, Error> {
    use crate::traits::Bosonic;

    // Normalize τ to [0, β] using periodicity
    // G(τ + β) = G(τ) for bosons
    let tau_normalized = normalize_tau::<Bosonic>(tau, beta)?.0;
    require_finite("omega", omega)?;

    // Avoid overflow for large negative ω by factoring out exp(βω): both
    // branches keep the exponents non-positive. expm1 keeps the Bose
    // denominator 1 - exp(-β|ω|) accurate for small β|ω|. This is the same
    // form as the bosonic arm of `DiscreteLehmannRepresentation::evaluate_tau`.
    // At ω = ±0 the denominator is a signed zero and the result is ∓inf.
    if omega >= 0.0 {
        // 1 - exp(-βω) = -expm1(-βω)
        let denominator = -(-beta * omega).exp_m1();
        Ok(-(-omega * tau_normalized).exp() / denominator)
    } else {
        // -exp(-ωτ) / (1 - exp(-βω)) = exp(ω(β - τ)) / (1 - exp(βω))
        let denominator = -(beta * omega).exp_m1();
        Ok((omega * (beta - tau_normalized)).exp() / denominator)
    }
}

/// Generic single-pole Green's function at Matsubara frequency
///
/// Computes G(iν) = 1/(iν - ω) for a single pole at frequency ω.
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
/// Complex-valued Green's function G(iν)
///
/// # Errors
///
/// [`Error::InvalidParameter`] if `beta` is not positive and finite, or
/// `omega` is not finite
pub fn giwn_single_pole<S: StatisticsType>(
    matsubara_freq: &MatsubaraFreq<S>,
    omega: f64,
    beta: f64,
) -> Result<Complex<f64>, Error> {
    require_positive_finite("beta", beta)?;
    require_finite("omega", omega)?;
    // G(iν) = 1/(iν - ω)
    let wn = matsubara_freq.value(beta);
    let denominator = Complex::new(0.0, 1.0) * wn - Complex::new(omega, 0.0);
    Ok(Complex::new(1.0, 0.0) / denominator)
}

// ============================================================================
// Discrete Lehmann Representation
// ============================================================================

/// Discrete Lehmann Representation (DLR)
///
/// The DLR is a variant of the IR basis based on a "sketching" of the analytic
/// continuation kernel K. Instead of using singular value expansion, it represents
/// Green's functions as a linear combination of poles on the real-frequency axis:
///
/// ```text
/// G(iν) = Σ_i a[i] * reg[i] / (iν - ω[i])
/// ```
///
/// where:
/// - `ω[i]` are pole positions on the real axis
/// - `a[i]` are expansion coefficients
/// - `reg[i]` are the kernel regularizers `w(β, ω_i)`
///
/// [`regularizers()`](Self::regularizers) returns `w(β, ω_i)`: 1 for fermions
/// and `tanh(βω_i/2)` for bosons with `LogisticKernel`, and `ω_i` for
/// `RegularizedBoseKernel`. The DLR functions are `-K(τ, ω_i)` and its Fourier
/// transform for the physical kernel `K(τ, ω) = Σ_l u_l(τ) s_l v_l(ω)` of the
/// source basis.
///
/// Two constructions are available:
/// - **independent (default)**: [`DiscreteLehmannRepresentation::new`] /
///   [`DlrBuilder`] select the poles by an interpolative decomposition of the
///   discretized logistic kernel (Kaye, Chen, Parcollet, PRB 105, 235115),
///   without building an IR basis;
/// - **IR-derived**: `DiscreteLehmannRepresentation::from_ir` (trait
///   `DlrFromIr` of the IR crate) uses the default real-frequency sampling
///   points of an IR basis as poles, and `from_ir_with_poles` takes the poles.
///
/// Conversions to and from an IR basis are provided by [`IrDlrTransform`];
/// an IR-derived DLR carries one for its source basis.
///
/// Imaginary-time and Matsubara nodes for square interpolation are selected
/// by a row interpolative decomposition and are exposed through
/// [`Basis::default_tau_sampling_points`](crate::basis_trait::Basis::default_tau_sampling_points)
/// and
/// [`Basis::default_matsubara_sampling_points`](crate::basis_trait::Basis::default_matsubara_sampling_points),
/// so [`TauSampling::new`](crate::TauSampling::new) and
/// [`MatsubaraSampling::new`](crate::MatsubaraSampling::new) work on a DLR.
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
pub struct DiscreteLehmannRepresentation<S>
where
    S: StatisticsType,
{
    /// Pole positions on the real-frequency axis ω ∈ [-ωmax, ωmax]
    poles: Vec<f64>,

    /// Inverse temperature β
    beta: f64,

    /// Maximum frequency ωmax
    wmax: f64,

    /// Power with which the source kernel scales the spectral variable.
    kernel_ypower: i32,

    /// Accuracy of the representation
    accuracy: f64,

    /// Regularizers for each pole: regularizer[i] = w(β, ω_i) of the logistic
    /// kernel, or of the kernel of the source IR basis for an IR-derived DLR
    regularizers: Vec<f64>,

    /// Pole weights used in tau and Matsubara evaluations.
    ///
    /// They equal `regularizers`: the τ functions are
    /// `-w_i e^{-τω_i} / (1 ± e^{-βω_i})` and the Matsubara functions
    /// `w_i / (iν - ω_i)`, which is `-K(τ, ω_i)` and its Fourier transform for
    /// the physical kernel of the source basis.
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

/// Regularizer `w(β, ω)` of the logistic kernel: 1 for fermions and
/// `tanh(βω/2)` for bosons
fn logistic_regularizer<S: StatisticsType>(beta: f64, omega: f64) -> f64 {
    match S::STATISTICS {
        Statistics::Fermionic => 1.0,
        Statistics::Bosonic => (0.5 * beta * omega).tanh(),
    }
}

/// Builder for the independent (interpolative-decomposition) DLR.
///
/// ```
/// use sparse_ir::{DlrBuilder, Fermionic};
/// let dlr = DlrBuilder::<Fermionic>::new(10.0, 10.0)
///     .accuracy(1e-10)
///     .build()
///     .unwrap();
/// assert!(!dlr.poles().is_empty());
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
    /// Default target accuracy, `1e-14`.
    ///
    /// The poles are selected by a pivoted Gram–Schmidt factorization of the
    /// kernel in double precision, whose residuals cannot fall much below
    /// `1e-15`. An accuracy below about `1e-14` is therefore beyond what the
    /// selection can resolve: it keeps adding near-redundant poles chosen by
    /// rounding, which makes the representation larger and its fits *less*
    /// accurate. `1e-14` is the tolerance the reference implementations use
    /// at full precision (cppdlr warns below it).
    pub const DEFAULT_ACCURACY: f64 = 1e-14;

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
    ///
    /// Values below about `1e-14` are beyond double-precision resolution; see
    /// [`Self::DEFAULT_ACCURACY`].
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
    /// [`Error::InvalidParameter`] if `beta` or `wmax` is not positive and
    /// finite, the accuracy is not in (0, 1), or `max_size` is 0.
    pub fn build(self) -> Result<DiscreteLehmannRepresentation<S>, Error> {
        require_positive_finite("beta", self.beta)?;
        require_positive_finite("wmax", self.wmax)?;
        crate::error::require_accuracy("accuracy", Some(self.accuracy))?;
        crate::error::require_nonzero_size("max_size", self.max_size)?;
        let lambda = self.beta * self.wmax;
        let poles: Vec<f64> = crate::dlr_id::select_poles(lambda, self.accuracy, self.max_size)
            .into_iter()
            .map(|w| w / self.beta)
            .collect();
        let regularizers = poles
            .iter()
            .map(|&pole| logistic_regularizer::<S>(self.beta, pole))
            .collect();
        Ok(DiscreteLehmannRepresentation::from_parts(
            self.beta,
            self.wmax,
            self.accuracy,
            poles,
            regularizers,
            0,
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

    /// Pole positions on the real-frequency axis, in the order they were
    /// given to `from_ir_with_poles` (sorted ascending when chosen by
    /// [`Self::new`] or `from_ir`)
    pub fn poles(&self) -> &[f64] {
        &self.poles
    }

    /// Regularizers of the poles: `regularizers[i] = w(β, poles[i])` of the
    /// kernel of the IR basis this DLR was built from (of the logistic
    /// kernel for an independent DLR)
    pub fn regularizers(&self) -> &[f64] {
        &self.regularizers
    }

    /// Number of functions of the IR basis this DLR was built from: the
    /// extent of the IR axis of [`Self::from_ir_nd`] and [`Self::to_ir_nd`].
    /// `None` for a DLR built independently of an IR basis.
    pub fn ir_basis_size(&self) -> Option<usize> {
        self.ir.as_ref().map(IrDlrTransform::ir_size)
    }

    /// IR <-> DLR transform of the source basis (IR-derived DLR only).
    pub fn ir_transform(&self) -> Option<&IrDlrTransform> {
        self.ir.as_ref()
    }

    /// Build a DLR independently of any IR basis (default construction).
    ///
    /// Equivalent to `DlrBuilder::new(beta, wmax).accuracy(accuracy).build()`.
    /// An `accuracy` below about `1e-14` is beyond double-precision
    /// resolution and makes the DLR larger without making it more accurate;
    /// see [`DlrBuilder::DEFAULT_ACCURACY`].
    ///
    /// # Errors
    /// See [`DlrBuilder::build`].
    pub fn new(beta: f64, wmax: f64, accuracy: f64) -> Result<Self, Error>
    where
        S: 'static,
    {
        DlrBuilder::new(beta, wmax).accuracy(accuracy).build()
    }

    /// Assemble a DLR from checked parts: the constructors of the IR crate
    /// build on this after validating the poles against their basis.
    ///
    /// `regularizers[i]` is the kernel regularizer `w(β, poles[i])`, which is
    /// also the weight of the pole functions; `kernel_ypower` is 0 or 1 when a
    /// bosonic pole lies at 0. `ir` is the transform to the source IR basis,
    /// if any.
    #[doc(hidden)]
    pub fn from_parts(
        beta: f64,
        wmax: f64,
        accuracy: f64,
        poles: Vec<f64>,
        regularizers: Vec<f64>,
        kernel_ypower: i32,
        ir: Option<IrDlrTransform>,
    ) -> Self {
        let pole_weights = regularizers.clone();
        Self {
            poles,
            beta,
            wmax,
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

    fn zero_pole_tau_limit(&self) -> f64 {
        match self.kernel_ypower {
            // -lim_{ω→0} w(β, ω) e^{-τω} / (1 - e^{-βω}) with w = tanh(βω/2) or ω
            0 => -0.5,
            1 => -1.0 / self.beta,
            // Unreachable: from_ir_with_poles rejects a bosonic pole at 0 for any
            // other ypower, and neither the poles nor the kernel can be
            // changed afterwards.
            _ => panic!(
                "DLR tau evaluation does not support kernel ypower = {}",
                self.kernel_ypower
            ),
        }
    }

    fn zero_pole_matsubara_limit(&self) -> f64 {
        match self.kernel_ypower {
            // lim_{ω→0} w(β, ω) / (0 - ω) at n = 0 with w = tanh(βω/2) or ω
            0 => -0.5 * self.beta,
            1 => -1.0,
            // Unreachable: from_ir_with_poles rejects a bosonic pole at 0 for any
            // other ypower, and neither the poles nor the kernel can be
            // changed afterwards.
            _ => panic!(
                "DLR Matsubara evaluation does not support kernel ypower = {}",
                self.kernel_ypower
            ),
        }
    }

    // ========================================================================
    // Public API (generic, user-friendly)
    // ========================================================================

    /// Convert IR coefficients to DLR (N-dimensional, generic over real/complex)
    ///
    /// # Type Parameters
    /// * `T` - Element type (`f64` or `Complex<f64>`)
    ///
    /// # Arguments
    /// * `gl` - IR coefficients as N-D tensor
    /// * `dim` - Dimension along which to transform
    ///
    /// # Returns
    /// DLR coefficients as N-D tensor, with [`Basis::size`](crate::basis_trait::Basis::size)
    /// (the number of poles) entries along `dim`. An empty batch (a zero
    /// extent on another axis) gives an empty result.
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `gl`
    /// * [`Error::ShapeMismatch`] of the input if `gl` does not have
    ///   [`Self::ir_basis_size`] entries along `dim`
    /// * [`Error::DecompositionFailed`] if the SVD of the fitting matrix fails
    ///   (its entries are finite, since the poles are)
    pub fn from_ir_nd<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        gl: &TypedTensor<T>,
        dim: usize,
    ) -> Result<TypedTensor<T>, Error> {
        // An empty batch gives an empty result; the fit fails only if the
        // SVD of the fitting matrix does.
        self.require_ir()?.ir_to_dlr_nd(backend, gl, dim)
    }

    /// Convert DLR coefficients to IR (N-dimensional, generic over real/complex)
    ///
    /// # Type Parameters
    /// * `T` - Element type (`f64` or `Complex<f64>`)
    ///
    /// # Arguments
    /// * `g_dlr` - DLR coefficients as N-D tensor
    /// * `dim` - Dimension along which to transform
    ///
    /// # Returns
    /// IR coefficients as N-D tensor, with [`Self::ir_basis_size`] entries
    /// along `dim`. An empty batch gives an empty result.
    ///
    /// # Errors
    ///
    /// * [`Error::AxisOutOfRange`] if `dim` is not an axis of `g_dlr`
    /// * [`Error::ShapeMismatch`] of the input if `g_dlr` does not have
    ///   [`Basis::size`](crate::basis_trait::Basis::size) (the number of
    ///   poles) entries along `dim`
    pub fn to_ir_nd<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        g_dlr: &TypedTensor<T>,
        dim: usize,
    ) -> Result<TypedTensor<T>, Error> {
        self.require_ir()?.dlr_to_ir_nd(backend, g_dlr, dim)
    }

    fn require_ir(&self) -> Result<&IrDlrTransform, Error> {
        self.ir.as_ref().ok_or_else(|| Error::NotSupported {
            what: "IR <-> DLR conversion of a DLR that was not built from an IR basis; \
                   build an IrDlrTransform for the IR basis instead"
                .to_string(),
        })
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
            let mat = self
                .evaluate_tau(&taus)
                .expect("the candidate times lie in [0, beta]");
            let rows = crate::dlr_id::select_rows(
                mat.host_data().expect("an owned matrix is host-resident"),
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
        let mat = self
            .evaluate_matsubara(&freqs)
            .expect("evaluating the DLR functions at Matsubara frequencies cannot fail");
        let data = mat.host_data().expect("an owned matrix is host-resident");
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
    /// The transform with the `ir_size x dlr_size` matrix `T` mapping DLR to
    /// IR coefficients (see the type documentation). The IR crate builds it
    /// from its basis.
    #[doc(hidden)]
    pub fn from_matrix(matrix: Mat<f64>) -> Self {
        Self {
            fitter: RealMatrixFitter::new(matrix),
        }
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
    /// The errors of [`DiscreteLehmannRepresentation::from_ir_nd`].
    pub fn ir_to_dlr_nd<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        gl: &TypedTensor<T>,
        dim: usize,
    ) -> Result<TypedTensor<T>, Error> {
        self.fitter.fit_nd(backend, gl, dim)
    }

    /// DLR -> IR along axis `dim`.
    ///
    /// # Errors
    /// The errors of [`DiscreteLehmannRepresentation::to_ir_nd`].
    pub fn dlr_to_ir_nd<T: FitScalar>(
        &self,
        backend: Option<&GemmBackendHandle>,
        g_dlr: &TypedTensor<T>,
        dim: usize,
    ) -> Result<TypedTensor<T>, Error> {
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

    fn default_tau_sampling_points(&self) -> Result<Vec<f64>, Error> {
        Ok(self.tau_nodes().to_vec())
    }

    fn default_matsubara_sampling_points(
        &self,
        positive_only: bool,
    ) -> Result<Vec<crate::freq::MatsubaraFreq<S>>, Error> {
        self.matsubara_nodes(positive_only)
            .iter()
            .map(|&n| crate::freq::MatsubaraFreq::new(n))
            .collect()
    }

    fn evaluate_tau(&self, tau: &[f64]) -> Result<crate::Matrix<f64>, Error> {
        let n_poles = self.poles.len();
        // Normalize every τ first: this rejects a τ outside [-β, β] and NaN
        // for every pole, including the bosonic pole at 0, whose limit does
        // not depend on τ.
        let normalized = tau
            .iter()
            .map(|&t| normalize_tau::<S>(t, self.beta))
            .collect::<Result<Vec<(f64, f64)>, Error>>()?;
        if normalized.is_empty() {
            // An empty set of points gives an empty matrix.
            return Ok(Mat::<f64>::from_elem([0, n_poles], 0.0).into_typed());
        }
        Ok(Mat::<f64>::from_fn([normalized.len(), n_poles], |idx| {
            let (tau_norm, sign) = normalized[idx[0]];
            let pole = self.poles[idx[1]];
            let pole_weight = self.pole_weights[idx[1]];
            match S::STATISTICS {
                Statistics::Fermionic => {
                    sign * fermionic_single_pole_unchecked(tau_norm, pole, self.beta) * pole_weight
                }
                Statistics::Bosonic => {
                    // The bosonic sign of normalize_tau is always 1.
                    if pole == 0.0 {
                        self.zero_pole_tau_limit()
                    } else if pole > 0.0 {
                        let denominator = -(-self.beta * pole).exp_m1();
                        -(-tau_norm * pole).exp() * pole_weight / denominator
                    } else {
                        let denominator = -(self.beta * pole).exp_m1();
                        (pole * (self.beta - tau_norm)).exp() * pole_weight / denominator
                    }
                }
            }
        })
        .into_typed())
    }

    fn evaluate_matsubara(
        &self,
        freqs: &[crate::freq::MatsubaraFreq<S>],
    ) -> Result<crate::Matrix<num_complex::Complex<f64>>, Error> {
        use num_complex::Complex;

        let n_points = freqs.len();
        let n_poles = self.poles.len();
        if n_points == 0 {
            // See evaluate_tau.
            return Ok(
                Mat::<Complex<f64>>::from_elem([0, n_poles], Complex::new(0.0, 0.0)).into_typed(),
            );
        }

        // Evaluate MatsubaraPoles basis functions
        Ok(Mat::<Complex<f64>>::from_fn([n_points, n_poles], |idx| {
            let freq = &freqs[idx[0]];
            let pole = self.poles[idx[1]];
            let pole_weight = self.pole_weights[idx[1]];

            // iν = iπn/β, with n = freq.n() (odd for fermions, even for bosons)
            let iv = freq.value_imaginary(self.beta);

            // u_i(iν) = pole_weight / (iν - pole_i), where `pole_weight` is the
            // regularizer w(β, ω_i) of the source kernel.
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
        .into_typed())
    }

    fn evaluate_omega(&self, _omega: &[f64]) -> Result<crate::Matrix<f64>, Error> {
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
        // Until the semantics are clarified, this is NotSupported.
        Err(Error::NotSupported {
            what: "evaluate_omega of a DLR: its pole functions have no real-frequency \
                   representation (#205); use the IR basis"
                .to_string(),
        })
    }

    fn default_omega_sampling_points(&self) -> Result<Vec<f64>, Error> {
        // DLR poles ARE the omega sampling points
        Ok(self.poles.clone())
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
            let g_tau = gtau_single_pole::<S>(tau, omega, beta).unwrap();
            let g_tau_minus_beta = gtau_single_pole::<S>(tau - beta, omega, beta).unwrap();

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
        let g_f_specific = fermionic_single_pole(tau, omega, beta).unwrap();
        let g_f_generic = gtau_single_pole::<Fermionic>(tau, omega, beta).unwrap();

        let g_b_specific = bosonic_single_pole(tau, omega, beta).unwrap();
        let g_b_generic = gtau_single_pole::<Bosonic>(tau, omega, beta).unwrap();

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
