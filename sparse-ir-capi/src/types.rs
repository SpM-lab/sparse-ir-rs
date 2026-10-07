//! Opaque types for C API
//!
//! All Rust objects are wrapped in opaque pointers to hide implementation
//! details from C code.

use crate::{SPIR_STATISTICS_BOSONIC, SPIR_STATISTICS_FERMIONIC};
use num_complex::Complex;
use sparse_ir::basis::FiniteTempBasis;
use sparse_ir::basis_trait::Basis;
use sparse_ir::dlr::DiscreteLehmannRepresentation;
use sparse_ir::fitters::InplaceFitter;
use sparse_ir::freq::MatsubaraFreq;
use sparse_ir::gemm::GemmBackendHandle;
use sparse_ir::kernel::{AbstractKernel, CentrosymmKernel, LogisticKernel, RegularizedBoseKernel};
use sparse_ir::poly::PiecewiseLegendrePolyVector;
use sparse_ir::polyfourier::PiecewiseLegendreFTVector;
use sparse_ir::sve::SVEResult;
use sparse_ir::taufuncs::normalize_tau;
use sparse_ir::traits::{Statistics, StatisticsType};
use sparse_ir::{Bosonic, Fermionic};
use sparse_ir::{TypedTensorView, TypedTensorViewMut};
use std::sync::Arc;

/// Convert Statistics enum to C-API integer
#[inline]
#[allow(dead_code)]
pub(crate) fn statistics_to_c(stats: Statistics) -> i32 {
    match stats {
        Statistics::Fermionic => SPIR_STATISTICS_FERMIONIC,
        Statistics::Bosonic => SPIR_STATISTICS_BOSONIC,
    }
}

/// Convert C-API integer to Statistics enum
#[inline]
#[allow(dead_code)]
pub(crate) fn statistics_from_c(value: i32) -> Result<Statistics, i32> {
    match value {
        SPIR_STATISTICS_FERMIONIC => Ok(Statistics::Fermionic),
        SPIR_STATISTICS_BOSONIC => Ok(Statistics::Bosonic),
        _ => Err(value),
    }
}

/// Function domain type for continuous functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FunctionDomain {
    /// Tau domain with periodicity (statistics-dependent)
    Tau(Statistics),
    /// Omega (frequency) domain without periodicity
    Omega,
}

impl FunctionDomain {
    /// Check if this is a tau function with the given statistics
    #[allow(dead_code)]
    pub(crate) fn is_tau_with_statistics(&self, stats: Statistics) -> bool {
        matches!(self, FunctionDomain::Tau(s) if *s == stats)
    }

    /// Check if this is an omega function
    #[allow(dead_code)]
    pub(crate) fn is_omega(&self) -> bool {
        matches!(self, FunctionDomain::Omega)
    }
}

/// Opaque kernel type for C API (compatible with libsparseir)
///
/// This is a tagged union that can hold either LogisticKernel or RegularizedBoseKernel.
/// The actual type is determined by which constructor was used.
///
/// Note: Named `spir_kernel` to match libsparseir C++ API exactly.
/// The internal structure is hidden using a void pointer to prevent exposing KernelType to C.
#[repr(C)]
pub struct spir_kernel {
    pub(crate) _private: *const std::ffi::c_void,
}

/// Opaque SVE result type for C API (compatible with libsparseir)
///
/// Contains singular values and singular functions from SVE computation.
///
/// Note: Named `spir_sve_result` to match libsparseir C++ API exactly.
/// The internal structure is hidden using a void pointer to prevent exposing `Arc<SVEResult>` to C.
#[repr(C)]
pub struct spir_sve_result {
    pub(crate) _private: *const std::ffi::c_void,
}

/// Opaque basis type for C API (compatible with libsparseir)
///
/// Represents a finite temperature basis (IR or DLR).
///
/// Note: Named `spir_basis` to match libsparseir C++ API exactly.
/// The internal structure is hidden using a void pointer to prevent exposing BasisType to C.
#[repr(C)]
pub struct spir_basis {
    pub(crate) _private: *const std::ffi::c_void,
}

/// Internal basis type (not exposed to C)
#[derive(Clone)]
pub(crate) enum BasisType {
    LogisticFermionic(Arc<FiniteTempBasis<LogisticKernel, Fermionic>>),
    LogisticBosonic(Arc<FiniteTempBasis<LogisticKernel, Bosonic>>),
    // No C ABI constructor creates this combination (see #241); the dispatch
    // arms below still handle it defensively for opaque handles.
    #[allow(dead_code)]
    RegularizedBoseFermionic(Arc<FiniteTempBasis<RegularizedBoseKernel, Fermionic>>),
    RegularizedBoseBosonic(Arc<FiniteTempBasis<RegularizedBoseKernel, Bosonic>>),
    // DLR (Discrete Lehmann Representation) variants
    // Note: DLR always uses LogisticKernel internally, regardless of input kernel type
    DLRFermionic(Arc<sparse_ir::dlr::DiscreteLehmannRepresentation<Fermionic>>),
    DLRBosonic(Arc<sparse_ir::dlr::DiscreteLehmannRepresentation<Bosonic>>),
}

/// Internal kernel type (not exposed to C)
#[derive(Clone)]
pub(crate) enum KernelType {
    Logistic(Arc<LogisticKernel>),
    RegularizedBose(Arc<RegularizedBoseKernel>),
}

impl spir_kernel {
    /// Get a reference to the inner KernelType
    pub(crate) fn inner(&self) -> &KernelType {
        unsafe { &*(self._private as *const KernelType) }
    }

    pub(crate) fn new_logistic(lambda: f64) -> Result<Self, sparse_ir::Error> {
        let inner = KernelType::Logistic(Arc::new(LogisticKernel::new(lambda)?));
        Ok(Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        })
    }

    // spir_reg_bose_kernel_new stays available until the kernel is removed (#273).
    #[allow(deprecated)]
    pub(crate) fn new_regularized_bose(lambda: f64) -> Result<Self, sparse_ir::Error> {
        let inner = KernelType::RegularizedBose(Arc::new(RegularizedBoseKernel::new(lambda)?));
        Ok(Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        })
    }

    pub(crate) fn lambda(&self) -> f64 {
        match self.inner() {
            KernelType::Logistic(k) => k.lambda(),
            KernelType::RegularizedBose(k) => k.lambda(),
        }
    }

    pub(crate) fn compute(&self, x: f64, y: f64) -> f64 {
        match self.inner() {
            KernelType::Logistic(k) => k.compute(x, y),
            KernelType::RegularizedBose(k) => k.compute(x, y),
        }
    }

    /// Get the inner kernel for SVE computation
    pub(crate) fn as_logistic(&self) -> Option<&Arc<LogisticKernel>> {
        match self.inner() {
            KernelType::Logistic(k) => Some(k),
            _ => None,
        }
    }

    pub(crate) fn as_regularized_bose(&self) -> Option<&Arc<RegularizedBoseKernel>> {
        match self.inner() {
            KernelType::RegularizedBose(k) => Some(k),
            _ => None,
        }
    }

    /// Get kernel domain boundaries (xmin, xmax, ymin, ymax)
    pub(crate) fn domain(&self) -> (f64, f64, f64, f64) {
        // Both kernel types have domain [-1, 1] × [-1, 1]
        (-1.0, 1.0, -1.0, 1.0)
    }
}

impl Clone for spir_kernel {
    fn clone(&self) -> Self {
        // Cheap clone: KernelType::clone internally uses Arc::clone which is cheap
        let inner = self.inner().clone();
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }
}

impl Drop for spir_kernel {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *const KernelType as *mut KernelType);
            }
        }
    }
}

impl spir_sve_result {
    /// Get a reference to the inner Arc<SVEResult>
    fn inner_arc(&self) -> &Arc<SVEResult> {
        unsafe { &*(self._private as *const Arc<SVEResult>) }
    }

    pub(crate) fn new(sve_result: SVEResult) -> Self {
        let inner = Arc::new(sve_result);
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.inner_arc().s().len()
    }

    pub(crate) fn svals(&self) -> &[f64] {
        self.inner_arc().s()
    }

    #[allow(dead_code)]
    pub(crate) fn epsilon(&self) -> f64 {
        self.inner_arc().epsilon()
    }

    /// Get inner SVEResult for basis construction
    pub(crate) fn inner(&self) -> &Arc<SVEResult> {
        self.inner_arc()
    }
}

impl Clone for spir_sve_result {
    fn clone(&self) -> Self {
        // Cheap clone: Arc::clone just increments reference count
        let inner = self.inner_arc().clone();
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }
}

impl Drop for spir_sve_result {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ =
                    Box::from_raw(self._private as *const Arc<SVEResult> as *mut Arc<SVEResult>);
            }
        }
    }
}

impl spir_basis {
    /// Get a reference to the inner BasisType (for internal use by other modules)
    pub(crate) fn inner(&self) -> &BasisType {
        unsafe { &*(self._private as *const BasisType) }
    }

    fn inner_type(&self) -> &BasisType {
        self.inner()
    }

    pub(crate) fn new_logistic_fermionic(
        basis: FiniteTempBasis<LogisticKernel, Fermionic>,
    ) -> Self {
        let inner = BasisType::LogisticFermionic(Arc::new(basis));
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }

    pub(crate) fn new_logistic_bosonic(basis: FiniteTempBasis<LogisticKernel, Bosonic>) -> Self {
        let inner = BasisType::LogisticBosonic(Arc::new(basis));
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }

    /// Kept for defensive completeness: no C ABI constructor creates a
    /// fermionic RegularizedBose basis (rejected at the boundary, see #241).
    #[allow(dead_code)]
    pub(crate) fn new_regularized_bose_fermionic(
        basis: FiniteTempBasis<RegularizedBoseKernel, Fermionic>,
    ) -> Self {
        let inner = BasisType::RegularizedBoseFermionic(Arc::new(basis));
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }

    pub(crate) fn new_regularized_bose_bosonic(
        basis: FiniteTempBasis<RegularizedBoseKernel, Bosonic>,
    ) -> Self {
        let inner = BasisType::RegularizedBoseBosonic(Arc::new(basis));
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }

    pub(crate) fn new_dlr_fermionic(
        dlr: Arc<sparse_ir::dlr::DiscreteLehmannRepresentation<Fermionic>>,
    ) -> Self {
        let inner = BasisType::DLRFermionic(dlr);
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }

    pub(crate) fn new_dlr_bosonic(
        dlr: Arc<sparse_ir::dlr::DiscreteLehmannRepresentation<Bosonic>>,
    ) -> Self {
        let inner = BasisType::DLRBosonic(dlr);
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }

    pub(crate) fn size(&self) -> usize {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b.size(),
            BasisType::LogisticBosonic(b) => b.size(),
            BasisType::RegularizedBoseFermionic(b) => b.size(),
            BasisType::RegularizedBoseBosonic(b) => b.size(),
            BasisType::DLRFermionic(dlr) => dlr.poles().len(),
            BasisType::DLRBosonic(dlr) => dlr.poles().len(),
        }
    }

    pub(crate) fn svals(&self) -> Vec<f64> {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b.s().to_vec(),
            BasisType::LogisticBosonic(b) => b.s().to_vec(),
            BasisType::RegularizedBoseFermionic(b) => b.s().to_vec(),
            BasisType::RegularizedBoseBosonic(b) => b.s().to_vec(),
            // DLR: no singular values, return empty
            BasisType::DLRFermionic(_) | BasisType::DLRBosonic(_) => vec![],
        }
    }

    pub(crate) fn statistics(&self) -> i32 {
        // 0 = Bosonic, 1 = Fermionic (matching libsparseir)
        match self.inner_type() {
            BasisType::LogisticFermionic(_) => 1,
            BasisType::LogisticBosonic(_) => 0,
            BasisType::RegularizedBoseFermionic(_) => 1,
            BasisType::RegularizedBoseBosonic(_) => 0,
            BasisType::DLRFermionic(_) => 1,
            BasisType::DLRBosonic(_) => 0,
        }
    }

    pub(crate) fn beta(&self) -> f64 {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b.beta(),
            BasisType::LogisticBosonic(b) => b.beta(),
            BasisType::RegularizedBoseFermionic(b) => b.beta(),
            BasisType::RegularizedBoseBosonic(b) => b.beta(),
            BasisType::DLRFermionic(dlr) => dlr.beta(),
            BasisType::DLRBosonic(dlr) => dlr.beta(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn wmax(&self) -> f64 {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b.wmax(),
            BasisType::LogisticBosonic(b) => b.wmax(),
            BasisType::RegularizedBoseFermionic(b) => b.wmax(),
            BasisType::RegularizedBoseBosonic(b) => b.wmax(),
            BasisType::DLRFermionic(dlr) => dlr.wmax(),
            BasisType::DLRBosonic(dlr) => dlr.wmax(),
        }
    }

    pub(crate) fn default_tau_sampling_points(&self) -> Result<Vec<f64>, sparse_ir::Error> {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b.default_tau_sampling_points(),
            BasisType::LogisticBosonic(b) => b.default_tau_sampling_points(),
            BasisType::RegularizedBoseFermionic(b) => b.default_tau_sampling_points(),
            BasisType::RegularizedBoseBosonic(b) => b.default_tau_sampling_points(),
            // DLR: the interpolation nodes, one per pole
            BasisType::DLRFermionic(dlr) => Ok(dlr.tau_nodes().to_vec()),
            BasisType::DLRBosonic(dlr) => Ok(dlr.tau_nodes().to_vec()),
        }
    }

    pub(crate) fn default_tau_sampling_points_size_requested(
        &self,
        size_requested: usize,
    ) -> Result<Vec<f64>, sparse_ir::Error> {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => {
                b.default_tau_sampling_points_size_requested(size_requested)
            }
            BasisType::LogisticBosonic(b) => {
                b.default_tau_sampling_points_size_requested(size_requested)
            }
            BasisType::RegularizedBoseFermionic(b) => {
                b.default_tau_sampling_points_size_requested(size_requested)
            }
            BasisType::RegularizedBoseBosonic(b) => {
                b.default_tau_sampling_points_size_requested(size_requested)
            }
            // A DLR has no default points; the C API reports none (success)
            BasisType::DLRFermionic(_) | BasisType::DLRBosonic(_) => Ok(vec![]),
        }
    }

    pub(crate) fn default_matsubara_sampling_points(
        &self,
        positive_only: bool,
    ) -> Result<Vec<i64>, sparse_ir::Error> {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => {
                b.default_matsubara_sampling_points_i64(positive_only)
            }
            BasisType::LogisticBosonic(b) => b.default_matsubara_sampling_points_i64(positive_only),
            BasisType::RegularizedBoseFermionic(b) => {
                b.default_matsubara_sampling_points_i64(positive_only)
            }
            BasisType::RegularizedBoseBosonic(b) => {
                b.default_matsubara_sampling_points_i64(positive_only)
            }
            // DLR: the interpolation nodes, one per pole
            BasisType::DLRFermionic(dlr) => Ok(dlr.matsubara_nodes(positive_only).to_vec()),
            BasisType::DLRBosonic(dlr) => Ok(dlr.matsubara_nodes(positive_only).to_vec()),
        }
    }

    pub(crate) fn default_matsubara_sampling_points_with_mitigate(
        &self,
        positive_only: bool,
        mitigate: bool,
        n_points: usize,
    ) -> Result<Vec<i64>, sparse_ir::Error> {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b
                .default_matsubara_sampling_points_i64_with_mitigate(
                    positive_only,
                    mitigate,
                    n_points,
                ),
            BasisType::LogisticBosonic(b) => b.default_matsubara_sampling_points_i64_with_mitigate(
                positive_only,
                mitigate,
                n_points,
            ),
            BasisType::RegularizedBoseFermionic(b) => b
                .default_matsubara_sampling_points_i64_with_mitigate(
                    positive_only,
                    mitigate,
                    n_points,
                ),
            BasisType::RegularizedBoseBosonic(b) => b
                .default_matsubara_sampling_points_i64_with_mitigate(
                    positive_only,
                    mitigate,
                    n_points,
                ),
            // DLR: no default Matsubara sampling points
            BasisType::DLRFermionic(_) | BasisType::DLRBosonic(_) => Ok(vec![]),
        }
    }

    pub(crate) fn default_omega_sampling_points(&self) -> Result<Vec<f64>, sparse_ir::Error> {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b.default_omega_sampling_points(),
            BasisType::LogisticBosonic(b) => b.default_omega_sampling_points(),
            BasisType::RegularizedBoseFermionic(b) => b.default_omega_sampling_points(),
            BasisType::RegularizedBoseBosonic(b) => b.default_omega_sampling_points(),
            // DLR: return poles as omega sampling points
            BasisType::DLRFermionic(dlr) => Ok(dlr.poles().to_vec()),
            BasisType::DLRBosonic(dlr) => Ok(dlr.poles().to_vec()),
        }
    }
}

impl Clone for spir_basis {
    fn clone(&self) -> Self {
        // Cheap clone: BasisType::clone internally uses Arc::clone which is cheap
        let inner = self.inner_type().clone();
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }
}

impl Drop for spir_basis {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *const BasisType as *mut BasisType);
            }
        }
    }
}

// ============================================================================
// Wrapper types for different function representations
// ============================================================================

/// Wrapper for PiecewiseLegendrePolyVector with domain information
#[derive(Clone)]
pub(crate) struct PolyVectorFuncs {
    pub poly: Arc<PiecewiseLegendrePolyVector>,
    pub domain: FunctionDomain,
}

impl PolyVectorFuncs {
    /// Evaluate all functions at a single point
    ///
    /// # Errors
    /// [`sparse_ir::Error::OutOfDomain`] if `x` is NaN or outside the
    /// domain: [-β, β] for τ functions, the knots for ω functions
    pub fn evaluate_at(&self, x: f64, beta: f64) -> Result<Vec<f64>, sparse_ir::Error> {
        let (x_reg, sign) = self.normalize(x, beta)?;
        self.poly
            .get_polys()
            .iter()
            .map(|p| Ok(sign * p.try_evaluate(x_reg)?))
            .collect()
    }

    /// Batch evaluate all functions at multiple points
    /// Returns Vec<Vec<f64>> where result[i][j] is function i evaluated at point j
    ///
    /// # Errors
    /// As [`Self::evaluate_at`], for the first point outside the domain;
    /// nothing is evaluated then
    pub fn batch_evaluate_at(
        &self,
        xs: &[f64],
        beta: f64,
    ) -> Result<Vec<Vec<f64>>, sparse_ir::Error> {
        let normalized = xs
            .iter()
            .map(|&x| self.normalize(x, beta))
            .collect::<Result<Vec<(f64, f64)>, sparse_ir::Error>>()?;
        let xs_reg: Vec<f64> = normalized.iter().map(|&(x, _)| x).collect();
        self.poly
            .get_polys()
            .iter()
            .map(|p| {
                let values = p.try_evaluate_many(&xs_reg)?;
                Ok(values
                    .iter()
                    .zip(&normalized)
                    .map(|(&value, &(_, sign))| sign * value)
                    .collect())
            })
            .collect()
    }

    /// The point `x` in the domain of the polynomials, and the sign of the
    /// (anti)periodic continuation for a τ function
    fn normalize(&self, x: f64, beta: f64) -> Result<(f64, f64), sparse_ir::Error> {
        match self.domain {
            FunctionDomain::Tau(Statistics::Fermionic) => normalize_tau::<Fermionic>(x, beta),
            FunctionDomain::Tau(Statistics::Bosonic) => normalize_tau::<Bosonic>(x, beta),
            FunctionDomain::Omega => Ok((x, 1.0)),
        }
    }
}

/// Wrapper for Fourier-transformed functions (PiecewiseLegendreFTVector)
#[derive(Clone)]
pub(crate) struct FTVectorFuncs {
    pub ft_fermionic: Option<Arc<PiecewiseLegendreFTVector<Fermionic>>>,
    pub ft_bosonic: Option<Arc<PiecewiseLegendreFTVector<Bosonic>>>,
    pub statistics: Statistics,
}

/// The DLR that DLR functions are taken from
#[derive(Clone)]
pub(crate) enum DlrOf {
    Fermionic(Arc<DiscreteLehmannRepresentation<Fermionic>>),
    Bosonic(Arc<DiscreteLehmannRepresentation<Bosonic>>),
}

impl DlrOf {
    pub(crate) fn beta(&self) -> f64 {
        match self {
            Self::Fermionic(dlr) => dlr.beta(),
            Self::Bosonic(dlr) => dlr.beta(),
        }
    }

    fn n_poles(&self) -> usize {
        match self {
            Self::Fermionic(dlr) => dlr.poles().len(),
            Self::Bosonic(dlr) => dlr.poles().len(),
        }
    }

    /// `values[i][j]`: the τ function of pole `poles[i]` at `taus[j]`
    ///
    /// # Errors
    /// [`sparse_ir::Error::OutOfDomain`] if a τ is NaN or outside [-β, β]
    fn tau_values(&self, poles: &[usize], taus: &[f64]) -> Result<Vec<Vec<f64>>, sparse_ir::Error> {
        let values = match self {
            Self::Fermionic(dlr) => dlr.evaluate_tau(taus)?,
            Self::Bosonic(dlr) => dlr.evaluate_tau(taus)?,
        };
        Ok(columns(&values, poles, taus.len()))
    }

    /// `values[i][j]`: the Matsubara function of pole `poles[i]` at `ns[j]`
    ///
    /// # Errors
    /// [`sparse_ir::Error::InvalidMatsubaraIndex`] if an index has the wrong
    /// parity for the statistics
    fn matsubara_values(
        &self,
        poles: &[usize],
        ns: &[i64],
    ) -> Result<Vec<Vec<Complex<f64>>>, sparse_ir::Error> {
        fn values<S: StatisticsType + 'static>(
            dlr: &DiscreteLehmannRepresentation<S>,
            poles: &[usize],
            ns: &[i64],
        ) -> Result<Vec<Vec<Complex<f64>>>, sparse_ir::Error> {
            let freqs = ns
                .iter()
                .map(|&n| MatsubaraFreq::<S>::new(n))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(columns(&dlr.evaluate_matsubara(&freqs)?, poles, ns.len()))
        }
        match self {
            Self::Fermionic(dlr) => values(dlr.as_ref(), poles, ns),
            Self::Bosonic(dlr) => values(dlr.as_ref(), poles, ns),
        }
    }
}

/// The columns `poles` of an `n_points × n_poles` matrix, as rows
fn columns<T: sparse_ir::TensorScalar + Copy>(
    matrix: &sparse_ir::Matrix<T>,
    poles: &[usize],
    n_points: usize,
) -> Vec<Vec<T>> {
    // Owned tensors are compact column-major: column i starts at i * n_points.
    let data = matrix
        .host_data()
        .expect("an owned matrix is host-resident");
    poles
        .iter()
        .map(|&i| data[i * n_points..(i + 1) * n_points].to_vec())
        .collect()
}

/// DLR functions in the τ domain: those of the poles `indices` of `dlr`
#[derive(Clone)]
pub(crate) struct DLRTauFuncs {
    pub dlr: DlrOf,
    pub indices: Vec<usize>,
}

/// DLR functions in the Matsubara domain: those of the poles `indices` of `dlr`
#[derive(Clone)]
pub(crate) struct DLRMatsubaraFuncs {
    pub dlr: DlrOf,
    pub indices: Vec<usize>,
}

// ============================================================================
// Internal enum to hold different function types
// ============================================================================

/// Internal enum to hold different function types
#[derive(Clone)]
pub(crate) enum FuncsType {
    /// Continuous functions (u or v): PiecewiseLegendrePolyVector
    PolyVector(PolyVectorFuncs),

    /// Fourier-transformed functions (uhat): PiecewiseLegendreFTVector
    FTVector(FTVectorFuncs),

    /// DLR functions in tau domain (discrete poles)
    DLRTau(DLRTauFuncs),

    /// DLR functions in Matsubara domain (discrete poles)
    DLRMatsubara(DLRMatsubaraFuncs),
}

/// Opaque funcs type for C API (compatible with libsparseir)
///
/// Wraps piecewise Legendre polynomial representations:
/// - PiecewiseLegendrePolyVector for u and v
/// - PiecewiseLegendreFTVector for uhat
///
/// Note: Named `spir_funcs` to match libsparseir C++ API exactly.
/// The internal FuncsType is hidden using a void pointer, but beta is kept as a public field.
#[repr(C)]
pub struct spir_funcs {
    pub(crate) _private: *const std::ffi::c_void,
    pub(crate) beta: f64,
}

impl spir_funcs {
    /// Get a reference to the inner FuncsType
    pub(crate) fn inner_type(&self) -> &FuncsType {
        unsafe { &*(self._private as *const FuncsType) }
    }

    /// Create u funcs (tau-domain, Fermionic)
    pub(crate) fn from_u_fermionic(poly: Arc<PiecewiseLegendrePolyVector>, beta: f64) -> Self {
        let inner = FuncsType::PolyVector(PolyVectorFuncs {
            poly,
            domain: FunctionDomain::Tau(Statistics::Fermionic),
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
            beta,
        }
    }

    /// Create u funcs (tau-domain, Bosonic)
    pub(crate) fn from_u_bosonic(poly: Arc<PiecewiseLegendrePolyVector>, beta: f64) -> Self {
        let inner = FuncsType::PolyVector(PolyVectorFuncs {
            poly,
            domain: FunctionDomain::Tau(Statistics::Bosonic),
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
            beta,
        }
    }

    /// Create v funcs (omega-domain, no statistics)
    pub(crate) fn from_v(poly: Arc<PiecewiseLegendrePolyVector>, beta: f64) -> Self {
        let inner = FuncsType::PolyVector(PolyVectorFuncs {
            poly,
            domain: FunctionDomain::Omega,
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
            beta,
        }
    }

    /// Create uhat funcs (Matsubara-domain, Fermionic, truncated)
    pub(crate) fn from_uhat_fermionic(
        ft: Arc<PiecewiseLegendreFTVector<Fermionic>>,
        beta: f64,
    ) -> Self {
        let inner = FuncsType::FTVector(FTVectorFuncs {
            ft_fermionic: Some(ft),
            ft_bosonic: None,
            statistics: Statistics::Fermionic,
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
            beta,
        }
    }

    /// Create uhat funcs (Matsubara-domain, Bosonic, truncated)
    pub(crate) fn from_uhat_bosonic(
        ft: Arc<PiecewiseLegendreFTVector<Bosonic>>,
        beta: f64,
    ) -> Self {
        let inner = FuncsType::FTVector(FTVectorFuncs {
            ft_fermionic: None,
            ft_bosonic: Some(ft),
            statistics: Statistics::Bosonic,
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
            beta,
        }
    }

    /// Create uhat_full funcs (Matsubara-domain, Fermionic, untruncated)
    ///
    /// Creates funcs from the full (untruncated) basis functions `uhat_full`.
    /// This accesses `basis.uhat_full` which contains all basis functions
    /// from the SVE result, not just the truncated ones.
    pub(crate) fn from_uhat_full_fermionic(
        ft: Arc<PiecewiseLegendreFTVector<Fermionic>>,
        beta: f64,
    ) -> Self {
        let inner = FuncsType::FTVector(FTVectorFuncs {
            ft_fermionic: Some(ft),
            ft_bosonic: None,
            statistics: Statistics::Fermionic,
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
            beta,
        }
    }

    /// Create uhat_full funcs (Matsubara-domain, Bosonic, untruncated)
    ///
    /// Creates funcs from the full (untruncated) basis functions `uhat_full`.
    /// This accesses `basis.uhat_full` which contains all basis functions
    /// from the SVE result, not just the truncated ones.
    pub(crate) fn from_uhat_full_bosonic(
        ft: Arc<PiecewiseLegendreFTVector<Bosonic>>,
        beta: f64,
    ) -> Self {
        let inner = FuncsType::FTVector(FTVectorFuncs {
            ft_fermionic: None,
            ft_bosonic: Some(ft),
            statistics: Statistics::Bosonic,
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
            beta,
        }
    }

    /// Create the DLR τ functions of every pole of `dlr`
    pub(crate) fn from_dlr_tau(dlr: DlrOf) -> Self {
        let beta = dlr.beta();
        let indices = (0..dlr.n_poles()).collect();
        let inner = FuncsType::DLRTau(DLRTauFuncs { dlr, indices });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
            beta,
        }
    }

    /// Create the DLR Matsubara functions of every pole of `dlr`
    pub(crate) fn from_dlr_matsubara(dlr: DlrOf) -> Self {
        let beta = dlr.beta();
        let indices = (0..dlr.n_poles()).collect();
        let inner = FuncsType::DLRMatsubara(DLRMatsubaraFuncs { dlr, indices });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
            beta,
        }
    }

    /// Get the number of basis functions
    pub(crate) fn size(&self) -> usize {
        match self.inner_type() {
            FuncsType::PolyVector(pv) => pv.poly.get_polys().len(),
            FuncsType::FTVector(ftv) => {
                if let Some(ft) = &ftv.ft_fermionic {
                    ft.get_polys().len()
                } else if let Some(ft) = &ftv.ft_bosonic {
                    ft.get_polys().len()
                } else {
                    0
                }
            }
            FuncsType::DLRTau(dlr) => dlr.indices.len(),
            FuncsType::DLRMatsubara(dlr) => dlr.indices.len(),
        }
    }

    /// Get knots for continuous functions (PolyVector only)
    pub(crate) fn knots(&self) -> Option<Vec<f64>> {
        match self.inner_type() {
            FuncsType::PolyVector(pv) => {
                // Get unique knots from all polynomials
                let mut all_knots = Vec::new();
                for p in pv.poly.get_polys() {
                    for &knot in p.get_knots() {
                        if !all_knots.iter().any(|&k: &f64| (k - knot).abs() < 1e-14) {
                            all_knots.push(knot);
                        }
                    }
                }
                all_knots.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Some(all_knots)
            }
            _ => None, // FT vectors don't have knots in the traditional sense
        }
    }

    /// Evaluate at a single tau/omega point (for continuous functions only)
    ///
    /// # Arguments
    /// * `x` - A point in the domain of the functions: for u,
    ///   tau ∈ [-beta, beta]; for v, omega ∈ [-omega_max, omega_max]
    ///
    /// # Returns
    /// `None` if the functions are not of this kind; `Some(Err(..))` for a
    /// point outside the domain
    pub(crate) fn eval_continuous(&self, x: f64) -> Option<Result<Vec<f64>, sparse_ir::Error>> {
        match self.inner_type() {
            FuncsType::PolyVector(pv) => Some(pv.evaluate_at(x, self.beta)),
            FuncsType::DLRTau(f) => Some(
                f.dlr
                    .tau_values(&f.indices, &[x])
                    .map(|rows| rows.into_iter().map(|row| row[0]).collect()),
            ),
            _ => None,
        }
    }

    /// Evaluate at a single Matsubara frequency (for FT functions only)
    ///
    /// # Arguments
    /// * `n` - Matsubara frequency index
    ///
    /// # Returns
    /// `None` if the functions are not of this kind; `Some(Err(..))` for an
    /// index of the wrong parity
    pub(crate) fn eval_matsubara(
        &self,
        n: i64,
    ) -> Option<Result<Vec<num_complex::Complex64>, sparse_ir::Error>> {
        match self.inner_type() {
            FuncsType::FTVector(ftv) => {
                if ftv.statistics == Statistics::Fermionic {
                    // Fermionic
                    let ft = ftv.ft_fermionic.as_ref()?;
                    let freq = match MatsubaraFreq::<Fermionic>::new(n) {
                        Ok(freq) => freq,
                        Err(e) => return Some(Err(e)),
                    };
                    let mut result = Vec::with_capacity(ft.get_polys().len());
                    for p in ft.get_polys() {
                        result.push(p.evaluate(&freq));
                    }
                    Some(Ok(result))
                } else {
                    // Bosonic
                    let ft = ftv.ft_bosonic.as_ref()?;
                    let freq = match MatsubaraFreq::<Bosonic>::new(n) {
                        Ok(freq) => freq,
                        Err(e) => return Some(Err(e)),
                    };
                    let mut result = Vec::with_capacity(ft.get_polys().len());
                    for p in ft.get_polys() {
                        result.push(p.evaluate(&freq));
                    }
                    Some(Ok(result))
                }
            }
            FuncsType::DLRMatsubara(f) => Some(
                f.dlr
                    .matsubara_values(&f.indices, &[n])
                    .map(|rows| rows.into_iter().map(|row| row[0]).collect()),
            ),
            _ => None,
        }
    }

    /// Batch evaluate at multiple tau/omega points
    ///
    /// # Returns
    /// `None` if the functions are not of this kind; `Some(Err(..))` for a
    /// point outside the domain
    pub(crate) fn batch_eval_continuous(
        &self,
        xs: &[f64],
    ) -> Option<Result<Vec<Vec<f64>>, sparse_ir::Error>> {
        match self.inner_type() {
            FuncsType::PolyVector(pv) => Some(pv.batch_evaluate_at(xs, self.beta)),
            FuncsType::DLRTau(f) => Some(f.dlr.tau_values(&f.indices, xs)),
            _ => None,
        }
    }

    /// Batch evaluate at multiple Matsubara frequencies (for FT functions only)
    ///
    /// # Arguments
    /// * `ns` - Matsubara frequency indices
    ///
    /// # Returns
    /// `None` if the functions are not of this kind; `Some(Err(..))` for an
    /// index of the wrong parity
    pub(crate) fn batch_eval_matsubara(
        &self,
        ns: &[i64],
    ) -> Option<Result<Vec<Vec<num_complex::Complex64>>, sparse_ir::Error>> {
        match self.inner_type() {
            FuncsType::FTVector(ftv) => {
                if ftv.statistics == Statistics::Fermionic {
                    // Fermionic
                    let ft = ftv.ft_fermionic.as_ref()?;
                    let n_funcs = ft.get_polys().len();
                    let n_points = ns.len();
                    let mut result =
                        vec![vec![num_complex::Complex64::new(0.0, 0.0); n_points]; n_funcs];

                    for (j, &n) in ns.iter().enumerate() {
                        let freq = match MatsubaraFreq::<Fermionic>::new(n) {
                            Ok(freq) => freq,
                            Err(e) => return Some(Err(e)),
                        };
                        for (i, p) in ft.get_polys().iter().enumerate() {
                            result[i][j] = p.evaluate(&freq);
                        }
                    }
                    Some(Ok(result))
                } else {
                    // Bosonic
                    let ft = ftv.ft_bosonic.as_ref()?;
                    let n_funcs = ft.get_polys().len();
                    let n_points = ns.len();
                    let mut result =
                        vec![vec![num_complex::Complex64::new(0.0, 0.0); n_points]; n_funcs];

                    for (j, &n) in ns.iter().enumerate() {
                        let freq = match MatsubaraFreq::<Bosonic>::new(n) {
                            Ok(freq) => freq,
                            Err(e) => return Some(Err(e)),
                        };
                        for (i, p) in ft.get_polys().iter().enumerate() {
                            result[i][j] = p.evaluate(&freq);
                        }
                    }
                    Some(Ok(result))
                }
            }
            FuncsType::DLRMatsubara(f) => Some(f.dlr.matsubara_values(&f.indices, ns)),
            FuncsType::DLRTau(_) => {
                // DLRTau is for tau, not Matsubara frequencies
                None
            }
            _ => None,
        }
    }

    /// Extract a slice of functions by indices (creates a new subset)
    ///
    /// # Arguments
    /// * `indices` - Indices of functions to extract
    ///
    /// # Returns
    /// New funcs object with the selected subset, or None if operation not supported
    pub(crate) fn get_slice(&self, indices: &[usize]) -> Option<Self> {
        match self.inner_type() {
            FuncsType::PolyVector(pv) => {
                let mut new_polys = Vec::with_capacity(indices.len());
                for &idx in indices {
                    if idx >= pv.poly.get_polys().len() {
                        return None;
                    }
                    new_polys.push(pv.poly.get_polys()[idx].clone());
                }
                // The C API rejects an empty selection and invalid indices
                // before this call; a failure here is an internal
                // inconsistency (SPIR_INTERNAL_ERROR, as before).
                let new_poly_vec = PiecewiseLegendrePolyVector::new(new_polys).ok()?;
                Some(Self {
                    _private: Box::into_raw(Box::new(FuncsType::PolyVector(PolyVectorFuncs {
                        poly: Arc::new(new_poly_vec),
                        domain: pv.domain,
                    }))) as *mut std::ffi::c_void,
                    beta: self.beta,
                })
            }
            FuncsType::FTVector(ftv) => {
                // Extract slice from PiecewiseLegendreFTVector
                if ftv.statistics == Statistics::Fermionic {
                    let ft = ftv.ft_fermionic.as_ref()?;
                    let mut new_polyvec = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        if idx >= ft.get_polys().len() {
                            return None;
                        }
                        new_polyvec.push(ft.get_polys()[idx].clone());
                    }
                    let new_ft_vector =
                        Arc::new(PiecewiseLegendreFTVector::from_vector(new_polyvec));
                    Some(Self {
                        _private: Box::into_raw(Box::new(FuncsType::FTVector(FTVectorFuncs {
                            ft_fermionic: Some(new_ft_vector),
                            ft_bosonic: None,
                            statistics: ftv.statistics,
                        }))) as *mut std::ffi::c_void,
                        beta: self.beta,
                    })
                } else {
                    let ft = ftv.ft_bosonic.as_ref()?;
                    let mut new_polyvec = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        if idx >= ft.get_polys().len() {
                            return None;
                        }
                        new_polyvec.push(ft.get_polys()[idx].clone());
                    }
                    let new_ft_vector =
                        Arc::new(PiecewiseLegendreFTVector::from_vector(new_polyvec));
                    Some(Self {
                        _private: Box::into_raw(Box::new(FuncsType::FTVector(FTVectorFuncs {
                            ft_fermionic: None,
                            ft_bosonic: Some(new_ft_vector),
                            statistics: ftv.statistics,
                        }))) as *mut std::ffi::c_void,
                        beta: self.beta,
                    })
                }
            }
            FuncsType::DLRTau(f) => {
                // Select a subset of the poles the functions are taken from
                let mut new_indices = Vec::with_capacity(indices.len());
                for &idx in indices {
                    if idx >= f.indices.len() {
                        return None;
                    }
                    new_indices.push(f.indices[idx]);
                }
                Some(Self {
                    _private: Box::into_raw(Box::new(FuncsType::DLRTau(DLRTauFuncs {
                        dlr: f.dlr.clone(),
                        indices: new_indices,
                    }))) as *mut std::ffi::c_void,
                    beta: f.dlr.beta(),
                })
            }
            FuncsType::DLRMatsubara(f) => {
                // Select a subset of the poles the functions are taken from
                let mut new_indices = Vec::with_capacity(indices.len());
                for &idx in indices {
                    if idx >= f.indices.len() {
                        return None;
                    }
                    new_indices.push(f.indices[idx]);
                }
                Some(Self {
                    _private: Box::into_raw(Box::new(FuncsType::DLRMatsubara(DLRMatsubaraFuncs {
                        dlr: f.dlr.clone(),
                        indices: new_indices,
                    }))) as *mut std::ffi::c_void,
                    beta: f.dlr.beta(),
                })
            }
        }
    }
}

impl Clone for spir_funcs {
    fn clone(&self) -> Self {
        // Cheap clone: FuncsType::clone internally uses Arc::clone which is cheap
        let inner = self.inner_type().clone();
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
            beta: self.beta,
        }
    }
}

impl Drop for spir_funcs {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *const FuncsType as *mut FuncsType);
            }
        }
    }
}

// Helper function for tau normalization is now provided by sparse_ir::taufuncs::normalize_tau

#[cfg(test)]
mod tests {

    #[test]
    fn test_funcs_creation() {
        // Basic test that funcs types can be created
        // More comprehensive tests should be in integration tests
    }
}
/// Sampling type for C API (unified type for all domains)
///
/// This wraps different sampling implementations:
/// - TauSampling (for tau-domain)
/// - MatsubaraSampling (for Matsubara frequencies, full range or positive-only)
/// The internal structure is hidden using a void pointer to prevent exposing SamplingType to C.
#[repr(C)]
pub struct spir_sampling {
    pub(crate) _private: *const std::ffi::c_void,
}

impl spir_sampling {
    /// Get a reference to the inner SamplingType (for internal use by other modules)
    pub(crate) fn inner(&self) -> &SamplingType {
        unsafe { &*(self._private as *const SamplingType) }
    }
}

impl Clone for spir_sampling {
    fn clone(&self) -> Self {
        // Cheap clone: SamplingType::clone internally uses Arc::clone which is cheap
        let inner = self.inner().clone();
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }
}

impl Drop for spir_sampling {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *const SamplingType as *mut SamplingType);
            }
        }
    }
}

/// Internal enum to distinguish between different sampling types
#[derive(Clone)]
pub(crate) enum SamplingType {
    TauFermionic(Arc<sparse_ir::sampling::TauSampling<Fermionic>>),
    TauBosonic(Arc<sparse_ir::sampling::TauSampling<Bosonic>>),
    MatsubaraFermionic(Arc<sparse_ir::matsubara_sampling::MatsubaraSampling<Fermionic>>),
    MatsubaraBosonic(Arc<sparse_ir::matsubara_sampling::MatsubaraSampling<Bosonic>>),
    MatsubaraPositiveOnlyFermionic(
        Arc<sparse_ir::matsubara_sampling::MatsubaraSamplingPositiveOnly<Fermionic>>,
    ),
    MatsubaraPositiveOnlyBosonic(
        Arc<sparse_ir::matsubara_sampling::MatsubaraSamplingPositiveOnly<Bosonic>>,
    ),
}

/// InplaceFitter implementation for SamplingType
///
/// Delegates to the underlying sampling type's InplaceFitter implementation.
/// Returns `Error::NotSupported` for the operations that the sampling type does not support.
impl InplaceFitter for SamplingType {
    fn n_points(&self) -> usize {
        match self {
            SamplingType::TauFermionic(s) => s.n_sampling_points(),
            SamplingType::TauBosonic(s) => s.n_sampling_points(),
            SamplingType::MatsubaraFermionic(s) => s.n_sampling_points(),
            SamplingType::MatsubaraBosonic(s) => s.n_sampling_points(),
            SamplingType::MatsubaraPositiveOnlyFermionic(s) => s.n_sampling_points(),
            SamplingType::MatsubaraPositiveOnlyBosonic(s) => s.n_sampling_points(),
        }
    }

    fn basis_size(&self) -> usize {
        match self {
            SamplingType::TauFermionic(s) => s.basis_size(),
            SamplingType::TauBosonic(s) => s.basis_size(),
            SamplingType::MatsubaraFermionic(s) => s.basis_size(),
            SamplingType::MatsubaraBosonic(s) => s.basis_size(),
            SamplingType::MatsubaraPositiveOnlyFermionic(s) => s.basis_size(),
            SamplingType::MatsubaraPositiveOnlyBosonic(s) => s.basis_size(),
        }
    }

    fn evaluate_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<(), sparse_ir::Error> {
        match self {
            SamplingType::TauFermionic(s) => {
                InplaceFitter::evaluate_nd_dd_to(s.as_ref(), backend, coeffs, dim, out)
            }
            SamplingType::TauBosonic(s) => {
                InplaceFitter::evaluate_nd_dd_to(s.as_ref(), backend, coeffs, dim, out)
            }
            // Matsubara doesn't support dd (real → real)
            _ => Err(unsupported("evaluate_nd_dd_to")),
        }
    }

    fn evaluate_nd_dz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<(), sparse_ir::Error> {
        match self {
            SamplingType::MatsubaraFermionic(s) => {
                InplaceFitter::evaluate_nd_dz_to(s.as_ref(), backend, coeffs, dim, out)
            }
            SamplingType::MatsubaraBosonic(s) => {
                InplaceFitter::evaluate_nd_dz_to(s.as_ref(), backend, coeffs, dim, out)
            }
            SamplingType::MatsubaraPositiveOnlyFermionic(s) => {
                InplaceFitter::evaluate_nd_dz_to(s.as_ref(), backend, coeffs, dim, out)
            }
            SamplingType::MatsubaraPositiveOnlyBosonic(s) => {
                InplaceFitter::evaluate_nd_dz_to(s.as_ref(), backend, coeffs, dim, out)
            }
            // Tau doesn't support dz (real → complex)
            _ => Err(unsupported("evaluate_nd_dz_to")),
        }
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<(), sparse_ir::Error> {
        match self {
            SamplingType::TauFermionic(s) => {
                InplaceFitter::evaluate_nd_zz_to(s.as_ref(), backend, coeffs, dim, out)
            }
            SamplingType::TauBosonic(s) => {
                InplaceFitter::evaluate_nd_zz_to(s.as_ref(), backend, coeffs, dim, out)
            }
            SamplingType::MatsubaraFermionic(s) => {
                InplaceFitter::evaluate_nd_zz_to(s.as_ref(), backend, coeffs, dim, out)
            }
            SamplingType::MatsubaraBosonic(s) => {
                InplaceFitter::evaluate_nd_zz_to(s.as_ref(), backend, coeffs, dim, out)
            }
            SamplingType::MatsubaraPositiveOnlyFermionic(s) => {
                InplaceFitter::evaluate_nd_zz_to(s.as_ref(), backend, coeffs, dim, out)
            }
            SamplingType::MatsubaraPositiveOnlyBosonic(s) => {
                InplaceFitter::evaluate_nd_zz_to(s.as_ref(), backend, coeffs, dim, out)
            }
        }
    }

    fn fit_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<(), sparse_ir::Error> {
        match self {
            SamplingType::TauFermionic(s) => {
                InplaceFitter::fit_nd_dd_to(s.as_ref(), backend, values, dim, out)
            }
            SamplingType::TauBosonic(s) => {
                InplaceFitter::fit_nd_dd_to(s.as_ref(), backend, values, dim, out)
            }
            // Matsubara doesn't support dd (real → real)
            _ => Err(unsupported("fit_nd_dd_to")),
        }
    }

    fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<(), sparse_ir::Error> {
        match self {
            SamplingType::MatsubaraFermionic(s) => {
                InplaceFitter::fit_nd_zd_to(s.as_ref(), backend, values, dim, out)
            }
            SamplingType::MatsubaraBosonic(s) => {
                InplaceFitter::fit_nd_zd_to(s.as_ref(), backend, values, dim, out)
            }
            SamplingType::MatsubaraPositiveOnlyFermionic(s) => {
                InplaceFitter::fit_nd_zd_to(s.as_ref(), backend, values, dim, out)
            }
            SamplingType::MatsubaraPositiveOnlyBosonic(s) => {
                InplaceFitter::fit_nd_zd_to(s.as_ref(), backend, values, dim, out)
            }
            // Tau doesn't support zd (complex → real)
            _ => Err(unsupported("fit_nd_zd_to")),
        }
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<(), sparse_ir::Error> {
        match self {
            SamplingType::TauFermionic(s) => {
                InplaceFitter::fit_nd_zz_to(s.as_ref(), backend, values, dim, out)
            }
            SamplingType::TauBosonic(s) => {
                InplaceFitter::fit_nd_zz_to(s.as_ref(), backend, values, dim, out)
            }
            SamplingType::MatsubaraFermionic(s) => {
                InplaceFitter::fit_nd_zz_to(s.as_ref(), backend, values, dim, out)
            }
            SamplingType::MatsubaraBosonic(s) => {
                InplaceFitter::fit_nd_zz_to(s.as_ref(), backend, values, dim, out)
            }
            SamplingType::MatsubaraPositiveOnlyFermionic(s) => {
                InplaceFitter::fit_nd_zz_to(s.as_ref(), backend, values, dim, out)
            }
            SamplingType::MatsubaraPositiveOnlyBosonic(s) => {
                InplaceFitter::fit_nd_zz_to(s.as_ref(), backend, values, dim, out)
            }
        }
    }
}

/// The error of an N-D operation that a sampling type does not support
fn unsupported(operation: &str) -> sparse_ir::Error {
    sparse_ir::Error::NotSupported {
        what: format!("{operation} for this sampling"),
    }
}

#[cfg(test)]
mod sampling_tests {

    #[test]
    fn test_sampling_creation() {
        // Basic test that sampling types can be created
        // More comprehensive tests should be in integration tests
    }
}
// Re-export status codes from lib.rs to avoid duplication
// (StatusCode and constants are defined in lib.rs)
