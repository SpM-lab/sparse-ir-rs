//! Opaque types for C API
//!
//! All Rust objects are wrapped in opaque pointers to hide implementation
//! details from C code.

use sparseir_rust::basis::FiniteTempBasis;
use sparseir_rust::freq::MatsubaraFreq;
use sparseir_rust::kernel::{AbstractKernel, CentrosymmKernel, LogisticKernel, RegularizedBoseKernel};
use sparseir_rust::poly::PiecewiseLegendrePolyVector;
use sparseir_rust::polyfourier::PiecewiseLegendreFTVector;
use sparseir_rust::sve::SVEResult;
use sparseir_rust::taufuncs::normalize_tau;
use sparseir_rust::traits::Statistics;
use sparseir_rust::{Bosonic, Fermionic};
use std::sync::Arc;

/// Convert Statistics enum to C-API integer
#[inline]
#[allow(dead_code)]
pub(crate) fn statistics_to_c(stats: Statistics) -> i32 {
    match stats {
        Statistics::Fermionic => 1,
        Statistics::Bosonic => 0,
    }
}

/// Convert C-API integer to Statistics enum
#[inline]
#[allow(dead_code)]
pub(crate) fn statistics_from_c(value: i32) -> Statistics {
    match value {
        1 => Statistics::Fermionic,
        _ => Statistics::Bosonic, // Default to Bosonic for invalid values
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
    pub(crate) _private: *mut std::ffi::c_void,
}

/// Opaque SVE result type for C API (compatible with libsparseir)
///
/// Contains singular values and singular functions from SVE computation.
///
/// Note: Named `spir_sve_result` to match libsparseir C++ API exactly.
/// The internal structure is hidden using a void pointer to prevent exposing Arc<SVEResult> to C.
#[repr(C)]
pub struct spir_sve_result {
    pub(crate) _private: *mut std::ffi::c_void,
}

/// Opaque basis type for C API (compatible with libsparseir)
///
/// Represents a finite temperature basis (IR or DLR).
///
/// Note: Named `spir_basis` to match libsparseir C++ API exactly.
/// The internal structure is hidden using a void pointer to prevent exposing BasisType to C.
#[repr(C)]
pub struct spir_basis {
    pub(crate) _private: *mut std::ffi::c_void,
}

/// Internal basis type (not exposed to C)
#[derive(Clone)]
pub(crate) enum BasisType {
    LogisticFermionic(Arc<FiniteTempBasis<LogisticKernel, Fermionic>>),
    LogisticBosonic(Arc<FiniteTempBasis<LogisticKernel, Bosonic>>),
    RegularizedBoseFermionic(Arc<FiniteTempBasis<RegularizedBoseKernel, Fermionic>>),
    RegularizedBoseBosonic(Arc<FiniteTempBasis<RegularizedBoseKernel, Bosonic>>),
    // DLR (Discrete Lehmann Representation) variants
    // Note: DLR always uses LogisticKernel internally, regardless of input kernel type
    DLRFermionic(Arc<sparseir_rust::dlr::DiscreteLehmannRepresentation<Fermionic>>),
    DLRBosonic(Arc<sparseir_rust::dlr::DiscreteLehmannRepresentation<Bosonic>>),
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
        unsafe {
            &*(self._private as *const KernelType)
        }
    }

    pub(crate) fn new_logistic(lambda: f64) -> Self {
        let inner = KernelType::Logistic(Arc::new(LogisticKernel::new(lambda)));
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }

    pub(crate) fn new_regularized_bose(lambda: f64) -> Self {
        let inner = KernelType::RegularizedBose(Arc::new(RegularizedBoseKernel::new(lambda)));
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
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
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }
}

impl Drop for spir_kernel {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut KernelType);
            }
        }
    }
}

impl spir_sve_result {
    /// Get a reference to the inner Arc<SVEResult>
    fn inner_arc(&self) -> &Arc<SVEResult> {
        unsafe {
            &*(self._private as *const Arc<SVEResult>)
        }
    }

    pub(crate) fn new(sve_result: SVEResult) -> Self {
        let inner = Arc::new(sve_result);
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.inner_arc().s.len()
    }

    pub(crate) fn svals(&self) -> &[f64] {
        &self.inner_arc().s
    }

    #[allow(dead_code)]
    pub(crate) fn epsilon(&self) -> f64 {
        self.inner_arc().epsilon
    }

    #[allow(dead_code)]
    pub(crate) fn truncate(&self, epsilon: f64, max_size: Option<usize>) -> Self {
        let (u_part, s_part, v_part) = self.inner_arc().part(Some(epsilon), max_size);
        let truncated = SVEResult::new(u_part, s_part, v_part, epsilon);
        Self::new(truncated)
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
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }
}

impl Drop for spir_sve_result {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut Arc<SVEResult>);
            }
        }
    }
}

impl spir_basis {
    /// Get a reference to the inner BasisType (for internal use by other modules)
    pub(crate) fn inner(&self) -> &BasisType {
        unsafe {
            &*(self._private as *const BasisType)
        }
    }

    fn inner_type(&self) -> &BasisType {
        self.inner()
    }

    pub(crate) fn new_logistic_fermionic(
        basis: FiniteTempBasis<LogisticKernel, Fermionic>,
    ) -> Self {
        let inner = BasisType::LogisticFermionic(Arc::new(basis));
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }

    pub(crate) fn new_logistic_bosonic(basis: FiniteTempBasis<LogisticKernel, Bosonic>) -> Self {
        let inner = BasisType::LogisticBosonic(Arc::new(basis));
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }

    pub(crate) fn new_regularized_bose_fermionic(
        basis: FiniteTempBasis<RegularizedBoseKernel, Fermionic>,
    ) -> Self {
        let inner = BasisType::RegularizedBoseFermionic(Arc::new(basis));
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }

    pub(crate) fn new_regularized_bose_bosonic(
        basis: FiniteTempBasis<RegularizedBoseKernel, Bosonic>,
    ) -> Self {
        let inner = BasisType::RegularizedBoseBosonic(Arc::new(basis));
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }

    pub(crate) fn new_dlr_fermionic(
        dlr: Arc<sparseir_rust::dlr::DiscreteLehmannRepresentation<Fermionic>>,
    ) -> Self {
        let inner = BasisType::DLRFermionic(dlr);
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }

    pub(crate) fn new_dlr_bosonic(
        dlr: Arc<sparseir_rust::dlr::DiscreteLehmannRepresentation<Bosonic>>,
    ) -> Self {
        let inner = BasisType::DLRBosonic(dlr);
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }

    pub(crate) fn size(&self) -> usize {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b.size(),
            BasisType::LogisticBosonic(b) => b.size(),
            BasisType::RegularizedBoseFermionic(b) => b.size(),
            BasisType::RegularizedBoseBosonic(b) => b.size(),
            BasisType::DLRFermionic(dlr) => dlr.poles.len(),
            BasisType::DLRBosonic(dlr) => dlr.poles.len(),
        }
    }

    pub(crate) fn svals(&self) -> Vec<f64> {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b.s.clone(),
            BasisType::LogisticBosonic(b) => b.s.clone(),
            BasisType::RegularizedBoseFermionic(b) => b.s.clone(),
            BasisType::RegularizedBoseBosonic(b) => b.s.clone(),
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
            BasisType::LogisticFermionic(b) => b.beta,
            BasisType::LogisticBosonic(b) => b.beta,
            BasisType::RegularizedBoseFermionic(b) => b.beta,
            BasisType::RegularizedBoseBosonic(b) => b.beta,
            BasisType::DLRFermionic(dlr) => dlr.beta,
            BasisType::DLRBosonic(dlr) => dlr.beta,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn wmax(&self) -> f64 {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b.wmax(),
            BasisType::LogisticBosonic(b) => b.wmax(),
            BasisType::RegularizedBoseFermionic(b) => b.wmax(),
            BasisType::RegularizedBoseBosonic(b) => b.wmax(),
            BasisType::DLRFermionic(dlr) => dlr.wmax,
            BasisType::DLRBosonic(dlr) => dlr.wmax,
        }
    }

    pub(crate) fn default_tau_sampling_points(&self) -> Vec<f64> {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b.default_tau_sampling_points(),
            BasisType::LogisticBosonic(b) => b.default_tau_sampling_points(),
            BasisType::RegularizedBoseFermionic(b) => b.default_tau_sampling_points(),
            BasisType::RegularizedBoseBosonic(b) => b.default_tau_sampling_points(),
            // DLR: no default tau sampling points
            BasisType::DLRFermionic(_) | BasisType::DLRBosonic(_) => vec![],
        }
    }

    pub(crate) fn default_matsubara_sampling_points(&self, positive_only: bool) -> Vec<i64> {
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
            // DLR: no default Matsubara sampling points
            BasisType::DLRFermionic(_) | BasisType::DLRBosonic(_) => vec![],
        }
    }

    pub(crate) fn default_matsubara_sampling_points_with_mitigate(
        &self,
        positive_only: bool,
        mitigate: bool,
        n_points: usize,
    ) -> Vec<i64> {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => {
                b.default_matsubara_sampling_points_i64_with_mitigate(positive_only, mitigate, n_points)
            }
            BasisType::LogisticBosonic(b) => {
                b.default_matsubara_sampling_points_i64_with_mitigate(positive_only, mitigate, n_points)
            }
            BasisType::RegularizedBoseFermionic(b) => {
                b.default_matsubara_sampling_points_i64_with_mitigate(positive_only, mitigate, n_points)
            }
            BasisType::RegularizedBoseBosonic(b) => {
                b.default_matsubara_sampling_points_i64_with_mitigate(positive_only, mitigate, n_points)
            }
            // DLR: no default Matsubara sampling points
            BasisType::DLRFermionic(_) | BasisType::DLRBosonic(_) => vec![],
        }
    }

    pub(crate) fn default_omega_sampling_points(&self) -> Vec<f64> {
        match self.inner_type() {
            BasisType::LogisticFermionic(b) => b.default_omega_sampling_points(),
            BasisType::LogisticBosonic(b) => b.default_omega_sampling_points(),
            BasisType::RegularizedBoseFermionic(b) => b.default_omega_sampling_points(),
            BasisType::RegularizedBoseBosonic(b) => b.default_omega_sampling_points(),
            // DLR: return poles as omega sampling points
            BasisType::DLRFermionic(dlr) => dlr.poles.clone(),
            BasisType::DLRBosonic(dlr) => dlr.poles.clone(),
        }
    }
}

impl Clone for spir_basis {
    fn clone(&self) -> Self {
        // Cheap clone: BasisType::clone internally uses Arc::clone which is cheap
        let inner = self.inner_type().clone();
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }
}

impl Drop for spir_basis {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut BasisType);
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
    pub fn evaluate_at(&self, x: f64, beta: f64) -> Vec<f64> {
        // Normalize x based on domain
        let (x_reg, sign) = match self.domain {
            FunctionDomain::Tau(Statistics::Fermionic) => {
                // u functions (fermionic): normalize tau to [0, beta]
                normalize_tau::<Fermionic>(x, beta)
            }
            FunctionDomain::Tau(Statistics::Bosonic) => {
                // u functions (bosonic): normalize tau to [0, beta]
                normalize_tau::<Bosonic>(x, beta)
            }
            FunctionDomain::Omega => {
                // v functions: no normalization needed
                (x, 1.0)
            }
        };

        // Evaluate all polynomials at the normalized point
        self.poly
            .polyvec
            .iter()
            .map(|p| sign * p.evaluate(x_reg))
            .collect()
    }

    /// Batch evaluate all functions at multiple points
    /// Returns Vec<Vec<f64>> where result[i][j] is function i evaluated at point j
    pub fn batch_evaluate_at(&self, xs: &[f64], beta: f64) -> Vec<Vec<f64>> {
        let n_funcs = self.poly.polyvec.len();
        let n_points = xs.len();
        let mut result = vec![vec![0.0; n_points]; n_funcs];

        // Normalize all points based on domain
        let normalized: Vec<(f64, f64)> = xs
            .iter()
            .map(|&x| match self.domain {
                FunctionDomain::Tau(Statistics::Fermionic) => normalize_tau::<Fermionic>(x, beta),
                FunctionDomain::Tau(Statistics::Bosonic) => normalize_tau::<Bosonic>(x, beta),
                FunctionDomain::Omega => (x, 1.0),
            })
            .collect();

        // Extract normalized x values and signs
        let xs_reg: Vec<f64> = normalized.iter().map(|(x, _)| *x).collect();
        let signs: Vec<f64> = normalized.iter().map(|(_, s)| *s).collect();

        // Evaluate each polynomial at all regularized points using evaluate_many
        for (i, p) in self.poly.polyvec.iter().enumerate() {
            let values = p.evaluate_many(&xs_reg);
            for (j, &val) in values.iter().enumerate() {
                result[i][j] = signs[j] * val;
            }
        }

        result
    }
}

/// Wrapper for Fourier-transformed functions (PiecewiseLegendreFTVector)
#[derive(Clone)]
pub(crate) struct FTVectorFuncs {
    pub ft_fermionic: Option<Arc<PiecewiseLegendreFTVector<Fermionic>>>,
    pub ft_bosonic: Option<Arc<PiecewiseLegendreFTVector<Bosonic>>>,
    pub statistics: Statistics,
}

/// Wrapper for DLR functions in tau domain
#[derive(Clone)]
pub(crate) struct DLRTauFuncs {
    pub poles: Vec<f64>,
    pub beta: f64,
    pub wmax: f64,
    pub inv_weights: Vec<f64>,
    pub statistics: Statistics,
}

impl DLRTauFuncs {
    /// Evaluate all DLR tau functions at a single point
    pub fn evaluate_at(&self, tau: f64) -> Vec<f64> {
        use sparseir_rust::kernel::LogisticKernel;

        // Normalize tau to [0, beta] using the appropriate statistics
        let (tau_reg, sign) = match self.statistics {
            Statistics::Fermionic => normalize_tau::<Fermionic>(tau, self.beta),
            Statistics::Bosonic => normalize_tau::<Bosonic>(tau, self.beta),
        };

        // Compute kernel parameters
        // DLR always uses LogisticKernel in tau domain.
        // Statistics-dependent regularization factors (e.g. tanh(βω/2) for bosons)
        // are applied only in the Matsubara representation via inv_weights and
        // do NOT enter the tau basis functions themselves.
        let lambda = self.beta * self.wmax;
        let kernel = LogisticKernel::new(lambda);
        let x_kern = 2.0 * tau_reg / self.beta - 1.0; // x_kern ∈ [-1, 1]

        // Evaluate DLR tau basis functions: u_l(τ) = sign * (-K(x, y_l))
        self.poles
            .iter()
            .map(|&pole| {
                let y = pole / self.wmax;
                let k_val = kernel.compute(x_kern, y);
                sign * (-k_val)
            })
            .collect()
    }

    /// Batch evaluate all DLR tau functions at multiple points
    /// Returns Vec<Vec<f64>> where result[i][j] is function i evaluated at point j
    pub fn batch_evaluate_at(&self, taus: &[f64]) -> Vec<Vec<f64>> {
        use sparseir_rust::kernel::LogisticKernel;

        let n_funcs = self.poles.len();
        let n_points = taus.len();
        let mut result = vec![vec![0.0; n_points]; n_funcs];

        // DLR always uses LogisticKernel in tau domain
        let lambda = self.beta * self.wmax;
        let kernel = LogisticKernel::new(lambda);

        // Evaluate at each point
        for (j, &tau) in taus.iter().enumerate() {
            // Normalize tau to [0, beta] using the appropriate statistics
            let (tau_reg, sign) = match self.statistics {
                Statistics::Fermionic => normalize_tau::<Fermionic>(tau, self.beta),
                Statistics::Bosonic => normalize_tau::<Bosonic>(tau, self.beta),
            };
            let x_kern = 2.0 * tau_reg / self.beta - 1.0;

            for (i, &pole) in self.poles.iter().enumerate() {
                let y = pole / self.wmax;
                let k_val = kernel.compute(x_kern, y);
                result[i][j] = sign * (-k_val);
            }
        }

        result
    }
}

/// Wrapper for DLR functions in Matsubara domain
#[derive(Clone)]
pub(crate) struct DLRMatsubaraFuncs {
    pub poles: Vec<f64>,
    pub beta: f64,
    pub inv_weights: Vec<f64>,
    pub statistics: Statistics,
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
    pub(crate) _private: *mut std::ffi::c_void,
    pub(crate) beta: f64,
}

impl spir_funcs {
    /// Get a reference to the inner FuncsType
    pub(crate) fn inner_type(&self) -> &FuncsType {
        unsafe {
            &*(self._private as *const FuncsType)
        }
    }

    /// Create u funcs (tau-domain, Fermionic)
    pub(crate) fn from_u_fermionic(poly: Arc<PiecewiseLegendrePolyVector>, beta: f64) -> Self {
        let inner = FuncsType::PolyVector(PolyVectorFuncs {
            poly,
            domain: FunctionDomain::Tau(Statistics::Fermionic),
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
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
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
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
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
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
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
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
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
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
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
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
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
            beta,
        }
    }

    /// Create DLR tau funcs (tau-domain, Fermionic)
    /// Note: DLR always uses LogisticKernel regardless of the IR basis kernel type
    pub(crate) fn from_dlr_tau_fermionic(
        poles: Vec<f64>,
        beta: f64,
        wmax: f64,
        inv_weights: Vec<f64>,
    ) -> Self {
        let inner = FuncsType::DLRTau(DLRTauFuncs {
            poles,
            beta,
            wmax,
            inv_weights,
            statistics: Statistics::Fermionic,
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
            beta,
        }
    }

    /// Create DLR tau funcs (tau-domain, Bosonic)
    /// Note: DLR always uses LogisticKernel regardless of the IR basis kernel type
    pub(crate) fn from_dlr_tau_bosonic(
        poles: Vec<f64>,
        beta: f64,
        wmax: f64,
        inv_weights: Vec<f64>,
    ) -> Self {
        let inner = FuncsType::DLRTau(DLRTauFuncs {
            poles,
            beta,
            wmax,
            inv_weights,
            statistics: Statistics::Bosonic,
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
            beta,
        }
    }

    /// Create DLR Matsubara funcs (Matsubara-domain, Fermionic)
    pub(crate) fn from_dlr_matsubara_fermionic(
        poles: Vec<f64>,
        beta: f64,
        inv_weights: Vec<f64>,
    ) -> Self {
        let inner = FuncsType::DLRMatsubara(DLRMatsubaraFuncs {
            poles,
            beta,
            inv_weights,
            statistics: Statistics::Fermionic,
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
            beta,
        }
    }

    /// Create DLR Matsubara funcs (Matsubara-domain, Bosonic)
    pub(crate) fn from_dlr_matsubara_bosonic(
        poles: Vec<f64>,
        beta: f64,
        inv_weights: Vec<f64>,
    ) -> Self {
        let inner = FuncsType::DLRMatsubara(DLRMatsubaraFuncs {
            poles,
            beta,
            inv_weights,
            statistics: Statistics::Bosonic,
        });
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
            beta,
        }
    }

    /// Get the number of basis functions
    pub(crate) fn size(&self) -> usize {
        match self.inner_type() {
            FuncsType::PolyVector(pv) => pv.poly.polyvec.len(),
            FuncsType::FTVector(ftv) => {
                if let Some(ft) = &ftv.ft_fermionic {
                    ft.polyvec.len()
                } else if let Some(ft) = &ftv.ft_bosonic {
                    ft.polyvec.len()
                } else {
                    0
                }
            }
            FuncsType::DLRTau(dlr) => dlr.poles.len(),
            FuncsType::DLRMatsubara(dlr) => dlr.poles.len(),
        }
    }

    /// Get knots for continuous functions (PolyVector only)
    pub(crate) fn knots(&self) -> Option<Vec<f64>> {
        match self.inner_type() {
            FuncsType::PolyVector(pv) => {
                // Get unique knots from all polynomials
                let mut all_knots = Vec::new();
                for p in &pv.poly.polyvec {
                    for &knot in &p.knots {
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
    /// * `x` - For u: tau ∈ [-beta, beta], For v: omega ∈ [-omega_max, omega_max]
    ///
    /// # Returns
    /// Vector of function values, or None if not continuous
    pub(crate) fn eval_continuous(&self, x: f64) -> Option<Vec<f64>> {
        match self.inner_type() {
            FuncsType::PolyVector(pv) => Some(pv.evaluate_at(x, self.beta)),
            FuncsType::DLRTau(dlr) => Some(dlr.evaluate_at(x)),
            _ => None,
        }
    }

    /// Evaluate at a single Matsubara frequency (for FT functions only)
    ///
    /// # Arguments
    /// * `n` - Matsubara frequency index
    ///
    /// # Returns
    /// Vector of complex function values, or None if not FT type
    pub(crate) fn eval_matsubara(&self, n: i64) -> Option<Vec<num_complex::Complex64>> {
        match self.inner_type() {
            FuncsType::FTVector(ftv) => {
                if ftv.statistics == Statistics::Fermionic {
                    // Fermionic
                    let ft = ftv.ft_fermionic.as_ref()?;
                    let freq = MatsubaraFreq::<Fermionic>::new(n).ok()?;
                    let mut result = Vec::with_capacity(ft.polyvec.len());
                    for p in &ft.polyvec {
                        result.push(p.evaluate(&freq));
                    }
                    Some(result)
                } else {
                    // Bosonic
                    let ft = ftv.ft_bosonic.as_ref()?;
                    let freq = MatsubaraFreq::<Bosonic>::new(n).ok()?;
                    let mut result = Vec::with_capacity(ft.polyvec.len());
                    for p in &ft.polyvec {
                        result.push(p.evaluate(&freq));
                    }
                    Some(result)
                }
            }
            FuncsType::DLRMatsubara(dlr) => {
                // Evaluate DLR Matsubara functions: uhat_l(iν_n) = inv_weight[l] / (iν_n - pole_l)
                use num_complex::Complex;

                let mut result = Vec::with_capacity(dlr.poles.len());
                if dlr.statistics == Statistics::Fermionic {
                    // Fermionic
                    let freq = MatsubaraFreq::<Fermionic>::new(n).ok()?;
                    let iv = freq.value_imaginary(dlr.beta);
                    for (i, &pole) in dlr.poles.iter().enumerate() {
                        let inv_weight = dlr.inv_weights[i];
                        result.push(Complex::new(inv_weight, 0.0) / (iv - Complex::new(pole, 0.0)));
                    }
                } else {
                    // Bosonic
                    let freq = MatsubaraFreq::<Bosonic>::new(n).ok()?;
                    let iv = freq.value_imaginary(dlr.beta);
                    for (i, &pole) in dlr.poles.iter().enumerate() {
                        let inv_weight = dlr.inv_weights[i];
                        result.push(Complex::new(inv_weight, 0.0) / (iv - Complex::new(pole, 0.0)));
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Batch evaluate at multiple tau/omega points
    pub(crate) fn batch_eval_continuous(&self, xs: &[f64]) -> Option<Vec<Vec<f64>>> {
        match self.inner_type() {
            FuncsType::PolyVector(pv) => Some(pv.batch_evaluate_at(xs, self.beta)),
            FuncsType::DLRTau(dlr) => Some(dlr.batch_evaluate_at(xs)),
            _ => None,
        }
    }

    /// Batch evaluate at multiple Matsubara frequencies (for FT functions only)
    ///
    /// # Arguments
    /// * `ns` - Matsubara frequency indices
    ///
    /// # Returns
    /// Matrix of complex function values (size = `[n_funcs, n_freqs]`), or None if not FT type
    pub(crate) fn batch_eval_matsubara(
        &self,
        ns: &[i64],
    ) -> Option<Vec<Vec<num_complex::Complex64>>> {
        match self.inner_type() {
            FuncsType::FTVector(ftv) => {
                if ftv.statistics == Statistics::Fermionic {
                    // Fermionic
                    let ft = ftv.ft_fermionic.as_ref()?;
                    let n_funcs = ft.polyvec.len();
                    let n_points = ns.len();
                    let mut result =
                        vec![vec![num_complex::Complex64::new(0.0, 0.0); n_points]; n_funcs];

                    for (j, &n) in ns.iter().enumerate() {
                        let freq = MatsubaraFreq::<Fermionic>::new(n).ok()?;
                        for (i, p) in ft.polyvec.iter().enumerate() {
                            result[i][j] = p.evaluate(&freq);
                        }
                    }
                    Some(result)
                } else {
                    // Bosonic
                    let ft = ftv.ft_bosonic.as_ref()?;
                    let n_funcs = ft.polyvec.len();
                    let n_points = ns.len();
                    let mut result =
                        vec![vec![num_complex::Complex64::new(0.0, 0.0); n_points]; n_funcs];

                    for (j, &n) in ns.iter().enumerate() {
                        let freq = MatsubaraFreq::<Bosonic>::new(n).ok()?;
                        for (i, p) in ft.polyvec.iter().enumerate() {
                            result[i][j] = p.evaluate(&freq);
                        }
                    }
                    Some(result)
                }
            }
            FuncsType::DLRMatsubara(dlr) => {
                // Batch evaluate DLR Matsubara functions: uhat_l(iν_n) = inv_weight[l] / (iν_n - pole_l)
                use num_complex::Complex;

                let n_funcs = dlr.poles.len();
                let n_points = ns.len();
                let mut result = vec![vec![Complex::new(0.0, 0.0); n_points]; n_funcs];

                for (j, &n) in ns.iter().enumerate() {
                    if dlr.statistics == Statistics::Fermionic {
                        // Fermionic
                        let freq = MatsubaraFreq::<Fermionic>::new(n).ok()?;
                        let iv = freq.value_imaginary(dlr.beta);
                        for (i, &pole) in dlr.poles.iter().enumerate() {
                            let inv_weight = dlr.inv_weights[i];
                            result[i][j] =
                                Complex::new(inv_weight, 0.0) / (iv - Complex::new(pole, 0.0));
                        }
                    } else {
                        // Bosonic
                        let freq = MatsubaraFreq::<Bosonic>::new(n).ok()?;
                        let iv = freq.value_imaginary(dlr.beta);
                        for (i, &pole) in dlr.poles.iter().enumerate() {
                            let inv_weight = dlr.inv_weights[i];
                            result[i][j] =
                                Complex::new(inv_weight, 0.0) / (iv - Complex::new(pole, 0.0));
                        }
                    }
                }

                Some(result)
            }
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
                    if idx >= pv.poly.polyvec.len() {
                        return None;
                    }
                    new_polys.push(pv.poly.polyvec[idx].clone());
                }
                let new_poly_vec = PiecewiseLegendrePolyVector::new(new_polys);
                Some(Self {
                    _private: Box::into_raw(Box::new(FuncsType::PolyVector(PolyVectorFuncs {
                        poly: Arc::new(new_poly_vec),
                        domain: pv.domain,
                    }))) as *mut std::ffi::c_void,
                    beta: self.beta,
                })
            }
            FuncsType::FTVector(_) => {
                // FTVector slicing not yet supported (requires public constructor)
                None
            }
            FuncsType::DLRTau(dlr) => {
                // Select subset of poles
                let mut new_poles = Vec::with_capacity(indices.len());
                let mut new_inv_weights = Vec::with_capacity(indices.len());
                for &idx in indices {
                    if idx >= dlr.poles.len() {
                        return None;
                    }
                    new_poles.push(dlr.poles[idx]);
                    new_inv_weights.push(dlr.inv_weights[idx]);
                }
                Some(Self {
                    _private: Box::into_raw(Box::new(FuncsType::DLRTau(DLRTauFuncs {
                        poles: new_poles,
                        beta: dlr.beta,
                        wmax: dlr.wmax,
                        inv_weights: new_inv_weights,
                        statistics: dlr.statistics,
                    }))) as *mut std::ffi::c_void,
                    beta: dlr.beta,
                })
            }
            FuncsType::DLRMatsubara(dlr) => {
                // Select subset of poles
                let mut new_poles = Vec::with_capacity(indices.len());
                let mut new_inv_weights = Vec::with_capacity(indices.len());
                for &idx in indices {
                    if idx >= dlr.poles.len() {
                        return None;
                    }
                    new_poles.push(dlr.poles[idx]);
                    new_inv_weights.push(dlr.inv_weights[idx]);
                }
                Some(Self {
                    _private: Box::into_raw(Box::new(FuncsType::DLRMatsubara(DLRMatsubaraFuncs {
                        poles: new_poles,
                        beta: dlr.beta,
                        inv_weights: new_inv_weights,
                        statistics: dlr.statistics,
                    }))) as *mut std::ffi::c_void,
                    beta: dlr.beta,
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
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
            beta: self.beta,
        }
    }
}

impl Drop for spir_funcs {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut FuncsType);
            }
        }
    }
}

// Helper function for tau normalization is now provided by sparseir_rust::taufuncs::normalize_tau

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
    pub(crate) _private: *mut std::ffi::c_void,
}

impl spir_sampling {
    /// Get a reference to the inner SamplingType (for internal use by other modules)
    pub(crate) fn inner(&self) -> &SamplingType {
        unsafe {
            &*(self._private as *const SamplingType)
        }
    }
}

impl Clone for spir_sampling {
    fn clone(&self) -> Self {
        // Cheap clone: SamplingType::clone internally uses Arc::clone which is cheap
        let inner = self.inner().clone();
        Self {
            _private: Box::into_raw(Box::new(inner)) as *mut std::ffi::c_void,
        }
    }
}

impl Drop for spir_sampling {
    fn drop(&mut self) {
        unsafe {
            if !self._private.is_null() {
                let _ = Box::from_raw(self._private as *mut SamplingType);
            }
        }
    }
}

/// Internal enum to distinguish between different sampling types
#[derive(Clone)]
pub(crate) enum SamplingType {
    TauFermionic(Arc<sparseir_rust::sampling::TauSampling<Fermionic>>),
    TauBosonic(Arc<sparseir_rust::sampling::TauSampling<Bosonic>>),
    MatsubaraFermionic(Arc<sparseir_rust::matsubara_sampling::MatsubaraSampling<Fermionic>>),
    MatsubaraBosonic(Arc<sparseir_rust::matsubara_sampling::MatsubaraSampling<Bosonic>>),
    MatsubaraPositiveOnlyFermionic(
        Arc<sparseir_rust::matsubara_sampling::MatsubaraSamplingPositiveOnly<Fermionic>>,
    ),
    MatsubaraPositiveOnlyBosonic(
        Arc<sparseir_rust::matsubara_sampling::MatsubaraSamplingPositiveOnly<Bosonic>>,
    ),
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
