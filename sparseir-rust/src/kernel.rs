//! Kernel implementations for SparseIR
//!
//! This module provides kernel implementations for analytical continuation in quantum many-body physics.
//! The kernels are used in Fredholm integral equations of the first kind.
//!
//! u(x) = integral of K(x, y) v(y) dy
//!
//! where x ∈ [xmin, xmax] and y ∈ [ymin, ymax].
//!
//! In general, the kernel is applied to a scaled spectral function rho'(y) as:
//!
//! integral of K(x, y) rho'(y) dy,
//!
//! where ρ'(y) = w(y) ρ(y). The weight function w(y) transforms the original spectral
//! function ρ(y) into the scaled version ρ'(y) used in the integral equation.

use crate::numeric::CustomNumeric;
use crate::traits::{Statistics, StatisticsType};
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryType {
    Even,
    Odd,
}

impl SymmetryType {
    pub fn sign(self) -> i32 {
        match self {
            SymmetryType::Even => 1,
            SymmetryType::Odd => -1,
        }
    }
}

/// Trait for SVE (Singular Value Expansion) hints
///
/// Provides discretization hints for singular value expansion of a given kernel.
/// This includes segment information and numerical parameters for efficient computation.
pub trait SVEHints<T>: Debug + Send + Sync
where
    T: Copy + Debug + Send + Sync,
{
    /// Get the x-axis segments for discretization
    ///
    /// For centrosymmetric kernels, returns only positive values (x >= 0) including the endpoints.
    /// The returned vector contains segments from [0, xmax] where xmax is the
    /// upper bound of the x domain.
    ///
    /// For non-centrosymmetric kernels, returns segments covering the full domain
    /// [-xmax, xmax].
    fn segments_x(&self) -> Vec<T>;

    /// Get the y-axis segments for discretization
    ///
    /// For centrosymmetric kernels, returns only positive values (y >= 0) including the endpoints.
    /// The returned vector contains segments from [0, ymax] where ymax is the
    /// upper bound of the y domain.
    ///
    /// For non-centrosymmetric kernels, returns segments covering the full domain
    /// [-ymax, ymax].
    fn segments_y(&self) -> Vec<T>;

    /// Get the number of singular values hint
    fn nsvals(&self) -> usize;

    /// Get the number of Gauss points for quadrature
    fn ngauss(&self) -> usize;
}

/// Trait for kernel type properties (static characteristics)
pub trait KernelProperties {
    /// Associated type for SVE hints
    type SVEHintsType<T>: SVEHints<T> + Clone
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
    /// Power with which the y coordinate scales.
    ///
    /// For most kernels, this is 0 (no scaling).
    /// For RegularizedBoseKernel, this is 1 (linear scaling).
    fn ypower(&self) -> i32;

    /// Convergence radius of the Matsubara basis asymptotic model
    ///
    /// For improved numerical accuracy, IR basis functions on Matsubara axis
    /// can be evaluated from asymptotic expression for |n| > conv_radius.
    fn conv_radius(&self) -> f64;

    /// Get the upper bound of the x domain
    fn xmax(&self) -> f64;

    /// Get the upper bound of the y domain
    fn ymax(&self) -> f64;

    /// Weight function for given statistics.
    ///
    /// The kernel is applied to a scaled spectral function ρ'(y) as:
    ///     ∫ K(x, y) ρ'(y) dy,
    /// where ρ'(y) = w(y) ρ(y).
    ///
    /// This function returns w(beta, omega) that transforms the original spectral
    /// function ρ(y) into the scaled version ρ'(y) used in the integral equation.
    ///
    /// @param beta Inverse temperature
    /// @param omega Frequency
    /// @return The weight value w(beta, omega)
    fn weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64;

    /// Inverse weight function to avoid division by zero.
    ///
    /// This is a safer API that returns the inverse weight.
    ///
    /// @param beta Inverse temperature
    /// @param omega Frequency
    /// @return The inverse weight value
    fn inv_weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64;

    /// Create SVE hints for this kernel type.
    ///
    /// Provides discretization hints for singular value expansion computation.
    /// The hints include segment information and numerical parameters optimized
    /// for the specific kernel type.
    ///
    /// @param epsilon Target accuracy for the SVE computation
    /// @return SVE hints specific to this kernel type
    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
}

/// Trait for general kernels (both centrosymmetric and non-centrosymmetric)
///
/// This trait provides the basic interface for computing kernel values.
/// Centrosymmetric kernels should implement `CentrosymmKernel` instead,
/// which provides additional optimizations.
pub trait AbstractKernel: Send + Sync {
    /// Compute the kernel value K(x, y) with high precision
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate (typically in [-xmax, xmax])
    /// * `y` - The y coordinate (typically in [-ymax, ymax])
    ///
    /// # Returns
    ///
    /// The kernel value K(x, y)
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T;

    /// Check if the kernel is centrosymmetric
    ///
    /// Returns true if and only if K(x, y) == K(-x, -y) for all values of x and y.
    /// This allows the kernel to be block-diagonalized, speeding up the
    /// singular value expansion by a factor of 4.
    ///
    /// # Returns
    ///
    /// True if the kernel is centrosymmetric, false otherwise.
    fn is_centrosymmetric(&self) -> bool {
        false
    }
}

/// Trait for centrosymmetric kernels
///
/// Centrosymmetric kernels satisfy K(x, y) = K(-x, -y) and can be decomposed
/// into even and odd components for efficient computation.
pub trait CentrosymmKernel: AbstractKernel {
    /// Compute the reduced kernel value
    ///
    /// K_red(x, y) = K(x, y) + sign * K(x, -y)
    /// where sign = 1 for even symmetry and sign = -1 for odd symmetry
    fn compute_reduced<T: CustomNumeric + Copy + Debug>(
        &self,
        x: T,
        y: T,
        symmetry: SymmetryType,
    ) -> T;

    /// Get the cutoff parameter Λ (lambda)
    fn lambda(&self) -> f64;
}

/// Logistic kernel for fermionic analytical continuation
///
/// This kernel implements K(x, y) = exp(-Λy(x + 1)/2)/(1 + exp(-Λy))
/// where x ∈ [-1, 1] and y ∈ [-1, 1]
#[derive(Debug, Clone, Copy)]
pub struct LogisticKernel {
    lambda: f64,
}

impl LogisticKernel {
    /// Create a new logistic kernel with the given cutoff parameter
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Get the cutoff parameter
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl KernelProperties for LogisticKernel {
    type SVEHintsType<T>
        = LogisticSVEHints<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
    fn ypower(&self) -> i32 {
        0 // No y-power scaling for LogisticKernel
    }

    fn conv_radius(&self) -> f64 {
        40.0 * self.lambda
    }

    fn xmax(&self) -> f64 {
        1.0
    }
    fn ymax(&self) -> f64 {
        1.0
    }

    fn weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64 {
        match S::STATISTICS {
            Statistics::Fermionic => {
                // For fermionic statistics: w(beta, omega) = 1.0
                // The kernel K(x, y) is already in the correct form for fermions
                1.0
            }
            Statistics::Bosonic => {
                // For bosonic statistics: w(beta, omega) = 1.0 / tanh(0.5 * beta * omega)
                // This transforms the fermionic kernel to work with bosonic correlation functions
                // The tanh factor accounts for the different statistics
                1.0 / (0.5 * beta * omega).tanh()
            }
        }
    }

    fn inv_weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64 {
        match S::STATISTICS {
            Statistics::Fermionic => {
                // For fermionic statistics: 1/w = 1.0 (safe, no division by zero)
                1.0
            }
            Statistics::Bosonic => {
                // For bosonic statistics: 1/w = tanh(0.5 * beta * omega) (safe, handles omega=0 case)
                // This avoids division by zero when tanh(0.5 * beta * omega) approaches zero
                (0.5 * beta * omega).tanh()
            }
        }
    }

    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        LogisticSVEHints::new(*self, epsilon)
    }
}

pub fn compute_logistic_kernel<T: CustomNumeric>(lambda: f64, x: T, y: T) -> T {
    let x_plus: T = T::from_f64_unchecked(1.0) + x;
    let x_minus: T = T::from_f64_unchecked(1.0) - x;

    let u_plus: T = T::from_f64_unchecked(0.5) * x_plus;
    let u_minus: T = T::from_f64_unchecked(0.5) * x_minus;
    let v: T = T::from_f64_unchecked(lambda) * y;

    let mabs_v: T = -v.abs_as_same_type();
    let numerator: T = if v >= T::from_f64_unchecked(0.0) {
        (u_plus * mabs_v).exp()
    } else {
        (u_minus * mabs_v).exp()
    };
    let denominator: T = T::from_f64_unchecked(1.0) + mabs_v.exp();
    numerator / denominator
}

fn compute_logistic_kernel_reduced_odd<T: CustomNumeric>(lambda: f64, x: T, y: T) -> T {
    // For x * y around 0, antisymmetrization introduces cancellation, which
    // reduces the relative precision. To combat this, we replace the
    // values with the explicit form
    let v_half: T = T::from_f64_unchecked(lambda * 0.5) * y;
    // Use direct comparison to match C++ implementation (avoid precision loss from to_f64())
    let xy_small: bool = x * v_half < T::from_f64_unchecked(1.0);
    let cosh_finite: bool = v_half < T::from_f64_unchecked(85.0);
    if xy_small && cosh_finite {
        -(v_half * x).sinh() / v_half.cosh()
    } else {
        let k_plus = compute_logistic_kernel(lambda, x, y);
        let k_minus = compute_logistic_kernel(lambda, x, -y);
        k_plus - k_minus
    }
}

impl AbstractKernel for LogisticKernel {
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
        compute_logistic_kernel(self.lambda, x, y)
    }

    fn is_centrosymmetric(&self) -> bool {
        true
    }
}

impl CentrosymmKernel for LogisticKernel {
    fn compute_reduced<T: CustomNumeric + Copy + Debug>(
        &self,
        x: T,
        y: T,
        symmetry: SymmetryType,
    ) -> T {
        match symmetry {
            SymmetryType::Even => self.compute(x, y) + self.compute(x, -y),
            SymmetryType::Odd => compute_logistic_kernel_reduced_odd(self.lambda, x, y),
        }
    }

    fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// SVE hints for LogisticKernel
#[derive(Debug, Clone)]
pub struct LogisticSVEHints<T> {
    kernel: LogisticKernel,
    epsilon: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> LogisticSVEHints<T>
where
    T: Copy + Debug + Send + Sync,
{
    pub fn new(kernel: LogisticKernel, epsilon: f64) -> Self {
        Self {
            kernel,
            epsilon,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> SVEHints<T> for LogisticSVEHints<T>
where
    T: Copy + Debug + Send + Sync + CustomNumeric,
{
    fn segments_x(&self) -> Vec<T> {
        // Direct implementation that generates only non-negative sample points
        // This is equivalent to the C++ implementation but without the full symmetric array creation
        let nzeros = std::cmp::max((15.0 * self.kernel.lambda().log10()).round() as usize, 1);

        // Create a range of values
        let mut temp = vec![0.0; nzeros];
        for i in 0..nzeros {
            temp[i] = 0.143 * i as f64;
        }

        // Calculate diffs using the inverse hyperbolic cosine
        let mut diffs = vec![0.0; nzeros];
        for i in 0..nzeros {
            diffs[i] = 1.0 / temp[i].cosh();
        }

        // Calculate cumulative sum of diffs
        let mut zeros = vec![0.0; nzeros];
        zeros[0] = diffs[0];
        for i in 1..nzeros {
            zeros[i] = zeros[i - 1] + diffs[i];
        }

        // Normalize zeros
        let last_zero = zeros[nzeros - 1];
        for i in 0..nzeros {
            zeros[i] /= last_zero;
        }

        // Create segments with only non-negative values (x >= 0) including endpoints [0, xmax]
        let mut segments = Vec::with_capacity(nzeros + 1);

        // Add 0.0 endpoint
        segments.push(<T as CustomNumeric>::from_f64_unchecked(0.0));

        // Add positive zeros (already in [0, 1] range)
        for i in 0..nzeros {
            segments.push(<T as CustomNumeric>::from_f64_unchecked(zeros[i]));
        }

        // Ensure segments are sorted in ascending order [0, ..., xmax]
        segments.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        segments
    }

    fn segments_y(&self) -> Vec<T> {
        // Direct implementation that generates only non-negative sample points
        // This is equivalent to the C++ implementation but without the full symmetric array creation
        let nzeros = std::cmp::max((20.0 * self.kernel.lambda().log10()).round() as usize, 2);

        // Initial differences (from C++ implementation)
        let mut diffs = vec![
            0.01523, 0.03314, 0.04848, 0.05987, 0.06703, 0.07028, 0.07030, 0.06791, 0.06391,
            0.05896, 0.05358, 0.04814, 0.04288, 0.03795, 0.03342, 0.02932, 0.02565, 0.02239,
            0.01951, 0.01699,
        ];

        // Truncate diffs if necessary
        if nzeros < diffs.len() {
            diffs.truncate(nzeros);
        }

        // Calculate trailing differences
        for i in 20..nzeros {
            let x = 0.141 * i as f64;
            diffs.push(0.25 * (-x).exp());
        }

        // Calculate cumulative sum of diffs
        let mut zeros = Vec::with_capacity(nzeros);
        zeros.push(diffs[0]);
        for i in 1..nzeros {
            zeros.push(zeros[i - 1] + diffs[i]);
        }

        // Normalize zeros
        let last_zero = zeros[nzeros - 1];
        for i in 0..nzeros {
            zeros[i] /= last_zero;
        }
        zeros.pop(); // Remove last element

        // Updated nzeros
        let nzeros = zeros.len();

        // Adjust zeros
        for i in 0..nzeros {
            zeros[i] -= 1.0;
        }

        // Generate segments directly from negative zeros
        let mut segments: Vec<T> = Vec::new();

        segments.push(<T as CustomNumeric>::from_f64_unchecked(1.0));

        // Add absolute values of negative zeros
        for i in 0..nzeros {
            let abs_val = -zeros[i];
            segments.push(<T as CustomNumeric>::from_f64_unchecked(abs_val));
        }

        if segments[segments.len() - 1].abs_as_same_type() > T::epsilon() {
            segments.push(<T as CustomNumeric>::from_f64_unchecked(0.0));
        }

        // Sort in ascending order
        segments.sort_by(|a, b| a.partial_cmp(b).unwrap());

        segments
    }

    fn nsvals(&self) -> usize {
        let log10_lambda = self.kernel.lambda().log10().max(1.0);
        ((25.0 + log10_lambda) * log10_lambda).round() as usize
    }

    fn ngauss(&self) -> usize {
        if self.epsilon >= 1e-8 { 10 } else { 16 }
    }
}

// ============================================================================
// RegularizedBoseKernel
// ============================================================================

/// Regularized bosonic analytical continuation kernel
///
/// In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is:
///
/// ```text
/// K(x, y) = y * exp(-Λ y (x + 1) / 2) / (1 - exp(-Λ y))
/// ```
///
/// This kernel is used for bosonic Green's functions. The factor y regularizes
/// the singularity at ω = 0, making the kernel well-behaved for numerical work.
///
/// The dimensionalized kernel is related by:
/// ```text
/// K(τ, ω) = ωmax * K(2τ/β - 1, ω/ωmax)
/// ```
/// where ωmax = Λ/β.
///
/// # Properties
/// - **Centrosymmetric**: K(x, y) = K(-x, -y)
/// - **ypower = 1**: Spectral function transforms as ρ'(y) = y * ρ(y)
/// - **Bosonic only**: Does not support fermionic statistics
/// - **Weight function**: w(β, ω) = 1/ω for bosonic statistics
///
/// # Numerical Stability
/// The expression v / (exp(v) - 1) is evaluated using expm1 for small |v|.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegularizedBoseKernel {
    /// Kernel cutoff parameter Λ = β × ωmax
    pub lambda: f64,
}

impl RegularizedBoseKernel {
    /// Create a new RegularizedBoseKernel
    ///
    /// # Arguments
    /// * `lambda` - Kernel cutoff Λ (must be non-negative)
    ///
    /// # Panics
    /// Panics if lambda < 0
    pub fn new(lambda: f64) -> Self {
        if lambda < 0.0 {
            panic!("Kernel cutoff Λ must be non-negative, got {}", lambda);
        }
        Self { lambda }
    }

    /// Compute kernel value with numerical stability
    ///
    /// Evaluates K(x, y) = y * exp(-Λy(x+1)/2) / (1 - exp(-Λy))
    /// using expm1 for better accuracy near y = 0.
    ///
    /// # Arguments
    /// * `x` - Normalized time coordinate (x ∈ [-1, 1])
    /// * `y` - Normalized frequency coordinate (y ∈ [-1, 1])
    /// * `x_plus` - Precomputed x + 1 (optional, for efficiency)
    /// * `x_minus` - Precomputed 1 - x (optional, for efficiency)
    fn compute_impl<T>(&self, x: T, y: T, x_plus: Option<T>, x_minus: Option<T>) -> T
    where
        T: CustomNumeric,
    {
        // u_plus = (x + 1) / 2, u_minus = (1 - x) / 2
        // x_plus and x_minus are (x+1) and (1-x), so we need to multiply by 0.5
        let u_plus = x_plus
            .map(|xp| T::from_f64_unchecked(0.5) * xp)
            .unwrap_or_else(|| T::from_f64_unchecked(0.5) * (T::from_f64_unchecked(1.0) + x));
        let u_minus = x_minus
            .map(|xm| T::from_f64_unchecked(0.5) * xm)
            .unwrap_or_else(|| T::from_f64_unchecked(0.5) * (T::from_f64_unchecked(1.0) - x));

        let v = T::from_f64_unchecked(self.lambda) * y;
        let absv = v.abs_as_same_type();

        // Handle y ≈ 0 using Taylor expansion
        // K(x,y) = 1/Λ - xy/2 + (1/24)Λ(3x² - 1)y² + O(y³)
        // For |Λy| < 2e-14, use first-order approximation
        // This avoids division by zero when exp(-|Λy|) ≈ 1
        if absv.to_f64() < 2e-14 {
            let term0 = T::from_f64_unchecked(1.0 / self.lambda);
            let term1 = T::from_f64_unchecked(0.5) * x * y;
            return term0 - term1;
        }

        // enum_val = exp(-|v| * (v >= 0 ? u_plus : u_minus))
        let enum_val = if v >= T::from_f64_unchecked(0.0) {
            (-absv * u_plus).exp()
        } else {
            (-absv * u_minus).exp()
        };

        // Handle v / (exp(v) - 1) with numerical stability
        // Follows C++ implementation using expm1 pattern
        // denom = absv / expm1(-absv) = absv / (exp(-absv) - 1)
        let exp_neg_absv = (-absv).exp();
        let denom = absv / (exp_neg_absv - T::from_f64_unchecked(1.0));

        // K(x, y) = -1/Λ * enum_val * denom
        // Since denom is negative (exp(-absv) < 1), final result is positive
        T::from_f64_unchecked(-1.0 / self.lambda) * enum_val * denom
    }
}

impl KernelProperties for RegularizedBoseKernel {
    type SVEHintsType<T>
        = RegularizedBoseSVEHints<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;

    fn ypower(&self) -> i32 {
        1 // Spectral function transforms as ρ'(y) = y * ρ(y)
    }

    fn conv_radius(&self) -> f64 {
        40.0 * self.lambda
    }

    fn xmax(&self) -> f64 {
        1.0
    }

    fn ymax(&self) -> f64 {
        1.0
    }

    fn weight<S: StatisticsType + 'static>(&self, _beta: f64, omega: f64) -> f64 {
        match S::STATISTICS {
            Statistics::Fermionic => {
                panic!("RegularizedBoseKernel does not support fermionic functions");
            }
            Statistics::Bosonic => {
                // weight = 1/ω
                if omega.abs() < 1e-300 {
                    panic!("RegularizedBoseKernel: omega too close to zero");
                }
                1.0 / omega
            }
        }
    }

    fn inv_weight<S: StatisticsType + 'static>(&self, _beta: f64, omega: f64) -> f64 {
        match S::STATISTICS {
            Statistics::Fermionic => {
                panic!("RegularizedBoseKernel does not support fermionic functions");
            }
            Statistics::Bosonic => {
                // inv_weight = ω (safe, no division)
                omega
            }
        }
    }

    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        RegularizedBoseSVEHints::new(*self, epsilon)
    }
}

impl AbstractKernel for RegularizedBoseKernel {
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
        let x_plus = Some(T::from_f64_unchecked(1.0) + x);
        let x_minus = Some(T::from_f64_unchecked(1.0) - x);
        self.compute_impl(x, y, x_plus, x_minus)
    }

    fn is_centrosymmetric(&self) -> bool {
        true
    }
}

impl CentrosymmKernel for RegularizedBoseKernel {
    fn compute_reduced<T: CustomNumeric + Copy + Debug>(
        &self,
        x: T,
        y: T,
        symmetry: SymmetryType,
    ) -> T {
        match symmetry {
            SymmetryType::Even => self.compute(x, y) + self.compute(x, -y),
            SymmetryType::Odd => {
                // For RegularizedBoseKernel, use sinh formulation for numerical stability
                // K(x,y) - K(x,-y) = -y * sinh(Λ y x / 2) / sinh(Λ y / 2)
                let v_half = T::from_f64_unchecked(self.lambda * 0.5) * y;
                let xv_half = x * v_half;
                let xy_small = xv_half.to_f64().abs() < 1.0;
                let sinh_finite = v_half.to_f64().abs() < 85.0 && v_half.to_f64().abs() > 1e-200;

                if xy_small && sinh_finite {
                    // Use sinh formulation for numerical stability
                    // NOTE: Minus sign is critical! (matches Julia/C++ implementation)
                    -y * xv_half.sinh() / v_half.sinh()
                } else {
                    // Fall back to direct computation
                    self.compute(x, y) - self.compute(x, -y)
                }
            }
        }
    }

    fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// SVE hints for RegularizedBoseKernel
#[derive(Debug, Clone)]
pub struct RegularizedBoseSVEHints<T> {
    kernel: RegularizedBoseKernel,
    epsilon: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> RegularizedBoseSVEHints<T>
where
    T: Copy + Debug + Send + Sync,
{
    pub fn new(kernel: RegularizedBoseKernel, epsilon: f64) -> Self {
        Self {
            kernel,
            epsilon,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> SVEHints<T> for RegularizedBoseSVEHints<T>
where
    T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
{
    fn segments_x(&self) -> Vec<T> {
        // Returns segments for x >= 0 domain only
        // C++/Julia: nzeros = max(round(15 * log10(lambda)), 15)
        let nzeros = ((15.0 * self.kernel.lambda.log10()).round() as usize).max(15);

        // temp[i] = 0.18 * i
        let mut temp = vec![0.0; nzeros];
        for i in 0..nzeros {
            temp[i] = 0.18 * i as f64;
        }

        // diffs[i] = 1.0 / cosh(temp[i])
        let mut diffs = vec![0.0; nzeros];
        for i in 0..nzeros {
            diffs[i] = 1.0 / temp[i].cosh();
        }

        // Cumulative sum
        let mut zeros = vec![0.0; nzeros];
        zeros[0] = diffs[0];
        for i in 1..nzeros {
            zeros[i] = zeros[i - 1] + diffs[i];
        }

        // Normalize
        let last_zero = zeros[nzeros - 1];
        for i in 0..nzeros {
            zeros[i] /= last_zero;
        }

        // Create segments with only non-negative values: [0, zeros[0], zeros[1], ...]
        let mut segments = Vec::with_capacity(nzeros + 1);
        segments.push(T::from_f64_unchecked(0.0));
        for i in 0..nzeros {
            segments.push(T::from_f64_unchecked(zeros[i]));
        }

        // Ensure sorted (should already be sorted, but verify)
        segments.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        segments
    }

    fn segments_y(&self) -> Vec<T> {
        // Returns segments for y >= 0 domain only
        // C++/Julia: nzeros = max(round(20 * log10(lambda)), 20)
        let nzeros = ((20.0 * self.kernel.lambda.log10()).round() as usize).max(20);

        // diffs[j] = 0.12 / exp(0.0337 * j * log(j + 1))
        let mut diffs = vec![0.0; nzeros];
        for j in 0..nzeros {
            let j_f64 = j as f64;
            let exponent = 0.0337 * j_f64 * (j_f64 + 1.0).ln();
            diffs[j] = 0.12 / exponent.exp();
        }

        // Cumulative sum
        let mut zeros = vec![0.0; nzeros];
        zeros[0] = diffs[0];
        for i in 1..nzeros {
            zeros[i] = zeros[i - 1] + diffs[i];
        }

        // Normalize by last value, then remove it
        let last_zero = zeros[nzeros - 1];
        for i in 0..nzeros {
            zeros[i] /= last_zero;
        }
        zeros.pop();

        // After normalization and pop, zeros are in [0, 1) range
        // Create segments: [0, zeros[0], zeros[1], ..., 1]
        let mut segments = Vec::with_capacity(zeros.len() + 2);
        segments.push(T::from_f64_unchecked(0.0));
        for z in zeros {
            segments.push(T::from_f64_unchecked(z));
        }
        segments.push(T::from_f64_unchecked(1.0));

        // Ensure sorted (should already be sorted, but verify)
        segments.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        segments
    }

    fn nsvals(&self) -> usize {
        // C++: int(round(28 * max(1.0, log10(lambda))))
        let log10_lambda = self.kernel.lambda.log10().max(1.0);
        (28.0 * log10_lambda).round() as usize
    }

    fn ngauss(&self) -> usize {
        if self.epsilon >= 1e-8 { 10 } else { 16 }
    }
}

#[cfg(test)]
#[path = "kernel_tests.rs"]
mod tests;
