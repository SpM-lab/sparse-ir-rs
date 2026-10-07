//! Discrete Lehmann representations built from an IR basis
//!
//! The DLR itself needs no IR basis (see
//! [`DiscreteLehmannRepresentation::new`]). This module connects the two:
//! [`IrBasis`] is a [`Basis`] with a kernel (the IR basis
//! [`FiniteTempBasis`]), [`DlrFromIr`] builds a DLR whose poles come from such
//! a basis, and [`IrBasis::dlr_transform`] the linear map between the
//! coefficients of the basis and of any DLR on the same domain.

use crate::basis::FiniteTempBasis;
use crate::basis_trait::Basis;
use crate::dlr::{DiscreteLehmannRepresentation, IrDlrTransform};
use crate::error::Error;
use crate::kernel::{CentrosymmKernel, KernelProperties};
use crate::matrix::Mat;
use crate::traits::{Statistics, StatisticsType};

/// A basis with a kernel: the IR basis [`FiniteTempBasis`]
///
/// The kernel supplies the pole weights `w(β, ω)` of a DLR built from the
/// basis. A DLR is not an `IrBasis`, so no DLR can be built from a DLR.
pub trait IrBasis<S: StatisticsType>: Basis<S> {
    /// Kernel type
    type Kernel: KernelProperties;

    /// The kernel the basis was built from
    fn kernel(&self) -> &Self::Kernel;

    /// The linear map between the coefficients of this basis and of `dlr`.
    ///
    /// # Errors
    /// * [`Error::KernelStatisticsMismatch`] if the kernel of this basis does not
    ///   support the statistics
    /// * [`Error::InvalidParameter`] named `basis` if the β of this basis differs from that
    ///   of `dlr`, or its kernel weight vanishes at a pole where the DLR
    ///   weight does not
    /// * [`Error::OutOfDomain`] named `poles` for a pole of `dlr` outside
    ///   [-ωmax, ωmax] of this basis
    fn dlr_transform(&self, dlr: &DiscreteLehmannRepresentation<S>) -> Result<IrDlrTransform, Error>
    where
        S: 'static,
    {
        if S::STATISTICS == Statistics::Fermionic && self.kernel().ypower() == 1 {
            return Err(Error::KernelStatisticsMismatch);
        }
        let (beta, wmax) = (Basis::beta(self), Basis::wmax(self));
        let rel = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
        if !rel(beta, dlr.beta()) {
            return Err(Error::InvalidParameter {
                name: "basis",
                value: format!("beta = {beta:?}"),
                reason: format!("must equal beta = {:?} of the DLR", dlr.beta()),
            });
        }
        if let Some(&pole) = dlr.poles().iter().find(|p| !(-wmax..=wmax).contains(*p)) {
            return Err(Error::OutOfDomain {
                name: "poles",
                value: pole,
                domain: (-wmax, wmax),
            });
        }
        let mut ratio = Vec::with_capacity(dlr.poles().len());
        for (&pole, &w) in dlr.poles().iter().zip(dlr.pole_weights()) {
            let wir = self.kernel().regularizer::<S>(beta, pole);
            if w == wir {
                ratio.push(1.0);
            } else if wir == 0.0 {
                return Err(Error::InvalidParameter {
                    name: "basis",
                    value: format!("kernel weight 0 at pole {pole:?}"),
                    reason: "must not vanish where the DLR weight does not".to_string(),
                });
            } else {
                ratio.push(w / wir);
            }
        }
        let v_at_poles =
            crate::sampling::mat_from_matrix(&Basis::evaluate_omega(self, dlr.poles())?)?;
        let s = Basis::svals(self);
        let fitmat = Mat::<f64>::from_fn([Basis::size(self), dlr.poles().len()], |idx| {
            let (l, i) = (idx[0], idx[1]);
            -s[l] * v_at_poles[[i, l]] * ratio[i]
        });
        Ok(IrDlrTransform::from_matrix(fitmat))
    }
}

impl<K, S> IrBasis<S> for FiniteTempBasis<K, S>
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType + 'static,
{
    type Kernel = K;

    fn kernel(&self) -> &K {
        FiniteTempBasis::kernel(self)
    }
}

/// Constructors of a [`DiscreteLehmannRepresentation`] from an IR basis
///
/// The returned DLR carries the [`IrDlrTransform`] of its source basis, so
/// [`DiscreteLehmannRepresentation::from_ir_nd`] and
/// [`DiscreteLehmannRepresentation::to_ir_nd`] work on it.
pub trait DlrFromIr<S: StatisticsType>: Sized {
    /// Create a DLR from an IR basis with custom poles
    ///
    /// The tau-domain pole basis is built from the logistic representation, while
    /// kernel-specific regularizers are preserved for compatible kernels.
    ///
    /// # Arguments
    /// * `basis` - The IR basis to construct DLR from
    /// * `poles` - Pole positions on the real-frequency axis, in
    ///   [-ωmax, ωmax] of `basis`
    ///
    /// # Errors
    /// * [`Error::KernelStatisticsMismatch`] if the kernel does not support
    ///   the requested statistics (e.g. `RegularizedBoseKernel` with fermionic
    ///   statistics)
    /// * [`Error::EmptyInput`] if `poles` is empty
    /// * [`Error::NonFiniteInput`] if a pole is NaN or infinite, and
    ///   [`Error::OutOfDomain`] if a pole is outside [-ωmax, ωmax] of `basis`,
    ///   both named `poles`, for the first such pole
    /// * [`Error::NotSupported`] for a bosonic pole at 0 if the kernel has a
    ///   `ypower` other than 0 or 1 (the DLR knows the limit at 0 for those
    ///   only)
    ///
    /// Duplicate poles are accepted. They make [`DiscreteLehmannRepresentation::from_ir_nd`]
    /// ill-conditioned: the coefficients of equal poles are not unique,
    /// although the round trip through [`DiscreteLehmannRepresentation::to_ir_nd`] still recovers the
    /// IR coefficients.
    fn from_ir_with_poles<B: IrBasis<S>>(basis: &B, poles: Vec<f64>) -> Result<Self, Error>;

    /// Create a DLR from an IR basis with its default pole locations
    ///
    /// Uses the default omega sampling points from the basis.
    ///
    /// # Arguments
    /// * `basis` - The IR basis to construct DLR from
    ///
    /// # Errors
    /// * [`Error::InsufficientDefaultPoles`] if the basis has fewer default
    ///   poles than functions. This can happen with certain kernel types
    ///   (e.g. `RegularizedBoseKernel`) due to numerical precision limitations
    ///   in root finding.
    /// * [`Error::KernelStatisticsMismatch`] as in [`Self::from_ir_with_poles`]
    /// * [`Error::NotSupported`] as in [`Self::from_ir_with_poles`]: for a default
    ///   pole at 0 of a bosonic basis whose kernel has a `ypower` other than
    ///   0 or 1
    /// * The errors of
    ///   [`Basis::default_omega_sampling_points`]
    ///   (NotSupported for an SVE with too few singular functions)
    fn from_ir<B: IrBasis<S>>(basis: &B) -> Result<Self, Error>;
}

impl<S: StatisticsType + 'static> DlrFromIr<S> for DiscreteLehmannRepresentation<S> {
    fn from_ir_with_poles<B: IrBasis<S>>(basis: &B, poles: Vec<f64>) -> Result<Self, Error> {
        // RegularizedBoseKernel (ypower == 1) is meaningful only for bosonic
        // statistics; its regularizer panics for fermionic input. Reject the
        // combination before computing anything.
        if S::STATISTICS == Statistics::Fermionic && basis.kernel().ypower() == 1 {
            return Err(Error::KernelStatisticsMismatch);
        }
        // Without poles the fitting matrix has no columns and the DLR has no
        // functions.
        if poles.is_empty() {
            return Err(Error::EmptyInput { name: "poles" });
        }

        let beta = Basis::beta(basis);
        let wmax = Basis::wmax(basis);
        let accuracy = Basis::accuracy(basis);
        let kernel_ypower = basis.kernel().ypower();

        // Each pole must be finite and in [-ωmax, ωmax], the domain of V_l.
        // evaluate_omega below checks the domain of the basis again.
        for (i, &pole) in poles.iter().enumerate() {
            if !pole.is_finite() {
                return Err(Error::NonFiniteInput {
                    name: "poles",
                    index: vec![i],
                    value: pole,
                });
            }
            if !(-wmax..=wmax).contains(&pole) {
                return Err(Error::OutOfDomain {
                    name: "poles",
                    value: pole,
                    domain: (-wmax, wmax),
                });
            }
        }
        // A bosonic pole at 0 is evaluated through its finite limit, which is
        // known for ypower 0 (regularizer tanh(βω/2)) and 1 (regularizer ω)
        // only; see zero_pole_tau_limit.
        if S::STATISTICS == Statistics::Bosonic
            && !(0..=1).contains(&kernel_ypower)
            && poles.contains(&0.0)
        {
            return Err(Error::NotSupported {
                what: format!(
                    "a bosonic DLR pole at 0 for a kernel with ypower = {kernel_ypower}: \
                     its limit is known for ypower 0 and 1 only"
                ),
            });
        }

        // Compute fitting matrix: fitmat = -s · V(poles)
        // This transforms DLR coefficients to IR coefficients
        let v_at_poles = crate::sampling::mat_from_matrix(&Basis::evaluate_omega(basis, &poles)?)?; // shape: [n_poles, basis_size]
        let s = Basis::svals(basis); // Non-normalized singular values (same as C++)

        let basis_size = Basis::size(basis);
        let n_poles = poles.len();

        // fitmat[l, i] = -s[l] * V_l(pole[i])
        // C++: fitmat = (-A_array * s_array.replicate(1, A.cols())).matrix()
        let fitmat = Mat::<f64>::from_fn([basis_size, n_poles], |idx| {
            let l = idx[0];
            let i = idx[1];
            -s[l] * v_at_poles[[i, l]]
        });

        let ir = IrDlrTransform::from_matrix(fitmat);

        let regularizers: Vec<f64> = poles
            .iter()
            .map(|&pole| basis.kernel().regularizer::<S>(beta, pole))
            .collect();

        Ok(DiscreteLehmannRepresentation::from_parts(
            beta,
            wmax,
            accuracy,
            poles,
            regularizers,
            kernel_ypower,
            Some(ir),
        ))
    }

    fn from_ir<B: IrBasis<S>>(basis: &B) -> Result<Self, Error> {
        let poles = Basis::default_omega_sampling_points(basis)?;
        let basis_size = Basis::size(basis);
        if basis_size > poles.len() {
            return Err(Error::InsufficientDefaultPoles {
                basis_size,
                n_poles: poles.len(),
            });
        }
        Self::from_ir_with_poles(basis, poles)
    }
}
