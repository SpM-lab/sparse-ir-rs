//! The lattice and basis setup shared by the applied examples that solve a
//! single-band Hubbard model on a square lattice.
//!
//! Everything here is independent of the interaction, so a parameter scan
//! builds one [`Lattice`] and reuses it, and a scan over temperature can at
//! least reuse the singular value expansion.

use sparse_ir::{
    Basis, Bosonic, Error, Fermionic, FiniteTempBasis, LogisticKernel, SVEResult, TworkType,
    compute_sve,
};

use crate::mesh::{IrMesh, MomentumGrid};

/// The singular value expansion shared by both statistics, as
/// `FiniteTempBasisSet` computes it in Python.
///
/// Sharing it halves the setup cost, and it is what makes the fermionic and
/// bosonic τ grids identical — which every susceptibility below relies on.
pub fn sve_for(beta: f64, wmax: f64, eps: f64) -> Result<(LogisticKernel, SVEResult), Error> {
    let kernel = LogisticKernel::new(beta * wmax)?;
    let sve = compute_sve(kernel, Some(eps), None, None, TworkType::Auto)?;
    Ok((kernel, sve))
}

/// The fermionic and bosonic bases of one inverse temperature, and the
/// sampling meshes that go with them.
///
/// This is `FiniteTempBasisSet` of the Python implementation: both statistics
/// out of a single expansion, which is what makes their τ grids identical.
pub struct Bases {
    basis_f: FiniteTempBasis<LogisticKernel, Fermionic>,
    basis_b: FiniteTempBasis<LogisticKernel, Bosonic>,
    mesh_f: IrMesh<Fermionic>,
    mesh_b: IrMesh<Bosonic>,
    /// `u^F_l(0)` and `u^B_l(0)`, the rows that turn a Matsubara sum into an
    /// evaluation.
    uf_at_zero: sparse_ir::Matrix<f64>,
    ub_at_zero: sparse_ir::Matrix<f64>,
    /// The fermionic frequencies `ν = nπ/β` as plain numbers.
    nu: Vec<f64>,
    iw0_f: usize,
    iw0_b: usize,
}

impl Bases {
    pub fn new(beta: f64, wmax: f64, eps: f64) -> Result<Self, Error> {
        let (kernel, sve) = sve_for(beta, wmax, eps)?;
        Self::from_sve(kernel, sve, beta, eps)
    }

    /// The same pair of bases, but from an expansion computed elsewhere.
    ///
    /// A scan that holds `Λ = β ω_max` fixed while `β` varies keeps one
    /// expansion for the whole sweep: the basis functions are the same, only
    /// their scaling with `β` changes. That also keeps the basis the same
    /// size at every temperature, so results can be carried from one step to
    /// the next.
    pub fn from_sve(
        kernel: LogisticKernel,
        sve: SVEResult,
        beta: f64,
        eps: f64,
    ) -> Result<Self, Error> {
        let basis_f = FiniteTempBasis::<LogisticKernel, Fermionic>::from_sve_result(
            kernel,
            beta,
            sve.clone(),
            Some(eps),
            None,
        )?;
        let basis_b = FiniteTempBasis::<LogisticKernel, Bosonic>::from_sve_result(
            kernel,
            beta,
            sve,
            Some(eps),
            None,
        )?;
        let mesh_f = IrMesh::<Fermionic>::new(&basis_f)?;
        let mesh_b = IrMesh::<Bosonic>::new(&basis_b)?;
        // Several of the examples build a quantity from `G` at the fermionic
        // times and then fit it as a bosonic function, which is only
        // meaningful because the two grids agree. They do, because the
        // logistic kernel gives both statistics the same imaginary-time basis
        // functions.
        assert_eq!(
            mesh_f.tau_points(),
            mesh_b.tau_points(),
            "the fermionic and bosonic sampling times must coincide"
        );

        let uf_at_zero = basis_f.evaluate_tau(&[0.0])?;
        let ub_at_zero = basis_b.evaluate_tau(&[0.0])?;
        let nu: Vec<f64> = mesh_f
            .wn()
            .iter()
            .map(|w| w.n() as f64 * std::f64::consts::PI / beta)
            .collect();
        let iw0_f = mesh_f
            .wn()
            .iter()
            .position(|w| w.n() == 1)
            .expect("the fermionic grid contains n = 1");
        let iw0_b = mesh_b
            .wn()
            .iter()
            .position(|w| w.n() == 0)
            .expect("the bosonic grid contains n = 0");
        Ok(Self {
            basis_f,
            basis_b,
            mesh_f,
            mesh_b,
            uf_at_zero,
            ub_at_zero,
            nu,
            iw0_f,
            iw0_b,
        })
    }

    /// `u^F_l(0)`, the row that turns a fermionic Matsubara sum into one
    /// evaluation.
    pub fn uf_at_zero(&self) -> &sparse_ir::Matrix<f64> {
        &self.uf_at_zero
    }

    /// `u^B_l(0)`, the same for a bosonic sum.
    pub fn ub_at_zero(&self) -> &sparse_ir::Matrix<f64> {
        &self.ub_at_zero
    }

    pub fn basis_f(&self) -> &FiniteTempBasis<LogisticKernel, Fermionic> {
        &self.basis_f
    }

    pub fn basis_b(&self) -> &FiniteTempBasis<LogisticKernel, Bosonic> {
        &self.basis_b
    }

    pub fn mesh_f(&self) -> &IrMesh<Fermionic> {
        &self.mesh_f
    }

    pub fn mesh_b(&self) -> &IrMesh<Bosonic> {
        &self.mesh_b
    }

    /// The fermionic frequencies as plain numbers, in grid order.
    pub fn nu(&self) -> &[f64] {
        &self.nu
    }

    /// Index of `ν = πT`, the lowest fermionic frequency.
    pub fn iw0_f(&self) -> usize {
        self.iw0_f
    }

    /// Index of `ν = 0`, the lowest bosonic frequency.
    pub fn iw0_b(&self) -> usize {
        self.iw0_b
    }
}

/// Everything about the lattice and the basis that does not depend on `U`.
///
/// The scan builds one of these and reuses it for every interaction strength;
/// the single-point example builds one and uses it once.
pub struct Lattice {
    grid: MomentumGrid,
    ek: Vec<f64>,
    bases: Bases,
}

impl Lattice {
    /// `nk1 × nk2` square lattice with nearest-neighbour hopping `t`, at
    /// inverse temperature `beta`.
    pub fn new(
        nk1: usize,
        nk2: usize,
        t: f64,
        beta: f64,
        wmax: f64,
        eps: f64,
    ) -> Result<Self, Error> {
        let (kernel, sve) = sve_for(beta, wmax, eps)?;
        Self::from_sve(kernel, sve, nk1, nk2, t, beta, eps)
    }

    /// The same lattice, but from an expansion computed elsewhere.
    pub fn from_sve(
        kernel: LogisticKernel,
        sve: SVEResult,
        nk1: usize,
        nk2: usize,
        t: f64,
        beta: f64,
        eps: f64,
    ) -> Result<Self, Error> {
        let bases = Bases::from_sve(kernel, sve, beta, eps)?;
        let grid = MomentumGrid::new(nk1, nk2);
        let ek = grid.square_lattice_dispersion(t);
        Ok(Self { grid, ek, bases })
    }

    pub fn grid(&self) -> &MomentumGrid {
        &self.grid
    }

    /// The bases and meshes of this lattice.
    pub fn bases(&self) -> &Bases {
        &self.bases
    }

    pub fn uf_at_zero(&self) -> &sparse_ir::Matrix<f64> {
        self.bases.uf_at_zero()
    }

    pub fn ub_at_zero(&self) -> &sparse_ir::Matrix<f64> {
        self.bases.ub_at_zero()
    }

    pub fn dispersion(&self) -> &[f64] {
        &self.ek
    }

    pub fn basis_f(&self) -> &FiniteTempBasis<LogisticKernel, Fermionic> {
        self.bases.basis_f()
    }

    pub fn basis_b(&self) -> &FiniteTempBasis<LogisticKernel, Bosonic> {
        self.bases.basis_b()
    }

    pub fn mesh_f(&self) -> &IrMesh<Fermionic> {
        self.bases.mesh_f()
    }

    pub fn mesh_b(&self) -> &IrMesh<Bosonic> {
        self.bases.mesh_b()
    }

    pub fn nu(&self) -> &[f64] {
        self.bases.nu()
    }

    pub fn iw0_f(&self) -> usize {
        self.bases.iw0_f()
    }

    pub fn iw0_b(&self) -> usize {
        self.bases.iw0_b()
    }

    /// Number of momentum points, `nk1 × nk2`.
    pub fn nk(&self) -> usize {
        self.grid.len()
    }
}

/// The indices of Γ → X → M → Γ on an `nk × nk` grid, in fractional
/// coordinates `(0,0) → (½,0) → (½,½) → (0,0)`.
pub fn high_symmetry_path(nk_lin: usize) -> Vec<usize> {
    let half = nk_lin / 2;
    let mut path = Vec::new();
    for i in 0..=half {
        path.push(i * nk_lin);
    }
    for j in 1..=half {
        path.push(half * nk_lin + j);
    }
    for step in 1..=half {
        let i = half - step;
        path.push(i * nk_lin + i);
    }
    path
}
