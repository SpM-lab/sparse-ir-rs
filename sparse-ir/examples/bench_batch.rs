//! Batched fit/evaluate throughput benchmark.
//!
//! Rust port of the Fortran `fortran/test/test_timing.f90` benchmark. The
//! same Matsubara Green's function `G(iν) = 1/(iν - ω0)` is replicated over
//! `lsize` batch rows and pushed through
//! `fit_matsubara -> evaluate_tau -> fit_tau -> evaluate_matsubara`
//! `num / lsize` times, on column-major `[lsize, npoints]` arrays transformed
//! along axis 1 (the Fortran `TARGET_DIM = 2` layout), so each call has
//! `pre = lsize, post = 1`. The calls go through [`InplaceFitter`] with caller-owned
//! buffers, exactly as `spir_sampling_{eval,fit}_*` do for the C/Fortran API.
//!
//! Run a single configuration (same positional arguments as the Fortran code):
//!
//! ```text
//! cargo run --release --example bench_batch -- \
//!     [nlambda] [ndigit] [positive_only] [statistics F|B] [lreal_ir] [lreal_tau] [num] [lsize]
//! ```
//!
//! With no arguments (or `--sweep`) the Fortran "short" pattern set is run for
//! `lsize = 1, 10, 120`. Pin threads externally (`RAYON_NUM_THREADS=1`).
//!
//! `--blas` injects the system Fortran BLAS (`dgemm_`/`zgemm_`, Accelerate on
//! macOS) as the GEMM backend, as the Fortran wrapper does via
//! `spir_gemm_backend_new_from_fblas_lp64`; the default is the built-in faer
//! backend (`backend = None`).
//!
//! `--engine=einsum|plan|plan1s` replaces the in-house fitters with the same
//! pseudoinverse two-step contractions written as tenferro-einsum calls
//! (`einsum_read_into` from a string, a prepared `ConcreteEinsumPlan` with one
//! backend session per call, or one session per timed loop). The
//! positive-only/complex-coefficient patterns are skipped for these engines.
//! tenferro-cpu uses faer unless built with
//! `--features tenferro-cpu/blas-accelerate` (or another BLAS provider).

use num_complex::Complex;
use sparse_ir::gemm::{DgemmFnPtr, ExternalBlasBackend, GemmBackendHandle, ZgemmFnPtr};
use sparse_ir::{
    Bosonic, Fermionic, FiniteTempBasis, InplaceFitter, LogisticKernel, MatsubaraSampling,
    MatsubaraSamplingPositiveOnly, StatisticsType, TauSampling, TypedTensorView,
    TypedTensorViewMut,
};
use std::time::Instant;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{ConcreteEinsumPlan, TypedTensorReadEinsumIntoExt};
use tenferro_linalg::TypedTensorLinalgExt;
use tenferro_tensor::{
    BackendSession, BackendSessionHost, TensorRead, TensorScalar, TensorWrite, TypedTensor,
};

type C64 = Complex<f64>;

unsafe extern "C" {
    fn dgemm_();
    fn zgemm_();
}

/// GEMM backend backed by the system Fortran BLAS linked into `sparse-ir`.
fn fblas_backend() -> GemmBackendHandle {
    // SAFETY: `dgemm_`/`zgemm_` are the reference Fortran BLAS entry points,
    // whose ABI is exactly `DgemmFnPtr`/`ZgemmFnPtr` (LP64).
    let (d, z) = unsafe {
        (
            std::mem::transmute::<unsafe extern "C" fn(), DgemmFnPtr>(dgemm_),
            std::mem::transmute::<unsafe extern "C" fn(), ZgemmFnPtr>(zgemm_),
        )
    };
    GemmBackendHandle::new(Box::new(ExternalBlasBackend::new(d, z)))
}

#[derive(Clone, Copy)]
struct Config {
    nlambda: i32,
    ndigit: i32,
    positive_only: bool,
    fermionic: bool,
    lreal_ir: bool,
    lreal_tau: bool,
    num: usize,
    lsize: usize,
}

#[derive(Default)]
struct Timings {
    fit_matsu: f64,
    eval_tau: f64,
    fit_tau: f64,
    eval_matsu: f64,
}

impl Timings {
    fn total(&self) -> f64 {
        self.fit_matsu + self.eval_tau + self.fit_tau + self.eval_matsu
    }
}

fn view<'a, T: 'static>(data: &'a [T], shape: &[usize; 2]) -> TypedTensorView<'a, T> {
    TypedTensorView::from_slice(shape, [1, shape[0] as isize], 0, data).unwrap()
}

fn view_mut<'a, T: 'static>(data: &'a mut [T], shape: &[usize; 2]) -> TypedTensorViewMut<'a, T> {
    TypedTensorViewMut::from_slice(shape, [1, shape[0] as isize], 0, data).unwrap()
}

/// Time `iters` repetitions of `f` in seconds, after one untimed warm-up call
/// (which absorbs the lazy pseudoinverse SVD of the in-house fitters).
fn time_loop(iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64()
}

/// Real IR coefficients and real tau values (Fortran `dd`/`dc` loops):
/// fit_matsubara zd, evaluate_tau dd, fit_tau dd, evaluate_matsubara dz.
fn loop_real(
    backend: Option<&GemmBackendHandle>,
    tau: &dyn InplaceFitter,
    mats: &dyn InplaceFitter,
    iters: usize,
    lsize: usize,
    giv: &[C64],
    giv_reconst: &mut [C64],
) -> Timings {
    let (nfreq, l, ntau) = (mats.n_points(), mats.basis_size(), tau.n_points());
    let mut gl = vec![0.0; lsize * l];
    let mut gtau = vec![0.0; lsize * ntau];
    let mut t = Timings::default();
    t.fit_matsu = time_loop(iters, || {
        mats.fit_nd_zd_to(
            backend,
            &view(giv, &[lsize, nfreq]),
            1,
            &mut view_mut(&mut gl, &[lsize, l]),
        )
        .unwrap();
    });
    t.eval_tau = time_loop(iters, || {
        tau.evaluate_nd_dd_to(
            backend,
            &view(&gl, &[lsize, l]),
            1,
            &mut view_mut(&mut gtau, &[lsize, ntau]),
        )
        .unwrap();
    });
    t.fit_tau = time_loop(iters, || {
        tau.fit_nd_dd_to(
            backend,
            &view(&gtau, &[lsize, ntau]),
            1,
            &mut view_mut(&mut gl, &[lsize, l]),
        )
        .unwrap();
    });
    t.eval_matsu = time_loop(iters, || {
        mats.evaluate_nd_dz_to(
            backend,
            &view(&gl, &[lsize, l]),
            1,
            &mut view_mut(giv_reconst, &[lsize, nfreq]),
        )
        .unwrap();
    });
    t
}

/// Complex IR coefficients and complex tau values (Fortran `cd`/`cc` loops):
/// all four transforms are zz.
fn loop_complex(
    backend: Option<&GemmBackendHandle>,
    tau: &dyn InplaceFitter,
    mats: &dyn InplaceFitter,
    iters: usize,
    lsize: usize,
    giv: &[C64],
    giv_reconst: &mut [C64],
) -> Timings {
    let (nfreq, l, ntau) = (mats.n_points(), mats.basis_size(), tau.n_points());
    let mut gl = vec![C64::new(0.0, 0.0); lsize * l];
    let mut gtau = vec![C64::new(0.0, 0.0); lsize * ntau];
    let mut t = Timings::default();
    t.fit_matsu = time_loop(iters, || {
        mats.fit_nd_zz_to(
            backend,
            &view(giv, &[lsize, nfreq]),
            1,
            &mut view_mut(&mut gl, &[lsize, l]),
        )
        .unwrap();
    });
    t.eval_tau = time_loop(iters, || {
        tau.evaluate_nd_zz_to(
            backend,
            &view(&gl, &[lsize, l]),
            1,
            &mut view_mut(&mut gtau, &[lsize, ntau]),
        )
        .unwrap();
    });
    t.fit_tau = time_loop(iters, || {
        tau.fit_nd_zz_to(
            backend,
            &view(&gtau, &[lsize, ntau]),
            1,
            &mut view_mut(&mut gl, &[lsize, l]),
        )
        .unwrap();
    });
    t.eval_matsu = time_loop(iters, || {
        mats.evaluate_nd_zz_to(
            backend,
            &view(&gl, &[lsize, l]),
            1,
            &mut view_mut(giv_reconst, &[lsize, nfreq]),
        )
        .unwrap();
    });
    t
}

fn run<S: StatisticsType + 'static>(
    cfg: Config,
    backend: Option<&GemmBackendHandle>,
    engine: Engine,
) {
    let lambda = 10f64.powi(cfg.nlambda);
    let beta = 100.0;
    let omega0 = 1.0 / beta;
    let eps = 0.1f64.powi(cfg.ndigit);

    let basis = FiniteTempBasis::<LogisticKernel, S>::new(
        LogisticKernel::new(lambda),
        beta,
        Some(eps),
        None,
    );
    let tau = TauSampling::<S>::new(&basis);
    let (mats, mats_a, freqs): (Box<dyn InplaceFitter>, Vec<C64>, Vec<C64>) = if cfg.positive_only {
        let s = MatsubaraSamplingPositiveOnly::<S>::new(&basis);
        let a = s.matrix().host_data().unwrap().to_vec();
        let f = s
            .sampling_points()
            .iter()
            .map(|w| w.value_imaginary(beta))
            .collect();
        (Box::new(s), a, f)
    } else {
        let s = MatsubaraSampling::<S>::new(&basis);
        let a = s.matrix().host_data().unwrap().to_vec();
        let f = s
            .sampling_points()
            .iter()
            .map(|w| w.value_imaginary(beta))
            .collect();
        (Box::new(s), a, f)
    };
    let nfreq = freqs.len();
    // Smallest positive frequency (skipping 0 for bosons), as in the Fortran code.
    let freq_1_idx = match (cfg.positive_only, cfg.fermionic) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => nfreq / 2,
        (false, false) => nfreq / 2 + 1,
    };

    let lsize = cfg.lsize;
    let iters = cfg.num / lsize;
    let mut giv = vec![C64::new(0.0, 0.0); lsize * nfreq];
    for (n, &iw) in freqs.iter().enumerate() {
        giv[n * lsize..(n + 1) * lsize].fill(1.0 / (iw - omega0));
    }
    let mut giv_reconst = vec![C64::new(0.0, 0.0); lsize * nfreq];

    let real_path = cfg.positive_only && cfg.lreal_ir;
    let dims = (nfreq, basis.size(), tau.n_points());
    let tau_a = tau.matrix().host_data().unwrap();
    let t = if let Engine::Einsum(mode) = engine {
        if real_path {
            einsum_loop_real(
                mode,
                tau_a,
                &mats_a,
                dims,
                iters,
                lsize,
                &giv,
                &mut giv_reconst,
            )
        } else if !cfg.positive_only {
            einsum_loop_complex(
                mode,
                tau_a,
                &mats_a,
                dims,
                iters,
                lsize,
                &giv,
                &mut giv_reconst,
            )
        } else {
            // Positive-only with complex IR coefficients is not ported.
            return;
        }
    } else if real_path {
        loop_real(
            backend,
            &tau,
            mats.as_ref(),
            iters,
            lsize,
            &giv,
            &mut giv_reconst,
        )
    } else {
        loop_complex(
            backend,
            &tau,
            mats.as_ref(),
            iters,
            lsize,
            &giv,
            &mut giv_reconst,
        )
    };

    let expected = 1.0 / (freqs[freq_1_idx] - omega0);
    let computed = giv_reconst[freq_1_idx * lsize];
    let rel_err = ((computed.re + computed.im) - (expected.re + expected.im)).abs()
        / (expected.re + expected.im).abs();
    assert!(
        rel_err <= 1e2 * eps,
        "relative error {rel_err:e} exceeds tolerance {:e}",
        1e2 * eps
    );

    let flag = |b: bool| if b { 'T' } else { 'F' };
    println!(
        "{:>3} {:>3} {:>4} {:>4} {:>4} {:>4} {:>8} {:>6} {:>5} {:>5} {:>5} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>12.4e} {:>10.2e}",
        cfg.nlambda,
        cfg.ndigit,
        flag(cfg.positive_only),
        if cfg.fermionic { 'F' } else { 'B' },
        flag(cfg.lreal_ir),
        flag(cfg.lreal_tau),
        iters * lsize,
        lsize,
        basis.size(),
        nfreq,
        tau.n_points(),
        t.fit_matsu,
        t.eval_tau,
        t.fit_tau,
        t.eval_matsu,
        t.total(),
        t.total() / (iters * lsize) as f64,
        rel_err,
    );
}

fn run_cfg(cfg: Config, backend: Option<&GemmBackendHandle>, engine: Engine) {
    if cfg.fermionic {
        run::<Fermionic>(cfg, backend, engine)
    } else {
        run::<Bosonic>(cfg, backend, engine)
    }
}

fn parse_bool(s: &str) -> bool {
    matches!(s, "T" | "t" | "true" | ".true." | "1")
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let blas = args.iter().any(|a| a == "--blas");
    args.retain(|a| a != "--blas");
    let engine = args
        .iter()
        .find_map(|a| a.strip_prefix("--engine="))
        .map_or(Engine::Inhouse, Engine::parse);
    args.retain(|a| !a.starts_with("--engine="));
    let handle = blas.then(fblas_backend);
    let backend = handle.as_ref();
    if engine == Engine::Inhouse {
        println!(
            "# engine: inhouse, GEMM backend: {}",
            if blas {
                "system BLAS (injected)"
            } else {
                "faer (default)"
            }
        );
    } else {
        println!("# engine: {}", engine.name());
    }
    println!(
        "{:>3} {:>3} {:>4} {:>4} {:>4} {:>4} {:>8} {:>6} {:>5} {:>5} {:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}",
        "nλ",
        "nd",
        "pos",
        "stat",
        "rIR",
        "rTau",
        "num",
        "lsize",
        "L",
        "nfreq",
        "ntau",
        "fitM(s)",
        "evalT(s)",
        "fitT(s)",
        "evalM(s)",
        "total(s)",
        "per_vec(s)",
        "rel_err",
    );
    if args.is_empty() || args[0] == "--sweep" {
        // Fortran run_timing_benchmark.sh "short" patterns.
        let patterns = [
            (true, true, true, true),
            (true, true, false, false),
            (false, true, false, false),
            (true, false, true, true),
            (true, false, false, false),
            (false, false, false, false),
        ];
        for (positive_only, fermionic, lreal_ir, lreal_tau) in patterns {
            for lsize in [1, 10, 120] {
                run_cfg(
                    Config {
                        nlambda: 6,
                        ndigit: 8,
                        positive_only,
                        fermionic,
                        lreal_ir,
                        lreal_tau,
                        num: 185_640,
                        lsize,
                    },
                    backend,
                    engine,
                );
            }
        }
        return;
    }
    let get = |i: usize, default: &str| args.get(i).cloned().unwrap_or_else(|| default.to_string());
    let cfg = Config {
        nlambda: get(0, "6").parse().unwrap(),
        ndigit: get(1, "8").parse().unwrap(),
        positive_only: parse_bool(&get(2, "T")),
        fermionic: get(3, "F") != "B",
        lreal_ir: parse_bool(&get(4, "T")),
        lreal_tau: parse_bool(&get(5, "T")),
        num: get(6, "185640").parse().unwrap(),
        lsize: get(7, "1").parse().unwrap(),
    };
    assert!(
        cfg.num % cfg.lsize == 0,
        "num ({}) must be divisible by lsize ({})",
        cfg.num,
        cfg.lsize
    );
    run_cfg(cfg, backend, engine);
}

// ---------------------------------------------------------------------------
// tenferro-einsum engine
// ---------------------------------------------------------------------------

/// How the tenferro-einsum engine drives each contraction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EinsumMode {
    /// `einsum_read_into` from a subscript string, one backend session per call.
    String,
    /// `ConcreteEinsumPlan` prepared once, one backend session per call.
    Plan,
    /// `ConcreteEinsumPlan` prepared once, one backend session per timed loop.
    PlanOneSession,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Engine {
    /// sparse-ir's `InplaceFitter` (built-in faer or injected BLAS GEMM).
    Inhouse,
    Einsum(EinsumMode),
}

impl Engine {
    fn parse(s: &str) -> Self {
        match s {
            "inhouse" => Engine::Inhouse,
            "einsum" => Engine::Einsum(EinsumMode::String),
            "plan" => Engine::Einsum(EinsumMode::Plan),
            "plan1s" => Engine::Einsum(EinsumMode::PlanOneSession),
            _ => panic!("unknown engine {s:?} (inhouse|einsum|plan|plan1s)"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Engine::Inhouse => "inhouse",
            Engine::Einsum(EinsumMode::String) => "einsum (string, session per call)",
            Engine::Einsum(EinsumMode::Plan) => "einsum (prepared plan, session per call)",
            Engine::Einsum(EinsumMode::PlanOneSession) => {
                "einsum (prepared plan, one session per loop)"
            }
        }
    }
}

/// Scalars the einsum engine contracts.
trait EScalar:
    tenferro_linalg::LinalgScalar
    + TensorScalar<Real = f64>
    + num_traits::Zero
    + Copy
    + Send
    + Sync
    + 'static
{
    fn conj_(self) -> Self;
    fn scale_(self, s: f64) -> Self;
}

impl EScalar for f64 {
    fn conj_(self) -> Self {
        self
    }
    fn scale_(self, s: f64) -> Self {
        self * s
    }
}

impl EScalar for C64 {
    fn conj_(self) -> Self {
        self.conj()
    }
    fn scale_(self, s: f64) -> Self {
        self * s
    }
}

fn col_major(shape: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut s = 1isize;
    for &n in shape {
        strides.push(s);
        s *= n as isize;
    }
    strides
}

fn tv<'a, T: 'static>(data: &'a [T], shape: &[usize]) -> TypedTensorView<'a, T> {
    TypedTensorView::from_slice(shape, col_major(shape), 0, data).unwrap()
}

fn tv_mut<'a, T: 'static>(data: &'a mut [T], shape: &[usize]) -> TypedTensorViewMut<'a, T> {
    TypedTensorViewMut::from_slice(shape, col_major(shape), 0, data).unwrap()
}

/// A complex column-major `[p, ...]` buffer as a real `[2, p, ...]` buffer.
fn as_real(x: &[C64]) -> &[f64] {
    // SAFETY: `Complex<f64>` is `repr(C)` `{ re, im }`.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), 2 * x.len()) }
}

fn as_real_mut(x: &mut [C64]) -> &mut [f64] {
    // SAFETY: as in `as_real`.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), 2 * x.len()) }
}

/// Pseudoinverse `A^+ = V diag(1/s) U^H` of a column-major `n x m` matrix as
/// `(uh: r x n, v_scaled: m x r)`, via the tenferro-linalg SVD.
fn pinv<T: EScalar>(a: &[T], n: usize, m: usize) -> (Vec<T>, Vec<T>, usize) {
    let tensor = TypedTensor::<T>::from_vec_col_major(vec![n, m], a.to_vec()).unwrap();
    let (u, s, vt) = CpuBackend::new()
        .with_backend_session(|session| tensor.svd(session))
        .unwrap();
    let (ldu, ldvt) = (u.shape()[0], vt.shape()[0]);
    let (u, s, vt) = (
        u.host_data().unwrap(),
        s.host_data().unwrap(),
        vt.host_data().unwrap(),
    );
    let r = n.min(m);
    let mut uh = vec![T::zero(); r * n];
    for i in 0..n {
        for l in 0..r {
            uh[l + r * i] = u[i + ldu * l].conj_();
        }
    }
    let mut vs = vec![T::zero(); m * r];
    for l in 0..r {
        let inv = 1.0 / s[l];
        for j in 0..m {
            vs[j + m * l] = vt[l + ldvt * j].conj_().scale_(inv);
        }
    }
    (uh, vs, r)
}

/// One binary contraction `out = einsum(subs, a, b)`.
struct Op {
    subs: &'static str,
    plan: Option<ConcreteEinsumPlan>,
}

impl Op {
    fn new<T: EScalar>(
        mode: EinsumMode,
        subs: &'static str,
        a: TypedTensorView<'_, T>,
        b: TypedTensorView<'_, T>,
    ) -> Self {
        let plan = (mode != EinsumMode::String).then(|| {
            ConcreteEinsumPlan::prepare_read(
                [
                    TensorRead::View(T::tensor_view(a)),
                    TensorRead::View(T::tensor_view(b)),
                ],
                subs,
            )
            .unwrap()
        });
        Self { subs, plan }
    }

    fn run<T: EScalar>(
        &self,
        session: &mut dyn BackendSession,
        a: TypedTensorView<'_, T>,
        b: TypedTensorView<'_, T>,
        out: TypedTensorViewMut<'_, T>,
    ) {
        match &self.plan {
            None => [a, b].einsum_read_into(self.subs, session, out).unwrap(),
            Some(plan) => plan
                .execute_read_into(
                    [
                        TensorRead::View(T::tensor_view(a)),
                        TensorRead::View(T::tensor_view(b)),
                    ],
                    session,
                    TensorWrite::View(T::tensor_view_mut(out)),
                )
                .unwrap(),
        }
    }
}

/// Time `iters` repetitions of `body`, opening backend sessions per `mode`.
fn time_session_loop(
    host: &mut CpuBackend,
    mode: EinsumMode,
    iters: usize,
    mut body: impl FnMut(&mut dyn BackendSession) + Send,
) -> f64 {
    if mode == EinsumMode::PlanOneSession {
        host.with_backend_session(|s| time_loop(iters, || body(s)))
    } else {
        time_loop(iters, || host.with_backend_session(|s| body(s)))
    }
}

/// Positive-only Matsubara with real IR coefficients, as `loop_real`.
///
/// The complex `n x L` Matsubara matrix `A` is the real `[n, 2, L]` tensor
/// `B[j, c, l]` (`c` = re/im), i.e. the stacked `[Re A; Im A]`. Complex data
/// `[p, n]` is read as real `[2, p, n]`.
#[allow(clippy::too_many_arguments)]
fn einsum_loop_real(
    mode: EinsumMode,
    tau_a: &[f64],
    mats_a: &[C64],
    (nfreq, l, ntau): (usize, usize, usize),
    iters: usize,
    p: usize,
    giv: &[C64],
    giv_reconst: &mut [C64],
) -> Timings {
    let mut host = CpuBackend::with_threads(1).unwrap();
    let mut b = vec![0.0; 2 * nfreq * l];
    for (k, z) in mats_a.iter().enumerate() {
        let (j, ll) = (k % nfreq, k / nfreq);
        b[j + 2 * nfreq * ll] = z.re;
        b[j + nfreq + 2 * nfreq * ll] = z.im;
    }
    let (uhb, vsb, rb) = pinv(&b, 2 * nfreq, l);
    let (uht, vst, rt) = pinv(tau_a, ntau, l);
    let mut gl = vec![0.0; p * l];
    let mut gtau = vec![0.0; p * ntau];
    let mut tmp = vec![0.0; p * rb.max(rt)];
    let giv_r = as_real(giv);

    let fit_m1 = Op::new(
        mode,
        "rjc,cpj->pr",
        tv(&uhb, &[rb, nfreq, 2]),
        tv(giv_r, &[2, p, nfreq]),
    );
    let fit_m2 = Op::new(mode, "lr,pr->pl", tv(&vsb, &[l, rb]), tv(&tmp, &[p, rb]));
    let eval_t = Op::new(mode, "ij,pj->pi", tv(tau_a, &[ntau, l]), tv(&gl, &[p, l]));
    let fit_t1 = Op::new(
        mode,
        "rj,pj->pr",
        tv(&uht, &[rt, ntau]),
        tv(&gtau, &[p, ntau]),
    );
    let fit_t2 = Op::new(mode, "lr,pr->pl", tv(&vst, &[l, rt]), tv(&tmp, &[p, rt]));
    let eval_m = Op::new(
        mode,
        "jcl,pl->cpj",
        tv(&b, &[nfreq, 2, l]),
        tv(&gl, &[p, l]),
    );

    let mut t = Timings::default();
    t.fit_matsu = time_session_loop(&mut host, mode, iters, |s| {
        fit_m1.run(
            s,
            tv(&uhb, &[rb, nfreq, 2]),
            tv(giv_r, &[2, p, nfreq]),
            tv_mut(&mut tmp[..p * rb], &[p, rb]),
        );
        fit_m2.run(
            s,
            tv(&vsb, &[l, rb]),
            tv(&tmp[..p * rb], &[p, rb]),
            tv_mut(&mut gl, &[p, l]),
        );
    });
    t.eval_tau = time_session_loop(&mut host, mode, iters, |s| {
        eval_t.run(
            s,
            tv(tau_a, &[ntau, l]),
            tv(&gl, &[p, l]),
            tv_mut(&mut gtau, &[p, ntau]),
        );
    });
    t.fit_tau = time_session_loop(&mut host, mode, iters, |s| {
        fit_t1.run(
            s,
            tv(&uht, &[rt, ntau]),
            tv(&gtau, &[p, ntau]),
            tv_mut(&mut tmp[..p * rt], &[p, rt]),
        );
        fit_t2.run(
            s,
            tv(&vst, &[l, rt]),
            tv(&tmp[..p * rt], &[p, rt]),
            tv_mut(&mut gl, &[p, l]),
        );
    });
    let out = as_real_mut(giv_reconst);
    t.eval_matsu = time_session_loop(&mut host, mode, iters, |s| {
        eval_m.run(
            s,
            tv(&b, &[nfreq, 2, l]),
            tv(&gl, &[p, l]),
            tv_mut(out, &[2, p, nfreq]),
        );
    });
    t
}

/// Full Matsubara with complex IR coefficients, as `loop_complex`.
///
/// Matsubara transforms are complex contractions; the real tau matrix acts on
/// complex data read as real `[2, p, n]`.
#[allow(clippy::too_many_arguments)]
fn einsum_loop_complex(
    mode: EinsumMode,
    tau_a: &[f64],
    mats_a: &[C64],
    (nfreq, l, ntau): (usize, usize, usize),
    iters: usize,
    p: usize,
    giv: &[C64],
    giv_reconst: &mut [C64],
) -> Timings {
    let mut host = CpuBackend::with_threads(1).unwrap();
    let (uhm, vsm, rm) = pinv(mats_a, nfreq, l);
    let (uht, vst, rt) = pinv(tau_a, ntau, l);
    let zero = C64::new(0.0, 0.0);
    let mut gl = vec![zero; p * l];
    let mut gtau = vec![zero; p * ntau];
    let mut tmp = vec![zero; p * rm.max(rt)];

    let fit_m1 = Op::new(
        mode,
        "rj,pj->pr",
        tv(&uhm, &[rm, nfreq]),
        tv(giv, &[p, nfreq]),
    );
    let fit_m2 = Op::new(mode, "lr,pr->pl", tv(&vsm, &[l, rm]), tv(&tmp, &[p, rm]));
    let eval_t = Op::new(
        mode,
        "ij,cpj->cpi",
        tv(tau_a, &[ntau, l]),
        tv(as_real(&gl), &[2, p, l]),
    );
    let fit_t1 = Op::new(
        mode,
        "rj,cpj->cpr",
        tv(&uht, &[rt, ntau]),
        tv(as_real(&gtau), &[2, p, ntau]),
    );
    let fit_t2 = Op::new(
        mode,
        "lr,cpr->cpl",
        tv(&vst, &[l, rt]),
        tv(as_real(&tmp[..p * rt]), &[2, p, rt]),
    );
    let eval_m = Op::new(mode, "jl,pl->pj", tv(mats_a, &[nfreq, l]), tv(&gl, &[p, l]));

    let mut t = Timings::default();
    t.fit_matsu = time_session_loop(&mut host, mode, iters, |s| {
        fit_m1.run(
            s,
            tv(&uhm, &[rm, nfreq]),
            tv(giv, &[p, nfreq]),
            tv_mut(&mut tmp[..p * rm], &[p, rm]),
        );
        fit_m2.run(
            s,
            tv(&vsm, &[l, rm]),
            tv(&tmp[..p * rm], &[p, rm]),
            tv_mut(&mut gl, &[p, l]),
        );
    });
    t.eval_tau = time_session_loop(&mut host, mode, iters, |s| {
        eval_t.run(
            s,
            tv(tau_a, &[ntau, l]),
            tv(as_real(&gl), &[2, p, l]),
            tv_mut(as_real_mut(&mut gtau), &[2, p, ntau]),
        );
    });
    t.fit_tau = time_session_loop(&mut host, mode, iters, |s| {
        fit_t1.run(
            s,
            tv(&uht, &[rt, ntau]),
            tv(as_real(&gtau), &[2, p, ntau]),
            tv_mut(as_real_mut(&mut tmp[..p * rt]), &[2, p, rt]),
        );
        fit_t2.run(
            s,
            tv(&vst, &[l, rt]),
            tv(as_real(&tmp[..p * rt]), &[2, p, rt]),
            tv_mut(as_real_mut(&mut gl), &[2, p, l]),
        );
    });
    t.eval_matsu = time_session_loop(&mut host, mode, iters, |s| {
        eval_m.run(
            s,
            tv(mats_a, &[nfreq, l]),
            tv(&gl, &[p, l]),
            tv_mut(giv_reconst, &[p, nfreq]),
        );
    });
    t
}
