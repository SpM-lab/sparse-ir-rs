//! The sum-of-squares parametrization, the objective `Φ` and its
//! Gauss–Newton model.
//!
//! Parameters `θ = [x (M), s (M), Re V (N K M), Im V (N K M)]`, with
//! `ξ_j = x_j - i γ_j`, `γ_j = e^{s_j}`, and the `N x K` factors
//! `R_j = √γ_j V_j`, `R[a + N (c + K j)]`. The scaling keeps the diagonal term
//! `R_j R_j† / γ_j = V_j V_j†` fixed as `γ_j → 0`, so that a pole can approach
//! the real axis (a bath level with weight `V_j V_j†`) without a curved
//! valley in `(s, R)`.
//! The weights are
//!
//! ```text
//! W_j = Σ_k C_jk R_j R_k†,   C_jk = -2i / (ξ_j - ξ_k*),
//! ```
//!
//! so that `Σ_j W_j / (z - ξ_j)` is the Cauchy transform of a positive
//! semidefinite matrix measure (see the module documentation).
//!
//! Channels `x = p + N q` index the `N x N` entries; arrays are column-major.

use super::C64;
use crate::error::{Error, Result};
use std::f64::consts::PI;
use tenferro_einsum::ConcreteEinsumPlan;
use tenferro_tensor::{
    BackendSession, TensorRead, TensorScalar, TensorWrite, TypedTensorView, TypedTensorViewMut,
};

const ZERO: C64 = C64::new(0.0, 0.0);
const I: C64 = C64::new(0.0, 1.0);

/// `⟨(ω - p)^{-a}, (ω - q)^{-b}⟩ = ∫ conj((ω - p)^{-a}) (ω - q)^{-b} dω` over
/// the real axis, `a, b ∈ {1, 2}`, for `p, q` off the real axis. It vanishes
/// for poles in opposite half planes.
pub(super) fn gram(p: C64, a: u8, q: C64, b: u8) -> C64 {
    if (p.im < 0.0) != (q.im < 0.0) {
        return ZERO;
    }
    // Close the contour around `p*`: upwards for `p` in the lower half plane.
    let sign = if p.im < 0.0 { 1.0 } else { -1.0 };
    let tpi = C64::new(0.0, 2.0 * PI);
    let d = p.conj() - q;
    sign * match (a, b) {
        (1, 1) => tpi / d,
        (1, 2) => tpi / (d * d),
        (2, 1) => -tpi / (d * d),
        _ => -2.0 * tpi / (d * d * d),
    }
}

/// Column-major strides of `shape`.
fn col_major(shape: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut s = 1isize;
    for &n in shape {
        strides.push(s);
        s *= n as isize;
    }
    strides
}

fn read<'a>(data: &'a [C64], shape: &[usize]) -> Result<TensorRead<'a>> {
    let view = TypedTensorView::from_slice(shape, col_major(shape), 0, data)?;
    Ok(TensorRead::View(C64::tensor_view(view)))
}

fn write<'a>(data: &'a mut [C64], shape: &[usize]) -> Result<TensorWrite<'a>> {
    let view = TypedTensorViewMut::from_slice(shape, col_major(shape), 0, data)?;
    Ok(TensorWrite::View(C64::tensor_view_mut(view)))
}

fn einsum_error(e: tenferro_einsum::Error) -> Error {
    Error::Tensor {
        reason: e.to_string(),
    }
}

/// A binary contraction prepared once for fixed column-major shapes.
struct Contraction {
    plan: ConcreteEinsumPlan,
    a: Vec<usize>,
    b: Vec<usize>,
    out: Vec<usize>,
}

impl Contraction {
    fn new(subs: &str, a: &[usize], b: &[usize], out: &[usize]) -> Result<Self> {
        let (da, db) = (
            vec![ZERO; a.iter().product()],
            vec![ZERO; b.iter().product()],
        );
        let plan = ConcreteEinsumPlan::prepare_read([read(&da, a)?, read(&db, b)?], subs)
            .map_err(einsum_error)?;
        Ok(Self {
            plan,
            a: a.to_vec(),
            b: b.to_vec(),
            out: out.to_vec(),
        })
    }

    fn run(
        &self,
        session: &mut dyn BackendSession,
        a: &[C64],
        b: &[C64],
        out: &mut [C64],
    ) -> Result<()> {
        self.plan
            .execute_read_into(
                [read(a, &self.a)?, read(b, &self.b)?],
                session,
                write(out, &self.out)?,
            )
            .map_err(einsum_error)
    }
}

/// The fixed data of one refinement.
#[derive(Clone)]
pub(super) struct Problem<'a> {
    /// Number of poles `M`
    pub m: usize,
    /// Matrix size `N` (1 for scalar data)
    pub n: usize,
    /// Rank `K` of the factors
    pub k: usize,
    /// Evaluation points `z_n`, `Im z_n > 0`
    pub z: &'a [C64],
    /// Data `d[n + L x]`
    pub d: &'a [C64],
    /// Height `η > 0` of the line `Im z = η` on which H² distances are
    /// measured
    pub eta: f64,
    /// Reference poles shifted by `-iη`, those in the lower half plane
    pub ref_poles: Vec<C64>,
    /// Their weights `[M_ref, X]`
    pub ref_weights: Vec<C64>,
    /// `‖g_ref‖²` in H² over all reference poles and channels (0 without a
    /// reference)
    pub ref_norm: f64,
    /// Weight of the H² term, `1 / ‖g_ref‖²` for a reference; 0 drops it
    pub h2_weight: f64,
    /// Discrepancy target `ε` (rms over the `L X` data entries)
    pub eps: f64,
}

impl Problem<'_> {
    /// Number of channels `X = N²`
    pub fn channels(&self) -> usize {
        self.n * self.n
    }

    /// Number of real parameters `P`
    pub fn params(&self) -> usize {
        self.m * (2 + 2 * self.n * self.k)
    }
}

/// Value (and optionally derivatives) of `Φ(θ) = F_H / ‖g_ref‖² + c Σ |r|²`
/// with `c = α / (L X ε²)`.
pub(super) struct Eval {
    pub phi: f64,
    /// Relative H² distance squared `F_H / ‖g_ref‖²`
    pub h2: f64,
    /// Rms data residual
    pub rms: f64,
    /// Gradient `[P]`
    pub grad: Vec<f64>,
    /// Gauss–Newton matrix `[P, P]`
    pub gn: Vec<f64>,
}

/// Buffers and prepared contractions, reused across all evaluations.
pub(super) struct Workspace {
    xi: Vec<C64>,
    growth: Vec<f64>,
    r: Vec<C64>,
    c: Vec<C64>,
    h: Vec<C64>,
    y: Vec<C64>,
    s: Vec<C64>,
    pub(super) w: Vec<C64>,
    gamma: Vec<C64>,
    gamma_ref_w: Vec<C64>,
    u: Vec<C64>,
    kext: Vec<C64>,
    kconj: Vec<C64>,
    res: Vec<C64>,
    kr: Vec<C64>,
    kk: Vec<C64>,
    q: Vec<C64>,
    v: Vec<C64>,
    a: Vec<C64>,
    aconj: Vec<C64>,
    t: Vec<C64>,
    gn: Vec<C64>,
    grad: Vec<C64>,
    model: Contraction,
    adjoint: Contraction,
    normal: Contraction,
    apply_q: Contraction,
    gauss_newton: Contraction,
    gradient: Contraction,
}

impl Workspace {
    pub fn new(pb: &Problem) -> Result<Self> {
        let (m, n, k, l) = (pb.m, pb.n, pb.k, pb.z.len());
        let (x, p, b) = (pb.channels(), pb.params(), 2 * pb.m);
        Ok(Self {
            xi: vec![ZERO; m],
            growth: vec![0.0; m],
            r: vec![ZERO; n * k * m],
            c: vec![ZERO; m * m],
            h: vec![ZERO; k * n * m],
            y: vec![ZERO; m * m * x],
            s: vec![ZERO; m * x],
            w: vec![ZERO; m * x],
            gamma: vec![ZERO; b * b],
            gamma_ref_w: vec![ZERO; b * x],
            u: vec![ZERO; b * x],
            kext: vec![ZERO; l * b],
            kconj: vec![ZERO; l * b],
            res: vec![ZERO; l * x],
            kr: vec![ZERO; b * x],
            kk: vec![ZERO; b * b],
            q: vec![ZERO; b * b],
            v: vec![ZERO; b * x],
            a: vec![ZERO; b * x * p],
            aconj: vec![ZERO; b * x * p],
            t: vec![ZERO; b * x * p],
            gn: vec![ZERO; p * p],
            grad: vec![ZERO; p],
            model: Contraction::new("nj,jx->nx", &[l, m], &[m, x], &[l, x])?,
            adjoint: Contraction::new("nb,nx->bx", &[l, b], &[l, x], &[b, x])?,
            normal: Contraction::new("nb,nc->bc", &[l, b], &[l, b], &[b, b])?,
            apply_q: Contraction::new("bc,cxp->bxp", &[b, b], &[b, x, p], &[b, x, p])?,
            gauss_newton: Contraction::new("bxp,bxq->pq", &[b, x, p], &[b, x, p], &[p, p])?,
            gradient: Contraction::new("bxp,bx->p", &[b, x, p], &[b, x], &[p])?,
        })
    }

    /// Poles `ξ` of the last evaluation.
    pub fn poles(&self) -> &[C64] {
        &self.xi
    }

    /// Factors `R[a + N (c + K j)]` of the last evaluation.
    pub fn factors(&self) -> &[C64] {
        &self.r
    }

    /// Evaluate `Φ` at `theta` with multiplier `alpha`; the gradient and the
    /// Gauss–Newton matrix only when `derivatives` is set.
    pub fn eval(
        &mut self,
        pb: &Problem,
        theta: &[f64],
        alpha: f64,
        derivatives: bool,
        session: &mut dyn BackendSession,
    ) -> Result<Eval> {
        let (m, n, k, l) = (pb.m, pb.n, pb.k, pb.z.len());
        let (x, p, nb) = (pb.channels(), pb.params(), 2 * pb.m);
        let nr = n * k * m;
        for j in 0..m {
            self.growth[j] = theta[m + j].exp();
            self.xi[j] = C64::new(theta[j], -self.growth[j]);
        }
        for i in 0..nr {
            let scale = self.growth[i / (n * k)].sqrt();
            self.r[i] = scale * C64::new(theta[2 * m + i], theta[2 * m + nr + i]);
        }
        for kk in 0..m {
            for j in 0..m {
                self.c[j + m * kk] = (self.xi[j] - self.xi[kk].conj()).inv();
            }
        }
        // H_j[c, q] = Σ_k C_jk conj(R_k[q, c]),  W_j = -2i R_j H_j
        self.h.fill(ZERO);
        for j in 0..m {
            for kk in 0..m {
                let cjk = self.c[j + m * kk];
                for q in 0..n {
                    for c in 0..k {
                        self.h[c + k * (q + n * j)] += cjk * self.r[q + n * (c + k * kk)].conj();
                    }
                }
            }
        }
        for j in 0..m {
            for q in 0..n {
                for pp in 0..n {
                    let mut acc = ZERO;
                    for c in 0..k {
                        acc += self.r[pp + n * (c + k * j)] * self.h[c + k * (q + n * j)];
                    }
                    self.w[j + m * (pp + n * q)] = C64::new(0.0, -2.0) * acc;
                }
            }
        }

        // H² part on the line Im z = η, i.e. on the real axis with the poles
        // shifted by -iη, on the extended basis [1/(ω-ξ_j), 1/(ω-ξ_j)²]
        let shift = C64::new(0.0, pb.eta);
        for bc in 0..nb {
            for br in 0..nb {
                self.gamma[br + nb * bc] = gram(
                    self.xi[br % m] - shift,
                    1 + (br / m) as u8,
                    self.xi[bc % m] - shift,
                    1 + (bc / m) as u8,
                );
            }
        }
        self.gamma_ref_w.fill(ZERO);
        let mr = pb.ref_poles.len();
        for (i, &pr) in pb.ref_poles.iter().enumerate() {
            for br in 0..nb {
                let g = gram(self.xi[br % m] - shift, 1 + (br / m) as u8, pr, 1);
                for ch in 0..x {
                    self.gamma_ref_w[br + nb * ch] += g * pb.ref_weights[i + mr * ch];
                }
            }
        }
        // u = Γ [W; 0] - Γ_ref W_ref,  F_H = ‖g_ref‖² + Re Σ conj(W) (u - Γ_ref W_ref)
        let mut fh = pb.ref_norm;
        for ch in 0..x {
            for br in 0..nb {
                let mut acc = -self.gamma_ref_w[br + nb * ch];
                for j in 0..m {
                    acc += self.gamma[br + nb * j] * self.w[j + m * ch];
                }
                self.u[br + nb * ch] = acc;
            }
            for j in 0..m {
                fh += (self.w[j + m * ch].conj()
                    * (self.u[j + nb * ch] - self.gamma_ref_w[j + nb * ch]))
                    .re;
            }
        }
        let fh = fh.max(0.0);

        // Data part
        for b in 0..nb {
            for (i, &zi) in pb.z.iter().enumerate() {
                let f = (zi - self.xi[b % m]).inv();
                let v = if b < m { f } else { f * f };
                self.kext[i + l * b] = v;
                self.kconj[i + l * b] = v.conj();
            }
        }
        self.model
            .run(session, &self.kext[..l * m], &self.w, &mut self.res)?;
        let mut sumsq = 0.0;
        for (r, &d) in self.res.iter_mut().zip(pb.d) {
            *r -= d;
            sumsq += r.norm_sqr();
        }
        let lx = (l * x) as f64;
        let cdata = alpha / (lx * pb.eps * pb.eps);
        let h2 = fh * pb.h2_weight;
        let mut out = Eval {
            phi: h2 + cdata * sumsq,
            h2,
            rms: (sumsq / lx).sqrt(),
            grad: Vec::new(),
            gn: Vec::new(),
        };
        if !derivatives {
            return Ok(out);
        }

        // Q = w Γ + c Kext† Kext,  v = w u + c Kext† r,  w = h2_weight
        self.adjoint
            .run(session, &self.kconj, &self.res, &mut self.kr)?;
        self.normal
            .run(session, &self.kconj, &self.kext, &mut self.kk)?;
        let inv_norm = pb.h2_weight;
        for (qv, (&g, &kk)) in self.q.iter_mut().zip(self.gamma.iter().zip(&self.kk)) {
            *qv = g * inv_norm + kk * cdata;
        }
        for (vv, (&u, &kr)) in self.v.iter_mut().zip(self.u.iter().zip(&self.kr)) {
            *vv = u * inv_norm + kr * cdata;
        }

        self.jacobian(pb);
        self.apply_q.run(session, &self.q, &self.a, &mut self.t)?;
        self.gauss_newton
            .run(session, &self.aconj, &self.t, &mut self.gn)?;
        self.gradient
            .run(session, &self.aconj, &self.v, &mut self.grad)?;
        out.gn = self.gn.iter().map(|g| 2.0 * g.re).collect();
        out.grad = self.grad.iter().map(|g| 2.0 * g.re).collect();
        debug_assert_eq!(out.gn.len(), p * p);
        Ok(out)
    }

    /// `A[b, x, θ]`: derivatives of the extended coefficients `[W; 0]`, i.e.
    /// `dW_j/dθ` (rows `b < M`) and `W_j dξ_j/dθ` (rows `M + j`).
    fn jacobian(&mut self, pb: &Problem) {
        let (m, n, k) = (pb.m, pb.n, pb.k);
        let (x, nb) = (pb.channels(), 2 * pb.m);
        let nrk = n * k * m;
        let at = |b: usize, ch: usize, t: usize| b + nb * (ch + x * t);
        // Y_jm = R_j R_m†,  S_j = Σ_k C_jk² Y_jk
        for ch in 0..x {
            let (pp, q) = (ch % n, ch / n);
            for mm in 0..m {
                for j in 0..m {
                    let mut acc = ZERO;
                    for c in 0..k {
                        acc += self.r[pp + n * (c + k * j)] * self.r[q + n * (c + k * mm)].conj();
                    }
                    self.y[j + m * (mm + m * ch)] = acc;
                }
            }
            for j in 0..m {
                let mut acc = ZERO;
                for kk in 0..m {
                    let cjk = self.c[j + m * kk];
                    acc += cjk * cjk * self.y[j + m * (kk + m * ch)];
                }
                self.s[j + m * ch] = acc;
            }
        }
        self.a.fill(ZERO);
        for ch in 0..x {
            for mm in 0..m {
                let gm = self.growth[mm];
                for j in 0..m {
                    let cjm = self.c[j + m * mm];
                    let cy = cjm * cjm * self.y[j + m * (mm + m * ch)];
                    self.a[at(j, ch, mm)] -= 2.0 * I * cy;
                    self.a[at(j, ch, m + mm)] += 2.0 * gm * cy;
                }
                let sj = self.s[mm + m * ch];
                self.a[at(mm, ch, mm)] += 2.0 * I * sj;
                self.a[at(mm, ch, m + mm)] += 2.0 * gm * sj;
                let wm = self.w[mm + m * ch];
                self.a[at(m + mm, ch, mm)] = wm;
                self.a[at(m + mm, ch, m + mm)] = -I * gm * wm;
            }
        }
        for mm in 0..m {
            for c in 0..k {
                for aa in 0..n {
                    let t_re = 2 * m + aa + n * (c + k * mm);
                    let t_im = t_re + nrk;
                    // δ_jm δ_pa H_j[c, q]
                    for q in 0..n {
                        let hv = self.h[c + k * (q + n * mm)];
                        let ch = aa + n * q;
                        self.a[at(mm, ch, t_re)] += -2.0 * I * hv;
                        self.a[at(mm, ch, t_im)] += 2.0 * hv;
                    }
                    // C_jm R_j[p, c] δ_qa
                    for j in 0..m {
                        let cjm = self.c[j + m * mm];
                        for pp in 0..n {
                            let cr = cjm * self.r[pp + n * (c + k * j)];
                            let ch = pp + n * aa;
                            self.a[at(j, ch, t_re)] += -2.0 * I * cr;
                            self.a[at(j, ch, t_im)] += -2.0 * cr;
                        }
                    }
                }
            }
        }
        // Chain rule for R_j = √γ_j V_j: dR/ds_j = R_j / 2, dR/dV_j = √γ_j.
        for mm in 0..m {
            let sq = self.growth[mm].sqrt();
            let col_s = m + mm;
            for e in n * k * mm..n * k * (mm + 1) {
                let (t_re, t_im) = (2 * m + e, 2 * m + nrk + e);
                let r = self.r[e];
                for ch in 0..x {
                    for b in 0..nb {
                        let d =
                            0.5 * (r.re * self.a[at(b, ch, t_re)] + r.im * self.a[at(b, ch, t_im)]);
                        self.a[at(b, ch, col_s)] += d;
                        self.a[at(b, ch, t_re)] *= sq;
                        self.a[at(b, ch, t_im)] *= sq;
                    }
                }
            }
        }
        for (ac, &a) in self.aconj.iter_mut().zip(&self.a) {
            *ac = a.conj();
        }
    }
}
