//! Parity with the reference implementation Green-Phys/MiniPole.
//!
//! The fixtures in `tests/reference/` hold inputs and outputs of the
//! reference (commit 15e4a54), written by `tests/reference/gen_reference.py`.

use num_complex::Complex;
use sparse_ir::mpm::{
    ErrType, Esprit, EspritParams, MiniPoleDlrParams, MiniPoleParams, MiniPoleResult, N0, Plane,
    mini_pole, mini_pole_dlr,
};
use std::collections::HashMap;
use tenferro_tensor::TypedTensor;

type C64 = Complex<f64>;

enum Arr {
    R(Vec<f64>),
    C(Vec<C64>),
    I(Vec<i64>),
}

struct Case(HashMap<String, Arr>);

impl Case {
    fn load(name: &str) -> Self {
        let path = format!("{}/tests/reference/{name}.txt", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut map = HashMap::new();
        for line in text.lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            let n: usize = t[2].parse().unwrap();
            let v = &t[3..];
            let arr = match t[1] {
                "r" => Arr::R(v.iter().map(|x| x.parse().unwrap()).collect()),
                "i" => Arr::I(v.iter().map(|x| x.parse().unwrap()).collect()),
                "c" => Arr::C(
                    (0..n)
                        .map(|i| C64::new(v[2 * i].parse().unwrap(), v[2 * i + 1].parse().unwrap()))
                        .collect(),
                ),
                k => panic!("kind {k}"),
            };
            map.insert(t[0].to_string(), arr);
        }
        Case(map)
    }
    fn has(&self, k: &str) -> bool {
        self.0.contains_key(k)
    }
    fn r(&self, k: &str) -> Vec<f64> {
        match &self.0[k] {
            Arr::R(v) => v.clone(),
            Arr::I(v) => v.iter().map(|&x| x as f64).collect(),
            Arr::C(_) => panic!("{k} is complex"),
        }
    }
    fn c(&self, k: &str) -> Vec<C64> {
        match &self.0[k] {
            Arr::C(v) => v.clone(),
            Arr::R(v) => v.iter().map(|&x| C64::new(x, 0.0)).collect(),
            Arr::I(v) => v.iter().map(|&x| C64::new(x as f64, 0.0)).collect(),
        }
    }
    fn i(&self, k: &str) -> i64 {
        match &self.0[k] {
            Arr::I(v) => v[0],
            _ => panic!("{k} is not an integer"),
        }
    }
    fn opt_f(&self, k: &str) -> Option<f64> {
        self.has(k).then(|| self.r(k)[0])
    }
    fn opt_i(&self, k: &str) -> Option<i64> {
        self.has(k).then(|| self.i(k))
    }
}

/// C-order `(n, rest...)` data to column-major `[n, rest]` with `rest`
/// flattened in C order, as channel `c` of the reference.
fn c_to_colmajor(v: &[C64], n: usize) -> Vec<C64> {
    let d = v.len() / n;
    let mut out = vec![C64::new(0.0, 0.0); v.len()];
    for i in 0..n {
        for c in 0..d {
            out[i + n * c] = v[i * d + c];
        }
    }
    out
}

/// Channel of the reference (C order `a n + b`) to ours (`a + n b`).
fn chan(c_ref: usize, n_orb: usize) -> usize {
    (c_ref / n_orb) + n_orb * (c_ref % n_orb)
}

fn max_diff(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

fn check_esprit(name: &str, tol: f64) {
    let case = Case::load(name);
    let h = case.c("h");
    // omega is M x d.
    let dim = case.c("omega").len() / case.i("M") as usize;
    let n = h.len() / dim;
    let p = EspritParams {
        err: case.opt_f("arg_err"),
        err_type: if case.opt_i("arg_err_type") == Some(1) {
            ErrType::Rel
        } else {
            ErrType::Abs
        },
        m: case.opt_i("arg_M").map(|m| m as usize),
        lfactor: case.opt_f("arg_Lfactor").unwrap_or(0.4),
        ..EspritParams::default()
    };
    let e = Esprit::new(&c_to_colmajor(&h, n), n, dim, &p).unwrap();
    assert_eq!(e.m as i64, case.i("M"), "{name}: M");
    let s_ref = case.r("S");
    let ds =
        e.s.iter()
            .zip(&s_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
    // Match nodes by nearest neighbour.
    let g_ref = case.c("gamma");
    let dg = g_ref
        .iter()
        .map(|g| {
            e.gamma
                .iter()
                .map(|x| (x - g).norm())
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max);
    eprintln!(
        "{name}: |dS| {ds:.1e} |dgamma| {dg:.1e} err_max {:.3e} vs {:.3e}",
        e.err_max,
        case.r("err_max")[0]
    );
    assert!(ds < tol * s_ref[0] && dg < tol, "{name}");
}

#[test]
fn esprit_matches_reference() {
    for name in [
        "esprit_abs",
        "esprit_rel",
        "esprit_fixed_m",
        "esprit_matrix",
        "esprit_real",
    ] {
        check_esprit(name, 1e-12);
    }
}

fn compare_poles(name: &str, rep: &MiniPoleResult, case: &Case, n_orb: usize) -> (f64, f64) {
    let loc_ref = case.c("pole_location");
    let w_ref = case.c("pole_weight");
    assert_eq!(
        rep.pole_location.len(),
        loc_ref.len(),
        "{name}: number of poles {:?} vs {:?}",
        rep.pole_location,
        loc_ref
    );
    let r = loc_ref.len();
    let d = w_ref.len().checked_div(r).unwrap_or(n_orb * n_orb);
    let w = rep.pole_weight.host_data().unwrap();
    let dl = max_diff(&rep.pole_location, &loc_ref);
    let mut dw = 0.0_f64;
    for j in 0..r {
        for c in 0..d {
            let ours = w[j + r * chan(c, n_orb)];
            dw = dw.max((ours - w_ref[j * d + c]).norm());
        }
    }
    (dl, dw)
}

fn check_dlr(name: &str, n_orb: usize, tol: f64) {
    let case = Case::load(name);
    let xl = case.r("xl");
    let r = xl.len();
    let al = case.c("al");
    let mut shape = vec![r];
    if n_orb > 1 {
        shape.extend([n_orb, n_orb]);
    }
    // Reference channel a n + b is ours a + n b: transpose the trailing axes.
    let mut al_ours = vec![C64::new(0.0, 0.0); al.len()];
    let d = n_orb * n_orb;
    for l in 0..r {
        for c in 0..d {
            al_ours[l + r * chan(c, n_orb)] = al[l * d + c];
        }
    }
    let al = TypedTensor::from_vec_col_major(shape, al_ours).unwrap();
    let params = MiniPoleDlrParams {
        n0: case.i("arg_n0") as usize,
        nmax: case.opt_f("arg_nmax"),
        err: case.opt_f("arg_err"),
        err_type: if case.opt_i("arg_err_type") == Some(1) {
            ErrType::Rel
        } else {
            ErrType::Abs
        },
        m: case.opt_i("arg_M").map(|m| m as usize),
        symmetry: case.opt_i("arg_symmetry") == Some(1),
        ..MiniPoleDlrParams::new(0, 0.0)
    };
    let rep = mini_pole_dlr(&al, &xl, case.r("beta")[0], &params).unwrap();
    let hk_ref = case.c("h_k");
    let nk = hk_ref.len() / d;
    let hk = rep.h_k.host_data().unwrap();
    let mut dh = 0.0_f64;
    for k in 0..nk {
        for c in 0..d {
            dh = dh.max((hk[k + nk * chan(c, n_orb)] - hk_ref[k * d + c]).norm());
        }
    }
    let (dl, dw) = compare_poles(name, &rep, &case, n_orb);
    eprintln!(
        "{name}: |dh| {dh:.1e} M {} vs {} |dloc| {dl:.1e} |dweight| {dw:.1e}",
        rep.esprit.m,
        case.i("M")
    );
    assert_eq!(rep.esprit.m as i64, case.i("M"), "{name}: M");
    assert!(dh < 1e-12 && dl < tol && dw < tol, "{name}");
}

#[test]
fn mini_pole_dlr_matches_reference() {
    for (name, n_orb) in [
        ("dlr_exact", 1),
        ("dlr_fermion", 1),
        ("dlr_fermion_rel", 1),
        ("dlr_fermion_fixed_m", 1),
        ("dlr_boson", 1),
        ("dlr_matrix", 2),
        ("dlr_symmetric", 1),
    ] {
        check_dlr(name, n_orb, 1e-10);
    }
}

/// Metrics of a pole representation against the reference: nearest-pole
/// matching; `(max |Δξ| of poles with |A| > 1e-6, max |Δξ| |A| of the
/// others, max |ΔA|, max |ΔG(iω_n)|)`.
fn pole_metrics(
    rep: &MiniPoleResult,
    case: &Case,
    n_orb: usize,
    w: &[f64],
) -> (f64, f64, f64, f64) {
    let loc_ref = case.c("pole_location");
    let w_ref = case.c("pole_weight");
    let r = loc_ref.len();
    let d = n_orb * n_orb;
    let ours = rep.pole_weight.host_data().unwrap();
    let mut used = vec![false; r];
    let (mut dphys, mut dspur, mut dw) = (0.0_f64, 0.0_f64, 0.0_f64);
    for j in 0..r {
        let wn = (0..d).map(|c| w_ref[j * d + c].norm()).fold(0.0, f64::max);
        let (jo, dl) = (0..r)
            .filter(|&k| !used[k])
            .map(|k| (k, (rep.pole_location[k] - loc_ref[j]).norm()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        used[jo] = true;
        if wn > 1e-6 {
            dphys = dphys.max(dl);
        } else {
            dspur = dspur.max(dl * wn);
        }
        for c in 0..d {
            dw = dw.max((ours[jo + r * chan(c, n_orb)] - w_ref[j * d + c]).norm());
        }
    }
    let z: Vec<C64> = w.iter().map(|&x| C64::new(0.0, x)).collect();
    let g = rep.evaluate(&z).unwrap();
    let g = g.host_data().unwrap();
    let cref = case.c("const");
    let mut dg = 0.0_f64;
    for (i, &zi) in z.iter().enumerate() {
        for c in 0..d {
            let gr: C64 = cref[c]
                + (0..r)
                    .map(|j| w_ref[j * d + c] / (zi - loc_ref[j]))
                    .sum::<C64>();
            dg = dg.max((g[i + z.len() * chan(c, n_orb)] - gr).norm());
        }
    }
    (dphys, dspur, dw, dg)
}

/// Runs the case and checks the discrete outputs exactly (`n0`, number of
/// moments and poles) and the continuous ones to the given tolerances.
fn check_mats(name: &str, n_orb: usize, tol_pole: f64, tol_g: f64) {
    let case = Case::load(name);
    let w = case.r("w");
    let nw = w.len();
    let g = case.c("G_w");
    let d = n_orb * n_orb;
    let mut g_ours = vec![C64::new(0.0, 0.0); g.len()];
    for i in 0..nw {
        for c in 0..d {
            g_ours[i + nw * chan(c, n_orb)] = g[i * d + c];
        }
    }
    let shape = if n_orb > 1 {
        vec![nw, n_orb, n_orb]
    } else {
        vec![nw]
    };
    let g_t = TypedTensor::from_vec_col_major(shape, g_ours).unwrap();
    let params = MiniPoleParams {
        n0: match case.opt_i("arg_n0") {
            Some(n) => N0::Fixed(n as usize),
            None => N0::Auto { shift: 0 },
        },
        m: case.opt_i("arg_M").map(|m| m as usize),
        symmetry: case.opt_i("arg_symmetry") == Some(1),
        g_symmetric: case.opt_i("arg_G_symmetric") == Some(1),
        compute_const: case.opt_i("arg_compute_const") == Some(1),
        plane: case
            .opt_i("arg_plane")
            .map(|p| if p == 1 { Plane::W } else { Plane::Z }),
        ..MiniPoleParams::new(case.r("arg_err")[0])
    };
    let rep = mini_pole(&g_t, &w, &params).unwrap();
    let hk_ref = case.c("h_k");
    let nk_ref = hk_ref.len() / d;
    let hk = rep.h_k.host_data().unwrap();
    let nk = rep.h_k.shape()[0];
    let mut dh = 0.0_f64;
    for k in 0..nk.min(nk_ref) {
        for c in 0..d {
            dh = dh.max((hk[k + nk * chan(c, n_orb)] - hk_ref[k * d + c]).norm());
        }
    }
    let em_ref = case.r("err_max")[0];
    let em = rep.err_max.unwrap();
    let npoles_ref = case.c("pole_location").len();
    let (dphys, dspur, dw, dg) = pole_metrics(&rep, &case, n_orb, &w);
    let dc = max_diff(
        &(0..d)
            .map(|c| rep.constant[chan(c, n_orb)])
            .collect::<Vec<_>>(),
        &case.c("const"),
    );
    eprintln!(
        "{name}: n0 {}/{} K {nk}/{nk_ref} poles {}/{npoles_ref} err_max {em:.2e}/{em_ref:.2e} \
         |dh| {dh:.1e} |dxi| {dphys:.1e} |dxi A| {dspur:.1e} |dA| {dw:.1e} |dG| {dg:.1e} |dC| {dc:.1e}",
        rep.n0,
        case.i("n0"),
        rep.pole_location.len()
    );
    assert_eq!(rep.n0 as i64, case.i("n0"), "{name}: n0");
    assert_eq!(
        rep.pole_location.len(),
        npoles_ref,
        "{name}: number of poles"
    );
    assert!((em - em_ref).abs() < 0.2 * em_ref, "{name}: err_max");
    assert!(dh < 0.05 * em_ref, "{name}: h_k");
    assert!(
        dphys < tol_pole && dspur < tol_g && dw < tol_g && dc < tol_g && dg < tol_g,
        "{name}"
    );
}

/// The DLR path of the port on a DLR of this crate, for both statistics.
#[test]
fn mini_pole_dlr_from_sparse_ir_dlr() {
    use sparse_ir::{
        Bosonic, DiscreteLehmannRepresentation, Fermionic, MatsubaraSampling, StatisticsType,
    };
    fn run<S: StatisticsType + 'static>(beta: f64, n0: usize) {
        let spec = [(-1.42, 0.3), (0.26, 0.5), (1.16, 0.2)];
        let dlr = DiscreteLehmannRepresentation::<S>::new(beta, 2.0, 1e-12).unwrap();
        let s = MatsubaraSampling::new(&dlr).unwrap();
        let v: Vec<C64> = s
            .sampling_points()
            .iter()
            .map(|f| {
                let z = f.value_imaginary(beta);
                spec.iter().map(|&(x, a)| a / (z - x)).sum()
            })
            .collect();
        let v = TypedTensor::from_vec_col_major(vec![v.len()], v).unwrap();
        let g = s.fit_nd(None, &v, 0).unwrap();
        let rep = sparse_ir::mpm::mini_pole_dlr_from(&dlr, &g, &MiniPoleDlrParams::new(n0, 1e-8))
            .unwrap();
        assert_eq!(rep.pole_location.len(), 3, "{:?}", rep.pole_location);
        let a = rep.pole_weight.host_data().unwrap();
        for (j, &(x, w)) in spec.iter().enumerate() {
            assert!(
                (rep.pole_location[j] - x).norm() < 1e-6,
                "{} vs {x}",
                rep.pole_location[j]
            );
            assert!((a[j] - w).norm() < 1e-6, "{} vs {w}", a[j]);
        }
    }
    run::<Fermionic>(50.0, 5);
    run::<Bosonic>(50.0, 5);
}

#[test]
fn symmetric_mini_pole_rejects_zero_frequency_start() {
    let beta = 100.0;
    let w: Vec<f64> = (0..50)
        .map(|n| 2.0 * n as f64 * std::f64::consts::PI / beta)
        .collect();
    let g: Vec<C64> = w
        .iter()
        .map(|&x| 0.5 / C64::new(-0.3, x) + 0.5 / C64::new(0.3, x))
        .collect();
    let g = TypedTensor::from_vec_col_major(vec![w.len()], g).unwrap();
    let params = MiniPoleParams {
        n0: N0::Fixed(0),
        symmetry: true,
        ..MiniPoleParams::new(1e-10)
    };
    assert!(mini_pole(&g, &w, &params).is_err());
}

/// Exact data (`err = 1e-10`): the moments agree to about 1e-13 and the
/// poles to about 1e-10. Noisy data (`η = 1e-7`, `err = 1e-6`): the moments
/// agree to 1e-8, within the tolerance `0.01 err_max` of the reference's own
/// quadrature, and the poles, being that sensitive to the moments, to 1e-4.
#[test]
fn mini_pole_matches_reference() {
    let mut failed = Vec::new();
    for (name, n_orb, tol_pole, tol_g) in [
        ("mats_fermion_n0", 1, 1e-8, 1e-7),
        ("mats_fermion_const", 1, 1e-8, 1e-7),
        ("mats_boson", 1, 1e-8, 1e-7),
        ("mats_matrix", 2, 1e-8, 1e-7),
        ("mats_matrix_gsym", 2, 1e-8, 1e-7),
        ("mats_symmetric", 1, 1e-8, 1e-7),
        ("mats_matrix_symmetric", 2, 1e-8, 1e-7),
        ("mats_fermion_auto", 1, 1e-4, 1e-4),
        ("mats_fermion_plane_w", 1, 1e-4, 1e-4),
        ("mats_fermion_fixed_m", 1, 1e-4, 1e-4),
    ] {
        if std::panic::catch_unwind(|| check_mats(name, n_orb, tol_pole, tol_g)).is_err() {
            failed.push(name);
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}
