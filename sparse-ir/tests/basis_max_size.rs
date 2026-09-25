//! Size-limited bases (`max_size`) against SparseIR.jl (issue #285).
//!
//! `FiniteTempBasis::new(kernel, beta, epsilon, Some(max_size))` must truncate
//! the basis, not the SVE. The default sampling points of a basis of size L
//! come from singular functions beyond the basis: the roots of u_L (tau) and
//! v_L (omega), and the sign changes of uhat_L or uhat_{L+1} (Matsubara). An
//! SVE truncated to `max_size` functions does not contain them, so the point
//! selection fell back to the extrema of the last function: the points
//! differed from SparseIR.jl, and a basis could get fewer Matsubara points
//! than L (6 for L = 7 at Λ = 10, fermionic). `accuracy()` reported
//! s_{L-1}/s_0 instead of s_L/s_0 for the same reason.
//!
//! Reference data (a compatibility check against the reference
//! implementation): SparseIR.jl 1.1.4, the native Julia implementation (not
//! the libsparseir wrapper), git-tree-sha1
//! 175036112d3e0ff85e967ae1919b33fefa0cb637, run with Julia 1.12.5 by
//! `basis_max_size_reference.jl` next to this file. It evaluates
//! `FiniteTempBasis(statistics, β, ωmax, ε; kernel, max_size)` with the default
//! (untruncated) SVE and prints the generated block at the end of this file:
//! Float64 values as shortest round-trip `repr`, tau points as SparseIR.jl's
//! τ ∈ [0, β] in ascending order, omega points in ascending order.

use sparse_ir::basis::FiniteTempBasis;
use sparse_ir::kernel::{
    CentrosymmKernel, KernelProperties, LogisticKernel, RegularizedBoseKernel,
};
use sparse_ir::sve::{TworkType, compute_sve};
use sparse_ir::traits::{Bosonic, Fermionic, StatisticsType};

/// Tolerance for the default tau and omega points, relative to β (tau) or
/// ωmax (omega).
///
/// Tau and omega points are compared only for groups whose SVE runs in
/// double-double arithmetic (ε = 1e-10). For bases without `max_size`, which
/// issue #285 does not affect, Rust and SparseIR.jl agree on these points to
/// 5.3e-16 of the interval at Λ = 10, 100 and 1000: both store the singular
/// functions as Float64 coefficients and locate their roots in Float64, which
/// is accurate to a few ulp. 1e-12 leaves a margin of over 1000. The defect
/// guarded against (extrema of the last function instead of the roots of the
/// next one) moved a point of every case here by at least 6.9e-3 of the
/// interval.
const POINT_TOL: f64 = 1e-12;

/// Relative tolerance for `accuracy()` = s_L / s_0.
///
/// A singular value s_l carries an absolute error of about eps(Twork) * s_0,
/// so s_L / s_0 has a relative error of about eps(Twork) * s_0 / s_L. The worst
/// case is the Float64 group (ε = 1e-6, L = 15, s_L / s_0 = 1.4e-4), about
/// 1.6e-12; the double-double groups are far below. 1e-10 leaves a margin of
/// over 60. The defect (s_{L-1} / s_0 instead of s_L / s_0) changed the value
/// by a factor of 1.5 or more.
const ACCURACY_RTOL: f64 = 1e-10;

#[derive(Clone, Copy, Debug)]
enum KernelKind {
    Logistic,
    RegularizedBose,
}

/// One kernel, β and ε, with the bases of several `max_size` values.
struct Group {
    kernel: KernelKind,
    lambda: f64,
    beta: f64,
    eps: f64,
    cases: &'static [SizeCase],
}

/// SparseIR.jl data of the bases with one `max_size`.
///
/// Size, accuracy and the tau and omega points depend on the SVE only, so they
/// are shared by the fermionic and bosonic bases.
struct SizeCase {
    max_size: usize,
    size: usize,
    accuracy: f64,
    /// `default_tau_sampling_points`, τ ∈ [0, β], ascending (`None` for the
    /// Float64 group, see [`POINT_TOL`])
    tau: Option<&'static [f64]>,
    /// `default_omega_sampling_points`, ascending (`None` as for `tau`)
    omega: Option<&'static [f64]>,
    fermionic: Option<Matsu>,
    bosonic: Option<Matsu>,
}

/// Default Matsubara sampling points as indices n of iν_n = iπn/β.
struct Matsu {
    /// `positive_only = false`
    all: &'static [i64],
    /// `positive_only = true`
    positive: &'static [i64],
}

fn check_group(group: &Group) {
    match group.kernel {
        KernelKind::Logistic => check_kernel(group, LogisticKernel::new(group.lambda)),
        KernelKind::RegularizedBose => {
            // Deprecated, but still supported until it is removed (#273).
            #[allow(deprecated)]
            let kernel = RegularizedBoseKernel::new(group.lambda);
            check_kernel(group, kernel)
        }
    }
}

fn check_kernel<K>(group: &Group, kernel: K)
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
{
    // The untruncated SVE, whose functions a size-limited basis must keep.
    let n_sve = compute_sve(kernel.clone(), group.eps, None, None, TworkType::Auto)
        .s
        .len();
    let mut failures = Vec::new();
    for case in group.cases {
        if let Some(matsu) = &case.fermionic {
            check_case::<K, Fermionic>(group, &kernel, case, matsu, n_sve, &mut failures);
        }
        if let Some(matsu) = &case.bosonic {
            check_case::<K, Bosonic>(group, &kernel, case, matsu, n_sve, &mut failures);
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatch(es) against SparseIR.jl 1.1.4:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn check_case<K, S>(
    group: &Group,
    kernel: &K,
    case: &SizeCase,
    matsu: &Matsu,
    n_sve: usize,
    failures: &mut Vec<String>,
) where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType + 'static,
{
    let basis = FiniteTempBasis::<K, S>::new(
        kernel.clone(),
        group.beta,
        Some(group.eps),
        Some(case.max_size),
    );
    let label = format!(
        "{:?} Λ = {}, β = {}, ε = {:e}, {:?}, max_size = {}",
        group.kernel,
        group.lambda,
        group.beta,
        group.eps,
        S::STATISTICS,
        case.max_size
    );
    let mut fail = |what: String| failures.push(format!("{label}: {what}"));

    if basis.size() != case.size {
        fail(format!("size {} (SparseIR.jl {})", basis.size(), case.size));
    }

    let (n_svals, n_uhat_full) = (basis.sve_result().s.len(), basis.uhat_full().len());
    if n_svals != n_sve || n_uhat_full != n_sve {
        fail(format!(
            "SVE truncated: sve_result has {n_svals} and uhat_full {n_uhat_full} functions, \
             the untruncated SVE {n_sve}"
        ));
    }

    let accuracy_err = (basis.accuracy() - case.accuracy).abs() / case.accuracy;
    if accuracy_err > ACCURACY_RTOL {
        fail(format!(
            "accuracy {:e} (SparseIR.jl {:e}, relative error {accuracy_err:.1e} > {ACCURACY_RTOL:e})",
            basis.accuracy(),
            case.accuracy
        ));
    }

    for (positive_only, expected) in [(false, matsu.all), (true, matsu.positive)] {
        let got: Vec<i64> = basis
            .default_matsubara_sampling_points(positive_only)
            .iter()
            .map(|w| w.n())
            .collect();
        if got != expected {
            fail(format!(
                "positive_only = {positive_only}: {} default Matsubara points {got:?}, \
                 SparseIR.jl {} {expected:?}",
                got.len(),
                expected.len()
            ));
        }
        // The symptom of issue #285: fewer points than basis functions makes
        // the Matsubara fit underdetermined.
        if !positive_only && expected.len() >= case.size && got.len() < basis.size() {
            fail(format!(
                "{} default Matsubara points for basis size {} (SparseIR.jl {})",
                got.len(),
                basis.size(),
                expected.len()
            ));
        }
    }

    if let Some(expected) = case.tau {
        // Rust folds τ into [-β/2, β/2]; unfold to SparseIR.jl's [0, β].
        let mut got: Vec<f64> = basis
            .default_tau_sampling_points()
            .iter()
            .map(|&tau| if tau < 0.0 { tau + group.beta } else { tau })
            .collect();
        got.sort_by(f64::total_cmp);
        compare_points("tau", &got, expected, POINT_TOL * group.beta, &mut fail);
    }
    if let Some(expected) = case.omega {
        let mut got = basis.default_omega_sampling_points();
        got.sort_by(f64::total_cmp);
        compare_points("omega", &got, expected, POINT_TOL * basis.wmax(), &mut fail);
    }
}

fn compare_points(
    what: &str,
    got: &[f64],
    expected: &[f64],
    tol: f64,
    fail: &mut impl FnMut(String),
) {
    if got.len() != expected.len() {
        fail(format!(
            "{} default {what} points {got:?}, SparseIR.jl {} {expected:?}",
            got.len(),
            expected.len()
        ));
        return;
    }
    let (i, err) = got
        .iter()
        .zip(expected)
        .map(|(g, e)| (g - e).abs())
        .enumerate()
        .fold((0, 0.0), |acc, (i, e)| if e > acc.1 { (i, e) } else { acc });
    if err > tol {
        fail(format!(
            "default {what} point {i} is {} (SparseIR.jl {}, |error| {err:.1e} > {tol:.1e})",
            got[i], expected[i]
        ));
    }
}

#[test]
fn logistic_lambda_10_max_size_matches_sparseir_jl() {
    check_group(&LOGISTIC_LAMBDA_10);
}

#[test]
fn logistic_lambda_100_max_size_matches_sparseir_jl() {
    check_group(&LOGISTIC_LAMBDA_100);
}

/// Float64 working precision: sizes, accuracy and Matsubara points only.
#[test]
fn logistic_lambda_100_eps_1e_6_max_size_matches_sparseir_jl() {
    check_group(&LOGISTIC_LAMBDA_100_EPS_1E_6);
}

#[test]
fn logistic_lambda_1000_max_size_matches_sparseir_jl() {
    check_group(&LOGISTIC_LAMBDA_1000);
}

#[test]
fn regularized_bose_lambda_10_max_size_matches_sparseir_jl() {
    check_group(&REGULARIZED_BOSE_LAMBDA_10);
}

// ---- BEGIN GENERATED by basis_max_size_reference.jl (SparseIR.jl 1.1.4, Julia 1.12.5) ----
#[rustfmt::skip]
const LOGISTIC_LAMBDA_10: Group = Group {
    kernel: KernelKind::Logistic,
    lambda: 10.0,
    beta: 10.0,
    eps: 1.0e-10,
    cases: &[
        SizeCase {
            max_size: 5,
            size: 5,
            accuracy: 0.006307734902438445,
            tau: Some(&[
                0.37527067597536434, 2.0496711867937387, 5.0, 7.95032881320626, 9.624729324024635,
            ]),
            omega: Some(&[
                -0.7923081073597402, -0.3248792194728739, 1.364173964400305e-16, 0.3248792194728739,
                0.7923081073597402,
            ]),
            fermionic: Some(Matsu {
                all: &[-9, -3, -1, 1, 3, 9],
                positive: &[1, 3, 9],
            }),
            bosonic: Some(Matsu {
                all: &[-6, -2, 0, 2, 6],
                positive: &[0, 2, 6],
            }),
        },
        SizeCase {
            max_size: 6,
            size: 6,
            accuracy: 0.0012650587530826224,
            tau: Some(&[
                0.28211354383062215, 1.5062100711529847, 3.6811462049547696, 6.31885379504523,
                8.493789928847015, 9.717886456169378,
            ]),
            omega: Some(&[
                -0.8582936084309079, -0.46515375024884953, -0.131318103044386, 0.131318103044386,
                0.4651537502488494, 0.8582936084309079,
            ]),
            fermionic: Some(Matsu {
                all: &[-9, -3, -1, 1, 3, 9],
                positive: &[1, 3, 9],
            }),
            bosonic: Some(Matsu {
                all: &[-12, -4, -2, 0, 2, 4, 12],
                positive: &[0, 2, 4, 12],
            }),
        },
        SizeCase {
            max_size: 7,
            size: 7,
            accuracy: 0.00022044190542023527,
            tau: Some(&[
                0.21976980976083027, 1.1608366860149932, 2.8258095555016913, 5.000000000000001,
                7.174190444498309, 8.839163313985006, 9.78023019023917,
            ]),
            omega: Some(&[
                -0.899230231740475, -0.5800172265020634, -0.2527668173454177, 0.0,
                0.2527668173454177, 0.5800172265020636, 0.899230231740475,
            ]),
            fermionic: Some(Matsu {
                all: &[-15, -5, -3, -1, 1, 3, 5, 15],
                positive: &[1, 3, 5, 15],
            }),
            bosonic: Some(Matsu {
                all: &[-12, -4, -2, 0, 2, 4, 12],
                positive: &[0, 2, 4, 12],
            }),
        },
        SizeCase {
            max_size: 8,
            size: 8,
            accuracy: 3.388219748878556e-5,
            tau: Some(&[
                0.17588048505886467, 0.9242631256852296, 2.2438442147830906, 4.020160506835126,
                5.979839493164874, 7.75615578521691, 9.07573687431477, 9.824119514941135,
            ]),
            omega: Some(&[
                -0.9255792020303564, -0.6687938433234504, -0.36535430758333276,
                -0.10984042975145991, 0.10984042975145991, 0.36535430758333276, 0.6687938433234505,
                0.9255792020303564,
            ]),
            fermionic: Some(Matsu {
                all: &[-15, -5, -3, -1, 1, 3, 5, 15],
                positive: &[1, 3, 5, 15],
            }),
            bosonic: Some(Matsu {
                all: &[-20, -6, -4, -2, 0, 2, 4, 6, 20],
                positive: &[0, 2, 4, 6, 20],
            }),
        },
        SizeCase {
            max_size: 9,
            size: 9,
            accuracy: 4.652204208612409e-6,
            tau: Some(&[
                0.14381086972666268, 0.7539188172343514, 1.8281756700955498, 3.295742947640673, 5.0,
                6.704257052359327, 8.17182432990445, 9.246081182765648, 9.856189130273338,
            ]),
            omega: Some(&[
                -0.9432179772813258, -0.7358929070487296, -0.46468497428938094, -0.2113257301034317,
                0.0, 0.2113257301034317, 0.4646849742893808, 0.7358929070487297, 0.9432179772813258,
            ]),
            fermionic: Some(Matsu {
                all: &[-23, -7, -5, -3, -1, 1, 3, 5, 7, 23],
                positive: &[1, 3, 5, 7, 23],
            }),
            bosonic: Some(Matsu {
                all: &[-20, -6, -4, -2, 0, 2, 4, 6, 20],
                positive: &[0, 2, 4, 6, 20],
            }),
        },
        SizeCase {
            max_size: 10,
            size: 10,
            accuracy: 5.76749190298595e-7,
            tau: Some(&[
                0.11968141782803032, 0.6267736255645506, 1.5196069524112277, 2.74996248691348,
                4.220505372131992, 5.779494627868008, 7.25003751308652, 8.480393047588771,
                9.37322637443545, 9.88031858217197,
            ]),
            omega: Some(&[
                -0.9554661910241915, -0.7864829976189045, -0.5487566240865254, -0.3060896537687696,
                -0.0951704924885707, 0.0951704924885707, 0.3060896537687696, 0.5487566240865251,
                0.7864829976189045, 0.9554661910241913,
            ]),
            fermionic: Some(Matsu {
                all: &[-23, -7, -5, -3, -1, 1, 3, 5, 7, 23],
                positive: &[1, 3, 5, 7, 23],
            }),
            bosonic: Some(Matsu {
                all: &[-28, -8, -6, -4, -2, 0, 2, 4, 6, 8, 28],
                positive: &[0, 2, 4, 6, 8, 28],
            }),
        },
        SizeCase {
            max_size: 13,
            size: 13,
            accuracy: 6.47185753092162e-10,
            tau: Some(&[
                0.07476726936977762, 0.3915416693976248, 0.9510219545130882, 1.7331330056844652,
                2.70426325827944, 3.814847542062893, 5.000000000000001, 6.185152457937106,
                7.29573674172056, 8.266866994315535, 9.048978045486912, 9.608458330602375,
                9.925232730630222,
            ]),
            omega: Some(&[
                -0.9756413381062321, -0.8774639030133529, -0.7216007819947255, -0.5345148469500631,
                -0.34172140656309646, -0.1623693872331598, 1.364173964400305e-16,
                0.1623693872331598, 0.34172140656309646, 0.5345148469500629, 0.7216007819947257,
                0.8774639030133529, 0.9756413381062319,
            ]),
            fermionic: Some(Matsu {
                all: &[-43, -15, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 15, 43],
                positive: &[1, 3, 5, 7, 9, 15, 43],
            }),
            bosonic: Some(Matsu {
                all: &[-38, -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12, 38],
                positive: &[0, 2, 4, 6, 8, 12, 38],
            }),
        },
    ],
};
#[rustfmt::skip]
const LOGISTIC_LAMBDA_100: Group = Group {
    kernel: KernelKind::Logistic,
    lambda: 100.0,
    beta: 10.0,
    eps: 1.0e-10,
    cases: &[
        SizeCase {
            max_size: 7,
            size: 7,
            accuracy: 0.0402142528478818,
            tau: Some(&[
                0.06417028463099994, 0.42509385479543194, 1.6374759478853995, 4.999999999999999,
                8.362524052114601, 9.574906145204569, 9.935829715369,
            ]),
            omega: Some(&[
                -6.418711526597188, -1.9360879757110583, -0.47925877693015345,
                1.1726646147170549e-15, 0.47925877693015345, 1.9360879757110583, 6.418711526597189,
            ]),
            fermionic: Some(Matsu {
                all: &[-51, -11, -3, -1, 1, 3, 11, 51],
                positive: &[1, 3, 11, 51],
            }),
            bosonic: Some(Matsu {
                all: &[-44, -8, -2, 0, 2, 8, 44],
                positive: &[0, 2, 8, 44],
            }),
        },
        SizeCase {
            max_size: 10,
            size: 10,
            accuracy: 0.005699321145803073,
            tau: Some(&[
                0.04064587654145746, 0.2341667774384104, 0.6864905751001926, 1.6723999009670099,
                3.6503442963564825, 6.349655703643517, 8.32760009903299, 9.313509424899808,
                9.76583322256159, 9.959354123458542,
            ]),
            omega: Some(&[
                -8.138588126887893, -4.150196843634329, -1.78176706277135, -0.6738819586624432,
                -0.16041144471730592, 0.16041144471730592, 0.6738819586624432, 1.78176706277135,
                4.150196843634329, 8.138588126887893,
            ]),
            fermionic: Some(Matsu {
                all: &[-69, -17, -7, -3, -1, 1, 3, 7, 17, 69],
                positive: &[1, 3, 7, 17, 69],
            }),
            bosonic: Some(Matsu {
                all: &[-80, -22, -8, -4, -2, 0, 2, 4, 8, 22, 80],
                positive: &[0, 2, 4, 8, 22, 80],
            }),
        },
        SizeCase {
            max_size: 15,
            size: 15,
            accuracy: 0.00013685597490293338,
            tau: Some(&[
                0.024178594413840626, 0.13121713586569328, 0.34106769008565907, 0.6903915239122183,
                1.2469526264958108, 2.114612374827205, 3.3882782690610433, 5.0, 6.611721730938957,
                7.885387625172795, 8.753047373504188, 9.309608476087782, 9.658932309914341,
                9.868782864134307, 9.97582140558616,
            ]),
            omega: Some(&[
                -9.243962317777932, -6.860760428117921, -4.434795847914188, -2.6653646099249895,
                -1.5067331475556711, -0.7637743828789403, -0.29596949695555347, 0.0,
                0.29596949695555347, 0.7637743828789403, 1.5067331475556711, 2.6653646099249895,
                4.434795847914188, 6.860760428117918, 9.243962317777932,
            ]),
            fermionic: Some(Matsu {
                all: &[-131, -39, -21, -11, -7, -5, -3, -1, 1, 3, 5, 7, 11, 21, 39, 131],
                positive: &[1, 3, 5, 7, 11, 21, 39, 131],
            }),
            bosonic: Some(Matsu {
                all: &[-120, -36, -18, -10, -6, -4, -2, 0, 2, 4, 6, 10, 18, 36, 120],
                positive: &[0, 2, 4, 6, 10, 18, 36, 120],
            }),
        },
        SizeCase {
            max_size: 20,
            size: 20,
            accuracy: 1.9046720878034658e-6,
            tau: Some(&[
                0.016520510110292763, 0.08821371414425627, 0.22218990953513318, 0.4279733957594578,
                0.7213390829195476, 1.1261246848361872, 1.674568435321957, 2.401719469586032,
                3.326661988522959, 4.419547728502938, 5.580452271497062, 6.673338011477041,
                7.598280530413968, 8.325431564678041, 8.873875315163811, 9.278660917080453,
                9.572026604240541, 9.777810090464866, 9.911786285855742, 9.983479489889707,
            ]),
            omega: Some(&[
                -9.627665097275862, -8.264643224184361, -6.471746036555501, -4.764381801412788,
                -3.3607838701654758, -2.280650790832304, -1.4699053438462388, -0.8668385426957764,
                -0.42701674754753716, -0.12191363445843048, 0.12191363445843048,
                0.42701674754753716, 0.8668385426957764, 1.4699053438462388, 2.280650790832304,
                3.3607838701654758, 4.764381801412788, 6.471746036555501, 8.264643224184361,
                9.627665097275862,
            ]),
            fermionic: Some(Matsu {
                all: &[
                    -175, -55, -31, -19, -13, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 13, 19, 31, 55,
                    175,
                ],
                positive: &[1, 3, 5, 7, 9, 13, 19, 31, 55, 175],
            }),
            bosonic: Some(Matsu {
                all: &[
                    -188, -60, -34, -22, -14, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 14, 22, 34,
                    60, 188,
                ],
                positive: &[0, 2, 4, 6, 8, 10, 14, 22, 34, 60, 188],
            }),
        },
        SizeCase {
            max_size: 25,
            size: 25,
            accuracy: 1.5750908652138604e-8,
            tau: Some(&[
                0.012163382118338673, 0.06453956226540769, 0.16065338546014885, 0.3039299425672026,
                0.4996656873465488, 0.7553988714232285, 1.0811857258437108, 1.4894250276043541,
                1.9935056320172384, 2.6042122723107486, 3.32313287840973, 4.134396257652064, 5.0,
                5.865603742347936, 6.67686712159027, 7.395787727689251, 8.00649436798276,
                8.510574972395645, 8.91881427415629, 9.244601128576772, 9.50033431265345,
                9.696070057432797, 9.839346614539851, 9.935460437734593, 9.987836617881662,
            ]),
            omega: Some(&[
                -9.790918418754211, -8.972447606272668, -7.747697193425136, -6.384335107818904,
                -5.076718857262216, -3.9225845343864627, -2.9494403246353467, -2.1480667835768177,
                -1.4959717202749112, -0.9700867174943937, -0.5532603215502627, -0.23755700880894418,
                1.1726646147170549e-15, 0.23755700880894418, 0.5532603215502627, 0.9700867174943937,
                1.4959717202749112, 2.1480667835768177, 2.9494403246353467, 3.9225845343864627,
                5.076718857262216, 6.3843351078189015, 7.747697193425136, 8.972447606272668,
                9.790918418754211,
            ]),
            fermionic: Some(Matsu {
                all: &[
                    -253, -83, -47, -31, -23, -17, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11,
                    13, 17, 23, 31, 47, 83, 253,
                ],
                positive: &[1, 3, 5, 7, 9, 11, 13, 17, 23, 31, 47, 83, 253],
            }),
            bosonic: Some(Matsu {
                all: &[
                    -240, -78, -44, -30, -20, -16, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12,
                    16, 20, 30, 44, 78, 240,
                ],
                positive: &[0, 2, 4, 6, 8, 10, 12, 16, 20, 30, 44, 78, 240],
            }),
        },
    ],
};
#[rustfmt::skip]
const LOGISTIC_LAMBDA_100_EPS_1E_6: Group = Group {
    kernel: KernelKind::Logistic,
    lambda: 100.0,
    beta: 10.0,
    eps: 1.0e-6,
    cases: &[
        SizeCase {
            max_size: 7,
            size: 7,
            accuracy: 0.040214252847881805,
            tau: None,
            omega: None,
            fermionic: Some(Matsu {
                all: &[-51, -11, -3, -1, 1, 3, 11, 51],
                positive: &[1, 3, 11, 51],
            }),
            bosonic: Some(Matsu {
                all: &[-44, -8, -2, 0, 2, 8, 44],
                positive: &[0, 2, 8, 44],
            }),
        },
        SizeCase {
            max_size: 10,
            size: 10,
            accuracy: 0.005699321145803077,
            tau: None,
            omega: None,
            fermionic: Some(Matsu {
                all: &[-69, -17, -7, -3, -1, 1, 3, 7, 17, 69],
                positive: &[1, 3, 7, 17, 69],
            }),
            bosonic: Some(Matsu {
                all: &[-80, -22, -8, -4, -2, 0, 2, 4, 8, 22, 80],
                positive: &[0, 2, 4, 8, 22, 80],
            }),
        },
        SizeCase {
            max_size: 15,
            size: 15,
            accuracy: 0.00013685597490293165,
            tau: None,
            omega: None,
            fermionic: Some(Matsu {
                all: &[-131, -39, -21, -11, -7, -5, -3, -1, 1, 3, 5, 7, 11, 21, 39, 131],
                positive: &[1, 3, 5, 7, 11, 21, 39, 131],
            }),
            bosonic: Some(Matsu {
                all: &[-120, -36, -18, -10, -6, -4, -2, 0, 2, 4, 6, 10, 18, 36, 120],
                positive: &[0, 2, 4, 6, 10, 18, 36, 120],
            }),
        },
    ],
};
#[rustfmt::skip]
const LOGISTIC_LAMBDA_1000: Group = Group {
    kernel: KernelKind::Logistic,
    lambda: 1000.0,
    beta: 100.0,
    eps: 1.0e-10,
    cases: &[
        SizeCase {
            max_size: 20,
            size: 20,
            accuracy: 0.0007103544269340659,
            tau: Some(&[
                0.029739045116394802, 0.16410620961617717, 0.4412934783047717, 0.9456501050432387,
                1.861663067124697, 3.561778720695974, 6.747165409422196, 12.668328218824133,
                23.245248617502124, 39.95736765183505, 60.04263234816495, 76.75475138249787,
                87.33167178117587, 93.2528345905778, 96.43822127930403, 98.1383369328753,
                99.05434989495676, 99.55870652169523, 99.83589379038382, 99.9702609548836,
            ]),
            omega: Some(&[
                -8.910768900117116, -5.8805901631879145, -3.337979003320242, -1.7998256455735389,
                -0.9549487258761596, -0.5027482660100356, -0.2610446773682865, -0.129585895411091,
                -0.05577457295503318, -0.014461112373105937, 0.014461112373105937,
                0.05577457295503318, 0.129585895411091, 0.2610446773682865, 0.5027482660100356,
                0.9549487258761596, 1.7998256455735389, 3.337979003320242, 5.880590163187915,
                8.910768900117116,
            ]),
            fermionic: Some(Matsu {
                all: &[
                    -975, -285, -133, -67, -35, -19, -9, -5, -3, -1, 1, 3, 5, 9, 19, 35, 67, 133,
                    285, 975,
                ],
                positive: &[1, 3, 5, 9, 19, 35, 67, 133, 285, 975],
            }),
            bosonic: Some(Matsu {
                all: &[
                    -1032, -306, -146, -76, -42, -22, -12, -6, -4, -2, 0, 2, 4, 6, 12, 22, 42, 76,
                    146, 306, 1032,
                ],
                positive: &[0, 2, 4, 6, 12, 22, 42, 76, 146, 306, 1032],
            }),
        },
        SizeCase {
            max_size: 40,
            size: 40,
            accuracy: 4.47090584505109e-8,
            tau: Some(&[
                0.013078716096309728, 0.06950025185115116, 0.17349225361918963, 0.3296767652333221,
                0.5455084160000068, 0.8322949572044602, 1.2066702717693056, 1.6926016450059977,
                2.323940644532918, 3.1474551639412818, 4.226292256055242, 5.643936559302554,
                7.508636531479973, 9.957517616981903, 13.157869654277988, 17.299930266883067,
                22.570679170230967, 29.09452292815187, 36.83560358626811, 45.49616626187479,
                54.50383373812521, 63.16439641373189, 70.90547707184814, 77.42932082976904,
                82.70006973311693, 86.84213034572201, 90.0424823830181, 92.49136346852002,
                94.35606344069744, 95.77370774394475, 96.85254483605871, 97.67605935546709,
                98.30739835499399, 98.79332972823069, 99.16770504279555, 99.454491584,
                99.67032323476667, 99.82650774638081, 99.93049974814885, 99.9869212839037,
            ]),
            omega: Some(&[
                -9.768765016374502, -8.874473255282712, -7.568932014094807, -6.163020790606004,
                -4.864955489342272, -3.7660244910677614, -2.8808602581228393, -2.187866100105488,
                -1.6539077463928216, -1.245985736484763, -0.9355641273286832, -0.6994896003396817,
                -0.5195770356565476, -0.38181664884034416, -0.27559246467814824,
                -0.1930286199325461, -0.12845948694884618, -0.07802463356779683,
                -0.03951362998113868, -0.01153331832926761, 0.01153331832926761,
                0.03951362998113868, 0.07802463356779683, 0.12845948694884618, 0.1930286199325461,
                0.27559246467814824, 0.38181664884034416, 0.5195770356565476, 0.6994896003396817,
                0.9355641273286832, 1.245985736484763, 1.6539077463928216, 2.187866100105488,
                2.8808602581228393, 3.7660244910677614, 4.864955489342272, 6.163020790606004,
                7.568932014094807, 8.874473255282712, 9.768765016374502,
            ]),
            fermionic: Some(Matsu {
                all: &[
                    -2235, -725, -413, -273, -191, -139, -103, -77, -57, -43, -31, -23, -17, -13,
                    -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 17, 23, 31, 43, 57, 77, 103,
                    139, 191, 273, 413, 725, 2235,
                ],
                positive: &[
                    1, 3, 5, 7, 9, 11, 13, 17, 23, 31, 43, 57, 77, 103, 139, 191, 273, 413, 725,
                    2235,
                ],
            }),
            bosonic: Some(Matsu {
                all: &[
                    -2302, -748, -426, -284, -200, -146, -108, -82, -62, -46, -34, -26, -20, -16,
                    -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 16, 20, 26, 34, 46, 62, 82,
                    108, 146, 200, 284, 426, 748, 2302,
                ],
                positive: &[
                    0, 2, 4, 6, 8, 10, 12, 16, 20, 26, 34, 46, 62, 82, 108, 146, 200, 284, 426, 748,
                    2302,
                ],
            }),
        },
    ],
};
#[rustfmt::skip]
const REGULARIZED_BOSE_LAMBDA_10: Group = Group {
    kernel: KernelKind::RegularizedBose,
    lambda: 10.0,
    beta: 10.0,
    eps: 1.0e-10,
    cases: &[
        SizeCase {
            max_size: 7,
            size: 7,
            accuracy: 0.00029938752020373146,
            tau: Some(&[
                0.21595518129077118, 1.1443829806197692, 2.804308739440558, 4.999999999999999,
                7.1956912605594425, 8.855617019380231, 9.784044818709228,
            ]),
            omega: Some(&[
                -0.9236011758416132, -0.655062994547515, -0.32179542212054335, 0.0,
                0.32179542212054335, 0.655062994547515, 0.9236011758416132,
            ]),
            fermionic: None,
            bosonic: Some(Matsu {
                all: &[-12, -4, -2, 0, 2, 4, 12],
                positive: &[0, 2, 4, 12],
            }),
        },
        SizeCase {
            max_size: 10,
            size: 10,
            accuracy: 7.968271069229755e-7,
            tau: Some(&[
                0.11886645655032502, 0.6230067934129568, 1.5126388769745136, 2.742557088284816,
                4.217239634018651, 5.782760365981349, 7.257442911715184, 8.487361123025485,
                9.376993206587043, 9.881133543449675,
            ]),
            omega: Some(&[
                -0.9633166124679355, -0.8196861092180456, -0.6042422014501574, -0.3616750649576711,
                -0.11928552977297832, 0.11928552977297832, 0.3616750649576711, 0.6042422014501572,
                0.8196861092180456, 0.9633166124679355,
            ]),
            fermionic: None,
            bosonic: Some(Matsu {
                all: &[-28, -8, -6, -4, -2, 0, 2, 4, 6, 8, 28],
                positive: &[0, 2, 4, 6, 8, 28],
            }),
        },
    ],
};
// ---- END GENERATED ----
