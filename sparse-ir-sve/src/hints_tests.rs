use super::*;
use crate::kernel::{AbstractKernel, ExpandableKernel};
use crate::sve::{TworkType, compute_sve};
use crate::test_kernels::{ExpKernel, TiltedExpKernel};

/// K(x, y) = exp(-(x - y)²) on [-1, 1] × [0, 2]: not centrosymmetric, on a
/// domain not symmetric about 0 in y
#[derive(Debug, Clone, Copy)]
struct GaussianKernel {
    nseg: usize,
}

impl AbstractKernel for GaussianKernel {
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
        (-(x - y) * (x - y)).exp()
    }
}

impl SVEKernel for GaussianKernel {
    type SVEHintsType<T>
        = SegmentHints
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;

    fn xmax(&self) -> f64 {
        1.0
    }

    fn ymax(&self) -> f64 {
        2.0
    }

    fn ymin(&self) -> f64 {
        0.0
    }

    fn sve_hints<T>(&self, _epsilon: f64) -> SegmentHints
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        SegmentHints::uniform(self, self.nseg, self.nseg, 16).unwrap()
    }
}

impl ExpandableKernel for GaussianKernel {}

fn segments(hints: &SegmentHints) -> (Vec<f64>, Vec<f64>) {
    (
        SVEHints::<f64>::segments_x(hints),
        SVEHints::<f64>::segments_y(hints),
    )
}

/// The hints keep the boundaries of the full domain; the SVE of a
/// centrosymmetric kernel discretizes those in (0, xmax], with 0 added where
/// it is not one of them
#[test]
fn test_centrosymmetric_kernel_discretizes_the_half_domain() {
    use crate::sve::utils::half_domain_segments;
    let half = |s: &[f64]| half_domain_segments(s);

    let kernel = ExpKernel::new(1.0);
    let hints = SegmentHints::uniform(&kernel, 4, 2, 16).unwrap();
    let (x, y) = segments(&hints);
    assert_eq!(
        (x.clone(), y.clone()),
        (vec![-1.0, -0.5, 0.0, 0.5, 1.0], vec![-1.0, 0.0, 1.0])
    );
    assert_eq!((half(&x), half(&y)), (vec![0.0, 0.5, 1.0], vec![0.0, 1.0]));

    let hints = SegmentHints::uniform(&kernel, 3, 3, 16).unwrap();
    assert_eq!(half(&segments(&hints).0), vec![0.0, 1.0 / 3.0, 1.0]);

    let hints = SegmentHints::new(&kernel, &[-1.0, -0.2, 0.7, 1.0], &[-1.0, 1.0], 8).unwrap();
    assert_eq!(segments(&hints).0, vec![-1.0, -0.2, 0.7, 1.0]);
    assert_eq!(half(&segments(&hints).0), vec![0.0, 0.7, 1.0]);
}

/// Other kernels are discretized on all the boundaries
#[test]
fn test_other_kernels_keep_the_full_domain() {
    let hints = SegmentHints::uniform(&TiltedExpKernel(ExpKernel::new(1.0)), 4, 2, 16).unwrap();
    assert_eq!(
        segments(&hints),
        (vec![-1.0, -0.5, 0.0, 0.5, 1.0], vec![-1.0, 0.0, 1.0])
    );

    let kernel = GaussianKernel { nseg: 1 };
    let hints = SegmentHints::uniform(&kernel, 2, 4, 10).unwrap();
    assert_eq!(
        segments(&hints),
        (vec![-1.0, 0.0, 1.0], vec![0.0, 0.5, 1.0, 1.5, 2.0])
    );
    assert_eq!(SVEHints::<f64>::ngauss(&hints), 10);
    // At most ngauss singular values per segment of the axis with fewer
    assert_eq!(SVEHints::<f64>::nsvals(&hints), 20);
}

/// Uniform boundaries are exactly symmetric about the midpoint of the domain
#[test]
fn test_uniform_segments_are_symmetric() {
    for n in 1..=40 {
        let s = uniform_segments("n", n, -3.0, 3.0).unwrap();
        assert_eq!((s[0], s[n]), (-3.0, 3.0));
        assert!(s.windows(2).all(|w| w[0] < w[1]), "{s:?}");
        assert!((0..=n).all(|i| s[i] == -s[n - i]), "{s:?}");
    }
}

#[test]
fn test_invalid_segments_are_rejected() {
    let kernel = GaussianKernel { nseg: 1 };
    let y = [0.0, 2.0];
    let cases: [(&[f64], &[f64], usize, &str); 7] = [
        (&[-1.0, 1.0], &y, 0, "ngauss"),
        (&[-1.0], &y, 16, "segments_x"),
        (&[-1.0, f64::NAN, 1.0], &y, 16, "segments_x"),
        (&[-1.0, 0.5, 0.5, 1.0], &y, 16, "segments_x"),
        (&[-1.0, 0.5, 0.2, 1.0], &y, 16, "segments_x"),
        (&[-0.9, 1.0], &y, 16, "segments_x"),
        (&[-1.0, 1.0], &[-2.0, 2.0], 16, "segments_y"),
    ];
    for (x, y, ngauss, name) in cases {
        match SegmentHints::new(&kernel, x, y, ngauss) {
            Err(Error::InvalidParameter { name: got, .. }) => assert_eq!(got, name, "{x:?} {y:?}"),
            other => panic!("{x:?} {y:?} {ngauss}: {other:?}"),
        }
    }
    for (nx, ny, name) in [(0, 1, "nseg_x"), (1, 0, "nseg_y")] {
        match SegmentHints::uniform(&kernel, nx, ny, 16) {
            Err(Error::InvalidParameter { name: got, .. }) => assert_eq!(got, name),
            other => panic!("{nx} {ny}: {other:?}"),
        }
    }
}

/// The SVE of a kernel on a domain that is not symmetric about 0, with
/// uniform hints, reproduces the kernel
#[test]
fn test_sve_with_uniform_hints_on_a_shifted_domain() {
    let kernel = GaussianKernel { nseg: 4 };
    let sve = compute_sve(kernel, Some(1e-10), None, None, TworkType::Float64X2).unwrap();
    let s = sve.s();
    for (x, y) in [(-1.0, 0.0), (-0.3, 1.7), (0.9, 2.0), (0.25, 0.4)] {
        let k: f64 = (0..s.len())
            .map(|l| s[l] * sve.u()[l].evaluate(x) * sve.v()[l].evaluate(y))
            .sum();
        let exact: f64 = kernel.compute(x, y);
        assert!(
            (k - exact).abs() < 1e-9 * s[0],
            "({x}, {y}): {k} vs {exact}"
        );
    }
}
