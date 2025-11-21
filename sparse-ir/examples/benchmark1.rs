//! Benchmark for sparseir-rust operations
//!
//! This benchmark is a Rust port of the C benchmark in capi_benchmark/benchmark1.c
//! It measures the performance of fit_zz, eval_zz, and eval_dz operations
//! for both Matsubara and Tau sampling.

use sparse_ir::{
    FiniteTempBasis, Fermionic, LogisticKernel, MatsubaraSampling,
    MatsubaraSamplingPositiveOnly, TauSampling
};
use num_complex::Complex;
use mdarray::{DynRank, Tensor};
use std::time::Instant;

/// Simple benchmark timer
struct Benchmark {
    start: Instant,
    name: &'static str,
}

impl Benchmark {
    fn start(name: &'static str) -> Self {
        Self {
            start: Instant::now(),
            name,
        }
    }

    fn end(self) -> f64 {
        let elapsed = self.start.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        println!("{:<30}: {:10.6} ms", self.name, elapsed_ms);
        elapsed_ms
    }
}

fn benchmark(
    beta: f64,
    omega_max: f64,
    epsilon: f64,
    extra_size: usize,
    nrun: usize,
    positive_only: bool,
) {
    println!("beta: {}", beta);
    println!("omega_max: {}", omega_max);
    println!("epsilon: {}", epsilon);
    println!("Extra size: {}", extra_size);
    println!("Number of runs: {}", nrun);

    // Kernel creation
    let bench = Benchmark::start("Kernel creation");
    let kernel = LogisticKernel::new(beta * omega_max);
    bench.end();

    // SVE computation (happens inside basis creation)
    let bench = Benchmark::start("SVE computation");
    let epsilon_basis = 1e-10;
    let basis = FiniteTempBasis::<_, Fermionic>::new(kernel, beta, Some(epsilon_basis), None);
    bench.end();

    // Get basis size
    let n_basis = basis.size();
    println!("n_basis: {}", n_basis);

    // Get imaginary-time sampling points
    let tau_points = basis.default_tau_sampling_points();
    let n_tau = tau_points.len();
    println!("n_tau: {}", n_tau);

    // Create sampling object for imaginary-time domain
    let tau_sampling = TauSampling::with_sampling_points(&basis, tau_points);

    // Get Matsubara frequency sampling points
    let matsubara_points = basis.default_matsubara_sampling_points(positive_only);
    let n_matsubara = matsubara_points.len();
    println!("n_matsubara: {}", n_matsubara);

    // Create sampling object for Matsubara domain
    let matsubara_sampling = if positive_only {
        Box::new(MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, matsubara_points))
            as Box<dyn MatsubaraSamplingTrait>
    } else {
        Box::new(MatsubaraSampling::with_sampling_points(&basis, matsubara_points))
            as Box<dyn MatsubaraSamplingTrait>
    };

    // Create test data arrays
    // g_matsu_z: [n_matsubara, extra_size] (complex)
    let g_matsu_z = create_test_data_complex(n_matsubara, extra_size);

    // g_tau_z: [n_tau, extra_size] (complex)
    let g_tau_z = create_test_data_complex(n_tau, extra_size);

    // g_basis_d: [n_basis, extra_size] (real)
    let g_basis_d = create_test_data_real(n_basis, extra_size);

    // g_basis_z: [n_basis, extra_size] (complex)
    let g_basis_z = create_test_data_complex(n_basis, extra_size);

    // Test: matsubara, fit_zz (complex values -> complex coefficients)
    // First run to warm up
    let _ = matsubara_sampling.fit_nd(&g_matsu_z, 0);

    let bench = Benchmark::start("fit_zz (Matsubara)");
    for _ in 0..nrun {
        let _ = matsubara_sampling.fit_nd(&g_matsu_z, 0);
    }
    let elapsed = bench.end();
    println!("  Average per run: {:10.6} ms", elapsed / nrun as f64);

    // Test: matsubara, eval_zz (complex coefficients -> complex values)
    let bench = Benchmark::start("eval_zz (Matsubara)");
    for _ in 0..nrun {
        let _ = matsubara_sampling.evaluate_nd(&g_basis_z, 0);
    }
    let elapsed = bench.end();
    println!("  Average per run: {:10.6} ms", elapsed / nrun as f64);

    // Test: matsubara, eval_dz (real coefficients -> complex values)
    // Only for positive_only case (MatsubaraSamplingPositiveOnly)
    if positive_only {
        let bench = Benchmark::start("eval_dz (Matsubara)");
        for _ in 0..nrun {
            let _ = matsubara_sampling.evaluate_nd_real(&g_basis_d, 0);
        }
        let elapsed = bench.end();
        println!("  Average per run: {:10.6} ms", elapsed / nrun as f64);
    }

    // Test: tau, fit_zz (complex values -> complex coefficients)
    // First run to warm up
    let _ = tau_sampling.fit_nd::<Complex<f64>>(None, &g_tau_z, 0);

    let bench = Benchmark::start("fit_zz (Tau)");
    for _ in 0..nrun {
        let _ = tau_sampling.fit_nd::<Complex<f64>>(None, &g_tau_z, 0);
    }
    let elapsed = bench.end();
    println!("  Average per run: {:10.6} ms", elapsed / nrun as f64);

    // Test: tau, eval_zz (complex coefficients -> complex values)
    let bench = Benchmark::start("eval_zz (Tau)");
    for _ in 0..nrun {
        let _ = tau_sampling.evaluate_nd::<Complex<f64>>(None, &g_basis_z, 0);
    }
    let elapsed = bench.end();
    println!("  Average per run: {:10.6} ms", elapsed / nrun as f64);
}

/// Trait to unify MatsubaraSampling and MatsubaraSamplingPositiveOnly
trait MatsubaraSamplingTrait {
    fn fit_nd(&self, values: &Tensor<Complex<f64>, DynRank>, dim: usize) -> Tensor<Complex<f64>, DynRank>;
    fn evaluate_nd(&self, coeffs: &Tensor<Complex<f64>, DynRank>, dim: usize) -> Tensor<Complex<f64>, DynRank>;
    fn evaluate_nd_real(&self, coeffs: &Tensor<f64, DynRank>, dim: usize) -> Tensor<Complex<f64>, DynRank>;
}

impl<S: sparse_ir::traits::StatisticsType> MatsubaraSamplingTrait for MatsubaraSampling<S> {
    fn fit_nd(&self, values: &Tensor<Complex<f64>, DynRank>, dim: usize) -> Tensor<Complex<f64>, DynRank> {
        MatsubaraSampling::fit_nd(self, None, values, dim)
    }

    fn evaluate_nd(&self, coeffs: &Tensor<Complex<f64>, DynRank>, dim: usize) -> Tensor<Complex<f64>, DynRank> {
        MatsubaraSampling::evaluate_nd(self, None, coeffs, dim)
    }

    fn evaluate_nd_real(&self, coeffs: &Tensor<f64, DynRank>, dim: usize) -> Tensor<Complex<f64>, DynRank> {
        MatsubaraSampling::evaluate_nd_real(self, None, coeffs, dim)
    }
}

impl<S: sparse_ir::traits::StatisticsType> MatsubaraSamplingTrait for MatsubaraSamplingPositiveOnly<S> {
    fn fit_nd(&self, values: &Tensor<Complex<f64>, DynRank>, dim: usize) -> Tensor<Complex<f64>, DynRank> {
        // For positive_only, fit returns real coefficients, but we need complex
        // So we convert real to complex
        let real_coeffs = MatsubaraSamplingPositiveOnly::fit_nd(self, None, values, dim);
        let shape_vec: Vec<usize> = real_coeffs.shape().dims().to_vec();
        let mut complex_coeffs = Tensor::<Complex<f64>, DynRank>::zeros(&shape_vec[..]);
        for i in 0..real_coeffs.len() {
            complex_coeffs[i] = Complex::new(real_coeffs[i], 0.0);
        }
        complex_coeffs
    }

    fn evaluate_nd(&self, coeffs: &Tensor<Complex<f64>, DynRank>, dim: usize) -> Tensor<Complex<f64>, DynRank> {
        // For positive_only, we need to convert complex to real first
        let shape_vec: Vec<usize> = coeffs.shape().dims().to_vec();
        let mut real_coeffs = Tensor::<f64, DynRank>::zeros(&shape_vec[..]);
        for i in 0..coeffs.len() {
            real_coeffs[i] = coeffs[i].re;
        }
        MatsubaraSamplingPositiveOnly::evaluate_nd(self, None, &real_coeffs, dim)
    }

    fn evaluate_nd_real(&self, coeffs: &Tensor<f64, DynRank>, dim: usize) -> Tensor<Complex<f64>, DynRank> {
        MatsubaraSamplingPositiveOnly::evaluate_nd(self, None, coeffs, dim)
    }
}

/// Create test data with complex values
fn create_test_data_complex(dim0: usize, dim1: usize) -> Tensor<Complex<f64>, DynRank> {
    // Create a simple test pattern: pole at 0.5 * omega_max
    // For benchmark purposes, we use a simple pattern
    let shape = vec![dim0, dim1];
    let mut data = Tensor::<Complex<f64>, DynRank>::zeros(&shape[..]);

    // Fill with simple test pattern
    for i in 0..dim0 {
        for j in 0..dim1 {
            let re = ((i + j) as f64) * 0.001;
            let im = ((i * j) as f64) * 0.001;
            data[&[i, j][..]] = Complex::new(re, im);
        }
    }
    data
}

/// Create test data with real values
fn create_test_data_real(dim0: usize, dim1: usize) -> Tensor<f64, DynRank> {
    let shape = vec![dim0, dim1];
    let mut data = Tensor::<f64, DynRank>::zeros(&shape[..]);

    // Fill with simple test pattern
    for i in 0..dim0 {
        for j in 0..dim1 {
            data[&[i, j][..]] = ((i + j) as f64) * 0.001;
        }
    }
    data
}

fn benchmark_internal(beta: f64, epsilon: f64) {
    let omega_max = 1.0; // Ultraviolet cutoff
    let extra_size = 1000; // dimension of the extra space
    let nrun = 10000; // Number of runs to average over

    println!("Benchmark (positive only = false)");
    benchmark(beta, omega_max, epsilon, extra_size, nrun, false);
    println!("\n");

    println!("Benchmark (positive only = true)");
    benchmark(beta, omega_max, epsilon, extra_size, nrun, true);
    println!("\n");
}

fn main() {
    println!("Benchmark (beta = 1e+3, epsilon = 1e-6)");
    benchmark_internal(1e+3, 1e-6);
    println!("\n");

    println!("Benchmark (beta = 1e+5, epsilon = 1e-10)");
    benchmark_internal(1e+5, 1e-10);
}

