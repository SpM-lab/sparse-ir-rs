//! Benchmark for sparseir-rust operations
//!
//! This benchmark is a Rust port of the C benchmark in capi_benchmark/benchmark1.c
//! It measures the performance of fit_zz, eval_zz, and eval_dz operations
//! for both Matsubara and Tau sampling.

use sparseir_rust::{
    FiniteTempBasis, Fermionic, LogisticKernel, MatsubaraSampling,
};
use num_complex::Complex;
use mdarray::{DTensor, DynRank, Tensor};
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

    // Get Matsubara frequency sampling points
    let matsubara_points = basis.default_matsubara_sampling_points(positive_only);
    let n_matsubara = matsubara_points.len();
    println!("n_matsubara: {}", n_matsubara);

    // Create sampling object for Matsubara domain
    let matsubara_sampling = Box::new(MatsubaraSampling::with_sampling_points(&basis, matsubara_points));

    // Create test data arrays
    // g_matsu_z: [n_matsubara, extra_size] (complex)
    let g_matsu_z = create_test_data_complex(n_matsubara, extra_size);

    let values_2d_dyn = g_matsu_z.reshape(&[n_matsubara, extra_size][..]).to_tensor();
    let values_2d = DTensor::<Complex<f64>, 2>::from_fn([n_matsubara, extra_size], |idx| {
        values_2d_dyn[&[idx[0], idx[1]][..]]
    });
    println!("values_2d: {:?}", values_2d.shape());

    // Test: matsubara, fit_zz (complex values -> complex coefficients)
    // First run to warm up
    {
        let _result = matsubara_sampling.fit_2d(&values_2d);
    }

    let bench = Benchmark::start("fit_zz (Matsubara)");
    for _ in 0..nrun {
        let _result = matsubara_sampling.fit_2d(&values_2d);
    }
    let elapsed = bench.end();
    println!("  Average per run: {:10.6} ms", elapsed / nrun as f64);

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

fn benchmark_internal(beta: f64, epsilon: f64) {
    let omega_max = 1.0; // Ultraviolet cutoff
    let extra_size = 1000; // dimension of the extra space
    let nrun = 10000; // Number of runs to average over

    println!("Benchmark (positive only = false)");
    benchmark(beta, omega_max, epsilon, extra_size, nrun, false);
    println!("\n");
}

fn main() {
    println!("Benchmark (beta = 1e+3, epsilon = 1e-6)");
    benchmark_internal(1e+3, 1e-6);
    println!("\n");
}

