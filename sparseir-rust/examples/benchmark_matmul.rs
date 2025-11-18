//! Simple matmul_par throughput benchmark.
//!
//! Run with:
//!     cargo run --example benchmark_matmul --release

use mdarray::DTensor;
use num_complex::Complex;
use sparseir_rust::gemm::{get_backend_info, matmul_par};
use std::time::Instant;

fn dummy_fit2d(values_2d: &DTensor<Complex<f64>, 2>) -> DTensor<Complex<f64>, 2> {
    // create random complex matrix with shape [values_2d.shape().0, values_2d.shape().0]
    let uh = DTensor::<Complex<f64>, 2>::from_fn(
        [values_2d.shape().0, values_2d.shape().0],
        |_idx| Complex::new(rand::random::<f64>(), rand::random::<f64>()),
    );

    matmul_par(&uh, values_2d)
}

fn main() {
    let (backend_name, is_external, is_ilp64) = get_backend_info();
    println!(
        "matmul_par backend: {} (external: {}, ilp64: {})",
        backend_name, is_external, is_ilp64
    );

    let values_2d = DTensor::<Complex<f64>, 2>::from_fn([52, 1024], |_idx| {
        Complex::new(rand::random::<f64>(), rand::random::<f64>())
    });
    let uh_values = dummy_fit2d(&values_2d);
    println!("uh_values: {:?}", uh_values.shape());

    let start = Instant::now();
    for _ in 0..20000 {
        let _ = dummy_fit2d(&values_2d);
    }
    let elapsed = start.elapsed();
    println!("Time taken: {:?}", elapsed);
}
