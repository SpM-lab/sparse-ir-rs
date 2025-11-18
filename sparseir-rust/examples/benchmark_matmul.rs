//! Simple matmul_par throughput benchmark.
//!
//! Run with:
//!     cargo run --example benchmark_matmul --release

use mdarray::DTensor;
use num_complex::Complex;
use std::time::Instant;
use sparseir_rust::gemm::matmul_par;
use faer::rand;

// Memory usage reporting
fn get_memory_usage() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let output = Command::new("ps")
            .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok();

        if let Some(output) = output {
            String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<usize>()
                .unwrap_or(0)
        } else {

            0
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

fn matmul(
    a: &DTensor<Complex<f64>, 2>,
    b: &DTensor<Complex<f64>, 2>,
) -> DTensor<Complex<f64>, 2> {
    matmul_par(a, b, None)
}

fn main() {
    let a = DTensor::<Complex<f64>, 2>::from_fn([256, 256], |_idx| {
        Complex::new(rand::random::<f64>(), rand::random::<f64>())
    });

    let b = DTensor::<Complex<f64>, 2>::from_fn([a.shape().0, a.shape().1], |_idx| {
        Complex::new(rand::random::<f64>(), rand::random::<f64>())
    });

    let mem_before = get_memory_usage();
    println!("Memory before loop: {} KB", mem_before);

    let start = Instant::now();
    for i in 0..20000 {
        matmul(&a, &b);

        // Report memory every 2000 iterations
        if i % 2000 == 0 && i > 0 {
            let mem_now = get_memory_usage();
            println!("Iteration {}: Memory = {} KB (delta: {} KB)",
                     i, mem_now, mem_now as i64 - mem_before as i64);
        }
    }
    let elapsed = start.elapsed();

    let mem_after = get_memory_usage();
    println!("Time taken: {:?}", elapsed);
    println!("Memory after loop: {} KB (delta: {} KB)",
             mem_after, mem_after as i64 - mem_before as i64);
}
