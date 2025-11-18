//! Simple Faer.matmul(a, b).parallelize().eval() throughput benchmark.
//!
//! Run with:
//!     cargo run --example matmul --release

use mdarray::DTensor;
use num_complex::Complex;
use std::time::Instant;

// Memory usage reporting
#[cfg(target_os = "macos")]
fn get_memory_usage() -> usize {
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

#[cfg(target_os = "linux")]
fn get_memory_usage() -> usize {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    if let Ok(file) = File::open("/proc/self/status") {
        for line in BufReader::new(file).lines().flatten() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return parts[1].parse::<usize>().unwrap_or(0);
                }
            }
        }
    }
    0
}

#[cfg(all(not(target_os = "macos"), not(target_os = "linux")))]
fn get_memory_usage() -> usize {
    0
}

fn matmul(
    a: &DTensor<Complex<f64>, 2>,
    b: &DTensor<Complex<f64>, 2>,
) -> DTensor<Complex<f64>, 2> {
    use mdarray_linalg::prelude::MatMul;
    use mdarray_linalg::matmul::MatMulBuilder;
    use mdarray_linalg_faer::Faer;
    Faer.matmul(a, b).parallelize().eval()
}

fn main() {
    let a = DTensor::<Complex<f64>, 2>::from_elem([256, 256], Complex::new(1.0, 1.0));
    let b = DTensor::<Complex<f64>, 2>::from_elem([256, 256], Complex::new(1.0, 1.0));

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