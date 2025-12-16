//! Example: Using sparse-ir-rs C-API directly (equivalent to debug.py using pylibsparseir)
//!
//! This demonstrates how to use the C-API functions directly from Rust,
//! matching the low-level API usage in Python's pylibsparseir.

use sparse_ir_capi::{
    SPIR_COMPUTATION_SUCCESS, SPIR_STATISTICS_FERMIONIC, SPIR_TWORK_FLOAT64X2,
    spir_basis_get_default_taus, spir_basis_get_n_default_taus, spir_basis_get_size,
    spir_basis_new, spir_basis_release, spir_kernel_release, spir_logistic_kernel_new,
    spir_sampling_release, spir_sve_result_new, spir_sve_result_release, spir_tau_sampling_new,
};

fn main() {
    let t = 0.1; // temperature
    let beta = 1.0 / t;
    let wmax = 10.0;
    let ir_tol = 1e-10;

    // Step 1: Create kernel
    let lambda_ = beta * wmax;
    let mut status = 0;
    let kernel = unsafe { spir_logistic_kernel_new(lambda_, &mut status as *mut i32) };
    if status != SPIR_COMPUTATION_SUCCESS || kernel.is_null() {
        panic!("Failed to create logistic kernel: status = {}", status);
    }
    println!("✅ Created kernel with lambda = {}", lambda_);

    // Step 2: Compute SVE
    let sve = unsafe {
        spir_sve_result_new(
            kernel,
            ir_tol,
            -1, // lmax: -1 means auto
            -1, // n_gauss: -1 means auto
            SPIR_TWORK_FLOAT64X2,
            &mut status as *mut i32,
        )
    };
    if status != SPIR_COMPUTATION_SUCCESS || sve.is_null() {
        unsafe {
            spir_kernel_release(kernel);
        }
        panic!("Failed to create SVE result: status = {}", status);
    }
    println!("✅ Computed SVE with tolerance = {}", ir_tol);

    // Step 3: Create fermionic basis
    let basis_f = unsafe {
        spir_basis_new(
            SPIR_STATISTICS_FERMIONIC,
            beta,
            wmax,
            ir_tol,
            kernel,
            sve,
            -1, // max_size: -1 means no limit
            &mut status as *mut i32,
        )
    };
    if status != SPIR_COMPUTATION_SUCCESS || basis_f.is_null() {
        unsafe {
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
        }
        panic!("Failed to create basis: status = {}", status);
    }
    // Get basis size
    let mut basis_size = 0;
    let status_size = spir_basis_get_size(basis_f, &mut basis_size as *mut i32);
    if status_size != SPIR_COMPUTATION_SUCCESS {
        unsafe {
            spir_basis_release(basis_f);
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
        }
        panic!("Failed to get basis size: status = {}", status_size);
    }
    println!("✅ Created fermionic basis (size = {})", basis_size);

    // Step 4: Get default tau sampling points from basis
    let mut n_tau_points = 0;
    let status_n = spir_basis_get_n_default_taus(basis_f, &mut n_tau_points as *mut i32);
    if status_n != SPIR_COMPUTATION_SUCCESS {
        unsafe {
            spir_basis_release(basis_f);
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
        }
        panic!(
            "Failed to get number of default tau points: status = {}",
            status_n
        );
    }

    let mut sampling_points = vec![0.0; n_tau_points as usize];
    let status_tau = spir_basis_get_default_taus(basis_f, sampling_points.as_mut_ptr());
    if status_tau != SPIR_COMPUTATION_SUCCESS {
        unsafe {
            spir_basis_release(basis_f);
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
        }
        panic!("Failed to get default tau points: status = {}", status_tau);
    }

    // Step 5: Create tau sampling with default sampling points
    let smpl_tau = spir_tau_sampling_new(
        basis_f,
        sampling_points.len() as i32,
        sampling_points.as_ptr(),
        &mut status as *mut i32,
    );
    if status != SPIR_COMPUTATION_SUCCESS || smpl_tau.is_null() {
        unsafe {
            spir_basis_release(basis_f);
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
        }
        panic!("Failed to create tau sampling: status = {}", status);
    }
    println!(
        "✅ Created tau sampling with {} default sampling points",
        sampling_points.len()
    );

    // Cleanup
    unsafe {
        spir_sampling_release(smpl_tau);
        spir_basis_release(basis_f);
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
    }
    println!("✅ Successfully completed and cleaned up all resources");
}
