//! Tests for README code examples
//!
//! This file contains tests that verify all code examples in the README compile and run correctly.

use sparse_ir::*;

#[test]
fn test_basic_example() -> Result<(), Error> {
    // Basic Example from README

    // Create a finite temperature basis
    let beta = 10.0;
    let lambda = 10.0; // beta * omega_max
    let kernel = LogisticKernel::new(lambda)?;
    let basis = FermionicBasis::new(kernel, beta, None, None)?;

    // Generate sampling points
    let sampling = TauSampling::new(&basis)?;

    // Use the basis for calculations
    let tau_points = sampling.sampling_points();
    println!("Generated {} sampling points", tau_points.len());

    // Verify that we got some sampling points
    assert!(!tau_points.is_empty());
    assert!(tau_points.len() >= basis.size());
    Ok(())
}

#[test]
fn test_sve_example() -> Result<(), Error> {
    // SVE Example from README (corrected)

    // Create a kernel for analytical continuation
    let kernel = LogisticKernel::new(1.0)?;

    // Compute the SVE to an accuracy of 1e-12, keeping at most 100 singular values
    let sve_result = compute_sve(kernel, Some(1e-12), None, Some(100), TworkType::Auto)?;

    println!("SVE computed with {} singular values", sve_result.s().len());

    // Verify that we got some singular values
    assert!(!sve_result.s().is_empty());
    assert!(sve_result.s().len() <= 100);
    Ok(())
}
