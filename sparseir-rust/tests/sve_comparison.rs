//! Comparison tests between Rust and Julia SVE implementations

use sparse_ir::{LogisticKernel, TworkType, compute_sve};

// ============================================================================
// Reference data for λ=5.0, ε=1.0e-6
// Generated from SparseIR.jl v1.0
// ============================================================================

/// Reference singular values (8 significant values for λ=5, ε=1e-6)
const REFERENCE_SVALS: [f64; 8] = [
    0.7299119550913774,
    0.38682851834442455,
    0.1095936259267352,
    0.02263801679340181,
    0.0035312837500997263,
    0.00044208295394039266,
    4.6148981858334375e-5,
    4.129184929608994e-6,
];

/// Reference u matrix: 5 test points × 8 singular functions
/// Test points: x = [-0.9, -0.5, 0.0, 0.5, 0.9]
const REFERENCE_U: [[f64; 8]; 5] = [
    [
        0.8605593593193338,
        -1.1852905059002685,
        1.0519446878631924,
        -0.7570653957841662,
        0.30865216867618306,
        0.20865950114886975,
        -0.6929848609632565,
        1.0460413561787945,
    ],
    [
        0.6683045339679808,
        -0.5276473876625923,
        -0.3420090737759205,
        0.8581513835197583,
        -0.5417179134312096,
        -0.29947299753003703,
        0.8474946653870142,
        -0.5637045385457694,
    ],
    [
        0.5961501592674083,
        2.0775316713822338e-16,
        -0.8146130792158527,
        -1.883628715386559e-15,
        0.813975162998825,
        -1.2021983271731859e-14,
        -0.807640206256603,
        2.403288637454968e-13,
    ],
    [
        0.6683045339679808,
        0.5276473876625923,
        -0.3420090737759205,
        -0.8581513835197583,
        -0.5417179134312096,
        0.29947299753003703,
        0.8474946653870142,
        0.5637045385457694,
    ],
    [
        0.8605593593193338,
        1.1852905059002685,
        1.0519446878631924,
        0.7570653957841662,
        0.30865216867618306,
        -0.20865950114886975,
        -0.6929848609632565,
        -1.0460413561787945,
    ],
];

/// Reference v matrix: 5 test points × 8 singular functions
/// Test points: y = [-0.9, -0.5, 0.0, 0.5, 0.9]
const REFERENCE_V: [[f64; 8]; 5] = [
    [
        0.4471665485427396,
        0.805992069523501,
        1.0401394241216797,
        1.078481411457772,
        0.8773652103635484,
        0.4841698089214613,
        -0.01905661796672064,
        -0.5285289988353953,
    ],
    [
        0.6696008770990369,
        0.8150855256780701,
        0.26988891972204054,
        -0.5225675865866158,
        -0.8647463859673025,
        -0.3952258364322118,
        0.45425095908698737,
        0.86136149598233,
    ],
    [
        0.9593225235542324,
        -1.2524951131469452e-14,
        -0.8978389151052752,
        1.1486613213834679e-14,
        0.8269164035575688,
        -8.933167218051167e-13,
        -0.8114684884703242,
        -3.405592619154411e-11,
    ],
    [
        0.6696008770990369,
        -0.8150855256780701,
        0.26988891972204054,
        0.5225675865866158,
        -0.8647463859673025,
        0.3952258364322118,
        0.45425095908698737,
        -0.86136149598233,
    ],
    [
        0.4471665485427396,
        -0.805992069523501,
        1.0401394241216797,
        -1.078481411457772,
        0.8773652103635484,
        -0.4841698089214613,
        -0.01905661796672064,
        0.5285289988353953,
    ],
];

#[test]
fn test_sve_singular_values_lambda_5() {
    let lambda = 5.0;
    let epsilon = 1e-6;

    println!("Testing SVE for λ={}, ε={}", lambda, epsilon);

    let kernel = LogisticKernel::new(lambda);
    let result = compute_sve(
        kernel,
        epsilon,
        None, // cutoff
        None, // max_num_svals
        TworkType::Auto,
    );

    // Load reference values
    let s_ref: Vec<f64> = REFERENCE_SVALS.to_vec();

    println!("Rust: {} singular values", result.s.len());
    println!("Julia: {} singular values", s_ref.len());

    // Print all Rust singular values
    println!("\nAll Rust singular values:");
    for (i, &s) in result.s.iter().enumerate() {
        println!("  Rust s[{}] = {:.6e}", i, s);
    }

    // Filter significant singular values in Rust result
    let threshold = epsilon * result.s[0];
    println!(
        "\nThreshold = {} * {} = {:.6e}",
        epsilon, result.s[0], threshold
    );

    let significant_rust: Vec<f64> = result
        .s
        .iter()
        .filter(|&&s| s > threshold)
        .copied()
        .collect();

    println!("Rust significant (> threshold): {}", significant_rust.len());
    println!("First Rust s[0] = {}", result.s[0]);

    // Compare number of significant singular values
    assert_eq!(
        significant_rust.len(),
        s_ref.len(),
        "Number of significant singular values mismatch"
    );

    // Compare singular values
    println!("\nComparing singular values:");
    for i in 0..significant_rust.len() {
        let rust_val = significant_rust[i];
        let julia_val = s_ref[i];
        let abs_error = (rust_val - julia_val).abs();
        let rel_error = abs_error / julia_val;

        println!(
            "  s[{}]: rust={:.6e}, julia={:.6e}, rel_err={:.2e}",
            i, rust_val, julia_val, rel_error
        );

        assert!(
            rel_error < 1e-10,
            "Singular value {} mismatch: rust={}, julia={}, rel_error={}",
            i,
            rust_val,
            julia_val,
            rel_error
        );
    }

    println!(
        "\n✓ All {} singular values matched within 1e-10 relative error",
        significant_rust.len()
    );
}

#[test]
fn test_sve_singular_functions_lambda_5() {
    let lambda = 5.0;
    let epsilon = 1e-6;

    println!(
        "Testing SVE singular functions for λ={}, ε={}",
        lambda, epsilon
    );

    let kernel = LogisticKernel::new(lambda);
    let result = compute_sve(kernel, epsilon, None, None, TworkType::Auto);

    // Test points
    let x_test = [-0.9, -0.5, 0.0, 0.5, 0.9];

    // Load reference u and v values
    let u_ref = mdarray::DTensor::<f64, 2>::from_fn([5, 8], |idx| REFERENCE_U[idx[0]][idx[1]]);
    let v_ref = mdarray::DTensor::<f64, 2>::from_fn([5, 8], |idx| REFERENCE_V[idx[0]][idx[1]]);

    let n_funcs = u_ref.shape().1;
    println!("Comparing first {} singular functions", n_funcs);

    // Compare u functions
    println!("\nComparing u functions:");
    for i in 0..n_funcs {
        let mut max_error: f64 = 0.0;

        for (j, &x) in x_test.iter().enumerate() {
            let u_rust = result.u.get_polys()[i].evaluate(x);
            let u_julia = u_ref[[j, i]];
            let abs_error = (u_rust - u_julia).abs();
            max_error = max_error.max(abs_error);
        }

        println!("  u[{}]: max_error={:.2e}", i, max_error);

        // Allow slightly larger error for higher-order singular functions
        let tolerance = if i < 5 { 1e-10 } else { 1e-9 };

        assert!(
            max_error < tolerance,
            "u[{}] error too large: {:.2e} (tolerance: {:.2e})",
            i,
            max_error,
            tolerance
        );
    }

    // Compare v functions
    println!("\nComparing v functions:");
    for i in 0..n_funcs {
        let mut max_error: f64 = 0.0;

        for (j, &x) in x_test.iter().enumerate() {
            let v_rust = result.v.get_polys()[i].evaluate(x);
            let v_julia = v_ref[[j, i]];
            let abs_error = (v_rust - v_julia).abs();
            max_error = max_error.max(abs_error);
        }

        println!("  v[{}]: max_error={:.2e}", i, max_error);

        // Allow slightly larger error for higher-order singular functions
        let tolerance = if i < 5 { 1e-10 } else { 1e-9 };

        assert!(
            max_error < tolerance,
            "v[{}] error too large: {:.2e} (tolerance: {:.2e})",
            i,
            max_error,
            tolerance
        );
    }

    println!("\n✓ All singular functions matched within 1e-10 absolute error");
}
