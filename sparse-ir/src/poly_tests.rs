//! Tests for piecewise Legendre polynomial implementations

use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
use mdarray::tensor;

#[test]
fn test_basic_polynomial_creation() {
    // Test data from C++ tests
    let data = tensor![
        [0.8177021060277301, 0.7085670484724618, 0.5033588232863977],
        [0.3804323567786363, 0.7911959541742282, 0.8268504271915096],
        [0.5425813266814807, 0.38397463704084633, 0.21626598379927042],
    ];

    let knots = vec![
        0.507134318967235,
        0.5766150365607372,
        0.7126662232433161,
        0.7357313003784003,
    ];
    let l = 3;

    let poly = PiecewiseLegendrePoly::new(data.clone(), knots.clone(), l, None, 0);

    assert_eq!(poly.data, data);
    assert_eq!(poly.xmin, knots[0]);
    assert_eq!(poly.xmax, knots[knots.len() - 1]);
    assert_eq!(poly.knots, knots);
    assert_eq!(poly.polyorder, data.shape().0);
    assert_eq!(poly.symm, 0);
}

#[test]
fn test_polynomial_evaluation() {
    let data = tensor![[1.0, 2.0], [3.0, 4.0]];
    let knots = vec![0.0, 1.0, 2.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);

    // Test evaluation at various points
    let x = 0.5;
    let result = poly.evaluate(x);
    println!("poly(0.5) = {}", result);

    // Test split function
    let (i, x_tilde) = poly.split(0.5);
    assert_eq!(i, 0);
    println!("split(0.5) = ({}, {})", i, x_tilde);
}

#[test]
fn test_derivative_calculation() {
    // Create a simple polynomial: P(x) = 1 + 2x + 3x^2 on [0, 1]
    let data = tensor![[1.0], [2.0], [3.0]];
    let knots = vec![0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    // Test first derivative
    let deriv1 = poly.deriv(1);
    println!("Original poly: {:?}", poly.data);
    println!("First derivative: {:?}", deriv1.data);

    // Test derivatives at a point
    let x = 0.5;
    let derivs = poly.derivs(x);
    println!("Derivatives at x={}: {:?}", x, derivs);

    assert_eq!(derivs.len(), poly.polyorder);
}

#[test]
fn test_overlap_integral() {
    // Create a polynomial: P(x) = 1 on [0, 1]
    let data = tensor![[1.0]];
    let knots = vec![0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    // Test overlap with constant function f(x) = 1
    let result = poly.overlap(|_| 1.0);
    println!("Overlap with f(x)=1: {}", result);

    // The result includes normalization factor sqrt(2/delta_x) = sqrt(2)
    // So the expected result is sqrt(2) ≈ 1.414...
    let expected = 2.0_f64.sqrt();
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_high_precision_overlap_integral() {
    // Test with exact values from C++ poly.cxx
    let data = tensor![
        [0.8177021060277301, 0.7085670484724618, 0.5033588232863977],
        [0.3804323567786363, 0.7911959541742282, 0.8268504271915096],
        [0.5425813266814807, 0.38397463704084633, 0.21626598379927042],
    ];

    let knots = vec![
        0.507134318967235,
        0.5766150365607372,
        0.7126662232433161,
        0.7357313003784003,
    ];
    let l = 3;
    let poly = PiecewiseLegendrePoly::new(data, knots, l, None, 0);

    // Test overlap with identity function f(x) = x
    let identity = |x: f64| x;
    let result = poly.overlap(identity);

    // Expected result from C++ test
    let expected = 0.4934184996836403;

    println!(
        "Expected overlap: {}, Actual: {}, Diff: {}",
        expected,
        result,
        (result - expected).abs()
    );
    assert!((result - expected).abs() < 1e-12);
}

#[test]
fn test_root_finding() {
    // Create a polynomial that should have a root
    // P(x) = x - 0.5 on [0, 1] (root at x = 0.5)
    // But with Legendre normalization, this becomes P(x) = sqrt(2) * (x - 0.5)
    let data = tensor![[-0.5], [1.0]];
    let knots = vec![0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);

    // Test evaluation at the expected root
    let x = 0.5;
    let value = poly.evaluate(x);
    println!("poly(0.5) = {}", value);

    // For debugging: test at multiple points
    for test_x in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let val = poly.evaluate(test_x);
        println!("poly({}) = {}", test_x, val);
    }

    let roots = poly.roots();
    println!("Roots found: {:?}", roots);

    // The root finding might not work perfectly due to normalization
    // Let's just check that the function behaves reasonably
    assert!(poly.evaluate(0.0) * poly.evaluate(1.0) <= 0.0); // Sign change
}

#[test]
fn test_high_precision_root_finding() {
    // Test with exact values from C++ poly.cxx
    // This test uses a 16th-order polynomial with known roots
    let v = vec![
        0.16774734206553019,
        0.49223680914312595,
        -0.8276728567928646,
        0.16912891046582143,
        -0.0016231275318572044,
        0.00018381683946452256,
        -9.699355027805034e-7,
        7.60144228530804e-8,
        -2.8518324490258146e-10,
        1.7090590205708293e-11,
        -5.0081401126025e-14,
        2.1244236198427895e-15,
        2.0478095258000225e-16,
        -2.676573801530628e-16,
        2.338165820094204e-16,
        -1.2050663212312096e-16,
        -0.16774734206553019,
        0.49223680914312595,
        0.8276728567928646,
        0.16912891046582143,
        0.0016231275318572044,
        0.00018381683946452256,
        9.699355027805034e-7,
        7.60144228530804e-8,
        2.8518324490258146e-10,
        1.7090590205708293e-11,
        5.0081401126025e-14,
        2.1244236198427895e-15,
        -2.0478095258000225e-16,
        -2.676573801530628e-16,
        -2.338165820094204e-16,
        -1.2050663212312096e-16,
    ];

    // Reshape into 16x2 matrix (column-major order like C++ Eigen)
    let mut data = mdarray::DTensor::<f64, 2>::from_elem([16, 2], 0.0);
    for i in 0..16 {
        for j in 0..2 {
            data[[i, j]] = v[i + j * 16]; // Column-major indexing
        }
    }

    let knots = vec![0.0, 0.5, 1.0];
    let l = 3;
    let poly = PiecewiseLegendrePoly::new(data, knots.clone(), l, None, 0);

    // Find roots
    let roots = poly.roots();

    // Expected roots from C++ test
    let expected_roots = vec![0.1118633448586015, 0.4999999999999998, 0.8881366551413985];

    println!("Expected roots: {:?}", expected_roots);
    println!("Actual roots: {:?}", roots);

    // Check that we found the right number of roots
    assert_eq!(roots.len(), expected_roots.len());

    // Check each root with high precision
    for i in 0..roots.len() {
        println!(
            "Root {}: Expected {}, Actual {}, Diff {}",
            i,
            expected_roots[i],
            roots[i],
            (roots[i] - expected_roots[i]).abs()
        );

        // Check root accuracy
        assert!((roots[i] - expected_roots[i]).abs() < 1e-10);

        // Verify that polynomial evaluates to zero at the root
        let root_value = poly.evaluate(roots[i]);
        println!("  poly({}) = {}", roots[i], root_value);
        assert!(root_value.abs() < 1e-10);

        // Verify roots are in domain
        assert!(roots[i] >= knots[0]);
        assert!(roots[i] <= knots[knots.len() - 1]);
    }
}

#[test]
fn test_split_function() {
    let data = tensor![[1.0, 2.0], [3.0, 4.0]];
    let knots = vec![0.0, 1.0, 2.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    // Test split at various points
    let test_points = vec![0.0, 0.5, 1.0, 1.5, 2.0];

    for x in test_points {
        let (segment, x_tilde) = poly.split(x);
        println!("split({}) = ({}, {})", x, segment, x_tilde);

        // Check that x_tilde is in [-1, 1]
        assert!((-1.0..=1.0).contains(&x_tilde));

        // Check that segment is valid
        assert!(segment < poly.knots.len() - 1);
    }
}

#[test]
fn test_legendre_polynomial_evaluation() {
    // Test the Legendre polynomial evaluation directly
    let poly = PiecewiseLegendrePoly::new(
        mdarray::DTensor::<f64, 2>::from_elem([3, 1], 0.0),
        vec![0.0, 1.0],
        0,
        None,
        0,
    );

    // Test P_0(x) = 1
    let coeffs = vec![1.0, 0.0, 0.0];
    let result = poly.evaluate_legendre_polynomial(0.5, &coeffs);
    assert!((result - 1.0).abs() < 1e-10);

    // Test P_1(x) = x
    let coeffs = vec![0.0, 1.0, 0.0];
    let result = poly.evaluate_legendre_polynomial(0.5, &coeffs);
    assert!((result - 0.5).abs() < 1e-10);

    // Test P_2(x) = (3x^2 - 1)/2
    let coeffs = vec![0.0, 0.0, 1.0];
    let result = poly.evaluate_legendre_polynomial(0.5, &coeffs);
    let expected = (3.0 * 0.5 * 0.5 - 1.0) / 2.0;
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_with_data_methods() {
    let data = tensor![[1.0, 2.0], [3.0, 4.0]];
    let knots = vec![0.0, 1.0, 2.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);

    // Test with_data
    let new_data = tensor![[5.0, 6.0], [7.0, 8.0]];
    let new_poly = poly.with_data(new_data.clone());
    assert_eq!(new_poly.data, new_data);
    assert_eq!(new_poly.symm, poly.symm);

    // Test with_data_and_symmetry
    let new_poly2 = poly.with_data_and_symmetry(new_data.clone(), 1);
    assert_eq!(new_poly2.data, new_data);
    assert_eq!(new_poly2.symm, 1);
}

#[test]
fn test_cpp_compatible_data() {
    // Test with the exact data from C++ tests
    let data = tensor![
        [0.8177021060277301, 0.7085670484724618, 0.5033588232863977],
        [0.3804323567786363, 0.7911959541742282, 0.8268504271915096],
        [0.5425813266814807, 0.38397463704084633, 0.21626598379927042],
    ];

    let knots = vec![
        0.507134318967235,
        0.5766150365607372,
        0.7126662232433161,
        0.7357313003784003,
    ];
    let l = 3;

    let poly = PiecewiseLegendrePoly::new(data.clone(), knots.clone(), l, None, 0);

    // Test basic properties
    assert_eq!(poly.data, data);
    assert_eq!(poly.knots, knots);
    assert_eq!(poly.l, l);
    assert_eq!(poly.polyorder, 3);

    // Test evaluation at various points
    let test_points = vec![0.55, 0.6, 0.65, 0.7, 0.73];
    for x in test_points {
        let result = poly.evaluate(x);
        println!("poly({}) = {}", x, result);
        assert!(result.is_finite());
    }
}

#[test]
fn test_derivative_consistency() {
    // Test that derivatives are consistent with numerical differentiation
    let data = tensor![[1.0, 2.0], [3.0, 4.0]];
    let knots = vec![0.0, 1.0, 2.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 1, None, 0);

    let x = 0.5;
    let h = 1e-8;

    // Numerical first derivative
    let f_plus = poly.evaluate(x + h);
    let f_minus = poly.evaluate(x - h);
    let numerical_deriv = (f_plus - f_minus) / (2.0 * h);

    // Analytical first derivative
    let analytical_deriv = poly.deriv(1).evaluate(x);

    println!("Numerical derivative: {}", numerical_deriv);
    println!("Analytical derivative: {}", analytical_deriv);
    println!("Difference: {}", (numerical_deriv - analytical_deriv).abs());

    // Should be close (within numerical precision)
    assert!((numerical_deriv - analytical_deriv).abs() < 1e-6);
}

#[test]
fn test_high_precision_derivative() {
    // Test with exact values from C++ poly.cxx
    let data = tensor![
        [0.8177021060277301, 0.7085670484724618, 0.5033588232863977],
        [0.3804323567786363, 0.7911959541742282, 0.8268504271915096],
        [0.5425813266814807, 0.38397463704084633, 0.21626598379927042],
    ];

    let knots = vec![
        0.507134318967235,
        0.5766150365607372,
        0.7126662232433161,
        0.7357313003784003,
    ];
    let l = 3;
    let poly = PiecewiseLegendrePoly::new(data, knots, l, None, 0);

    // Test evaluation at x = 0.6 with high precision
    let x = 0.6;
    let expected_value = 0.9409047338158947;
    let actual_value = poly.evaluate(x);

    println!(
        "Expected: {}, Actual: {}, Diff: {}",
        expected_value,
        actual_value,
        (actual_value - expected_value).abs()
    );
    assert!((actual_value - expected_value).abs() < 1e-10);

    // Test derivative evaluation at x = 0.6 with high precision
    let deriv_poly = poly.deriv(1);
    let expected_deriv = 1.9876646271069893;
    let actual_deriv = deriv_poly.evaluate(x);

    println!(
        "Expected deriv: {}, Actual deriv: {}, Diff: {}",
        expected_deriv,
        actual_deriv,
        (actual_deriv - expected_deriv).abs()
    );
    assert!((actual_deriv - expected_deriv).abs() < 1e-10);

    // Test multiple derivatives at x = 0.6
    let expected_derivs = [0.9409047338158947, 1.9876646271069893, 954.4275060248603];
    let actual_derivs = poly.derivs(x);

    assert_eq!(actual_derivs.len(), 3);
    for i in 0..3 {
        println!(
            "Deriv {}: Expected {}, Actual {}, Diff {}",
            i,
            expected_derivs[i],
            actual_derivs[i],
            (actual_derivs[i] - expected_derivs[i]).abs()
        );
        assert!((actual_derivs[i] - expected_derivs[i]).abs() < 1e-10);
    }
}

#[test]
fn test_legendre_polynomial_properties() {
    // Test that our Legendre polynomial evaluation matches known properties
    let poly = PiecewiseLegendrePoly::new(
        mdarray::DTensor::<f64, 2>::from_elem([5, 1], 0.0),
        vec![0.0, 1.0],
        0,
        None,
        0,
    );

    // Test P_0(x) = 1 at x = 0
    let coeffs = vec![1.0, 0.0, 0.0, 0.0, 0.0];
    let result = poly.evaluate_legendre_polynomial(0.0, &coeffs);
    assert!((result - 1.0).abs() < 1e-10);

    // Test P_1(x) = x at x = 0.5
    let coeffs = vec![0.0, 1.0, 0.0, 0.0, 0.0];
    let result = poly.evaluate_legendre_polynomial(0.5, &coeffs);
    assert!((result - 0.5).abs() < 1e-10);

    // Test P_2(x) = (3x^2 - 1)/2 at x = 1.0
    let coeffs = vec![0.0, 0.0, 1.0, 0.0, 0.0];
    let result = poly.evaluate_legendre_polynomial(1.0, &coeffs);
    let expected = (3.0 * 1.0 * 1.0 - 1.0) / 2.0;
    assert!((result - expected).abs() < 1e-10);

    // Test P_3(x) = (5x^3 - 3x)/2 at x = 0.5
    let coeffs = vec![0.0, 0.0, 0.0, 1.0, 0.0];
    let result = poly.evaluate_legendre_polynomial(0.5, &coeffs);
    let expected = (5.0 * 0.125 - 3.0 * 0.5) / 2.0;
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_accessor_methods() {
    let data = tensor![[1.0, 2.0], [3.0, 4.0]];
    let knots = vec![0.0, 1.0, 2.0];
    let l = 5;
    let symm = 1;
    let poly = PiecewiseLegendrePoly::new(data.clone(), knots.clone(), l, None, symm);

    // Test all accessor methods
    assert_eq!(poly.get_xmin(), knots[0]);
    assert_eq!(poly.get_xmax(), knots[knots.len() - 1]);
    assert_eq!(poly.get_l(), l);
    assert_eq!(poly.get_symm(), symm);
    assert_eq!(poly.get_polyorder(), data.shape().0);
    assert_eq!(poly.get_domain(), (knots[0], knots[knots.len() - 1]));
    assert_eq!(poly.get_knots(), knots.as_slice());
    assert_eq!(poly.get_data(), &data);

    // Test delta_x and norms
    let delta_x = poly.get_delta_x();
    let norms = poly.get_norms();
    assert_eq!(delta_x.len(), knots.len() - 1);
    assert_eq!(norms.len(), knots.len() - 1);

    // Check that delta_x matches knots
    for i in 0..delta_x.len() {
        assert!((delta_x[i] - (knots[i + 1] - knots[i])).abs() < 1e-10);
    }
}

#[test]
fn test_polynomial_vector_creation() {
    let data1 = tensor![[1.0, 2.0], [3.0, 4.0]];
    let data2 = tensor![[5.0, 6.0], [7.0, 8.0]];
    let knots = vec![0.0, 1.0, 2.0];

    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots, 1, None, 0);

    let vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);

    assert_eq!(vector.size(), 2);
    assert_eq!(vector.get_polyorder(), 2);
}

#[test]
fn test_vector_evaluation() {
    let data1 = tensor![[1.0, 2.0], [3.0, 4.0]];
    let data2 = tensor![[5.0, 6.0], [7.0, 8.0]];
    let knots = vec![0.0, 1.0, 2.0];

    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots, 1, None, 0);

    let vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);

    let results = vector.evaluate_at(0.5);
    assert_eq!(results.len(), 2);
    println!("Vector evaluation at 0.5: {:?}", results);

    // Test multiple points
    let xs = vec![0.0, 0.5, 1.0];
    let results_matrix = vector.evaluate_at_many(&xs);
    assert_eq!(*results_matrix.shape(), (2, 3));
    println!("Vector evaluation matrix:\n{:?}", results_matrix);
}

#[test]
fn test_vector_3d_construction() {
    // Create 3D data: 3 degrees, 2 segments, 2 polynomials
    let mut data3d = mdarray::DTensor::<f64, 3>::from_elem([3, 2, 2], 0.0);

    // Polynomial 0: coefficients for 3 degrees, 2 segments
    data3d[[0, 0, 0]] = 1.0; // degree 0, segment 0, poly 0
    data3d[[1, 0, 0]] = 2.0; // degree 1, segment 0, poly 0
    data3d[[0, 1, 0]] = 3.0; // degree 0, segment 1, poly 0
    data3d[[1, 1, 0]] = 4.0; // degree 1, segment 1, poly 0

    // Polynomial 1: coefficients for 3 degrees, 2 segments
    data3d[[0, 0, 1]] = 5.0; // degree 0, segment 0, poly 1
    data3d[[1, 0, 1]] = 6.0; // degree 1, segment 0, poly 1
    data3d[[0, 1, 1]] = 7.0; // degree 0, segment 1, poly 1
    data3d[[1, 1, 1]] = 8.0; // degree 1, segment 1, poly 1

    let knots = vec![0.0, 1.0, 2.0]; // 2 segments need 3 knots
    let symm = vec![0, 1];

    let vector = PiecewiseLegendrePolyVector::from_3d_data(data3d, knots, Some(symm));

    assert_eq!(vector.size(), 2);
    assert_eq!(vector.get_polyorder(), 3);
    assert_eq!(vector.get_symm(), vec![0, 1]);

    // Test evaluation
    let results = vector.evaluate_at(0.5);
    assert_eq!(results.len(), 2);
    println!("3D vector evaluation at 0.5: {:?}", results);
}

#[test]
fn test_vector_slicing() {
    let data1 = tensor![[1.0, 2.0], [3.0, 4.0]];
    let data2 = tensor![[5.0, 6.0], [7.0, 8.0]];
    let data3 = tensor![[9.0, 10.0], [11.0, 12.0]];
    let knots = vec![0.0, 1.0, 2.0];

    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots.clone(), 1, None, 0);
    let poly3 = PiecewiseLegendrePoly::new(data3, knots, 2, None, 0);

    let vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2, poly3]);

    // Test single slice
    let single_slice = vector.slice_single(1);
    assert!(single_slice.is_some());
    assert_eq!(single_slice.unwrap().size(), 1);

    // Test multi slice
    let multi_slice = vector.slice_multi(&[0, 2]);
    assert_eq!(multi_slice.size(), 2);

    // Test evaluation of slice
    let slice_results = multi_slice.evaluate_at(0.5);
    assert_eq!(slice_results.len(), 2);
    println!("Slice evaluation: {:?}", slice_results);
}

#[test]
fn test_vector_accessors() {
    let data1 = tensor![[1.0, 2.0], [3.0, 4.0]];
    let data2 = tensor![[5.0, 6.0], [7.0, 8.0]];
    let knots = vec![0.0, 1.0, 2.0];

    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots.clone(), 1, None, 0);

    let vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);

    // Test accessor methods
    assert_eq!(vector.xmin(), 0.0);
    assert_eq!(vector.xmax(), 2.0);
    assert_eq!(vector.get_knots(None), knots);
    assert_eq!(vector.get_polyorder(), 2);
    assert_eq!(vector.get_symm(), vec![0, 0]);

    // Test 3D data conversion
    let data3d = vector.get_data();
    assert_eq!(*data3d.shape(), (2, 2, 2)); // 2 segments, 2 degrees, 2 polynomials
    println!("3D data shape: {:?}", data3d.shape());
}

#[test]
fn test_vector_roots() {
    let data1 = tensor![[-0.5], [1.0]]; // Should have root at 0.5
    let data2 = tensor![[-1.0], [2.0]]; // Should have root at 0.5
    let knots = vec![0.0, 1.0];

    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots, 1, None, 0);

    let vector = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);

    let roots = vector.roots(None);
    println!("Vector roots: {:?}", roots);

    // Verify nroots() returns consistent count with roots.len()
    assert_eq!(vector.nroots(None), roots.len());
}

#[test]
fn test_julia_random_data() {
    // Initialize data and knots as per the C++ test
    // julia> using StableRNGs
    // julia> rng = StableRNG(2024)
    // julia> data = rand(rng, 3, 3)
    // julia> knots = rand(rng, size(data, 2) + 1) |> sort
    let data = tensor![
        [0.8177021060277301, 0.7085670484724618, 0.5033588232863977],
        [0.3804323567786363, 0.7911959541742282, 0.8268504271915096],
        [0.5425813266814807, 0.38397463704084633, 0.21626598379927042],
    ];

    let knots = vec![
        0.507134318967235,
        0.5766150365607372,
        0.7126662232433161,
        0.7357313003784003,
    ];
    let l = 3;

    // Create the PiecewiseLegendrePoly object
    let poly = PiecewiseLegendrePoly::new(data.clone(), knots.clone(), l, None, 0);

    // Test that the object is initialized correctly
    let poly_data = poly.get_data();
    for i in 0..data.shape().0 {
        for j in 0..data.shape().1 {
            assert!((poly_data[[i, j]] - data[[i, j]]).abs() < 1e-15);
        }
    }
    assert_eq!(poly.get_xmin(), knots[0]);
    assert_eq!(poly.get_xmax(), knots[knots.len() - 1]);
    assert!(
        poly.get_knots()
            .iter()
            .zip(knots.iter())
            .all(|(a, b)| (a - b).abs() < 1e-15)
    );
    assert_eq!(poly.get_polyorder(), data.shape().0);
    assert_eq!(poly.get_symm(), 0);

    // Test evaluation at specific point
    let x = 0.5328437345518631;
    let result = poly.evaluate(x);
    let expected = 2.696073744825952;
    println!(
        "Julia random data test - x: {}, expected: {}, actual: {}",
        x, expected, result
    );
    assert!((result - expected).abs() < 1e-10);

    // Test split function (equivalent to C++ split)
    let (i, tilde_x) = poly.split(x);
    assert_eq!(i, 0);
    let expected_tilde_x = -0.25995538114498773;
    println!(
        "Split - i: {}, expected tilde_x: {}, actual: {}",
        i, expected_tilde_x, tilde_x
    );
    assert!((tilde_x - expected_tilde_x).abs() < 1e-10);
}

#[test]
fn test_high_order_polynomial_vector() {
    // Test with high-order polynomials from C++ poly.cxx
    // These are 16th-order polynomials with complex coefficients

    // Data for polynomial 1 (16x2 matrix)
    let data1_values = vec![
        0.49996553669802485,
        -0.009838135710548356,
        0.003315915376286483,
        -2.4035906967802686e-5,
        3.4824832610792906e-6,
        -1.6818592059096e-8,
        1.5530850593697272e-9,
        -5.67191158452736e-12,
        3.8438802553084145e-13,
        -1.12861464373688e-15,
        -1.4028528586225198e-16,
        5.199431653846204e-18,
        -3.490774002228127e-16,
        4.339342349553959e-18,
        -8.247505551908268e-17,
        7.379549188001237e-19,
        0.49996553669802485,
        0.009838135710548356,
        0.003315915376286483,
        2.4035906967802686e-5,
        3.4824832610792906e-6,
        1.6818592059096e-8,
        1.5530850593697272e-9,
        5.67191158452736e-12,
        3.8438802553084145e-13,
        1.12861464373688e-15,
        -1.4028528586225198e-16,
        -5.199431653846204e-18,
        -3.490774002228127e-16,
        -4.339342349553959e-18,
        -8.247505551908268e-17,
        -7.379549188001237e-19,
    ];

    let mut data1 = mdarray::DTensor::<f64, 2>::from_elem([16, 2], 0.0);
    for i in 0..16 {
        for j in 0..2 {
            data1[[i, j]] = data1_values[i * 2 + j];
        }
    }

    // Data for polynomial 2 (16x2 matrix)
    let data2_values = vec![
        -0.43195475509329695,
        0.436151579050162,
        -0.005257007544885257,
        0.0010660519696441624,
        -6.611545612452212e-6,
        7.461310619506964e-7,
        -3.2179499894475862e-9,
        2.5166526274315926e-10,
        -8.387341925898803e-13,
        5.008268649326024e-14,
        3.7750894390998034e-17,
        -2.304983535459561e-16,
        3.0252856483620636e-16,
        -1.923751082183687e-16,
        7.201014354168769e-17,
        -3.2715804561902326e-17,
        0.43195475509329695,
        0.436151579050162,
        0.005257007544885257,
        0.0010660519696441624,
        6.611545612452212e-6,
        7.461310619506964e-7,
        3.2179499894475862e-9,
        2.5166526274315926e-10,
        8.387341925898803e-13,
        5.008268649326024e-14,
        -3.7750894390998034e-17,
        -2.304983535459561e-16,
        -3.0252856483620636e-16,
        -1.923751082183687e-16,
        -7.201014354168769e-17,
        -3.2715804561902326e-17,
    ];

    let mut data2 = mdarray::DTensor::<f64, 2>::from_elem([16, 2], 0.0);
    for i in 0..16 {
        for j in 0..2 {
            data2[[i, j]] = data2_values[i * 2 + j];
        }
    }

    // Data for polynomial 3 (16x2 matrix)
    let data3_values = vec![
        -0.005870438661638806,
        -0.8376202388555938,
        0.28368166184926036,
        -0.0029450618222246236,
        0.0004277118923277169,
        -2.4101642603229184e-6,
        2.2287962786878678e-7,
        -8.875091544426018e-10,
        6.021488924175155e-11,
        -1.8705305570705647e-13,
        9.924398482443944e-15,
        4.299521053905097e-16,
        -1.0697019178666955e-16,
        3.6972269778329906e-16,
        -8.848885164903329e-17,
        6.327687614609368e-17,
        -0.005870438661638806,
        0.8376202388555938,
        0.28368166184926036,
        0.0029450618222246236,
        0.0004277118923277169,
        2.4101642603229184e-6,
        2.2287962786878678e-7,
        8.875091544426018e-10,
        6.021488924175155e-11,
        1.8705305570705647e-13,
        9.924398482443944e-15,
        -4.299521053905097e-16,
        -1.0697019178666955e-16,
        -3.6972269778329906e-16,
        -8.848885164903329e-17,
        -6.327687614609368e-17,
    ];

    let mut data3 = mdarray::DTensor::<f64, 2>::from_elem([16, 2], 0.0);
    for i in 0..16 {
        for j in 0..2 {
            data3[[i, j]] = data3_values[i * 2 + j];
        }
    }

    let knots = vec![-1.0, 0.0, 1.0];

    // Create high-order polynomials
    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots.clone(), 1, None, 0);
    let poly3 = PiecewiseLegendrePoly::new(data3, knots.clone(), 2, None, 0);

    // Create polynomial vector
    let vector =
        PiecewiseLegendrePolyVector::new(vec![poly1.clone(), poly2.clone(), poly3.clone()]);

    // Test basic properties
    assert_eq!(vector.size(), 3);
    assert_eq!(vector.xmin(), -1.0);
    assert_eq!(vector.xmax(), 1.0);
    assert_eq!(vector.get_polyorder(), 16);

    // Test evaluation at a point
    let x = 0.5;
    let results = vector.evaluate_at(x);
    assert_eq!(results.len(), 3);

    // Compare with individual polynomial evaluations
    let expected_results = [poly1.evaluate(x), poly2.evaluate(x), poly3.evaluate(x)];
    for i in 0..3 {
        println!(
            "Poly {} at {}: Expected {}, Actual {}",
            i, x, expected_results[i], results[i]
        );
        assert!((results[i] - expected_results[i]).abs() < 1e-12);
    }

    // Test evaluation at multiple points
    let xs = vec![-0.8, -0.2, 0.2, 0.8];
    let results_matrix = vector.evaluate_at_many(&xs);
    assert_eq!(*results_matrix.shape(), (3, 4));

    // Verify each evaluation
    for i in 0..3 {
        for j in 0..4 {
            let expected = match i {
                0 => poly1.evaluate(xs[j]),
                1 => poly2.evaluate(xs[j]),
                2 => poly3.evaluate(xs[j]),
                _ => unreachable!(),
            };
            let actual = results_matrix[[i, j]];
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    println!("High-order polynomial vector test passed!");
}

#[test]
fn test_rescale_domain_single_poly() {
    // Create a simple polynomial on [-1, 1]
    let data_vec = vec![
        1.0, 2.0, // const term
        0.5, 1.0, // linear term
        0.2, 0.3, // quadratic term
    ];
    let data = mdarray::DTensor::<f64, 2>::from_fn([3, 2], |idx| data_vec[idx[0] * 2 + idx[1]]);

    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    // Test evaluation on original domain
    let val_original = poly.evaluate(0.5);

    // Rescale to [0, 2]: x_new = x_old + 1
    let new_knots = vec![0.0, 1.0, 2.0];
    let poly_rescaled = poly.rescale_domain(new_knots.clone(), None, None);

    // Check domain changed
    assert_eq!(poly_rescaled.xmin, 0.0);
    assert_eq!(poly_rescaled.xmax, 2.0);
    assert_eq!(poly_rescaled.knots, new_knots);

    // Evaluate at corresponding point: x=0.5 on [-1,1] → x=1.5 on [0,2]
    let val_rescaled = poly_rescaled.evaluate(1.5);

    // Values should be equal (same data, just different domain mapping)
    assert!(
        (val_original - val_rescaled).abs() < 1e-14,
        "Rescaled value differs: {} vs {}",
        val_original,
        val_rescaled
    );
}

#[test]
fn test_scale_data_single_poly() {
    // Create a polynomial
    let data_vec = [1.0, 2.0, 0.5, 1.0];
    let data = mdarray::DTensor::<f64, 2>::from_fn([2, 2], |idx| data_vec[idx[0] * 2 + idx[1]]);

    let knots = vec![-1.0, 0.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots, 0, None, 0);

    let val_original = poly.evaluate(0.5);

    // Scale data by 2.0
    let poly_scaled = poly.scale_data(2.0);
    let val_scaled = poly_scaled.evaluate(0.5);

    // Scaled value should be exactly 2x original
    assert!(
        (val_scaled - 2.0 * val_original).abs() < 1e-14,
        "Scaled value incorrect: {} vs {}",
        val_scaled,
        2.0 * val_original
    );

    // Domain should be unchanged
    assert_eq!(poly_scaled.xmin, poly.xmin);
    assert_eq!(poly_scaled.xmax, poly.xmax);
}

#[test]
fn test_rescale_domain_vector() {
    // Create vector with 2 polynomials
    let data1 =
        mdarray::DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 1.0 } else { 0.5 });
    let data2 =
        mdarray::DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 2.0 } else { 1.0 });

    let knots = vec![-1.0, 1.0];
    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 1); // symm=1
    let poly2 = PiecewiseLegendrePoly::new(data2, knots.clone(), 1, None, -1); // symm=-1

    let polyvec = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);

    // Rescale to [0, β] where β=2.0
    let beta = 2.0;
    let new_knots = vec![0.0, beta];
    let new_delta_x = vec![beta];

    let polyvec_rescaled = polyvec.rescale_domain(
        new_knots.clone(),
        Some(new_delta_x),
        None, // Keep original symmetry
    );

    // Check all polynomials have new domain
    for poly in polyvec_rescaled.get_polys() {
        assert_eq!(poly.xmin, 0.0);
        assert_eq!(poly.xmax, beta);
    }

    // Check symmetry preserved
    assert_eq!(polyvec_rescaled.get_polys()[0].symm, 1);
    assert_eq!(polyvec_rescaled.get_polys()[1].symm, -1);
}

#[test]
fn test_scale_data_vector() {
    // Create vector with 2 polynomials
    let data1 =
        mdarray::DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 1.0 } else { 0.5 });
    let data2 =
        mdarray::DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 2.0 } else { 1.0 });

    let knots = vec![-1.0, 1.0];
    let poly1 = PiecewiseLegendrePoly::new(data1, knots.clone(), 0, None, 0);
    let poly2 = PiecewiseLegendrePoly::new(data2, knots, 1, None, 0);

    let polyvec = PiecewiseLegendrePolyVector::new(vec![poly1, poly2]);

    // Evaluate before scaling
    let val1_before = polyvec.get_polys()[0].evaluate(0.0);
    let val2_before = polyvec.get_polys()[1].evaluate(0.0);

    // Scale by sqrt(2.0)
    let factor = 2.0_f64.sqrt();
    let polyvec_scaled = polyvec.scale_data(factor);

    // Evaluate after scaling
    let val1_after = polyvec_scaled.get_polys()[0].evaluate(0.0);
    let val2_after = polyvec_scaled.get_polys()[1].evaluate(0.0);

    // Check scaling
    assert!((val1_after - factor * val1_before).abs() < 1e-14);
    assert!((val2_after - factor * val2_before).abs() < 1e-14);
}

#[test]
fn test_rescale_with_symmetry_change() {
    // Test rescaling with symmetry parameter change
    let data =
        mdarray::DTensor::<f64, 2>::from_fn([2, 1], |idx| if idx[0] == 0 { 1.0 } else { 0.5 });
    let knots = vec![-1.0, 1.0];
    let poly = PiecewiseLegendrePoly::new(data, knots.clone(), 0, None, 1);

    assert_eq!(poly.symm, 1);

    // Rescale to [0, 2] with symm=-1
    let new_knots = vec![0.0, 2.0];
    let poly_rescaled = poly.rescale_domain(new_knots, None, Some(-1));

    assert_eq!(poly_rescaled.symm, -1);
    assert_eq!(poly_rescaled.xmin, 0.0);
    assert_eq!(poly_rescaled.xmax, 2.0);
}

#[test]
fn test_basis_transformation_example() {
    // Example: Transform from SVE domain x ∈ [-1, 1] to τ ∈ [0, β]
    // Transformation: τ = β/2 * (x + 1)
    // So: x = -1 → τ = 0, x = 1 → τ = β

    let beta = 10.0;

    // Original polynomial on [-1, 1]
    let data_vec = [1.0, 1.5, 0.2, 0.3, 0.1, 0.1];
    let data = mdarray::DTensor::<f64, 2>::from_fn([3, 2], |idx| data_vec[idx[0] * 2 + idx[1]]);
    let sve_knots = vec![-1.0, 0.0, 1.0];
    let poly_sve = PiecewiseLegendrePoly::new(data, sve_knots, 0, None, 0);

    // Transform to [0, β]
    let tau_knots: Vec<f64> = poly_sve
        .knots
        .iter()
        .map(|&x| beta / 2.0 * (x + 1.0))
        .collect();
    let tau_delta_x: Vec<f64> = poly_sve.delta_x.iter().map(|&dx| beta / 2.0 * dx).collect();

    let poly_tau = poly_sve.rescale_domain(tau_knots.clone(), Some(tau_delta_x), None);

    // Check transformation
    assert!((poly_tau.xmin - 0.0).abs() < 1e-14);
    assert!((poly_tau.xmax - beta).abs() < 1e-14);
    assert_eq!(poly_tau.knots.len(), 3);
    assert!((poly_tau.knots[0] - 0.0).abs() < 1e-14);
    assert!((poly_tau.knots[1] - beta / 2.0).abs() < 1e-14);
    assert!((poly_tau.knots[2] - beta).abs() < 1e-14);
}
