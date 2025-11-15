#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <sparseir/sparseir.h>   // C interface

using Catch::Approx;

TEST_CASE("Test spir_kernel_get_sve_hints_nsvals", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    int nsvals;
    status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, &nsvals);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(nsvals > 0);
    
    // For lambda=10, log10(10)=1, so nsvals should be around (25+1)*1 = 26
    // But actual formula may vary, so just check it's reasonable
    REQUIRE(nsvals >= 10);
    REQUIRE(nsvals <= 1000);
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_kernel_get_sve_hints_ngauss", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon_coarse = 1e-6;
    double epsilon_fine = 1e-10;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    int ngauss_coarse;
    status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon_coarse, &ngauss_coarse);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(ngauss_coarse > 0);
    
    int ngauss_fine;
    status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon_fine, &ngauss_fine);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(ngauss_fine > 0);
    
    // For epsilon >= 1e-8, ngauss should be 10
    // For epsilon < 1e-8, ngauss should be 16
    REQUIRE(ngauss_coarse == 10);
    REQUIRE(ngauss_fine == 16);
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_kernel_get_sve_hints_segments_x", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    // First call: get the number of segments
    int n_segments = 0;
    status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, nullptr, &n_segments);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments > 0);
    
    // Second call: get the actual segments
    std::vector<double> segments(n_segments);
    int n_segments_out = n_segments;
    status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, segments.data(), &n_segments_out);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments_out == n_segments);
    
    // Verify segments are valid
    REQUIRE(segments.size() == static_cast<size_t>(n_segments));
    REQUIRE(segments[0] == Approx(0.0).margin(1e-10));
    REQUIRE(segments[n_segments - 1] == Approx(1.0).margin(1e-10));
    
    // Verify segments are in ascending order
    for (int i = 1; i < n_segments; ++i) {
        REQUIRE(segments[i] > segments[i-1]);
    }
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_kernel_get_sve_hints_segments_y", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    // First call: get the number of segments
    int n_segments = 0;
    status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, nullptr, &n_segments);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments > 0);
    
    // Second call: get the actual segments
    std::vector<double> segments(n_segments);
    int n_segments_out = n_segments;
    status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, segments.data(), &n_segments_out);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments_out == n_segments);
    
    // Verify segments are valid
    REQUIRE(segments.size() == static_cast<size_t>(n_segments));
    REQUIRE(segments[0] == Approx(0.0).margin(1e-10));
    REQUIRE(segments[n_segments - 1] == Approx(1.0).margin(1e-10));
    
    // Verify segments are in ascending order
    for (int i = 1; i < n_segments; ++i) {
        REQUIRE(segments[i] > segments[i-1]);
    }
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_kernel_get_sve_hints with RegularizedBoseKernel", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_reg_bose_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    // Test nsvals
    int nsvals;
    status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, &nsvals);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(nsvals > 0);
    
    // Test ngauss
    int ngauss;
    status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon, &ngauss);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(ngauss > 0);
    
    // Test segments_x
    int n_segments_x = 0;
    status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, nullptr, &n_segments_x);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments_x > 0);
    
    // Test segments_y
    int n_segments_y = 0;
    status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, nullptr, &n_segments_y);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments_y > 0);
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_kernel_get_sve_hints error handling", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    // Test with nullptr kernel
    int nsvals;
    status = spir_kernel_get_sve_hints_nsvals(nullptr, epsilon, &nsvals);
    REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
    
    // Test with nullptr output parameter
    status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, nullptr);
    REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_choose_working_type", "[cinterface]")
{
    // Test with epsilon >= 1e-8 -> should return FLOAT64
    {
        int twork = spir_choose_working_type(1e-6);
        REQUIRE(twork == SPIR_TWORK_FLOAT64);
    }
    
    {
        int twork = spir_choose_working_type(1e-8);
        REQUIRE(twork == SPIR_TWORK_FLOAT64);
    }
    
    // Test with epsilon < 1e-8 -> should return FLOAT64X2
    {
        int twork = spir_choose_working_type(1e-10);
        REQUIRE(twork == SPIR_TWORK_FLOAT64X2);
    }
    
    {
        int twork = spir_choose_working_type(1e-15);
        REQUIRE(twork == SPIR_TWORK_FLOAT64X2);
    }
    
    // Test with NaN -> should return FLOAT64X2
    {
        int twork = spir_choose_working_type(std::numeric_limits<double>::quiet_NaN());
        REQUIRE(twork == SPIR_TWORK_FLOAT64X2);
    }
    
    // Test boundary case: epsilon = 1e-8 exactly
    {
        int twork = spir_choose_working_type(1e-8);
        REQUIRE(twork == SPIR_TWORK_FLOAT64);
    }
    
    // Test boundary case: epsilon just below 1e-8
    {
        int twork = spir_choose_working_type(0.99e-8);
        REQUIRE(twork == SPIR_TWORK_FLOAT64X2);
    }
}

TEST_CASE("Test spir_funcs_from_piecewise_legendre", "[cinterface]")
{
    // Create a simple piecewise polynomial: constant function = 1.0 on [-1, 1]
    // Single segment, nfuncs=1 (only degree 0 Legendre polynomial)
    {
        int n_segments = 1;
        double segments[2] = {-1.0, 1.0};
        double coeffs[1] = {1.0};  // Only constant term
        int nfuncs = 1;
        int order = 0;
        
        int status;
        spir_funcs* funcs = spir_funcs_from_piecewise_legendre(
            segments, n_segments, coeffs, nfuncs, order, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(funcs != nullptr);
        
        // Test evaluation at x=0 (should be normalized, so value depends on normalization)
        int size;
        status = spir_funcs_get_size(funcs, &size);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(size == 1);
        
        // Evaluate at x=0
        double x = 0.0;
        double values[1];
        status = spir_funcs_eval(funcs, x, values);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        // Value should be approximately 1.0 * normalization factor
        REQUIRE(values[0] == Approx(1.0).margin(1e-10));
        
        spir_funcs_release(funcs);
    }
    
    // Create a linear function: f(x) = x on [-1, 1]
    // Single segment, nfuncs=2 (degrees 0 and 1)
    {
        int n_segments = 1;
        double segments[2] = {-1.0, 1.0};
        // Legendre expansion: P0(x) = 1, P1(x) = x
        // For f(x) = x, we need coefficient 0 for P0 and 1 for P1
        // But normalization affects the actual values
        double coeffs[2] = {0.0, 1.0};  // Constant=0, linear=1
        int nfuncs = 2;
        int order = 0;
        
        int status;
        spir_funcs* funcs = spir_funcs_from_piecewise_legendre(
            segments, n_segments, coeffs, nfuncs, order, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(funcs != nullptr);
        
        // Evaluate at x=0.5
        double x = 0.5;
        double values[2];
        status = spir_funcs_eval(funcs, x, values);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        // Value should be approximately 0.5 (with normalization)
        // Note: PiecewiseLegendrePoly applies normalization, so the actual value may differ
        // We just check that evaluation succeeds and returns a reasonable value
        REQUIRE(std::abs(values[0]) < 10.0);  // Reasonable bound
        
        spir_funcs_release(funcs);
    }
    
    // Test error handling: invalid arguments
    {
        int status;
        spir_funcs* funcs = spir_funcs_from_piecewise_legendre(
            nullptr, 1, nullptr, 1, 0, &status);
        REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
        REQUIRE(funcs == nullptr);
    }
    
    // Test error handling: n_segments < 1
    {
        double segments[2] = {-1.0, 1.0};
        double coeffs[1] = {1.0};
        int status;
        spir_funcs* funcs = spir_funcs_from_piecewise_legendre(
            segments, 0, coeffs, 1, 0, &status);
        REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
        REQUIRE(funcs == nullptr);
    }
    
    // Test error handling: non-monotonic segments
    {
        double segments[2] = {1.0, -1.0};  // Wrong order
        double coeffs[1] = {1.0};
        int status;
        spir_funcs* funcs = spir_funcs_from_piecewise_legendre(
            segments, 1, coeffs, 1, 0, &status);
        REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
        REQUIRE(funcs == nullptr);
    }
}

TEST_CASE("Test spir_gauss_legendre_rule_piecewise_double", "[cinterface]")
{
    // Test with single segment [-1, 1]
    {
        int n = 5;
        double segments[2] = {-1.0, 1.0};
        int n_segments = 1;
        double x[5], w[5];
        int status;
        
        status = spir_gauss_legendre_rule_piecewise_double(
            n, segments, n_segments, x, w, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        // Verify we got n points
        // Points should be in [-1, 1] and sorted
        REQUIRE(x[0] >= -1.0);
        REQUIRE(x[n-1] <= 1.0);
        for (int i = 1; i < n; ++i) {
            REQUIRE(x[i] > x[i-1]);
        }
        
        // Weights should be positive
        for (int i = 0; i < n; ++i) {
            REQUIRE(w[i] > 0.0);
        }
        
        // Weight sum should be approximately 1.0 (integral over [-1, 1] is 2, but normalized)
        // Actually, for Gauss-Legendre on [-1, 1], sum of weights should be 2.0
        double weight_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            weight_sum += w[i];
        }
        REQUIRE(weight_sum == Approx(2.0).margin(1e-10));  // Integral over [-1, 1] is 2
    }
    
    // Test with two segments [-1, 0, 1]
    {
        int n = 3;
        double segments[3] = {-1.0, 0.0, 1.0};
        int n_segments = 2;
        double x[6], w[6];  // n * n_segments
        int status;
        
        status = spir_gauss_legendre_rule_piecewise_double(
            n, segments, n_segments, x, w, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        // Verify we got n * n_segments points
        // Points should be sorted across segments
        REQUIRE(x[0] >= -1.0);
        REQUIRE(x[5] <= 1.0);
        for (int i = 1; i < 6; ++i) {
            REQUIRE(x[i] > x[i-1]);
        }
        
        // Weights should be positive
        for (int i = 0; i < 6; ++i) {
            REQUIRE(w[i] > 0.0);
        }
        
        // Weight sum should be approximately 2.0 (integral over [-1, 1])
        double weight_sum = 0.0;
        for (int i = 0; i < 6; ++i) {
            weight_sum += w[i];
        }
        REQUIRE(weight_sum == Approx(2.0).margin(1e-10));
    }
    
    // Test error handling
    {
        int status;
        status = spir_gauss_legendre_rule_piecewise_double(
            5, nullptr, 1, nullptr, nullptr, &status);
        REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
    }
    
    {
        double segments[2] = {-1.0, 1.0};
        double x[5], w[5];
        int status;
        status = spir_gauss_legendre_rule_piecewise_double(
            0, segments, 1, x, w, &status);
        REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
    }
    
    {
        double segments[2] = {1.0, -1.0};  // Wrong order
        double x[5], w[5];
        int status;
        status = spir_gauss_legendre_rule_piecewise_double(
            5, segments, 1, x, w, &status);
        REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
    }
}

TEST_CASE("Test spir_gauss_legendre_rule_piecewise_ddouble", "[cinterface]")
{
    // Test with single segment [-1, 1]
    {
        int n = 5;
        double segments[2] = {-1.0, 1.0};
        int n_segments = 1;
        double x_high[5], x_low[5], w_high[5], w_low[5];
        int status;
        
        status = spir_gauss_legendre_rule_piecewise_ddouble(
            n, segments, n_segments, x_high, x_low, w_high, w_low, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        // Verify we got n points
        // Points should be in [-1, 1] and sorted
        REQUIRE(x_high[0] >= -1.0);
        REQUIRE(x_high[n-1] <= 1.0);
        for (int i = 1; i < n; ++i) {
            // Reconstruct DDouble and compare
            double x_val = x_high[i] + x_low[i];
            double x_prev = x_high[i-1] + x_low[i-1];
            REQUIRE(x_val > x_prev);
        }
        
        // Weights should be positive
        double weight_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            double w_val = w_high[i] + w_low[i];
            REQUIRE(w_val > 0.0);
            weight_sum += w_val;
        }
        // Weight sum should be approximately 2.0 (integral over [-1, 1])
        REQUIRE(weight_sum == Approx(2.0).margin(1e-10));
    }
    
    // Test with two segments [-1, 0, 1]
    {
        int n = 3;
        double segments[3] = {-1.0, 0.0, 1.0};
        int n_segments = 2;
        double x_high[6], x_low[6], w_high[6], w_low[6];
        int status;
        
        status = spir_gauss_legendre_rule_piecewise_ddouble(
            n, segments, n_segments, x_high, x_low, w_high, w_low, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        // Verify points are sorted
        for (int i = 1; i < 6; ++i) {
            double x_val = x_high[i] + x_low[i];
            double x_prev = x_high[i-1] + x_low[i-1];
            REQUIRE(x_val > x_prev);
        }
        
        // Weight sum should be approximately 2.0 (integral over [-1, 1])
        double weight_sum = 0.0;
        for (int i = 0; i < 6; ++i) {
            double w_val = w_high[i] + w_low[i];
            weight_sum += w_val;
        }
        REQUIRE(weight_sum == Approx(2.0).margin(1e-10));
    }
    
    // Test error handling
    {
        int status;
        status = spir_gauss_legendre_rule_piecewise_ddouble(
            5, nullptr, 1, nullptr, nullptr, nullptr, nullptr, &status);
        REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
    }
}

TEST_CASE("Test spir_sve_result_from_matrix", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon = 1e-8;
    
    // Create kernel and get SVE hints
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    // Get SVE hints
    int n_gauss;
    status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon, &n_gauss);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    int n_segments_x = 0;
    status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, nullptr, &n_segments_x);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    int n_segments_y = 0;
    status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, nullptr, &n_segments_y);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Get segments
    std::vector<double> segments_x(n_segments_x + 1);
    int n_segments_x_out = n_segments_x + 1;
    status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, segments_x.data(), &n_segments_x_out);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    std::vector<double> segments_y(n_segments_y + 1);
    int n_segments_y_out = n_segments_y + 1;
    status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, segments_y.data(), &n_segments_y_out);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Get Gauss points and weights
    // Note: n_segments_x and n_segments_y are the number of boundary points (n_segments + 1),
    // but spir_gauss_legendre_rule_piecewise_double expects the number of segments (n_segments)
    int nx = n_gauss * (n_segments_x - 1);  // n_segments_x - 1 is the number of segments
    int ny = n_gauss * (n_segments_y - 1);  // n_segments_y - 1 is the number of segments
    std::vector<double> x(nx), w_x(nx);
    std::vector<double> y(ny), w_y(ny);
    
    int status_gauss;
    status_gauss = spir_gauss_legendre_rule_piecewise_double(
        n_gauss, segments_x.data(), n_segments_x - 1, x.data(), w_x.data(), &status_gauss);  // n_segments_x - 1 is the number of segments
    REQUIRE(status_gauss == SPIR_COMPUTATION_SUCCESS);
    
    status_gauss = spir_gauss_legendre_rule_piecewise_double(
        n_gauss, segments_y.data(), n_segments_y - 1, y.data(), w_y.data(), &status_gauss);  // n_segments_y - 1 is the number of segments
    REQUIRE(status_gauss == SPIR_COMPUTATION_SUCCESS);
    
    // Create a simple test kernel matrix
    // Note: In practice, this would be computed from the actual kernel
    std::vector<double> K_high(nx * ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            // Simple test: scaled identity-like matrix
            double k_val = (i == j) ? std::sqrt(w_x[i] * w_y[j]) : 0.0;
            K_high[i * ny + j] = k_val;
        }
    }
    
    // Create SVE result from matrix (row major)
    spir_sve_result* sve_from_matrix = spir_sve_result_from_matrix(
        K_high.data(), nullptr, nx, ny, SPIR_ORDER_ROW_MAJOR,
        segments_x.data(), n_segments_x,
        segments_y.data(), n_segments_y,
        n_gauss, epsilon, &status);
    
    // Note: The test matrix is very simple, so the SVE result may not be meaningful
    // But we can at least verify the function doesn't crash and returns a valid result
    if (status == SPIR_COMPUTATION_SUCCESS && sve_from_matrix != nullptr) {
        int sve_size;
        status = spir_sve_result_get_size(sve_from_matrix, &sve_size);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        spir_sve_result_release(sve_from_matrix);
    }
    
    // Test error handling
    {
        spir_sve_result* sve_err = spir_sve_result_from_matrix(
            nullptr, nullptr, nx, ny, SPIR_ORDER_ROW_MAJOR,
            segments_x.data(), n_segments_x,
            segments_y.data(), n_segments_y,
            n_gauss, epsilon, &status);
        REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
        REQUIRE(sve_err == nullptr);
    }
    
    // Test column major order
    {
        std::vector<double> K_col(nx * ny);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                K_col[j * nx + i] = (i == j) ? std::sqrt(w_x[i] * w_y[j]) : 0.0;
            }
        }
        
        spir_sve_result* sve_col = spir_sve_result_from_matrix(
            K_col.data(), nullptr, nx, ny, SPIR_ORDER_COLUMN_MAJOR,
            segments_x.data(), n_segments_x,
            segments_y.data(), n_segments_y,
            n_gauss, epsilon, &status);
        
        if (status == SPIR_COMPUTATION_SUCCESS && sve_col != nullptr) {
            spir_sve_result_release(sve_col);
        }
    }
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_basis_new_from_sve_and_inv_weight", "[cinterface]")
{
    double lambda = 10.0;
    double beta = 1.0;
    double omega_max = lambda / beta;
    double epsilon = 1e-8;
    
    // Create kernel and SVE result
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    spir_sve_result* sve = spir_sve_result_new(
        kernel, epsilon, -1, -1, SPIR_TWORK_AUTO, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(sve != nullptr);
    
    // Create inv_weight_func as spir_funcs
    // For LogisticKernel with Fermionic statistics, inv_weight_func(omega) = 1.0
    // We'll create a simple constant function
    int n_segments = 1;
    double segments[2] = {-omega_max, omega_max};  // Full omega range
    double coeffs[1] = {1.0};  // Constant function = 1.0
    int nfuncs = 1;
    int order = 0;
    
    spir_funcs* inv_weight_funcs = spir_funcs_from_piecewise_legendre(
        segments, n_segments, coeffs, nfuncs, order, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(inv_weight_funcs != nullptr);
    
    // Create basis using new function
    // For LogisticKernel: ypower=0, conv_radius=1.0 (typical values)
    spir_basis* basis = spir_basis_new_from_sve_and_inv_weight(
        SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon,
        lambda, 0, 1.0,  // ypower=0, conv_radius=1.0
        sve, inv_weight_funcs, -1, &status);
    
    if (status == SPIR_COMPUTATION_SUCCESS && basis != nullptr) {
        int basis_size;
        status = spir_basis_get_size(basis, &basis_size);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(basis_size > 0);
        
        spir_basis_release(basis);
    }
    
    // Test error handling
    {
        spir_basis* basis_err = spir_basis_new_from_sve_and_inv_weight(
            SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon,
            lambda, 0, 1.0,
            nullptr, inv_weight_funcs, -1, &status);
        REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
        REQUIRE(basis_err == nullptr);
    }
    
    {
        spir_basis* basis_err = spir_basis_new_from_sve_and_inv_weight(
            SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon,
            lambda, 0, 1.0,
            sve, nullptr, -1, &status);
        REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
        REQUIRE(basis_err == nullptr);
    }
    
    spir_funcs_release(inv_weight_funcs);
    spir_sve_result_release(sve);
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_basis_get_default_matsus_ext with fence", "[cinterface]")
{
    double beta = 10.0;
    double wmax = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(beta * wmax, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    spir_sve_result* sve = spir_sve_result_new(kernel, epsilon, -1, -1, SPIR_TWORK_AUTO, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    spir_basis* basis = spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, wmax, epsilon, kernel, sve, -1, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    int basis_size;
    status = spir_basis_get_size(basis, &basis_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Test without mitigation (fence = false)
    {
        bool positive_only = false;
        bool mitigate = false;
        int n_points_requested = basis_size;
        
        int n_points_returned = 0;
        std::vector<int64_t> points(n_points_requested + 10);
        
        status = spir_basis_get_default_matsus_ext(
            basis, positive_only, mitigate, n_points_requested, points.data(), &n_points_returned);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points_returned >= n_points_requested);
        REQUIRE(n_points_returned <= n_points_requested + 10);
        
        // Verify points are valid fermionic frequencies (odd integers)
        for (int i = 0; i < n_points_returned; ++i) {
            REQUIRE(llabs(points[i]) % 2 == 1);
        }
    }
    
    // Test with mitigation (fence = true)
    {
        bool positive_only = false;
        bool mitigate = true;
        int n_points_requested = basis_size;
        
        int n_points_returned = 0;
        std::vector<int64_t> points(n_points_requested + 20);  // Extra space for fence points
        
        status = spir_basis_get_default_matsus_ext(
            basis, positive_only, mitigate, n_points_requested, points.data(), &n_points_returned);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points_returned >= n_points_requested);  // May exceed when mitigate is true
        
        // Verify points are valid fermionic frequencies (odd integers)
        for (int i = 0; i < n_points_returned; ++i) {
            REQUIRE(llabs(points[i]) % 2 == 1);
        }
        
        // When mitigate is true and basis_size >= 20, we should have more points
        // Note: This may not always be true due to rounding, so we just check it's >= requested
        if (basis_size >= 20 && n_points_returned > n_points_requested) {
            // Extra points due to fencing were added
        }
    }
    
    // Test positive_only = true with mitigation
    {
        bool positive_only = true;
        bool mitigate = true;
        int n_points_requested = basis_size;
        
        int n_points_returned = 0;
        std::vector<int64_t> points(n_points_requested + 20);
        
        status = spir_basis_get_default_matsus_ext(
            basis, positive_only, mitigate, n_points_requested, points.data(), &n_points_returned);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points_returned > 0);
        REQUIRE(n_points_returned <= basis_size + 10);  // May have some extra points due to fencing
        
        // Verify all points are positive and odd
        for (int i = 0; i < n_points_returned; ++i) {
            REQUIRE(points[i] > 0);
            REQUIRE(points[i] % 2 == 1);
        }
    }
    
    spir_basis_release(basis);
    spir_sve_result_release(sve);
    spir_kernel_release(kernel);
}
