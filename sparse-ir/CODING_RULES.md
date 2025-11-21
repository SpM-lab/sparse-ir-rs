# Coding Rules for sparseir-rust

This document outlines the coding style and special considerations for the sparseir-rust project, particularly regarding high-precision numerical computation and TwoFloat limitations.

## General Coding Style

### 1. Generic Functions and Type Parameters
- **Prefer generic functions** over code duplication
- Use generic type parameters with proper trait bounds
- Example: `fn test_kernel_interpolation_precision_generic<T: CustomNumeric + Clone + 'static>`
- Always include `'static` lifetime bound when using `std::any::TypeId`

### 2. Trait Design
- **Add methods to existing traits** rather than creating workarounds
- When a method is needed across multiple types, implement it in the base trait
- Example: Added `max()` method to `CustomNumeric` trait for both `f64` and `TwoFloat`

### 3. Error Handling
- Use `panic!` for unrecoverable errors (e.g., points outside cell boundaries)
- Prefer explicit error messages with context
- Example: `"x={} is outside cell bounds [{}, {}]", x, self.x_min, self.x_max`

### 4. Code Organization
- **Move related functions to appropriate modules** (e.g., interpolation functions to `interpolation1d.rs`, `interpolation2d.rs`)
- Keep tests in separate `tests/` directory, not in `src/`
- Use descriptive module and function names

### 5. Comments and Documentation
- Add comments explaining **why** something is done, especially for precision-related decisions
- Document precision limitations and bottlenecks
- Example: Comments about TwoFloat's `cos()` function having f64-level precision

## TwoFloat Precision Limitations

### Critical Limitation: Arithmetic Function Precision
- **TwoFloat's arithmetic functions (`sin`, `cos`, `exp`) have only f64-level precision (~15-16 digits)**
- This is NOT the full theoretical 30-digit precision of double-double arithmetic
- This limitation affects the overall precision of interpolation and numerical computations

### Practical Implications
- TwoFloat interpolation accuracy is limited to ~1e-16, not 1e-30
- When implementing high-precision tests, set realistic tolerances based on actual precision
- Example: TwoFloat 2D interpolation achieves ~1e-12 absolute error, not 1e-30

### Code Comments Required
When using TwoFloat, always add comments explaining precision limitations:
```rust
// Note: TwoFloat's cos() has only f64-level precision (~15-16 digits), not the full 
// theoretical 30-digit precision. This limits TwoFloat interpolation accuracy to ~1e-16,
// not the 1e-30 that might be theoretically possible with perfect double-double arithmetic.
```

## Numerical Computation Guidelines

### 1. Tolerance Settings
- **Base tolerances on observed precision**, not theoretical limits
- Allow reasonable margins above observed errors
- Example: If observed error is ~2.86e-13, set tolerance to 1e-12

### 2. Type Conversions
- Use `T::from_f64()` for explicit type conversions
- Avoid unnecessary intermediate conversions
- Example: `let x_norm = T::from_f64(2.0) * (x - gauss_x.a) / (gauss_x.b - gauss_x.a) - T::from_f64(1.0);`

### 3. Matrix Operations
- Use manual matrix multiplication for generic types to avoid `ndarray::dot` recursion limits
- Example: Manual implementation of `coeffs = C_x * values * C_y^T`

### 4. Coordinate Normalization
- Always normalize coordinates to `[-1, 1]` range for Legendre polynomial evaluation
- Apply normalization in both coefficient computation and evaluation phases

## Testing Guidelines

### 1. Generic Test Functions
- Create generic test functions to avoid code duplication
- Pass parameters (lambda, epsilon, tolerances) as arguments
- Example: `test_kernel_interpolation_precision_generic<T>(lambda, epsilon, tolerance_abs, tolerance_rel)`

### 2. Precision Testing Strategy
- Use `DBig` (arbitrary precision) as reference for high-precision comparisons
- Implement `to_dbig()` methods for accurate type conversion
- Test both `f64` and `TwoFloat` implementations

### 3. Tolerance Configuration
- Set different tolerances for different numeric types based on their actual capabilities
- f64: `tolerance_abs = 1e-12`, `tolerance_rel = 1e-10`
- TwoFloat: `tolerance_abs = 1e-11`, `tolerance_rel = 1e-10` (slightly stricter due to extended precision)

### 4. Debug Information
- Include debug prints showing actual error values during development
- Remove debug prints before committing (unless they provide ongoing value)

### 5. Commit Message Guidelines
- **Focus on substantial changes**: what was implemented, removed, or fixed
- Avoid mentioning intermediate development steps, temporary code, or trial-and-error activities
- Write commit messages from the perspective of the final result
- Example: ✅ "Add symmetry type parameter to kernel interpolation tests"
- Example: ✅ "Fix memory leak in matrix operations" (important fixes)
- Example: ✅ "Remove deprecated API functions" (important removals)
- Example: ❌ "Remove temporary debug code added during development" (intermediate steps)
- Example: ❌ "Remove unnecessary Parity enum, use existing SymmetryType instead" (trial-and-error)

## API Design Principles

### 1. Configuration via Hints
- Use `SVEHints` to get configuration parameters (segments, polynomial degrees)
- Example: `let gauss_per_cell = hints.ngauss();` instead of hardcoded values

### 2. Generic Kernel Support
- Design kernels to be generic over numeric types
- Use `CentrosymmKernel` trait with generic methods for `dyn` compatibility

### 3. Struct-based Interfaces
- Prefer struct-based interfaces over function-based ones for complex objects
- Example: `Interpolate1D<T>`, `Interpolate2D<T>`, `InterpolatedKernel<T>`

## Performance Considerations

### 1. Pre-computation
- Store pre-computed polynomial coefficients to avoid repeated computation
- Example: `Interpolate2D` stores coefficients computed once during construction

### 2. Efficient Cell Finding
- Use binary search for cell lookup in interpolation grids
- Implement `binary_search_segments()` for O(log n) cell finding

### 3. Memory Management
- Use `Array2` for 2D data structures
- Clone data structures when necessary to avoid move errors

## File Organization

### 1. Module Structure
```
src/
├── kernel.rs           # Kernel definitions and SVE hints
├── kernelmatrix.rs     # DiscretizedKernel and InterpolatedKernel
├── interpolation1d.rs  # 1D interpolation functions and structs
├── interpolation2d.rs  # 2D interpolation functions and structs
├── gauss.rs           # Gauss quadrature rules and utilities
└── numeric.rs         # CustomNumeric trait and implementations

tests/
├── kernel_tests.rs                    # Kernel precision tests
├── kernel_interpolation_tests.rs      # Interpolation precision tests
├── interpolation1d_tests.rs           # 1D interpolation tests
├── interpolation2d_tests.rs           # 2D interpolation tests
└── kernelmatrix_tests.rs              # Matrix generation tests
```

### 2. Import Organization
- Group imports by module (sparseir-rust, external crates, std)
- Use specific imports when possible
- Remove unused imports

## Common Patterns

### 1. Generic Function Design
- Use generic type parameters with appropriate trait bounds
- Include lifetime bounds (`'static`) when using `std::any::TypeId`
- Convert between types using trait methods (`T::from_f64()`)

### 2. Precision Testing Strategy
- Use high-precision reference implementations for validation
- Implement conversion methods to arbitrary precision types
- Base tolerances on observed rather than theoretical precision

### 3. Trait Method Implementation
- Use built-in methods when available for standard types
- Implement manual logic for custom types that lack built-in methods
- Ensure consistent behavior across all implementations

## Anti-patterns to Avoid

### 1. Hardcoded Configuration Values
- Avoid magic numbers and hardcoded parameters
- Use configuration objects or hints to provide parameters
- Make functions configurable through parameters

### 2. Unnecessary Type Conversions
- Avoid redundant conversions between types
- Use types directly when they're already in the correct format
- Minimize intermediate conversion steps

### 3. Missing Documentation
- Always document precision limitations and bottlenecks
- Explain why certain approaches are taken
- Provide context for numerical decisions

### 4. Unrealistic Expectations
- Set tolerances based on actual capabilities, not theoretical limits
- Account for implementation-specific limitations
- Allow reasonable margins for numerical stability

---

This document should be updated as new patterns emerge and precision limitations are discovered.
