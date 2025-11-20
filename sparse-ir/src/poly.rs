//! Piecewise Legendre polynomial implementations for SparseIR
//!
//! This module provides high-performance piecewise Legendre polynomial
//! functionality compatible with the C++ implementation.

/// A single piecewise Legendre polynomial
#[derive(Debug, Clone)]
pub struct PiecewiseLegendrePoly {
    /// Polynomial order (degree of Legendre polynomials in each segment)
    pub polyorder: usize,
    /// Minimum x value of the domain
    pub xmin: f64,
    /// Maximum x value of the domain
    pub xmax: f64,
    /// Knot points defining the segments
    pub knots: Vec<f64>,
    /// Segment widths (for numerical stability)
    pub delta_x: Vec<f64>,
    /// Coefficient matrix: [degree][segment_index]
    pub data: mdarray::DTensor<f64, 2>,
    /// Symmetry parameter
    pub symm: i32,
    /// Polynomial parameter (used in power moments calculation)
    pub l: i32,
    /// Segment midpoints
    pub xm: Vec<f64>,
    /// Inverse segment widths
    pub inv_xs: Vec<f64>,
    /// Normalization factors
    pub norms: Vec<f64>,
}

impl PiecewiseLegendrePoly {
    /// Create a new PiecewiseLegendrePoly from data and knots
    pub fn new(
        data: mdarray::DTensor<f64, 2>,
        knots: Vec<f64>,
        l: i32,
        delta_x: Option<Vec<f64>>,
        symm: i32,
    ) -> Self {
        let polyorder = data.shape().0;
        let nsegments = data.shape().1;

        if knots.len() != nsegments + 1 {
            panic!(
                "Invalid knots array: expected {} knots, got {}",
                nsegments + 1,
                knots.len()
            );
        }

        // Validate knots are sorted
        for i in 1..knots.len() {
            if knots[i] <= knots[i - 1] {
                panic!("Knots must be monotonically increasing");
            }
        }

        // Compute delta_x if not provided
        let delta_x =
            delta_x.unwrap_or_else(|| (1..knots.len()).map(|i| knots[i] - knots[i - 1]).collect());

        // Validate delta_x matches knots
        for i in 0..delta_x.len() {
            let expected = knots[i + 1] - knots[i];
            if (delta_x[i] - expected).abs() > 1e-10 {
                panic!("delta_x must match knots");
            }
        }

        // Compute segment midpoints
        let xm: Vec<f64> = (0..nsegments)
            .map(|i| 0.5 * (knots[i] + knots[i + 1]))
            .collect();

        // Compute inverse segment widths
        let inv_xs: Vec<f64> = delta_x.iter().map(|&dx| 2.0 / dx).collect();

        // Compute normalization factors
        let norms: Vec<f64> = inv_xs.iter().map(|&inv_x| inv_x.sqrt()).collect();

        Self {
            polyorder,
            xmin: knots[0],
            xmax: knots[knots.len() - 1],
            knots,
            delta_x,
            data,
            symm,
            l,
            xm,
            inv_xs,
            norms,
        }
    }

    /// Create a new PiecewiseLegendrePoly with new data but same structure
    pub fn with_data(&self, new_data: mdarray::DTensor<f64, 2>) -> Self {
        Self {
            data: new_data,
            ..self.clone()
        }
    }

    /// Get the symmetry parameter
    pub fn symm(&self) -> i32 {
        self.symm
    }

    /// Create a new PiecewiseLegendrePoly with new data and symmetry
    pub fn with_data_and_symmetry(
        &self,
        new_data: mdarray::DTensor<f64, 2>,
        new_symm: i32,
    ) -> Self {
        Self {
            data: new_data,
            symm: new_symm,
            ..self.clone()
        }
    }

    /// Rescale domain: create a new polynomial with the same data but different knots
    ///
    /// This is useful for transforming from one domain to another, e.g.,
    /// from x ∈ [-1, 1] to τ ∈ [0, β].
    ///
    /// # Arguments
    ///
    /// * `new_knots` - New knot points
    /// * `new_delta_x` - Optional new segment widths (computed from knots if None)
    /// * `new_symm` - Optional new symmetry parameter (keeps old if None)
    ///
    /// # Returns
    ///
    /// New polynomial with rescaled domain
    pub fn rescale_domain(
        &self,
        new_knots: Vec<f64>,
        new_delta_x: Option<Vec<f64>>,
        new_symm: Option<i32>,
    ) -> Self {
        Self::new(
            self.data.clone(),
            new_knots,
            self.l,
            new_delta_x,
            new_symm.unwrap_or(self.symm),
        )
    }

    /// Scale all data values by a constant factor
    ///
    /// This is useful for normalizations, e.g., multiplying by √β for
    /// Fourier transform preparations.
    ///
    /// # Arguments
    ///
    /// * `factor` - Scaling factor to multiply all data by
    ///
    /// # Returns
    ///
    /// New polynomial with scaled data
    pub fn scale_data(&self, factor: f64) -> Self {
        Self::with_data(
            self,
            mdarray::DTensor::<f64, 2>::from_fn(*self.data.shape(), |idx| self.data[idx] * factor),
        )
    }

    /// Evaluate the polynomial at a given point
    pub fn evaluate(&self, x: f64) -> f64 {
        let (i, x_tilde) = self.split(x);
        // Extract column i into a Vec
        let coeffs: Vec<f64> = (0..self.data.shape().0)
            .map(|row| self.data[[row, i]])
            .collect();
        let value = self.evaluate_legendre_polynomial(x_tilde, &coeffs);
        value * self.norms[i]
    }

    /// Evaluate the polynomial at multiple points
    pub fn evaluate_many(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }

    /// Split x into segment index and normalized x
    pub fn split(&self, x: f64) -> (usize, f64) {
        if x < self.xmin || x > self.xmax {
            panic!("x = {} is outside domain [{}, {}]", x, self.xmin, self.xmax);
        }

        // Find the segment containing x
        for i in 0..self.knots.len() - 1 {
            if x >= self.knots[i] && x <= self.knots[i + 1] {
                // Transform x to [-1, 1] for Legendre polynomials
                let x_tilde = 2.0 * (x - self.xm[i]) / self.delta_x[i];
                return (i, x_tilde);
            }
        }

        // Handle edge case: x exactly at the last knot
        let last_idx = self.knots.len() - 2;
        let x_tilde = 2.0 * (x - self.xm[last_idx]) / self.delta_x[last_idx];
        (last_idx, x_tilde)
    }

    /// Evaluate Legendre polynomial using recurrence relation
    pub fn evaluate_legendre_polynomial(&self, x: f64, coeffs: &[f64]) -> f64 {
        if coeffs.is_empty() {
            return 0.0;
        }

        let mut result = 0.0;
        let mut p_prev = 1.0; // P_0(x) = 1
        let mut p_curr = x; // P_1(x) = x

        // Add first two terms
        if !coeffs.is_empty() {
            result += coeffs[0] * p_prev;
        }
        if coeffs.len() > 1 {
            result += coeffs[1] * p_curr;
        }

        // Use recurrence relation: P_{n+1}(x) = ((2n+1)x*P_n(x) - n*P_{n-1}(x))/(n+1)
        for n in 1..coeffs.len() - 1 {
            let p_next =
                ((2.0 * (n as f64) + 1.0) * x * p_curr - (n as f64) * p_prev) / ((n + 1) as f64);
            result += coeffs[n + 1] * p_next;
            p_prev = p_curr;
            p_curr = p_next;
        }

        result
    }

    /// Compute derivative of the polynomial
    pub fn deriv(&self, n: usize) -> Self {
        if n == 0 {
            return self.clone();
        }

        // Compute derivative coefficients
        let mut ddata = self.data.clone();
        for _ in 0..n {
            ddata = self.compute_derivative_coefficients(&ddata);
        }

        // Apply scaling factors (C++: ddata.col(i) *= std::pow(inv_xs[i], n))
        let ddata_shape = *ddata.shape();
        for i in 0..ddata_shape.1 {
            let inv_x_power = self.inv_xs[i].powi(n as i32);
            for j in 0..ddata_shape.0 {
                ddata[[j, i]] *= inv_x_power;
            }
        }

        // Update symmetry: C++: int new_symm = std::pow(-1, n) * symm;
        let new_symm = if n % 2 == 0 { self.symm } else { -self.symm };

        Self {
            data: ddata,
            symm: new_symm,
            ..self.clone()
        }
    }

    /// Compute derivative coefficients using the same algorithm as C++ legder function
    fn compute_derivative_coefficients(
        &self,
        coeffs: &mdarray::DTensor<f64, 2>,
    ) -> mdarray::DTensor<f64, 2> {
        let mut c = coeffs.clone();
        let c_shape = *c.shape();
        let mut n = c_shape.0;

        // Single derivative step (equivalent to C++ legder with cnt=1)
        if n <= 1 {
            return mdarray::DTensor::<f64, 2>::from_elem([1, c.shape().1], 0.0);
        }

        n -= 1;
        let mut der = mdarray::DTensor::<f64, 2>::from_elem([n, c.shape().1], 0.0);

        // C++ implementation: for (int j = n; j >= 2; --j)
        for j in (2..=n).rev() {
            // C++: der.row(j - 1) = (2 * j - 1) * c.row(j);
            for col in 0..c_shape.1 {
                der[[j - 1, col]] = (2.0 * (j as f64) - 1.0) * c[[j, col]];
            }
            // C++: c.row(j - 2) += c.row(j);
            for col in 0..c_shape.1 {
                c[[j - 2, col]] += c[[j, col]];
            }
        }

        // C++: if (n > 1) der.row(1) = 3 * c.row(2);
        if n > 1 {
            for col in 0..c_shape.1 {
                der[[1, col]] = 3.0 * c[[2, col]];
            }
        }

        // C++: der.row(0) = c.row(1);
        for col in 0..c_shape.1 {
            der[[0, col]] = c[[1, col]];
        }

        der
    }

    /// Compute derivatives at a point x
    pub fn derivs(&self, x: f64) -> Vec<f64> {
        let mut results = Vec::new();

        // Compute up to polyorder derivatives
        for n in 0..self.polyorder {
            let deriv_poly = self.deriv(n);
            results.push(deriv_poly.evaluate(x));
        }

        results
    }

    /// Compute overlap integral with a function
    pub fn overlap<F>(&self, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let mut integral = 0.0;

        for i in 0..self.knots.len() - 1 {
            let segment_integral =
                self.gauss_legendre_quadrature(self.knots[i], self.knots[i + 1], |x| {
                    self.evaluate(x) * f(x)
                });
            integral += segment_integral;
        }

        integral
    }

    /// Gauss-Legendre quadrature over [a, b]
    fn gauss_legendre_quadrature<F>(&self, a: f64, b: f64, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        // 5-point Gauss-Legendre quadrature
        const XG: [f64; 5] = [
            -0.906179845938664,
            -0.538469310105683,
            0.0,
            0.538469310105683,
            0.906179845938664,
        ];
        const WG: [f64; 5] = [
            0.236926885056189,
            0.478628670499366,
            0.568888888888889,
            0.478628670499366,
            0.236926885056189,
        ];

        let c1 = (b - a) / 2.0;
        let c2 = (b + a) / 2.0;

        let mut integral = 0.0;
        for j in 0..5 {
            let x = c1 * XG[j] + c2;
            integral += WG[j] * f(x);
        }

        integral * c1
    }

    /// Find roots of the polynomial using C++ compatible algorithm
    pub fn roots(&self) -> Vec<f64> {
        // Refine the grid by factor of 4 for better root finding
        // (C++ uses 2, but RegularizedBoseKernel needs finer resolution)
        let refined_grid = self.refine_grid(&self.knots, 4);

        // Find all roots using the refined grid
        self.find_all_roots(&refined_grid)
    }

    /// Refine grid by factor alpha (C++ compatible)
    fn refine_grid(&self, grid: &[f64], alpha: usize) -> Vec<f64> {
        let mut refined = Vec::new();

        for i in 0..grid.len() - 1 {
            let start = grid[i];
            let step = (grid[i + 1] - grid[i]) / (alpha as f64);
            for j in 0..alpha {
                refined.push(start + (j as f64) * step);
            }
        }
        refined.push(grid[grid.len() - 1]);
        refined
    }

    /// Find all roots using refined grid (C++ compatible)
    fn find_all_roots(&self, xgrid: &[f64]) -> Vec<f64> {
        if xgrid.is_empty() {
            return Vec::new();
        }

        // Evaluate function at all grid points
        let fx: Vec<f64> = xgrid.iter().map(|&x| self.evaluate(x)).collect();

        // Find exact zeros (direct hits)
        let mut x_hit = Vec::new();
        for i in 0..fx.len() {
            if fx[i] == 0.0 {
                x_hit.push(xgrid[i]);
            }
        }

        // Find sign changes
        let mut sign_change = Vec::new();
        for i in 0..fx.len() - 1 {
            let has_sign_change = fx[i].signum() != fx[i + 1].signum();
            let not_hit = fx[i] != 0.0 && fx[i + 1] != 0.0;
            sign_change.push(has_sign_change && not_hit);
        }

        // If no sign changes, return only direct hits
        if sign_change.iter().all(|&sc| !sc) {
            x_hit.sort_by(|a, b| a.partial_cmp(b).unwrap());
            return x_hit;
        }

        // Find intervals with sign changes
        let mut a_intervals = Vec::new();
        let mut b_intervals = Vec::new();
        let mut fa_values = Vec::new();

        for i in 0..sign_change.len() {
            if sign_change[i] {
                a_intervals.push(xgrid[i]);
                b_intervals.push(xgrid[i + 1]);
                fa_values.push(fx[i]);
            }
        }

        // Calculate epsilon for convergence
        let max_elm = xgrid.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let epsilon_x = f64::EPSILON * max_elm;

        // Use bisection for each interval with sign change
        for i in 0..a_intervals.len() {
            let root = self.bisect(a_intervals[i], b_intervals[i], fa_values[i], epsilon_x);
            x_hit.push(root);
        }

        // Sort and return
        x_hit.sort_by(|a, b| a.partial_cmp(b).unwrap());
        x_hit
    }

    /// Bisection method to find root (C++ compatible)
    fn bisect(&self, a: f64, b: f64, fa: f64, eps: f64) -> f64 {
        let mut a = a;
        let mut b = b;
        let mut fa = fa;

        loop {
            let mid = (a + b) / 2.0;
            if self.close_enough(a, mid, eps) {
                return mid;
            }

            let fmid = self.evaluate(mid);
            if fa.signum() != fmid.signum() {
                b = mid;
            } else {
                a = mid;
                fa = fmid;
            }
        }
    }

    /// Check if two values are close enough (C++ compatible)
    fn close_enough(&self, a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    // Accessor methods to match C++ interface
    pub fn get_xmin(&self) -> f64 {
        self.xmin
    }
    pub fn get_xmax(&self) -> f64 {
        self.xmax
    }
    pub fn get_l(&self) -> i32 {
        self.l
    }
    pub fn get_domain(&self) -> (f64, f64) {
        (self.xmin, self.xmax)
    }
    pub fn get_knots(&self) -> &[f64] {
        &self.knots
    }
    pub fn get_delta_x(&self) -> &[f64] {
        &self.delta_x
    }
    pub fn get_symm(&self) -> i32 {
        self.symm
    }
    pub fn get_data(&self) -> &mdarray::DTensor<f64, 2> {
        &self.data
    }
    pub fn get_norms(&self) -> &[f64] {
        &self.norms
    }
    pub fn get_polyorder(&self) -> usize {
        self.polyorder
    }
}

/// Vector of piecewise Legendre polynomials
#[derive(Debug, Clone)]
pub struct PiecewiseLegendrePolyVector {
    /// Individual polynomials
    pub polyvec: Vec<PiecewiseLegendrePoly>,
}

impl PiecewiseLegendrePolyVector {
    /// Constructor with a vector of PiecewiseLegendrePoly
    ///
    /// # Panics
    /// Panics if the input vector is empty, as empty PiecewiseLegendrePolyVector is not meaningful
    pub fn new(polyvec: Vec<PiecewiseLegendrePoly>) -> Self {
        if polyvec.is_empty() {
            panic!("Cannot create empty PiecewiseLegendrePolyVector");
        }
        Self { polyvec }
    }

    /// Get the polynomials
    pub fn get_polys(&self) -> &[PiecewiseLegendrePoly] {
        &self.polyvec
    }

    /// Constructor with a 3D array, knots, and symmetry vector
    pub fn from_3d_data(
        data3d: mdarray::DTensor<f64, 3>,
        knots: Vec<f64>,
        symm: Option<Vec<i32>>,
    ) -> Self {
        let npolys = data3d.shape().2;
        let mut polyvec = Vec::with_capacity(npolys);

        if let Some(ref symm_vec) = symm {
            if symm_vec.len() != npolys {
                panic!("Sizes of data and symm don't match");
            }
        }

        // Compute delta_x from knots
        let delta_x: Vec<f64> = (1..knots.len()).map(|i| knots[i] - knots[i - 1]).collect();

        for i in 0..npolys {
            // Extract 2D data for this polynomial
            let data3d_shape = data3d.shape();
            let mut data =
                mdarray::DTensor::<f64, 2>::from_elem([data3d_shape.0, data3d_shape.1], 0.0);
            for j in 0..data3d_shape.0 {
                for k in 0..data3d_shape.1 {
                    data[[j, k]] = data3d[[j, k, i]];
                }
            }

            let poly = PiecewiseLegendrePoly::new(
                data,
                knots.clone(),
                i as i32,
                Some(delta_x.clone()),
                symm.as_ref().map_or(0, |s| s[i]),
            );

            polyvec.push(poly);
        }

        Self { polyvec }
    }

    /// Get the size of the vector
    pub fn size(&self) -> usize {
        self.polyvec.len()
    }

    /// Rescale domain for all polynomials in the vector
    ///
    /// Creates a new PiecewiseLegendrePolyVector where each polynomial has
    /// the same data but new knots and delta_x.
    ///
    /// # Arguments
    ///
    /// * `new_knots` - New knot points (same for all polynomials)
    /// * `new_delta_x` - Optional new segment widths
    /// * `new_symm` - Optional vector of new symmetry parameters (one per polynomial)
    ///
    /// # Returns
    ///
    /// New vector with rescaled domains
    pub fn rescale_domain(
        &self,
        new_knots: Vec<f64>,
        new_delta_x: Option<Vec<f64>>,
        new_symm: Option<Vec<i32>>,
    ) -> Self {
        let polyvec = self
            .polyvec
            .iter()
            .enumerate()
            .map(|(i, poly)| {
                let symm = new_symm.as_ref().map(|s| s[i]);
                poly.rescale_domain(new_knots.clone(), new_delta_x.clone(), symm)
            })
            .collect();

        Self { polyvec }
    }

    /// Scale all data values by a constant factor
    ///
    /// Multiplies the data of all polynomials by the same factor.
    ///
    /// # Arguments
    ///
    /// * `factor` - Scaling factor to multiply all data by
    ///
    /// # Returns
    ///
    /// New vector with scaled data
    pub fn scale_data(&self, factor: f64) -> Self {
        let polyvec = self
            .polyvec
            .iter()
            .map(|poly| poly.scale_data(factor))
            .collect();

        Self { polyvec }
    }

    /// Get polynomial by index (immutable)
    pub fn get(&self, index: usize) -> Option<&PiecewiseLegendrePoly> {
        self.polyvec.get(index)
    }

    /// Get polynomial by index (mutable) - deprecated, use immutable design instead
    #[deprecated(
        note = "PiecewiseLegendrePolyVector is designed to be immutable. Use get() and create new instances for modifications."
    )]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut PiecewiseLegendrePoly> {
        self.polyvec.get_mut(index)
    }

    /// Extract a single polynomial as a vector
    pub fn slice_single(&self, index: usize) -> Option<Self> {
        self.polyvec.get(index).map(|poly| Self {
            polyvec: vec![poly.clone()],
        })
    }

    /// Extract multiple polynomials by indices
    pub fn slice_multi(&self, indices: &[usize]) -> Self {
        // Validate indices
        for &idx in indices {
            if idx >= self.polyvec.len() {
                panic!("Index {} out of range", idx);
            }
        }

        // Check for duplicates
        {
            let mut unique_indices = indices.to_vec();
            unique_indices.sort();
            unique_indices.dedup();
            if unique_indices.len() != indices.len() {
                panic!("Duplicate indices not allowed");
            }
        }

        let new_polyvec: Vec<_> = indices
            .iter()
            .map(|&idx| self.polyvec[idx].clone())
            .collect();

        Self {
            polyvec: new_polyvec,
        }
    }

    /// Evaluate all polynomials at a single point
    pub fn evaluate_at(&self, x: f64) -> Vec<f64> {
        self.polyvec.iter().map(|poly| poly.evaluate(x)).collect()
    }

    /// Evaluate all polynomials at multiple points
    pub fn evaluate_at_many(&self, xs: &[f64]) -> mdarray::DTensor<f64, 2> {
        let n_funcs = self.polyvec.len();
        let n_points = xs.len();
        let mut results = mdarray::DTensor::<f64, 2>::from_elem([n_funcs, n_points], 0.0);

        for (i, poly) in self.polyvec.iter().enumerate() {
            for (j, &x) in xs.iter().enumerate() {
                results[[i, j]] = poly.evaluate(x);
            }
        }

        results
    }

    // Accessor methods to match C++ interface
    pub fn xmin(&self) -> f64 {
        if self.polyvec.is_empty() {
            panic!("Cannot get xmin from empty PiecewiseLegendrePolyVector");
        }
        self.polyvec[0].xmin
    }

    pub fn xmax(&self) -> f64 {
        if self.polyvec.is_empty() {
            panic!("Cannot get xmax from empty PiecewiseLegendrePolyVector");
        }
        self.polyvec[0].xmax
    }

    pub fn get_knots(&self, tolerance: Option<f64>) -> Vec<f64> {
        if self.polyvec.is_empty() {
            panic!("Cannot get knots from empty PiecewiseLegendrePolyVector");
        }
        const DEFAULT_TOLERANCE: f64 = 1e-10;
        let tolerance = tolerance.unwrap_or(DEFAULT_TOLERANCE);

        // Collect all knots from all polynomials
        let mut all_knots = Vec::new();
        for poly in &self.polyvec {
            for &knot in &poly.knots {
                all_knots.push(knot);
            }
        }

        // Sort and remove duplicates
        {
            all_knots.sort_by(|a, b| a.partial_cmp(b).unwrap());
            all_knots.dedup_by(|a, b| (*a - *b).abs() < tolerance);
        }
        all_knots
    }

    pub fn get_delta_x(&self) -> Vec<f64> {
        if self.polyvec.is_empty() {
            panic!("Cannot get delta_x from empty PiecewiseLegendrePolyVector");
        }
        self.polyvec[0].delta_x.clone()
    }

    pub fn get_polyorder(&self) -> usize {
        if self.polyvec.is_empty() {
            panic!("Cannot get polyorder from empty PiecewiseLegendrePolyVector");
        }
        self.polyvec[0].polyorder
    }

    pub fn get_norms(&self) -> &[f64] {
        if self.polyvec.is_empty() {
            panic!("Cannot get norms from empty PiecewiseLegendrePolyVector");
        }
        &self.polyvec[0].norms
    }

    pub fn get_symm(&self) -> Vec<i32> {
        if self.polyvec.is_empty() {
            panic!("Cannot get symm from empty PiecewiseLegendrePolyVector");
        }
        self.polyvec.iter().map(|poly| poly.symm).collect()
    }

    /// Get data as 3D tensor: [segment][degree][polynomial]
    pub fn get_data(&self) -> mdarray::DTensor<f64, 3> {
        if self.polyvec.is_empty() {
            panic!("Cannot get data from empty PiecewiseLegendrePolyVector");
        }

        let nsegments = self.polyvec[0].data.shape().1;
        let polyorder = self.polyvec[0].polyorder;
        let npolys = self.polyvec.len();

        let mut data = mdarray::DTensor::<f64, 3>::from_elem([nsegments, polyorder, npolys], 0.0);

        for (poly_idx, poly) in self.polyvec.iter().enumerate() {
            for segment in 0..nsegments {
                for degree in 0..polyorder {
                    data[[segment, degree, poly_idx]] = poly.data[[degree, segment]];
                }
            }
        }

        data
    }

    /// Find roots of all polynomials
    pub fn roots(&self, tolerance: Option<f64>) -> Vec<f64> {
        if self.polyvec.is_empty() {
            panic!("Cannot get roots from empty PiecewiseLegendrePolyVector");
        }
        const DEFAULT_TOLERANCE: f64 = 1e-10;
        let tolerance = tolerance.unwrap_or(DEFAULT_TOLERANCE);
        let mut all_roots = Vec::new();

        for poly in &self.polyvec {
            let poly_roots = poly.roots();
            for root in poly_roots {
                all_roots.push(root);
            }
        }

        // Sort in descending order and remove duplicates (like C++ implementation)
        {
            all_roots.sort_by(|a, b| b.partial_cmp(a).unwrap());
            all_roots.dedup_by(|a, b| (*a - *b).abs() < tolerance);
        }
        all_roots
    }

    /// Get reference to last polynomial
    ///
    /// C++ equivalent: u.polyvec.back()
    pub fn last(&self) -> &PiecewiseLegendrePoly {
        self.polyvec
            .last()
            .expect("Cannot get last from empty PiecewiseLegendrePolyVector")
    }

    /// Get the number of roots
    pub fn nroots(&self, tolerance: Option<f64>) -> usize {
        if self.polyvec.is_empty() {
            panic!("Cannot get nroots from empty PiecewiseLegendrePolyVector");
        }
        self.roots(tolerance).len()
    }
}

impl std::ops::Index<usize> for PiecewiseLegendrePolyVector {
    type Output = PiecewiseLegendrePoly;

    fn index(&self, index: usize) -> &Self::Output {
        &self.polyvec[index]
    }
}

/// Get default sampling points in [-1, 1]
///
/// C++ implementation: libsparseir/include/sparseir/basis.hpp:287-310
///
/// For orthogonal polynomials (the high-T limit of IR), we know that the
/// ideal sampling points for a basis of size L are the roots of the L-th
/// polynomial. We empirically find that these stay good sampling points
/// for our kernels (probably because the kernels are totally positive).
///
/// If we do not have enough polynomials in the basis, we approximate the
/// roots of the L'th polynomial by the extrema of the last basis function,
/// which is sensible due to the strong interleaving property of these
/// functions' roots.
pub fn default_sampling_points(u: &PiecewiseLegendrePolyVector, l: usize) -> Vec<f64> {
    // C++: if (u.xmin() != -1.0 || u.xmax() != 1.0)
    //          throw std::runtime_error("Expecting unscaled functions here.");
    if (u.xmin() - (-1.0)).abs() > 1e-10 || (u.xmax() - 1.0).abs() > 1e-10 {
        panic!("Expecting unscaled functions here.");
    }

    let x0 = if l < u.polyvec.len() {
        // C++: return u.polyvec[L].roots();
        u[l].roots()
    } else {
        // C++: PiecewiseLegendrePoly poly = u.polyvec.back();
        //      Eigen::VectorXd maxima = poly.deriv().roots();
        let poly = u.last();
        let poly_deriv = poly.deriv(1);
        let maxima = poly_deriv.roots();

        // C++: double left = (maxima[0] + poly.xmin) / 2.0;
        let left = (maxima[0] + poly.xmin) / 2.0;

        // C++: double right = (maxima[maxima.size() - 1] + poly.xmax) / 2.0;
        let right = (maxima[maxima.len() - 1] + poly.xmax) / 2.0;

        // C++: Eigen::VectorXd x0(maxima.size() + 2);
        //      x0[0] = left;
        //      x0.segment(1, maxima.size()) = maxima;
        //      x0[x0.size() - 1] = right;
        let mut x0_vec = Vec::with_capacity(maxima.len() + 2);
        x0_vec.push(left);
        x0_vec.extend_from_slice(&maxima);
        x0_vec.push(right);

        x0_vec
    };

    // C++: if (x0.size() != L) { warning }
    if x0.len() != l {
        eprintln!(
            "Warning: Expecting to get {} sampling points for corresponding basis function, \
             instead got {}. This may happen if not enough precision is left in the polynomial.",
            l,
            x0.len()
        );
    }

    x0
}

// IndexMut implementation removed - PiecewiseLegendrePolyVector is designed to be immutable
// If modification is needed, create a new instance instead

// Note: FnOnce implementation removed due to experimental nature
// Use evaluate_at() and evaluate_at_many() methods directly

#[cfg(test)]
#[path = "poly_tests.rs"]
mod poly_tests;
