//! Matrix operations for equilibrium solvers.
//!
//! Provides matrix types optimized for equilibrium solving:
//! - [`DenseMatrix`] - Full dense matrix (row-major)
//! - [`SparseMatrix`] - CSR sparse matrix for large systems
//!
//! These are designed for equilibrium models where the operator
//! is a learned weight matrix (not just diagonal).

/// Dense matrix in row-major order.
///
/// Optimized for small-to-medium matrices (up to ~1024x1024).
/// For larger matrices, consider using a BLAS backend.
#[derive(Debug, Clone)]
pub struct DenseMatrix {
    /// Matrix data in row-major order
    pub data: Vec<f32>,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
}

impl DenseMatrix {
    /// Create a new dense matrix from data.
    ///
    /// # Arguments
    /// * `data` - Row-major matrix data
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Panics
    /// Panics if data.len() != rows * cols
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length {} doesn't match {}x{} = {}",
            data.len(),
            rows,
            cols,
            rows * cols
        );
        Self { data, rows, cols }
    }

    /// Create a zeros matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create an identity matrix.
    pub fn identity(dim: usize) -> Self {
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Self {
            data,
            rows: dim,
            cols: dim,
        }
    }

    /// Create a diagonal matrix from diagonal values.
    pub fn from_diagonal(diag: &[f32]) -> Self {
        let dim = diag.len();
        let mut data = vec![0.0; dim * dim];
        for (i, &d) in diag.iter().enumerate() {
            data[i * dim + i] = d;
        }
        Self {
            data,
            rows: dim,
            cols: dim,
        }
    }

    /// Get element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        debug_assert!(row < self.rows && col < self.cols);
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        debug_assert!(row < self.rows && col < self.cols);
        self.data[row * self.cols + col] = value;
    }

    /// Matrix-vector multiply: y = A @ x
    ///
    /// # Arguments
    /// * `x` - Input vector of length cols
    ///
    /// # Returns
    /// Output vector of length rows
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);

        let mut y = vec![0.0; self.rows];
        self.matvec_into(x, &mut y);
        y
    }

    /// Matrix-vector multiply into existing buffer.
    pub fn matvec_into(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);

        for i in 0..self.rows {
            let row_start = i * self.cols;
            let mut sum = 0.0;

            // Process 4 elements at a time for better cache utilization
            let chunks = self.cols / 4;
            let remainder = self.cols % 4;

            for j in 0..chunks {
                let idx = row_start + j * 4;
                let x_idx = j * 4;
                sum += self.data[idx] * x[x_idx];
                sum += self.data[idx + 1] * x[x_idx + 1];
                sum += self.data[idx + 2] * x[x_idx + 2];
                sum += self.data[idx + 3] * x[x_idx + 3];
            }

            // Handle remainder
            for j in (self.cols - remainder)..self.cols {
                sum += self.data[row_start + j] * x[j];
            }

            y[i] = sum;
        }
    }

    /// Matrix-vector multiply and add: y = A @ x + b
    pub fn matvec_add(&self, x: &[f32], b: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(b.len(), self.rows);

        let mut y = self.matvec(x);
        for (yi, &bi) in y.iter_mut().zip(b.iter()) {
            *yi += bi;
        }
        y
    }

    /// Matrix-vector multiply add into buffer: y = A @ x + b
    pub fn matvec_add_into(&self, x: &[f32], b: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(b.len(), self.rows);
        debug_assert_eq!(y.len(), self.rows);

        self.matvec_into(x, y);
        for (yi, &bi) in y.iter_mut().zip(b.iter()) {
            *yi += bi;
        }
    }

    /// Compute Frobenius norm.
    pub fn frobenius_norm(&self) -> f32 {
        self.data.iter().map(|&v| v * v).sum::<f32>().sqrt()
    }

    /// Estimate spectral norm (largest singular value) via power iteration.
    ///
    /// # Arguments
    /// * `max_iters` - Maximum iterations for power method
    /// * `tol` - Convergence tolerance
    pub fn spectral_norm_estimate(&self, max_iters: usize, tol: f32) -> f32 {
        if self.rows != self.cols {
            // For non-square, use A^T @ A
            return self.ata_spectral_norm(max_iters, tol).sqrt();
        }

        let n = self.rows;
        let mut x = vec![1.0 / (n as f32).sqrt(); n];
        let mut y = vec![0.0; n];
        let mut sigma = 0.0;

        for _ in 0..max_iters {
            // y = A @ x
            self.matvec_into(&x, &mut y);

            // Normalize
            let norm: f32 = y.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm < 1e-10 {
                return 0.0;
            }

            let sigma_new = norm;
            for v in y.iter_mut() {
                *v /= norm;
            }

            // Check convergence
            if (sigma_new - sigma).abs() < tol * sigma_new {
                return sigma_new;
            }

            sigma = sigma_new;
            std::mem::swap(&mut x, &mut y);
        }

        sigma
    }

    /// Estimate spectral norm of A^T @ A (for non-square matrices).
    fn ata_spectral_norm(&self, max_iters: usize, tol: f32) -> f32 {
        let n = self.cols;
        let mut x = vec![1.0 / (n as f32).sqrt(); n];
        let mut ax = vec![0.0; self.rows];
        let mut atax = vec![0.0; n];
        let mut sigma = 0.0;

        for _ in 0..max_iters {
            // ax = A @ x
            self.matvec_into(&x, &mut ax);

            // atax = A^T @ ax
            for (i, v) in atax.iter_mut().enumerate() {
                *v = 0.0;
                for j in 0..self.rows {
                    *v += self.data[j * self.cols + i] * ax[j];
                }
            }

            // Normalize
            let norm: f32 = atax.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm < 1e-10 {
                return 0.0;
            }

            let sigma_new = norm;
            for v in atax.iter_mut() {
                *v /= norm;
            }

            if (sigma_new - sigma).abs() < tol * sigma_new {
                return sigma_new;
            }

            sigma = sigma_new;
            std::mem::swap(&mut x, &mut atax);
        }

        sigma
    }
}

/// CSR Sparse matrix for large systems.
///
/// Compressed Sparse Row format is efficient for matrix-vector products.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Non-zero values
    pub data: Vec<f32>,
    /// Column indices for each non-zero
    pub indices: Vec<usize>,
    /// Row pointers (indices[indptr[i]..indptr[i+1]] are in row i)
    pub indptr: Vec<usize>,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
}

impl SparseMatrix {
    /// Create a sparse matrix from COO format.
    ///
    /// # Arguments
    /// * `row_indices` - Row index for each non-zero
    /// * `col_indices` - Column index for each non-zero
    /// * `values` - Non-zero values
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    pub fn from_coo(
        row_indices: &[usize],
        col_indices: &[usize],
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Self {
        assert_eq!(row_indices.len(), col_indices.len());
        assert_eq!(row_indices.len(), values.len());

        let nnz = values.len();

        // Sort by row then column
        let mut entries: Vec<(usize, usize, f32)> = row_indices
            .iter()
            .zip(col_indices.iter())
            .zip(values.iter())
            .map(|((&r, &c), &v)| (r, c, v))
            .collect();
        entries.sort_by_key(|&(r, c, _)| (r, c));

        // Convert to CSR
        let mut data = Vec::with_capacity(nnz);
        let mut indices = Vec::with_capacity(nnz);
        let mut indptr = vec![0usize; rows + 1];

        for (row, col, val) in entries {
            data.push(val);
            indices.push(col);
            indptr[row + 1] += 1;
        }

        // Cumulative sum for indptr
        for i in 1..=rows {
            indptr[i] += indptr[i - 1];
        }

        Self {
            data,
            indices,
            indptr,
            rows,
            cols,
        }
    }

    /// Create sparse identity matrix.
    pub fn identity(dim: usize) -> Self {
        Self {
            data: vec![1.0; dim],
            indices: (0..dim).collect(),
            indptr: (0..=dim).collect(),
            rows: dim,
            cols: dim,
        }
    }

    /// Create sparse diagonal matrix.
    pub fn from_diagonal(diag: &[f32]) -> Self {
        let dim = diag.len();
        Self {
            data: diag.to_vec(),
            indices: (0..dim).collect(),
            indptr: (0..=dim).collect(),
            rows: dim,
            cols: dim,
        }
    }

    /// Matrix-vector multiply: y = A @ x
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);

        let mut y = vec![0.0; self.rows];
        self.matvec_into(x, &mut y);
        y
    }

    /// Matrix-vector multiply into buffer.
    pub fn matvec_into(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);

        for i in 0..self.rows {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];

            let mut sum = 0.0;
            for j in start..end {
                sum += self.data[j] * x[self.indices[j]];
            }
            y[i] = sum;
        }
    }

    /// Matrix-vector multiply and add: y = A @ x + b
    pub fn matvec_add(&self, x: &[f32], b: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(b.len(), self.rows);

        let mut y = self.matvec(x);
        for (yi, &bi) in y.iter_mut().zip(b.iter()) {
            *yi += bi;
        }
        y
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Sparsity ratio (nnz / total elements).
    pub fn sparsity(&self) -> f32 {
        self.nnz() as f32 / (self.rows * self.cols) as f32
    }
}

/// Generic matrix operator trait.
pub trait MatrixOperator {
    /// Apply operator: y = A @ x + b
    fn apply(&self, x: &[f32], bias: &[f32]) -> Vec<f32>;

    /// Apply operator into buffer
    fn apply_into(&self, x: &[f32], bias: &[f32], y: &mut [f32]);

    /// Get output dimension
    fn output_dim(&self) -> usize;

    /// Get input dimension
    fn input_dim(&self) -> usize;
}

impl MatrixOperator for DenseMatrix {
    fn apply(&self, x: &[f32], bias: &[f32]) -> Vec<f32> {
        self.matvec_add(x, bias)
    }

    fn apply_into(&self, x: &[f32], bias: &[f32], y: &mut [f32]) {
        self.matvec_add_into(x, bias, y);
    }

    fn output_dim(&self) -> usize {
        self.rows
    }

    fn input_dim(&self) -> usize {
        self.cols
    }
}

impl MatrixOperator for SparseMatrix {
    fn apply(&self, x: &[f32], bias: &[f32]) -> Vec<f32> {
        self.matvec_add(x, bias)
    }

    fn apply_into(&self, x: &[f32], bias: &[f32], y: &mut [f32]) {
        self.matvec_into(x, y);
        for (yi, &bi) in y.iter_mut().zip(bias.iter()) {
            *yi += bi;
        }
    }

    fn output_dim(&self) -> usize {
        self.rows
    }

    fn input_dim(&self) -> usize {
        self.cols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_matvec() {
        // 2x3 matrix
        let m = DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let x = vec![1.0, 2.0, 3.0];
        let y = m.matvec(&x);

        // [1,2,3] @ [1,2,3]^T = 1+4+9 = 14
        // [4,5,6] @ [1,2,3]^T = 4+10+18 = 32
        assert!((y[0] - 14.0).abs() < 1e-5);
        assert!((y[1] - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_dense_identity() {
        let m = DenseMatrix::identity(3);
        let x = vec![1.0, 2.0, 3.0];
        let y = m.matvec(&x);
        assert_eq!(y, x);
    }

    #[test]
    fn test_dense_diagonal() {
        let m = DenseMatrix::from_diagonal(&[2.0, 3.0, 4.0]);
        let x = vec![1.0, 2.0, 3.0];
        let y = m.matvec(&x);
        assert!((y[0] - 2.0).abs() < 1e-5);
        assert!((y[1] - 6.0).abs() < 1e-5);
        assert!((y[2] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_matvec() {
        // Same 2x3 matrix as sparse
        let row = vec![0, 0, 0, 1, 1, 1];
        let col = vec![0, 1, 2, 0, 1, 2];
        let val = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = SparseMatrix::from_coo(&row, &col, &val, 2, 3);

        let x = vec![1.0, 2.0, 3.0];
        let y = m.matvec(&x);

        assert!((y[0] - 14.0).abs() < 1e-5);
        assert!((y[1] - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_identity() {
        let m = SparseMatrix::identity(3);
        let x = vec![1.0, 2.0, 3.0];
        let y = m.matvec(&x);
        assert_eq!(y, x);
    }

    #[test]
    fn test_spectral_norm() {
        // Identity matrix has spectral norm 1
        let m = DenseMatrix::identity(10);
        let sigma = m.spectral_norm_estimate(100, 1e-6);
        assert!((sigma - 1.0).abs() < 1e-4);

        // Diagonal matrix with entries [0.5, 0.5, ...] has spectral norm 0.5
        let m2 = DenseMatrix::from_diagonal(&[0.5; 10]);
        let sigma2 = m2.spectral_norm_estimate(100, 1e-6);
        assert!((sigma2 - 0.5).abs() < 1e-4);
    }
}
