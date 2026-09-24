//! Crate-level error type for fallible tensor operations.

use crate::gemm::GemmError;

/// Errors returned by sampling, fitting, and tensor-valued APIs.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The target axis does not exist in the input tensor.
    #[error("axis {dim} out of range for a rank-{rank} tensor")]
    AxisOutOfRange { dim: usize, rank: usize },
    /// Input or output extents do not match the operator.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    /// The requested scalar-type combination is not provided.
    #[error("unsupported operation: {0}")]
    Unsupported(&'static str),
    /// A buffer is not host-resident compact column-major storage where one
    /// is required (for example a strided output view).
    #[error("layout error: {0}")]
    Layout(String),
    /// A caller-supplied argument is invalid.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// A numerical kernel (for example an SVD) failed.
    #[error("numerical failure: {0}")]
    Numerical(String),
    /// The GEMM backend rejected the call.
    #[error(transparent)]
    Gemm(#[from] GemmError),
    /// A tenferro tensor operation failed.
    #[error(transparent)]
    Tensor(#[from] tenferro_tensor::Error),
}

/// Crate result alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;
