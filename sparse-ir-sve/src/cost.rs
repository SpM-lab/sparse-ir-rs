//! Work estimates for the threading decisions, in `f64` operations
//!
//! Every parallel region of this crate passes its estimated work to
//! [`crate::parallel::for_each_chunk`] or [`crate::parallel::map_on_pool`],
//! which run it on the calling thread below
//! [`crate::parallel::PARALLEL_MIN_WORK`]. The estimates count `f64`
//! operations and are rough: they need only order the regions correctly
//! against that one threshold. They were calibrated on an M1 Pro, where the
//! threshold is about 130 µs of serial work.

/// A double-double operation
pub(crate) const DF64_OP: usize = 16;

/// A term of a double-double matrix product ([`crate::ddgemm`]), with its
/// error-free transformations
pub(crate) const DF64_PRODUCT_TERM: usize = 12;

/// An SVE kernel entry (a few exponentials), counted in operations of its
/// element type
pub(crate) const KERNEL_ENTRY: usize = 64;
