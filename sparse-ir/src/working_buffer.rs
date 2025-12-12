//! Working buffer for in-place operations
//!
//! Provides a reusable buffer that can be used for temporary storage during
//! evaluate/fit operations, avoiding repeated allocations.

use std::alloc::{self, Layout};
use std::ptr::NonNull;

/// A reusable working buffer for temporary storage
///
/// This buffer manages raw memory that can be interpreted as different types
/// (f64 or Complex<f64>) depending on the operation. It automatically grows
/// when more space is needed.
///
/// # Safety
/// This struct manages raw memory and must be used carefully:
/// - The buffer is aligned for Complex<f64> (16 bytes)
/// - When casting to different types, ensure alignment requirements are met
pub struct WorkingBuffer {
    /// Raw pointer to the buffer
    ptr: NonNull<u8>,
    /// Capacity in bytes
    capacity_bytes: usize,
    /// Current layout (for deallocation)
    layout: Option<Layout>,
}

impl WorkingBuffer {
    /// Create a new empty working buffer
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            capacity_bytes: 0,
            layout: None,
        }
    }

    /// Create a new working buffer with initial capacity (in bytes)
    pub fn with_capacity_bytes(capacity_bytes: usize) -> Self {
        if capacity_bytes == 0 {
            return Self::new();
        }

        // Align to 16 bytes for Complex<f64> compatibility
        let layout = Layout::from_size_align(capacity_bytes, 16)
            .expect("Invalid layout");

        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).expect("Allocation failed");

        Self {
            ptr,
            capacity_bytes,
            layout: Some(layout),
        }
    }

    /// Ensure the buffer has at least the specified capacity in bytes
    ///
    /// If the current capacity is insufficient, the buffer is reallocated.
    /// Existing data is NOT preserved.
    pub fn ensure_capacity_bytes(&mut self, required_bytes: usize) {
        if required_bytes <= self.capacity_bytes {
            return;
        }

        // Deallocate old buffer if any
        self.deallocate();

        // Allocate new buffer with some extra room to avoid frequent reallocations
        let new_capacity = required_bytes.max(required_bytes * 3 / 2);
        let layout = Layout::from_size_align(new_capacity, 16)
            .expect("Invalid layout");

        let ptr = unsafe { alloc::alloc(layout) };
        self.ptr = NonNull::new(ptr).expect("Allocation failed");
        self.capacity_bytes = new_capacity;
        self.layout = Some(layout);
    }

    /// Ensure the buffer can hold at least `count` elements of type T
    pub fn ensure_capacity<T>(&mut self, count: usize) {
        let required_bytes = count * std::mem::size_of::<T>();
        self.ensure_capacity_bytes(required_bytes);
    }

    /// Get the buffer as a mutable slice of f64
    ///
    /// # Safety
    /// Caller must ensure:
    /// - The buffer has enough capacity for `count` f64 elements
    /// - No other references to this buffer exist
    pub unsafe fn as_f64_slice_mut(&mut self, count: usize) -> &mut [f64] {
        debug_assert!(count * std::mem::size_of::<f64>() <= self.capacity_bytes);
        std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut f64, count)
    }

    /// Get the buffer as a mutable slice of Complex<f64>
    ///
    /// # Safety
    /// Caller must ensure:
    /// - The buffer has enough capacity for `count` Complex<f64> elements
    /// - No other references to this buffer exist
    pub unsafe fn as_complex_slice_mut(
        &mut self,
        count: usize,
    ) -> &mut [num_complex::Complex<f64>] {
        debug_assert!(count * std::mem::size_of::<num_complex::Complex<f64>>() <= self.capacity_bytes);
        std::slice::from_raw_parts_mut(
            self.ptr.as_ptr() as *mut num_complex::Complex<f64>,
            count,
        )
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get current capacity in bytes
    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Deallocate the buffer
    fn deallocate(&mut self) {
        if let Some(layout) = self.layout.take() {
            unsafe {
                alloc::dealloc(self.ptr.as_ptr(), layout);
            }
        }
        self.ptr = NonNull::dangling();
        self.capacity_bytes = 0;
    }
}

impl Default for WorkingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for WorkingBuffer {
    fn drop(&mut self) {
        self.deallocate();
    }
}

// WorkingBuffer is Send + Sync because it owns its memory
unsafe impl Send for WorkingBuffer {}
unsafe impl Sync for WorkingBuffer {}

/// Copy data from a strided view to a contiguous slice
///
/// This is useful for copying permuted views to a contiguous buffer
/// before performing GEMM operations.
///
/// # Arguments
/// * `src` - Source slice (may be strided)
/// * `dst` - Destination slice (must be contiguous, same total elements)
pub fn copy_to_contiguous<T: Copy>(
    src: &mdarray::Slice<T, mdarray::DynRank, mdarray::Strided>,
    dst: &mut [T],
) {
    assert_eq!(dst.len(), src.len(), "Destination size mismatch");

    // mdarray's iter() returns elements in row-major order
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s;
    }
}

/// Copy data from a contiguous slice to a strided view
///
/// This is useful for copying GEMM results back to a permuted output view.
///
/// # Arguments
/// * `src` - Source slice (contiguous)
/// * `dst` - Destination slice (may be strided)
pub fn copy_from_contiguous<T: Copy>(
    src: &[T],
    dst: &mut mdarray::Slice<T, mdarray::DynRank, mdarray::Strided>,
) {
    assert_eq!(src.len(), dst.len(), "Source size mismatch");

    // mdarray's iter_mut() returns elements in row-major order
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_working_buffer_new() {
        let buf = WorkingBuffer::new();
        assert_eq!(buf.capacity_bytes(), 0);
    }

    #[test]
    fn test_working_buffer_with_capacity() {
        let buf = WorkingBuffer::with_capacity_bytes(1024);
        assert!(buf.capacity_bytes() >= 1024);
    }

    #[test]
    fn test_working_buffer_ensure_capacity() {
        let mut buf = WorkingBuffer::new();
        buf.ensure_capacity::<f64>(100);
        assert!(buf.capacity_bytes() >= 100 * std::mem::size_of::<f64>());
    }

    #[test]
    fn test_working_buffer_as_f64_slice() {
        let mut buf = WorkingBuffer::new();
        let count = 10;
        buf.ensure_capacity::<f64>(count);

        unsafe {
            let slice = buf.as_f64_slice_mut(count);
            assert_eq!(slice.len(), count);

            // Write some data
            for i in 0..count {
                slice[i] = i as f64;
            }

            // Read it back
            let slice = buf.as_f64_slice_mut(count);
            for i in 0..count {
                assert_eq!(slice[i], i as f64);
            }
        }
    }

    #[test]
    fn test_working_buffer_as_complex_slice() {
        let mut buf = WorkingBuffer::new();
        let count = 10;
        buf.ensure_capacity::<Complex<f64>>(count);

        unsafe {
            let slice = buf.as_complex_slice_mut(count);
            assert_eq!(slice.len(), count);

            // Write some data
            for i in 0..count {
                slice[i] = Complex::new(i as f64, (i + 1) as f64);
            }

            // Read it back
            let slice = buf.as_complex_slice_mut(count);
            for i in 0..count {
                assert_eq!(slice[i], Complex::new(i as f64, (i + 1) as f64));
            }
        }
    }

    #[test]
    fn test_working_buffer_reallocation() {
        let mut buf = WorkingBuffer::with_capacity_bytes(100);
        let old_capacity = buf.capacity_bytes();

        buf.ensure_capacity_bytes(1000);
        assert!(buf.capacity_bytes() >= 1000);
        assert!(buf.capacity_bytes() > old_capacity);
    }

    #[test]
    fn test_copy_to_from_contiguous() {
        use mdarray::{tensor, Tensor};

        // Create a 2D tensor: shape [2, 3]
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let original = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Permute it (creates strided view): shape [3, 2]
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]
        let permuted = original.permute([1, 0]);

        // Copy to contiguous buffer (row-major iteration of permuted view)
        // [1, 4, 2, 5, 3, 6]
        let mut buffer = vec![0.0f64; 6];
        super::copy_to_contiguous(&permuted.into_dyn(), &mut buffer);

        assert_eq!(buffer, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Test copy back: create [3, 2] tensor, permute to [2, 3], copy source data
        // Source represents row-major [2, 3] data: [[10, 20, 30], [40, 50, 60]]
        let mut output: Tensor<f64, mdarray::DynRank> = Tensor::from_elem(&[3, 2][..], 0.0);
        {
            // permute_mut [1, 0] gives view of shape [2, 3]
            let mut output_permuted = output.permute_mut([1, 0]);
            let source = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
            super::copy_from_contiguous(&source, &mut output_permuted.into_dyn());
        }

        // Output tensor is [3, 2], which is transpose of [[10, 20, 30], [40, 50, 60]]
        // So output is [[10, 40], [20, 50], [30, 60]]
        assert_eq!(output[[0, 0]], 10.0);
        assert_eq!(output[[0, 1]], 40.0);
        assert_eq!(output[[1, 0]], 20.0);
        assert_eq!(output[[1, 1]], 50.0);
        assert_eq!(output[[2, 0]], 30.0);
        assert_eq!(output[[2, 1]], 60.0);
    }
}
