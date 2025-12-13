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
}
