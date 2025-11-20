//! Common macros for opaque type implementations
//!
//! This module provides macros to generate common C API functions for opaque types,
//! following libsparseir's DECLARE_OPAQUE_TYPE pattern.

/// Generate common opaque type functions: release, clone, is_assigned, get_raw_ptr
///
/// This macro implements the standard lifecycle functions for opaque C API types,
/// matching libsparseir's DECLARE_OPAQUE_TYPE pattern.
///
/// # Requirements
/// - The type must implement `Clone`
/// - The type name should follow the pattern `spir_*`
///
/// # Generated functions
/// - `spir_<TYPE>_release()` - Drops the object
/// - `spir_<TYPE>_clone()` - Creates a shallow copy (Arc-based, cheap)
/// - `spir_<TYPE>_is_assigned()` - Checks if pointer is valid
/// - `_spir_<TYPE>_get_raw_ptr()` - Returns raw pointer for debugging
///
/// # Example
/// ```ignore
/// // In types.rs
/// #[derive(Clone)]
/// #[repr(C)]
/// pub struct spir_kernel {
///     inner: KernelType,
/// }
///
/// // In kernel.rs
/// impl_opaque_type_common!(kernel);
/// ```
#[macro_export]
macro_rules! impl_opaque_type_common {
    ($type_name:ident) => {
        paste::paste! {
            /// Release the object by dropping it
            ///
            /// # Safety
            /// The caller must ensure that the pointer is valid and not used after this call.
            #[unsafe(no_mangle)]
            pub extern "C" fn [<spir_ $type_name _release>](obj: *mut [<spir_ $type_name>]) {
                if obj.is_null() {
                    return;
                }
                // Convert back to Box and drop - safe because we check for null
                unsafe {
                    let _ = Box::from_raw(obj);
                }
            }

            /// Clone the object (shallow copy with Arc reference counting)
            ///
            /// # Safety
            /// The caller must ensure that the source pointer is valid.
            /// The returned pointer must be freed with `spir_<type>_release()`.
            ///
            /// # Returns
            /// A new pointer to a cloned object, or null if input is null or panic occurs.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<spir_ $type_name _clone>](
                src: *const [<spir_ $type_name>]
            ) -> *mut [<spir_ $type_name>] {
                if src.is_null() {
                    return std::ptr::null_mut();
                }

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    let src_ref = &*src;
                    let cloned = (*src_ref).clone();
                    // Clone uses Arc reference counting (cheap operation)
                    Box::into_raw(Box::new(cloned))
                }));

                result.unwrap_or(std::ptr::null_mut())
            }

            /// Check if the object pointer is valid (non-null and dereferenceable)
            ///
            /// # Returns
            /// 1 if the object is valid, 0 otherwise
            #[unsafe(no_mangle)]
            pub extern "C" fn [<spir_ $type_name _is_assigned>](
                obj: *const [<spir_ $type_name>]
            ) -> i32 {
                if obj.is_null() {
                    return 0;
                }

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    let _ = &*obj;
                    1
                }));

                result.unwrap_or(0)
            }

            /// Get the raw pointer for debugging purposes (internal use only)
            ///
            /// # Returns
            /// The raw pointer cast to `void*`, or null if input is null
            #[unsafe(no_mangle)]
            pub extern "C" fn [<_spir_ $type_name _get_raw_ptr>](
                obj: *const [<spir_ $type_name>]
            ) -> *const std::ffi::c_void {
                if obj.is_null() {
                    return std::ptr::null();
                }

                obj as *const std::ffi::c_void
            }
        }
    };
}
