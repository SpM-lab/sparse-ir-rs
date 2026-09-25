//! Common macros for opaque type implementations
//!
//! This module provides macros to generate common C API functions for opaque types,
//! following libsparseir's DECLARE_OPAQUE_TYPE pattern.

/// Debug print macro that prints to stderr only if `SPARSEIR_DEBUG` enables
/// debug output (see [`is_debug_enabled`](crate::is_debug_enabled))
///
/// Usage: `debug_println!("format string", args...)`
#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        if $crate::is_debug_enabled() {
            eprintln!("[SPARSEIR DEBUG] {}", format!($($arg)*));
        }
    };
}

/// Debug print macro for errors that prints to stderr only if `SPARSEIR_DEBUG`
/// enables debug output (see [`is_debug_enabled`](crate::is_debug_enabled))
///
/// Usage: `debug_eprintln!("format string", args...)`
#[macro_export]
macro_rules! debug_eprintln {
    ($($arg:tt)*) => {
        if $crate::is_debug_enabled() {
            eprintln!("[SPARSEIR DEBUG ERROR] {}", format!($($arg)*));
        }
    };
}

/// Generate common opaque type functions: release, clone, is_assigned, get_raw_ptr
///
/// This macro implements the standard lifecycle functions for opaque C API types,
/// matching libsparseir's DECLARE_OPAQUE_TYPE pattern.
///
/// # Requirements
/// - The type must implement `Clone`
/// - The type name should follow the pattern `spir_*`
/// - The invoking crate must depend on `paste`: the expansion calls
///   `paste::paste!`
///
/// # Generated functions
/// - `spir_<TYPE>_release()` - Drops the object
/// - `spir_<TYPE>_clone()` - Creates a shallow copy (Arc-based, cheap)
/// - `spir_<TYPE>_is_assigned()` - Checks if pointer is valid
/// - `_spir_<TYPE>_get_raw_ptr()` - Returns raw pointer for debugging
///
/// # Example
/// ```
/// use sparse_ir_capi::impl_opaque_type_common;
///
/// // Any `spir_` name that this library does not define already: the
/// // generated functions are exported unmangled.
/// #[allow(non_camel_case_types)]
/// #[derive(Clone)]
/// #[repr(C)]
/// pub struct spir_demo {
///     value: f64,
/// }
///
/// // Defines spir_demo_release, spir_demo_clone, spir_demo_is_assigned and
/// // _spir_demo_get_raw_ptr.
/// impl_opaque_type_common!(demo);
///
/// let obj = Box::into_raw(Box::new(spir_demo { value: 1.5 }));
/// assert_eq!(spir_demo_is_assigned(obj), 1);
/// assert_eq!(_spir_demo_get_raw_ptr(obj), obj as *const std::ffi::c_void);
///
/// let copy = spir_demo_clone(obj);
/// assert!(!copy.is_null() && copy != obj);
/// // SAFETY: `copy` was just returned non-null by `spir_demo_clone`.
/// assert_eq!(unsafe { (*copy).value }, 1.5);
///
/// spir_demo_release(copy);
/// spir_demo_release(obj);
///
/// // NULL is accepted by every generated function.
/// assert_eq!(spir_demo_is_assigned(std::ptr::null()), 0);
/// assert!(spir_demo_clone(std::ptr::null()).is_null());
/// assert!(_spir_demo_get_raw_ptr(std::ptr::null()).is_null());
/// spir_demo_release(std::ptr::null_mut());
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
            pub extern "C" fn [<spir_ $type_name _clone>](
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

            /// Check if the object pointer is non-null.
            ///
            /// Note: This only performs a null check. It cannot detect dangling
            /// pointers; dereferencing an arbitrary non-null pointer would be
            /// undefined behaviour that `catch_unwind` cannot reliably catch.
            ///
            /// # Returns
            /// 1 if the pointer is non-null, 0 otherwise
            #[unsafe(no_mangle)]
            pub extern "C" fn [<spir_ $type_name _is_assigned>](
                obj: *const [<spir_ $type_name>]
            ) -> i32 {
                if obj.is_null() { 0 } else { 1 }
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
