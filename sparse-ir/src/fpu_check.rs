//! FPU state checking and correction for numerical stability
//!
//! This module detects dangerous FPU settings (particularly Flush-to-Zero and
//! Denormals-Are-Zero flags) that can cause incorrect SVD results when called
//! from Intel Fortran programs compiled with `-O3` without `-fp-model precise`.
//!
//! # Background
//!
//! Intel Fortran's `-O3` optimization may set the MXCSR register's FZ and DAZ bits
//! at program startup for performance. However, this causes problems for SVD
//! computations that rely on proper handling of denormalized (subnormal) numbers.
//!
//! # Usage
//!
//! The [`FpuGuard`] RAII guard automatically saves, corrects, and restores FPU state:
//!
//! ```ignore
//! let _guard = FpuGuard::new_protect_computation();
//! // Computation here - FZ/DAZ are disabled
//! // FPU state is automatically restored when _guard is dropped
//! ```
//!
//! # Performance
//!
//! The `stmxcsr`/`ldmxcsr` instructions are very lightweight (a few CPU cycles),
//! so the overhead of checking and restoring FPU state is negligible compared
//! to actual matrix computations.

use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};

/// MXCSR bit positions
const MXCSR_FZ_BIT: u32 = 15; // Flush to Zero
const MXCSR_DAZ_BIT: u32 = 6; // Denormals Are Zero

/// Flag to track if warning has been shown (only show once per process)
static WARNING_SHOWN: AtomicBool = AtomicBool::new(false);

/// Lazy initialization to check FPU state at library load time
static FPU_CHECK_INIT: Lazy<bool> = Lazy::new(|| {
    let state = get_fpu_state();
    if state.is_dangerous() {
        print_fpu_warning(&state);
        WARNING_SHOWN.store(true, Ordering::SeqCst);
        true // dangerous state detected
    } else {
        false
    }
});

/// Result of FPU state check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FpuState {
    /// Raw MXCSR register value
    pub mxcsr: u32,
    /// Flush to Zero flag
    pub flush_to_zero: bool,
    /// Denormals Are Zero flag
    pub denormals_are_zero: bool,
}

impl FpuState {
    /// Check if FPU state is dangerous for numerical computation
    pub fn is_dangerous(&self) -> bool {
        self.flush_to_zero || self.denormals_are_zero
    }
}

impl std::fmt::Display for FpuState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MXCSR=0x{:08X}, FZ={}, DAZ={}",
            self.mxcsr, self.flush_to_zero as u8, self.denormals_are_zero as u8
        )
    }
}

/// Print FPU warning message (called only once)
fn print_fpu_warning(state: &FpuState) {
    eprintln!();
    eprintln!("================================================================================");
    eprintln!("sparse-ir WARNING: Dangerous FPU settings detected!");
    eprintln!("================================================================================");
    eprintln!();
    eprintln!("  Current FPU state: {}", state);
    eprintln!();
    eprintln!("  Problem: Flush-to-Zero (FZ) or Denormals-Are-Zero (DAZ) is enabled.");
    eprintln!("           This causes subnormal numbers to be treated as zero, which");
    eprintln!("           can produce INCORRECT results in SVD/SVE computations.");
    eprintln!();
    eprintln!("  Common cause: Intel Fortran compiler (ifort/ifx) with -O3 optimization");
    eprintln!("                sets FZ/DAZ flags at program startup for performance.");
    eprintln!();
    eprintln!("  Solution: Add '-fp-model precise' flag when compiling your Fortran code:");
    eprintln!();
    eprintln!("      ifort -O3 -fp-model precise your_program.f90");
    eprintln!("      ifx   -O3 -fp-model precise your_program.f90");
    eprintln!();
    eprintln!("  For Quantum ESPRESSO/EPW, add to make.inc:");
    eprintln!();
    eprintln!("      FFLAGS += -fp-model precise");
    eprintln!();
    eprintln!("  Action: sparse-ir will temporarily disable FZ/DAZ during each computation");
    eprintln!("          and restore the original settings afterward.");
    eprintln!("          Results will be correct, but please fix the compiler flags");
    eprintln!("          to avoid this warning.");
    eprintln!();
    eprintln!("================================================================================");
    eprintln!();
}

/// Get current FPU state (x86/x86_64 only)
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub fn get_fpu_state() -> FpuState {
    let mut mxcsr: u32 = 0;
    unsafe {
        std::arch::asm!(
            "stmxcsr [{}]",
            in(reg) &mut mxcsr,
            options(nostack)
        );
    }

    FpuState {
        mxcsr,
        flush_to_zero: (mxcsr >> MXCSR_FZ_BIT) & 1 != 0,
        denormals_are_zero: (mxcsr >> MXCSR_DAZ_BIT) & 1 != 0,
    }
}

/// Get current FPU state (non-x86 fallback)
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
pub fn get_fpu_state() -> FpuState {
    // On non-x86 platforms, assume safe defaults
    FpuState {
        mxcsr: 0,
        flush_to_zero: false,
        denormals_are_zero: false,
    }
}

/// Set MXCSR register value (x86/x86_64 only)
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn set_mxcsr(value: u32) {
    unsafe {
        std::arch::asm!(
            "ldmxcsr [{}]",
            in(reg) &value,
            options(nostack)
        );
    }
}

/// Set MXCSR register value (non-x86 fallback - no-op)
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
fn set_mxcsr(_value: u32) {
    // No-op on non-x86 platforms
}

/// Initialize FPU check (call this early to trigger warning if needed)
///
/// This function triggers the lazy initialization which checks FPU state
/// and prints a warning if dangerous settings are detected.
/// The warning is only printed once per process.
pub fn init_fpu_check() {
    let _ = *FPU_CHECK_INIT;
}

/// RAII guard that protects a computation from dangerous FPU settings
///
/// On creation:
/// 1. Triggers one-time FPU check and warning (if not already done)
/// 2. If FZ or DAZ is enabled, disables them temporarily
///
/// On drop:
/// - Restores the original FPU state
///
/// # Example
///
/// ```ignore
/// {
///     let _guard = FpuGuard::new_protect_computation();
///     // Computation here - FZ/DAZ are disabled
///     perform_svd_computation();
/// } // Original FPU state is restored here
/// ```
///
/// # Performance
///
/// The overhead is negligible (a few CPU cycles for stmxcsr/ldmxcsr).
pub struct FpuGuard {
    original_mxcsr: u32,
    needs_restore: bool,
}

impl FpuGuard {
    /// Create a new guard that protects computation from FZ/DAZ
    ///
    /// - Triggers one-time warning if dangerous FPU settings are detected
    /// - Temporarily disables FZ/DAZ if they are enabled
    /// - Restores original state when dropped
    pub fn new_protect_computation() -> Self {
        // Trigger one-time FPU check and warning
        let _ = *FPU_CHECK_INIT;

        let state = get_fpu_state();
        let original_mxcsr = state.mxcsr;

        if state.is_dangerous() {
            // Clear FZ and DAZ bits
            let safe_mxcsr = original_mxcsr & !((1 << MXCSR_FZ_BIT) | (1 << MXCSR_DAZ_BIT));
            set_mxcsr(safe_mxcsr);

            Self {
                original_mxcsr,
                needs_restore: true,
            }
        } else {
            Self {
                original_mxcsr,
                needs_restore: false,
            }
        }
    }

    /// Check if the guard needed to modify FPU state
    pub fn was_modified(&self) -> bool {
        self.needs_restore
    }
}

impl Drop for FpuGuard {
    fn drop(&mut self) {
        if self.needs_restore {
            // Restore original FPU state
            set_mxcsr(self.original_mxcsr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_fpu_state() {
        let state = get_fpu_state();
        // Just verify we can read the state without panicking
        println!("Current FPU state: {}", state);
    }

    #[test]
    fn test_fpu_guard_creation() {
        let guard = FpuGuard::new_protect_computation();
        // Guard should be created successfully
        drop(guard);
    }

    #[test]
    fn test_fpu_state_display() {
        let state = FpuState {
            mxcsr: 0x1F80,
            flush_to_zero: false,
            denormals_are_zero: false,
        };
        let display = format!("{}", state);
        assert!(display.contains("MXCSR=0x00001F80"));
        assert!(display.contains("FZ=0"));
        assert!(display.contains("DAZ=0"));
    }

    #[test]
    fn test_fpu_state_dangerous() {
        let safe_state = FpuState {
            mxcsr: 0x1F80,
            flush_to_zero: false,
            denormals_are_zero: false,
        };
        assert!(!safe_state.is_dangerous());

        let dangerous_fz = FpuState {
            mxcsr: 0x9F80,
            flush_to_zero: true,
            denormals_are_zero: false,
        };
        assert!(dangerous_fz.is_dangerous());

        let dangerous_daz = FpuState {
            mxcsr: 0x1FC0,
            flush_to_zero: false,
            denormals_are_zero: true,
        };
        assert!(dangerous_daz.is_dangerous());
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[test]
    fn test_fpu_guard_restores_state() {
        let original_state = get_fpu_state();

        {
            let _guard = FpuGuard::new_protect_computation();
            // State might be modified here
        }

        let restored_state = get_fpu_state();
        assert_eq!(original_state.mxcsr, restored_state.mxcsr);
    }
}
