//! CPU features the library is built for (#332)
//!
//! The default build uses the fused multiply-add (`.cargo/config.toml`). On a
//! CPU without it the first FMA instruction raises SIGILL. The bindings call
//! these functions at import, before any numerical code runs, and raise a
//! clear error when `spir_build_uses_fma() && !spir_cpu_has_fma()`.

use std::sync::OnceLock;

/// Whether this build of the library may execute fused multiply-add
/// instructions
///
/// # Returns
/// 1 if the library was compiled with the FMA target feature (the default
/// build on x86-64 and every aarch64 build), 0 if it was built without it,
/// e.g. with `RUSTFLAGS="-C target-feature=-fma"`. A build that returns 0
/// runs on any CPU of its architecture.
#[unsafe(no_mangle)]
pub extern "C" fn spir_build_uses_fma() -> libc::c_int {
    // FMA is in the aarch64 base instruction set, which has no `fma` target
    // feature to test
    libc::c_int::from(cfg!(any(target_feature = "fma", target_arch = "aarch64")))
}

/// Whether this CPU, and the operating system, support fused multiply-add
/// instructions
///
/// On x86-64 this reads CPUID and the enabled register state directly,
/// because `is_x86_feature_detected!` answers "yes" without asking the CPU
/// in a build compiled with the feature. On aarch64 FMA is part of the base
/// instruction set. The result is computed once and cached. The function
/// executes no FMA instruction itself, so it is safe to call on any CPU.
///
/// # Returns
/// 1 if the CPU can execute FMA instructions, 0 otherwise (and on
/// architectures other than x86-64 and aarch64)
#[unsafe(no_mangle)]
pub extern "C" fn spir_cpu_has_fma() -> libc::c_int {
    static HAS_FMA: OnceLock<bool> = OnceLock::new();
    libc::c_int::from(*HAS_FMA.get_or_init(detect_fma))
}

#[cfg(target_arch = "x86_64")]
fn detect_fma() -> bool {
    use std::arch::x86_64::__cpuid;

    const FMA: u32 = 1 << 12;
    const OSXSAVE: u32 = 1 << 27;
    const AVX: u32 = 1 << 28;
    // XMM and YMM register state enabled by the OS (XCR0 bits 1 and 2)
    const XMM_YMM_STATE: u64 = 0b110;

    // CPUID leaf 1 exists on every x86-64 CPU
    let ecx = __cpuid(1).ecx;
    if ecx & (FMA | OSXSAVE | AVX) != FMA | OSXSAVE | AVX {
        return false;
    }
    // SAFETY: OSXSAVE (checked above) means XGETBV is available and the OS
    // has enabled it.
    let xcr0 = unsafe { xgetbv0() };
    xcr0 & XMM_YMM_STATE == XMM_YMM_STATE
}

/// XCR0, read in a function compiled with the `xsave` feature that
/// `_xgetbv` requires
///
/// # Safety
/// The CPU must support XGETBV (CPUID.1:ECX.OSXSAVE).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "xsave")]
unsafe fn xgetbv0() -> u64 {
    // SAFETY: the caller guarantees XGETBV (see above).
    unsafe { std::arch::x86_64::_xgetbv(0) }
}

#[cfg(target_arch = "aarch64")]
fn detect_fma() -> bool {
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_fma() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_uses_fma_matches_the_compile_flags() {
        let expected = cfg!(any(target_feature = "fma", target_arch = "aarch64"));
        assert_eq!(spir_build_uses_fma(), i32::from(expected));
    }

    /// The detection must not depend on how the library was compiled. In a
    /// build without the `fma` feature std asks the CPU, so it is an
    /// independent answer; in a build with it std answers "yes" unasked.
    /// Rosetta 2, for one, runs x86-64 code without advertising FMA.
    #[test]
    fn cpu_has_fma_asks_the_cpu() {
        let has = spir_cpu_has_fma();
        assert!(has == 0 || has == 1);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(has, 1);
        #[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
        assert_eq!(has == 1, std::is_x86_feature_detected!("fma"));
        assert_eq!(spir_cpu_has_fma(), has, "cached value");
    }
}
