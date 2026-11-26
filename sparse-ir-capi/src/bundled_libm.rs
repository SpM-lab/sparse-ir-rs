//! The C math functions that the library and its dependencies call, computed
//! with the pure-Rust `libm` crate (the `bundled-libm` feature).
//!
//! Without it, `f64::exp`, `f64::sin`, ... and the same methods inside
//! `num-complex` and the linear-algebra crates become calls to the system
//! libm, whose last bits differ between versions: glibc 2.35 and 2.39 round
//! `expm1` and `cosh` differently, and the same `libsparse_ir_capi.so` returns
//! different Matsubara transforms on the two. The definitions below give the
//! library its own copy. On ELF targets `build.rs` links the library with
//! `-Bsymbolic`, so that its calls bind to these definitions even when the
//! system libm is loaded first, as it is in every Python or Julia process.
//!
//! The list is what the C library imports from libm without the feature;
//! the `Rust-CAPI` workflow checks that none of it is left undefined.

macro_rules! unary {
    ($($name:ident => $f:path : $t:ty),* $(,)?) => {
        $(
            #[unsafe(no_mangle)]
            pub extern "C" fn $name(x: $t) -> $t {
                $f(x)
            }
        )*
    };
}

macro_rules! binary {
    ($($name:ident => $f:path : $t:ty),* $(,)?) => {
        $(
            #[unsafe(no_mangle)]
            pub extern "C" fn $name(x: $t, y: $t) -> $t {
                $f(x, y)
            }
        )*
    };
}

unary! {
    asin => libm::asin: f64, cos => libm::cos: f64, cosh => libm::cosh: f64, exp => libm::exp: f64,
    exp2 => libm::exp2: f64, expm1 => libm::expm1: f64, log => libm::log: f64,
    log10 => libm::log10: f64, log1p => libm::log1p: f64, log2 => libm::log2: f64,
    sin => libm::sin: f64, sinh => libm::sinh: f64, tanh => libm::tanh: f64,
    cosf => libm::cosf: f32, coshf => libm::coshf: f32, expf => libm::expf: f32,
    expm1f => libm::expm1f: f32, logf => libm::logf: f32, log1pf => libm::log1pf: f32,
    sinf => libm::sinf: f32, sinhf => libm::sinhf: f32, tanhf => libm::tanhf: f32,
}

binary! {
    atan2 => libm::atan2: f64, hypot => libm::hypot: f64, pow => libm::pow: f64,
    atan2f => libm::atan2f: f32, hypotf => libm::hypotf: f32, powf => libm::powf: f32,
}

/// `sin(x)` and `cos(x)` into `*s` and `*c`, as the GNU C function.
///
/// # Safety
///
/// `s` and `c` must be valid for writes of one value each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sincos(x: f64, s: *mut f64, c: *mut f64) {
    let (sin, cos) = libm::sincos(x);
    // SAFETY: the caller guarantees that `s` and `c` are valid for writes;
    // compiler-generated calls pass two distinct stack slots.
    unsafe {
        s.write(sin);
        c.write(cos);
    }
}

/// `sinf(x)` and `cosf(x)` into `*s` and `*c`, as the GNU C function.
///
/// # Safety
///
/// `s` and `c` must be valid for writes of one value each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sincosf(x: f32, s: *mut f32, c: *mut f32) {
    let (sin, cos) = libm::sincosf(x);
    // SAFETY: as in `sincos`.
    unsafe {
        s.write(sin);
        c.write(cos);
    }
}

#[cfg(test)]
mod tests {
    // The definitions must forward to the libm crate unchanged, special
    // values included: the C library relies on their bits.
    const POINTS: [f64; 9] = [
        -700.5,
        -3.25,
        -0.0,
        0.0,
        1e-300,
        0.75,
        2.5e5,
        f64::INFINITY,
        f64::NAN,
    ];

    #[test]
    fn unary_and_binary_forward_bitwise() {
        let pairs: [(extern "C" fn(f64) -> f64, fn(f64) -> f64); 13] = [
            (super::asin, libm::asin),
            (super::cos, libm::cos),
            (super::cosh, libm::cosh),
            (super::exp, libm::exp),
            (super::exp2, libm::exp2),
            (super::expm1, libm::expm1),
            (super::log, libm::log),
            (super::log10, libm::log10),
            (super::log1p, libm::log1p),
            (super::log2, libm::log2),
            (super::sin, libm::sin),
            (super::sinh, libm::sinh),
            (super::tanh, libm::tanh),
        ];
        for (ours, theirs) in pairs {
            for x in POINTS {
                assert_eq!(ours(x).to_bits(), theirs(x).to_bits(), "x = {x}");
            }
        }
        for x in POINTS {
            assert_eq!(super::pow(x, 1.5).to_bits(), libm::pow(x, 1.5).to_bits());
            assert_eq!(
                super::atan2(x, -2.0).to_bits(),
                libm::atan2(x, -2.0).to_bits()
            );
            assert_eq!(
                super::hypot(x, 3.0).to_bits(),
                libm::hypot(x, 3.0).to_bits()
            );
            let y = x as f32;
            assert_eq!(super::expf(y).to_bits(), libm::expf(y).to_bits());
            assert_eq!(super::powf(y, 0.5).to_bits(), libm::powf(y, 0.5).to_bits());
        }
    }

    #[test]
    fn sincos_writes_both_outputs() {
        for x in POINTS {
            let (mut s, mut c) = (f64::NAN, f64::NAN);
            // SAFETY: `s` and `c` are distinct local variables.
            unsafe { super::sincos(x, &mut s, &mut c) };
            let (sin, cos) = libm::sincos(x);
            assert_eq!((s.to_bits(), c.to_bits()), (sin.to_bits(), cos.to_bits()));
            let (mut sf, mut cf) = (f32::NAN, f32::NAN);
            // SAFETY: as above.
            unsafe { super::sincosf(x as f32, &mut sf, &mut cf) };
            let (sinf, cosf) = libm::sincosf(x as f32);
            assert_eq!(
                (sf.to_bits(), cf.to_bits()),
                (sinf.to_bits(), cosf.to_bits())
            );
        }
    }
}
