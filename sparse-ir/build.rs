fn main() {
    // macOS なら Accelerate.framework を使う
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:warning=Using macOS Accelerate framework for BLAS");
    } else {
        // 他のOSなら pkg-config で探す
        let candidates = ["openblas", "blas", "cblas"];
        let mut found = false;

        for lib in candidates {
            if pkg_config::probe_library(lib).is_ok() {
                println!("cargo:warning=Found system BLAS: {}", lib);
                found = true;
                break;
            }
        }

        if !found {
            println!("cargo:warning=No system BLAS found! Tests may not link properly.");
            println!("cargo:warning=Consider installing OpenBLAS or another BLAS implementation.");
        }
    }
}
