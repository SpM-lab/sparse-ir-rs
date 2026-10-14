//! Parallel execution on the rayon thread pool.
//!
//! The threaded work of the library (the SVE and the batched Matsubara
//! evaluations) runs on the current rayon pool (the global one unless the
//! caller runs inside `ThreadPool::install`) through the functions here. Its
//! size (`RAYON_NUM_THREADS`) therefore bounds the number of threads the
//! library uses.
//!
//! Every job handed out here runs under the FPU state of the thread that
//! hands it out (`crate::fpu_check::FpuContext`), so the threaded paths
//! compute exactly what the serial ones would. They are used only where each
//! value is computed by the same operations whatever the threading, so the
//! results do not depend on the number of threads.
//!
//! Each parallel region runs serially below a size threshold, set where it
//! is decided: [`PARALLEL_MIN_WORK`] (in `f64` operations) here, and next to
//! the code for the Matsubara evaluations.
//!
//! Internal to the sparse-ir crates; not part of the public API.

/// Number of threads of the current rayon pool
pub fn n_threads() -> usize {
    rayon::current_num_threads()
}

/// Parallel regions with less work than this, in `f64` operations, run on the
/// calling thread (dispatching to the pool would cost more than it saves).
/// On an M1 Pro, `compute_sve` was fastest at about this threshold both in
/// `f64` (ε ≥ 10⁻⁸) and, counting a double-double operation as 16 `f64`
/// ones, in double-double.
pub const PARALLEL_MIN_WORK: usize = 1 << 19;

/// Run `jobs` on the rayon pool and wait for all of them. Pool threads do not
/// have the caller's FPU state, so every job runs under it.
pub fn run_on_pool<'a, F: FnOnce() + Send + 'a>(jobs: impl IntoIterator<Item = F>) {
    // Collected first: the scope closure must be `Send`, the iterator need not be.
    let jobs: Vec<F> = jobs.into_iter().collect();
    let fpu = crate::fpu_check::FpuContext::current();
    rayon::scope(|scope| {
        for job in jobs {
            scope.spawn(move |_| fpu.run(job));
        }
    });
}

/// Run `f(start, chunk)` over contiguous chunks of `out`, where `start` is the
/// index of `chunk[0]` in `out`; on the thread pool when `work` (in `f64`
/// operations) is large. Each element must depend only on its own
/// index, so the result does not depend on the chunking.
pub fn for_each_chunk<T: Send>(out: &mut [T], work: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let n_threads = n_threads().min(out.len());
    if n_threads <= 1 || work < PARALLEL_MIN_WORK {
        f(0, out);
        return;
    }
    let chunk_len = out.len().div_ceil(n_threads);
    let f = &f;
    run_on_pool(
        out.chunks_mut(chunk_len)
            .enumerate()
            .map(|(c, chunk)| move || f(c * chunk_len, chunk)),
    );
}

/// `[f(0), ..., f(n - 1)]`, computed on the pool.
pub fn map_on_pool<R: Send>(n: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    let mut slots: Vec<Option<R>> = (0..n).map(|_| None).collect();
    let f = &f;
    run_on_pool(
        slots
            .iter_mut()
            .enumerate()
            .map(|(i, slot)| move || *slot = Some(f(i))),
    );
    slots
        .into_iter()
        .map(|r| r.expect("every job has run"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `for_each_chunk` hands each chunk its offset in the whole slice, so
    /// the pool result equals the one computed on the calling thread.
    #[test]
    fn for_each_chunk_matches_serial() {
        let fill = |work| {
            let mut out = vec![0.0f64; 10_000];
            for_each_chunk(&mut out, work, |start, chunk| {
                for (i, x) in chunk.iter_mut().enumerate() {
                    *x = ((start + i) as f64 * 0.37).sin();
                }
            });
            out
        };
        assert_eq!(fill(usize::MAX), fill(0));
        assert_eq!(fill(0)[9_999], (9_999.0f64 * 0.37).sin());
    }

    #[test]
    fn map_on_pool_keeps_the_order() {
        assert_eq!(
            map_on_pool(100, |i| i * i),
            (0..100).map(|i| i * i).collect::<Vec<_>>()
        );
        assert!(map_on_pool(0, |i| i).is_empty());
    }

    /// Inside `ThreadPool::install`, the library uses that pool's threads.
    #[test]
    fn uses_the_installed_pool() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(n_threads(), 1);
            let ids = map_on_pool(8, |_| std::thread::current().id());
            assert!(ids.iter().all(|id| *id == ids[0]));
        });
    }
}
