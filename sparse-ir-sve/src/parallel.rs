//! Parallel execution on the rayon thread pool.
//!
//! The threaded work of the library (the SVE) goes through the functions
//! here. Without the `parallel` feature (the default; never enabled by the
//! C API), and inside [`sequential`], it runs on the calling thread. (The
//! SVDs of the fitters go through tenferro's `CpuBackend`, which has a pool
//! of its own, outside these functions; see #344.)
//! With the feature it runs on the current rayon pool (the global one unless
//! the caller runs inside `ThreadPool::install`), whose size
//! (`RAYON_NUM_THREADS`) therefore bounds the number of threads the library
//! uses.
//!
//! Every job handed out here runs under the FPU state of the thread that
//! hands it out (`crate::fpu_check::FpuContext`), so the threaded paths
//! compute exactly what the serial ones would. They are used only where each
//! value is computed by the same operations whatever the threading, so the
//! results do not depend on the number of threads.
//!
//! Each parallel region states its estimated work in `f64` operations and
//! runs on the calling thread below the one threshold [`PARALLEL_MIN_WORK`]
//! (or with a single thread).
//!
//! Internal to the sparse-ir crates; not part of the public API.

use std::cell::Cell;

thread_local! {
    /// Whether the calling thread is inside [`sequential`]
    static SEQUENTIAL: Cell<bool> = const { Cell::new(false) };
}

/// Run `f` with all the parallel regions it reaches on the calling thread,
/// even with the `parallel` feature (as the C API does for every call)
pub fn sequential<R>(f: impl FnOnce() -> R) -> R {
    /// Restores the previous state, also when `f` panics
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SEQUENTIAL.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(SEQUENTIAL.with(|s| s.replace(true)));
    f()
}

/// Number of threads the parallel regions may use: that of the current
/// rayon pool, or 1 without the `parallel` feature or inside [`sequential`]
pub fn n_threads() -> usize {
    if SEQUENTIAL.with(Cell::get) {
        return 1;
    }
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Parallel regions with less work than this, in `f64` operations, run on the
/// calling thread (dispatching to the pool would cost more than it saves).
/// On an M1 Pro, `compute_sve` was fastest at about this threshold both in
/// `f64` (ε ≥ 10⁻⁸) and, counting a double-double operation as 16 `f64`
/// ones, in double-double.
pub const PARALLEL_MIN_WORK: usize = 1 << 19;

/// Whether a region of `work` `f64` operations goes to the pool
fn on_pool(work: usize) -> bool {
    n_threads() > 1 && work >= PARALLEL_MIN_WORK
}

/// [`for_each_chunk`] splits its work into this many chunks per thread, so
/// that idle threads take over the chunks of slower ones (on a CPU with
/// performance and efficiency cores, the latter would otherwise hold up the
/// whole region). On an M1 Pro, this made the double-double `compute_sve`
/// 10–15 % faster on 8 threads than one chunk per thread.
const CHUNKS_PER_THREAD: usize = 4;

/// Run `jobs` on the rayon pool and wait for all of them. Pool threads do not
/// have the caller's FPU state, so every job runs under it. Without the
/// `parallel` feature, or with a single thread, the jobs run in order on the
/// calling thread.
pub fn run_on_pool<'a, F: FnOnce() + Send + 'a>(jobs: impl IntoIterator<Item = F>) {
    #[cfg(feature = "parallel")]
    if n_threads() > 1 {
        // Collected first: the scope closure must be `Send`, the iterator need not be.
        let jobs: Vec<F> = jobs.into_iter().collect();
        let fpu = crate::fpu_check::FpuContext::current();
        rayon::scope(|scope| {
            for job in jobs {
                scope.spawn(move |_| fpu.run(job));
            }
        });
        return;
    }
    for job in jobs {
        job();
    }
}

/// Run `f(start, chunk)` over contiguous chunks of `out`, where `start` is the
/// index of `chunk[0]` in `out`; on the thread pool when `work` (in `f64`
/// operations) is large. Each element must depend only on its own
/// index, so the result does not depend on the chunking.
pub fn for_each_chunk<T: Send>(out: &mut [T], work: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if out.len() <= 1 || !on_pool(work) {
        f(0, out);
        return;
    }
    let chunk_len = out
        .len()
        .div_ceil(n_threads().min(out.len()) * CHUNKS_PER_THREAD);
    let f = &f;
    run_on_pool(
        out.chunks_mut(chunk_len)
            .enumerate()
            .map(|(c, chunk)| move || f(c * chunk_len, chunk)),
    );
}

/// `[f(0), ..., f(n - 1)]`, one pool job per item when `work` (in `f64`
/// operations, for all items) is large
///
/// A single large item goes to the pool too: the parallel regions inside it
/// then start from a pool thread, which is cheaper than from outside the
/// pool (on an M1 Pro, 3 % of a double-double SVE of a single block).
pub fn map_on_pool<R: Send>(n: usize, work: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    if !on_pool(work) {
        return (0..n).map(f).collect();
    }
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
            map_on_pool(100, usize::MAX, |i| i * i),
            (0..100).map(|i| i * i).collect::<Vec<_>>()
        );
        assert!(map_on_pool(0, usize::MAX, |i| i).is_empty());
        assert_eq!(map_on_pool(3, 0, |i| i), vec![0, 1, 2]);
    }

    /// A single large item runs on a pool thread, a small one on the caller's
    /// (a test thread, outside the pool).
    #[cfg(feature = "parallel")]
    #[test]
    fn map_on_pool_sends_a_single_large_item_to_the_pool() {
        let thread_index = |work| map_on_pool(1, work, |_| rayon::current_thread_index())[0];
        assert_eq!(thread_index(0), None);
        if n_threads() > 1 {
            assert!(thread_index(usize::MAX).is_some());
        }
    }

    /// Inside `ThreadPool::install`, the library uses that pool's threads.
    #[cfg(feature = "parallel")]
    #[test]
    fn uses_the_installed_pool() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(n_threads(), 1);
            let ids = map_on_pool(8, usize::MAX, |_| std::thread::current().id());
            assert!(ids.iter().all(|id| *id == ids[0]));
        });
    }

    /// Inside `sequential` everything runs on the calling thread, also with
    /// the `parallel` feature and a large pool, and also in nested calls; the
    /// state is restored afterwards, also after a panic.
    #[test]
    fn sequential_runs_on_the_calling_thread() {
        let caller = std::thread::current().id();
        let on_caller = || {
            map_on_pool(16, usize::MAX, |_| std::thread::current().id())
                .iter()
                .all(|id| *id == caller)
        };
        sequential(|| {
            assert_eq!(n_threads(), 1);
            assert!(on_caller());
            sequential(|| assert!(on_caller()));
            assert_eq!(n_threads(), 1);
        });
        let panicked = std::panic::catch_unwind(|| sequential(|| panic!("in sequential")));
        assert!(panicked.is_err());
        assert!(!SEQUENTIAL.with(Cell::get));
        #[cfg(not(feature = "parallel"))]
        assert_eq!(n_threads(), 1);
    }
}
