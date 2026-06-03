#[cfg(feature = "profile")]
mod imp {
    use std::{
        cell::RefCell,
        collections::HashMap,
        sync::{Mutex, OnceLock},
        time::{Duration, Instant},
    };

    #[derive(Clone, Copy, Debug, Default)]
    struct LocalStat {
        calls: u64,
        total: Duration,
        max: Duration,
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct ProfileStat {
        pub calls: u64,
        pub total: Duration,
        pub max: Duration,
    }

    thread_local! {
        static LOCAL_STATS: RefCell<HashMap<&'static str, LocalStat>> = RefCell::new(HashMap::new());
    }

    static GLOBAL_STATS: OnceLock<Mutex<HashMap<&'static str, ProfileStat>>> = OnceLock::new();

    pub struct ScopeTimer {
        name: &'static str,
        start: Instant,
    }

    impl ScopeTimer {
        #[inline(always)]
        pub fn new(name: &'static str) -> Self {
            Self {
                name,
                start: Instant::now(),
            }
        }
    }

    impl Drop for ScopeTimer {
        fn drop(&mut self) {
            record(self.name, self.start.elapsed());
        }
    }

    #[inline(always)]
    pub fn record(name: &'static str, elapsed: Duration) {
        LOCAL_STATS.with(|stats| {
            let mut stats = stats.borrow_mut();
            let entry = stats.entry(name).or_default();
            entry.calls += 1;
            entry.total += elapsed;
            entry.max = entry.max.max(elapsed);
        });
    }

    pub fn flush_thread() {
        let drained = LOCAL_STATS.with(|stats| std::mem::take(&mut *stats.borrow_mut()));
        if drained.is_empty() {
            return;
        }

        let mut global = GLOBAL_STATS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("profile stats mutex poisoned");
        for (name, stat) in drained {
            let entry = global.entry(name).or_default();
            entry.calls += stat.calls;
            entry.total += stat.total;
            entry.max = entry.max.max(stat.max);
        }
    }

    pub fn report() -> Vec<(&'static str, ProfileStat)> {
        flush_thread();
        let mut rows = GLOBAL_STATS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("profile stats mutex poisoned")
            .iter()
            .map(|(&name, &stat)| (name, stat))
            .collect::<Vec<_>>();
        rows.sort_by(|(_, left), (_, right)| right.total.cmp(&left.total));
        rows
    }

    pub fn print_report() {
        let rows = report();
        if rows.is_empty() {
            eprintln!("profile: no samples");
            return;
        }

        eprintln!("profile:");
        eprintln!(
            "{:<36} {:>12} {:>12} {:>12} {:>12}",
            "name", "calls", "total_ms", "avg_us", "max_us"
        );
        for (name, stat) in rows {
            let total_us = stat.total.as_micros();
            let avg_us = total_us / stat.calls.max(1) as u128;
            eprintln!(
                "{:<36} {:>12} {:>12.3} {:>12} {:>12}",
                name,
                stat.calls,
                total_us as f64 / 1000.0,
                avg_us,
                stat.max.as_micros()
            );
        }
    }
}

#[cfg(feature = "profile")]
pub use imp::{ScopeTimer, print_report, report};

#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn print_report() {}

#[macro_export]
macro_rules! scope_profile {
    ($name:literal) => {
        #[cfg(feature = "profile")]
        let _scope_profile_guard = $crate::profile::ScopeTimer::new($name);
    };
}
