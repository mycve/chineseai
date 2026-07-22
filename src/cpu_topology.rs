use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct CpuPlacement {
    pub cpu: usize,
    pub node: usize,
    pub package: usize,
    pub core: usize,
    pub llc: usize,
    pub smt_level: usize,
}

#[cfg(target_os = "linux")]
fn read_usize(path: impl AsRef<std::path::Path>) -> Option<usize> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(target_os = "linux")]
fn cpu_node(cpu: usize) -> usize {
    let path = format!("/sys/devices/system/cpu/cpu{cpu}");
    std::fs::read_dir(path)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().to_str().map(str::to_owned))
        .find_map(|name| name.strip_prefix("node")?.parse().ok())
        .unwrap_or(0)
}

#[cfg(target_os = "linux")]
fn cpu_llc(cpu: usize, package: usize) -> usize {
    let path = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
    std::fs::read_dir(path)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .find_map(|entry| {
            let path = entry.path();
            (read_usize(path.join("level")) == Some(3))
                .then(|| read_usize(path.join("id")))
                .flatten()
        })
        // 保证不同封装在缺失 sysfs cache id 时不会被错误归为同一 LLC。
        .unwrap_or(package)
}

fn interleave_by_llc(mut placements: Vec<CpuPlacement>) -> Vec<CpuPlacement> {
    let mut groups = BTreeMap::<(usize, usize, usize), Vec<CpuPlacement>>::new();
    for placement in placements.drain(..) {
        groups
            .entry((placement.node, placement.package, placement.llc))
            .or_default()
            .push(placement);
    }
    for group in groups.values_mut() {
        group.sort_by_key(|placement| (placement.core, placement.cpu));
    }
    let max_group_len = groups.values().map(Vec::len).max().unwrap_or(0);
    let mut out = Vec::new();
    for index in 0..max_group_len {
        for group in groups.values() {
            if let Some(&placement) = group.get(index) {
                out.push(placement);
            }
        }
    }
    out
}

#[cfg(target_os = "linux")]
fn allowed_cpus() -> Vec<usize> {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let result =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    if result != 0 {
        return (0..std::thread::available_parallelism().map_or(1, usize::from)).collect();
    }
    (0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) })
        .collect()
}

pub fn cpu_placements() -> Vec<CpuPlacement> {
    #[cfg(target_os = "linux")]
    {
        let mut siblings = BTreeMap::<(usize, usize), Vec<(usize, usize, usize)>>::new();
        for cpu in allowed_cpus() {
            let topology = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
            let package = read_usize(format!("{topology}/physical_package_id")).unwrap_or(0);
            let core = read_usize(format!("{topology}/core_id")).unwrap_or(cpu);
            siblings.entry((package, core)).or_default().push((
                cpu_node(cpu),
                cpu_llc(cpu, package),
                cpu,
            ));
        }
        for cpus in siblings.values_mut() {
            cpus.sort_unstable();
        }
        let max_smt = siblings.values().map(Vec::len).max().unwrap_or(1);
        let mut out = Vec::new();
        for smt_level in 0..max_smt {
            let level = siblings
                .iter()
                .filter_map(|(&(package, core), cpus)| {
                    let &(node, llc, cpu) = cpus.get(smt_level)?;
                    Some(CpuPlacement {
                        cpu,
                        node,
                        package,
                        core,
                        llc,
                        smt_level,
                    })
                })
                .collect::<Vec<_>>();
            out.extend(interleave_by_llc(level));
        }
        return out;
    }
    #[cfg(not(target_os = "linux"))]
    {
        interleave_by_llc(
            (0..std::thread::available_parallelism().map_or(1, usize::from))
                .map(|cpu| CpuPlacement {
                    cpu,
                    node: 0,
                    package: 0,
                    core: cpu,
                    llc: 0,
                    smt_level: 0,
                })
                .collect(),
        )
    }
}

pub fn numa_nodes(placements: &[CpuPlacement]) -> Vec<(usize, usize)> {
    let mut seen = BTreeSet::new();
    placements
        .iter()
        .filter_map(|placement| {
            seen.insert(placement.node)
                .then_some((placement.node, placement.cpu))
        })
        .collect()
}

pub fn pin_current_thread(cpu: usize) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
        unsafe { libc::CPU_SET(cpu, &mut set) };
        let result =
            unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
        if result != 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placements_are_unique_and_physical_cores_come_first() {
        let placements = cpu_placements();
        assert!(!placements.is_empty());
        let unique = placements.iter().map(|p| p.cpu).collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), placements.len());
        let first_smt = placements
            .iter()
            .position(|p| p.smt_level > 0)
            .unwrap_or(placements.len());
        assert!(placements[..first_smt].iter().all(|p| p.smt_level == 0));
        assert!(placements[first_smt..].iter().all(|p| p.smt_level > 0));
    }

    #[test]
    fn placements_visit_each_llc_before_reusing_one() {
        let placements = interleave_by_llc(vec![
            CpuPlacement {
                cpu: 0,
                node: 0,
                package: 0,
                core: 0,
                llc: 0,
                smt_level: 0,
            },
            CpuPlacement {
                cpu: 1,
                node: 0,
                package: 0,
                core: 1,
                llc: 0,
                smt_level: 0,
            },
            CpuPlacement {
                cpu: 2,
                node: 0,
                package: 0,
                core: 2,
                llc: 1,
                smt_level: 0,
            },
            CpuPlacement {
                cpu: 3,
                node: 0,
                package: 0,
                core: 3,
                llc: 1,
                smt_level: 0,
            },
        ]);
        assert_eq!(
            placements.iter().map(|p| p.llc).collect::<Vec<_>>(),
            vec![0, 1, 0, 1]
        );
    }
}
