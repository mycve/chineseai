"""在 Windows 同一逻辑核心交替测量旧模型、压缩模型与增量缓存，保存原始 NPS。"""
import argparse
import ctypes
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline_exe")
parser.add_argument("candidate_exe")
parser.add_argument("baseline_model")
parser.add_argument("candidate_model")
parser.add_argument("fens")
parser.add_argument("output")
args = parser.parse_args()
api = ctypes.WinDLL("kernel32", use_last_error=True)
api.GetCurrentProcess.restype = ctypes.c_void_p
api.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
if not api.SetProcessAffinityMask(api.GetCurrentProcess(), 4):
    raise ctypes.WinError(ctypes.get_last_error())
fens = [None, *Path(args.fens).read_text(encoding="utf-8").splitlines()[:4]]
cases = {
    "baseline": (args.baseline_exe, args.baseline_model, []),
    "compressed": (args.candidate_exe, args.candidate_model, []),
    "incremental": (args.candidate_exe, args.candidate_model, ["--threat-incremental"]),
}
results = []
for fen in fens:
    samples = {name: [] for name in cases}
    for repeat in range(3):
        order = list(cases)
        for name in order[repeat:] + order[:repeat]:
            exe, model, flags = cases[name]
            command = [exe, "az-bench", *flags, model, "800", "200", "1"]
            if fen:
                command.append(fen)
            output = subprocess.check_output(command, text=True)
            samples[name].append(int(re.search(r"sims/sec\s*:\s*(\d+)", output)[1]))
    medians = {k: statistics.median(v) for k, v in samples.items()}
    results.append({"fen": fen or "startpos", "samples": samples, "medians": medians})
    print(medians, flush=True)
ratios = {name: math.exp(statistics.mean(math.log(r["medians"][name] / r["medians"]["baseline"]) for r in results)) for name in cases}
Path(args.output).write_text(json.dumps({"nodes": 800, "repeat": 200, "rounds": 3, "cpu_mask": 4, "results": results, "geomean_ratios": ratios}, ensure_ascii=False, indent=2), encoding="utf-8")
print("geomean ratios", ratios)
