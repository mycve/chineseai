"""按固定局面配对比较NNUE特征基准；多轮结果先在每个局面内取中位数。"""
import argparse
import csv
import json
import statistics
from pathlib import Path


IDENTITY = ("case", "fen", "pieces", "moves", "relation_rows", "motif_rows")
METRICS = (
    ("move_features_ns", "候选后果提取", 1000, "µs", False),
    ("incremental_features_ns", "单步增量维护", 1000, "µs", False),
    ("full_features_ns", "全量特征提取", 1000, "µs", False),
    ("full_eval_ns", "全量前向", 1000, "µs", False),
    ("cached_eval_ns", "已有累加和的前向", 1000, "µs", False),
    ("search_full_nps", "完整搜索：全量模式", 1, "sims/s", True),
    ("search_incremental_nps", "完整搜索：增量模式", 1, "sims/s", True),
    ("search_incremental_bytes", "增量搜索保留内存", 1048576, "MiB", False),
)


def load(paths):
    runs = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 64, f"{path}: expected 64 positions"
        assert len({r["case"] for r in rows}) == 64, f"{path}: duplicate case"
        if runs:
            assert [[r[k] for k in IDENTITY] for r in rows] == [
                [r[k] for k in IDENTITY] for r in runs[0]
            ], f"{path}: position/feature mismatch"
        runs.append(rows)
    result = []
    for index, row in enumerate(runs[0]):
        aggregate = {k: row[k] for k in IDENTITY}
        for key in row:
            if key not in aggregate:
                aggregate[key] = statistics.median(float(run[index][key]) for run in runs)
        result.append(aggregate)
    return result


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, nargs="+", required=True)
    parser.add_argument("--after", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix", type=Path, default=Path("NNUE_FEATURE_LOCAL"))
    args = parser.parse_args()
    before, after = load(args.before), load(args.after)
    assert [[r[k] for k in IDENTITY] for r in before] == [
        [r[k] for k in IDENTITY] for r in after
    ], "before/after position or feature mismatch"
    pairs = []
    for old, new in zip(before, after):
        pair = {k: old[k] for k in IDENTITY}
        for key, _, _, _, _ in METRICS:
            pair[f"before_{key}"] = old[key]
            pair[f"after_{key}"] = new[key]
            pair[f"ratio_{key}"] = new[key] / old[key]
        pairs.append(pair)
    prefix = str(args.prefix)
    write_csv(Path(prefix + "_BEFORE.csv"), before)
    write_csv(Path(prefix + "_AFTER.csv"), after)
    write_csv(Path(prefix + "_PAIRED.csv"), pairs)
    lines = [
        "# NNUE局部差分基准配对结果", "",
        f"64个局面；优化前{len(args.before)}轮、优化后{len(args.after)}轮。每个局面先跨轮取中位数，再统计局面中位数。",
        "棋盘、候选数、R/M活跃行数逐项相等。配对变化按各局面的after/before比值取中位数，不用两个总体中位数相除。", "",
        "| 指标 | 优化前中位数 | 优化后中位数 | 配对变化 |",
        "|---|---:|---:|---:|",
    ]
    summary = {}
    for key, label, divisor, unit, higher in METRICS:
        a = statistics.median(row[key] for row in before) / divisor
        b = statistics.median(row[key] for row in after) / divisor
        ratio = statistics.median(row[f"ratio_{key}"] for row in pairs)
        change = (ratio - 1) * 100
        lines.append(f"| {label} | {a:.3f} {unit} | {b:.3f} {unit} | {change:+.1f}% |")
        summary[key] = {"before": a, "after": b, "paired_ratio": ratio,
                        "gain": ratio - 1 if higher else 1 - ratio}
    faster = sum(row["ratio_search_incremental_nps"] > 1 for row in pairs)
    versus_full = statistics.median(r["search_incremental_nps"] / r["search_full_nps"] for r in after)
    lines += ["", f"增量搜索在{faster}/64个局面的三轮中位数高于基线；优化后的增量/全量搜索吞吐比中位数为{versus_full:.3f}×。"]
    if "local_updates" in after[0]:
        updates = sum(row["feature_updates"] for row in after)
        local = sum(row["local_updates"] for row in after)
        lines += ["", f"完整增量搜索中，{local / updates:.1%}的维护使用已缓存祖先的局部行差分；其余因小残局策略、缓存缺失或周期刷新而准确重建。"]
    if "two_ply_features_ns" in after[0]:
        rows = [r for r in after if r["two_ply_full_features_ns"] > 0]
        delta = statistics.median(r["two_ply_features_ns"] for r in rows) / 1000
        full = statistics.median(r["two_ply_full_features_ns"] for r in rows) / 1000
        lines += ["", f"跨两步维护中位数：{delta:.3f} µs；同一终点的全量特征提取：{full:.3f} µs。"]
    lines += ["", "负耗时变化表示更快；正NPS变化表示更快。内存为工作区保留容量，包含有界图缓存，不是进程RSS。", ""]
    Path(prefix + "_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
