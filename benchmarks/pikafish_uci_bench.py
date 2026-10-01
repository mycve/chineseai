"""同局面、单线程、固定节点预算的 UCI 引擎基准。"""
import argparse
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("engine", type=Path)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--positions", type=Path, default=Path("benchmarks/pikafish-search-positions.fen"))
parser.add_argument("--nodes", type=int, default=262144)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
process = subprocess.Popen([str(args.engine.resolve())], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, text=True, encoding="utf-8")


def command(text):
    process.stdin.write(text + "\n")
    process.stdin.flush()


def until(prefix):
    lines = []
    for line in process.stdout:
        lines.append(line.strip())
        if line.startswith(prefix):
            return lines
    raise RuntimeError(f"引擎提前退出，等待 {prefix}")


results = []
try:
    command("uci")
    until("uciok")
    command("setoption name Threads value 1")
    command("setoption name Hash value 16")
    command(f"setoption name EvalFile value {args.model.resolve()}")
    command("isready")
    until("readyok")
    positions = [fen for fen in args.positions.read_text(encoding="utf-8").splitlines() if fen]
    # 首次加载网络不计入计时。
    command(f"position fen {positions[0]}")
    command("go nodes 8192")
    until("bestmove")
    for _ in range(args.repeats):
        for fen in positions:
            command("ucinewgame")
            command("isready")
            until("readyok")
            command(f"position fen {fen}")
            started = time.perf_counter()
            command(f"go nodes {args.nodes}")
            lines = until("bestmove")
            elapsed = time.perf_counter() - started
            info = next(line.split() for line in reversed(lines)
                        if line.startswith("info depth") and "nodes" in line.split())
            nodes = int(info[info.index("nodes") + 1])
            results.append({"fen": fen, "nodes": nodes, "seconds": elapsed,
                            "bestmove": lines[-1].split()[1]})
    seconds = sum(row["seconds"] for row in results)
    nodes = sum(row["nodes"] for row in results)
    print(json.dumps({"engine": str(args.engine), "threads": 1, "node_budget": args.nodes,
                      "search_nodes": nodes, "search_seconds": seconds,
                      "search_nps": nodes / seconds, "searches": results}, ensure_ascii=False))
finally:
    if process.poll() is None:
        command("quit")
        process.wait(timeout=10)
