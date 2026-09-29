"""Compare UCI eval output and fixed-depth searches on identical positions.

Matching scores and node counts alone do not prove identical pruning.
"""

import argparse
import json
import queue
import re
import subprocess
import threading
import time
from pathlib import Path


SCORE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)")
DEPTH = re.compile(r"\bdepth\s+(\d+)")
NODES = re.compile(r"\bnodes\s+(\d+)")
PV = re.compile(r"\bpv\s+(.+)$")
SELDEPTH = re.compile(r"\bseldepth\s+(\d+)")
INTERNAL = re.compile(r"NNUE evaluation:?\s+([+-]?\d+)\s*\((?:side to move, )?internal units\)")
FINAL = re.compile(r"Final evaluation:?\s+([+-]?\d+(?:\.\d+)?)")
STARTPOS = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


class Engine:
    def __init__(self, executable: Path, weights: Path, timeout: float):
        self.process = subprocess.Popen(
            [str(executable.resolve())], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.timeout = timeout
        self.send("uci")
        options = self.until(lambda line: line == "uciok")
        if not any(line.startswith("option name EvalFile ") for line in options):
            raise RuntimeError(f"{executable} has no EvalFile option")
        self.send(f"setoption name EvalFile value {weights.resolve()}")
        self.send("isready")
        ready = self.until(lambda line: line == "readyok")
        if any("failed to load" in line.lower() for line in ready):
            raise RuntimeError("\n".join(ready))

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\r\n"))
        self.lines.put(None)

    def send(self, command: str):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def until(self, done):
        result = []
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise TimeoutError(f"engine timed out: {result[-5:]}") from exc
            if line is None:
                raise RuntimeError(f"engine exited: {result[-5:]}")
            result.append(line)
            if done(line):
                return result

    def probe(self, position: str, depth: int, searchmoves: list[str]):
        self.send("ucinewgame")
        self.send(position)
        self.send("isready")
        self.until(lambda line: line == "readyok")
        self.send("eval")
        # eval is a non-standard UCI extension. The ready marker drains its output.
        self.send("isready")
        evaluation = self.until(lambda line: line == "readyok")
        search = []
        if depth > 0:
            suffix = " searchmoves " + " ".join(searchmoves) if searchmoves else ""
            self.send(f"go depth {depth}{suffix}")
            search = self.until(lambda line: line.startswith("bestmove "))
        info = [line for line in search if line.startswith("info ") and SCORE.search(line)]
        score_line = info[-1] if info else ""
        score = SCORE.search(score_line)
        observed_depth = DEPTH.search(score_line)
        nodes = NODES.search(score_line)
        pv = PV.search(score_line)
        seldepth = SELDEPTH.search(score_line)
        internal = next((INTERNAL.search(line) for line in evaluation if INTERNAL.search(line)), None)
        final = next((FINAL.search(line) for line in evaluation if FINAL.search(line)), None)
        return {
            "eval_internal_units": int(internal.group(1)) if internal else None,
            "eval_final": float(final.group(1)) if final else None,
            "eval_lines": [line for line in evaluation if line.startswith(("NNUE evaluation", "Final evaluation", "info string eval"))],
            "depth": int(observed_depth.group(1)) if observed_depth else None,
            "score_type": score.group(1) if score else None,
            "score": int(score.group(2)) if score else None,
            "nodes": int(nodes.group(1)) if nodes else None,
            "seldepth": int(seldepth.group(1)) if seldepth else None,
            "pv": pv.group(1) if pv else None,
            "bestmove": search[-1].split()[1] if search else None,
        }

    def close(self):
        if self.process.poll() is None:
            try:
                self.send("quit")
            except OSError:
                pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chineseai", type=Path, required=True)
    parser.add_argument("--pikafish", type=Path, required=True)
    parser.add_argument("--chineseai-nnue", type=Path, required=True)
    parser.add_argument("--pikafish-nnue", type=Path, required=True)
    parser.add_argument("--fens", type=Path, help="UTF-8 file with one FEN or UCI position command per line")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--searchmoves", nargs="+", default=[], help="restrict both engines to the same root moves")
    parser.add_argument("--eval-only", action="store_true", help="compare raw NNUE integers without search")
    parser.add_argument("--require-internal-equal", action="store_true", help="exit nonzero if any raw evaluation differs or is absent")
    parser.add_argument("--allow-unavailable", action="store_true", help="allow positions where an engine does not report a raw value, such as check")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    for path in (args.chineseai, args.pikafish, args.chineseai_nnue, args.pikafish_nnue):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    positions = ([line.strip() for line in args.fens.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
            if args.fens else [STARTPOS])
    if not positions:
        parser.error("no positions supplied")
    engines = []
    try:
        engines = [
            Engine(args.chineseai, args.chineseai_nnue, args.timeout),
            Engine(args.pikafish, args.pikafish_nnue, args.timeout),
        ]
        mismatch = 0
        unavailable = 0
        for entry in positions:
            position = entry if entry.startswith("position ") else f"position fen {entry}"
            for depth in ([0] if args.eval_only else args.depths):
                left, right = (engine.probe(position, depth, args.searchmoves) for engine in engines)
                internal_equal = (left["eval_internal_units"] == right["eval_internal_units"]
                                  if left["eval_internal_units"] is not None
                                  and right["eval_internal_units"] is not None else None)
                if internal_equal is False:
                    mismatch += 1
                elif internal_equal is None:
                    unavailable += 1
                print(json.dumps({
                    "position": position, "requested_depth": depth, "searchmoves": args.searchmoves,
                    "chineseai": left, "pikafish": right,
                    "internal_equal": internal_equal,
                    "score_equal": (left["score"] is not None and right["score"] is not None
                                    and (left["score_type"], left["score"]) ==
                                    (right["score_type"], right["score"])),
                    "nodes_equal": (left["nodes"] == right["nodes"]
                                    if left["nodes"] is not None and right["nodes"] is not None else None),
                    "bestmove_equal": left["bestmove"] == right["bestmove"],
                }, ensure_ascii=False))
        if args.require_internal_equal and (mismatch or (unavailable and not args.allow_unavailable)):
            raise SystemExit(f"raw NNUE mismatch in {mismatch} probe(s), unavailable in {unavailable}")
    finally:
        for engine in engines:
            engine.close()


if __name__ == "__main__":
    main()
