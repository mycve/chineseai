#!/usr/bin/env python3

import argparse
import json
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass


DEFAULT_MOVETIME_MS = 200
DEFAULT_GAMES = 4
DEFAULT_MAX_PLIES = 200


@dataclass
class SearchSummary:
    depth: int = 0
    seldepth: int = 0
    score_kind: str = "cp"
    score_value: int = 0
    nodes: int = 0
    time_ms: int = 0
    nps: int = 0
    pv: str = ""
    bestmove: str = "0000"


def parse_info_line(line: str) -> SearchSummary | None:
    if not line.startswith("info depth "):
        return None

    tokens = line.split()
    summary = SearchSummary()
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "depth" and index + 1 < len(tokens):
            summary.depth = int(tokens[index + 1])
            index += 2
        elif token == "seldepth" and index + 1 < len(tokens):
            summary.seldepth = int(tokens[index + 1])
            index += 2
        elif token == "score" and index + 2 < len(tokens):
            summary.score_kind = tokens[index + 1]
            summary.score_value = int(tokens[index + 2])
            index += 3
        elif token == "nodes" and index + 1 < len(tokens):
            summary.nodes = int(tokens[index + 1])
            index += 2
        elif token == "time" and index + 1 < len(tokens):
            summary.time_ms = int(tokens[index + 1])
            index += 2
        elif token == "nps" and index + 1 < len(tokens):
            summary.nps = int(tokens[index + 1])
            index += 2
        elif token == "pv":
            summary.pv = " ".join(tokens[index + 1 :]).strip()
            break
        else:
            index += 1
    return summary


class UciEngine:
    def __init__(self, command: list[str], cwd: pathlib.Path, extra_init: list[str] | None = None):
        self.command = command
        self.cwd = cwd
        self.extra_init = extra_init or []
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self):
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.wait_for("uciok")
        for command in self.extra_init:
            self.send(command)
        self.send("isready")
        self.wait_for("readyok")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.send("quit")
        except Exception:
            pass
        if self.process is not None:
            self.process.wait(timeout=5)

    def send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("engine is not running")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_line(self) -> str:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("engine is not running")
        line = self.process.stdout.readline()
        if line == "":
            raise RuntimeError("engine terminated unexpectedly")
        return line.rstrip("\n")

    def wait_for(self, token: str) -> str:
        while True:
            line = self.read_line()
            if token in line:
                return line

    def sync_ready(self) -> None:
        self.send("isready")
        self.wait_for("readyok")

    def set_position(self, position: str, moves: list[str]) -> None:
        self.send("ucinewgame")
        self.sync_ready()
        if position == "startpos":
            command = "position startpos"
        else:
            command = f"position fen {position}"
        if moves:
            command += " moves " + " ".join(moves)
        self.send(command)

    def search(self, position: str, moves: list[str], movetime_ms: int) -> SearchSummary:
        self.set_position(position, moves)
        self.send(f"go movetime {movetime_ms}")
        summary = SearchSummary()
        while True:
            line = self.read_line()
            if line.startswith("info depth "):
                parsed = parse_info_line(line)
                if parsed is not None:
                    summary = parsed
            elif line.startswith("bestmove "):
                summary.bestmove = line.split()[1]
                return summary

    def current_fen(self) -> str:
        self.send("d")
        while True:
            line = self.read_line()
            if line.startswith("info string fen "):
                return line.removeprefix("info string fen ").strip()

    def legal_move_count(self) -> int:
        self.send("go perft 1")
        nodes = None
        while True:
            line = self.read_line()
            if line.startswith("info depth 1 "):
                parsed = parse_info_line(line)
                if parsed is not None:
                    nodes = parsed.nodes
            elif line.startswith("bestmove "):
                return nodes or 0


def ensure_exists(path_text: str, label: str) -> pathlib.Path:
    path = pathlib.Path(path_text).resolve()
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    return path


def load_openings(openings_file: str | None, openings: list[str]) -> list[str]:
    positions = list(openings)
    if openings_file:
        raw = pathlib.Path(openings_file).read_text(encoding="utf-8")
        for line in raw.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                positions.append(line)
    return positions or ["startpos"]


def format_score(summary: SearchSummary) -> str:
    return f"{summary.score_kind} {summary.score_value}"


def side_from_fen(fen: str) -> str:
    parts = fen.split()
    return parts[1] if len(parts) > 1 else "w"


def engine_name(label: str, swap: bool, side: str) -> str:
    if side == "w":
        return "pikafish" if swap else label
    return label if swap else "pikafish"


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run Xiangqi UCI matches between ChineseAI and Pikafish."
    )
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="number of games")
    parser.add_argument(
        "--movetime-ms",
        type=int,
        default=DEFAULT_MOVETIME_MS,
        help="fixed movetime for each side on each move",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=DEFAULT_MAX_PLIES,
        help="adjudicate as draw after this many plies",
    )
    parser.add_argument(
        "--opening",
        action="append",
        default=[],
        help="opening position: 'startpos' or a Xiangqi FEN; may be repeated",
    )
    parser.add_argument(
        "--openings-file",
        help="text file with one opening per line",
    )
    parser.add_argument(
        "--ours",
        default=str(root / "target" / "release" / "chineseai"),
        help="path to our engine executable",
    )
    parser.add_argument(
        "--pikafish",
        default=str(root / "tools" / "pikafish"),
        help="path to official Pikafish executable",
    )
    parser.add_argument(
        "--pikafish-nnue",
        default=str(root / "tools" / "pikafish.nnue"),
        help="path to Pikafish NNUE file",
    )
    parser.add_argument(
        "--ours-eval",
        help="path to a ChineseAI text NNUE model loaded through UCI EvalFile",
    )
    parser.add_argument(
        "--report-out",
        type=pathlib.Path,
        help="write a JSON match report to this path",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="print one progress line every N plies; use 0 to disable",
    )
    parser.add_argument(
        "--label",
        default="ours",
        help="label to print for the first engine",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    ours = ensure_exists(args.ours, "our engine")
    pikafish = ensure_exists(args.pikafish, "Pikafish")
    pikafish_nnue = ensure_exists(args.pikafish_nnue, "Pikafish NNUE")
    ours_eval = ensure_exists(args.ours_eval, "our EvalFile") if args.ours_eval else None
    openings = load_openings(args.openings_file, args.opening)

    ours_cmd = [str(ours), "uci"]
    ours_init = [f"setoption name EvalFile value {ours_eval}"] if ours_eval else []
    pikafish_cmd = [str(pikafish)]
    pikafish_init = [f"setoption name EvalFile value {pikafish_nnue}"]

    results = {args.label: 0, "pikafish": 0, "draw": 0}
    game_reports = []
    match_started = time.monotonic()

    print(
        f"match start: games={args.games} movetime_ms={args.movetime_ms} "
        f"max_plies={args.max_plies} openings={len(openings)} ours_eval={ours_eval or '(none)'}",
        flush=True,
    )

    with UciEngine(ours_cmd, root, ours_init) as our_engine, UciEngine(
        pikafish_cmd, root, pikafish_init
    ) as pikafish_engine, UciEngine(ours_cmd, root) as referee:
        for game_index in range(args.games):
            game_started = time.monotonic()
            opening = openings[game_index % len(openings)]
            swap_colors = game_index % 2 == 1
            moves: list[str] = []

            print(
                f"game {game_index + 1:>2} start: opening={opening} "
                f"red={engine_name(args.label, swap_colors, 'w')} "
                f"black={engine_name(args.label, swap_colors, 'b')}",
                flush=True,
            )

            referee.set_position(opening, moves)
            current_fen = referee.current_fen()
            repetition = {current_fen: 1}
            result = "draw"
            reason = "max plies"

            for ply in range(args.max_plies):
                side = side_from_fen(current_fen)
                engine = (
                    pikafish_engine if (swap_colors == (side == "w")) else our_engine
                )
                mover_name = engine_name(args.label, swap_colors, side)
                summary = engine.search(opening, moves, args.movetime_ms)
                bestmove = summary.bestmove
                if bestmove == "0000":
                    result = "pikafish" if mover_name == args.label else args.label
                    reason = f"{mover_name} has no legal move"
                    break

                moves.append(bestmove)
                referee.set_position(opening, moves)
                current_fen = referee.current_fen()
                legal_moves = referee.legal_move_count()
                repetition[current_fen] = repetition.get(current_fen, 0) + 1

                if repetition[current_fen] >= 3:
                    result = "draw"
                    reason = "threefold repetition"
                    break
                if legal_moves == 0:
                    result = mover_name
                    reason = "opponent has no legal move"
                    break

                if args.progress_every > 0 and (ply + 1) % args.progress_every == 0:
                    print(
                        f"game {game_index + 1:>2} progress: ply={ply + 1} "
                        f"last={mover_name}:{bestmove} depth={summary.depth} "
                        f"nodes={summary.nodes} elapsed={time.monotonic() - game_started:.1f}s",
                        flush=True,
                    )

                if ply == args.max_plies - 1:
                    result = "draw"
                    reason = "max plies"

            results[result] += 1
            game_reports.append(
                {
                    "game": game_index + 1,
                    "opening": opening,
                    "red": engine_name(args.label, swap_colors, "w"),
                    "black": engine_name(args.label, swap_colors, "b"),
                    "result": result,
                    "reason": reason,
                    "plies": len(moves),
                    "moves": moves,
                }
            )
            print(
                f"game {game_index + 1:>2}: opening={opening} "
                f"red={engine_name(args.label, swap_colors, 'w')} "
                f"black={engine_name(args.label, swap_colors, 'b')} "
                f"result={result} reason={reason} plies={len(moves)} "
                f"elapsed={time.monotonic() - game_started:.1f}s",
                flush=True,
            )

    total = sum(results.values())
    print("\nscoreboard")
    print(f"  {args.label:<8} {results[args.label]}")
    print(f"  {'pikafish':<8} {results['pikafish']}")
    print(f"  {'draw':<8} {results['draw']}")
    if total:
        score = results[args.label] + results["draw"] * 0.5
        print(f"  score    {score:.1f}/{total}")
    print(f"  elapsed  {time.monotonic() - match_started:.1f}s")
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "label": args.label,
            "ours": str(ours),
            "ours_eval": str(ours_eval) if ours_eval else None,
            "pikafish": str(pikafish),
            "pikafish_nnue": str(pikafish_nnue),
            "games": args.games,
            "movetime_ms": args.movetime_ms,
            "max_plies": args.max_plies,
            "results": results,
            "score": results[args.label] + results["draw"] * 0.5,
            "game_reports": game_reports,
        }
        args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"report   {args.report_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
