#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import sys
from dataclasses import dataclass

BOARD_FILES = 9
BOARD_RANKS = 10
BOARD_SIZE = BOARD_FILES * BOARD_RANKS
INPUT_SIZE = BOARD_SIZE * 14 + 1
V2_INPUT_SIZE = INPUT_SIZE + 2 * 9 * 14 * BOARD_SIZE
STARTPOS_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
MATE_SCORE = 30000


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

    def search(
        self,
        fen: str,
        depth: int | None = None,
        movetime_ms: int | None = None,
    ) -> SearchSummary:
        if depth is None and movetime_ms is None:
            raise RuntimeError("either depth or movetime_ms must be provided")

        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok")
        self.send(f"position fen {fen}")
        if movetime_ms is not None:
            self.send(f"go movetime {movetime_ms}")
        else:
            self.send(f"go depth {depth}")

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


def piece_index(ch: str) -> int:
    color_base = 0 if ch.isupper() else 7
    kind = ch.lower()
    kind_index = {
        "k": 0,
        "a": 1,
        "b": 2,
        "e": 2,
        "n": 3,
        "h": 3,
        "r": 4,
        "c": 5,
        "p": 6,
    }.get(kind)
    if kind_index is None:
        raise ValueError(f"unsupported piece: {ch}")
    return color_base + kind_index


def fen_to_features(fen: str, feature_set: str) -> list[int]:
    if fen == "startpos":
        fen = STARTPOS_FEN
    parts = fen.split()
    if len(parts) < 2:
        raise ValueError(f"invalid FEN: {fen}")
    board_part, side_part = parts[0], parts[1]
    ranks = board_part.split("/")
    if len(ranks) != BOARD_RANKS:
        raise ValueError(f"expected 10 ranks in FEN: {fen}")

    pieces: list[tuple[int, str]] = []
    features: list[int] = []
    for rank_index, rank in enumerate(ranks):
        file_index = 0
        for ch in rank:
            if ch.isdigit():
                file_index += int(ch)
                continue
            if file_index >= BOARD_FILES:
                raise ValueError(f"file overflow in FEN: {fen}")
            sq = rank_index * BOARD_FILES + file_index
            pieces.append((sq, ch))
            features.append(piece_index(ch) * BOARD_SIZE + sq)
            file_index += 1
        if file_index != BOARD_FILES:
            raise ValueError(f"rank width mismatch in FEN: {fen}")

    if side_part == "w":
        features.append(INPUT_SIZE - 1)
    if feature_set == "v2":
        red_king_bucket = find_general_bucket(pieces, red=True)
        black_king_bucket = find_general_bucket(pieces, red=False)
        for sq, ch in pieces:
            index = piece_index(ch)
            features.append(king_aware_feature_index(0, red_king_bucket, index, sq))
            features.append(king_aware_feature_index(1, black_king_bucket, index, sq))
    return features


def find_general_bucket(pieces: list[tuple[int, str]], red: bool) -> int:
    target = "K" if red else "k"
    for sq, ch in pieces:
        if ch == target:
            return palace_bucket(sq, red)
    return 4


def palace_bucket(sq: int, red: bool) -> int:
    file_index = min(5, max(3, sq % BOARD_FILES)) - 3
    rank = sq // BOARD_FILES
    rank_index = min(9, max(7, rank)) - 7 if red else min(2, max(0, rank))
    return rank_index * 3 + file_index


def king_aware_feature_index(
    perspective: int, king_bucket: int, piece: int, square: int
) -> int:
    return INPUT_SIZE + (((perspective * 9 + king_bucket) * 14 + piece) * BOARD_SIZE + square)


def normalize_input_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line == "startpos":
        return STARTPOS_FEN
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        maybe_label, rest = parts
        try:
            float(maybe_label)
            line = rest
        except ValueError:
            pass
    return STARTPOS_FEN if line == "startpos" else line


def teacher_score(summary: SearchSummary, mate_score: int, clamp: int) -> int:
    if summary.score_kind == "mate":
        score = mate_score if summary.score_value > 0 else -mate_score
    else:
        score = summary.score_value
    return max(-clamp, min(clamp, score))


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Distill Pikafish evaluations into ChineseAI NNUE training samples."
    )
    parser.add_argument("input", type=pathlib.Path, help="text file of FENs or selfplay-dump rows")
    parser.add_argument("output", type=pathlib.Path, help="output label\\tfeature file")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--movetime-ms", type=int)
    parser.add_argument("--clamp", type=int, default=1200)
    parser.add_argument("--mate-score", type=int, default=5000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dedup", action="store_true", help="skip duplicate FENs")
    parser.add_argument("--feature-set", choices=["v1", "v2"], default="v1")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    pikafish = pathlib.Path(args.pikafish).resolve()
    pikafish_nnue = pathlib.Path(args.pikafish_nnue).resolve()
    if not pikafish.exists():
        raise SystemExit(f"pikafish not found: {pikafish}")
    if not pikafish_nnue.exists():
        raise SystemExit(f"pikafish nnue not found: {pikafish_nnue}")

    raw_lines = args.input.read_text(encoding="utf-8").splitlines()
    fens: list[str] = []
    seen: set[str] = set()
    for raw in raw_lines:
        fen = normalize_input_line(raw)
        if fen is None:
            continue
        if args.dedup:
            if fen in seen:
                continue
            seen.add(fen)
        fens.append(fen)
        if args.limit is not None and len(fens) >= args.limit:
            break

    if not fens:
        raise SystemExit("no valid positions found")

    init = [f"setoption name EvalFile value {pikafish_nnue}"]
    with UciEngine([str(pikafish)], root, init) as engine, args.output.open(
        "w", encoding="utf-8"
    ) as out:
        for index, fen in enumerate(fens, start=1):
            summary = engine.search(fen, depth=args.depth, movetime_ms=args.movetime_ms)
            label = teacher_score(summary, args.mate_score, args.clamp)
            features = fen_to_features(fen, args.feature_set)
            input_size = V2_INPUT_SIZE if args.feature_set == "v2" else INPUT_SIZE
            if any(feature >= input_size for feature in features):
                raise RuntimeError("feature index exceeded input size")
            out.write(f"{label}\t{' '.join(str(value) for value in features)}\n")
            if index % 50 == 0 or index == len(fens):
                print(
                    f"distilled {index}/{len(fens)} positions "
                    f"(last depth={summary.depth} score={summary.score_kind} {summary.score_value})"
                )

    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
