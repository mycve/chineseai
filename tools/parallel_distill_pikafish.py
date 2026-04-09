#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import pathlib
import sys
from dataclasses import dataclass

from distill_pikafish import (
    INPUT_SIZE,
    V2_INPUT_SIZE,
    UciEngine,
    fen_to_features,
    normalize_input_line,
    teacher_score,
)


@dataclass(frozen=True)
class DistillJob:
    worker_id: int
    fens: list[str]
    root: str
    pikafish: str
    pikafish_nnue: str
    feature_set: str
    depth: int | None
    movetime_ms: int | None
    clamp: int
    mate_score: int


def distill_chunk(job: DistillJob) -> tuple[int, list[str]]:
    root = pathlib.Path(job.root)
    init = [f"setoption name EvalFile value {job.pikafish_nnue}"]
    rows: list[str] = []
    input_size = V2_INPUT_SIZE if job.feature_set == "v2" else INPUT_SIZE
    with UciEngine([job.pikafish], root, init) as engine:
        for fen in job.fens:
            summary = engine.search(fen, depth=job.depth, movetime_ms=job.movetime_ms)
            label = teacher_score(summary, job.mate_score, job.clamp)
            features = fen_to_features(fen, job.feature_set)
            if any(feature >= input_size for feature in features):
                raise RuntimeError("feature index exceeded input size")
            rows.append(f"{label}\t{' '.join(str(value) for value in features)}\n")
    return job.worker_id, rows


def load_fens(path: pathlib.Path, limit: int | None, dedup: bool) -> list[str]:
    fens: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        fen = normalize_input_line(raw)
        if fen is None:
            continue
        if dedup:
            if fen in seen:
                continue
            seen.add(fen)
        fens.append(fen)
        if limit is not None and len(fens) >= limit:
            break
    if not fens:
        raise SystemExit("no valid positions found")
    return fens


def make_chunks(values: list[str], chunk_size: int) -> list[list[str]]:
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def parse_args() -> argparse.Namespace:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Parallel Pikafish teacher distillation into ChineseAI NNUE samples."
    )
    parser.add_argument("input", type=pathlib.Path, help="FEN file or selfplay-dump rows")
    parser.add_argument(
        "output",
        type=pathlib.Path,
        nargs="?",
        default=root / "data" / "pikafish_distill_v2.samples.txt",
    )
    parser.add_argument(
        "--workers",
        default=str(max(1, min(4, os.cpu_count() or mp.cpu_count()))),
        help="number of Pikafish worker processes, or 'auto'",
    )
    parser.add_argument(
        "--auto-worker-reserve",
        type=int,
        default=2,
        help="cores to leave idle when --workers auto is used",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="cap auto-selected workers",
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--movetime-ms", type=int)
    parser.add_argument("--feature-set", choices=["v1", "v2"], default="v2")
    parser.add_argument("--clamp", type=int, default=1200)
    parser.add_argument("--mate-score", type=int, default=5000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dedup", action="store_true")
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
    workers = resolve_worker_count(args.workers, args.auto_worker_reserve, args.max_workers)

    fens = load_fens(args.input, args.limit, args.dedup)
    chunks = make_chunks(fens, max(1, args.chunk_size))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    jobs = [
        DistillJob(
            worker_id=index,
            fens=chunk,
            root=str(root),
            pikafish=str(pikafish),
            pikafish_nnue=str(pikafish_nnue),
            feature_set=args.feature_set,
            depth=None if args.movetime_ms is not None else args.depth,
            movetime_ms=args.movetime_ms,
            clamp=args.clamp,
            mate_score=args.mate_score,
        )
        for index, chunk in enumerate(chunks)
    ]

    distilled = 0
    print(
        f"parallel distill: positions={len(fens)} workers={workers} "
        f"chunk_size={args.chunk_size} feature_set={args.feature_set}",
        flush=True,
    )
    with args.output.open("w", encoding="utf-8") as out:
        if workers <= 1:
            iterator = map(distill_chunk, jobs)
        else:
            pool = mp.Pool(processes=workers)
            iterator = pool.imap_unordered(distill_chunk, jobs)
        try:
            for _, rows in iterator:
                out.writelines(rows)
                out.flush()
                distilled += len(rows)
                print(f"distilled {distilled}/{len(fens)} positions -> {args.output}", flush=True)
        finally:
            if workers > 1:
                pool.close()
                pool.join()

    return 0


def resolve_worker_count(workers: str, reserve: int, max_workers: int | None) -> int:
    if workers == "auto":
        count = max(1, (os.cpu_count() or mp.cpu_count()) - max(0, reserve))
        if max_workers is not None:
            count = min(count, max_workers)
        return count
    try:
        count = int(workers)
    except ValueError as exc:
        raise SystemExit("--workers must be an integer or 'auto'") from exc
    return max(1, count)


if __name__ == "__main__":
    sys.exit(main())
